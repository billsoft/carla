"""
纯 LiDAR 点云体素生成器
直接将语义激光雷达点云体素化，不依赖图像融合
"""

import numpy as np
from dense_occupancy_collection.config.occupancy_config import CARLA_TO_OCCUPANCY_MAPPING

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("警告: 未安装numba，将使用慢速模式")

if HAS_NUMBA:
    @jit(nopython=True, parallel=False, cache=True)
    def _lidar_ray_cast_numba(points, labels, 
                             x_min, x_max, y_min, y_max, z_min, z_max,
                             resolution, grid_size):
        """
        Numba 加速的 LiDAR 射线投射体素化
        
        Args:
            points: (N, 3) float32
            labels: (N,) uint8 (Occupancy ID)
            ...范围参数...
        """
        X, Y, Z = grid_size
        occupancy = np.zeros((X, Y, Z), dtype=np.uint8)
        mask = np.zeros((X, Y, Z), dtype=np.uint8) # 0: Unknown, 1: Observed
        
        # 传感器原点 (近似)
        origin_x = 0.0
        origin_y = 0.0
        origin_z = 2.4 # 车顶高度
        
        # 原点体素索引
        ogx = int((origin_x - x_min) / resolution)
        ogy = int((origin_y - y_min) / resolution)
        ogz = int((origin_z - z_min) / resolution)
        
        num_points = points.shape[0]
        
        # 遍历每个点
        for i in range(num_points):
            px, py, pz = points[i, 0], points[i, 1], points[i, 2]
            label = labels[i]
            
            # 终点体素索引
            gx = int((px - x_min) / resolution)
            gy = int((py - y_min) / resolution)
            gz = int((pz - z_min) / resolution)
            
            # 检查终点是否在范围内
            if gx >= 0 and gx < X and gy >= 0 and gy < Y and gz >= 0 and gz < Z:
                # 标记 Occupied
                occupancy[gx, gy, gz] = label
                mask[gx, gy, gz] = 1
                
                # Ray Casting (Bresenham 3D 简化版)
                # 从原点到终点，沿途标记为 Free (0)
                # 为了性能，这里使用简单的线性插值采样
                
                dist = np.sqrt(px**2 + py**2 + (pz-origin_z)**2)
                steps = int(dist / (resolution * 0.8)) # 步长略小于分辨率
                
                if steps > 1:
                    dx = (px - origin_x) / steps
                    dy = (py - origin_y) / steps
                    dz = (pz - origin_z) / steps
                    
                    for s in range(1, steps): # 跳过起点(车身内部)和终点(物体)
                        sx = origin_x + dx * s
                        sy = origin_y + dy * s
                        sz = origin_z + dz * s
                        
                        ix = int((sx - x_min) / resolution)
                        iy = int((sy - y_min) / resolution)
                        iz = int((sz - z_min) / resolution)
                        
                        if ix >= 0 and ix < X and iy >= 0 and iy < Y and iz >= 0 and iz < Z:
                            # 只有当该体素还未被标记为 Occupied 时才标记为 Free
                            # 避免 Free 覆盖了后面的 Occupied (虽然理论上 RayCast 不会穿过物体，但有离散化误差)
                            if mask[ix, iy, iz] == 0:
                                occupancy[ix, iy, iz] = 0 # Free
                                mask[ix, iy, iz] = 1 # Observed
                                
        return occupancy, mask

class LidarVoxelGenerator:
    """
    纯 LiDAR 体素生成器
    """
    
    def __init__(self,
                 x_range=(-50.0, 50.0),
                 y_range=(-50.0, 50.0),
                 z_range=(-4.0, 4.0),
                 resolution=0.5):
        
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution
        
        self.grid_size = (
            int((x_range[1] - x_range[0]) / resolution),
            int((y_range[1] - y_range[0]) / resolution),
            int((z_range[1] - z_range[0]) / resolution),
        )
        
        print(f"LiDAR 体素网格: {self.grid_size}")
        
        if HAS_NUMBA:
            # 预热
            print("正在预热 Numba...", flush=True)
            dummy_points = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
            dummy_labels = np.array([1], dtype=np.uint8)
            _lidar_ray_cast_numba(
                dummy_points, dummy_labels,
                -50.0, 50.0, -50.0, 50.0, -4.0, 4.0,
                0.5, (200, 200, 16)
            )
            print("Numba 预热完成")

    def generate(self, points, labels):
        """
        生成体素
        
        Args:
            points: (N, 3)
            labels: (N,) CARLA Semantic ID
        """
        # 1. 映射标签
        mapped_labels = self._map_labels(labels)
        
        # 2. 体素化 + Ray Casting
        if HAS_NUMBA:
            occupancy, mask = _lidar_ray_cast_numba(
                points.astype(np.float32),
                mapped_labels.astype(np.uint8),
                float(self.x_range[0]), float(self.x_range[1]),
                float(self.y_range[0]), float(self.y_range[1]),
                float(self.z_range[0]), float(self.z_range[1]),
                float(self.resolution),
                self.grid_size
            )
        else:
            raise NotImplementedError("需要 Numba 支持")
            
        # 3. 填充自车
        self._fill_ego_vehicle(occupancy, mask)
        
        return occupancy, mask.astype(np.bool_)

    def _map_labels(self, semantic):
        """CARLA ID -> Occupancy ID"""
        mapped = np.zeros_like(semantic, dtype=np.uint8)
        for carla_id, occ_id in CARLA_TO_OCCUPANCY_MAPPING.items():
            mapped[semantic == carla_id] = occ_id
        return mapped

    def _fill_ego_vehicle(self, occupancy, mask):
        """填充自车体素 (简化版)"""
        x_min, x_max = -2.3, 2.3
        y_min, y_max = -0.9, 0.9
        z_min, z_max = -1.6, 0.1
        
        gx_min = int((x_min - self.x_range[0]) / self.resolution)
        gx_max = int((x_max - self.x_range[0]) / self.resolution)
        gy_min = int((y_min - self.y_range[0]) / self.resolution)
        gy_max = int((y_max - self.y_range[0]) / self.resolution)
        gz_min = int((z_min - self.z_range[0]) / self.resolution)
        gz_max = int((z_max - self.z_range[0]) / self.resolution)
        
        X, Y, Z = occupancy.shape
        gx_min, gx_max = max(0, gx_min), min(X, gx_max)
        gy_min, gy_max = max(0, gy_min), min(Y, gy_max)
        gz_min, gz_max = max(0, gz_min), min(Z, gz_max)
        
        occupancy[gx_min:gx_max, gy_min:gy_max, gz_min:gz_max] = 1 # Car
        mask[gx_min:gx_max, gy_min:gy_max, gz_min:gz_max] = 1

    def save_to_npz(self, filepath, occupancy, mask, metadata=None):
        save_dict = {
            'occupancy': occupancy,
            'mask': mask,
            'x_range': self.x_range,
            'y_range': self.y_range,
            'z_range': self.z_range,
            'resolution': self.resolution,
            'grid_size': self.grid_size
        }
        if metadata:
            save_dict.update(metadata)
        np.savez_compressed(filepath, **save_dict)
    
    def get_statistics(self, occupancy, mask):
        total_voxels = np.prod(self.grid_size)
        observed_voxels = np.sum(mask)
        occupied_voxels = np.sum(occupancy > 0)
        
        stats = {
            'total_voxels': int(total_voxels),
            'observed_voxels': int(observed_voxels),
            'occupied_voxels': int(occupied_voxels),
            'observation_rate': float(observed_voxels / total_voxels),
            'occupation_rate': float(occupied_voxels / observed_voxels) if observed_voxels > 0 else 0.0,
        }
        return stats

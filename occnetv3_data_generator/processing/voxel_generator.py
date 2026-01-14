"""
体素占用网格生成器 - 适配 OccNet V3
使用 Semantic Lidar 生成 Ground Truth
"""
import sys
from pathlib import Path
import numpy as np
import carla
from typing import Tuple, Dict

# 添加 carla_data_collection 到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "carla_data_collection"))

try:
    from utils.coordinate_transform import world_to_ego
except ImportError:
    print("Warning: Could not import world_to_ego from carla_data_collection")
    # Fallback implementation if import fails
    def world_to_ego(points_world, ego_transform):
        import math
        loc = ego_transform.location
        rot = ego_transform.rotation
        pitch = math.radians(rot.pitch)
        yaw = math.radians(rot.yaw)
        roll = math.radians(rot.roll)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [loc.x, loc.y, loc.z]
        T_inv = np.linalg.inv(T)
        points_homo = np.hstack([points_world, np.ones((len(points_world), 1))])
        points_ego = (T_inv @ points_homo.T).T[:, :3]
        return points_ego

# OccNet V3 配置
GRID_SIZE = (400, 400, 32)
PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]  # x_min, y_min, z_min, x_max, y_max, z_max
RESOLUTION = 0.2

# CARLA 语义标签 -> OccNet 类别映射
# OccNet Classes:
# 0: empty, 1: barrier, 2: bicycle, 3: bus, 4: car, 5: construction_vehicle,
# 6: motorcycle, 7: pedestrian, 8: traffic_cone, 9: trailer, 10: truck,
# 11: driveable_surface, 12: other_flat, 13: sidewalk, 14: terrain,
# 15: manmade, 16: vegetation, 17: free
CARLA_TO_OCCNET = {
    0: 0,   # Unlabeled -> empty
    1: 15,  # Building -> manmade
    2: 1,   # Fence -> barrier
    3: 15,  # Other -> manmade
    4: 7,   # Pedestrian -> pedestrian
    5: 15,  # Pole -> manmade
    6: 11,  # RoadLine -> driveable_surface
    7: 11,  # Road -> driveable_surface
    8: 13,  # SideWalk -> sidewalk
    9: 16,  # Vegetation -> vegetation
    10: 4,  # Vehicles -> car (Simplified)
    11: 15, # Wall -> manmade
    12: 15, # TrafficSign -> manmade
    13: 0,  # Sky -> empty
    14: 14, # Ground -> terrain
    15: 15, # Bridge -> manmade
    16: 1,  # RailTrack -> barrier
    17: 1,  # GuardRail -> barrier
    18: 15, # TrafficLight -> manmade
    19: 0,  # Static -> empty
    20: 0,  # Dynamic -> empty
    21: 0,  # Water -> empty
    22: 14, # Terrain -> terrain
}

class OccNetVoxelGenerator:
    """
    生成符合 OccNet V3 要求的体素占用网格
    """
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.pc_range = PC_RANGE
        self.resolution = RESOLUTION
        
        # 计算网格维度
        self.nx = int((self.pc_range[3] - self.pc_range[0]) / self.resolution)
        self.ny = int((self.pc_range[4] - self.pc_range[1]) / self.resolution)
        self.nz = int((self.pc_range[5] - self.pc_range[2]) / self.resolution)
        
        assert (self.nx, self.ny, self.nz) == self.grid_size, \
            f"Grid size mismatch: calculated {(self.nx, self.ny, self.nz)} != expected {self.grid_size}"

        print(f"[VoxelGenerator] Init: {self.grid_size}, Res: {self.resolution}m")

    def generate(self, lidar_data: Dict, ego_transform: carla.Transform) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        从语义激光雷达数据生成体素网格
        """
        import time
        t_start = time.time()
        
        # 1. 解析 Lidar 数据
        # lidar_data['raw_data'] 是 bytes
        points = np.frombuffer(lidar_data['raw_data'], dtype=np.float32).reshape(-1, 6)
        xyz_world = points[:, :3]
        semantic_tags = points[:, 5].astype(np.int32)
        
        # 2. 转换到车辆坐标系 (Ego Frame)
        xyz_ego = world_to_ego(xyz_world, ego_transform)
        
        # 3. 计算体素索引
        # x, y, z indices
        x_idx = ((xyz_ego[:, 0] - self.pc_range[0]) / self.resolution).astype(np.int32)
        y_idx = ((xyz_ego[:, 1] - self.pc_range[1]) / self.resolution).astype(np.int32)
        z_idx = ((xyz_ego[:, 2] - self.pc_range[2]) / self.resolution).astype(np.int32)
        
        # 4. 过滤范围外的点
        valid_mask = (
            (x_idx >= 0) & (x_idx < self.nx) &
            (y_idx >= 0) & (y_idx < self.ny) &
            (z_idx >= 0) & (z_idx < self.nz)
        )
        
        x_idx = x_idx[valid_mask]
        y_idx = y_idx[valid_mask]
        z_idx = z_idx[valid_mask]
        tags = semantic_tags[valid_mask]
        
        # 5. 填充网格
        occupancy = np.zeros(self.grid_size, dtype=np.uint8)
        mask = np.zeros(self.grid_size, dtype=np.uint8) # 0 or 1
        
        # 映射语义标签
        # 优化: 向量化映射
        # 创建映射数组 (假设最大 tag 是 22, 给一点余量到 30)
        map_arr = np.zeros(30, dtype=np.uint8)
        for k, v in CARLA_TO_OCCNET.items():
            if k < 30:
                map_arr[k] = v
                
        # 处理超出范围的标签 (默认为 0)
        safe_tags = np.clip(tags, 0, 29)
        occ_labels = map_arr[safe_tags]
        
        # 简单的体素填充: 后面的点覆盖前面的 (或者使用 max, 这里的逻辑可以优化)
        # 为了简单和速度，直接赋值。如果同一个体素有多个点，最后一个生效。
        # 更好的做法是: 非空覆盖空。
        
        # 利用 numpy 高级索引，重复索引只有最后一个生效
        # 我们希望保留非空类别。
        # 我们可以先按标签排序，把 0 排在前面，非0排在后面，这样非0会覆盖0
        
        sort_idx = np.argsort(occ_labels)
        x_idx = x_idx[sort_idx]
        y_idx = y_idx[sort_idx]
        z_idx = z_idx[sort_idx]
        occ_labels = occ_labels[sort_idx]
        
        occupancy[x_idx, y_idx, z_idx] = occ_labels
        mask[x_idx, y_idx, z_idx] = 1
        
        # 6. 统计
        non_empty = np.count_nonzero(occupancy)
        valid = np.count_nonzero(mask)
        elapsed = time.time() - t_start
        
        metadata = {
            'non_empty_voxels': non_empty,
            'valid_voxels': valid,
            'generation_time': elapsed
        }
        
        return occupancy, mask, metadata

    def transform_to_matrix(self, transform: carla.Transform) -> np.ndarray:
        """Helper to convert transform to 4x4 matrix"""
        import math
        loc = transform.location
        rot = transform.rotation
        pitch = math.radians(rot.pitch)
        yaw = math.radians(rot.yaw)
        roll = math.radians(rot.roll)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = [loc.x, loc.y, loc.z]
        return T

    def compute_ego_motion(self, pose_t0, pose_t1):
        return pose_t1 @ np.linalg.inv(pose_t0)

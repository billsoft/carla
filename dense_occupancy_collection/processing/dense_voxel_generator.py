"""
稠密体素生成器
将点云转换为稠密的3D Occupancy体素网格
"""

import numpy as np


class DenseVoxelGenerator:
    """
    稠密体素生成器

    将融合后的点云填充到3D网格中，生成Occupancy和Mask
    """

    def __init__(self, x_range, y_range, z_range, resolution):
        """
        初始化体素生成器

        Args:
            x_range: [x_min, x_max] 米
            y_range: [y_min, y_max] 米
            z_range: [z_min, z_max] 米
            resolution: 体素分辨率 米
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution

        # 计算网格尺寸
        self.grid_size = [
            int((x_range[1] - x_range[0]) / resolution),
            int((y_range[1] - y_range[0]) / resolution),
            int((z_range[1] - z_range[0]) / resolution)
        ]

    def generate(self, points, labels):
        """
        生成稠密体素网格

        Args:
            points: (N, 3) 车体坐标系点云 [x, y, z]
            labels: (N,) Occupancy语义标签 [0-17]

        Returns:
            occupancy: (X, Y, Z) uint8 体素语义标签
            mask: (X, Y, Z) bool 有效观测掩码
        """
        # 初始化网格
        occupancy = np.zeros(self.grid_size, dtype=np.uint8)
        mask = np.zeros(self.grid_size, dtype=np.bool_)

        if len(points) == 0:
            return occupancy, mask

        # 将点云坐标转换为网格索引
        grid_x = ((points[:, 0] - self.x_range[0]) / self.resolution).astype(np.int32)
        grid_y = ((points[:, 1] - self.y_range[0]) / self.resolution).astype(np.int32)
        grid_z = ((points[:, 2] - self.z_range[0]) / self.resolution).astype(np.int32)

        # 边界检查
        valid_mask = (
            (grid_x >= 0) & (grid_x < self.grid_size[0]) &
            (grid_y >= 0) & (grid_y < self.grid_size[1]) &
            (grid_z >= 0) & (grid_z < self.grid_size[2])
        )

        grid_x = grid_x[valid_mask]
        grid_y = grid_y[valid_mask]
        grid_z = grid_z[valid_mask]
        occ_labels = labels[valid_mask]

        # 填充体素网格
        for i in range(len(grid_x)):
            x, y, z = grid_x[i], grid_y[i], grid_z[i]
            label = occ_labels[i]

            # 标记为已观测
            mask[x, y, z] = True

            # 填充语义标签
            # 策略: 如果当前是0 (Unlabeled)，则接受任何标签
            #       如果当前非0，则只接受非0标签 (覆盖)
            if label != 0 or occupancy[x, y, z] == 0:
                occupancy[x, y, z] = label

        return occupancy, mask

    def generate_from_panorama(self, depth_pano, semantic_pano, vehicle_height=2.0):
        """
        从全景图直接生成稠密体素网格 (反向投影法)
        解决了正向投影产生的体素间隙问题

        Args:
            depth_pano: (H, W) 深度图 (米)
            semantic_pano: (H, W) 语义图
            vehicle_height: 车辆高度偏移 (米), 默认2.0米 (全景相机相对于车底的高度)
                            注意：VoxelGenerator的z_range通常是相对于车底的
        
        Returns:
            occupancy: (X, Y, Z) uint8 体素语义标签
            mask: (X, Y, Z) bool 有效观测掩码 (Free or Occupied)
        """
        # 初始化网格
        occupancy = np.zeros(self.grid_size, dtype=np.uint8)
        mask = np.zeros(self.grid_size, dtype=np.bool_)
        
        H, W = depth_pano.shape
        
        # 1. 生成体素中心坐标 (相对于Grid原点)
        # grid_indices: (3, X, Y, Z)
        gx = np.arange(self.grid_size[0])
        gy = np.arange(self.grid_size[1])
        gz = np.arange(self.grid_size[2])
        
        # 使用 meshgrid 生成所有体素的索引
        # indexing='ij' 保证维度顺序为 (X, Y, Z)
        grid_x, grid_y, grid_z = np.meshgrid(gx, gy, gz, indexing='ij')
        
        # 2. 转换为物理坐标 (相对于车辆中心)
        # Voxel Center = min + index * res + res/2
        x = self.x_range[0] + (grid_x + 0.5) * self.resolution
        y = self.y_range[0] + (grid_y + 0.5) * self.resolution
        z = self.z_range[0] + (grid_z + 0.5) * self.resolution
        
        # 修正高度: 全景相机位于车顶，Z=0通常指地面
        # 但 PanoramaTools 的 unproject 是以相机为原点
        # 所以我们需要把 Voxel 的坐标转换到 相机坐标系
        # Camera Z = Voxel Z - vehicle_height
        z_cam = z - vehicle_height
        x_cam = x
        y_cam = y
        
        # 3. 转换为球坐标 (d, lat, lon)
        # CARLA/Unreal: X前, Y右, Z上
        # d = sqrt(x^2 + y^2 + z^2)
        d = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
        
        # 避免除以0
        d[d < 1e-3] = 1e-3
        
        # lat = arcsin(z / d) -> [-pi/2, pi/2]
        lat = np.arcsin(np.clip(z_cam / d, -1.0, 1.0))
        
        # lon = arctan2(y, x) -> [-pi, pi]
        lon = np.arctan2(y_cam, x_cam)
        
        # 4. 转换为全景图 UV 坐标
        # u: [-pi, pi] -> [0, 1] -> [0, W]
        # v: [-pi/2, pi/2] -> [0, 1] -> [0, H] (注意 lat 定义: -pi/2是下，对应图像底部 V=H; 但通常 lat=0是中间)
        # 检查 PanoramaTools: lat = -(v / H - 0.5) * pi
        # => v / H - 0.5 = -lat / pi
        # => v = H * (0.5 - lat / pi)
        
        u_norm = (lon / (2 * np.pi)) + 0.5
        v_norm = 0.5 - (lat / np.pi)
        
        u = (u_norm * W).astype(np.int32)
        v = (v_norm * H).astype(np.int32)
        
        # 边界处理
        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)
        
        # 5. 采样深度和语义
        # 使用 numpy 高级索引
        sampled_depth = depth_pano[v, u]
        sampled_label = semantic_pano[v, u]
        
        # 6. 确定 Occupancy 和 Mask
        # 阈值: 体素分辨率的 1.0 倍 (或者 0.866 倍即 sqrt(3)/2)
        threshold = self.resolution * 0.8
        
        # 距离差
        diff = d - sampled_depth
        
        # 情况 A: 表面 (Occupied)
        # |d - sample| < threshold
        is_surface = np.abs(diff) < threshold
        
        # 情况 B: 自由空间 (Free)
        # d < sample - threshold
        is_free = diff < -threshold
        
        # 情况 C: 未知/遮挡 (Unknown)
        # d > sample + threshold (Default mask=False)
        
        # 填充
        # 1. 标记所有观测到的区域 (Free + Surface)
        mask[is_free | is_surface] = True
        
        # 2. 填充语义 (仅 Surface)
        # 注意: 这里的 sampled_label 是对应射线的表面标签
        # 只有当体素确实在表面时，才赋予该标签
        occupancy[is_surface] = sampled_label[is_surface]
        
        # 3. 过滤无效标签 (如果全景图中有些区域无效)
        # valid_pano = (sampled_depth > 0.1) & (sampled_depth < 100.0)
        # mask = mask & valid_pano
        # occupancy[~valid_pano] = 0
        
        return occupancy, mask

    def get_config(self):
        """
        获取体素配置信息

        Returns:
            dict: 配置字典
        """
        return {
            'x_range': self.x_range,
            'y_range': self.y_range,
            'z_range': self.z_range,
            'resolution': self.resolution,
            'grid_size': self.grid_size
        }

    def save_to_npz(self, filepath, occupancy, mask, metadata=None):
        """
        保存体素数据为NPZ格式

        Args:
            filepath: 输出文件路径
            occupancy: (X, Y, Z) 体素数组
            mask: (X, Y, Z) 掩码数组
            metadata: 额外的元数据字典 (可选)
        """
        save_dict = {
            'occupancy': occupancy,
            'mask': mask,
            'x_range': self.x_range,
            'y_range': self.y_range,
            'z_range': self.z_range,
            'resolution': self.resolution,
            'grid_size': self.grid_size
        }

        # 添加元数据
        if metadata is not None:
            save_dict.update(metadata)

        np.savez_compressed(filepath, **save_dict)

    @staticmethod
    def load_from_npz(filepath):
        """
        从NPZ文件加载体素数据

        Args:
            filepath: NPZ文件路径

        Returns:
            data: 包含 occupancy, mask, 配置信息的字典
        """
        data = np.load(filepath)
        return {
            'occupancy': data['occupancy'],
            'mask': data['mask'],
            'x_range': data['x_range'].tolist(),
            'y_range': data['y_range'].tolist(),
            'z_range': data['z_range'].tolist(),
            'resolution': float(data['resolution']),
            'grid_size': data['grid_size'].tolist()
        }

    def get_statistics(self, occupancy, mask):
        """
        计算体素统计信息

        Args:
            occupancy: (X, Y, Z) 体素数组
            mask: (X, Y, Z) 掩码数组

        Returns:
            stats: 统计信息字典
        """
        total_voxels = np.prod(self.grid_size)
        observed_voxels = np.sum(mask)
        occupied_voxels = np.sum(occupancy > 0)
        empty_voxels = np.sum((mask) & (occupancy == 0))

        # 每个类别的体素数
        label_counts = np.bincount(occupancy.flatten(), minlength=18)

        stats = {
            'total_voxels': int(total_voxels),
            'observed_voxels': int(observed_voxels),
            'occupied_voxels': int(occupied_voxels),
            'empty_voxels': int(empty_voxels),
            'observation_rate': float(observed_voxels / total_voxels),
            'occupation_rate': float(occupied_voxels / observed_voxels) if observed_voxels > 0 else 0.0,
            'label_distribution': label_counts.tolist()
        }

        return stats

"""
可见性过滤器 (Visibility Filter) - Semantic LiDAR Version (V2)
核心逻辑:
1. 动态物体: 使用 Actor ID (obj_idx) 匹配保留 (保留整体)
2. 静态物体: 使用 LiDAR 点云击中 (Voxelization) 保留 (保留表面)
3. 地面/标线: 强制保留 (Ground Protection)

修复:
- 解决建筑/树木被剔除的问题 (通过点云击中保留)
- 解决地面标线丢失的问题 (通过 GROUND_LABELS 保留)
"""

import numpy as np
from config.occupancy_config import GROUND_LABELS

class VisibilityFilterSimple:
    def __init__(self, keep_alive_time=0.5):
        self.visibility_cache = {} # 缓存 ID，防止帧间闪烁
        self.keep_alive_time = keep_alive_time
        self.current_time = 0.0
        self.GROUND_LABELS = GROUND_LABELS
        
        # LiDAR 安装位置 (相对 Ego), 用于坐标转换
        self.lidar_offset = np.array([0.0, 0.0, 2.5], dtype=np.float32)

    def run(self, occupancy, actor_ids, grid_config, lidar_data, ego_id=None, dt=0.05):
        """
        基于 ID 聚类的可见性过滤器 (Instance-based Visibility Filter)
        
        逻辑:
        1. 找出 LiDAR 击中的体素坐标。
        2. 读取这些坐标处的 ID (包括动态 Actor ID 和静态虚拟 ID)。
        3. 将这些 ID 加入 "保留列表" (Keep Set)。
        4. 保留网格中所有 ID 在 "保留列表" 中的体素 (整体保留)。
        
        Args:
            ego_id: 自车 ID (强制保留)
        """
        self.current_time += dt
        
        # 1. 提取 LiDAR 点云
        if lidar_data is None:
            points = np.empty((0, 3), dtype=np.float32)
        else:
            points = lidar_data['points']

        # 2. 找出被击中的体素 ID (Hit IDs)
        current_hit_ids = set()
        
        if len(points) > 0:
            # 坐标转换: Sensor -> Ego
            points_ego = points + self.lidar_offset
            
            # 计算体素索引
            x_min, x_max = grid_config['x_range']
            y_min, y_max = grid_config['y_range']
            z_min, z_max = grid_config['z_range']
            res = grid_config['resolution']
            
            # (N, 3)
            indices = (points_ego - np.array([x_min, y_min, z_min])) / res
            indices = np.floor(indices).astype(np.int32)
            
            # 过滤越界点
            nx, ny, nz = occupancy.shape
            valid_mask = (
                (indices[:, 0] >= 0) & (indices[:, 0] < nx) &
                (indices[:, 1] >= 0) & (indices[:, 1] < ny) &
                (indices[:, 2] >= 0) & (indices[:, 2] < nz)
            )
            valid_indices = indices[valid_mask]
            
            if len(valid_indices) > 0:
                # 从 actor_ids 网格中读取被击中的 ID
                # actor_ids 存储了所有物体的 ID (动态为正，静态为负，空为0)
                hit_values = actor_ids[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]
                
                # 过滤掉 0 (空体素)
                unique_hits = np.unique(hit_values)
                current_hit_ids = set(unique_hits[unique_hits != 0])

        # 3. 更新时序缓存 (防止闪烁)
        for aid in current_hit_ids:
            self.visibility_cache[aid] = self.current_time
        
        # ⭐ 强制保留 Ego ID
        if ego_id is not None:
            self.visibility_cache[ego_id] = self.current_time
            
        # 清理过期缓存，生成最终保留列表
        final_keep_ids = []
        expired_ids = []
        for aid, last_time in self.visibility_cache.items():
            if (self.current_time - last_time) <= self.keep_alive_time:
                final_keep_ids.append(aid)
            else:
                expired_ids.append(aid)
        for aid in expired_ids:
            del self.visibility_cache[aid]
            
        # 转为 numpy 数组加速查询
        final_keep_ids = np.array(final_keep_ids)
        
        # 4. 生成最终 Mask (Broadcasting)
        # Mask A: 属于保留 ID 的所有体素 (整体保留)
        if len(final_keep_ids) > 0:
            instance_mask = np.isin(actor_ids, final_keep_ids)
        else:
            instance_mask = np.zeros(occupancy.shape, dtype=bool)
            
        # Mask B: 地面/标线保护 (强制保留)
        # 1. 语义标签保护 (Road, Sidewalk, etc.)
        semantic_ground_mask = np.isin(occupancy, self.GROUND_LABELS)
        
        # 2. 高度保护 (Z < threshold) - 防止语义映射错误导致地面被切
        # Z range: [-1.0, 5.4], Resolution: 0.2
        # Index 0 -> -1.0 ~ -0.8
        # Index 1 -> -0.8 ~ -0.6
        # Index 5 ->  0.0 ~  0.2
        # Index 10 -> 1.0 ~  1.2
        # 保护 Z <= 1.0m (Index <= 10) 的所有体素
        # 注意: 只保护非空体素 (occupancy > 0)，避免保留地下噪声
        z_indices = np.arange(occupancy.shape[2])
        z_mask_2d = z_indices <= 10  # Z <= 1.0m
        height_protection_mask = np.zeros_like(occupancy, dtype=bool)
        height_protection_mask[:, :, z_mask_2d] = True
        height_protection_mask = height_protection_mask & (occupancy > 0) # 仅保护非空体素

        ground_mask = semantic_ground_mask | height_protection_mask
        
        # 组合
        final_mask = instance_mask | ground_mask
        
        # [DEBUG] Stats
        total_voxels = np.count_nonzero(occupancy)
        kept_voxels = np.sum(final_mask)
        instance_voxels = np.sum(instance_mask)
        ground_voxels = np.sum(ground_mask)
        
        print(f"  [Visibility] ID-Broadcasting Stats:")
        print(f"    - Hit IDs: {len(current_hit_ids)} (Cached: {len(final_keep_ids)})")
        print(f"    - Kept Voxels: {kept_voxels} ({kept_voxels/(total_voxels+1)*100:.1f}%)")
        print(f"    - Details: Instance={instance_voxels}, Ground={ground_voxels}")
        
        # 应用过滤
        filtered_occupancy = occupancy.copy()
        filtered_ids = actor_ids.copy()
        
        remove_mask = (~final_mask)
        filtered_occupancy[remove_mask] = 0
        filtered_ids[remove_mask] = 0
        
        return filtered_occupancy, filtered_ids

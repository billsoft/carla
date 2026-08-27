"""
可见性过滤器 (Visibility Filter) - Semantic LiDAR Version (V2)
核心逻辑:
1. 动态物体 (真实 Actor ID): 用语义LiDAR原生返回的 obj_idx 直接匹配保留 (保留整体)。
2. 静态物体 (Buildings/Poles/Vegetation 等虚拟负数ID, 没有真实 obj_idx): 用点云坐标
   转换到体素网格再反查 actor_ids 数组来判断命中 (保留整体, 同一虚拟ID广播)。
3. 地面/标线: 按语义标签强制保留 (Ground Protection)，不参与上面两条的可见性判断。

2026-08-27 用 diag_visibility.py 对着真实一帧数据实测量化过两处问题并修复:
- lidar_offset 之前写死 (0,0,2.5)，但 SEMANTIC_LIDAR_CONFIG 实际挂载高度是 (0,0,1.0)
  ("设置 Z=1.0m (用户指定)" 改配置时没同步改这里)——用 obj_idx 当 ground truth 扫描
  offset，命中率峰值精确落在 z=1.0 (86.4%)，当前写死值只有 34.9%，说明本该判定为
  "LiDAR 打中了"的动态物体，一半以上被坐标错位误判成"没打中"从而错误地被过滤成 free。
  现在改成从 SEMANTIC_LIDAR_CONFIG['position'] 直接算，配置改了这里跟着变，不会再漂。
  (注意这个纯平移换算只有 SEMANTIC_LIDAR_CONFIG['rotation'] 是零旋转时才成立，目前确实
  是零旋转；如果以后给LiDAR装配角度，这里要换成完整的 sensor->vehicle 变换矩阵。)
- 上面①②两条之外原来还有一条"Z<=1.0m 就无条件强制保留"的高度保护，本意是给
  GROUND_LABELS 配置错误兜底防止地面被切——但 GROUND_LABELS 现在已经修好、
  survey_actor_types.py 也验证过 CityObjectLabel 覆盖率 100%，这条兜底规则的原始目的
  已经不需要了，只剩副作用: 实测它在同一帧里让 14.8%（36万+）的非空体素仅仅因为
  "矮"就被强制保留，跟看不看得见完全无关——包括看不见的车/卡车/自行车/摩托车/行人，
  以及大量建筑墙根、树干的"幽灵矮桩"。已删除，只保留按语义标签的地面强制保留。
"""

import numpy as np
from config.occupancy_config import GROUND_LABELS, SEMANTIC_LIDAR_CONFIG

class VisibilityFilterSimple:
    def __init__(self, keep_alive_time=0.5):
        self.visibility_cache = {} # 缓存 ID，防止帧间闪烁
        self.keep_alive_time = keep_alive_time
        self.current_time = 0.0
        self.GROUND_LABELS = GROUND_LABELS

        # LiDAR 安装位置 (相对 Ego), 用于把点云坐标从 sensor 局部系换算到 ego 系。
        # 直接从传感器配置取，不再另外写死一份数值 (避免和实际挂载参数脱节)。
        _pos = SEMANTIC_LIDAR_CONFIG['position']
        self.lidar_offset = np.array([_pos['x'], _pos['y'], _pos['z']], dtype=np.float32)

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
                # 从 actor_ids 网格中读取被击中的 ID (对静态虚拟负数ID有效；对真实
                # actor 这条路径只是补充，下面直接用 obj_idx 的路径更准，见下)
                # actor_ids 存储了所有物体的 ID (动态为正，静态为负，空为0)
                hit_values = actor_ids[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]

                # 过滤掉 0 (空体素)
                unique_hits = np.unique(hit_values)
                current_hit_ids = set(unique_hits[unique_hits != 0].tolist())

        # 2.5 真实 Actor 直接用语义LiDAR原生返回的 obj_idx 匹配 (不用点云坐标转换/
        # snap到体素网格再反查，天然精确、不受任何坐标对齐偏差影响)。obj_idx==0
        # 表示这根射线打中的是没有对应 Actor 的静态关卡几何 (道路/建筑等)，这些
        # 只能靠上面的体素坐标回查(虚拟负数ID)判断，obj_idx 帮不上忙。
        # 即使有了上面的体素坐标回查，这条路径依然有必要保留: 用真实一帧数据实测
        # (diag_visibility.py) 两条路径在正确 lidar_offset 下的一致率是 86.4%，也就是
        # 说仍有约 13.6% 真实命中的点因为体素量化边界效应在回查路径里对不上——只靠
        # 回查会让这部分本该保留的 actor 被误判成"没打中"。
        if lidar_data is not None and 'obj_idx' in lidar_data:
            obj_idx = lidar_data['obj_idx']
            real_hits = obj_idx[obj_idx != 0]
            if len(real_hits) > 0:
                current_hit_ids |= set(np.unique(real_hits).tolist())

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
            
        # Mask B: 地面/标线保护 (强制保留，不管LiDAR有没有打中)
        # 只按语义标签保护 (Road/Sidewalk/Terrain/OtherFlat, 见 GROUND_LABELS)。
        # 2026-08-27 移除了之前额外叠加的"Z<=1.0m 全部强制保留"高度兜底——那是给
        # GROUND_LABELS 配置错误兜底用的，现在 GROUND_LABELS 已经修好、
        # survey_actor_types.py 也验证过 CityObjectLabel 映射覆盖率 100%，这条
        # 兜底规则的原始目的已经不需要了。实测 (diag_visibility.py) 它在同一帧里
        # 让 14.8% 的非空体素只因为矮就被强制保留，跟看不看得见完全无关——包含看
        # 不见的车/卡车/自行车/摩托车/行人，以及建筑墙根、树干的"幽灵矮桩"，纯粹是
        # 图片和体素对不上的来源，删掉之后如果发现地面又被切了，去查 GROUND_LABELS
        # 或 CITY_OBJECT_MAPPING 有没有遗漏，而不是重新加回这条高度兜底。
        ground_mask = np.isin(occupancy, self.GROUND_LABELS)
        
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

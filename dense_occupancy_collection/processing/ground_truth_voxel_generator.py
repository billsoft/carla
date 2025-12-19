"""
Ground Truth 体素生成器
直接利用 CARLA/UE5 的 Actor Bounding Box 和地图信息生成真实的体素标签
"""

import carla
import numpy as np
import math
import logging
from dense_occupancy_collection.config.occupancy_config import (
    CARLA_TO_OCCUPANCY_MAPPING, OCCUPANCY_LABELS
)
from dense_occupancy_collection.config.actor_occupancy_mapping import (
    get_occupancy_label_from_actor
)

# 配置日志
logging.basicConfig(filename='voxel_mapping.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

class GroundTruthVoxelGenerator:
    """
    基于 Ground Truth (Bounding Box + Map) 的体素生成器
    不依赖 LiDAR 点云，直接查询世界中的 Actor 和地图信息
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
        
        self.grid_size = [
            int((x_range[1] - x_range[0]) / resolution),
            int((y_range[1] - y_range[0]) / resolution),
            int((z_range[1] - z_range[0]) / resolution)
        ]
        
    def generate(self, world, ego_vehicle, visibility_lidar_data=None):
        """
        生成一帧的体素数据

        Args:
            world: carla.World
            ego_vehicle: carla.Actor (hero vehicle)
            visibility_lidar_data: Optional[bytes] - 512线语义激光雷达原始数据，用于可见性过滤

        Returns:
            occupancy: (X, Y, Z) uint8 array - 体素类别
            actor_ids: (X, Y, Z) uint32 array - 每个体素对应的Actor ID
            mask: (X, Y, Z) uint8 array - (1=Observed, 0=Unknown) -> GT 模式下全是 1
        """
        occupancy = np.zeros(self.grid_size, dtype=np.uint8)
        actor_ids = np.zeros(self.grid_size, dtype=np.int32)  # ⭐ 新增：记录Actor ID（使用int32支持负数虚拟ID）
        
        ego_transform = ego_vehicle.get_transform()
        ego_matrix = np.array(ego_transform.get_matrix())
        
        # 1. 填充静态环境 (地面、道路)
        self._fill_static_environment(occupancy, actor_ids, world, ego_transform)

        # 2. 获取动态 Actors (车辆、行人)
        actors = world.get_actors()
        vehicles = actors.filter('vehicle.*')
        walkers = actors.filter('walker.pedestrian.*')
        all_actors = list(vehicles) + list(walkers)

        print(f"\n[体素生成] 场景中总Actor数: 车辆={len(vehicles)}, 行人={len(walkers)}")

        # 3. 遍历 Actor，光栅化 Bounding Box
        filled_actor_ids = []
        filled_by_type = {'vehicles': [], 'walkers': []}  # ⭐ 统计各类型
        filtered_by_distance = {'vehicles': 0, 'walkers': 0}  # ⭐ 统计距离过滤

        print(f"\n[行人调试] ========== 开始填充Actor到体素 ==========")

        for actor in all_actors:
            # 距离粗筛
            dist = actor.get_location().distance(ego_vehicle.get_location())

            # ⭐ 调试：记录actor类型
            is_walker = 'walker.pedestrian' in actor.type_id.lower()

            if dist > 60.0: # 略大于 grid 半径
                if is_walker:
                    filtered_by_distance['walkers'] += 1
                    print(f"  [行人调试] ID={actor.id} 距离过远被跳过 (dist={dist:.1f}m > 60m)")
                else:
                    filtered_by_distance['vehicles'] += 1
                continue

            # ⭐ 行人调试：记录处理的行人
            if is_walker:
                loc = actor.get_location()
                print(f"  [行人调试] ID={actor.id} 开始填充 (dist={dist:.1f}m, world_loc=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}))")

            self._fill_actor_bb(occupancy, actor_ids, actor, ego_matrix, is_ego=(actor.id == ego_vehicle.id))
            filled_actor_ids.append(actor.id)

            # ⭐ 按类型统计
            if is_walker:
                filled_by_type['walkers'].append(actor.id)
            else:
                filled_by_type['vehicles'].append(actor.id)

        print(f"[行人调试] ========== 填充完成 ==========")
        print(f"[行人调试] 距离过滤: 车辆 {filtered_by_distance['vehicles']} 个, 行人 {filtered_by_distance['walkers']} 个")

        # 4. 填充自车
        self._fill_actor_bb(occupancy, actor_ids, ego_vehicle, ego_matrix, is_ego=True)
        filled_actor_ids.append(ego_vehicle.id)

        print(f"[体素生成] 填充到体素的Actor IDs ({len(filled_actor_ids)}个): {sorted(filled_actor_ids)}")
        print(f"[体素生成]   车辆: {len(filled_by_type['vehicles'])}个, 行人: {len(filled_by_type['walkers'])}个")

        # 5. 可见性过滤 (如果提供了激光雷达数据)
        if visibility_lidar_data is not None:
            occupancy, actor_ids = self._apply_visibility_filter(
                occupancy, actor_ids, visibility_lidar_data, world, ego_vehicle.id
            )

        # 6. Mask (Ground Truth is fully observed)
        mask = np.ones_like(occupancy)

        return occupancy, actor_ids, mask

    def _fill_static_environment(self, occupancy, actor_ids, world, ego_transform):
        """
        填充静态环境 (Road, Ground, Sidewalk)
        由于全图 RayCast 太慢，这里采用基于 Map 的启发式方法：
        1. 假设 Z < 0.2 (相对于车轮接地处) 为地面层
        2. 查询 Map 区分 Road 和 Ground (Sidewalk/Terrain)
        3. 获取静态物体 (Buildings, Street Lights, etc.) 的 Bounding Box 并填充
        """
        map_instance = world.get_map()
        ego_location = ego_transform.location
        ego_matrix = np.array(ego_transform.get_matrix())
        
        # --- A. 地面与道路 (基于高度启发式) ---
        
        # 计算 Ego 在世界坐标系中的 Z (通常路面 Z=0，Ego Z > 0)
        # 我们假设路面在 Z_world = 0 附近 (Town10)
        # 或者更严谨地，查询 Ego 所在处的路面高度
        
        start_waypoint = map_instance.get_waypoint(ego_location, project_to_road=True, lane_type=carla.LaneType.Any)
        ground_z_world = start_waypoint.transform.location.z if start_waypoint else 0.0
        
        # 1. 基础填充：将 Z < 0.2 的所有体素设为 Ground (12)
        # 找到 Z < 0.2 对应的 grid index
        z_threshold = 0.2
        gz_max = int((z_threshold - self.z_range[0]) / self.resolution)
        gz_max = max(0, min(self.grid_size[2], gz_max))

        if gz_max > 0:
            occupancy[:, :, :gz_max] = 12 # Ground (terrain类别)
            actor_ids[:, :, :gz_max] = -1012  # ⭐ 地面虚拟ID: -(12+1000)=-1012
            
        # 2. 区分 Road (9) 和 Sidewalk (11)
        # 为了性能，我们不逐个体素查询，而是按步长采样，然后插值？
        # 或者只在 ego 周围 40m 内精细查询
        
        # 这里的实现：遍历 grid 的 (x, y)，步长为 2 (即 1.0m 分辨率)，查询 Map
        step = 2
        
        # 获取 Grid 的 X, Y 坐标网格 (相对于 Ego)
        x_indices = np.arange(0, self.grid_size[0], step)
        y_indices = np.arange(0, self.grid_size[1], step)
        
        # 转换到物理坐标
        x_coords = self.x_range[0] + (x_indices + 0.5) * self.resolution
        y_coords = self.y_range[0] + (y_indices + 0.5) * self.resolution
        
        # 构建网格点 (N, 3) in Ego
        xv, yv = np.meshgrid(x_coords, y_coords, indexing='ij')
        zv = np.zeros_like(xv) # Z=0 平面
        
        points_ego = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)
        
        # Transform to World
        # Homogeneous
        points_ego_h = np.concatenate([points_ego, np.ones((points_ego.shape[0], 1))], axis=1)
        points_world_h = points_ego_h @ ego_matrix.T
        points_world = points_world_h[:, :3]
        
        # 批量查询 Map (Python API 只能循环)
        # 限制范围：只查 60m 内的
        # 距离判断
        dists = np.linalg.norm(points_world[:, :2] - np.array([ego_location.x, ego_location.y]), axis=1)
        valid_mask = dists < 60.0 # 只更新 60m 内的地面
        
        valid_indices = np.where(valid_mask)[0]
        
        # 循环查询
        # 这是一个性能瓶颈，50m 半径约覆盖 2500-10000 个点 (取决于 step)
        # step=2 (1m), 100x100m = 10000 点。
        # 10000 次 get_waypoint 耗时约 0.2-0.5s，可以接受
        
        road_indices = []
        sidewalk_indices = []
        
        for idx in valid_indices:
            pw = points_world[idx]
            loc = carla.Location(x=pw[0], y=pw[1], z=pw[2])
            
            # project_to_road=True 会找到最近的路
            # 我们需要判断距离是否够近
            wp = map_instance.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            
            if wp:
                # 检查水平距离
                # wp.transform.location 是路中心
                # lane_width
                lane_width = wp.lane_width
                wp_loc = wp.transform.location
                
                dist_xy = math.sqrt((loc.x - wp_loc.x)**2 + (loc.y - wp_loc.y)**2)
                
                if dist_xy < (lane_width / 2.0 + 0.5): # 稍微宽容一点
                    road_indices.append(idx)
                    continue
            
            # 如果不是 Driving，检查 Sidewalk (Sidewalk 支持有限，尝试用 Raycast?)
            # 简单起见，非 Road 且 Z < 0.2 的都保留为 Ground (12)
            # 或者如果有 Sidewalk waypoint (LaneType.Sidewalk)
            wp_sw = map_instance.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Sidewalk)
            if wp_sw:
                lane_width = wp_sw.lane_width
                wp_loc = wp_sw.transform.location
                dist_xy = math.sqrt((loc.x - wp_loc.x)**2 + (loc.y - wp_loc.y)**2)
                if dist_xy < (lane_width / 2.0 + 0.5):
                     sidewalk_indices.append(idx)
        
        # 更新 Grid
        # 我们根据 road_indices 反推 grid 坐标
        # idx 是 points_ego (flattened meshgrid) 的索引
        
        # 恢复 (ix, iy) 索引
        # meshgrid shape: (len(x_indices), len(y_indices))
        nx = len(x_indices)
        ny = len(y_indices)
        
        # Road
        for idx in road_indices:
            # unravelling index
            ix_sub = idx // ny
            iy_sub = idx % ny
            
            ix_base = x_indices[ix_sub]
            iy_base = y_indices[iy_sub]
            
            # 填充 step x step 的区域
            # occupancy[ix_base:ix_base+step, iy_base:iy_base+step, :gz_max] = 9 # Road
            
            # 安全切片
            ix_end = min(self.grid_size[0], ix_base + step)
            iy_end = min(self.grid_size[1], iy_base + step)

            occupancy[ix_base:ix_end, iy_base:iy_end, :gz_max] = 11 # driveable_surface (11)
            actor_ids[ix_base:ix_end, iy_base:iy_end, :gz_max] = -1011  # 道路虚拟ID

        # Sidewalk
        for idx in sidewalk_indices:
            ix_sub = idx // ny
            iy_sub = idx % ny
            ix_base = x_indices[ix_sub]
            iy_base = y_indices[iy_sub]

            ix_end = min(self.grid_size[0], ix_base + step)
            iy_end = min(self.grid_size[1], iy_base + step)

            occupancy[ix_base:ix_end, iy_base:iy_end, :gz_max] = 13 # sidewalk (13)
            actor_ids[ix_base:ix_end, iy_base:iy_end, :gz_max] = -1013  # 人行道虚拟ID
            
        # --- B. 静态物体 (建筑物, 交通标志, 杆等) ---
        # 使用 world.get_environment_objects() 获取更详细的静态物体信息 (ID, Transform, BBox)
        # 替代旧的 get_level_bbs，以支持实例级可见性过滤
        
        # 感兴趣的静态物体类型映射
        # Label -> Occupancy Label
        static_type_mapping = {
            carla.CityObjectLabel.Buildings: 15,      # manmade
            carla.CityObjectLabel.Fences: 1,          # barrier
            carla.CityObjectLabel.TrafficLight: 15,   # manmade
            carla.CityObjectLabel.TrafficSigns: 15,   # manmade
            carla.CityObjectLabel.Poles: 15,          # manmade
            carla.CityObjectLabel.Vegetation: 16,     # vegetation
            carla.CityObjectLabel.Walls: 15,          # manmade
            carla.CityObjectLabel.Other: 17,          # general_object
            carla.CityObjectLabel.Static: 17,         # general_object
            carla.CityObjectLabel.Dynamic: 17,        # general_object
            carla.CityObjectLabel.Bridge: 2,          # construction (bridge -> 2/15?) 映射为 manmade 或 barrier? 这里选 15
            carla.CityObjectLabel.GuardRail: 1,       # barrier
            carla.CityObjectLabel.RailTrack: 17,      # general_object
        }
        
        # 获取所有环境物体
        env_objs = world.get_environment_objects(carla.CityObjectLabel.Any)
        
        # 预计算 Ego 逆矩阵
        try:
            ego_matrix_inv = np.linalg.inv(ego_matrix)
        except np.linalg.LinAlgError:
            return

        count_filled = 0
        
        for i, obj in enumerate(env_objs):
            # 1. 类型过滤
            if obj.type not in static_type_mapping:
                continue
                
            occ_label = static_type_mapping[obj.type]
            
            # 2. 距离过滤
            # obj.transform.location 是世界坐标
            dist = obj.transform.location.distance(ego_location)
            if dist > 60.0:
                continue
                
            # 3. 获取 Bounding Box 顶点 (World Frame)
            # 经过调试发现：UE5 CARLA 0.10.0 中，EnvironmentObject.bounding_box 已经是世界坐标 (World Space AABB)
            # 与 get_level_bbs 返回的一致。
            # 因此，不能再应用 obj.transform，否则会导致双重变换（物体飞到天上或消失）
            bb = obj.bounding_box
            
            # 使用 Identity Transform 获取世界顶点 (因为 BB 已经是 World AABB)
            verts_world = bb.get_world_vertices(carla.Transform())
            
            # 4. 转换到 Ego Frame
            verts_world_np = np.array([[v.x, v.y, v.z, 1.0] for v in verts_world]).T
            verts_ego_np = ego_matrix_inv @ verts_world_np
            
            xs_ego = verts_ego_np[0, :]
            ys_ego = verts_ego_np[1, :]
            zs_ego = verts_ego_np[2, :]
            
            # 5. 计算 Grid 范围
            min_ix = int(np.floor((np.min(xs_ego) - self.x_range[0]) / self.resolution))
            max_ix = int(np.ceil((np.max(xs_ego) - self.x_range[0]) / self.resolution))
            min_iy = int(np.floor((np.min(ys_ego) - self.y_range[0]) / self.resolution))
            max_iy = int(np.ceil((np.max(ys_ego) - self.y_range[0]) / self.resolution))
            min_iz = int(np.floor((np.min(zs_ego) - self.z_range[0]) / self.resolution))
            max_iz = int(np.ceil((np.max(zs_ego) - self.z_range[0]) / self.resolution))
            
            # Clip
            min_ix = max(0, min_ix)
            max_ix = min(self.grid_size[0], max_ix)
            min_iy = max(0, min_iy)
            max_iy = min(self.grid_size[1], max_iy)
            min_iz = max(0, min_iz)
            max_iz = min(self.grid_size[2], max_iz)
            
            if min_ix >= max_ix or min_iy >= max_iy or min_iz >= max_iz:
                continue
                
            # 6. OBB 光栅化
            # 生成 Ego Grid Points
            lx = np.linspace(self.x_range[0] + (min_ix + 0.5)*self.resolution, 
                             self.x_range[0] + (max_ix - 0.5)*self.resolution, max_ix - min_ix)
            ly = np.linspace(self.y_range[0] + (min_iy + 0.5)*self.resolution, 
                             self.y_range[0] + (max_iy - 0.5)*self.resolution, max_iy - min_iy)
            lz = np.linspace(self.z_range[0] + (min_iz + 0.5)*self.resolution, 
                             self.z_range[0] + (max_iz - 0.5)*self.resolution, max_iz - min_iz)
            
            sub_xv, sub_yv, sub_zv = np.meshgrid(lx, ly, lz, indexing='ij')
            sub_points_ego = np.stack([sub_xv, sub_yv, sub_zv, np.ones_like(sub_xv)], axis=-1).reshape(-1, 4)
            
            # Ego -> World -> Local
            # T_obj_inv * T_ego * P_ego
            # 注意：由于 bb 已经是 World AABB，我们不需要转换到 Local Object Space
            # 我们只需要检查 World Points 是否在 World AABB 内
            # 也就是检查 abs(P_world - BB_Center) <= BB_Extent
            
            # Ego -> World
            sub_points_world_h = sub_points_ego @ ego_matrix.T
            sub_points_world = sub_points_world_h[:, :3]
            
            # World AABB Check
            # bb.location is World Center
            diff = np.abs(sub_points_world - np.array([bb.location.x, bb.location.y, bb.location.z]))
            in_x = diff[:, 0] <= bb.extent.x
            in_y = diff[:, 1] <= bb.extent.y
            in_z = diff[:, 2] <= bb.extent.z
            
            mask_in = in_x & in_y & in_z
            
            if not np.any(mask_in):
                continue
            
            # 7. 填充并应用地面保护
            nx, ny, nz = max_ix - min_ix, max_iy - min_iy, max_iz - min_iz
            mask_reshaped = mask_in.reshape(nx, ny, nz)
            
            roi = occupancy[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz]
            roi_ids = actor_ids[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz]
            
            # =========================================================
            # FIX: 地面保护机制 (Ground Protection for Static Objects)
            # =========================================================
            GROUND_LABELS = [11, 12, 13, 14]
            is_ground = np.isin(roi, GROUND_LABELS)
            final_mask = mask_reshaped & (~is_ground)
            
            if not np.any(final_mask):
                continue
            
            # 8. 分配唯一虚拟 ID
            # 使用 obj.id 会导致 int32 溢出 (CARLA EnvironmentObject ID 是 64位哈希)
            # 这里使用枚举索引生成唯一的小整数 ID
            # 加上 10000 偏移量，避免与潜在的小负数 ID 冲突
            virtual_id = -(i + 10000)
            
            roi[final_mask] = occ_label
            roi_ids[final_mask] = virtual_id
            
            occupancy[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz] = roi
            actor_ids[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz] = roi_ids
            
            count_filled += 1
            
        print(f"[体素生成] 已填充静态环境物体: {count_filled} 个")
                
        # --- C. 记录未映射物体 (Debug) ---
        # 记录每帧遇到的 CityObjectLabel 类型和 Actor 类型
        # 为了避免日志爆炸，我们可以维护一个集合，并在程序结束或每隔 N 帧打印一次
        # 这里为了实时性，直接打印
        
        # 实际生产中建议写入专门的日志文件
        # print(f"DEBUG: Processing {city_label} -> {occ_label}")

    def _get_adaptive_extent(self, bb, occ_label, actor):
        """
        根据对象类型自适应调整BoundingBox的填充范围

        Args:
            bb: carla.BoundingBox对象
            occ_label: Occupancy标签 (0-17)
            actor: carla.Actor对象

        Returns:
            (extent_x, extent_y, extent_z): 调整后的extent
        """
        original_x, original_y, original_z = bb.extent.x, bb.extent.y, bb.extent.z

        # 15: manmade (建筑、杆、标志等)
        if occ_label == 15:
            # 检查是否是细长物体（杆、标志杆等）
            # 特征：Z向很高，XY向很小
            if original_z > 2.0 and max(original_x, original_y) < 0.5:
                # 杆状物体：使用圆柱近似，收缩到半径
                radius = max(original_x, original_y) * 0.5  # 收缩50%
                return radius, radius, original_z

            # 交通标志：通常是薄板 + 杆
            # 特征：某一维度特别薄
            min_dim = min(original_x, original_y, original_z)
            if min_dim < 0.1:
                # 保持薄维度，其他维度收缩70%
                return (
                    original_x * 0.7 if original_x > 0.1 else original_x,
                    original_y * 0.7 if original_y > 0.1 else original_y,
                    original_z * 0.7 if original_z > 0.1 else original_z
                )

        # 8: traffic_cone (交通锥)
        elif occ_label == 8:
            # 锥形物体：底部圆形，收缩XY向
            return original_x * 0.6, original_y * 0.6, original_z * 0.9

        # 1: barrier (隔离栏)
        elif occ_label == 1:
            # 条状物体：通常某一维度很长
            # 保持长维度，收缩短维度
            dims = [original_x, original_y, original_z]
            max_dim = max(dims)
            return (
                original_x if original_x >= max_dim * 0.8 else original_x * 0.7,
                original_y if original_y >= max_dim * 0.8 else original_y * 0.7,
                original_z if original_z >= max_dim * 0.8 else original_z * 0.8
            )

        # 7: pedestrian (行人)
        elif occ_label == 7:
            # 行人BoundingBox实际就是胶囊体近似，已经比较准确
            # ⭐ 行人本来就很小(0.19m×0.19m)，不应该收缩，否则在0.2m分辨率下几乎看不见
            # 保持原始大小，甚至可以轻微扩大以确保可见性
            return original_x * 1.1, original_y * 1.1, original_z * 1.0

        # 2: bicycle, 6: motorcycle (自行车、摩托车)
        elif occ_label in [2, 6]:
            # 细长物体，收缩宽度
            return original_x * 0.7, original_y * 0.7, original_z * 0.9

        # 4: car, 3: bus, 10: truck, 5: construction_vehicle (车辆)
        elif occ_label in [4, 3, 10, 5]:
            # 车辆BoundingBox通常比较准确，保留大部分
            return original_x * 0.9, original_y * 0.9, original_z * 0.9

        # 16: vegetation (植被)
        elif occ_label == 16:
            # 树木、灌木：不规则形状，保守填充中心区域
            return original_x * 0.6, original_y * 0.6, original_z * 0.7

        # 其他类型：默认轻微收缩
        else:
            return original_x * 0.85, original_y * 0.85, original_z * 0.85

    def _fill_actor_bb(self, occupancy, actor_ids_grid, actor, ego_matrix, is_ego=False):
        """
        Helper to rasterize an actor's bounding box into the occupancy grid

        Args:
            occupancy: 体素类别数组
            actor_ids_grid: 体素Actor ID数组
            actor: CARLA Actor对象
            ego_matrix: Ego变换矩阵
            is_ego: 是否是自车
        """
        try:
            bb = actor.bounding_box
            actor_transform = actor.get_transform()
        except:
            return # Actor might be dead

        # Logging for debug
        # print(f"DEBUG: Actor {actor.type_id} -> Semantic Tag {actor.semantic_tags}")

        # Ego Matrix Inverse (World -> Ego)
        try:
            m_inv = np.linalg.inv(ego_matrix)
        except np.linalg.LinAlgError:
            return

        # Get 8 corners in World
        verts_world = bb.get_world_vertices(actor_transform)
        if not verts_world:
            return

        verts_world_np = np.array([[v.x, v.y, v.z, 1.0] for v in verts_world]).T

        # Transform to Ego
        verts_ego_np = m_inv @ verts_world_np

        xs_ego = verts_ego_np[0, :]
        ys_ego = verts_ego_np[1, :]
        zs_ego = verts_ego_np[2, :]

        # Grid Indices Range
        min_ix = int(np.floor((np.min(xs_ego) - self.x_range[0]) / self.resolution))
        max_ix = int(np.ceil((np.max(xs_ego) - self.x_range[0]) / self.resolution))
        min_iy = int(np.floor((np.min(ys_ego) - self.y_range[0]) / self.resolution))
        max_iy = int(np.ceil((np.max(ys_ego) - self.y_range[0]) / self.resolution))
        min_iz = int(np.floor((np.min(zs_ego) - self.z_range[0]) / self.resolution))
        max_iz = int(np.ceil((np.max(zs_ego) - self.z_range[0]) / self.resolution))

        # Clip
        min_ix = max(0, min_ix)
        max_ix = min(self.grid_size[0], max_ix)
        min_iy = max(0, min_iy)
        max_iy = min(self.grid_size[1], max_iy)
        min_iz = max(0, min_iz)
        max_iz = min(self.grid_size[2], max_iz)

        # 行人调试：记录边界框裁剪前后
        is_walker = 'walker.pedestrian' in actor.type_id.lower()
        if is_walker:
            original_grid_indices = (min_ix, max_ix, min_iy, max_iy, min_iz, max_iz)

        if min_ix >= max_ix or min_iy >= max_iy or min_iz >= max_iz:
            if is_walker:
                print(f"  [行人调试] ID={actor.id} 在网格范围外被跳过 - Grid范围: ix=[{min_ix},{max_ix}), iy=[{min_iy},{max_iy}), iz=[{min_iz},{max_iz})")
            return

        # Prepare Sub-grid for OBB check
        lx = np.linspace(self.x_range[0] + (min_ix + 0.5)*self.resolution, 
                         self.x_range[0] + (max_ix - 0.5)*self.resolution, max_ix - min_ix)
        ly = np.linspace(self.y_range[0] + (min_iy + 0.5)*self.resolution, 
                         self.y_range[0] + (max_iy - 0.5)*self.resolution, max_iy - min_iy)
        lz = np.linspace(self.z_range[0] + (min_iz + 0.5)*self.resolution, 
                         self.z_range[0] + (max_iz - 0.5)*self.resolution, max_iz - min_iz)
        
        sub_xv, sub_yv, sub_zv = np.meshgrid(lx, ly, lz, indexing='ij')
        
        # Flatten
        sub_points_ego = np.stack([sub_xv, sub_yv, sub_zv, np.ones_like(sub_xv)], axis=-1).reshape(-1, 4)
        
        # Transform Ego -> World -> Actor Local
        # P_local = T_actor_inv * P_world
        # P_world = T_ego * P_ego
        # P_local = T_actor_inv * T_ego * P_ego
        
        box_matrix = np.array(actor_transform.get_matrix())
        try:
            box_matrix_inv = np.linalg.inv(box_matrix)
        except np.linalg.LinAlgError:
            return
            
        transform_matrix = box_matrix_inv @ ego_matrix
        
        points_in_actor = sub_points_ego @ transform_matrix.T # (N, 4)
        
        # Subtract bb.location (which is local offset in actor frame)
        rel_x = points_in_actor[:, 0] - bb.location.x
        rel_y = points_in_actor[:, 1] - bb.location.y
        rel_z = points_in_actor[:, 2] - bb.location.z
        
        # Determine Label first (needed for adaptive extent)
        if is_ego:
            occ_label = 4  # Car (Default for Ego)
        else:
            occ_label = get_occupancy_label_from_actor(actor)

        # Adaptive BoundingBox extent based on object type
        # 根据对象类型自适应调整包围盒填充范围
        extent_x, extent_y, extent_z = self._get_adaptive_extent(
            bb, occ_label, actor
        )

        # 行人调试：记录extent信息
        if is_walker:
            print(f"  [行人调试] ID={actor.id} BBox原始extent=({bb.extent.x:.2f}, {bb.extent.y:.2f}, {bb.extent.z:.2f}), "
                  f"自适应extent=({extent_x:.2f}, {extent_y:.2f}, {extent_z:.2f})")

        # Check Extents with adaptive shrinking
        in_x = np.abs(rel_x) <= extent_x
        in_y = np.abs(rel_y) <= extent_y
        in_z = np.abs(rel_z) <= extent_z

        mask_in = in_x & in_y & in_z

        voxel_count = np.sum(mask_in)

        if not np.any(mask_in):
            if is_walker:
                print(f"  [行人调试] ID={actor.id} 没有体素通过extent检查 - 潜在网格: {(max_ix - min_ix) * (max_iy - min_iy) * (max_iz - min_iz)} 个")
            return

        # 行人调试：记录填充的体素数
        if is_walker:
            print(f"  [行人调试] ID={actor.id} ✓ 成功填充 {voxel_count} 个体素 (occupancy_label={occ_label})")

        # Fill (label already determined above)
        nx, ny, nz = max_ix - min_ix, max_iy - min_iy, max_iz - min_iz
        mask_reshaped = mask_in.reshape(nx, ny, nz)

        # 获取Actor ID
        current_actor_id = actor.id  # ⭐ 真实的CARLA Actor ID

        roi = occupancy[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz]
        roi_ids = actor_ids_grid[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz]

        # =========================================================
        # FIX: 地面保护机制 (Ground Protection)
        # 禁止 Actor BBox 覆盖已经是地面的体素
        # 解决 BBox 侵入地下导致地面出现"坑洞"的问题
        # =========================================================
        # 地面相关标签: 11(driveable), 12(other_flat), 13(sidewalk), 14(terrain)
        GROUND_LABELS = [11, 12, 13, 14]
        
        # 检查 ROI 区域内原本是否已经是地面
        is_ground = np.isin(roi, GROUND_LABELS)
        
        # 最终掩码：在 BBox 范围内 且 原本不是地面
        # 这样即使 BBox 插入地下，也不会破坏地面体素
        final_mask = mask_reshaped & (~is_ground)
        
        if not np.any(final_mask):
            return

        roi[final_mask] = occ_label
        roi_ids[final_mask] = current_actor_id  # ⭐ 记录Actor ID

        occupancy[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz] = roi
        actor_ids_grid[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz] = roi_ids

    def _apply_visibility_filter(self, occupancy, actor_ids, lidar_data, world, ego_vehicle_id):
        """
        应用可见性过滤：只保留激光雷达扫描到的物体

        核心逻辑：
        1. 解析64线语义激光雷达数据，提取可见的actor_id和tag（语义标签）
        2. 对于动态物体（obj_idx>0）：保留该actor_id的所有体素
        3. 对于静态环境（obj_idx=0）：根据tag映射到虚拟ID，只保留扫到的虚拟ID
        4. ⭐ Hero车辆始终保留（激光雷达在车顶扫不到自己）

        Args:
            occupancy: (X,Y,Z) uint8 - 体素类别数组
            actor_ids: (X,Y,Z) uint32 - 体素Actor ID数组
            lidar_data: bytes - 64线语义激光雷达原始数据
            world: carla.World对象
            ego_vehicle_id: int - Hero车辆的actor ID

        Returns:
            filtered_occupancy: 过滤后的体素类别数组
            filtered_actor_ids: 过滤后的Actor ID数组
        """
        # 1. 解析激光雷达数据
        dtype = np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('cos', np.float32),
            ('obj_idx', np.uint32),  # Actor ID
            ('tag', np.uint32)       # 语义标签 (CARLA semantic tag)
        ])
        points = np.frombuffer(lidar_data, dtype=dtype)

        # 2. 提取可见的actor ID（动态物体）+ 基于点云密度的可见性判断
        # ⭐ 关键改进：只有LiDAR点数足够多的actor才认为真正可见
        # 避免通过建筑物缝隙扫到的远处车辆被标记为"可见"

        dynamic_points = points[points['obj_idx'] > 0]  # 只看动态物体
        unique_ids, counts = np.unique(dynamic_points['obj_idx'], return_counts=True)

        # ⭐ 可见性阈值：降低到5点（极近距离的车辆可能被Hero车身严重遮挡）
        # 行人、自行车等小物体可能只有5-10个点
        # 如果阈值太高（10+），紧贴Hero车辆的车会被过滤
        MIN_POINTS_THRESHOLD = 5  # 最少5个点才算真正可见

        visible_actor_ids_set = set()
        filtered_by_density = []

        for actor_id, point_count in zip(unique_ids, counts):
            if point_count >= MIN_POINTS_THRESHOLD:
                visible_actor_ids_set.add(int(actor_id))
            else:
                filtered_by_density.append((int(actor_id), point_count))

        print(f"\n[可见性过滤] 激光雷达检测到 {len(points)} 点")
        print(f"[可见性过滤] 动态物体点数统计: {len(unique_ids)} 个actor")
        print(f"[可见性过滤] 点数充足（>={MIN_POINTS_THRESHOLD}点）: {len(visible_actor_ids_set)} 个")
        print(f"[可见性过滤] 点数不足（<{MIN_POINTS_THRESHOLD}点）被过滤: {len(filtered_by_density)} 个")
        if filtered_by_density:
            # ⭐ 显示所有被过滤的actor，方便诊断
            print(f"[可见性过滤] 点数不足的Actor（所有）: {filtered_by_density}")

        # ⭐ 强制保留Hero车辆（激光雷达在车顶扫不到自己）
        visible_actor_ids_set.add(ego_vehicle_id)
        print(f"[可见性过滤] 强制保留Hero车辆 ID={ego_vehicle_id}")

        # 3. 处理静态环境（obj_idx=0）：根据tag映射到虚拟ID
        # ⭐⭐⭐ CRITICAL FIX：静态环境需要分类处理 ⭐⭐⭐
        #
        # 问题：之前的逻辑无条件保留所有扫到的静态tag，导致：
        #   - 扫到远处建筑tag=1 → 虚拟ID -1015 → **所有建筑**都可见（错误！）
        #   - 扫到远处植被tag=9 → 虚拟ID -1016 → **所有植被**都可见（错误！）
        #
        # 新逻辑：
        #   1. 地面类型（道路、人行道）→ 不添加到可见集合，后续通过occupancy类型判断
        #   2. 大型静态物体（建筑、植被、墙）→ **不添加到可见集合**，让后续逻辑过滤
        #   3. 小型静态物体（杆、标志、围栏）→ 基于点云密度判断

        # 3. 处理静态环境（基于几何的点云匹配）
        # ⭐ 核心改进：解决静态物体无法区分实例的问题
        # 方法：
        #   1. 将 LiDAR 点云转换到 Ego 坐标系（体素网格坐标系）
        #   2. 计算点云落在哪个体素中
        #   3. 查询该体素当前的 actor_id
        #   4. 如果 ID 是静态物体（负数），则认为该静态物体可见
        
        # 3.1 坐标转换 Sensor -> Ego
        # Sensor 安装在 (0, 0, 1.5)，即 z += 1.5
        points_xyz = np.stack([points['x'], points['y'], points['z']], axis=-1)
        points_ego = points_xyz + np.array([0.0, 0.0, 1.5]) 

        # 3.2 提取静态点 (obj_idx == 0)
        # 其实所有点都可以用来 check，防止动态物体漏检，但动态物体已经由 obj_idx 处理了
        # 这里只关注 obj_idx == 0 的点，去“激活”静态物体
        dynamic_points_mask = points['obj_idx'] > 0
        static_points_ego = points_ego[~dynamic_points_mask]
        
        if len(static_points_ego) > 0:
            # 3.3 批量计算 Grid Indices (Vectorized)
            ix = np.floor((static_points_ego[:, 0] - self.x_range[0]) / self.resolution).astype(int)
            iy = np.floor((static_points_ego[:, 1] - self.y_range[0]) / self.resolution).astype(int)
            iz = np.floor((static_points_ego[:, 2] - self.z_range[0]) / self.resolution).astype(int)
            
            # 3.4 过滤有效范围内的点
            valid_mask = (ix >= 0) & (ix < self.grid_size[0]) & \
                         (iy >= 0) & (iy < self.grid_size[1]) & \
                         (iz >= 0) & (iz < self.grid_size[2])
            
            valid_ix = ix[valid_mask]
            valid_iy = iy[valid_mask]
            valid_iz = iz[valid_mask]
            
            # 3.5 查表获取 Hit Actor IDs
            if len(valid_ix) > 0:
                hit_ids = actor_ids[valid_ix, valid_iy, valid_iz]
                
                # 3.6 提取静态 ID (负数)
                # 忽略 -1011/-1012/-1013 (Road/Ground/Sidewalk)，它们由 GROUND_LABELS 保护
                # 我们主要关心 Buildings, Fences, Poles 等
                # 其实只要是负数都加进去也没关系，反正 Ground 也是永久可见
                hit_static_ids = hit_ids[hit_ids < 0]
                
                unique_static_ids = np.unique(hit_static_ids)
                
                # 更新可见集合
                visible_actor_ids_set.update(unique_static_ids.tolist())
                
                print(f"[可见性过滤] 几何匹配到的静态物体数: {len(unique_static_ids)}")

        # 4. 创建可见性mask - ⭐⭐⭐ 新逻辑：用户指定的简单可见性规则 ⭐⭐⭐
        #
        # 用户需求（2025-12-18）：
        #   1. 所有体素初始值 visible = False
        #   2. 地面类型(11,12,13,14) → visible = True（永久可见）
        #   3. Hero车辆 → visible = True（永久可见）
        #   4. 激光雷达检测到的ID → 该ID的所有体素 visible = True
        #   5. 最终：visible = False 的 → occupancy = 0（空气）
        #
        # 这样可以：
        #   - 过滤被遮挡的建筑物（静态环境也会被过滤）
        #   - 保护地面不出现"坑"（通过occupancy类型判断）
        #   - 保护Hero车辆（强制可见）

        # 步骤1: 所有体素初始值 = False
        visibility_mask = np.zeros(occupancy.shape, dtype=bool)

        # 步骤2: 地面相关类型永久可见（避免地面出现"坑"）
        # 11=driveable_surface, 12=other_flat, 13=sidewalk, 14=terrain
        GROUND_LABELS = [11, 12, 13, 14]
        ground_mask = np.isin(occupancy, GROUND_LABELS)
        visibility_mask[ground_mask] = True

        print(f"[可见性过滤] 地面体素（永久可见）: {np.sum(ground_mask)}")

        # 步骤3: Hero车辆永久可见
        hero_mask = (actor_ids == ego_vehicle_id)
        visibility_mask[hero_mask] = True
        print(f"[可见性过滤] Hero车辆体素（永久可见）: {np.sum(hero_mask)}")

        # 步骤4: 激光雷达检测到的ID → 所有该ID的体素可见
        lidar_detected_mask = np.isin(actor_ids, list(visible_actor_ids_set))
        visibility_mask[lidar_detected_mask] = True

        # ⭐ 统计
        all_voxel_actor_ids = np.unique(actor_ids[actor_ids > 0])  # 正数ID（真实actors）
        all_voxel_virtual_ids = np.unique(actor_ids[actor_ids < 0])  # 负数ID（虚拟actors）

        print(f"[可见性过滤] 体素中包含的真实Actor IDs: {sorted(list(all_voxel_actor_ids))}")
        print(f"[可见性过滤] 体素中真实Actor数量: {len(all_voxel_actor_ids)}")
        print(f"[可见性过滤] 体素中虚拟ID数量: {len(all_voxel_virtual_ids)}")

        # ⭐ 详细统计：哪些动态Actor被过滤了
        visible_dynamic_ids = [aid for aid in all_voxel_actor_ids if aid in visible_actor_ids_set]
        invisible_dynamic_ids = [aid for aid in all_voxel_actor_ids if aid not in visible_actor_ids_set]
        print(f"[可见性过滤] 激光雷达检测到的动态Actor IDs: {sorted(visible_dynamic_ids)}")
        print(f"[可见性过滤] 将被过滤的动态Actor IDs: {sorted(invisible_dynamic_ids)}")

        # 统计静态环境
        visible_static_ids = sorted([vid for vid in all_voxel_virtual_ids if vid in visible_actor_ids_set])
        invisible_static_ids = sorted([vid for vid in all_voxel_virtual_ids if vid not in visible_actor_ids_set])
        print(f"[可见性过滤] 激光雷达检测到的静态虚拟IDs: {visible_static_ids}")
        print(f"[可见性过滤] 将被过滤的静态虚拟IDs: {invisible_static_ids}")

        # 统计
        total_occupied = np.sum(occupancy > 0)
        visible_voxels = np.sum(visibility_mask & (occupancy > 0))
        filtered_voxels = total_occupied - visible_voxels

        print(f"[可见性过滤] 总占用体素: {total_occupied}")
        print(f"[可见性过滤] 可见体素: {visible_voxels}")
        print(f"[可见性过滤] 过滤掉: {filtered_voxels} ({filtered_voxels/total_occupied*100:.1f}%)")

        # 步骤5: 最终过滤 - visible = False 的体素 → occupancy = 0（空气）
        filtered_occupancy = occupancy.copy()
        filtered_actor_ids = actor_ids.copy()

        # ⭐⭐⭐ 关键：所有不可见的体素（无论动态还是静态）都变成空气 ⭐⭐⭐
        # ⭐⭐⭐ CRITICAL FIX: 地面类型永远不删除，即使被车辆BBox覆盖 ⭐⭐⭐
        invisible_mask = ~visibility_mask

        # 地面保护：即使invisible，只要occupancy是地面类型，就不删除
        # 这样可以避免车辆BBox底部与地面重叠时，地面被删除造成"坑"
        GROUND_LABELS = [11, 12, 13, 14]
        is_ground = np.isin(filtered_occupancy, GROUND_LABELS)

        # 最终要删除的：不可见 且 不是地面
        final_remove_mask = invisible_mask & (~is_ground)

        filtered_occupancy[final_remove_mask] = 0  # 不可见且非地面 → 空气
        filtered_actor_ids[final_remove_mask] = 0  # 清除Actor ID

        # 分类统计
        dynamic_filtered = np.sum(final_remove_mask & (actor_ids > 0))  # 动态物体被过滤
        static_filtered = np.sum(final_remove_mask & (actor_ids < 0))   # 静态环境被过滤
        ground_protected = np.sum(invisible_mask & is_ground)  # 不可见但因为是地面而保护

        print(f"[可见性过滤] 过滤掉的动态物体体素: {dynamic_filtered}")
        print(f"[可见性过滤] 过滤掉的静态环境体素（遮挡建筑等）: {static_filtered}")
        print(f"[可见性过滤] 保留的地面体素: {np.sum(is_ground)}")
        print(f"[可见性过滤] 地面被保护（不可见但保留）: {ground_protected}")

        return filtered_occupancy, filtered_actor_ids

    def save_to_npz(self, filepath, occupancy, actor_ids, mask, metadata=None):
        """
        保存体素数据到NPZ文件

        Args:
            filepath: 保存路径
            occupancy: (X,Y,Z) uint8 - 体素类别
            actor_ids: (X,Y,Z) uint32 - 体素Actor ID
            mask: (X,Y,Z) uint8 - 观测mask
            metadata: 额外元数据
        """
        save_dict = {
            'occupancy': occupancy,
            'actor_ids': actor_ids,  # ⭐ 保存Actor ID数组
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
        total_voxels = occupancy.size
        observed_voxels = np.sum(mask)
        occupied_voxels = np.sum(occupancy > 0)
        
        return {
            'total_voxels': total_voxels,
            'observed_voxels': int(observed_voxels),
            'occupied_voxels': int(occupied_voxels),
            'observation_rate': observed_voxels / total_voxels if total_voxels > 0 else 0
        }

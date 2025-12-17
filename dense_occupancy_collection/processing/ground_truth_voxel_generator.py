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
        
    def generate(self, world, ego_vehicle):
        """
        生成一帧的体素数据
        
        Args:
            world: carla.World
            ego_vehicle: carla.Actor (hero vehicle)
            
        Returns:
            occupancy: (X, Y, Z) uint8 array
            mask: (X, Y, Z) uint8 array (1=Observed, 0=Unknown) -> GT 模式下全是 1
        """
        occupancy = np.zeros(self.grid_size, dtype=np.uint8)
        
        ego_transform = ego_vehicle.get_transform()
        ego_matrix = np.array(ego_transform.get_matrix())
        
        # 1. 填充静态环境 (地面、道路)
        self._fill_static_environment(occupancy, world, ego_transform)

        # 2. 获取动态 Actors (车辆、行人)
        actors = world.get_actors()
        vehicles = actors.filter('vehicle.*')
        walkers = actors.filter('walker.pedestrian.*')
        all_actors = list(vehicles) + list(walkers)
        
        # 3. 遍历 Actor，光栅化 Bounding Box
        for actor in all_actors:
            # 距离粗筛
            dist = actor.get_location().distance(ego_vehicle.get_location())
            if dist > 60.0: # 略大于 grid 半径
                continue
            
            # 这里的 ego_matrix 是 ego -> world 还是 world -> ego?
            # get_matrix() 返回的是 Model to World (Transform Matrix)
            # 我们需要 World -> Ego 来把 actor 变换到 ego 坐标系
            # 所以需要 ego_matrix 的逆矩阵
            
            self._fill_actor_bb(occupancy, actor, ego_matrix, is_ego=(actor.id == ego_vehicle.id))

        # 4. 填充自车 (如果上面没包含自车)
        # 通常 world.get_actors() 包含自车，但在 filter 时可能会漏掉或者逻辑需要
        # 为了保险起见，再次显式填充自车
        self._fill_actor_bb(occupancy, ego_vehicle, ego_matrix, is_ego=True)

        # 5. Mask (Ground Truth is fully observed)
        mask = np.ones_like(occupancy)
        
        return occupancy, mask

    def _fill_static_environment(self, occupancy, world, ego_transform):
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
            occupancy[:, :, :gz_max] = 12 # Ground
            
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

        # Sidewalk
        for idx in sidewalk_indices:
            ix_sub = idx // ny
            iy_sub = idx % ny
            ix_base = x_indices[ix_sub]
            iy_base = y_indices[iy_sub]
            
            ix_end = min(self.grid_size[0], ix_base + step)
            iy_end = min(self.grid_size[1], iy_base + step)
            
            occupancy[ix_base:ix_end, iy_base:iy_end, :gz_max] = 13 # sidewalk (13)
            
        # --- B. 静态物体 (建筑物, 交通标志, 杆等) ---
        # 使用 world.get_level_bbs() 获取地图中的静态物体 BoundingBox
        # 这可能返回大量数据，需要距离过滤
        
        # 感兴趣的静态物体类型
        static_types = [
            (carla.CityObjectLabel.Buildings, 15),      # Building -> manmade (15)
            (carla.CityObjectLabel.Fences, 1),          # Fence -> barrier (1)
            (carla.CityObjectLabel.TrafficLight, 15),   # TrafficLight -> manmade (15)
            (carla.CityObjectLabel.TrafficSigns, 15),   # TrafficSign -> manmade (15)
            (carla.CityObjectLabel.Poles, 15),          # Poles -> manmade (15)
            (carla.CityObjectLabel.Vegetation, 16),     # Vegetation -> vegetation (16)
            (carla.CityObjectLabel.Walls, 15),          # Walls -> manmade (15)
            (carla.CityObjectLabel.Other, 17),          # Other -> general_object (17)
            (carla.CityObjectLabel.Static, 17),         # Static -> general_object (17)
            (carla.CityObjectLabel.Dynamic, 17),        # Dynamic -> general_object (17)
        ]
        
        # 预计算 Ego 逆矩阵
        try:
            ego_matrix_inv = np.linalg.inv(ego_matrix)
        except np.linalg.LinAlgError:
            return

        for city_label, occ_label in static_types:
            bbs = world.get_level_bbs(city_label)
            
            for bb in bbs:
                # bb.location 是世界坐标 (因为是 Level BB)
                # 距离粗筛
                dist = bb.location.distance(ego_location)
                if dist > 60.0:
                    continue
                
                # Level BB 的 rotation 也是世界坐标
                # 我们需要构建一个假的 Actor 或者是直接用 BB 属性
                # carla.BoundingBox 没有 get_transform() 方法
                # get_world_vertices(transform) 需要传入 Transform
                # 对于 Level BB，它们是 Axis Aligned 还是有 Rotation? 
                # 文档说 Level BB 是 world coordinates.
                # get_level_bbs returns list of carla.BoundingBox.
                # 但 BoundingBox 自身只包含 local location 和 extent.
                # 实际上对于 Level Objects，CARLA API 有点模糊。
                # 通常 Level BB 是 AABB (Axis Aligned in World) 或者 OBB。
                # get_level_bbs 返回的 BB，其 location 是世界坐标中心。
                # 但是 rotation 呢？carla.BoundingBox 包含 rotation 吗？
                # 0.9.10+ BoundingBox 有 rotation 属性吗？
                # 查阅 API: BoundingBox(location, extent). No rotation.
                # 这意味着 get_level_bbs 返回的通常是 AABB (Axis Aligned Bounding Box) 或者是已经变换过的？
                # 实际上，get_level_bbs 返回的是局部坐标系的 BB 吗？
                # 不，它是 "Bounding boxes of all the objects of a certain type in the level."
                # 实际上，我们需要 Transform 才能确定位置。
                # 但是 get_level_bbs 只返回 BB。
                # 这是一个已知的 CARLA API 痛点。通常 Level BB 是假定 Rotation=(0,0,0) 的 AABB？
                # 或者它们其实是 "World Space AABB"。
                # 如果是 World Space AABB，那么 rotation=Identity, location=WorldCenter.
                
                # 让我们假设它是 World Space AABB。
                # Construct a transform for the BB
                # bb_transform = carla.Transform(bb.location, carla.Rotation(0,0,0))
                # 但是 bb.location 是 center。
                # extent 是半长。
                
                # 如果是 AABB，我们不需要复杂的旋转变换
                # 直接转换 min/max 到 ego 坐标系 (带旋转)
                
                # 世界坐标系下的 8 个顶点
                # min_v = bb.location - bb.extent
                # max_v = bb.location + bb.extent
                # 这只有在它是 AABB 时才成立。
                
                # 假设 Rotation=0 (AABB)
                # bb_transform = carla.Transform(carla.Location(0,0,0), carla.Rotation(0,0,0))
                # verts = bb.get_world_vertices(bb_transform) # 这会加上 bb.location?
                # 不，get_world_vertices(tf): result = tf * (local_verts + bb.location)
                # 如果 bb.location 已经是世界坐标，那么 tf 应该是 Identity?
                # 实际上 get_level_bbs 返回的 BB，location 是世界坐标。
                # 所以我们用 Identity Transform 来获取顶点
                
                identity_tf = carla.Transform() # (0,0,0), (0,0,0)
                # 但是 bb.location 是 center。
                # get_world_vertices 会做: point = transform.location + transform.rotation * (bb.location + extent * sign)
                # 如果 transform 是 identity: point = bb.location + extent * sign
                # 这正是我们想要的 World AABB vertices。
                
                verts_world = bb.get_world_vertices(identity_tf)
                
                # 现在有了 8 个世界坐标点，转换到 Ego 坐标系
                verts_world_np = np.array([[v.x, v.y, v.z, 1.0] for v in verts_world]).T
                verts_ego_np = ego_matrix_inv @ verts_world_np
                
                xs_ego = verts_ego_np[0, :]
                ys_ego = verts_ego_np[1, :]
                zs_ego = verts_ego_np[2, :]
                
                # 找出 Grid 范围
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
                
                # 对于静态物体 (通常是长方体)，我们简单地填充其 3D 边界框覆盖的范围
                # 由于它们是 AABB (World)，转到 Ego 后可能是 OBB
                # 我们需要做 OBB 检查
                
                # 优化：对于建筑物等大物体，直接填充整个 Bounding Box 覆盖的范围 (Rasterize)
                # 检查每个体素中心是否在 OBB 内
                
                # OBB Check: Point P in Box?
                # Box is AABB in World Frame.
                # So just check if P_world is in [min_world, max_world]
                
                # 生成 Ego Grid Points
                lx = np.linspace(self.x_range[0] + (min_ix + 0.5)*self.resolution, 
                                 self.x_range[0] + (max_ix - 0.5)*self.resolution, max_ix - min_ix)
                ly = np.linspace(self.y_range[0] + (min_iy + 0.5)*self.resolution, 
                                 self.y_range[0] + (max_iy - 0.5)*self.resolution, max_iy - min_iy)
                lz = np.linspace(self.z_range[0] + (min_iz + 0.5)*self.resolution, 
                                 self.z_range[0] + (max_iz - 0.5)*self.resolution, max_iz - min_iz)
                
                sub_xv, sub_yv, sub_zv = np.meshgrid(lx, ly, lz, indexing='ij')
                sub_points_ego = np.stack([sub_xv, sub_yv, sub_zv, np.ones_like(sub_xv)], axis=-1).reshape(-1, 4)
                
                # Transform Ego -> World
                # P_world = T_ego * P_ego
                sub_points_world_h = sub_points_ego @ ego_matrix.T
                sub_points_world = sub_points_world_h[:, :3]
                
                # Check AABB in World
                # bb.contains(point, transform=Identity) works for AABB?
                # Yes, if we use Identity transform.
                # Or manual check: abs(p - center) < extent
                
                diff = np.abs(sub_points_world - np.array([bb.location.x, bb.location.y, bb.location.z]))
                in_x = diff[:, 0] <= bb.extent.x
                in_y = diff[:, 1] <= bb.extent.y
                in_z = diff[:, 2] <= bb.extent.z
                
                mask_in = in_x & in_y & in_z
                
                if not np.any(mask_in):
                    continue
                
                # Fill
                nx, ny, nz = max_ix - min_ix, max_iy - min_iy, max_iz - min_iz
                mask_reshaped = mask_in.reshape(nx, ny, nz)
                
                roi = occupancy[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz]
                
                # 如果当前是空的 (0) 或者是地/路 (11,12,13)，则覆盖
                # 这样可以避免覆盖已有的动态物体 (车辆行人通常优先级更高)
                # 或者是简单的覆盖逻辑
                # 考虑到静态物体通常在动态物体下面或周围
                # 我们希望静态物体不要覆盖车辆
                # 所以: if current <= 14 (static/ground) -> overwrite
                # 实际上 动态物体是在 _fill_actor_bb 里处理的，那是后处理，会覆盖静态物体
                # 所以这里直接赋值即可
                
                roi[mask_reshaped] = occ_label
                occupancy[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz] = roi
                
        # --- C. 记录未映射物体 (Debug) ---
        # 记录每帧遇到的 CityObjectLabel 类型和 Actor 类型
        # 为了避免日志爆炸，我们可以维护一个集合，并在程序结束或每隔 N 帧打印一次
        # 这里为了实时性，直接打印
        
        # 实际生产中建议写入专门的日志文件
        # print(f"DEBUG: Processing {city_label} -> {occ_label}")

    def _fill_actor_bb(self, occupancy, actor, ego_matrix, is_ego=False):
        """
        Helper to rasterize an actor's bounding box into the occupancy grid
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
        
        if min_ix >= max_ix or min_iy >= max_iy or min_iz >= max_iz:
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
        
        # Check Extents
        in_x = np.abs(rel_x) <= bb.extent.x
        in_y = np.abs(rel_y) <= bb.extent.y
        in_z = np.abs(rel_z) <= bb.extent.z
        
        mask_in = in_x & in_y & in_z
        
        if not np.any(mask_in):
            return
            
        # Determine Label
        if is_ego:
            occ_label = 4 # Car (Default for Ego)
        else:
            sem_tag = 0
            if hasattr(actor, 'semantic_tags') and actor.semantic_tags:
                sem_tag = actor.semantic_tags[0]
            
            occ_label = CARLA_TO_OCCUPANCY_MAPPING.get(sem_tag, 0)
            if occ_label == 0:
                if 'vehicle' in actor.type_id:
                    occ_label = 1
                elif 'walker' in actor.type_id:
                    occ_label = 6
        
        # Fill
        nx, ny, nz = max_ix - min_ix, max_iy - min_iy, max_iz - min_iz
        mask_reshaped = mask_in.reshape(nx, ny, nz)
        
        roi = occupancy[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz]
        roi[mask_reshaped] = occ_label
        occupancy[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz] = roi

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
        total_voxels = occupancy.size
        observed_voxels = np.sum(mask)
        occupied_voxels = np.sum(occupancy > 0)
        
        return {
            'total_voxels': total_voxels,
            'observed_voxels': int(observed_voxels),
            'occupied_voxels': int(occupied_voxels),
            'observation_rate': observed_voxels / total_voxels if total_voxels > 0 else 0
        }

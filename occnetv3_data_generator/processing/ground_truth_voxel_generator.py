"""
Ground Truth 体素生成器
直接利用 CARLA/UE5 的 Actor Bounding Box 和地图信息生成真实的体素标签
"""

import carla
import numpy as np
import math

# 不用客户端 get_matrix()/get_world_vertices()：5.5 客户端的 pitch/roll 符号与 UE 渲染相反，见 transform_utils.py
from processing.transform_utils import transform_to_matrix, bbox_world_vertices
from config.occupancy_config import (
    CARLA_TO_OCCUPANCY_MAPPING
)
from config.actor_occupancy_mapping import (
    get_occupancy_label_from_actor
)

class GroundTruthVoxelGenerator:
    """
    基于 Ground Truth (Bounding Box + Map) 的体素生成器
    不依赖 LiDAR 点云，直接查询世界中的 Actor 和地图信息
    """
    
    def __init__(self,
                 x_range=(-40.0, 40.0),  # ⭐ 更新默认值
                 y_range=(-40.0, 40.0),  # ⭐ 更新默认值
                 z_range=(-1.0, 5.4),
                 resolution=0.2,  # ⭐ 修正默认值: 0.5 -> 0.2
                 verbose=False):
        self.verbose = verbose
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution
        
        self.grid_size = [
            int((x_range[1] - x_range[0]) / resolution),
            int((y_range[1] - y_range[0]) / resolution),
            int((z_range[1] - z_range[0]) / resolution)
        ]
        
        # ⭐ 世界坐标系地面缓存
        # Key: (int(world_x/resolution), int(world_y/resolution)) - 使用体素分辨率网格索引
        # Value: {'label': 11/13/14, 'z': world_z}
        self.ground_cache = {}
        # self.cache_resolution = 0.5 # Deprecated: use self.resolution

        # ⭐ 添加验证
        expected_grid_size = [400, 400, 32]
        if self.grid_size != expected_grid_size:
            print(f"[警告] 体素网格尺寸 {self.grid_size} 与标准 {expected_grid_size} 不一致")

        
    def generate_flow(self, world, ego_vehicle, dt=0.05):
        """
        生成 3D 场景流 (Scene Flow) 和 动态掩码 (Flow Mask)

        Flow = 物体自身的世界速度旋转到当前自车坐标系 × dt (OpenOcc/Occ3D 占用流约定)：
        停着的车 flow=0，和旁边的建筑一致；自车自己的运动由 ego_motion 单独表达。
        2026-10-14 之前是"物体速度 - 自车速度"(相对速度)，结果停着的车 flow=-v_ego·dt 而
        建筑恒为 0，同一个静止世界里两种物体的流场互相矛盾。
        
        Args:
            world: carla.World
            ego_vehicle: carla.Actor (hero vehicle)
            dt: float, 帧间时间 (秒)
            
        Returns:
            flow: (3, X, Y, Z) float32 array - 位移向量 (dx, dy, dz)
            flow_mask: (X, Y, Z) bool array - 动态区域掩码
        """
        flow = np.zeros((3,) + tuple(self.grid_size), dtype=np.float32)
        flow_mask = np.zeros(self.grid_size, dtype=bool)
        
        ego_transform = ego_vehicle.get_transform()
        grid_to_world_matrix = transform_to_matrix(ego_transform)
        
        # World -> Grid (Ego) 旋转矩阵 (3x3)
        # V_ego = R_world_to_ego * V_world
        # R_world_to_ego = R_grid_to_world.T (因为是正交矩阵)
        world_to_grid_rot = grid_to_world_matrix[:3, :3].T
        
        # 获取所有动态 Actor
        actors = world.get_actors()
        vehicles = actors.filter('vehicle.*')
        walkers = actors.filter('walker.pedestrian.*')
        dynamic_actors = list(vehicles) + list(walkers)
        
        # 遍历动态 Actor
        for actor in dynamic_actors:
            # 距离粗筛
            if actor.get_location().distance(ego_vehicle.get_location()) > 60.0:
                continue
                
            # 1. 获取 Actor 速度 (World Frame)
            v_actor_world = actor.get_velocity()
            v_actor_vec_world = np.array([v_actor_world.x, v_actor_world.y, v_actor_world.z])
            
            # 2. 物体自身速度 (不减自车速度) 旋转到当前自车坐标系，× dt 得到位移
            displacement = (world_to_grid_rot @ v_actor_vec_world) * dt
            
            # 5. 填充 Flow Grid (Rasterize Actor BBox)
            # 复用 _fill_actor_bb 的逻辑，但不修改 occupancy，而是填充 flow
            # 为了代码复用，我们将核心光栅化逻辑提取出来，或者简化处理
            # 这里简化处理：直接光栅化 BBox
            
            try:
                bb = actor.bounding_box
                actor_transform = actor.get_transform()
            except:
                continue

            # World -> Grid Transform
            try:
                world_to_grid_matrix = np.linalg.inv(grid_to_world_matrix)
            except np.linalg.LinAlgError:
                continue

            verts_world = bbox_world_vertices(bb, actor_transform)  # [8, 3]
            
            verts_world_np = np.concatenate([verts_world, np.ones((8, 1))], axis=1).T
            verts_grid_np = world_to_grid_matrix @ verts_world_np
            
            xs, ys, zs = verts_grid_np[0, :], verts_grid_np[1, :], verts_grid_np[2, :]
            
            min_ix = max(0, int(np.floor((np.min(xs) - self.x_range[0]) / self.resolution)))
            max_ix = min(self.grid_size[0], int(np.ceil((np.max(xs) - self.x_range[0]) / self.resolution)))
            min_iy = max(0, int(np.floor((np.min(ys) - self.y_range[0]) / self.resolution)))
            max_iy = min(self.grid_size[1], int(np.ceil((np.max(ys) - self.y_range[0]) / self.resolution)))
            min_iz = max(0, int(np.floor((np.min(zs) - self.z_range[0]) / self.resolution)))
            max_iz = min(self.grid_size[2], int(np.ceil((np.max(zs) - self.z_range[0]) / self.resolution)))
            
            if min_ix >= max_ix or min_iy >= max_iy or min_iz >= max_iz:
                continue
                
            # 简化版 OBB 检测 (为了速度，直接使用 AABB 填充 Flow)
            # 如果需要更精确，可以复制 _fill_actor_bb 的 OBB 逻辑
            # 考虑到 Flow 主要是用于运动补偿，AABB 覆盖通常是可以接受的，
            # 或者我们可以复用 occupancy 的 mask (如果先生成了 occupancy)
            # 但这里是独立的 flow 生成。
            
            # 使用简单的 AABB 填充
            # flow[:, min_ix:max_ix, min_iy:max_iy, min_iz:max_iz] = displacement[:, None, None, None]
            # flow_mask[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz] = True
            
            # 为了更精确，我们还是使用 OBB check (复制简化版)
            lx = self.x_range[0] + (np.arange(min_ix, max_ix) + 0.5) * self.resolution
            ly = self.y_range[0] + (np.arange(min_iy, max_iy) + 0.5) * self.resolution
            lz = self.z_range[0] + (np.arange(min_iz, max_iz) + 0.5) * self.resolution
            sub_xv, sub_yv, sub_zv = np.meshgrid(lx, ly, lz, indexing='ij')
            sub_points_grid = np.stack([sub_xv, sub_yv, sub_zv, np.ones_like(sub_xv)], axis=-1).reshape(-1, 4)
            
            sub_points_world = sub_points_grid @ grid_to_world_matrix.T
            box_matrix_inv = np.linalg.inv(transform_to_matrix(actor_transform))
            points_in_actor = sub_points_world[:, :3] @ box_matrix_inv[:3, :3].T + box_matrix_inv[:3, 3]
            
            in_x = np.abs(points_in_actor[:, 0] - bb.location.x) <= bb.extent.x
            in_y = np.abs(points_in_actor[:, 1] - bb.location.y) <= bb.extent.y
            in_z = np.abs(points_in_actor[:, 2] - bb.location.z) <= bb.extent.z
            mask_in = (in_x & in_y & in_z).reshape(max_ix-min_ix, max_iy-min_iy, max_iz-min_iz)
            
            if np.any(mask_in):
                # 填充 Flow
                # displacement shape: (3,)
                # target slice shape: (3, nx, ny, nz)
                flow_slice = flow[:, min_ix:max_ix, min_iy:max_iy, min_iz:max_iz]
                # Broadcast displacement to mask
                for i in range(3):
                    flow_slice[i][mask_in] = displacement[i]
                
                flow_mask_slice = flow_mask[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz]
                flow_mask_slice[mask_in] = True

        return flow, flow_mask

    def generate(self, world, ego_vehicle):
        """
        生成一帧的体素数据

        Args:
            world: carla.World
            ego_vehicle: carla.Actor (hero vehicle)

        Returns:
            occupancy: (X, Y, Z) uint8 array - 体素类别
            actor_ids: (X, Y, Z) uint32 array - 每个体素对应的Actor ID
        """
        occupancy = np.zeros(self.grid_size, dtype=np.uint8)
        actor_ids = np.zeros(self.grid_size, dtype=np.int32)  # ⭐ 新增：记录Actor ID（使用int32支持负数虚拟ID）
        
        ego_transform = ego_vehicle.get_transform()

        # ⭐ 构建变换矩阵 (包含旋转，确保 Grid 是 Ego-Aligned)
        # 体素网格坐标系: 以 Ego 为原点，X轴向前，Y轴向右/左，Z轴向上
        # 这与 Viewer 和 Dataset 的预期一致
        grid_to_world_matrix = transform_to_matrix(ego_transform)
        
        # 1. 填充静态环境 (地面、道路)
        self._fill_static_environment(occupancy, actor_ids, world, ego_transform, grid_to_world_matrix)

        # 2. 获取动态 Actors (车辆、行人、静态道具)
        actors = world.get_actors()
        vehicles = actors.filter('vehicle.*')
        walkers = actors.filter('walker.pedestrian.*')
        props = actors.filter('static.prop.*')
        traffic = []  # 见下方 2026-08-28 说明: 不再把 traffic.* 当动态 Actor 光栅化

        # 2026-08-28 不再把 `traffic.*` (traffic_light/traffic_sign/stop/yield/...)
        # 当普通动态 Actor 用它们的 actor.bounding_box 光栅化。根因排查 (用户报告"车
        # 左后方路面出现一个图片里没有的白块"，加了 --debug-actor-ids 调试开关直接
        # 定位到是哪个 actor)：traffic.traffic_light 的 bounding_box 实测和它自己的
        # transform.location 偏移达 9~12m（例如 id=9: loc=(-34.4,-51.0,0.3) 但
        # bb.location=(-9.0,8.5,1.0)，换算到世界系中心在 (-43.4,-42.5,1.3)，离
        # actor 自己的位置十几米远），这是 CARLA 给红绿灯用的控制/触发体积，不是
        # 贴合灯杆网格的可见几何——光栅化出来就是一个凭空出现、和任何可见物体都对不上
        # 的方块。而且 traffic.* 的 type_id 在 actor_occupancy_mapping.py 里没有任何
        # 精确匹配 (VEHICLE_MAPPING/PROP_MAPPING 都没有 'traffic.' 前缀的兜底规则)，
        # 会一路掉到 semantic_tag 兜底，用 CITY_OBJECT_MAPPING[Pole]=15(manmade)，
        # 跟 _fill_static_environment() 那边 TrafficLight/TrafficSigns 环境物体统一
        # 映射到 8(traffic_cone) 还对不上，是两条并行、互相矛盾的分类路径。
        # 真实可见的红绿灯/路牌几何，本来就已经由 world.get_environment_objects() 的
        # TrafficLight/TrafficSigns/Poles 类别在 _fill_static_environment() 里用贴合
        # mesh 的世界系 AABB 正确光栅化了 (映射到 8/manmade 视具体类型)，traffic.*
        # 动态 Actor 这条路径纯属重复且不可靠，直接不参与光栅化，不产生覆盖缺口。
        # (traffic.unknown 之前单独排除过的写法已经不需要了，因为整个 traffic.* 都不
        # 再进入这条光栅化路径。)

        all_actors = list(vehicles) + list(walkers) + list(props) + list(traffic)

        self._dbg(f"\n[体素生成] 场景中总Actor数: 车辆={len(vehicles)}, 行人={len(walkers)}, Props={len(props)}, Traffic={len(traffic)}")

        # 3. 遍历 Actor，光栅化 Bounding Box
        filled_actor_ids = []
        filled_by_type = {'vehicles': [], 'walkers': [], 'props': [], 'traffic': []}
        filtered_by_distance = {'vehicles': 0, 'walkers': 0, 'props': 0, 'traffic': 0}

        self._dbg(f"\n[调试] ========== 开始填充Actor到体素 ==========")

        for actor in all_actors:
            # 距离粗筛
            dist = actor.get_location().distance(ego_vehicle.get_location())

            # ⭐ 调试：记录actor类型
            is_walker = 'walker.pedestrian' in actor.type_id.lower()
            is_prop = 'static.prop' in actor.type_id.lower()

            if dist > 60.0: # 略大于 grid 半径
                if is_walker:
                    filtered_by_distance['walkers'] += 1
                elif is_prop:
                    filtered_by_distance['props'] += 1
                else:
                    filtered_by_distance['vehicles'] += 1
                continue

            self._fill_actor_bb(occupancy, actor_ids, actor, grid_to_world_matrix, is_ego=(actor.id == ego_vehicle.id))
            filled_actor_ids.append(actor.id)

            # ⭐ 按类型统计
            if is_walker:
                filled_by_type['walkers'].append(actor.id)
            elif is_prop:
                filled_by_type['props'].append(actor.id)
            else:
                filled_by_type['vehicles'].append(actor.id)

        self._dbg(f"[调试] ========== 填充完成 ==========")
        self._dbg(f"[调试] 距离过滤: 车辆 {filtered_by_distance['vehicles']}, 行人 {filtered_by_distance['walkers']}, Props {filtered_by_distance['props']}")

        # 4. 填充自车
        self._fill_actor_bb(occupancy, actor_ids, ego_vehicle, grid_to_world_matrix, is_ego=True)
        filled_actor_ids.append(ego_vehicle.id)
        
        # [调试] 验证自车中心位置 (在填充后)
        # 查找 Label 4 (Car) 且 ID 为 Ego ID 的体素
        ego_mask = (occupancy == 4) & (actor_ids == ego_vehicle.id)
        if np.any(ego_mask):
            ego_indices = np.argwhere(ego_mask)
            ego_center = np.mean(ego_indices, axis=0)
            self._dbg(f"[体素生成] ✅ Ego Vehicle Voxel Center: {ego_center} (Expected ~[256, 256, 20])")
            
            # 检查偏移
            offset = ego_center - np.array([self.grid_size[0]/2, self.grid_size[1]/2, (0 - self.z_range[0])/self.resolution])
            if np.linalg.norm(offset[:2]) > 5.0: # >1m offset
                print(f"[体素生成] ⚠️ 警告: Ego 偏离中心 {offset} (Grid Units)")
        else:
            print(f"[体素生成] ❌ 错误: 未找到 Ego Vehicle 体素!")

        self._dbg(f"[体素生成] 填充到体素的Actor IDs ({len(filled_actor_ids)}个): {sorted(filled_actor_ids)}")
        self._dbg(f"[体素生成]   车辆: {len(filled_by_type['vehicles'])}个, 行人: {len(filled_by_type['walkers'])}个")

        # 可见性处理在外部：LiDAR 实例过滤 (visibility_filter_simple.py)、相机自由空间雕刻与
        # 观测掩码 (camera_visibility.py)，见 main_collection.py
        return occupancy, actor_ids

    def _fill_static_environment(self, occupancy, actor_ids, world, ego_transform, grid_to_world_matrix):
        """
        填充静态环境 (Road, Ground, Sidewalk)
        由于全图 RayCast 太慢，这里采用基于 Map 的启发式方法：
        1. 假设 Z < 0.2 (相对于车轮接地处) 为地面层
        2. 查询 Map 区分 Road 和 Ground (Sidewalk/Terrain)
        3. 获取静态物体 (Buildings, Street Lights, etc.) 的 Bounding Box 并填充
        """
        map_instance = world.get_map()
        ego_location = ego_transform.location

        # ⭐⭐⭐ CRITICAL FIX: 优化 Cache 策略 ⭐⭐⭐
        # 原策略: 每帧清空 Cache (Line 175) -> 每帧查询 26万次 Map API -> 极慢
        # 新策略: 使用世界坐标 Grid Index 作为 Cache Key
        #         Key = (int(round(world_x/res)), int(round(world_y/res)))
        #         仅查询 Cache 中缺失的点 (车辆移动产生的新边缘)
        #         Cache 持续保留，支持任意旋转和移动，无需清空
        
        # Cache 使用世界坐标网格索引作为 Key，无需每帧清空
        # 车辆移动/旋转后只有新进入范围的边缘点才会触发 Map API 查询

        # --- A. 地面与道路 (Inverse Mapping Grid -> World) ---
        # 修复: 使用世界网格索引作为 Cache Key
        
        # 计算 Ego 在世界坐标系中的 Z (作为默认地面高度)
        start_waypoint = map_instance.get_waypoint(ego_location, project_to_road=True, lane_type=carla.LaneType.Any)
        ground_z_world = start_waypoint.transform.location.z if start_waypoint else 0.0

        # 1. 生成网格索引和对应的世界坐标
        # 生成所有网格索引
        keys_x = np.arange(self.grid_size[0])  # [0, 1, 2, ..., 511]
        keys_y = np.arange(self.grid_size[1])  # [0, 1, 2, ..., 511]

        # 生成网格索引对 (ix, iy)
        keys_x_grid, keys_y_grid = np.meshgrid(keys_x, keys_y, indexing='ij')
        keys_x_flat = keys_x_grid.ravel()  # 展平后的 X 索引
        keys_y_flat = keys_y_grid.ravel()  # 展平后的 Y 索引

        # 计算每个网格索引对应的 Ego Frame 坐标
        gx_flat = keys_x_flat * self.resolution + self.x_range[0] + self.resolution/2.0
        gy_flat = keys_y_flat * self.resolution + self.y_range[0] + self.resolution/2.0

        # 转换到世界坐标（用于查询 Map API）
        points_grid_h = np.stack([gx_flat, gy_flat, np.zeros(len(gx_flat)), np.ones(len(gx_flat))], axis=1)
        points_world_h = points_grid_h @ grid_to_world_matrix.T

        flat_wx = points_world_h[:, 0]
        flat_wy = points_world_h[:, 1]

        num_points = len(keys_x_flat)

        # 3. Cache Optimization (使用世界网格索引作为 Key)
        # 将浮点世界坐标量化为整数索引
        # 使用 round() 避免 float 精度误差导致的抖动
        world_ix = np.round(flat_wx / self.resolution).astype(np.int64)
        world_iy = np.round(flat_wy / self.resolution).astype(np.int64)
        
        # 构造查询 Keys (list of tuples)
        # 注意: zip 在 Python 3 是 lazy 的，转换成 list 稍有开销但为了逻辑清晰
        query_keys = list(zip(world_ix, world_iy))
        
        # 4. 找出缺失的 Keys 并批量查询
        import time
        t_query_start = time.time()
        
        # 统计缺失
        missing_indices = []
        for i, key in enumerate(query_keys):
            if key not in self.ground_cache:
                missing_indices.append(i)
                
        num_missing = len(missing_indices)
        
        if num_missing > 0:
            self._dbg(f"[地面填充] 发现 {num_missing} 个新网格点 (总 {num_points}), 开始查询 Map API...")
            
            # 仅对缺失点查询 Map
            for idx in missing_indices:
                key = query_keys[idx]
                wx = flat_wx[idx]
                wy = flat_wy[idx]

                loc = carla.Location(x=wx, y=wy, z=0.0) # Z不重要，project_to_road=True
                l = 14  # Default Terrain
                z = ground_z_world # Default Z
                
                # Check Road
                wp = map_instance.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp:
                    wp_loc = wp.transform.location
                    dist = math.sqrt((wx - wp_loc.x)**2 + (wy - wp_loc.y)**2)
                    half_width = wp.lane_width / 2.0

                    # ⭐ 修正逻辑: 车道内 vs 车道边缘外侧
                    if dist < half_width:
                        # 在车道内: 默认为可行驶路面
                        l = 11  # driveable_surface
                        z = wp_loc.z

                        # 车道线仍是路面 (nuScenes/Occ3D 里 lane marking 属于 driveable_surface)。
                        # label_version 1 曾把车道边缘 0.2m 标成 traffic_cone(8)，让网络在
                        # 照片里只看到白线的位置去学"锥桶"，已去掉。

                    elif dist < half_width + 0.5:
                        # 车道外侧 0.5m: 路缘石 (马路牙子)，与人行道同属一类。
                        # label_version 1 标成 barrier(1)，但照片里这是路面高度的平面，不是护栏。
                        l = 13  # sidewalk
                        z = wp_loc.z
                
                # Check Sidewalk
                if l == 14:
                    wp_sw = map_instance.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Sidewalk)
                    if wp_sw:
                        dist = math.sqrt((wx - wp_sw.transform.location.x)**2 + (wy - wp_sw.transform.location.y)**2)
                        if dist < (wp_sw.lane_width / 2.0 + 0.5):
                            l = 13
                            z = wp_sw.transform.location.z
                            
                self.ground_cache[key] = {'label': l, 'z': z}

        self._dbg(f"[地面填充] Cache 查询完成 (新增 {num_missing}), 耗时: {time.time() - t_query_start:.3f}s")

        # 5. 从 Cache 直接填充到网格
        final_ix = keys_x_flat
        final_iy = keys_y_flat
        final_l = np.zeros(len(final_ix), dtype=np.uint8)
        final_z_world = np.zeros(len(final_ix), dtype=np.float32)

        # ⭐⭐⭐ 性能优化: 向量化赋值，消除 Python 循环 ⭐⭐⭐
        # 根因: Python for 循环在处理 26万次赋值时极慢 (10-180秒)
        # 解决: 使用 NumPy 向量化操作 (0.01-0.05秒)
        # 预计提速: 100-1000 倍

        # Step 1: 批量查询 Cache
        cache_hits = np.array([self.ground_cache.get(k, None) for k in query_keys], dtype=object)

        # Step 2: 创建命中 mask
        hit_mask = np.array([x is not None for x in cache_hits], dtype=bool)

        # Step 3: 向量化赋值 - Cache Hit
        if hit_mask.any():
            hit_indices = np.where(hit_mask)[0]
            hit_labels = np.array([cache_hits[i]['label'] for i in hit_indices], dtype=np.uint8)
            hit_z = np.array([cache_hits[i]['z'] for i in hit_indices], dtype=np.float32)
            final_l[hit_mask] = hit_labels
            final_z_world[hit_mask] = hit_z

        # Step 4: 向量化赋值 - Cache Miss (一次性填充所有 miss)
        miss_mask = ~hit_mask
        if miss_mask.any():
            final_l[miss_mask] = 14  # terrain
            final_z_world[miss_mask] = ground_z_world

        # 调试信息: 显示 Cache 命中率
        if len(query_keys) > 0:
            hit_rate = hit_mask.sum() / len(hit_mask) * 100
            self._dbg(f"  [向量化填充] Cache命中率: {hit_mask.sum()}/{len(hit_mask)} ({hit_rate:.1f}%)")

        # 计算 Z 索引
        # ⭐⭐⭐ FIX: 使用 round() 替代 floor (int cast) ⭐⭐⭐
        # 解决 Z 值在体素边界附近时产生的 1-voxel 抖动 (Random Gray Layer)
        flat_gz = final_z_world - ego_location.z
        flat_iz = np.round((flat_gz - self.z_range[0]) / self.resolution).astype(int)

        # 过滤有效 Z
        valid_z_mask = (flat_iz >= 0) & (flat_iz < self.grid_size[2])

        final_ix = final_ix[valid_z_mask]
        final_iy = final_iy[valid_z_mask]
        final_iz = flat_iz[valid_z_mask]
        final_l = final_l[valid_z_mask]

        # ========================================================================
        # 地面和地下层的逐层向下填充
        # ========================================================================
        # 逻辑:
        # 1. 第一遍: 填充地表层 (Map API 查询到的所有地面体素)
        # 2. 第二遍: 对每个 (x,y) 列,从顶部向下搜索,找到地表,向下填充

        # 计算地面层索引 (相对于 Ego Z, 仅用于日志显示)
        gz_ground = ground_z_world - ego_location.z
        iz_ground = int((gz_ground - self.z_range[0]) / self.resolution)
        iz_ground = max(0, min(iz_ground, self.grid_size[2] - 1))

        self._dbg(f"\n[地面填充] 地面层索引 iz_ground={iz_ground} (ground_z={ground_z_world:.2f}, ego_z={ego_location.z:.2f})")

        # 第一遍: 填充地表层 (Map API 查询到的所有地面体素)
        # ⚠️ 修复: 不限制 Z 层,填充所有 Map API 查询到的地面
        # ⭐⭐⭐ CRITICAL FIX: 移除高度限制,信任 Map API 查询的 Z 值 ⭐⭐⭐
        # 根因: 坡道、桥梁、起伏路面的 Z 高度可能远超 ego 当前位置 ±0.4m
        #       强制限制会导致这些地面被丢弃,出现空洞

        # ⭐⭐⭐ 性能优化: 向量化赋值,消除 Python 循环 ⭐⭐⭐
        # 根因: Python for 循环在处理 26万次赋值时极慢 (10-15秒)
        # 解决: 使用 NumPy 高级索引批量赋值 (0.1-0.5秒)
        # 预计提速: 20-150 倍

        import time
        t_start = time.time()

        # 1. 批量读取当前值 (利用 NumPy 的向量化操作)
        current_vals = occupancy[final_ix, final_iy, final_iz]

        # 2. 生成掩码: 只填充空位 (0) 或原有地面 (12)
        fill_mask = (current_vals == 0) | (current_vals == 12)

        # 3. 筛选需要填充的索引
        valid_ix = final_ix[fill_mask]
        valid_iy = final_iy[fill_mask]
        valid_iz = final_iz[fill_mask]
        valid_l = final_l[fill_mask]

        # 4. 批量赋值 (瞬间完成,利用 C 语言底层优化)
        occupancy[valid_ix, valid_iy, valid_iz] = valid_l
        actor_ids[valid_ix, valid_iy, valid_iz] = -(1000 + valid_l.astype(np.int32))

        surface_fill_count = len(valid_ix)
        t_elapsed = time.time() - t_start

        self._dbg(f"[地面填充] 地表层填充: {surface_fill_count} 个体素 (耗时: {t_elapsed:.3f}s)")

        # 第二遍: 对每个 (x,y) 列,从地面向下填充
        # ⭐⭐⭐ 性能优化: 向量化向下填充,消除双重循环 ⭐⭐⭐
        # 根因: 512×512 的双重 Python 循环极慢 (8-10秒)
        # 解决: 使用 NumPy 3D 广播一次性填充所有列 (预计 0.1-0.5秒)

        t_start_downfill = time.time()

        # 策略: 利用已经填充好的地表层 (第一遍的结果)
        # 对每列,找到最高的非零体素,然后向下填充

        # 1. 找到每列的地表高度索引 (最高的非零体素)
        # 创建掩码: 哪些体素非零
        non_zero_mask = (occupancy != 0)

        # 对每列 (x, y),从上到下找第一个非零值的 Z 索引
        # argmax 会返回第一个 True 的索引,但我们需要从上到下搜索
        # 技巧: 反转 Z 轴,然后用 argmax
        non_zero_mask_reversed = non_zero_mask[:, :, ::-1]  # 反转 Z 轴
        has_surface = np.any(non_zero_mask_reversed, axis=2)  # 每列是否有地表
        surface_z_reversed = np.argmax(non_zero_mask_reversed, axis=2)  # 地表在反转后的索引

        # 转换回原始 Z 索引
        surface_z = self.grid_size[2] - 1 - surface_z_reversed

        # 2. 提取每列的地表标签和 Actor ID
        # 使用高级索引获取地表层的值
        ix_grid, iy_grid = np.meshgrid(np.arange(self.grid_size[0]),
                                       np.arange(self.grid_size[1]),
                                       indexing='ij')

        surface_labels = np.where(has_surface,
                                  occupancy[ix_grid, iy_grid, surface_z],
                                  0)  # 没有地表的列设为 0
        surface_aids = np.where(has_surface,
                               actor_ids[ix_grid, iy_grid, surface_z],
                               0)

        # 3. 创建 3D Z 坐标网格
        Z_grid = np.arange(self.grid_size[2])[None, None, :]  # shape: (1, 1, 32)
        surface_z_3d = surface_z[:, :, None]  # shape: (400, 400, 1)

        # 4. 生成填充掩码: Z < surface_z 且 occupancy == 0
        fill_mask = (Z_grid < surface_z_3d) & (occupancy == 0) & has_surface[:, :, None]

        # 5. 向量化填充 (广播操作): 向下复制本列的地表标签，而不是统一填成 Terrain。
        # 2026-08-27 改回"复制地表"这个原始方案 (按用户要求): 相机/LiDAR 只能看到
        # 地表这一层，看不到地下埋了什么，与其瞎猜/统一填一个和实际地表毫无关系的
        # Terrain，不如假设"往下也是同一种东西"——这样同一位置在不同帧里地下内容
        # 稳定不变，不会给训练引入和真实场景无关的噪声。
        # 这条只在这个函数里、这个时间点安全: 此时 occupancy 里只有地面层的结果
        # (driveable_surface/sidewalk/terrain, 见上面
        # ground_cache 的取值范围)，_fill_static_environment 的建筑/杆/Actor光栅化
        # 都还没跑，所以不会把建筑/车辆的标签误往下拖成一根柱子。
        # 之前(2026-08-27之前)改成统一 Terrain 是因为"如果地表是 Barrier，地下会
        # 形成一堵灰色的墙，视觉效果突兀"——这是审美顾虑，不是正确性问题，柱子状的
        # 地下内容对训练没有坏处，反而比"和地表毫无关系的固定材质"更符合"我们看不到
        # 地下、只能假设它和地表一致"这个原则。
        surface_labels_3d = surface_labels[:, :, None]  # (400, 400, 1)
        surface_aids_3d = surface_aids[:, :, None]

        occupancy[fill_mask] = np.broadcast_to(surface_labels_3d, occupancy.shape)[fill_mask]
        actor_ids[fill_mask] = np.broadcast_to(surface_aids_3d, actor_ids.shape)[fill_mask]

        total_filled = np.count_nonzero(fill_mask)
        filled_columns = np.count_nonzero(has_surface)

        t_downfill_elapsed = time.time() - t_start_downfill
        self._dbg(f"[地面填充] 完成 {filled_columns} 列填充, 总填充 {total_filled} 个体素 (耗时: {t_downfill_elapsed:.3f}s)")


        # --- B. 静态物体 (建筑物, 交通标志, 杆等) ---
        # 使用 world.get_environment_objects() 获取更详细的静态物体信息 (ID, Transform, BBox)
        # 替代旧的 get_level_bbs，以支持实例级可见性过滤
        
        # 直接使用 occupancy_config.py 中的统一映射配置
        # 确保与 actor_occupancy_mapping.py 和全局配置保持一致
        static_type_mapping = CARLA_TO_OCCUPANCY_MAPPING
        
        # ⭐⭐⭐ CRITICAL FIX: 排除已由 Map API 处理的地面类型 ⭐⭐⭐
        # 根因: Map API 已经生成了平滑、准确的地面 (Road, Sidewalk, Terrain)
        #       如果再光栅化这些类型的 Mesh (EnvironmentObjects), 会因为 Mesh 与 Waypoint 高度的微小差异
        #       导致在 Map 地面之上叠加一层"浮空"的 Mesh 地面 (Double Layer / Gray Layer on top)
        #       用户反馈的"地面上随机覆盖一层灰色"即由此导致。
        SKIP_STATIC_TYPES = {
            carla.CityObjectLabel.Roads,
            carla.CityObjectLabel.Sidewalks,
            carla.CityObjectLabel.Terrain,
            carla.CityObjectLabel.Ground,
            carla.CityObjectLabel.RoadLines, 
            # carla.CityObjectLabel.Markings, # Markings usually are on road
        }

        # 获取所有环境物体
        env_objs = world.get_environment_objects(carla.CityObjectLabel.Any)
        
        # 调试：统计环境物体类型
        type_counts = {}
        skipped_types = set()
        
        for obj in env_objs:
            if obj.type not in type_counts:
                type_counts[obj.type] = 0
            type_counts[obj.type] += 1
            
            if obj.type not in static_type_mapping:
                skipped_types.add(obj.type)

        self._dbg(f"\n[调试] 环境物体统计 (总数: {len(env_objs)}):")
        for t, count in type_counts.items():
            mapped = "✓" if t in static_type_mapping else "✗ (未映射)"
            skipped = "⛔ (已排除)" if t in SKIP_STATIC_TYPES else ""
            self._dbg(f"  - {t}: {count} 个 {mapped} {skipped}")
            
        if skipped_types:
            print(f"[警告] 发现未映射的物体类型: {skipped_types}")
        
        # 预计算 Grid->World 的逆矩阵 (World->Grid)
        # grid_to_world_matrix 是纯平移矩阵，其逆矩阵也是纯平移 (T^-1 = -T)
        try:
            world_to_grid_matrix = np.linalg.inv(grid_to_world_matrix)
        except np.linalg.LinAlgError:
            return

        count_filled = 0

        # 填充顺序：包围盒体积从大到小，小物体后填、覆盖大物体。AABB 重叠时更小的盒子更贴合真实几何——
        # 树冠的大盒子会罩住树下停的车、长椅、花坛，先填树再填它们，它们才不会被标成植被
        # (2026-10-14 逐像素核对：UE 渲染为 Car/Static 的表面点分别有 11%、37~47% 落在 vegetation 体素里)。
        # 虚拟 ID 仍用 env_objs 里的原始下标 i (-(i+10000))，debug_actor_ids 反查不受影响。
        def _bb_volume(o):
            e = o.bounding_box.extent
            return e.x * e.y * e.z
        fill_order = sorted(range(len(env_objs)), key=lambda k: -_bb_volume(env_objs[k]))

        for i in fill_order:
            obj = env_objs[i]
            # 1. 类型过滤
            if obj.type not in static_type_mapping:
                continue
                
            # ⭐ 排除地面类型
            if obj.type in SKIP_STATIC_TYPES:
                continue

            occ_label = static_type_mapping[obj.type]
            
            # 2. 距离过滤
            # obj.transform.location 是世界坐标
            dist = obj.transform.location.distance(ego_location)
            
            # ⭐ 关键修改：增加到 100m 以覆盖整个体素网格范围 (此前为 60m)
            # 网格范围通常是 [-50, 50] 或类似，100m 足够覆盖对角线
            if dist > 100.0:
                continue
                
            # 3. 获取 Bounding Box 顶点 (World Frame)
            # 经过调试发现：UE5 CARLA 0.10.0 中，EnvironmentObject.bounding_box 已经是世界坐标 (World Space AABB)
            # 与 get_level_bbs 返回的一致。
            # 因此，不能再应用 obj.transform，否则会导致双重变换（物体飞到天上或消失）
            bb = obj.bounding_box
            
            # 使用 Identity Transform 获取世界顶点 (因为 BB 已经是 World AABB)
            verts_world = bbox_world_vertices(bb, None)  # [8, 3]
            
            # 4. 转换到 Grid Frame (对齐世界坐标轴)
            verts_world_np = np.concatenate([verts_world, np.ones((8, 1))], axis=1).T
            verts_grid_np = world_to_grid_matrix @ verts_world_np
            
            xs_grid = verts_grid_np[0, :]
            ys_grid = verts_grid_np[1, :]
            zs_grid = verts_grid_np[2, :]
            
            # 5. 计算 Grid 范围
            min_ix = int(np.floor((np.min(xs_grid) - self.x_range[0]) / self.resolution))
            max_ix = int(np.ceil((np.max(xs_grid) - self.x_range[0]) / self.resolution))
            min_iy = int(np.floor((np.min(ys_grid) - self.y_range[0]) / self.resolution))
            max_iy = int(np.ceil((np.max(ys_grid) - self.y_range[0]) / self.resolution))
            min_iz = int(np.floor((np.min(zs_grid) - self.z_range[0]) / self.resolution))
            max_iz = int(np.ceil((np.max(zs_grid) - self.z_range[0]) / self.resolution))
            
            # Clip
            min_ix = max(0, min_ix)
            max_ix = min(self.grid_size[0], max_ix)
            min_iy = max(0, min_iy)
            max_iy = min(self.grid_size[1], max_iy)
            min_iz = max(0, min_iz)
            max_iz = min(self.grid_size[2], max_iz)
            
            if min_ix >= max_ix or min_iy >= max_iy or min_iz >= max_iz:
                continue

            # ⭐ 调试: 验证建筑坐标 (针对静态道具)
            # obj.type 实际上是 CityObjectLabel 枚举，不是字符串
            # 我们直接跳过这个打印，或者将其转换为字符串
            # if str(obj.type).startswith('static.prop'):
            #     bb_center_world = (bb.location.x, bb.location.y)
            #     bb_center_grid = (min_ix + (max_ix-min_ix)/2, min_iy + (max_iy-min_iy)/2)
            #     # print(f"[建筑填充] World=({bb_center_world[0]:.1f}, {bb_center_world[1]:.1f}) -> Grid Index Center=({bb_center_grid[0]:.0f}, {bb_center_grid[1]:.0f})")

            # ⭐ 安全检查：防止异常巨大的物体导致 meshgrid 爆内存/卡死
            # 100x100x100 = 1,000,000 个体素点
            # 如果某个物体过大（如天空盒或错误BBox），这里会卡住
            total_voxels = (max_ix - min_ix) * (max_iy - min_iy) * (max_iz - min_iz)
            if total_voxels > 2000000: # 限制最大处理 200万体素/物体
                # print(f"[警告] 跳过过大物体 ID={obj.id} Name={obj.name} Voxels={total_voxels}")
                continue
                
            # 6. OBB 光栅化 (改进版：保守光栅化 Conservative Rasterization)
            # 生成 Grid Points
            # ⭐ 优化：使用 arange + 索引计算 (精度最高)
            lx = self.x_range[0] + (np.arange(min_ix, max_ix) + 0.5) * self.resolution
            ly = self.y_range[0] + (np.arange(min_iy, max_iy) + 0.5) * self.resolution
            lz = self.z_range[0] + (np.arange(min_iz, max_iz) + 0.5) * self.resolution
            
            sub_xv, sub_yv, sub_zv = np.meshgrid(lx, ly, lz, indexing='ij')
            sub_points_grid = np.stack([sub_xv, sub_yv, sub_zv, np.ones_like(sub_xv)], axis=-1).reshape(-1, 4)
            
            # Grid -> World -> Local
            # T_obj_inv * T_grid * P_grid
            # 注意：由于 bb 已经是 World AABB，我们不需要转换到 Local Object Space
            # 我们只需要检查 World Points 是否在 World AABB 内
            # 也就是检查 abs(P_world - BB_Center) <= BB_Extent
            
            # Grid -> World
            sub_points_world_h = sub_points_grid @ grid_to_world_matrix.T
            sub_points_world = sub_points_world_h[:, :3]
            
            # World AABB Check
            # bb.location is World Center
            diff = np.abs(sub_points_world - np.array([bb.location.x, bb.location.y, bb.location.z]))
            
            # ⭐ 关键改进：保守光栅化 (Conservative Rasterization)
            # 只要体素与物体有任何重叠，就应该被选中。
            # 简单的"中心点检测"对于细小物体（如杆子）会失败，因为体素中心可能恰好在杆子外面。
            # 解决方案：将检测盒 (Bounding Box) 扩大半个分辨率。
            # 这样，只要体素中心距离物体表面在 resolution/2 以内（即体素体积与物体相交），就会被选中。
            
            # 默认 padding
            # ⭐ 增强 Padding: 0.9 * resolution (0.18m)
            # 这足以覆盖旋转体素的对角线距离 (sqrt(3)/2 * res ≈ 0.866 * res)
            # 解决静态物体闪烁问题
            padding = self.resolution * 0.9 
            
            # 优化：对于非常大的物体（如道路、地形），不需要这么激进的 padding，以节省性能并防止过度膨胀
            # 如果 extent 很大，减少 padding
            if max(bb.extent.x, bb.extent.y) > 2.0:
                padding = 0.0 # 大物体不需要额外 padding，中心点足够
            
            # 对于细小物体（Pole/Sign），我们希望它是"实心"的
            in_x = diff[:, 0] <= (bb.extent.x + padding)
            in_y = diff[:, 1] <= (bb.extent.y + padding)
            in_z = diff[:, 2] <= (bb.extent.z + padding)
            
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
            
        self._dbg(f"[体素生成] 已填充静态环境物体: {count_filled} 个")
                
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
            # ⭐ 关键修复：不要收缩细小物体，反而要确保最小尺寸
            if original_z > 2.0 and max(original_x, original_y) < 0.5:
                # 杆状物体：确保最小半径 >= resolution/2
                min_radius = self.resolution * 0.6  # 稍微大一点点 (e.g. 0.12m)
                radius = max(max(original_x, original_y), min_radius)
                return radius, radius, original_z

            # 交通标志：通常是薄板 + 杆
            # ⭐ 关键修复：确保薄维度不小于分辨率的一半
            min_dim_val = self.resolution * 0.6
            return (
                max(original_x, min_dim_val),
                max(original_y, min_dim_val),
                max(original_z, min_dim_val)
            )

        # 8: traffic_cone (交通锥)
        elif occ_label == 8:
            # 锥形物体：底部圆形
            # ⭐ 关键修复：确保最小尺寸
            min_radius = self.resolution * 0.6
            radius = max(max(original_x, original_y), min_radius)
            return radius, radius, original_z

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
            # 细长物体，为了通过深度可见性检查，绝不能收缩到物体内部！
            # 必须保持原始大小或轻微膨胀，确保体素中心位于深度表面之前
            return original_x * 1.05, original_y * 1.05, original_z * 1.0

        # 4: car, 3: bus, 10: truck, 5: construction_vehicle (车辆)
        elif occ_label in [4, 3, 10, 5]:
            # 车辆通常是空心的，为了可见性，也不建议过度收缩
            # 改为 0.95 或 1.0
            return original_x * 0.95, original_y * 0.95, original_z * 0.95

        # 16: vegetation (植被)
        elif occ_label == 16:
            # 树木、灌木：不规则形状，保守填充中心区域
            return original_x * 0.6, original_y * 0.6, original_z * 0.7

        # 其他类型：默认轻微收缩
        else:
            return original_x * 0.85, original_y * 0.85, original_z * 0.85

    def _fill_actor_bb(self, occupancy, actor_ids_grid, actor, grid_to_world_matrix, is_ego=False):
        """
        Helper to rasterize an actor's bounding box into the occupancy grid

        Args:
            occupancy: 体素类别数组
            actor_ids_grid: 体素Actor ID数组
            actor: CARLA Actor对象
            grid_to_world_matrix: Grid到World的变换矩阵 (纯平移)
            is_ego: 是否是自车
        """
        try:
            bb = actor.bounding_box
            actor_transform = actor.get_transform()
        except:
            return # Actor might be dead

        # Logging for debug
        # print(f"DEBUG: Actor {actor.type_id} -> Semantic Tag {actor.semantic_tags}")

        # World -> Grid (Translation Only Inverse)
        try:
            world_to_grid_matrix = np.linalg.inv(grid_to_world_matrix)
        except np.linalg.LinAlgError:
            return

        # Get 8 corners in World
        verts_world = bbox_world_vertices(bb, actor_transform)  # [8, 3]

        verts_world_np = np.concatenate([verts_world, np.ones((8, 1))], axis=1).T

        # Transform to Grid Frame
        verts_grid_np = world_to_grid_matrix @ verts_world_np

        xs_grid = verts_grid_np[0, :]
        ys_grid = verts_grid_np[1, :]
        zs_grid = verts_grid_np[2, :]

        # Grid Indices Range
        min_ix = int(np.floor((np.min(xs_grid) - self.x_range[0]) / self.resolution))
        max_ix = int(np.ceil((np.max(xs_grid) - self.x_range[0]) / self.resolution))
        min_iy = int(np.floor((np.min(ys_grid) - self.y_range[0]) / self.resolution))
        max_iy = int(np.ceil((np.max(ys_grid) - self.y_range[0]) / self.resolution))
        min_iz = int(np.floor((np.min(zs_grid) - self.z_range[0]) / self.resolution))
        max_iz = int(np.ceil((np.max(zs_grid) - self.z_range[0]) / self.resolution))

        # Clip
        min_ix = max(0, min_ix)
        max_ix = min(self.grid_size[0], max_ix)
        min_iy = max(0, min_iy)
        max_iy = min(self.grid_size[1], max_iy)
        min_iz = max(0, min_iz)
        max_iz = min(self.grid_size[2], max_iz)

        # 行人调试：记录边界框裁剪前后
        is_walker = 'walker.pedestrian' in actor.type_id.lower()

        if min_ix >= max_ix or min_iy >= max_iy or min_iz >= max_iz:
            if is_walker:
                self._dbg(f"  [行人调试] ID={actor.id} 在网格范围外被跳过 - Grid范围: ix=[{min_ix},{max_ix}), iy=[{min_iy},{max_iy}), iz=[{min_iz},{max_iz})")
            return

        # Prepare Sub-grid for OBB check
        # ⭐ 优化：使用 arange + 索引计算 (精度最高)
        # 避免 linspace 的浮点累积误差
        lx = self.x_range[0] + (np.arange(min_ix, max_ix) + 0.5) * self.resolution
        ly = self.y_range[0] + (np.arange(min_iy, max_iy) + 0.5) * self.resolution
        lz = self.z_range[0] + (np.arange(min_iz, max_iz) + 0.5) * self.resolution
        
        sub_xv, sub_yv, sub_zv = np.meshgrid(lx, ly, lz, indexing='ij')
        
        # Flatten Voxel Centers
        # shape: (N, 3)
        # sub_points_grid: 体素中心在Grid坐标系中 (对齐世界坐标轴)
        sub_points_grid = np.stack([sub_xv, sub_yv, sub_zv, np.ones_like(sub_xv)], axis=-1).reshape(-1, 4)
        num_voxels = sub_points_grid.shape[0]

        # ==============================================================================
        # ⭐ 多点采样策略 (Multi-point Sampling) 
        # 解决细小物体在体素网格移动时的闪烁问题
        # ==============================================================================
        
        # 1. 判定是否启用多点采样
        # 细小物体 (extent < 1.0) 或 行人 (walker) 启用 27 点采样 (3x3x3)
        # 之前的 9 点 (3x3平面) 在Z轴方向仍有不足，导致上下跳动闪烁
        # 大物体 (车辆) 保持中心单点采样以节省性能
        max_dim = max(bb.extent.x, bb.extent.y, bb.extent.z)
        is_small_object = (max_dim < 1.0) or is_walker
        
        if is_small_object:
            # 27-point sampling (3x3x3 Grid)
            # 覆盖 Center, Corners, and Edge Centers
            half_res = self.resolution * 0.45 # 稍微收缩一点，避免跨越太多体素
            
            offsets = [-half_res, 0, half_res]
            sample_offsets = []
            for ox in offsets:
                for oy in offsets:
                    for oz in offsets:
                        sample_offsets.append([ox, oy, oz])
            sample_offsets = np.array(sample_offsets)
        else:
            # 1-point sampling (Center only)
            sample_offsets = np.array([[0, 0, 0]])
            
        # 2. 扩展采样点
        # (N, 1, 3) + (1, S, 3) -> (N, S, 3)
        centers_expanded = sub_points_grid[:, :3][:, np.newaxis, :] + sample_offsets[np.newaxis, :, :]
        
        # Reshape to (N*S, 3) for vectorized transformation
        all_samples = centers_expanded.reshape(-1, 3)
        
        # To Homogeneous: (N*S, 4)
        all_samples_h = np.concatenate([all_samples, np.ones((all_samples.shape[0], 1))], axis=1)
        
        # Grid -> World -> Actor Local
        # Step 1: Grid -> World (纯平移)
        sub_points_world = all_samples_h @ grid_to_world_matrix.T

        # Step 2: World -> Actor Local (用于BBox内点检测)
        box_matrix = transform_to_matrix(actor_transform)
        try:
            box_matrix_inv = np.linalg.inv(box_matrix)
        except np.linalg.LinAlgError:
            return
            
        points_in_actor = sub_points_world @ box_matrix_inv.T # (N*S, 4)
        
        # Subtract bb.location (which is local offset in actor frame)
        # Note: In actor local frame, bb center is usually at (0,0,0) + bb.location
        # But carla.BoundingBox.location is relative to actor origin.
        # So points_in_actor is relative to actor origin.
        # We need to shift by bb.location to center it around (0,0,0) for extent check.
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
            self._dbg(f"  [行人调试] ID={actor.id} BBox原始extent=({bb.extent.x:.2f}, {bb.extent.y:.2f}, {bb.extent.z:.2f}), "
                  f"自适应extent=({extent_x:.2f}, {extent_y:.2f}, {extent_z:.2f})")

        # Check Extents with adaptive shrinking
        # mask_in_samples: (N*S, )
        in_x = np.abs(rel_x) <= extent_x
        in_y = np.abs(rel_y) <= extent_y
        in_z = np.abs(rel_z) <= extent_z

        mask_in_samples = in_x & in_y & in_z
        
        # 3. 聚合采样结果
        # Reshape back to (N, S)
        mask_in_samples_reshaped = mask_in_samples.reshape(num_voxels, -1)
        
        # 只要任意一个采样点命中，该体素即为命中
        mask_in = np.any(mask_in_samples_reshaped, axis=1)

        voxel_count = np.sum(mask_in)

        if not np.any(mask_in):
            if is_walker:
                self._dbg(f"  [行人调试] ID={actor.id} 没有体素通过extent检查 - 潜在网格: {(max_ix - min_ix) * (max_iy - min_iy) * (max_iz - min_iz)} 个")
            return

        # 行人调试：记录填充的体素数
        if is_walker:
            self._dbg(f"  [行人调试] ID={actor.id} ✓ 成功填充 {voxel_count} 个体素 (occupancy_label={occ_label})")

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

    def _dbg(self, *args, **kw):
        """逐帧调试输出 (默认关闭，GroundTruthVoxelGenerator(verbose=True) 打开)。"""
        if self.verbose:
            print(*args, **kw)

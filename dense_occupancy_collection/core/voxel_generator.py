"""
体素生成器 (Voxel Generator)
负责将 CARLA 世界状态转换为 3D 语义体素网格
核心功能：
1. 静态环境光栅化 (Map Query + Environment Objects)
2. 动态物体光栅化 (Actors)
3. 保守光栅化算法 (Conservative Rasterization)
"""

import carla
import numpy as np
import math
from dense_occupancy_collection.config.actor_occupancy_mapping import get_occupancy_label_from_actor

class VoxelGenerator:
    def __init__(self, config):
        """
        Args:
            config: dict containing x_range, y_range, z_range, resolution, mapping
        """
        self.cfg = config
        self.x_range = config['x_range']
        self.y_range = config['y_range']
        self.z_range = config['z_range']
        self.res = config['resolution']
        
        self.grid_size = config['grid_size']
        self.mapping = config['mapping']
        
    def generate(self, world, ego_vehicle):
        """
        生成 Ground Truth 体素网格 (无过滤)
        
        Returns:
            occupancy: (X, Y, Z) uint8
            actor_ids: (X, Y, Z) int32
        """
        occupancy = np.zeros(self.grid_size, dtype=np.uint8)
        actor_ids = np.zeros(self.grid_size, dtype=np.int32)
        
        ego_trans = ego_vehicle.get_transform()
        ego_matrix = np.array(ego_trans.get_matrix())
        
        # 1. 静态环境 (地面, 建筑, 植被)
        self._fill_static_environment(occupancy, actor_ids, world, ego_trans, ego_matrix)
        
        # 2. 动态物体 (NPCs)
        self._fill_dynamic_actors(occupancy, actor_ids, world, ego_vehicle, ego_matrix)
        
        return occupancy, actor_ids

    def _fill_static_environment(self, occupancy, actor_ids, world, ego_trans, ego_matrix):
        """填充静态环境 (Ground, Buildings, etc.)"""
        map_inst = world.get_map()
        ego_loc = ego_trans.location
        
        # --- A. 地面 (Ground/Road/Sidewalk) ---
        # 简单高度启发式: Z < 0.2 (相对于路面) -> Ground
        # 更精细的逻辑: 采样查询 map.get_waypoint
        
        # 1. 基础填充 (Z < 0.2)
        z_threshold = 0.2
        gz_max = int((z_threshold - self.z_range[0]) / self.res)
        gz_max = max(0, min(self.grid_size[2], gz_max))
        
        if gz_max > 0:
            # 默认 Ground (12: other_flat)
            occupancy[:, :, :gz_max] = 12 
            actor_ids[:, :, :gz_max] = 1  # 临时用1代表地面，防止负数ID乱码
            
            # 2. 区分 Road (11) 和 Sidewalk (13)
            # 采样步长 2 (1m)
            step = 2
            x_indices = np.arange(0, self.grid_size[0], step)
            y_indices = np.arange(0, self.grid_size[1], step)
            
            xv, yv = np.meshgrid(
                self.x_range[0] + (x_indices + 0.5) * self.res,
                self.y_range[0] + (y_indices + 0.5) * self.res,
                indexing='ij'
            )
            
            # Ego -> World
            points_ego = np.stack([xv, yv, np.zeros_like(xv)], axis=-1).reshape(-1, 3)
            points_h = np.concatenate([points_ego, np.ones((len(points_ego), 1))], axis=1)
            points_world = (points_h @ ego_matrix.T)[:, :3]
            
            # 距离过滤 (只查 60m 内)
            dists = np.linalg.norm(points_world[:, :2] - np.array([ego_loc.x, ego_loc.y]), axis=1)
            valid_mask = dists < 60.0
            
            valid_indices = np.where(valid_mask)[0]
            
            for idx in valid_indices:
                pw = points_world[idx]
                loc = carla.Location(x=pw[0], y=pw[1], z=pw[2])
                
                # Check Road
                wp = map_inst.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp:
                    if loc.distance(wp.transform.location) < (wp.lane_width / 2.0 + 0.5):
                        # 是道路
                        self._fill_ground_patch(idx, x_indices, y_indices, step, occupancy, actor_ids, gz_max, 11, 1) # ID=1
                        continue
                        
                # Check Sidewalk
                wp_sw = map_inst.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Sidewalk)
                if wp_sw:
                    if loc.distance(wp_sw.transform.location) < (wp_sw.lane_width / 2.0 + 0.5):
                        # 是人行道
                        self._fill_ground_patch(idx, x_indices, y_indices, step, occupancy, actor_ids, gz_max, 13, 1) # ID=1

        # --- B. 静态物体 (Buildings, Poles, etc.) ---
        self._fill_env_objects(occupancy, actor_ids, world, ego_loc, ego_matrix)

    def _fill_ground_patch(self, idx, x_indices, y_indices, step, occupancy, actor_ids, z_max, label, aid):
        ny = len(y_indices)
        ix_sub = idx // ny
        iy_sub = idx % ny
        
        ix_base = x_indices[ix_sub]
        iy_base = y_indices[iy_sub]
        
        ix_end = min(self.grid_size[0], ix_base + step)
        iy_end = min(self.grid_size[1], iy_base + step)
        
        occupancy[ix_base:ix_end, iy_base:iy_end, :z_max] = label
        actor_ids[ix_base:ix_end, iy_base:iy_end, :z_max] = aid

    def _fill_env_objects(self, occupancy, actor_ids, world, ego_loc, ego_matrix):
        """填充环境物体 (Buildings, etc.)"""
        # Mapping CARLA CityObjectLabel -> Occupancy ID
        # Simplified mapping
        static_mapping = {
            carla.CityObjectLabel.Buildings: 15,
            carla.CityObjectLabel.Fences: 1,
            carla.CityObjectLabel.TrafficLight: 15,
            carla.CityObjectLabel.TrafficSigns: 15,
            carla.CityObjectLabel.Poles: 15,
            carla.CityObjectLabel.Vegetation: 16,
            carla.CityObjectLabel.Walls: 15,
            carla.CityObjectLabel.Bridge: 2,
            # Add missing ones
            carla.CityObjectLabel.RoadLines: 11,
            carla.CityObjectLabel.Roads: 11,
            carla.CityObjectLabel.Sidewalks: 13,
            carla.CityObjectLabel.Terrain: 14,
            carla.CityObjectLabel.Ground: 12,
            carla.CityObjectLabel.Static: 17,
            carla.CityObjectLabel.Dynamic: 17,
        }
        
        env_objs = world.get_environment_objects(carla.CityObjectLabel.Any)
        ego_matrix_inv = np.linalg.inv(ego_matrix)
        
        for i, obj in enumerate(env_objs):
            if obj.type not in static_mapping: continue
            
            # Distance check (100m)
            if obj.transform.location.distance(ego_loc) > 100.0: continue
            
            occ_label = static_mapping[obj.type]
            virtual_id = (i + 10000) # 保持正数ID
            
            self._rasterize_bbox(
                occupancy, actor_ids, 
                obj.bounding_box, carla.Transform(), # BB is World Space
                ego_matrix, ego_matrix_inv,
                occ_label, virtual_id,
                is_world_bbox=True
            )

    def _fill_dynamic_actors(self, occupancy, actor_ids, world, ego_vehicle, ego_matrix):
        """填充动态 Actor"""
        actors = world.get_actors()
        vehicles = actors.filter('vehicle.*')
        walkers = actors.filter('walker.pedestrian.*')
        props = actors.filter('static.prop.*')
        
        all_actors = list(vehicles) + list(walkers) + list(props)
        ego_matrix_inv = np.linalg.inv(ego_matrix)
        
        for actor in all_actors:
            dist = actor.get_location().distance(ego_vehicle.get_location())
            if dist > 60.0: continue
            
            is_ego = (actor.id == ego_vehicle.id)
            occ_label = 4 if is_ego else get_occupancy_label_from_actor(actor)
            
            self._rasterize_bbox(
                occupancy, actor_ids,
                actor.bounding_box, actor.get_transform(),
                ego_matrix, ego_matrix_inv,
                occ_label, actor.id,
                is_world_bbox=False,
                actor_type=actor.type_id
            )

    def _rasterize_bbox(self, occupancy, actor_ids, bb, transform, ego_mat, ego_mat_inv, label, aid, is_world_bbox=False, actor_type=""):
        """
        通用 BBox 光栅化函数 (Conservative)
        """
        # 1. Get Vertices in World
        if is_world_bbox:
            # BB is already World AABB (mostly), transform is Identity
            verts_world = bb.get_world_vertices(carla.Transform())
        else:
            verts_world = bb.get_world_vertices(transform)
            
        if not verts_world: return

        # 2. Transform to Ego
        verts_world_np = np.array([[v.x, v.y, v.z, 1.0] for v in verts_world]).T
        verts_ego = ego_mat_inv @ verts_world_np
        
        xs, ys, zs = verts_ego[0], verts_ego[1], verts_ego[2]
        
        # 3. Grid Bounds
        min_ix = int(np.floor((np.min(xs) - self.x_range[0]) / self.res))
        max_ix = int(np.ceil((np.max(xs) - self.x_range[0]) / self.res))
        min_iy = int(np.floor((np.min(ys) - self.y_range[0]) / self.res))
        max_iy = int(np.ceil((np.max(ys) - self.y_range[0]) / self.res))
        min_iz = int(np.floor((np.min(zs) - self.z_range[0]) / self.res))
        max_iz = int(np.ceil((np.max(zs) - self.z_range[0]) / self.res))
        
        # Clip
        min_ix = max(0, min_ix); max_ix = min(self.grid_size[0], max_ix)
        min_iy = max(0, min_iy); max_iy = min(self.grid_size[1], max_iy)
        min_iz = max(0, min_iz); max_iz = min(self.grid_size[2], max_iz)
        
        if min_ix >= max_ix or min_iy >= max_iy or min_iz >= max_iz: return
        
        # Safety check for huge objects
        if (max_ix-min_ix)*(max_iy-min_iy)*(max_iz-min_iz) > 2000000: return

        # 4. Rasterize (Conservative)
        # Create grid points
        lx = np.linspace(self.x_range[0] + (min_ix + 0.5)*self.res, 
                         self.x_range[0] + (max_ix - 0.5)*self.res, max_ix - min_ix)
        ly = np.linspace(self.y_range[0] + (min_iy + 0.5)*self.res, 
                         self.y_range[0] + (max_iy - 0.5)*self.res, max_iy - min_iy)
        lz = np.linspace(self.z_range[0] + (min_iz + 0.5)*self.res, 
                         self.z_range[0] + (max_iz - 0.5)*self.res, max_iz - min_iz)
        
        xv, yv, zv = np.meshgrid(lx, ly, lz, indexing='ij')
        pts_ego = np.stack([xv, yv, zv, np.ones_like(xv)], axis=-1).reshape(-1, 4)
        
        # Transform Ego -> World -> Local
        # If is_world_bbox, we just check World AABB
        
        pts_world_h = pts_ego @ ego_mat.T
        pts_world = pts_world_h[:, :3]
        
        if is_world_bbox:
            # Check against World AABB
            center = np.array([bb.location.x, bb.location.y, bb.location.z])
            diff = np.abs(pts_world - center)
            padding = 0.0 if max(bb.extent.x, bb.extent.y) > 2.0 else self.res * 0.6
            
            mask = (diff[:,0] <= bb.extent.x + padding) & \
                   (diff[:,1] <= bb.extent.y + padding) & \
                   (diff[:,2] <= bb.extent.z + padding)
        else:
            # Check against OBB (Actor Local)
            # P_local = T_actor_inv * P_world
            box_mat = np.array(transform.get_matrix())
            try:
                box_mat_inv = np.linalg.inv(box_mat)
            except:
                return
                
            pts_local = pts_world_h @ box_mat_inv.T
            # Subtract local offset (bb.location)
            pts_local[:, 0] -= bb.location.x
            pts_local[:, 1] -= bb.location.y
            pts_local[:, 2] -= bb.location.z
            
            diff = np.abs(pts_local[:, :3])
            
            # Adaptive Padding (Instance Completion helper)
            padding = self._get_padding(label, bb, actor_type)
            
            mask = (diff[:,0] <= bb.extent.x + padding[0]) & \
                   (diff[:,1] <= bb.extent.y + padding[1]) & \
                   (diff[:,2] <= bb.extent.z + padding[2])

        if not np.any(mask): return
        
        # 5. Fill
        mask_reshaped = mask.reshape(max_ix-min_ix, max_iy-min_iy, max_iz-min_iz)
        
        roi = occupancy[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz]
        roi_ids = actor_ids[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz]
        
        # Ground Protection
        GROUND_LABELS = [11, 12, 13, 14]
        is_ground = np.isin(roi, GROUND_LABELS)
        final_mask = mask_reshaped & (~is_ground)
        
        roi[final_mask] = label
        roi_ids[final_mask] = aid
        
        occupancy[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz] = roi
        actor_ids[min_ix:max_ix, min_iy:max_iy, min_iz:max_iz] = roi_ids

    def _get_padding(self, label, bb, actor_type):
        """自适应 Padding 策略"""
        # 默认
        px, py, pz = self.res * 0.6, self.res * 0.6, self.res * 0.6
        
        # 细长物体 (Bicycle/Moto) - 保持略大
        if label in [2, 6]: 
            return (self.res*0.6, self.res*0.6, self.res*0.6)
            
        # 车辆 (Car/Truck) - 不要过度收缩
        if label in [4, 3, 10]:
            # 原始代码有收缩逻辑，现在为了 Instance Completion，建议不要收缩太多
            # 或者收缩一点点 (negative padding) 来避免贴图边缘？
            # 保持 0 padding 或微小 padding
            return (0.05, 0.05, 0.05)
            
        return (px, py, pz)

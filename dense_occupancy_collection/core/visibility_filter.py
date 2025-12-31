"""
可见性过滤器 (Visibility Filter)
1. 基于深度图的全景遮挡剔除 (Numba加速)
2. 实例级可见性补全 (Instance Completion)
"""

import numpy as np
import numba
from numba import jit, prange
import math

# ==============================================================================
# Numba JIT Accelerated Core
# ==============================================================================
@jit(nopython=True, parallel=True)
def _check_visibility_numba(
    occupancy, 
    x_range_min, x_range_max, 
    y_range_min, y_range_max, 
    z_range_min, z_range_max, 
    resolution,
    depth_maps, 
    view_matrices,
    fov_rad,
    width,
    height
):
    """
    Numba 加速的可见性检查 (Parallelized)
    """
    nx, ny, nz = occupancy.shape
    visibility_mask = np.zeros((nx, ny, nz), dtype=numba.boolean)
    
    # Pre-calculate half sizes
    half_width = width / 2.0
    half_height = height / 2.0
    # Focal length for Vertical FOV
    focal_length = half_height / math.tan(fov_rad / 2.0)
    
    # Loop over all voxels (Parallelized)
    for ix in prange(nx):
        for iy in range(ny):
            for iz in range(nz):
                if occupancy[ix, iy, iz] == 0:
                    continue
                    
                # 1. Calculate Voxel Center in Ego Coordinates
                vx = x_range_min + (ix + 0.5) * resolution
                vy = y_range_min + (iy + 0.5) * resolution
                vz = z_range_min + (iz + 0.5) * resolution
                
                is_visible = False
                
                # Check against all 6 cameras
                num_cams = view_matrices.shape[0]
                for c in range(num_cams):
                    # Ego -> Camera
                    t_mat = view_matrices[c]
                    
                    # P_cam = Mat * P_ego
                    pc_x = t_mat[0, 0] * vx + t_mat[0, 1] * vy + t_mat[0, 2] * vz + t_mat[0, 3]
                    pc_y = t_mat[1, 0] * vx + t_mat[1, 1] * vy + t_mat[1, 2] * vz + t_mat[1, 3]
                    pc_z = t_mat[2, 0] * vx + t_mat[2, 1] * vy + t_mat[2, 2] * vz + t_mat[2, 3]
                    
                    # Check if point is in front of camera (X > 0.1 for UE4 Camera System)
                    if pc_x <= 0.1: 
                        continue
                        
                    # Project to Image Plane
                    # u = f * (y / x) + w/2  (UE4: Y is right)
                    # v = f * (-z / x) + h/2 (UE4: Z is up, Image V is down)
                    
                    u = focal_length * (pc_y / pc_x) + half_width
                    v = focal_length * (-pc_z / pc_x) + half_height
                    
                    # Check bounds
                    if u >= 0 and u < width and v >= 0 and v < height:
                        u_int = int(u)
                        v_int = int(v)
                        
                        # Get Depth from Map
                        depth_val = depth_maps[c, v_int, u_int]
                        
                        # Check Visibility
                        # Voxel distance is pc_x
                        # Tolerance: 0.5 meters (to account for discretization)
                        if pc_x < (depth_val + 0.5):
                            is_visible = True
                            break # Visible in at least one camera
                
                if is_visible:
                    visibility_mask[ix, iy, iz] = True
                    
    return visibility_mask

# ==============================================================================
# Visibility Filter Class
# ==============================================================================
class VisibilityFilter:
    def __init__(self, width=512, height=512, fov=60.0):
        self.width = width
        self.height = height
        self.fov = fov
        self.fov_rad = math.radians(fov)
        
    def run(self, occupancy, actor_ids, grid_config, depth_data, ego_matrix):
        """
        执行可见性过滤流程
        1. 计算深度可见性 (Depth Mask)
        2. 实例级补全 (Instance Completion)
        3. 强制保留规则 (Ground)
        """
        # Unpack Config
        x_range = grid_config['x_range']
        y_range = grid_config['y_range']
        z_range = grid_config['z_range']
        res = grid_config['resolution']
        
        depth_maps = depth_data['depth_maps']      # (6, H, W)
        cam_transforms_world = depth_data['cam_transforms'] # (6, 4, 4) Camera->World
        
        # 1. Prepare Matrices: Ego -> Camera
        # T_ego2cam = inv(T_cam2world) * T_ego2world
        view_matrices = np.zeros_like(cam_transforms_world)
        for i in range(6):
            t_world2cam = np.linalg.inv(cam_transforms_world[i])
            view_matrices[i] = t_world2cam @ ego_matrix
            
        # 2. Compute Raw Visibility (Numba)
        # Note: Passes view_matrices (Ego->Cam) instead of cam_transforms (Cam->World)
        # This matches the updated Numba signature
        raw_mask = _check_visibility_numba(
            occupancy,
            float(x_range[0]), float(x_range[1]),
            float(y_range[0]), float(y_range[1]),
            float(z_range[0]), float(z_range[1]),
            float(res),
            depth_maps.astype(np.float32),
            view_matrices.astype(np.float32),
            float(self.fov_rad),
            int(self.width),
            int(self.height)
        )
        
        # 3. Instance Completion (实例级补全)
        # 逻辑：只要 Actor 有任何一个体素可见，则该 Actor 的所有体素均可见
        visible_voxel_ids = actor_ids[raw_mask]
        unique_visible_ids = np.unique(visible_voxel_ids)
        unique_visible_ids = unique_visible_ids[unique_visible_ids != 0] # Exclude empty space/virtual IDs
        
        # Filter out negative IDs (static objects/ground) from instance completion logic if needed
        # Assuming static objects (ID < 0) or ID=1 (Ground) should rely on Ground Protection or raw mask
        # But typically we want instance completion for dynamic actors (ID > 100)
        
        final_mask = np.zeros_like(raw_mask)
        
        if len(unique_visible_ids) > 0:
            # 广播可见性
            final_mask = np.isin(actor_ids, unique_visible_ids)
        else:
            final_mask = raw_mask
            
        # 4. 强制保留规则 (地面)
        # 11=driveable, 12=other_flat, 13=sidewalk, 14=terrain
        GROUND_LABELS = [11, 12, 13, 14]
        ground_mask = np.isin(occupancy, GROUND_LABELS)
        final_mask[ground_mask] = True
        
        # Apply Mask
        filtered_occupancy = occupancy.copy()
        filtered_ids = actor_ids.copy()
        
        # 核心修改：不可见区域直接设为 0 (Free/Air)
        # 这保证了数据集中的 "不可见" == "Free"，避免模型混淆
        # 同时保留 final_mask 供后续可能的 mask loss 使用（如果需要区分未观测区域）
        remove_mask = (~final_mask)
        
        filtered_occupancy[remove_mask] = 0
        filtered_ids[remove_mask] = 0
        
        return filtered_occupancy, filtered_ids, final_mask

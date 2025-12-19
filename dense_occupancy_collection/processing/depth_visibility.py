import numpy as np
import numba
from numba import jit, prange
import math

@jit(nopython=True, parallel=True)
def _check_visibility_numba(
    occupancy, 
    x_range_min, x_range_max, 
    y_range_min, y_range_max, 
    z_range_min, z_range_max, 
    resolution,
    depth_maps, 
    cam_transforms,
    fov_rad,
    width,
    height
):
    """
    Numba 加速的可见性检查 (Parallelized)
    
    Args:
        occupancy: (X, Y, Z) uint8 array
        x_range, y_range, z_range: tuple (min, max)
        resolution: float
        depth_maps: (6, H, W) float32 array (Meters)
        cam_transforms: (6, 4, 4) float32 array (World -> Camera Matrices)
        fov_rad: float (Field of View in radians)
        width, height: int (Image dimensions)
        
    Returns:
        visibility_mask: (X, Y, Z) boolean array
    """
    nx, ny, nz = occupancy.shape
    visibility_mask = np.zeros((nx, ny, nz), dtype=numba.boolean)
    
    # Pre-calculate half sizes
    half_width = width / 2.0
    half_height = height / 2.0
    focal_length = half_width / math.tan(fov_rad / 2.0)
    
    # Loop over all voxels (Parallelized)
    # Flatten loop for better parallelism if needed, but 3D is fine with prange
    for ix in prange(nx):
        for iy in range(ny):
            for iz in range(nz):
                if occupancy[ix, iy, iz] == 0:
                    continue
                    
                # 1. Calculate Voxel Center in World Coordinates
                # (Assuming occupancy grid is axis-aligned and starts at range_min)
                # Note: This must match the generator's coordinate system
                
                # grid index -> world coord
                vx = x_range_min + (ix + 0.5) * resolution
                vy = y_range_min + (iy + 0.5) * resolution
                vz = z_range_min + (iz + 0.5) * resolution
                
                # Point in World (Homogeneous)
                # We do manual matrix multiplication for performance
                
                is_visible = False
                
                # Check against all 6 cameras
                for c in range(6):
                    # World -> Camera
                    # P_cam = T_world2cam * P_world
                    # But cam_transforms usually are Camera->World (Pose) or World->Camera (View)?
                    # Usually CARLA get_matrix() returns Camera->World (Pose).
                    # So we need Inverse. 
                    # Passing pre-inverted matrices (View Matrices) is better!
                    
                    # Assuming cam_transforms passed in are VIEW MATRICES (World -> Camera)
                    t_mat = cam_transforms[c]
                    
                    # P_cam = Mat * P_world
                    pc_x = t_mat[0, 0] * vx + t_mat[0, 1] * vy + t_mat[0, 2] * vz + t_mat[0, 3]
                    pc_y = t_mat[1, 0] * vx + t_mat[1, 1] * vy + t_mat[1, 2] * vz + t_mat[1, 3]
                    pc_z = t_mat[2, 0] * vx + t_mat[2, 1] * vy + t_mat[2, 2] * vz + t_mat[2, 3]
                    
                    # Check if point is in front of camera (Z > 0 for standard CV, X > 0 for UE4)
                    # CARLA/UE4 Camera Coordinate System: X=Forward, Y=Right, Z=Up
                    if pc_x <= 0.1: # Near plane
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
                        # Tolerance: 0.5 meters (to account for voxel size and discretization)
                        if pc_x < (depth_val + 0.5):
                            is_visible = True
                            break # Visible in at least one camera
                
                if is_visible:
                    visibility_mask[ix, iy, iz] = True
                    
    return visibility_mask

class DepthVisibilityFilter:
    def __init__(self, width=512, height=512, fov=90.0):
        self.width = width
        self.height = height
        self.fov = fov
        self.fov_rad = math.radians(fov)
        
    def compute_visibility_mask(self, occupancy, x_range, y_range, z_range, resolution, depth_maps, cam_transforms, ego_matrix=None):
        """
        计算可见性掩码
        
        Args:
            occupancy: (X, Y, Z) array
            x_range, y_range, z_range: tuples
            resolution: float
            depth_maps: (6, H, W) array
            cam_transforms: (6, 4, 4) array (Camera Poses: Camera -> World)
            ego_matrix: (4, 4) array (Ego Vehicle Pose: Ego -> World). If None, assumes World=Ego.
            
        Returns:
            mask: (X, Y, Z) boolean array
        """
        # Pre-compute Transform Matrices (Ego -> Camera)
        # We need T_ego2cam = T_world2cam * T_ego2world
        # T_world2cam = inv(cam_transforms)
        # T_ego2world = ego_matrix
        
        view_matrices = np.zeros_like(cam_transforms)
        
        if ego_matrix is not None:
            for i in range(6):
                # T_world2cam = inv(T_cam2world)
                t_world2cam = np.linalg.inv(cam_transforms[i])
                # T_ego2cam = T_world2cam @ T_ego2world
                view_matrices[i] = t_world2cam @ ego_matrix
        else:
            # Fallback: Assume World Frame inputs (or simple test)
            for i in range(6):
                view_matrices[i] = np.linalg.inv(cam_transforms[i])
            
        # Call Numba function
        # Ensure types are correct for Numba
        return _check_visibility_numba(
            occupancy,
            float(x_range[0]), float(x_range[1]),
            float(y_range[0]), float(y_range[1]),
            float(z_range[0]), float(z_range[1]),
            float(resolution),
            depth_maps.astype(np.float32),
            view_matrices.astype(np.float32),
            float(self.fov_rad),
            int(self.width),
            int(self.height)
        )

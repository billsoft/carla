
import numpy as np
import math
from dense_occupancy_collection.processing.depth_visibility import _check_visibility_numba, DepthVisibilityFilter

def test_visibility_logic():
    print("Testing Visibility Logic...")
    
    # 1. Setup Camera (Front)
    # Camera at (0,0,0) facing +X
    # World = Camera Frame
    cam_transform = np.eye(4)
    cam_transforms = np.array([cam_transform]) # (1, 4, 4)
    
    # View Matrix = Inv(Identity) = Identity
    view_matrices = np.array([np.eye(4)])
    
    # 2. Setup Depth Map
    # Flat wall at x=10.0m
    width = 10
    height = 10
    depth_map = np.full((1, height, width), 10.0, dtype=np.float32)
    
    # 3. Setup Voxels
    # Voxel A: 9.0m (In front of wall) -> Should be Visible (pc_x < depth)
    # Voxel B: 10.0m (On wall) -> Should be Visible
    # Voxel C: 11.0m (Behind wall) -> Should be Hidden (pc_x > depth)
    
    resolution = 0.5
    
    # We construct occupancy grid manually to map to these coordinates
    # Grid Origin at (0,0,0)
    # ix=18 -> x = 18*0.5 + 0.25 = 9.25m
    # ix=20 -> x = 20*0.5 + 0.25 = 10.25m
    # ix=22 -> x = 22*0.5 + 0.25 = 11.25m
    
    occupancy = np.zeros((30, 1, 1), dtype=np.uint8)
    occupancy[18, 0, 0] = 1 # 9.25m
    occupancy[20, 0, 0] = 1 # 10.25m
    occupancy[22, 0, 0] = 1 # 11.25m
    
    x_range = (0.0, 15.0)
    y_range = (-1.0, 1.0)
    z_range = (-1.0, 1.0)
    
    # Run Filter
    fov = 90.0
    fov_rad = math.radians(fov)
    
    mask = _check_visibility_numba(
        occupancy,
        float(x_range[0]), float(x_range[1]),
        float(y_range[0]), float(y_range[1]),
        float(z_range[0]), float(z_range[1]),
        float(resolution),
        depth_map,
        view_matrices,
        fov_rad,
        width,
        height
    )
    
    print(f"Voxel at 9.25m (Depth=10.0m): {'Visible' if mask[18,0,0] else 'Hidden'}")
    print(f"Voxel at 10.25m (Depth=10.0m): {'Visible' if mask[20,0,0] else 'Hidden'}")
    print(f"Voxel at 11.25m (Depth=10.0m): {'Visible' if mask[22,0,0] else 'Hidden'}")

    # Check Tolerance
    # 10.25 < 10.0 + 0.5? 10.25 < 10.5 -> True.
    # What if Voxel is 10.6m?
    # ix=21 -> 10.75m.
    occupancy[21, 0, 0] = 1
    mask = _check_visibility_numba(
        occupancy, float(x_range[0]), float(x_range[1]), float(y_range[0]), float(y_range[1]), float(z_range[0]), float(z_range[1]), float(resolution),
        depth_map, view_matrices, fov_rad, width, height
    )
    print(f"Voxel at 10.75m (Depth=10.0m): {'Visible' if mask[21,0,0] else 'Hidden'}")

if __name__ == "__main__":
    test_visibility_logic()

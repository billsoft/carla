
import os
import numpy as np
import glob
from tqdm import tqdm

def verify_dataset(dataset_root):
    print(f"Verifying dataset at: {dataset_root}")
    
    # Subdirectories to check
    subdirs = ['ego_pose', 'ego_motion', 'flow', 'flow_mask']
    
    # Collect all frame IDs from ego_pose as the baseline
    pose_files = sorted(glob.glob(os.path.join(dataset_root, 'ego_pose', '*.npy')))
    if not pose_files:
        print("Error: No ego_pose files found!")
        return
        
    frame_ids = [os.path.basename(f) for f in pose_files]
    print(f"Found {len(frame_ids)} frames. Checking integrity...")
    
    stats = {k: {'missing': 0, 'shape_error': 0, 'nan_inf': 0} for k in subdirs}
    
    for fid in tqdm(frame_ids):
        for subdir in subdirs:
            path = os.path.join(dataset_root, subdir, fid)
            
            if not os.path.exists(path):
                stats[subdir]['missing'] += 1
                continue
                
            try:
                data = np.load(path)
                
                # Check for NaN/Inf
                if np.isnan(data).any() or np.isinf(data).any():
                    stats[subdir]['nan_inf'] += 1
                    
                # Shape Checks
                if subdir == 'ego_pose':
                    if data.shape != (4, 4):
                         stats[subdir]['shape_error'] += 1
                elif subdir == 'ego_motion':
                    if data.shape != (4, 4):
                         stats[subdir]['shape_error'] += 1
                elif subdir == 'flow':
                    # Expecting (H, W, 2) or (D, H, W, 3) - just checking it's not empty
                    if data.size == 0:
                         stats[subdir]['shape_error'] += 1
                elif subdir == 'flow_mask':
                    if data.size == 0:
                         stats[subdir]['shape_error'] += 1

            except Exception as e:
                print(f"Error reading {path}: {e}")
                stats[subdir]['shape_error'] += 1

    print("\nVerification Results:")
    print(f"{'Type':<15} | {'Missing':<10} | {'Shape Err':<10} | {'NaN/Inf':<10}")
    print("-" * 55)
    
    all_good = True
    for subdir, s in stats.items():
        print(f"{subdir:<15} | {s['missing']:<10} | {s['shape_error']:<10} | {s['nan_inf']:<10}")
        if sum(s.values()) > 0:
            all_good = False
            
    if all_good:
        print("\n✅ Dataset integrity verified! All files look good.")
    else:
        print("\n⚠️ Issues found. Please review the table above.")

if __name__ == "__main__":
    verify_dataset(r"d:\code\carla\dataset_10k_bak")

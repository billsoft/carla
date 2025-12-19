import numpy as np
import argparse
from pathlib import Path
import sys

def diagnose(file_path):
    print(f"Diagnosing: {file_path}")
    try:
        data = np.load(file_path)
        print(f"Files in archive: {list(data.files)}")
        
        if 'occupancy' in data:
            occ = data['occupancy']
            print(f"\n[Occupancy]")
            print(f"  Shape: {occ.shape}")
            print(f"  Dtype: {occ.dtype}")
            print(f"  Order: {'C' if occ.flags['C_CONTIGUOUS'] else 'F'}")
            print(f"  Unique values: {np.unique(occ)}")
            
            # 检查是否有规律的条纹 (采样)
            print(f"  Sample [250, 250, :10]: {occ[250, 250, :10]}")
            
            # 统计非空比例
            non_zero = np.count_nonzero(occ)
            total = occ.size
            print(f"  Non-zero: {non_zero} / {total} ({non_zero/total*100:.2f}%)")
            
        if 'actor_ids' in data:
            ids = data['actor_ids']
            print(f"\n[Actor IDs]")
            print(f"  Shape: {ids.shape}")
            print(f"  Dtype: {ids.dtype}")
            print(f"  Unique values count: {len(np.unique(ids))}")
            print(f"  Sample unique: {np.unique(ids)[:20]}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # 默认找最新的文件
    base_dir = Path(r"d:\code\carla\dataset_output\occupancy")
    files = sorted(list(base_dir.glob("*.npz")))
    if files:
        diagnose(files[0]) # Frame 0 (God Mode)
        if len(files) > 1:
            diagnose(files[1]) # Frame 1 (Filtered)
    else:
        print("No npz files found.")

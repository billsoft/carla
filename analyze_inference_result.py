import numpy as np
import sys
from pathlib import Path

def analyze_npz(file_path):
    print(f"Analyzing: {file_path}")
    try:
        data = np.load(file_path)
        if 'occupancy' not in data:
            print("  Error: 'occupancy' key not found in npz file.")
            return

        occupancy = data['occupancy']
        print(f"  Shape: {occupancy.shape}")
        print(f"  Dtype: {occupancy.dtype}")

        unique, counts = np.unique(occupancy, return_counts=True)
        total_voxels = occupancy.size
        
        print("  Class Distribution:")
        for label, count in zip(unique, counts):
            percentage = (count / total_voxels) * 100
            print(f"    Label {label}: {count} voxels ({percentage:.2f}%)")
            
        if 0 in unique:
            print("  ✅ Label 0 (free) exists.")
        else:
            print("  ❌ Label 0 (free) DOES NOT exist!")

    except Exception as e:
        print(f"  Error loading file: {e}")

if __name__ == "__main__":
    results_dir = Path(r"d:\code\carla\inference_results_transformer")
    npz_files = list(results_dir.glob("*.npz"))
    
    if not npz_files:
        print("No .npz files found in d:\\code\\carla\\inference_results_transformer")
    else:
        # Analyze the first file as a sample
        analyze_npz(npz_files[0])

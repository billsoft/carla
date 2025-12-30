
import numpy as np
import os
import sys
from pathlib import Path

def compare_npz_files(file1, file2):
    """
    比较两个 NPZ 文件的结构和内容
    """
    print(f"对比文件:")
    print(f"  A (推理结果): {file1}")
    print(f"  B (训练数据): {file2}")
    
    if not os.path.exists(file1):
        print(f"❌ 文件 A 不存在")
        return
    if not os.path.exists(file2):
        print(f"❌ 文件 B 不存在")
        return

    data1 = np.load(file1)
    data2 = np.load(file2)
    
    keys1 = set(data1.files)
    keys2 = set(data2.files)
    
    all_keys = sorted(list(keys1 | keys2))
    
    print(f"\n字段对比:")
    print(f"{'Key':<20} | {'A Shape':<20} | {'B Shape':<20} | {'A Type':<10} | {'B Type':<10} | {'Match'}")
    print("-" * 100)
    
    for key in all_keys:
        shape1 = str(data1[key].shape) if key in data1 else "MISSING"
        shape2 = str(data2[key].shape) if key in data2 else "MISSING"
        
        dtype1 = str(data1[key].dtype) if key in data1 else "-"
        dtype2 = str(data2[key].dtype) if key in data2 else "-"
        
        match = "✅" if (shape1 == shape2 and dtype1 == dtype2) else "❌"
        if key not in keys1 or key not in keys2:
            match = "❌ (缺失)"
            
        print(f"{key:<20} | {shape1:<20} | {shape2:<20} | {dtype1:<10} | {dtype2:<10} | {match}")

    print("\n值范围检查:")
    for key in ['occupancy', 'mask', 'actor_ids']:
        if key in data1 and key in data2:
            v1 = data1[key]
            v2 = data2[key]
            print(f"{key}:")
            print(f"  A (推理): min={v1.min()}, max={v1.max()}, unique={len(np.unique(v1))}")
            print(f"  B (真值): min={v2.min()}, max={v2.max()}, unique={len(np.unique(v2))}")

    # 额外检查：分辨率和范围
    print("\n元数据检查:")
    for key in ['x_range', 'y_range', 'z_range', 'resolution', 'grid_size']:
        if key in data1 and key in data2:
            print(f"{key}:")
            print(f"  A (推理): {data1[key]}")
            print(f"  B (真值): {data2[key]}")

if __name__ == "__main__":
    # 推理结果
    inf_file = r"d:\code\carla\inference_results\000000.npz"
    # 训练数据 (假设在 dataset_10k/occupancy 下)
    train_file = r"d:\code\carla\dataset_10k\occupancy\000000.npz"
    
    compare_npz_files(inf_file, train_file)

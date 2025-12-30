#!/usr/bin/env python3
"""
检查数据集的类别分布,特别是 class 0 (空白类)
"""

import numpy as np
from pathlib import Path

def check_class_distribution():
    """检查数据集类别分布"""
    dataset_path = Path(r'd:\code\carla\dataset_10k\occupancy')

    # 获取所有 npz 文件
    files = list(dataset_path.glob('*.npz'))[:100]  # 检查前100个

    print(f"检查前 {len(files)} 个文件...")
    print("=" * 80)

    # 统计所有类别
    class_counts = {}
    total_voxels = 0

    for i, file_path in enumerate(files):
        data = np.load(file_path)
        occupancy = data['occupancy']
        mask = data['mask']

        # 只统计有效区域 (mask=True的体素)
        valid_occupancy = occupancy[mask]

        # 统计每个类别
        unique, counts = np.unique(valid_occupancy, return_counts=True)
        for cls, cnt in zip(unique, counts):
            class_counts[int(cls)] = class_counts.get(int(cls), 0) + int(cnt)
            total_voxels += int(cnt)

        if (i + 1) % 20 == 0:
            print(f"已处理 {i+1}/{len(files)} 个文件...")

    print(f"\n类别分布统计 (基于有效体素 mask=True):")
    print("=" * 80)
    print(f"{'类别':<10} | {'体素数量':>15} | {'占比':>10}")
    print("-" * 80)

    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        percentage = count / total_voxels * 100 if total_voxels > 0 else 0
        print(f"Class {cls:<3} | {count:>15,} | {percentage:>9.2f}%")

    print("-" * 80)
    print(f"{'总计':<10} | {total_voxels:>15,} | {100.0:>9.2f}%")
    print("=" * 80)

    # 关键检查
    print(f"\n关键发现:")
    print(f"  ✅ Class 0 存在: {0 in class_counts}")
    if 0 in class_counts:
        class0_count = class_counts[0]
        class0_pct = class0_count / total_voxels * 100
        print(f"  ✅ Class 0 数量: {class0_count:,} ({class0_pct:.2f}%)")
    else:
        print(f"  ❌ 数据集中没有 Class 0!")

    print(f"  总类别数: {len(class_counts)}")

    # 检查第一个文件的详细信息
    print(f"\n第一个文件详细信息:")
    first_file = files[0]
    data = np.load(first_file)
    occupancy = data['occupancy']
    mask = data['mask']

    print(f"  文件: {first_file.name}")
    print(f"  Occupancy shape: {occupancy.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask 有效体素: {mask.sum():,} / {mask.size:,} ({mask.sum()/mask.size*100:.1f}%)")
    print(f"  Occupancy 值范围: [{occupancy.min()}, {occupancy.max()}]")

    valid_occ = occupancy[mask]
    unique_in_first = np.unique(valid_occ)
    print(f"  有效区域中的类别: {unique_in_first.tolist()}")
    print(f"  Class 0 在第一个文件: {0 in unique_in_first}")

if __name__ == "__main__":
    check_class_distribution()

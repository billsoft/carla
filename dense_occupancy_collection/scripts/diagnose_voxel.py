"""
诊断体素数据
检查occupancy和actor_ids的对应关系
"""

import numpy as np
import sys
from pathlib import Path

def diagnose_npz(npz_path):
    """诊断NPZ文件"""
    print(f"\n{'='*60}")
    print(f"诊断文件: {npz_path}")
    print(f"{'='*60}\n")

    # 加载数据
    data = np.load(npz_path)

    print("📦 文件内容:")
    for key in data.files:
        print(f"  - {key}: {data[key].shape} {data[key].dtype}")

    occupancy = data['occupancy']
    actor_ids = data['actor_ids']
    mask = data['mask']

    # 基本统计
    total_voxels = occupancy.size
    occupied_voxels = np.sum(occupancy > 0)

    print(f"\n📊 基本统计:")
    print(f"  总体素数: {total_voxels:,}")
    print(f"  占用体素: {occupied_voxels:,} ({occupied_voxels/total_voxels*100:.2f}%)")

    # Actor ID统计
    print(f"\n🎭 Actor ID 分析:")

    # 正数ID (真实actors)
    positive_ids = actor_ids[actor_ids > 0]
    unique_positive_ids = np.unique(positive_ids)
    print(f"  真实Actor IDs: {sorted(list(unique_positive_ids))}")
    print(f"  真实Actor数量: {len(unique_positive_ids)}")

    # 负数ID (虚拟IDs - 静态环境)
    negative_ids = actor_ids[actor_ids < 0]
    unique_negative_ids = np.unique(negative_ids)
    print(f"  虚拟IDs (静态环境): {sorted(list(unique_negative_ids))}")
    print(f"  虚拟ID数量: {len(unique_negative_ids)}")

    # ID=0 (未分配或被过滤)
    zero_ids = actor_ids[actor_ids == 0]
    zero_with_occupancy = np.sum((actor_ids == 0) & (occupancy > 0))
    print(f"  ID=0的体素: {len(zero_ids):,} 个")
    print(f"  ⚠️ ID=0但有occupancy的: {zero_with_occupancy:,} 个 (异常!)")

    # 每个Actor的体素数量
    print(f"\n📦 每个Actor的体素数量:")
    all_unique_ids = np.unique(actor_ids[actor_ids != 0])

    actor_voxel_counts = []
    for aid in all_unique_ids:
        count = np.sum(actor_ids == aid)
        actor_voxel_counts.append((aid, count))

    # 按体素数量排序
    actor_voxel_counts.sort(key=lambda x: x[1], reverse=True)

    print(f"  前20个Actor (按体素数量):")
    for aid, count in actor_voxel_counts[:20]:
        id_type = "虚拟" if aid < 0 else "真实"
        print(f"    ID {aid:6d} ({id_type}): {count:8,} 体素")

    # Occupancy类别分布
    print(f"\n🏷️ Occupancy类别分布:")
    unique_labels, counts = np.unique(occupancy, return_counts=True)

    label_names = [
        'free', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
        'manmade', 'vegetation', 'general_object'
    ]

    for label, count in sorted(zip(unique_labels, counts), key=lambda x: -x[1])[:10]:
        if label < len(label_names):
            name = label_names[label]
        else:
            name = 'unknown'
        percentage = count / total_voxels * 100
        print(f"  [{label:2d}] {name:20s}: {count:10,} ({percentage:5.2f}%)")

    # 检查一致性
    print(f"\n✅ 一致性检查:")

    # 检查: occupancy>0的地方actor_ids应该!=0
    inconsistent = np.sum((occupancy > 0) & (actor_ids == 0))
    if inconsistent > 0:
        print(f"  ⚠️ 发现 {inconsistent:,} 个体素: occupancy>0 但 actor_ids=0")
    else:
        print(f"  ✓ 所有占用体素都有Actor ID")

    # 检查: actor_ids!=0的地方occupancy应该>0
    inconsistent2 = np.sum((occupancy == 0) & (actor_ids != 0))
    if inconsistent2 > 0:
        print(f"  ⚠️ 发现 {inconsistent2:,} 个体素: actor_ids!=0 但 occupancy=0")
    else:
        print(f"  ✓ 所有Actor ID都对应有效occupancy")

    print(f"\n{'='*60}\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python diagnose_voxel.py <path_to_npz_file>")
        print("\n示例:")
        print("  python dense_occupancy_collection/scripts/diagnose_voxel.py output/Town10HD_Opt_xxx/occupancy/000000.npz")
        sys.exit(1)

    npz_path = Path(sys.argv[1])

    if not npz_path.exists():
        print(f"❌ 文件不存在: {npz_path}")
        sys.exit(1)

    diagnose_npz(npz_path)

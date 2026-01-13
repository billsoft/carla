"""
地面层诊断脚本 - 分析体素分布
检查地面上方是否有异常的灰色/白色层

用法:
    python occnetv3_data_generator/diagnose_ground_layer.py --dataset d:/code/carla/dataset_10k_bak
"""

import numpy as np
import argparse
from pathlib import Path
from collections import Counter

# 语义类别映射 (与occupancy_config.py保持一致)
OCCUPANCY_LABELS = [
    'free',                  # 0: 空气/未标记
    'barrier',               # 1: 护栏/隔离带
    'bicycle',               # 2: 自行车
    'bus',                   # 3: 公交车
    'car',                   # 4: 轿车
    'construction_vehicle',  # 5: 工程车辆
    'motorcycle',            # 6: 摩托车
    'pedestrian',            # 7: 行人
    'traffic_cone',          # 8: 交通锥/车道线
    'trailer',               # 9: 拖车
    'truck',                 # 10: 卡车
    'driveable_surface',     # 11: 可行驶路面 (Road)
    'other_flat',            # 12: 其他平坦表面
    'sidewalk',              # 13: 人行道
    'terrain',               # 14: 地形/泥土
    'manmade',               # 15: 人造物 (建筑、杆、标志)
    'vegetation',            # 16: 植被
    'general_object',        # 17: 通用物体
]

# 定义地面相关类别
GROUND_LABELS = {11, 12, 13, 14}  # Road, OtherFlat, Sidewalk, Terrain

# 定义颜色 (用于打印)
COLORS = {
    0: "⬛",   # Free (空气) - 黑色
    1: "⬜",   # Barrier (隔离带) - 白色
    11: "🟩",  # Road (路面) - 绿色
    12: "🟨",  # OtherFlat - 黄色
    13: "🟦",  # Sidewalk (人行道) - 蓝色
    14: "🟫",  # Terrain (泥土) - 棕色
    15: "🟥",  # Manmade (建筑) - 红色
}

def load_occupancy(file_path):
    """加载occupancy数据 (支持 .npy 和 .npz 格式)"""
    file_path = Path(file_path)

    if file_path.suffix == '.npy':
        # 单独的 npy 文件
        occupancy = np.load(file_path)
        data = {
            'occupancy': occupancy,
            'x_range': [-51.2, 51.2],
            'y_range': [-51.2, 51.2],
            'z_range': [-4.0, 4.0],
            'resolution': 0.2,
            'grid_size': occupancy.shape
        }
    elif file_path.suffix == '.npz':
        # NPZ 文件
        data = np.load(file_path)
        occupancy = data['occupancy']
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    # 验证形状
    expected_shape = (512, 512, 40)
    if occupancy.shape != expected_shape:
        print(f"⚠️ 警告: 体素形状 {occupancy.shape} 与预期 {expected_shape} 不一致")

    return occupancy, data

def analyze_ground_layer(occupancy):
    """
    分析地面层体素分布

    检查:
    1. 地面层高度 (Z索引)
    2. 地面上方是否有异常层
    3. 每列的体素类型分布 (从下到上)
    """
    X, Y, Z = occupancy.shape
    print(f"\n{'='*80}")
    print(f"开始分析地面层 (Grid Size: {X}×{Y}×{Z})")
    print(f"{'='*80}\n")

    # 统计每列的地面层高度和类型
    ground_heights = np.full((X, Y), -1, dtype=int)  # -1表示没有地面
    ground_types = np.full((X, Y), -1, dtype=int)

    # 从上到下搜索地面
    for iz in range(Z - 1, -1, -1):
        layer = occupancy[:, :, iz]
        is_ground = np.isin(layer, list(GROUND_LABELS))

        # 更新还未找到地面的列
        mask = (ground_heights == -1) & is_ground
        ground_heights[mask] = iz
        ground_types[mask] = layer[mask]

    # 统计地面高度分布
    valid_heights = ground_heights[ground_heights >= 0]
    if len(valid_heights) == 0:
        print("❌ 错误: 没有找到任何地面体素!")
        return

    print(f"[1] 地面高度统计:")
    print(f"  - 最低地面 Z索引: {valid_heights.min()} (Z={valid_heights.min() * 0.2 - 4.0:.2f}m)")
    print(f"  - 最高地面 Z索引: {valid_heights.max()} (Z={valid_heights.max() * 0.2 - 4.0:.2f}m)")
    print(f"  - 平均地面 Z索引: {valid_heights.mean():.2f}")
    print(f"  - 地面高度方差: {valid_heights.std():.2f}")

    # 统计地面类型分布
    ground_type_counts = Counter(ground_types[ground_types >= 0].flatten())
    print(f"\n[2] 地面类型分布:")
    for gtype, count in ground_type_counts.most_common():
        label = OCCUPANCY_LABELS[gtype] if gtype < len(OCCUPANCY_LABELS) else f"Unknown({gtype})"
        percentage = count / len(valid_heights) * 100
        print(f"  - {label:20s}: {count:6d} ({percentage:5.2f}%)")

    # 检查地面上方是否有异常层
    print(f"\n[3] 检查地面上方体素:")

    abnormal_columns = []
    double_layer_columns = []  # 双层地面现象 (Ground -> Gap -> Ground)
    cover_layer_columns = []   # 覆盖层现象 (Ground -> Non-Ground, 如灰色Barrier覆盖棕色Terrain)

    for ix in range(X):
        for iy in range(Y):
            gh = ground_heights[ix, iy]
            if gh < 0 or gh >= Z - 2:
                continue

            # 向上搜索 3 层
            column = occupancy[ix, iy, gh:min(gh + 4, Z)]

            # 检查模式1: Ground -> Gap (0) -> Ground (双层地面)
            if len(column) >= 3:
                if column[0] in GROUND_LABELS and column[1] == 0 and column[2] in GROUND_LABELS:
                    double_layer_columns.append((ix, iy, column[:3]))

            # 检查模式2: Ground (Terrain) -> Cover Layer (Barrier/Road)
            if len(column) >= 2:
                if column[0] == 14 and column[1] in {1, 11, 12}:  # 泥土上覆盖灰色/路面
                    cover_layer_columns.append((ix, iy, column[:2]))

            # 检查一般异常: 地面上方有非零非地面体素 (排除合理物体如车辆/行人)
            reasonable_objects = {4, 7, 10, 15, 16}  # Car, Pedestrian, Truck, Manmade, Vegetation
            if len(column) >= 2:
                above = column[1]
                if above > 0 and above not in GROUND_LABELS and above not in reasonable_objects:
                    abnormal_columns.append((ix, iy, column))

    print(f"  - 双层地面异常 (Ground -> Gap -> Ground): {len(double_layer_columns)} 列")
    print(f"  - 覆盖层异常 (Ground -> Cover Layer): {len(cover_layer_columns)} 列")
    print(f"  - 一般异常 (地面上方有不合理体素): {len(abnormal_columns)} 列")

    # 显示详细样例
    if double_layer_columns:
        print(f"\n  [双层地面样例 (前5个)]:")
        for i, (ix, iy, col) in enumerate(double_layer_columns[:5]):
            types = [OCCUPANCY_LABELS[c] for c in col]
            print(f"    ({ix}, {iy}): {types}")

    if cover_layer_columns:
        print(f"\n  [覆盖层样例 (前5个)]:")
        for i, (ix, iy, col) in enumerate(cover_layer_columns[:5]):
            types = [OCCUPANCY_LABELS[c] for c in col]
            print(f"    ({ix}, {iy}): {types}")

    # 可视化: 选择一个典型列打印垂直剖面
    if abnormal_columns or double_layer_columns or cover_layer_columns:
        print(f"\n[4] 垂直剖面可视化 (选取第一个异常列):")

        example = None
        example_type = ""
        if double_layer_columns:
            ix, iy, _ = double_layer_columns[0]
            example = (ix, iy)
            example_type = "双层地面"
        elif cover_layer_columns:
            ix, iy, _ = cover_layer_columns[0]
            example = (ix, iy)
            example_type = "覆盖层"
        elif abnormal_columns:
            ix, iy, _ = abnormal_columns[0]
            example = (ix, iy)
            example_type = "一般异常"

        if example:
            ix, iy = example
            gh = ground_heights[ix, iy]

            print(f"  列坐标: ({ix}, {iy}), 地面高度: Z={gh}")
            print(f"  异常类型: {example_type}")
            print(f"\n  从下到上的体素类型 (Z={max(0, gh-2)} 到 Z={min(Z-1, gh+5)}):")

            for iz in range(max(0, gh - 2), min(Z, gh + 6)):
                voxel = occupancy[ix, iy, iz]
                label = OCCUPANCY_LABELS[voxel] if voxel < len(OCCUPANCY_LABELS) else f"Unknown({voxel})"
                color = COLORS.get(voxel, "⬜")

                marker = ""
                if iz == gh:
                    marker = " ← 地面层"
                elif iz > gh:
                    marker = f" ← +{iz - gh}"

                print(f"    Z={iz:2d} ({(iz * 0.2 - 4.0):5.2f}m): {color} {label:20s} {marker}")

    # 统计总结
    print(f"\n[5] 问题总结:")
    total_columns = X * Y
    abnormal_ratio = (len(double_layer_columns) + len(cover_layer_columns)) / total_columns * 100

    print(f"  - 总列数: {total_columns}")
    print(f"  - 异常列数: {len(double_layer_columns) + len(cover_layer_columns)}")
    print(f"  - 异常比例: {abnormal_ratio:.2f}%")

    if abnormal_ratio > 1.0:
        print(f"\n⚠️  警告: 发现显著的地面层异常 ({abnormal_ratio:.1f}%)!")
        print(f"   可能原因:")
        print(f"   1. Map API 和 Static Mesh 双重生成地面,导致双层")
        print(f"   2. UE5 Mesh 几何位置略高于 Map 逻辑位置,形成浮空层")
        print(f"   3. 向下填充逻辑复制了地表颜色,导致灰色墙")
        print(f"\n   建议修复:")
        print(f"   1. 在 ground_truth_voxel_generator.py 中排除地面类型的 Static Mesh")
        print(f"   2. 地下填充统一使用 Terrain (14) 而非复制地表材质")
    else:
        print(f"\n✅ 地面层正常,异常比例很低 ({abnormal_ratio:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='地面层诊断工具')
    parser.add_argument('--dataset', default='d:/code/carla/dataset_10k_bak',
                        help='数据集目录')
    parser.add_argument('--frame', type=int, default=0,
                        help='要分析的帧编号 (默认: 0)')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)

    # 查找occupancy文件
    occupancy_dir = dataset_dir / 'occupancy'
    if not occupancy_dir.exists():
        print(f"❌ 错误: 目录不存在: {occupancy_dir}")
        return

    # 尝试加载指定帧 (支持 .npy 和 .npz)
    npy_files = sorted(occupancy_dir.glob('*.npy'))
    npz_files = sorted(occupancy_dir.glob('*.npz'))
    all_files = sorted(npy_files + npz_files)

    if not all_files:
        print(f"❌ 错误: 未找到任何 .npy 或 .npz 文件")
        return

    if args.frame >= len(all_files):
        print(f"⚠️ 警告: 帧 {args.frame} 超出范围,使用第一帧")
        args.frame = 0

    target_file = all_files[args.frame]
    print(f"分析文件: {target_file}")

    # 加载并分析
    occupancy, data = load_occupancy(target_file)

    # 打印基本信息
    print(f"\n数据集信息:")
    print(f"  - 体素尺寸: {occupancy.shape}")
    print(f"  - X 范围: {data.get('x_range', 'N/A')}")
    print(f"  - Y 范围: {data.get('y_range', 'N/A')}")
    print(f"  - Z 范围: {data.get('z_range', 'N/A')}")
    print(f"  - 分辨率: {data.get('resolution', 'N/A')} m")

    # 体素统计
    total = occupancy.size
    occupied = np.count_nonzero(occupancy)
    print(f"  - 占用率: {occupied}/{total} ({occupied/total*100:.2f}%)")

    # 开始分析
    analyze_ground_layer(occupancy)


if __name__ == '__main__':
    main()

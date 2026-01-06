"""
可视化数据集中的 DNG 相机图像和占用网格
用于验证数据采集质量
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_dng_image(dng_path):
    """加载 DNG 图像 (Bayer RAW)"""
    try:
        import rawpy
        with rawpy.imread(dng_path) as raw:
            # 使用相机白平衡
            rgb = raw.postprocess(use_camera_wb=True)
        return rgb
    except ImportError:
        print("⚠️ rawpy 未安装,尝试使用 OpenCV...")
        import cv2
        img = cv2.imread(dng_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"❌ 无法加载: {dng_path}")
            return None

        # 简单的 Bayer → RGB 转换
        rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_RGGB2RGB)
        return rgb

def load_occupancy(npz_path):
    """加载占用网格"""
    data = np.load(npz_path)
    occupancy = data['occupancy']
    mask = data.get('mask', np.ones_like(occupancy, dtype=bool))

    return occupancy, mask, data

def visualize_sample(dataset_dir, sample_id):
    """可视化一个样本"""

    dataset_path = Path(dataset_dir)
    print(f"数据集路径: {dataset_path}")

    # 相机名称
    camera_names = [
        'cam_front_main',
        'cam_front_wide',
        'cam_front_narrow',
        'cam_left_pillar',
        'cam_right_pillar',
        'cam_left_repeater',
        'cam_right_repeater',
        'cam_rear'
    ]

    # 加载相机图像
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f"Sample {sample_id:06d} - Camera Views", fontsize=16)

    for idx, cam_name in enumerate(camera_names):
        row = idx // 3
        col = idx % 3

        dng_path = dataset_path / 'cameras' / cam_name / f"{sample_id:06d}.dng"

        if dng_path.exists():
            print(f"加载: {cam_name}")
            img = load_dng_image(str(dng_path))

            if img is not None:
                axes[row, col].imshow(img)
                axes[row, col].set_title(cam_name)
                axes[row, col].axis('off')
            else:
                axes[row, col].text(0.5, 0.5, f"{cam_name}\n加载失败",
                                   ha='center', va='center')
                axes[row, col].axis('off')
        else:
            axes[row, col].text(0.5, 0.5, f"{cam_name}\n不存在",
                               ha='center', va='center')
            axes[row, col].axis('off')

    # 隐藏多余的子图
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"visualization_{sample_id:06d}_cameras.png", dpi=150)
    print(f"✅ 保存相机图像: visualization_{sample_id:06d}_cameras.png")

    # 加载占用网格
    occ_path = dataset_path / 'occupancy' / f"{sample_id:06d}.npz"

    if not occ_path.exists():
        print(f"❌ 占用网格不存在: {occ_path}")
        plt.show()
        return

    print(f"加载占用网格: {occ_path}")
    occupancy, mask, data = load_occupancy(str(occ_path))

    # 可视化占用网格
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Sample {sample_id:06d} - Occupancy Grid", fontsize=16)

    # 1. 俯视图 (BEV)
    bev = np.max(occupancy, axis=2)  # Z 轴最大值投影
    im1 = axes[0, 0].imshow(bev.T, origin='lower', cmap='tab20', vmin=0, vmax=17)
    axes[0, 0].set_title("BEV (Bird's Eye View)")
    axes[0, 0].set_xlabel("X (forward)")
    axes[0, 0].set_ylabel("Y (left)")
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. 前视图 (Front View)
    front = np.max(occupancy[:, occupancy.shape[1]//2, :], axis=0)  # 中间切片
    im2 = axes[0, 1].imshow(front.T, origin='lower', cmap='tab20', vmin=0, vmax=17)
    axes[0, 1].set_title("Front View (Y=0)")
    axes[0, 1].set_xlabel("X (forward)")
    axes[0, 1].set_ylabel("Z (up)")
    plt.colorbar(im2, ax=axes[0, 1])

    # 3. 统计信息
    axes[1, 0].axis('off')

    # 统计类别分布
    unique, counts = np.unique(occupancy, return_counts=True)
    class_names = [
        "Free", "Building", "Fence", "Other", "Pedestrian", "Pole",
        "RoadLine", "Road", "Sidewalk", "Vegetation", "Vehicle", "Wall",
        "TrafficSign", "Sky", "Ground", "Bridge", "RailTrack", "GuardRail"
    ]

    stats_text = f"Grid Size: {occupancy.shape}\n"
    stats_text += f"Resolution: {data.get('resolution', 'N/A'):.2f}m\n"
    stats_text += f"X Range: {data.get('x_range', 'N/A')}\n"
    stats_text += f"Y Range: {data.get('y_range', 'N/A')}\n"
    stats_text += f"Z Range: {data.get('z_range', 'N/A')}\n\n"
    stats_text += "Class Distribution:\n"

    for cls, count in zip(unique, counts):
        if cls < len(class_names):
            name = class_names[int(cls)]
        else:
            name = f"Class {cls}"
        percentage = count / occupancy.size * 100
        stats_text += f"  {cls:2d} {name:15s}: {count:8d} ({percentage:5.2f}%)\n"

    axes[1, 0].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center')

    # 4. 类别分布柱状图
    axes[1, 1].bar(unique, counts)
    axes[1, 1].set_xlabel("Class ID")
    axes[1, 1].set_ylabel("Voxel Count")
    axes[1, 1].set_title("Class Distribution")
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"visualization_{sample_id:06d}_occupancy.png", dpi=150)
    print(f"✅ 保存占用网格: visualization_{sample_id:06d}_occupancy.png")

    plt.show()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="可视化数据集样本")
    parser.add_argument('--dataset', type=str, default='dataset_10k',
                       help='数据集目录路径')
    parser.add_argument('--sample', type=int, default=0,
                       help='样本 ID (默认 0)')

    args = parser.parse_args()

    # 验证数据集路径
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ 数据集不存在: {dataset_path}")
        print(f"提示: 请确认路径或使用绝对路径")
        return

    print(f"可视化样本 {args.sample:06d}...")
    visualize_sample(args.dataset, args.sample)

if __name__ == "__main__":
    main()

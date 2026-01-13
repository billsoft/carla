"""
可视化 OccNetV3 数据集 (合并窗口版)
支持加载 DNG (Bayer RGGB) 和 NPY 格式
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse

# 相机映射
CAMERA_MAPPING = {
    0: 'front_main',
    1: 'front_wide',
    2: 'front_narrow',
    3: 'left_pillar',
    4: 'right_pillar',
    5: 'left_repeater',
    6: 'right_repeater',
    7: 'rear',
}

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

        # 简单的 Bayer → RGB 转换 (假设 RGGB)
        rgb = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
        
        # 归一化并增强亮度 (因为线性RAW通常较暗)
        rgb = rgb.astype(np.float32) / 4095.0 # 12-bit
        rgb = np.clip(rgb * 2.5, 0, 1) # 简单曝光补偿
        return rgb

def load_npy_image(npy_path):
    """加载 NPY 图像 (float16)"""
    img = np.load(npy_path)
    # (1, H, W) -> (H, W)
    if len(img.shape) == 3:
        img = img[0]
    return img

def load_occupancy(npy_path):
    """加载占用网格"""
    occupancy = np.load(npy_path)
    return occupancy

def visualize_sample(dataset_dir, sample_id):
    """可视化一个样本 (所有视图在一个窗口)"""
    dataset_path = Path(dataset_dir)
    print(f"加载样本: {sample_id}")

    # 创建大图
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig)
    fig.suptitle(f"Sample {sample_id}", fontsize=16)

    # 1. 加载并显示相机图像
    img_dir = dataset_path / 'images' / sample_id
    
    for idx in range(8):
        cam_name = CAMERA_MAPPING.get(idx, f"cam_{idx}")
        
        # 尝试加载 DNG
        dng_path = img_dir / f"cam_{idx}.dng"
        npy_path = img_dir / f"cam_{idx}.npy"
        
        img = None
        if dng_path.exists():
            img = load_dng_image(str(dng_path))
            type_label = "DNG"
        elif npy_path.exists():
            img = load_npy_image(str(npy_path))
            type_label = "NPY"
            
        # 计算子图位置 (前两行)
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])
        
        if img is not None:
            if len(img.shape) == 2: # Gray
                ax.imshow(img, cmap='gray')
            else: # RGB
                ax.imshow(img)
            ax.set_title(f"{cam_name} ({type_label})")
        else:
            ax.text(0.5, 0.5, "Missing", ha='center', va='center')
            
        ax.axis('off')

    # 2. 加载并显示 Occupancy
    occ_path = dataset_path / 'occupancy' / f"{sample_id}.npy"
    
    if occ_path.exists():
        occupancy = load_occupancy(str(occ_path))
        
        # BEV View (Top-down) -> 放在第3行左侧
        ax_bev = fig.add_subplot(gs[2, 0:2])
        bev = np.max(occupancy, axis=2) # Z max
        im1 = ax_bev.imshow(bev.T, origin='lower', cmap='tab20', vmin=0, vmax=17)
        ax_bev.set_title("Occupancy BEV")
        ax_bev.set_xlabel("X (Forward)")
        ax_bev.set_ylabel("Y (Left)")
        plt.colorbar(im1, ax=ax_bev)
        
        # Front View (Side) -> 放在第3行右侧
        ax_front = fig.add_subplot(gs[2, 2:4])
        # Y max (side projection)
        front = np.max(occupancy, axis=1) 
        im2 = ax_front.imshow(front.T, origin='lower', cmap='tab20', vmin=0, vmax=17)
        ax_front.set_title("Occupancy Front View (Side Projection)")
        ax_front.set_xlabel("X (Forward)")
        ax_front.set_ylabel("Z (Up)")
        plt.colorbar(im2, ax=ax_front)
        
    else:
        ax_msg = fig.add_subplot(gs[2, :])
        ax_msg.text(0.5, 0.5, "Occupancy Missing", ha='center', va='center')
        ax_msg.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize OccNetV3 Dataset')
    parser.add_argument('--dataset', default='dataset_occnet_v3', help='Dataset root directory')
    parser.add_argument('--sample', help='Specific sample ID (optional)')
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    # 获取所有样本
    img_root = dataset_path / 'images'
    if not img_root.exists():
        print(f"Images directory not found: {img_root}")
        return
        
    samples = sorted([d.name for d in img_root.iterdir() if d.is_dir()])
    
    if not samples:
        print("No samples found.")
        return
        
    print(f"Found {len(samples)} samples.")
    
    if args.sample:
        if args.sample in samples:
            visualize_sample(args.dataset, args.sample)
        else:
            print(f"Sample {args.sample} not found.")
    else:
        # 默认显示第一个
        visualize_sample(args.dataset, samples[0])
        print(f"\nTip: Use --sample to specify a sample ID")

if __name__ == '__main__':
    main()

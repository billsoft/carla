#!/usr/bin/env python3
# visualize.py
"""
Occupancy Network 可视化脚本

功能:
1. 可视化预测结果（3D 体素渲染）
2. 可视化 BEV 特征
3. 对比 GT 和预测

用法:
    python visualize.py --checkpoint checkpoints/best.pth --data_root /path/to/dataset --sample_idx 0
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.occ_network import OccupancyNetwork
from datasets.carla_occ_dataset import CARLAOccDataset
from configs.default_config import CLASS_NAMES, CLASS_COLORS


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Occupancy Network')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='检查点路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='要可视化的样本索引')
    parser.add_argument('--save_dir', type=str, default='visualizations',
                        help='保存目录')
    parser.add_argument('--show', action='store_true',
                        help='显示图像')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> OccupancyNetwork:
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = OccupancyNetwork(
        num_cameras=8,
        img_size=(384, 640),
        backbone_type='resnet50',
        backbone_pretrained=False,
        embed_dim=256,
        num_heads=8,
        num_transformer_layers=6,
        bev_h=200,
        bev_w=200,
        num_classes=18,
        num_heights=16,
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def get_colormap():
    """获取类别颜色映射"""
    # 归一化颜色到 [0, 1]
    cmap = np.array(CLASS_COLORS) / 255.0
    return cmap


def visualize_camera_images(images: torch.Tensor, save_path: str = None):
    """
    可视化 8 个相机图像
    
    Args:
        images: [8, 3, H, W] 图像张量
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    camera_names = [
        'Front Main', 'Front Wide', 'Front Narrow', 'Left Pillar',
        'Right Pillar', 'Left Repeater', 'Right Repeater', 'Rear'
    ]
    
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu()
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(camera_names[i])
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved camera images to {save_path}')
    
    return fig


def visualize_bev_occupancy(
    occupancy: np.ndarray,
    title: str = 'BEV Occupancy',
    save_path: str = None,
):
    """
    可视化 BEV 视角的占用网格（沿 Z 轴投影）
    
    Args:
        occupancy: [H, W, Z] 体素网格
    """
    cmap = get_colormap()
    
    # 沿 Z 轴投影（取最常见的非空类别）
    H, W, Z = occupancy.shape
    bev = np.zeros((H, W), dtype=np.uint8)
    
    for i in range(H):
        for j in range(W):
            column = occupancy[i, j, :]
            non_free = column[column > 0]
            if len(non_free) > 0:
                # 取最常见的类别
                bev[i, j] = np.bincount(non_free).argmax()
    
    # 创建 RGB 图像
    bev_rgb = cmap[bev]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(bev_rgb, origin='lower')
    ax.set_title(title)
    ax.set_xlabel('Y (left-right)')
    ax.set_ylabel('X (front-back)')
    
    # 添加图例
    patches = []
    for i, name in enumerate(CLASS_NAMES):
        if i == 0:
            continue  # 跳过 free
        if np.any(bev == i):
            patch = plt.Rectangle((0, 0), 1, 1, fc=cmap[i])
            patches.append((patch, name))
    
    if patches:
        ax.legend(
            [p[0] for p in patches],
            [p[1] for p in patches],
            loc='upper right',
            fontsize=8,
        )
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved BEV visualization to {save_path}')
    
    return fig


def visualize_3d_occupancy(
    occupancy: np.ndarray,
    title: str = '3D Occupancy',
    save_path: str = None,
    subsample: int = 4,
):
    """
    可视化 3D 体素网格
    
    Args:
        occupancy: [H, W, Z] 体素网格
        subsample: 下采样因子（减少点数以加快渲染）
    """
    cmap = get_colormap()
    
    # 获取非空体素
    non_free_mask = occupancy > 0
    
    # 下采样
    if subsample > 1:
        non_free_mask = non_free_mask[::subsample, ::subsample, ::subsample]
        occupancy_sub = occupancy[::subsample, ::subsample, ::subsample]
    else:
        occupancy_sub = occupancy
    
    # 获取坐标
    coords = np.array(np.where(non_free_mask)).T  # [N, 3]
    
    if len(coords) == 0:
        print('Warning: No non-free voxels to visualize')
        return None
    
    # 限制点数
    max_points = 50000
    if len(coords) > max_points:
        indices = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[indices]
    
    # 获取颜色
    colors_rgba = []
    for coord in coords:
        cls = occupancy_sub[coord[0], coord[1], coord[2]]
        color = list(cmap[cls]) + [0.8]  # 添加 alpha
        colors_rgba.append(color)
    
    colors_rgba = np.array(colors_rgba)
    
    # 3D 绘图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        coords[:, 1],  # Y
        coords[:, 0],  # X
        coords[:, 2],  # Z
        c=colors_rgba,
        s=1,
        alpha=0.8,
    )
    
    ax.set_xlabel('Y (left-right)')
    ax.set_ylabel('X (front-back)')
    ax.set_zlabel('Z (height)')
    ax.set_title(title)
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved 3D visualization to {save_path}')
    
    return fig


def visualize_comparison(
    gt: np.ndarray,
    pred: np.ndarray,
    save_path: str = None,
):
    """
    对比 GT 和预测
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    cmap = get_colormap()
    
    # 投影到 BEV
    def project_to_bev(occ):
        H, W, Z = occ.shape
        bev = np.zeros((H, W), dtype=np.uint8)
        for i in range(H):
            for j in range(W):
                column = occ[i, j, :]
                non_free = column[column > 0]
                if len(non_free) > 0:
                    bev[i, j] = np.bincount(non_free).argmax()
        return bev
    
    bev_gt = project_to_bev(gt)
    bev_pred = project_to_bev(pred)
    
    # GT
    axes[0].imshow(cmap[bev_gt], origin='lower')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(cmap[bev_pred], origin='lower')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Difference
    diff = np.zeros((bev_gt.shape[0], bev_gt.shape[1], 3))
    diff[bev_gt == bev_pred] = [0, 1, 0]  # 正确: 绿色
    diff[bev_gt != bev_pred] = [1, 0, 0]  # 错误: 红色
    diff[(bev_gt == 0) & (bev_pred == 0)] = [0, 0, 0]  # 都是 free: 黑色
    
    axes[2].imshow(diff, origin='lower')
    axes[2].set_title('Difference (Green=Correct, Red=Error)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved comparison to {save_path}')
    
    return fig


def visualize_bev_features(
    bev_features: torch.Tensor,
    save_path: str = None,
):
    """
    可视化 BEV 特征图
    
    Args:
        bev_features: [C, H, W] BEV 特征
    """
    features = bev_features.cpu().numpy()
    
    # 选择几个通道可视化
    num_channels = min(16, features.shape[0])
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        if i < num_channels:
            im = ax.imshow(features[i], cmap='viridis')
            ax.set_title(f'Channel {i}')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle('BEV Feature Maps')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved BEV features to {save_path}')
    
    return fig


@torch.no_grad()
def main():
    args = parse_args()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, device)
    
    # 加载数据
    print(f'Loading dataset from {args.data_root}...')
    dataset = CARLAOccDataset(
        data_root=args.data_root,
        split='all',
        img_size=(384, 640),
        grid_size=(200, 200, 16),
    )
    
    print(f'Total samples: {len(dataset)}')
    
    # 获取样本
    sample = dataset[args.sample_idx]
    images = sample['images'].unsqueeze(0).to(device)
    gt_occupancy = sample['occupancy'].numpy()
    mask = sample['mask'].numpy()
    
    print(f'Sample {args.sample_idx}:')
    print(f'  Images shape: {images.shape}')
    print(f'  Occupancy shape: {gt_occupancy.shape}')
    
    # 推理
    print('Running inference...')
    outputs = model(images)
    pred_logits = outputs['occ_logits']
    bev_features = outputs['bev_features']
    
    pred_occupancy = pred_logits.argmax(dim=1).squeeze(0).cpu().numpy()
    
    # 可视化
    print('Generating visualizations...')
    
    # 1. 相机图像
    visualize_camera_images(
        sample['images'],
        save_path=save_dir / f'sample_{args.sample_idx}_cameras.png'
    )
    
    # 2. GT BEV
    visualize_bev_occupancy(
        gt_occupancy,
        title='Ground Truth (BEV)',
        save_path=save_dir / f'sample_{args.sample_idx}_gt_bev.png'
    )
    
    # 3. Pred BEV
    visualize_bev_occupancy(
        pred_occupancy,
        title='Prediction (BEV)',
        save_path=save_dir / f'sample_{args.sample_idx}_pred_bev.png'
    )
    
    # 4. 对比
    visualize_comparison(
        gt_occupancy,
        pred_occupancy,
        save_path=save_dir / f'sample_{args.sample_idx}_comparison.png'
    )
    
    # 5. 3D 可视化（GT）
    visualize_3d_occupancy(
        gt_occupancy,
        title='Ground Truth (3D)',
        save_path=save_dir / f'sample_{args.sample_idx}_gt_3d.png'
    )
    
    # 6. 3D 可视化（Pred）
    visualize_3d_occupancy(
        pred_occupancy,
        title='Prediction (3D)',
        save_path=save_dir / f'sample_{args.sample_idx}_pred_3d.png'
    )
    
    # 7. BEV 特征
    visualize_bev_features(
        bev_features.squeeze(0),
        save_path=save_dir / f'sample_{args.sample_idx}_bev_features.png'
    )
    
    print(f'\nAll visualizations saved to {save_dir}')
    
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Bayer Occupancy Network 推理脚本

用法:
    python inference_bayer.py --checkpoint outputs/bayer_raw/xxx/epoch_019.pth --dataset dataset_10k --num-samples 10
"""

import os
import sys
import argparse
from pathlib import Path
import time

import torch
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models import build_bayer_occ_net
from data.carla_dataset_bayer import CARLADatasetBayer


def parse_args():
    parser = argparse.ArgumentParser(description='Bayer OccNet Inference')

    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--dataset', type=str, required=True, help='数据集根目录')
    parser.add_argument('--num-samples', type=int, default=10, help='推理样本数量')
    parser.add_argument('--output-dir', type=str, default='inference_results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--img-size', type=int, nargs=2, default=[384, 640], help='图像尺寸')

    return parser.parse_args()


def compute_metrics(pred, gt, mask):
    """
    计算评估指标

    Args:
        pred: [X, Y, Z] 预测类别 (uint8)
        gt: [X, Y, Z] 真值类别 (uint8)
        mask: [X, Y, Z] 有效掩码 (bool)

    Returns:
        metrics: dict
    """
    # 只计算有效区域
    valid = mask > 0
    pred_valid = pred[valid]
    gt_valid = gt[valid]

    # 总体准确率
    accuracy = (pred_valid == gt_valid).sum() / len(pred_valid)

    # Per-class IoU
    num_classes = max(pred_valid.max(), gt_valid.max()) + 1
    iou_per_class = []

    for cls in range(num_classes):
        pred_cls = (pred_valid == cls)
        gt_cls = (gt_valid == cls)

        intersection = (pred_cls & gt_cls).sum()
        union = (pred_cls | gt_cls).sum()

        if union > 0:
            iou = intersection / union
            iou_per_class.append(iou)

    miou = np.mean(iou_per_class) if len(iou_per_class) > 0 else 0.0

    return {
        'accuracy': float(accuracy),
        'miou': float(miou),
        'num_classes': int(num_classes)
    }


def resize_occupancy(occ_pred, target_size):
    """
    调整占用网格尺寸

    Args:
        occ_pred: [X1, Y1, Z1] 预测网格
        target_size: (X2, Y2, Z2) 目标尺寸

    Returns:
        resized: [X2, Y2, Z2] 调整后的网格
    """
    from scipy.ndimage import zoom

    X1, Y1, Z1 = occ_pred.shape
    X2, Y2, Z2 = target_size

    zoom_factors = (X2 / X1, Y2 / Y1, Z2 / Z1)
    resized = zoom(occ_pred, zoom_factors, order=0)  # 最近邻插值

    return resized.astype(occ_pred.dtype)


def main():
    args = parse_args()

    print("=" * 80)
    print("Bayer Occupancy Network 推理".center(80))
    print("=" * 80)
    print(f"\n配置:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Num Samples: {args.num_samples}")
    print(f"  Output Dir: {args.output_dir}")

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n输出目录: {output_dir.absolute()}")

    # 加载数据集
    print(f"\n加载数据集...")
    dataset = CARLADatasetBayer(
        root=args.dataset,
        img_size=tuple(args.img_size),
        augment=False  # 推理时不增强
    )

    num_samples = min(args.num_samples, len(dataset))
    print(f"数据集样本总数: {len(dataset)}")
    print(f"推理样本数: {num_samples}")

    # 构建模型
    print(f"\n构建模型...")
    model = build_bayer_occ_net(
        num_classes=18,
        grid_size=(200, 200, 16),
        img_size=tuple(args.img_size),
        backbone_width_mult=1.0,
        fpn_channels=128,
        bev_size=(100, 100),
        hidden_channels=64
    ).to(device)

    # 加载检查点
    print(f"\n加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Epoch: {epoch}")

    # 推理
    print(f"\n开始推理...")
    print("=" * 80)

    all_metrics = []
    inference_times = []

    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="推理进度"):
            # 加载数据
            sample = dataset[idx]
            images = sample['images'].unsqueeze(0).to(device)  # [1, N_cam, 1, H, W]
            gt_occupancy = sample['occupancy'].numpy()  # [X_gt, Y_gt, Z_gt]
            gt_mask = sample['mask'].numpy()  # [X_gt, Y_gt, Z_gt]

            # 推理
            start_time = time.time()
            occ_logits = model(images)  # [1, 18, X, Y, Z]
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # 预测类别
            occ_pred = occ_logits.argmax(dim=1).cpu().numpy()[0]  # [200, 200, 16]

            # 注意: gt_occupancy 和 gt_mask 已经是 (200, 200, 16) 了（数据集加载器已下采样）
            # 直接计算指标，无需调整尺寸
            mask_valid = (gt_mask > 0)

            # 计算指标
            metrics = compute_metrics(occ_pred, gt_occupancy, mask_valid)
            all_metrics.append(metrics)

            # 保存预测结果（与 viewer 兼容的格式）
            # 注意: viewer 期望的是原始网格尺寸，这里我们保存模型的原始输出
            output_file = output_dir / f"{idx:06d}.npz"

            # 获取数据集的元数据
            dataset_root = Path(args.dataset)
            occ_file = dataset_root / 'occupancy' / f"{idx:06d}.npz"

            if occ_file.exists():
                gt_data = np.load(str(occ_file))
                x_range = gt_data['x_range']
                y_range = gt_data['y_range']
                z_range = gt_data['z_range']
                resolution = gt_data['resolution']
                town = str(gt_data['town'])
            else:
                # 默认值
                x_range = np.array([-25.0, 25.0])
                y_range = np.array([-25.0, 25.0])
                z_range = np.array([-2.0, 6.0])
                resolution = 0.1
                town = 'unknown'

            # 保存为 viewer 兼容格式
            # 直接保存模型输出 (200, 200, 16)，不进行上采样

            # 计算新的分辨率
            # 根据数据集元数据动态计算
            x_span = x_range[1] - x_range[0]  # 例如: 100米
            new_resolution = x_span / occ_pred.shape[0]  # 100 / 200 = 0.5米

            # gt_mask 已经是 (200, 200, 16) 了，直接使用
            # actor_ids 在推理中没有，不要保存它，否则 viewer 会因为全0而隐藏所有体素

            np.savez_compressed(
                output_file,
                occupancy=occ_pred.astype(np.uint8),
                # actor_ids 不保存，viewer 会默认显示所有体素
                mask=mask_valid.astype(bool),  # ✅ 修复：使用 bool 类型（与真值一致）
                town=town,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                resolution=np.array([new_resolution]),  # 更新分辨率 (0.5m)
                grid_size=np.array(occ_pred.shape)      # (200, 200, 16)
            )

    # 统计结果
    print("\n" + "=" * 80)
    print("推理统计".center(80))
    print("=" * 80)

    avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
    avg_miou = np.mean([m['miou'] for m in all_metrics])
    avg_time = np.mean(inference_times)

    print(f"\n平均指标 (前 {num_samples} 个样本):")
    print(f"  Accuracy: {avg_accuracy*100:.2f}%")
    print(f"  mIoU: {avg_miou*100:.2f}%")
    print(f"  推理时间: {avg_time*1000:.1f} ms/sample")

    # 保存详细指标
    metrics_file = output_dir / 'metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Bayer OccNet 推理指标\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Num Samples: {num_samples}\n\n")
        f.write(f"Average Accuracy: {avg_accuracy*100:.2f}%\n")
        f.write(f"Average mIoU: {avg_miou*100:.2f}%\n")
        f.write(f"Average Inference Time: {avg_time*1000:.1f} ms\n\n")
        f.write("Per-sample Results:\n")
        f.write("-" * 80 + "\n")
        for idx, m in enumerate(all_metrics):
            f.write(f"Sample {idx:03d}: Acc={m['accuracy']*100:.2f}%, mIoU={m['miou']*100:.2f}%\n")

    print(f"\n详细指标已保存到: {metrics_file}")
    print(f"\n✅ 推理完成！")
    print(f"   输出文件: {num_samples} 个 .npz 文件")
    print(f"   位置: {output_dir.absolute()}")
    print("\n使用 occupancy_viewer 查看:")
    print(f"   1. 修改 occupancy_viewer/run_viewer.py 中的 DATA_DIR 为: {output_dir.absolute()}")
    print(f"   2. 运行: python occupancy_viewer/run_viewer.py")
    print(f"   3. 浏览器访问: http://localhost:8085/")
    print("=" * 80)


if __name__ == '__main__':
    main()

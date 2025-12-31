#!/usr/bin/env python3
# inference_transformer.py
"""
Transformer Occupancy Network 推理脚本

用法:
    python inference_transformer.py --checkpoint best.pth --dataset /path/to/dataset
"""

import os
import sys
import argparse
from pathlib import Path
import time

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.transformer_occ import build_transformer_occ_net
from models import TransformerOccNetBalanced
from data.carla_dataset_bayer import CARLADatasetBayer


def parse_args():
    parser = argparse.ArgumentParser(description='Transformer OccNet Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点')
    parser.add_argument('--dataset', type=str, required=True, help='数据集目录')
    parser.add_argument('--num-samples', type=int, default=10, help='推理样本数')
    parser.add_argument('--output-dir', type=str, default='inference_results_transformer')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--img-size', type=int, nargs=2, default=[960, 1280])
    parser.add_argument('--model-type', type=str, default='lite', choices=['standard', 'lite', 'balanced'])
    return parser.parse_args()


def compute_metrics(pred, gt, mask):
    """计算评估指标"""
    valid = mask > 0
    pred_valid = pred[valid]
    gt_valid = gt[valid]
    
    if len(pred_valid) == 0:
        return {'accuracy': 0.0, 'miou': 0.0}
        
    accuracy = (pred_valid == gt_valid).sum() / len(pred_valid)
    
    num_classes = max(pred_valid.max(), gt_valid.max()) + 1
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_cls = (pred_valid == cls)
        gt_cls = (gt_valid == cls)
        intersection = (pred_cls & gt_cls).sum()
        union = (pred_cls | gt_cls).sum()
        if union > 0:
            iou_per_class.append(intersection / union)
            
    miou = np.mean(iou_per_class) if iou_per_class else 0.0
    
    return {
        'accuracy': float(accuracy),
        'miou': float(miou),
        'num_classes': int(num_classes)
    }


def main():
    args = parse_args()
    
    print("=" * 70)
    print("Transformer Occupancy Network 推理".center(70))
    print("=" * 70)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 数据集
    print("\n加载数据集...")
    dataset = CARLADatasetBayer(
        root=args.dataset,
        img_size=tuple(args.img_size),
        augment=False
    )
    num_samples = min(args.num_samples, len(dataset))
    print(f"样本数: {num_samples}")
    
    # 模型
    print(f"\n构建 {args.model_type} 模型...")
    if args.model_type == 'balanced':
        # Balanced-Pro 配置 (需与 train_balanced.py 一致)
        model = TransformerOccNetBalanced(
            num_cameras=8,
            img_size=tuple(args.img_size),
            embed_dim=256,
            encoder_layers=5,
            decoder_layers=4,
            num_heads=8,
            bev_size=(50, 50),
            num_height_levels=8,
            num_deform_points=6,
            output_grid_size=(200, 200, 16),
            use_checkpoint=False  # 推理时不需 checkpoint
        ).to(device)
    else:
        model = build_transformer_occ_net(
            model_type=args.model_type,
            num_classes=18,
            img_size=tuple(args.img_size),
            output_grid_size=(200, 200, 16)
        ).to(device)
    
    # 加载权重
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Epoch: {epoch}")
    
    # 参数统计
    params = model.get_params_summary()
    print(f"\n模型参数: {params['total']:.2f}M")
    
    # 推理
    print("\n" + "=" * 70)
    print("开始推理...")
    
    all_metrics = []
    inference_times = []
    
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="推理进度"):
            sample = dataset[idx]
            images = sample['images'].unsqueeze(0).to(device)
            gt_occ = sample['occupancy'].numpy()
            gt_mask = sample['mask'].numpy()
            
            # 推理
            start_time = time.time()
            occ_logits = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 预测
            occ_pred = occ_logits.argmax(dim=1).cpu().numpy()[0]
            
            # 评估
            metrics = compute_metrics(occ_pred, gt_occ, gt_mask)
            all_metrics.append(metrics)
            
            # 保存
            np.savez_compressed(
                output_dir / f'{idx:06d}.npz',
                occupancy=occ_pred.astype(np.uint8),
                mask=(gt_mask > 0).astype(bool)
            )
            
    # 统计
    print("\n" + "=" * 70)
    print("推理统计".center(70))
    print("=" * 70)
    
    avg_acc = np.mean([m['accuracy'] for m in all_metrics])
    avg_miou = np.mean([m['miou'] for m in all_metrics])
    avg_time = np.mean(inference_times)
    
    print(f"\n平均指标:")
    print(f"  Accuracy: {avg_acc*100:.2f}%")
    print(f"  mIoU: {avg_miou*100:.2f}%")
    print(f"  推理时间: {avg_time*1000:.1f} ms/sample")
    
    # 保存指标
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Samples: {num_samples}\n\n")
        f.write(f"Average Accuracy: {avg_acc*100:.2f}%\n")
        f.write(f"Average mIoU: {avg_miou*100:.2f}%\n")
        f.write(f"Average Time: {avg_time*1000:.1f} ms\n")
        
    print(f"\n✅ 推理完成！结果保存到: {output_dir}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# evaluate.py
"""
Occupancy Network 评估脚本

用法:
    python evaluate.py --checkpoint checkpoints/best.pth --data_root /path/to/dataset
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.occ_network import OccupancyNetwork
from datasets.carla_occ_dataset import build_dataloader
from utils.metrics import OccupancyMetrics, compute_distance_metrics
from configs.default_config import CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Occupancy Network')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='检查点路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='评估数据集分割')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--bev_size', type=int, default=200,
                        help='BEV 网格大小')
    parser.add_argument('--num_heights', type=int, default=16,
                        help='高度层数')
    parser.add_argument('--upsample', action='store_true',
                        help='是否上采样到完整分辨率')
    parser.add_argument('--save_results', type=str, default=None,
                        help='保存结果的 JSON 文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> OccupancyNetwork:
    """加载模型"""
    # 从检查点推断模型配置
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 创建模型（使用默认配置）
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
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate(
    model: OccupancyNetwork,
    dataloader,
    device: torch.device,
    upsample: bool = False,
) -> dict:
    """
    评估模型
    
    Returns:
        results: 包含各种指标的字典
    """
    model.eval()
    
    # 指标收集器
    metrics = OccupancyMetrics(num_classes=18, class_names=CLASS_NAMES)
    
    # 距离指标累积
    all_distance_metrics = []
    
    # 推理时间统计
    total_time = 0
    num_samples = 0
    
    for batch in tqdm(dataloader, desc='Evaluating'):
        images = batch['images'].to(device)
        occupancy = batch['occupancy'].to(device)
        mask = batch['mask'].to(device)
        
        # 推理计时
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        
        if start_time:
            start_time.record()
        
        # 前向传播
        outputs = model(images, upsample=upsample)
        occ_logits = outputs['occ_logits']
        occ_pred = occ_logits.argmax(dim=1)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            total_time += start_time.elapsed_time(end_time)
        
        num_samples += images.shape[0]
        
        # 更新指标
        metrics.update(occ_pred, occupancy, mask)
        
        # 距离指标
        dist_metrics = compute_distance_metrics(
            occ_pred, occupancy, mask,
            num_classes=18,
        )
        all_distance_metrics.append(dist_metrics)
    
    # 汇总结果
    results = metrics.compute()
    
    # 添加距离指标
    if all_distance_metrics:
        for key in all_distance_metrics[0].keys():
            values = [m[key] for m in all_distance_metrics if not np.isnan(m[key])]
            if values:
                results[key] = np.mean(values)
    
    # 添加推理速度
    if total_time > 0:
        results['avg_inference_time_ms'] = total_time / num_samples
        results['fps'] = 1000 * num_samples / total_time
    
    return results


def print_results(results: dict):
    """打印结果"""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # 主要指标
    print("\n📊 Main Metrics:")
    print(f"  mIoU:           {results['miou']:.4f}")
    print(f"  Mean Accuracy:  {results['mean_acc']:.4f}")
    print(f"  Overall Acc:    {results['overall_acc']:.4f}")
    
    # 距离指标
    print("\n📏 Distance-based mIoU:")
    if 'near_miou' in results:
        print(f"  Near (0-20m):   {results['near_miou']:.4f}")
    if 'mid_miou' in results:
        print(f"  Mid (20-50m):   {results['mid_miou']:.4f}")
    if 'far_miou' in results:
        print(f"  Far (50-100m):  {results['far_miou']:.4f}")
    
    # 速度
    print("\n⚡ Inference Speed:")
    if 'avg_inference_time_ms' in results:
        print(f"  Avg Time:       {results['avg_inference_time_ms']:.2f} ms")
        print(f"  FPS:            {results['fps']:.1f}")
    
    # 每类 IoU
    print("\n📋 Per-class IoU:")
    for i, name in enumerate(CLASS_NAMES):
        key = f'iou_{name}'
        if key in results:
            iou = results[key]
            if not np.isnan(iou):
                bar = '█' * int(iou * 20) + '░' * (20 - int(iou * 20))
                print(f"  {name:25s} {bar} {iou:.4f}")
    
    print("=" * 60)


def main():
    args = parse_args()
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, device)
    
    # 数据加载
    print(f'Loading {args.split} dataset...')
    dataloader = build_dataloader(
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(384, 640),
        grid_size=(args.bev_size, args.bev_size, args.num_heights),
        augment=False,
    )
    print(f'Samples: {len(dataloader.dataset)}')
    
    # 评估
    print('Evaluating...')
    results = evaluate(model, dataloader, device, upsample=args.upsample)
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    if args.save_results:
        # 转换 numpy 类型为 Python 原生类型
        results_json = {}
        for k, v in results.items():
            if isinstance(v, (np.floating, np.integer)):
                results_json[k] = float(v)
            elif isinstance(v, np.ndarray):
                results_json[k] = v.tolist()
            else:
                results_json[k] = v
        
        with open(args.save_results, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f'\nResults saved to {args.save_results}')


if __name__ == '__main__':
    main()

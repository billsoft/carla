# utils/metrics.py
"""
评估指标

包含:
- IoU (Intersection over Union)
- mIoU (Mean IoU)
- Per-class Accuracy
- Distance-aware Metrics
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int = 18,
) -> Tuple[np.ndarray, float]:
    """
    计算每类 IoU 和 mIoU
    
    Args:
        pred: [B, H, W, Z] 预测类别 (argmax 后)
        target: [B, H, W, Z] 真实类别
        mask: [B, H, W, Z] 可见性掩码
        num_classes: 类别数
        
    Returns:
        iou_per_class: [num_classes] 每类 IoU (NaN 表示该类不存在)
        miou: 平均 IoU
    """
    ious = []
    
    for cls in range(num_classes):
        # 仅在可见区域计算
        pred_cls = (pred == cls) & mask
        target_cls = (target == cls) & mask
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = float('nan')
        else:
            iou = (intersection / union).item()
            
        ious.append(iou)
    
    # 计算 mIoU (忽略 NaN)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    
    return np.array(ious), miou


def compute_miou(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int = 18,
    ignore_classes: Optional[List[int]] = None,
) -> float:
    """
    计算 mIoU
    
    Args:
        pred: 预测类别
        target: 真实类别
        mask: 可见性掩码
        num_classes: 类别数
        ignore_classes: 忽略的类别列表
        
    Returns:
        miou: 平均 IoU
    """
    if ignore_classes is None:
        ignore_classes = [0]  # 默认忽略 free 类
        
    ious = []
    
    for cls in range(num_classes):
        if cls in ignore_classes:
            continue
            
        pred_cls = (pred == cls) & mask
        target_cls = (target == cls) & mask
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            iou = (intersection / union).item()
            ious.append(iou)
    
    return np.mean(ious) if ious else 0.0


def compute_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int = 18,
) -> Tuple[np.ndarray, float]:
    """
    计算每类准确率和总体准确率
    
    Args:
        pred: 预测类别
        target: 真实类别
        mask: 可见性掩码
        num_classes: 类别数
        
    Returns:
        acc_per_class: [num_classes] 每类准确率
        overall_acc: 总体准确率
    """
    acc_per_class = []
    
    for cls in range(num_classes):
        target_cls_mask = (target == cls) & mask
        
        if target_cls_mask.sum() == 0:
            acc_per_class.append(float('nan'))
        else:
            correct = (pred[target_cls_mask] == cls).sum().float()
            total = target_cls_mask.sum().float()
            acc = (correct / total).item()
            acc_per_class.append(acc)
    
    # 总体准确率
    valid_mask = mask
    if valid_mask.sum() > 0:
        correct = ((pred == target) & valid_mask).sum().float()
        total = valid_mask.sum().float()
        overall_acc = (correct / total).item()
    else:
        overall_acc = 0.0
    
    return np.array(acc_per_class), overall_acc


def compute_distance_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    x_range: Tuple[float, float] = (-50.0, 50.0),
    y_range: Tuple[float, float] = (-50.0, 50.0),
    num_classes: int = 18,
    distance_bins: List[float] = [0, 20, 50, 100],
) -> Dict[str, float]:
    """
    按距离分层计算 mIoU
    
    Args:
        pred: [B, H, W, Z] 预测
        target: [B, H, W, Z] 目标
        mask: [B, H, W, Z] 掩码
        x_range: X 范围
        y_range: Y 范围
        num_classes: 类别数
        distance_bins: 距离分段
        
    Returns:
        metrics: {'near_miou': ..., 'mid_miou': ..., 'far_miou': ...}
    """
    device = pred.device
    B, H, W, Z = pred.shape
    
    # 创建距离图
    x = torch.linspace(x_range[0], x_range[1], H, device=device)
    y = torch.linspace(y_range[0], y_range[1], W, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    dist = torch.sqrt(xx ** 2 + yy ** 2)  # [H, W]
    dist = dist[None, :, :, None].expand(B, -1, -1, Z)  # [B, H, W, Z]
    
    metrics = {}
    bin_names = ['near', 'mid', 'far']
    
    for i in range(len(distance_bins) - 1):
        d_min, d_max = distance_bins[i], distance_bins[i + 1]
        bin_mask = (dist >= d_min) & (dist < d_max) & mask
        
        if bin_mask.sum() > 0:
            miou = compute_miou(pred, target, bin_mask, num_classes)
        else:
            miou = float('nan')
            
        name = bin_names[i] if i < len(bin_names) else f'dist_{d_min}_{d_max}'
        metrics[f'{name}_miou'] = miou
    
    return metrics


class OccupancyMetrics:
    """
    Occupancy 评估指标收集器
    
    用于累积多个 batch 的预测和目标，最终计算整体指标
    """
    
    def __init__(
        self,
        num_classes: int = 18,
        class_names: Optional[List[str]] = None,
        ignore_classes: Optional[List[int]] = None,
    ):
        self.num_classes = num_classes
        self.ignore_classes = ignore_classes or [0]
        
        if class_names is None:
            self.class_names = [
                'free', 'barrier', 'bicycle', 'bus', 'car',
                'construction_vehicle', 'motorcycle', 'pedestrian',
                'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                'other_flat', 'sidewalk', 'terrain', 'manmade',
                'vegetation', 'general_object'
            ]
        else:
            self.class_names = class_names
            
        self.reset()
        
    def reset(self):
        """重置累积器"""
        # 每类的交集和并集
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        
        # 每类的正确数和总数
        self.correct = np.zeros(self.num_classes)
        self.total = np.zeros(self.num_classes)
        
        # 总体统计
        self.total_correct = 0
        self.total_count = 0
        
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        更新统计量
        
        Args:
            pred: [B, H, W, Z] 预测类别
            target: [B, H, W, Z] 真实类别
            mask: [B, H, W, Z] 可见性掩码
        """
        # 转为 numpy
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        mask = mask.cpu().numpy()
        
        for cls in range(self.num_classes):
            pred_cls = (pred == cls) & mask
            target_cls = (target == cls) & mask
            
            self.intersection[cls] += (pred_cls & target_cls).sum()
            self.union[cls] += (pred_cls | target_cls).sum()
            
            self.correct[cls] += (pred[target_cls] == cls).sum()
            self.total[cls] += target_cls.sum()
        
        # 总体准确率
        valid = mask
        self.total_correct += ((pred == target) & valid).sum()
        self.total_count += valid.sum()
        
    def compute(self) -> Dict[str, float]:
        """
        计算最终指标
        
        Returns:
            metrics: 包含各类 IoU、mIoU、准确率的字典
        """
        metrics = {}
        
        # 每类 IoU
        ious = []
        for cls in range(self.num_classes):
            if self.union[cls] > 0:
                iou = self.intersection[cls] / self.union[cls]
            else:
                iou = float('nan')
                
            metrics[f'iou_{self.class_names[cls]}'] = iou
            
            if cls not in self.ignore_classes and not np.isnan(iou):
                ious.append(iou)
        
        # mIoU
        metrics['miou'] = np.mean(ious) if ious else 0.0
        
        # 每类准确率
        accs = []
        for cls in range(self.num_classes):
            if self.total[cls] > 0:
                acc = self.correct[cls] / self.total[cls]
            else:
                acc = float('nan')
                
            metrics[f'acc_{self.class_names[cls]}'] = acc
            
            if cls not in self.ignore_classes and not np.isnan(acc):
                accs.append(acc)
        
        # 平均准确率
        metrics['mean_acc'] = np.mean(accs) if accs else 0.0
        
        # 总体准确率
        if self.total_count > 0:
            metrics['overall_acc'] = self.total_correct / self.total_count
        else:
            metrics['overall_acc'] = 0.0
        
        return metrics
    
    def get_table(self) -> str:
        """
        生成可打印的指标表格
        """
        metrics = self.compute()
        
        lines = []
        lines.append("=" * 50)
        lines.append("Occupancy Prediction Metrics")
        lines.append("=" * 50)
        
        # 每类 IoU
        lines.append("\nPer-class IoU:")
        for cls in range(self.num_classes):
            iou = metrics.get(f'iou_{self.class_names[cls]}', float('nan'))
            if not np.isnan(iou):
                lines.append(f"  {self.class_names[cls]:25s}: {iou:.4f}")
                
        # 汇总
        lines.append("\nSummary:")
        lines.append(f"  mIoU:        {metrics['miou']:.4f}")
        lines.append(f"  Mean Acc:    {metrics['mean_acc']:.4f}")
        lines.append(f"  Overall Acc: {metrics['overall_acc']:.4f}")
        lines.append("=" * 50)
        
        return "\n".join(lines)


# 测试代码
if __name__ == '__main__':
    print("Testing Metrics...")
    
    device = torch.device('cpu')
    
    # 模拟数据
    B, H, W, Z = 2, 50, 50, 8
    num_classes = 18
    
    pred = torch.randint(0, num_classes, (B, H, W, Z))
    target = torch.randint(0, num_classes, (B, H, W, Z))
    mask = torch.rand(B, H, W, Z) > 0.3
    
    # 让一部分预测正确
    pred[mask & (torch.rand(B, H, W, Z) > 0.5)] = target[mask & (torch.rand(B, H, W, Z) > 0.5)]
    
    # 1. 测试 compute_iou
    print("\n1. Testing compute_iou...")
    iou_per_class, miou = compute_iou(pred, target, mask, num_classes)
    print(f"   mIoU: {miou:.4f}")
    print(f"   Valid IoUs: {sum(~np.isnan(iou_per_class))}/{num_classes}")
    
    # 2. 测试 compute_accuracy
    print("\n2. Testing compute_accuracy...")
    acc_per_class, overall_acc = compute_accuracy(pred, target, mask, num_classes)
    print(f"   Overall Accuracy: {overall_acc:.4f}")
    
    # 3. 测试 distance metrics
    print("\n3. Testing distance metrics...")
    dist_metrics = compute_distance_metrics(pred, target, mask)
    for k, v in dist_metrics.items():
        print(f"   {k}: {v:.4f}" if not np.isnan(v) else f"   {k}: N/A")
    
    # 4. 测试 OccupancyMetrics
    print("\n4. Testing OccupancyMetrics...")
    metrics = OccupancyMetrics(num_classes=num_classes)
    
    # 模拟多个 batch
    for i in range(5):
        pred_i = torch.randint(0, num_classes, (B, H, W, Z))
        target_i = torch.randint(0, num_classes, (B, H, W, Z))
        mask_i = torch.rand(B, H, W, Z) > 0.3
        
        metrics.update(pred_i, target_i, mask_i)
    
    print(metrics.get_table())
    
    print("\n✓ All tests passed!")

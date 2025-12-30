# losses/occ_loss.py
"""
Occupancy 预测损失函数

包含:
1. Masked Weighted Cross-Entropy: 处理类别不平衡和遮挡区域
2. Lovász-Softmax Loss: 直接优化 IoU
3. Focal Loss: 聚焦难分类样本
4. Combined Loss: 组合多种损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class MaskedWeightedCELoss(nn.Module):
    """
    带掩码的加权交叉熵损失
    
    特点:
    1. 忽略不可见区域 (mask=False)
    2. 使用类别权重处理不平衡
    """
    
    def __init__(
        self,
        num_classes: int = 18,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # 默认类别权重
        if class_weights is None:
            class_weights = [
                0.5,   # 0: free
                2.0,   # 1: barrier
                5.0,   # 2: bicycle
                3.0,   # 3: bus
                1.0,   # 4: car
                10.0,  # 5: construction_vehicle
                5.0,   # 6: motorcycle
                3.0,   # 7: pedestrian
                8.0,   # 8: traffic_cone
                10.0,  # 9: trailer
                2.0,   # 10: truck
                0.8,   # 11: driveable_surface
                1.5,   # 12: other_flat
                1.5,   # 13: sidewalk
                1.0,   # 14: terrain
                1.0,   # 15: manmade
                1.0,   # 16: vegetation
                2.0,   # 17: general_object
            ]
            
        self.register_buffer(
            'class_weights',
            torch.tensor(class_weights, dtype=torch.float32)
        )
        
    def forward(
        self,
        pred: torch.Tensor,    # [B, C, H, W, Z]
        target: torch.Tensor,  # [B, H, W, Z]
        mask: torch.Tensor,    # [B, H, W, Z]
    ) -> torch.Tensor:
        """
        计算损失
        
        Args:
            pred: 预测 logits [B, num_classes, H, W, Z]
            target: 目标类别 [B, H, W, Z]
            mask: 可见性掩码 [B, H, W, Z], True=可见
            
        Returns:
            loss: 标量损失
        """
        B, C, H, W, Z = pred.shape
        
        # 将不可见区域的 target 设为 ignore_index
        target_masked = target.clone()
        target_masked[~mask] = self.ignore_index
        
        # 调整维度: [B, C, H, W, Z] -> [B, C, H*W*Z] -> [B*H*W*Z, C]
        pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
        target_flat = target_masked.reshape(-1)
        
        # 计算加权交叉熵
        loss = F.cross_entropy(
            pred_flat,
            target_flat,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='mean',
        )
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    
    对难分类样本给予更高权重
    FL(p) = -α(1-p)^γ log(p)
    """
    
    def __init__(
        self,
        num_classes: int = 18,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算 Focal Loss"""
        B, C, H, W, Z = pred.shape
        
        # 处理 mask
        target_masked = target.clone()
        target_masked[~mask] = self.ignore_index
        
        # 计算 softmax 概率
        pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
        target_flat = target_masked.reshape(-1)
        
        # 过滤有效样本
        valid_mask = target_flat != self.ignore_index
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        if pred_valid.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Softmax
        p = F.softmax(pred_valid, dim=1)
        
        # 获取目标类别的概率
        ce_loss = F.cross_entropy(pred_valid, target_valid, reduction='none')
        p_t = p.gather(1, target_valid.unsqueeze(1)).squeeze(1)
        
        # Focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Focal loss
        focal_loss = (focal_weight * ce_loss).mean()
        
        return focal_loss


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    计算 Lovász 扩展的梯度
    """
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1 - intersection / union
    
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
        
    return jaccard


def lovasz_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str = 'present',
) -> torch.Tensor:
    """
    Lovász-Softmax loss (展平版本)
    
    Args:
        probas: [P, C] 预测概率
        labels: [P] 真实标签
        classes: 'present' 只计算数据中存在的类
    """
    C = probas.size(1)
    losses = []
    
    # 确定要计算的类别
    if classes == 'present':
        class_to_sum = torch.unique(labels)
    else:
        class_to_sum = torch.arange(C, device=labels.device)
        
    for c in class_to_sum:
        if c == 0:  # 忽略 background/free
            continue
            
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
            
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        
        grad = lovasz_grad(fg_sorted)
        loss = torch.dot(errors_sorted, grad)
        losses.append(loss)
        
    if len(losses) == 0:
        return torch.tensor(0.0, device=probas.device)
        
    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax Loss
    
    直接优化 IoU 指标
    """
    
    def __init__(
        self,
        classes: str = 'present',
        ignore_index: int = -100,
    ):
        super().__init__()
        
        self.classes = classes
        self.ignore_index = ignore_index
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算 Lovász-Softmax Loss"""
        B, C, H, W, Z = pred.shape
        
        # 计算 softmax
        probas = F.softmax(pred, dim=1)
        
        # 展平
        probas_flat = probas.permute(0, 2, 3, 4, 1).reshape(-1, C)
        target_flat = target.reshape(-1)
        mask_flat = mask.reshape(-1)
        
        # 过滤可见区域
        probas_valid = probas_flat[mask_flat]
        target_valid = target_flat[mask_flat]
        
        if probas_valid.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
            
        loss = lovasz_softmax_flat(probas_valid, target_valid, self.classes)
        
        return loss


class CombinedOccLoss(nn.Module):
    """
    组合损失函数
    
    CE Loss + Lovász Loss
    """
    
    def __init__(
        self,
        num_classes: int = 18,
        ce_weight: float = 0.7,
        lovasz_weight: float = 0.3,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        
        self.ce_loss = MaskedWeightedCELoss(
            num_classes=num_classes,
            class_weights=class_weights,
        )
        
        self.lovasz_loss = LovaszSoftmaxLoss()
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算组合损失
        
        Returns:
            loss: 总损失
            loss_dict: 各分项损失
        """
        # CE Loss
        ce = self.ce_loss(pred, target, mask)
        
        # Lovász Loss
        lovasz = self.lovasz_loss(pred, target, mask)
        
        # 总损失
        loss = self.ce_weight * ce + self.lovasz_weight * lovasz
        
        loss_dict = {
            'ce_loss': ce.item(),
            'lovasz_loss': lovasz.item(),
            'total_loss': loss.item(),
        }
        
        return loss, loss_dict


class GeometricAwareLoss(nn.Module):
    """
    几何感知损失
    
    对不同距离的区域使用不同权重
    近处精度更重要
    """
    
    def __init__(
        self,
        num_classes: int = 18,
        x_range: Tuple[float, float] = (-50.0, 50.0),
        y_range: Tuple[float, float] = (-50.0, 50.0),
        near_threshold: float = 20.0,
        far_threshold: float = 50.0,
        near_weight: float = 2.0,
        far_weight: float = 0.5,
    ):
        super().__init__()
        
        self.x_range = x_range
        self.y_range = y_range
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.near_weight = near_weight
        self.far_weight = far_weight
        
        self.base_loss = MaskedWeightedCELoss(num_classes)
        
    def _get_distance_weights(
        self,
        shape: Tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """计算距离权重"""
        H, W, Z = shape
        
        # 创建坐标网格
        x = torch.linspace(self.x_range[0], self.x_range[1], H, device=device)
        y = torch.linspace(self.y_range[0], self.y_range[1], W, device=device)
        
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        dist = torch.sqrt(xx ** 2 + yy ** 2)  # [H, W]
        
        # 计算权重
        weights = torch.ones_like(dist)
        weights[dist <= self.near_threshold] = self.near_weight
        weights[dist >= self.far_threshold] = self.far_weight
        
        # 扩展到 Z 维度
        weights = weights.unsqueeze(-1).expand(-1, -1, Z)  # [H, W, Z]
        
        return weights
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算几何感知损失"""
        B, C, H, W, Z = pred.shape
        
        # 获取距离权重
        dist_weights = self._get_distance_weights((H, W, Z), pred.device)
        dist_weights = dist_weights.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, Z]
        
        # 调整 mask
        weighted_mask = mask.float() * dist_weights
        
        # 计算加权损失
        target_masked = target.clone()
        target_masked[~mask] = -100
        
        pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
        target_flat = target_masked.reshape(-1)
        weight_flat = weighted_mask.reshape(-1)
        
        # 计算每个样本的 CE loss
        loss = F.cross_entropy(pred_flat, target_flat, reduction='none', ignore_index=-100)
        
        # 应用距离权重
        valid_mask = target_flat != -100
        if valid_mask.any():
            weighted_loss = (loss[valid_mask] * weight_flat[valid_mask]).mean()
        else:
            weighted_loss = torch.tensor(0.0, device=pred.device)
            
        return weighted_loss


def build_loss(config) -> nn.Module:
    """
    根据配置构建损失函数
    """
    return CombinedOccLoss(
        num_classes=config.occupancy.num_classes,
        ce_weight=config.loss.ce_weight,
        lovasz_weight=config.loss.lovasz_weight,
        class_weights=config.loss.class_weights,
    )


# losses/__init__.py 内容
__all__ = [
    'MaskedWeightedCELoss',
    'FocalLoss',
    'LovaszSoftmaxLoss', 
    'CombinedOccLoss',
    'GeometricAwareLoss',
    'build_loss',
]


# 测试代码
if __name__ == '__main__':
    print("Testing Loss Functions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 模拟数据
    B, C, H, W, Z = 2, 18, 50, 50, 8
    
    pred = torch.randn(B, C, H, W, Z).to(device)
    target = torch.randint(0, C, (B, H, W, Z)).to(device)
    mask = torch.rand(B, H, W, Z).to(device) > 0.3
    
    # 1. 测试 CE Loss
    print("\n1. Testing Masked Weighted CE Loss...")
    ce_loss = MaskedWeightedCELoss(num_classes=C).to(device)
    loss_ce = ce_loss(pred, target, mask)
    print(f"   CE Loss: {loss_ce.item():.4f}")
    
    # 2. 测试 Focal Loss
    print("\n2. Testing Focal Loss...")
    focal_loss = FocalLoss(num_classes=C).to(device)
    loss_focal = focal_loss(pred, target, mask)
    print(f"   Focal Loss: {loss_focal.item():.4f}")
    
    # 3. 测试 Lovász Loss
    print("\n3. Testing Lovász-Softmax Loss...")
    lovasz_loss = LovaszSoftmaxLoss().to(device)
    loss_lovasz = lovasz_loss(pred, target, mask)
    print(f"   Lovász Loss: {loss_lovasz.item():.4f}")
    
    # 4. 测试组合损失
    print("\n4. Testing Combined Loss...")
    combined_loss = CombinedOccLoss(num_classes=C).to(device)
    loss_total, loss_dict = combined_loss(pred, target, mask)
    print(f"   Total Loss: {loss_total.item():.4f}")
    print(f"   Loss breakdown: {loss_dict}")
    
    # 5. 测试几何感知损失
    print("\n5. Testing Geometric-Aware Loss...")
    geo_loss = GeometricAwareLoss(num_classes=C).to(device)
    loss_geo = geo_loss(pred, target, mask)
    print(f"   Geometric Loss: {loss_geo.item():.4f}")
    
    # 测试反向传播
    print("\n6. Testing backward pass...")
    pred.requires_grad = True
    loss_total, _ = combined_loss(pred, target, mask)
    loss_total.backward()
    print(f"   Gradient shape: {pred.grad.shape}")
    print(f"   Gradient mean: {pred.grad.abs().mean().item():.6f}")
    
    print("\n✓ All tests passed!")

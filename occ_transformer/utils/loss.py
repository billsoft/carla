# utils/loss.py
"""
损失函数模块

支持:
- MaskedWeightedCELoss: 带 mask 的加权交叉熵损失
- 类别权重配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class MaskedWeightedCELoss(nn.Module):
    """
    带 Mask 的加权交叉熵损失
    
    Args:
        class_weights: 类别权重列表
        ignore_index: 忽略的类别索引
    """
    
    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = 255
    ):
        super().__init__()
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights).float())
        else:
            self.class_weights = None
            
        self.ignore_index = ignore_index
        
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, C, X, Y, Z] 预测 logits
            target: [B, X, Y, Z] 目标标签
            mask: [B, X, Y, Z] 有效区域 mask
            
        Returns:
            loss: 标量损失
        """
        # 应用 mask
        if mask is not None:
            if mask.shape != target.shape:
                raise ValueError(f"Mask shape {mask.shape} != Target shape {target.shape}")
            
            target = target.clone()
            if mask.dtype == torch.bool:
                target[~mask] = self.ignore_index
            else:
                target[mask < 0.5] = self.ignore_index
        
        # 计算交叉熵损失
        loss = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    
    用于处理类别不平衡问题
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = 255
    ):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights).float())
        else:
            self.class_weights = None
            
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, C, X, Y, Z] 预测 logits
            target: [B, X, Y, Z] 目标标签
            mask: [B, X, Y, Z] 有效区域 mask
            
        Returns:
            loss: 标量损失
        """
        # 应用 mask
        if mask is not None:
            target = target.clone()
            if mask.dtype == torch.bool:
                target[~mask] = self.ignore_index
            else:
                target[mask < 0.5] = self.ignore_index
        
        # 计算 CE loss
        ce_loss = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # 计算 p_t
        p = F.softmax(logits, dim=1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # 忽略 ignore_index
        valid_mask = target != self.ignore_index
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # 只计算有效区域的平均
        if valid_mask.sum() > 0:
            loss = focal_loss[valid_mask].mean()
        else:
            loss = focal_loss.mean()
            
        return loss


class LovaszLoss(nn.Module):
    """
    Lovasz-Softmax Loss
    
    直接优化 IoU
    """
    
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """简化版实现，实际使用时建议用完整的 Lovasz loss"""
        # 应用 mask
        if mask is not None:
            target = target.clone()
            if mask.dtype == torch.bool:
                target[~mask] = self.ignore_index
            else:
                target[mask < 0.5] = self.ignore_index
        
        # 使用 CE loss 作为近似
        loss = F.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        
        return loss


def get_default_class_weights() -> List[float]:
    """
    获取默认的类别权重
    
    根据 CARLA 数据集的类别分布设置
    稀有类别（行人、自行车等）权重更高
    
    类别说明 (18 类):
    0: empty - 空
    1: barrier - 护栏
    2: bicycle - 自行车 (稀有)
    3: bus - 公交车
    4: car - 汽车
    5: construction - 施工区
    6: motorcycle - 摩托车 (稀有)
    7: pedestrian - 行人 (稀有, 安全关键)
    8: traffic_cone - 锥桶
    9: trailer - 拖车
    10: truck - 卡车
    11: drivable - 可行驶区域
    12: other - 其他
    13: sidewalk - 人行道
    14: terrain - 地形
    15: manmade - 人造物
    16: vegetation - 植被
    17: sky - 天空
    """
    weights = [
        0.1,   # 0: empty - 显著降低空类别权重 (背景占绝大多数)
        2.0,   # 1: barrier
        5.0,   # 2: bicycle - 稀有
        2.0,   # 3: bus
        2.0,   # 4: car
        3.0,   # 5: construction
        5.0,   # 6: motorcycle - 稀有
        5.0,   # 7: pedestrian - 安全关键
        3.0,   # 8: traffic_cone
        2.0,   # 9: trailer
        2.0,   # 10: truck
        1.5,   # 11: drivable - 提高路面权重，保证基础结构清晰
        1.0,   # 12: other
        1.5,   # 13: sidewalk
        1.0,   # 14: terrain
        1.0,   # 15: manmade
        1.0,   # 16: vegetation
        0.1,   # 17: sky - 降低天空权重
    ]
    return weights


def get_class_names() -> List[str]:
    """获取类别名称"""
    return [
        'empty',        # 0
        'barrier',      # 1
        'bicycle',      # 2
        'bus',          # 3
        'car',          # 4
        'construction', # 5
        'motorcycle',   # 6
        'pedestrian',   # 7
        'traffic_cone', # 8
        'trailer',      # 9
        'truck',        # 10
        'drivable',     # 11
        'other',        # 12
        'sidewalk',     # 13
        'terrain',      # 14
        'manmade',      # 15
        'vegetation',   # 16
        'sky',          # 17
    ]


if __name__ == '__main__':
    print("=" * 60)
    print("损失函数测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试 MaskedWeightedCELoss
    print("\n[1] MaskedWeightedCELoss:")
    criterion = MaskedWeightedCELoss(class_weights=get_default_class_weights()).to(device)
    
    logits = torch.randn(2, 18, 50, 50, 8, device=device)
    target = torch.randint(0, 18, (2, 50, 50, 8), device=device)
    mask = torch.ones(2, 50, 50, 8, dtype=torch.bool, device=device)
    
    loss = criterion(logits, target, mask)
    print(f"  Logits: {logits.shape}")
    print(f"  Target: {target.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # 测试 FocalLoss
    print("\n[2] FocalLoss:")
    focal = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    loss = focal(logits, target, mask)
    print(f"  Loss: {loss.item():.4f}")
    
    # 类别权重
    print("\n[3] 类别权重:")
    weights = get_default_class_weights()
    names = get_class_names()
    for i, (name, w) in enumerate(zip(names, weights)):
        print(f"  {i:2d}. {name:<15} {w:.1f}")
    
    print("\n" + "=" * 60)
    print("✅ 测试通过！")
    print("=" * 60)

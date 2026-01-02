"""
损失函数模块

支持:
- MaskedWeightedCELoss: 带 mask 的加权交叉熵损失
- FocalLoss: 聚焦难分样本的损失
- LovaszSoftmaxLoss: 直接优化 IoU 的损失 (IoU Surrogate)
- CombinedLoss: 组合损失 (如 Focal + Lovasz)
- 类别权重配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovasz-Softmax Loss
    直接优化 IoU (Intersection over Union)
    
    Reference: https://github.com/bermanmaxim/LovaszSoftmax
    """
    
    def __init__(self, ignore_index: int = 255, classes='present', per_image=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.classes = classes
        self.per_image = per_image
        
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, C, X, Y, Z] 预测 logits (未经过 softmax)
            target: [B, X, Y, Z] 目标标签
            mask: [B, X, Y, Z] 有效区域 mask
        """
        # 应用 mask
        if mask is not None:
            # 如果 mask 是 bool，直接选出有效区域进行计算
            # Lovasz loss 需要 flatten
            if mask.dtype != torch.bool:
                mask = mask > 0.5
            
            # 展平并只取有效像素
            # logits: [B, C, XYZ] -> permute -> [B, XYZ, C] -> flatten -> [N, C]
            # target: [B, XYZ] -> flatten -> [N]
            
            # 为了简单起见，这里先不处理 per_image 的情况，统一 flatten
            # 注意：Lovasz Softmax 需要 probas (softmax后的概率)
            probas = F.softmax(logits, dim=1)
            
            # [B, C, D1, D2, D3] -> [B, D1, D2, D3, C]
            probas = probas.permute(0, 2, 3, 4, 1)
            
            # 选取 mask 为 True 的区域
            valid_probas = probas[mask] # [N, C]
            valid_target = target[mask] # [N]
            
            # 过滤掉 ignore_index (如果有额外的 ignore_index，虽然 mask 应该已经处理了)
            if self.ignore_index is not None:
                valid = valid_target != self.ignore_index
                valid_probas = valid_probas[valid]
                valid_target = valid_target[valid]
                
            loss = lovasz_softmax_flat(valid_probas, valid_target, classes=self.classes)
            return loss
            
        else:
            # 没有 mask 的情况
            probas = F.softmax(logits, dim=1)
            probas = probas.permute(0, 2, 3, 4, 1).contiguous().view(-1, probas.size(1))
            target = target.view(-1)
            
            if self.ignore_index is not None:
                valid = target != self.ignore_index
                probas = probas[valid]
                target = target[valid]
                
            loss = lovasz_softmax_flat(probas, target, classes=self.classes)
            return loss


class MaskedWeightedCELoss(nn.Module):
    """
    带 Mask 的加权交叉熵损失
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
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25, # 这里的 alpha 可以是标量，也可以不用，因为我们有 class_weights
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
        
        # 1. 准备 Target
        if mask is not None:
            target_masked = target.clone()
            if mask.dtype == torch.bool:
                target_masked[~mask] = self.ignore_index
            else:
                target_masked[mask < 0.5] = self.ignore_index
        else:
            target_masked = target

        # 2. 计算 CE Loss (不进行 reduction，保留每个像素的 loss)
        # Logits: [B, C, ...] -> [B, C, N]
        # Target: [B, ...] -> [B, N]
        
        # 使用 PyTorch 的 cross_entropy 计算 -log(p_t)
        # 如果有 class_weights，这里会应用 weights * -log(p_t)
        ce_loss = F.cross_entropy(
            logits,
            target_masked,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # 3. 计算 p_t (预测概率)
        # 为了计算 (1-p_t)^gamma，我们需要知道正确类别的概率 p_t
        log_p = F.log_softmax(logits, dim=1) # [B, C, X, Y, Z]
        p = torch.exp(log_p)
        
        # Gather p_t: 选取 target 对应类别的概率
        # target 需要 unsqueeze 维度以匹配 gather
        # target_masked 可能包含 ignore_index，需要处理
        
        # 创建一个临时的 target，把 ignore_index 替换为 0 (或其他有效值)，避免 gather 越界
        # 之后会通过 mask 过滤掉这些位置
        target_safe = target_masked.clone()
        valid_mask = target_safe != self.ignore_index
        target_safe[~valid_mask] = 0 
        
        p_t = p.gather(1, target_safe.unsqueeze(1)).squeeze(1) # [B, X, Y, Z]
        
        # 4. 计算 Focal Term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # 5. 组合
        # loss = focal_term * ce_loss
        # 注意：如果 class_weights 已经被 cross_entropy 应用了，这里不需要再乘 alpha
        # 除非 alpha 是用来平衡正负样本的额外系数。
        # 在多分类中，通常 class_weights 充当了 alpha 的角色。
        loss = focal_term * ce_loss
        
        # 6. Reduction
        if valid_mask.sum() > 0:
            return loss[valid_mask].mean()
        else:
            return loss.sum() * 0.0


class CombinedLoss(nn.Module):
    """
    组合损失：Weighted CE + Lovasz-Softmax
    """
    def __init__(
        self,
        ce_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = 255
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        
        self.ce_loss = MaskedWeightedCELoss(class_weights=class_weights, ignore_index=ignore_index)
        self.lovasz_loss = LovaszSoftmaxLoss(ignore_index=ignore_index)
        
    def forward(self, logits, target, mask=None):
        loss = 0.0
        if self.ce_weight > 0:
            loss += self.ce_weight * self.ce_loss(logits, target, mask)
        if self.lovasz_weight > 0:
            loss += self.lovasz_weight * self.lovasz_loss(logits, target, mask)
        return loss


def get_default_class_weights() -> List[float]:
    """
    获取默认的类别权重
    
    调整策略：
    1. 避免 0.1 这种极低权重，防止模型完全忽略 free 类别。
    2. 保持对稀有类别（行人、自行车）的高权重。
    3. 适当提高 empty 权重，确保模型敢于预测空。
    """
    weights = [
        1.0,   # 0: empty - 恢复到 1.0 以平衡 Free/Occupied
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
        1.5,   # 11: drivable
        1.0,   # 12: other
        1.5,   # 13: sidewalk
        1.0,   # 14: terrain
        1.0,   # 15: manmade
        1.0,   # 16: vegetation
        0.5,   # 17: sky - 天空相对容易，权重稍低
    ]
    return weights

def get_class_names() -> List[str]:
    return [
        'empty', 'barrier', 'bicycle', 'bus', 'car', 'construction',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'drivable', 'other', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'sky'
    ]

if __name__ == '__main__':
    print("=" * 60)
    print("损失函数测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logits = torch.randn(2, 18, 50, 50, 8, device=device)
    target = torch.randint(0, 18, (2, 50, 50, 8), device=device)
    mask = torch.ones(2, 50, 50, 8, dtype=torch.bool, device=device)
    
    print("\n[1] MaskedWeightedCELoss:")
    criterion = MaskedWeightedCELoss(class_weights=get_default_class_weights()).to(device)
    loss = criterion(logits, target, mask)
    print(f"  Loss: {loss.item():.4f}")
    
    print("\n[2] FocalLoss:")
    focal = FocalLoss(gamma=2.0, class_weights=get_default_class_weights()).to(device)
    loss = focal(logits, target, mask)
    print(f"  Loss: {loss.item():.4f}")

    print("\n[3] LovaszSoftmaxLoss:")
    lovasz = LovaszSoftmaxLoss().to(device)
    loss = lovasz(logits, target, mask)
    print(f"  Loss: {loss.item():.4f}")
    
    print("\n[4] CombinedLoss:")
    combined = CombinedLoss(class_weights=get_default_class_weights()).to(device)
    loss = combined(logits, target, mask)
    print(f"  Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 测试通过！")

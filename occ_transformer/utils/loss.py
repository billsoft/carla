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


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining Loss
    
    只对 loss 最大的前 k% 样本进行反向传播
    """
    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = 255,
        thresh: float = 0.7,
        min_kept: int = 100000
    ):
        super().__init__()
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights).float())
        else:
            self.class_weights = None
            
        self.ignore_index = ignore_index
        self.thresh = thresh
        self.min_kept = min_kept
        
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # 1. 计算所有像素的 CE Loss (reduction='none')
        # Logits: [B, C, XYZ]
        # Target: [B, XYZ]
        
        if mask is not None:
            target_masked = target.clone()
            if mask.dtype == torch.bool:
                target_masked[~mask] = self.ignore_index
            else:
                target_masked[mask < 0.5] = self.ignore_index
        else:
            target_masked = target
            
        loss = F.cross_entropy(
            logits,
            target_masked,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # 2. OHEM 筛选
        # 展平 loss
        loss = loss.view(-1)
        
        # 移除 ignore_index 的 loss (已经是 0 或无效值，但为了安全重新过滤)
        # cross_entropy ignore_index 位置 loss 为 0，不影响排序，但最好只在有效像素中选
        if self.ignore_index is not None:
            valid_mask = target_masked.view(-1) != self.ignore_index
            loss = loss[valid_mask]
            
        if loss.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        # 排序
        num_kept = int(loss.numel() * self.thresh)
        num_kept = max(num_kept, self.min_kept)
        num_kept = min(num_kept, loss.numel())
        
        top_k_loss, _ = loss.topk(num_kept)
        
        return top_k_loss.mean()


class DistanceAwareLoss(nn.Module):
    """
    距离感知损失
    
    近距离 (15m内): x2.0 (Was x3.0)
    中距离 (30m内): x1.2 (Was x1.5)
    远距离: x1.0
    """
    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = 255,
        bev_size: tuple = (100, 100),
        voxel_size: tuple = (0.5, 0.5) # 假设 50m / 100 = 0.5m
    ):
        super().__init__()
        self.base_loss = MaskedWeightedCELoss(class_weights, ignore_index)
        self.bev_size = bev_size
        self.voxel_size = voxel_size
        
        # 预计算距离权重矩阵
        # self.register_buffer('dist_weights', self._make_dist_weights())
        self.dist_weights = None
        
    def _make_dist_weights(self):
        # 注意: 即使 self.bev_size 可能是 (100, 100)，但在 forward 中，
        # logits 的空间维度可能是 (50, 50) 或 (100, 100) 取决于模型输出。
        # 因此，这里我们只作为初始参考，或者在 forward 中动态调整。
        # 为了健壮性，我们将在 forward 中动态生成或插值。
        pass
        
    def forward(self, logits, target, mask=None):
        # 1. 计算逐像素 loss
        loss_map = F.cross_entropy(
            logits,
            target,
            weight=self.base_loss.class_weights,
            ignore_index=self.base_loss.ignore_index,
            reduction='none'
        )
        
        # 2. 动态生成距离权重 (匹配当前 logits 尺寸)
        # loss_map: [B, H, W, Z] 或 [B, H, W]
        # 获取空间维度 H, W
        if loss_map.dim() == 4:
            H, W = loss_map.shape[1], loss_map.shape[2]
        else:
            H, W = loss_map.shape[1], loss_map.shape[2] # 假设最后是 Z 或 logits 是 [B, H, W]
            
        # 检查是否已有缓存且尺寸匹配
        if self.dist_weights is None or self.dist_weights.shape[0] != H or self.dist_weights.shape[1] != W:
            cx, cy = H / 2, W / 2
            y, x = torch.meshgrid(torch.arange(H, device=logits.device), torch.arange(W, device=logits.device), indexing='ij')
            
            dist_sq = (x - cx)**2 + (y - cy)**2
            dist = torch.sqrt(dist_sq.float())
            
            # 距离转换: 假设总范围是 50m (不论分辨率多少)
            # dist_norm = dist / (H/2) # 归一化到 [0, 1] (边缘)
            # dist_m = dist_norm * 25.0 # 假设半径 25m
            
            # 或者简单地：假设输入 voxel_size 是基于 100x100 = 50m范围 -> 0.5m/pixel
            # 如果是 50x50 -> 1.0m/pixel
            # scale_factor = 50.0 / H
            scale_factor = 0.5 * (100.0 / H) # base 0.5m for 100px
            dist_m = dist * scale_factor
            
            weights = torch.ones_like(dist_m)
            weights[dist_m < 15] = 2.0  # Was 3.0
            weights[(dist_m >= 15) & (dist_m < 30)] = 1.2 # Was 1.5
            
            # [H, W] -> [H, W, 1]
            self.dist_weights = weights.unsqueeze(-1)
            
        # 2. 应用距离权重
        # loss_map: [B, H, W, Z]
        # dist_weights: [H, W, 1]
        
        if mask is not None:
            if mask.dtype == torch.bool:
                loss_map = loss_map * mask.float()
            else:
                loss_map = loss_map * (mask > 0.5).float()
                
        weighted_loss = loss_map * self.dist_weights
        
        # 3. Mean (只对有效区域)
        if mask is not None:
            return weighted_loss.sum() / (mask.float().sum() + 1e-6)
        else:
            return weighted_loss.mean()


class CombinedLoss(nn.Module):
    """
    组合损失：Weighted CE + Lovasz-Softmax + OHEM + Distance
    """
    def __init__(
        self,
        ce_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        ohem_weight: float = 0.0,
        distance_weight: float = 0.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = 255,
        bev_size: tuple = (100, 100) # 新增 bev_size 参数传递给 DistanceAwareLoss
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.ohem_weight = ohem_weight
        self.distance_weight = distance_weight
        
        self.ce_loss = MaskedWeightedCELoss(class_weights=class_weights, ignore_index=ignore_index)
        self.lovasz_loss = LovaszSoftmaxLoss(ignore_index=ignore_index)
        
        if ohem_weight > 0:
            self.ohem_loss = OHEMLoss(class_weights=class_weights, ignore_index=ignore_index)
            
        if distance_weight > 0:
            self.distance_loss = DistanceAwareLoss(class_weights=class_weights, ignore_index=ignore_index, bev_size=bev_size)
        
    def forward(self, logits, target, mask=None):
        loss = 0.0
        if self.ce_weight > 0:
            loss += self.ce_weight * self.ce_loss(logits, target, mask)
        if self.lovasz_weight > 0:
            loss += self.lovasz_weight * self.lovasz_loss(logits, target, mask)
        if self.ohem_weight > 0:
            loss += self.ohem_weight * self.ohem_loss(logits, target, mask)
        if self.distance_weight > 0:
            loss += self.distance_weight * self.distance_loss(logits, target, mask)
            
        return loss


def get_default_class_weights() -> List[float]:
    """
    获取默认的类别权重 (v5 - Fine-tuned)
    
    设计理念:
    1. 抑制误报 (False Positives): Free (0) 权重设为 1.0 (基准)。
    2. 关键障碍物均衡 (Safety Critical): 
       - 弱势交通参与者 (行人, 两轮车) 统一高权 (x5.0)
       - 车辆与交通设施 (车, 栏, 锥) 统一中高权 (x3.0 - x4.0)
    3. 环境背景 (Environment): 
       - 路面、人行道保持基准 (x1.0)
       - 降低 植被(16)、建筑(15)、地形(14) 权重 (x0.8)，因为树木容易遮挡且误报高。
    4. 提升未知障碍物 (General Object): 
       - x2.0，避免漏检不明物体。
    """
    weights = [
        1.0,   # 0: free - 基准，抑制噪点
        3.0,   # 1: barrier - 隔离带 (重要边界)
        5.0,   # 2: bicycle - VRU (高危)
        3.0,   # 3: bus - 大型车辆
        3.0,   # 4: car - 核心车辆
        3.0,   # 5: construction_vehicle - 异型车辆
        5.0,   # 6: motorcycle - VRU (高危)
        5.0,   # 7: pedestrian - VRU (高危)
        4.0,   # 8: traffic_cone - 小物体 (施工/警示)
        3.0,   # 9: trailer - 拖车
        3.0,   # 10: truck - 卡车
        1.0,   # 11: driveable_surface - 路面 (易检测)
        1.0,   # 12: other_flat
        1.0,   # 13: sidewalk
        0.8,   # 14: terrain - 略降 (0.8)
        0.8,   # 15: manmade - 略降 (0.8)，避免建筑墙面权重过高
        0.8,   # 16: vegetation - 略降 (0.8)，抑制树木误报
        2.0,   # 17: general_object - 提升 (2.0)，关注未知障碍
    ]
    return weights


def get_moving_class_weights() -> List[float]:
    """
    获取针对移动物体优化的类别权重 (v4 - Consistent)
    
    在 v4 策略中，我们保持与默认权重高度一致，
    仅对核心移动障碍物做极其微小的增强，避免破坏整体平衡。
    """
    weights = get_default_class_weights()
    
    # 既然用户要求"统一规划"且"不希望多出体素"，
    # 这里我们不再进行激进的加权，而是保持一致性。
    # 仅保留函数接口以便兼容，或者做微乎其微的调整。
    
    # 实际上，直接返回 default 可能是最稳健的，
    # 但为了区分 moving 模式，我们只对 VRU 再加一点点 (5.0 -> 6.0)
    
    weights[7] = 6.0  # pedestrian
    weights[2] = 6.0  # bicycle
    weights[6] = 6.0  # motorcycle
    
    return weights

def get_class_names() -> List[str]:
    """
    获取类别名称列表

    ⚠️ 注意: 与 dense_occupancy_collection/config/occupancy_config.py 中的
    OCCUPANCY_LABELS 保持一致
    """
    return [
        'free',               # 0 (was 'empty')
        'barrier',            # 1
        'bicycle',            # 2
        'bus',                # 3
        'car',                # 4
        'construction_vehicle', # 5 (was 'construction')
        'motorcycle',         # 6
        'pedestrian',         # 7
        'traffic_cone',       # 8
        'trailer',            # 9
        'truck',              # 10
        'driveable_surface',  # 11 (was 'drivable')
        'other_flat',         # 12 (was 'other')
        'sidewalk',           # 13
        'terrain',            # 14
        'manmade',            # 15
        'vegetation',         # 16
        'general_object'      # 17
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
    
    print("\n[4] CombinedLoss (Basic):")
    combined = CombinedLoss(class_weights=get_default_class_weights()).to(device)
    loss = combined(logits, target, mask)
    print(f"  Loss: {loss.item():.4f}")

    print("\n[5] OHEMLoss:")
    ohem = OHEMLoss(class_weights=get_default_class_weights(), thresh=0.5).to(device)
    loss = ohem(logits, target, mask)
    print(f"  Loss: {loss.item():.4f}")

    print("\n[6] DistanceAwareLoss:")
    dist_loss = DistanceAwareLoss(class_weights=get_default_class_weights(), bev_size=(50, 50)).to(device)
    loss = dist_loss(logits, target, mask)
    print(f"  Loss: {loss.item():.4f}")

    print("\n[7] CombinedLoss (All):")
    combined_all = CombinedLoss(
        ce_weight=1.0, 
        lovasz_weight=1.0, 
        ohem_weight=0.5, 
        distance_weight=0.5,
        class_weights=get_default_class_weights(),
        bev_size=(50, 50) # 传入 bev_size 以匹配测试数据
    ).to(device)
    loss = combined_all(logits, target, mask)
    print(f"  Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 测试通过！")

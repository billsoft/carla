import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class DistanceAwareLoss(nn.Module):
    """
    距离感知损失加权

    近距离体素的损失权重更高，因为安全性更重要
    """

    def __init__(
        self,
        voxel_size: Tuple[int, int, int] = (400, 400, 32),
        pc_range: List[float] = [-40, -40, -1, 40, 40, 5.4],
        decay_lambda: float = 20.0,
        base_weight: float = 0.5,
        max_weight: float = 3.0,
    ):
        super().__init__()

        self.decay_lambda = decay_lambda
        self.base_weight = base_weight
        self.max_weight = max_weight

        # 预计算距离权重图
        X, Y, Z = voxel_size

        # 体素中心坐标
        x_range = pc_range[3] - pc_range[0]  # 80m
        y_range = pc_range[4] - pc_range[1]  # 80m

        x = torch.linspace(pc_range[0] + x_range/(2*X),
                          pc_range[3] - x_range/(2*X), X)
        y = torch.linspace(pc_range[1] + y_range/(2*Y),
                          pc_range[4] - y_range/(2*Y), Y)

        xx, yy = torch.meshgrid(x, y, indexing='ij')
        distance = torch.sqrt(xx**2 + yy**2)  # [X, Y]

        # 计算权重: 近距离权重高，远距离权重低
        weight = torch.exp(-distance / decay_lambda) + base_weight
        weight = weight.clamp(max=max_weight)

        # 扩展到 Z 维度 (所有高度使用相同权重)
        weight = weight.unsqueeze(-1).expand(-1, -1, Z)  # [X, Y, Z]

        self.register_buffer('distance_weight', weight)

    def forward(
        self,
        pred: torch.Tensor,      # [B, C, X, Y, Z]
        target: torch.Tensor,    # [B, X, Y, Z]
        ignore_index: int = -100
    ) -> torch.Tensor:
        """
        计算距离加权交叉熵损失
        """
        B, C, X, Y, Z = pred.shape
        device = pred.device

        # Reshape for cross entropy
        pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
        target_flat = target.reshape(-1)

        # 有效掩码
        valid_mask = target_flat != ignore_index

        if valid_mask.sum() == 0:
            return pred_flat.sum() * 0

        # 计算逐体素交叉熵损失 (不 reduce)
        loss_flat = F.cross_entropy(pred_flat, target_flat.clamp(0), reduction='none')
        loss_flat = loss_flat * valid_mask.float()

        # 获取距离权重并 flatten (确保在同一设备上)
        weight = self.distance_weight.to(device).unsqueeze(0).expand(B, -1, -1, -1)  # [B, X, Y, Z]
        weight_flat = weight.reshape(-1)

        # 应用距离权重
        weighted_loss = loss_flat * weight_flat * valid_mask.float()

        # 归一化
        return weighted_loss.sum() / (weight_flat * valid_mask.float()).sum().clamp(min=1e-6)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, pred, target):
        num_classes = pred.shape[1]
        pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
        target = target.view(-1)
        valid_mask = target != self.ignore_index
        pred, target = pred[valid_mask], target[valid_mask]
        if pred.numel() == 0:
            return pred.sum() * 0
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        if self.class_weights is not None:
            focal_weight = focal_weight * self.class_weights.to(pred.device)[target]
        return (focal_weight * ce_loss).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=-100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        B, C, X, Y, Z = pred.shape
        target_onehot = F.one_hot(target.clamp(0), C).permute(0, 4, 1, 2, 3).float()
        valid_mask = (target != self.ignore_index).unsqueeze(1).expand_as(pred)
        pred, target_onehot = pred * valid_mask, target_onehot * valid_mask
        intersection = (pred * target_onehot).sum(dim=(0, 2, 3, 4))
        cardinality = (pred + target_onehot).sum(dim=(0, 2, 3, 4))
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

class FlowLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_flow, gt_flow, valid_mask=None):
        diff = (pred_flow - gt_flow).abs()
        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(1).expand_as(pred_flow)
            return (diff * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        return diff.mean()


class DepthSupervisionLoss(nn.Module):
    """
    深度监督损失 (改进版)

    改进点:
    1. 边缘感知: 在物体边缘处不强制深度平滑
    2. 多尺度监督: 支持高分辨率深度预测
    3. Log 空间 L1 损失: 对近距离更敏感
    """

    def __init__(self, depth_range=(0.5, 80.0), eps=1e-6, use_edge_aware=True, smooth_weight=0.1):
        super().__init__()
        self.min_depth = depth_range[0]
        self.max_depth = depth_range[1]
        self.eps = eps
        self.use_edge_aware = use_edge_aware
        self.smooth_weight = smooth_weight

    def forward(self, depth_pred, depth_gt, valid_mask=None, images=None):
        """
        Args:
            depth_pred: [B, N, H, W] 预测深度
            depth_gt: [B, N, H, W] 真值深度
            valid_mask: [B, N, H, W] 有效掩码
            images: [B, N, C, H_img, W_img] 原始图像 (用于边缘感知)

        Returns:
            depth_loss: 深度损失标量
        """
        # 裁剪深度到有效范围
        depth_pred = depth_pred.clamp(self.min_depth, self.max_depth)
        depth_gt = depth_gt.clamp(self.min_depth, self.max_depth)

        # Log 空间 L1 损失 (对近距离更敏感)
        log_pred = torch.log(depth_pred + self.eps)
        log_gt = torch.log(depth_gt + self.eps)
        base_loss = torch.abs(log_pred - log_gt)

        # 边缘感知平滑损失
        smooth_loss = torch.tensor(0.0, device=depth_pred.device)
        if self.use_edge_aware and images is not None:
            smooth_loss = self._compute_edge_aware_smooth_loss(depth_pred, images)

        # 应用有效掩码
        if valid_mask is not None:
            # 只计算有效深度的损失
            valid_mask = valid_mask.float()
            base_loss = (base_loss * valid_mask).sum() / (valid_mask.sum() + self.eps)
        else:
            base_loss = base_loss.mean()

        return base_loss + self.smooth_weight * smooth_loss

    def _compute_edge_aware_smooth_loss(self, depth_pred, images):
        """
        计算边缘感知平滑损失

        在物体边缘处，深度不连续是正常的，不应该惩罚
        """
        B, N, H_pred, W_pred = depth_pred.shape
        device = depth_pred.device

        # 处理图像维度
        if images.dim() == 5:
            _, _, C, H_img, W_img = images.shape
        else:
            # 假设已经是 [B*N, C, H, W]
            return torch.tensor(0.0, device=device)

        # 下采样图像到深度预测尺寸
        if H_img != H_pred or W_img != W_pred:
            images_down = F.interpolate(
                images.view(B * N, C, H_img, W_img),
                size=(H_pred, W_pred),
                mode='bilinear',
                align_corners=False
            ).view(B, N, C, H_pred, W_pred)
        else:
            images_down = images

        # 计算图像梯度 (边缘)
        img_gray = images_down.mean(dim=2)  # [B, N, H, W]

        grad_x = torch.abs(img_gray[:, :, :, 1:] - img_gray[:, :, :, :-1])
        grad_y = torch.abs(img_gray[:, :, 1:, :] - img_gray[:, :, :-1, :])

        # 边缘权重: 边缘处权重小 (不惩罚深度不连续)
        edge_weight_x = torch.exp(-grad_x * 10)
        edge_weight_y = torch.exp(-grad_y * 10)

        # 深度梯度
        depth_grad_x = torch.abs(depth_pred[:, :, :, 1:] - depth_pred[:, :, :, :-1])
        depth_grad_y = torch.abs(depth_pred[:, :, 1:, :] - depth_pred[:, :, :-1, :])

        # 边缘感知平滑损失
        smooth_loss = (edge_weight_x * depth_grad_x).mean() + \
                     (edge_weight_y * depth_grad_y).mean()

        return smooth_loss

class CoarseToFineLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, coarse_weight=0.3, focal_gamma=2.0, focal_alpha=0.25, flow_weight=0.5,
                 voxel_size=(400, 400, 32), pc_range=[-40, -40, -1, 40, 40, 5.4], use_distance_aware=True, distance_weight=0.2,
                 use_depth_supervision=True, depth_weight=0.5, depth_range=(0.5, 80.0)):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.flow_weight = flow_weight
        self.distance_loss_weight = distance_weight
        self.use_distance_aware = use_distance_aware
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma, class_weights)
        self.dice_loss = DiceLoss()
        self.flow_loss = FlowLoss()
        if use_distance_aware:
            self.distance_loss = DistanceAwareLoss(voxel_size=voxel_size, pc_range=pc_range)

        # 深度监督
        self.use_depth_supervision = use_depth_supervision
        self.depth_weight = depth_weight
        if use_depth_supervision:
            self.depth_loss = DepthSupervisionLoss(depth_range=depth_range)

    def forward(self, outputs, targets):
        losses = {}
        semantic_pred = outputs['semantic']
        semantic_gt = targets['semantic']
        losses['focal'] = self.focal_loss(semantic_pred, semantic_gt)
        losses['dice'] = self.dice_loss(semantic_pred, semantic_gt)
        # Distance-Aware Loss (近距离体素权重更高)
        if self.use_distance_aware:
            losses['distance'] = self.distance_loss(semantic_pred, semantic_gt) * self.distance_loss_weight
        if 'coarse_semantic' in outputs:
            coarse_gt = F.interpolate(semantic_gt.unsqueeze(1).float(), size=outputs['coarse_semantic'].shape[2:], mode='nearest').squeeze(1).long()
            losses['coarse_focal'] = self.focal_loss(outputs['coarse_semantic'], coarse_gt) * self.coarse_weight
            losses['coarse_dice'] = self.dice_loss(outputs['coarse_semantic'], coarse_gt) * self.coarse_weight
        if 'flow' in outputs and 'flow' in targets:
            flow_mask = targets.get('flow_mask')
            losses['flow'] = self.flow_loss(outputs['flow'], targets['flow'], flow_mask) * self.flow_weight
            if 'coarse_flow' in outputs:
                coarse_flow_gt = F.interpolate(targets['flow'], size=outputs['coarse_flow'].shape[2:], mode='trilinear', align_corners=False)
                coarse_mask = F.interpolate(flow_mask.unsqueeze(1).float(), size=outputs['coarse_flow'].shape[2:], mode='nearest').squeeze(1).bool() if flow_mask is not None else None
                losses['coarse_flow'] = self.flow_loss(outputs['coarse_flow'], coarse_flow_gt, coarse_mask) * self.flow_weight * self.coarse_weight

        # 深度监督损失 (改进版: 避免下采样精度损失)
        if self.use_depth_supervision and 'depth_pred' in outputs and 'depth' in targets:
            depth_pred = outputs['depth_pred']  # [B, N, H, W]
            depth_gt = targets['depth']          # [B, N, H, W]
            depth_valid = targets.get('depth_valid', None)
            images = targets.get('images', None)  # 用于边缘感知损失

            B, N, H_gt, W_gt = depth_gt.shape
            _, _, H_pred, W_pred = depth_pred.shape

            if H_gt != H_pred or W_gt != W_pred:
                # 改进: 使用最近邻下采样保留边缘深度
                # 或者使用 min-pooling (取局部最小深度，保守估计)
                depth_gt_reshaped = depth_gt.view(B * N, 1, H_gt, W_gt)

                # 方法1: 最近邻下采样 (保留边缘)
                depth_gt_down = F.interpolate(depth_gt_reshaped, size=(H_pred, W_pred), mode='nearest')

                # 方法2: min-pooling (可选，更保守)
                # kernel_h = H_gt // H_pred
                # kernel_w = W_gt // W_pred
                # depth_gt_down = -F.max_pool2d(-depth_gt_reshaped, kernel_size=(kernel_h, kernel_w))

                depth_gt = depth_gt_down.view(B, N, H_pred, W_pred)

                if depth_valid is not None:
                    depth_valid = depth_valid.view(B * N, 1, H_gt, W_gt).float()
                    # 有效掩码也使用最近邻下采样
                    depth_valid = F.interpolate(depth_valid, size=(H_pred, W_pred), mode='nearest')
                    depth_valid = depth_valid.view(B, N, H_pred, W_pred) > 0.5

            # 调用改进的深度损失 (支持边缘感知)
            losses['depth'] = self.depth_loss(depth_pred, depth_gt, depth_valid, images) * self.depth_weight

        losses['total'] = sum(losses.values())
        return losses

class OccLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 获取距离感知损失配置
        use_distance_aware = getattr(config, 'use_distance_aware', True)
        distance_weight = getattr(config, 'distance_loss_weight', 0.2)
        # 获取深度监督配置
        use_depth_supervision = getattr(config, 'use_depth_supervision', True)
        depth_weight = getattr(config, 'depth_loss_weight', 0.5)
        depth_range = getattr(config, 'depth_range', (0.5, 80.0))

        self.loss_fn = CoarseToFineLoss(
            num_classes=config.num_classes,
            class_weights=config.class_weights,
            coarse_weight=config.coarse_loss_weight,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
            flow_weight=config.flow_loss_weight,
            voxel_size=config.voxel_size,
            pc_range=config.pc_range,
            use_distance_aware=use_distance_aware,
            distance_weight=distance_weight,
            use_depth_supervision=use_depth_supervision,
            depth_weight=depth_weight,
            depth_range=depth_range,
        )

    def forward(self, outputs, targets):
        return self.loss_fn(outputs, targets)

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CoarseToFineLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, coarse_weight=0.3, focal_gamma=2.0, focal_alpha=0.25, flow_weight=0.5):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.flow_weight = flow_weight
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma, class_weights)
        self.dice_loss = DiceLoss()
        self.flow_loss = FlowLoss()

    def forward(self, outputs, targets):
        losses = {}
        semantic_pred = outputs['semantic']
        semantic_gt = targets['semantic']
        losses['focal'] = self.focal_loss(semantic_pred, semantic_gt)
        losses['dice'] = self.dice_loss(semantic_pred, semantic_gt)
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
        losses['total'] = sum(losses.values())
        return losses

class OccLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.loss_fn = CoarseToFineLoss(num_classes=config.num_classes, class_weights=config.class_weights, coarse_weight=config.coarse_loss_weight, focal_gamma=config.focal_gamma, focal_alpha=config.focal_alpha, flow_weight=config.flow_loss_weight)

    def forward(self, outputs, targets):
        return self.loss_fn(outputs, targets)

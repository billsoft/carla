import torch
import torch.nn as nn
import torch.nn.functional as F

class OccupancyLoss(nn.Module):
    def __init__(self, num_classes=18, ignore_index=255, lovasz_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lovasz_weight = lovasz_weight
    
    def forward(self, pred, target):
        B, C, X, Y, Z = pred.shape
        pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
        target_flat = target.reshape(-1)
        valid_mask = target_flat != self.ignore_index
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        if pred_valid.numel() == 0:
            return {'total': torch.tensor(0.0, device=pred.device), 'ce': torch.tensor(0.0, device=pred.device), 'lovasz': torch.tensor(0.0, device=pred.device)}
        ce_loss = F.cross_entropy(pred_valid, target_valid)
        lovasz_loss = self.lovasz_softmax(pred, target)
        total_loss = ce_loss + self.lovasz_weight * lovasz_loss
        return {'total': total_loss, 'ce': ce_loss, 'lovasz': lovasz_loss}
    
    def lovasz_softmax(self, pred, target):
        B, C, X, Y, Z = pred.shape
        pred = F.softmax(pred, dim=1)
        losses = []
        for b in range(B):
            for c in range(C):
                fg = (target[b] == c).float()
                if fg.sum() == 0:
                    continue
                errors = (fg - pred[b, c]).abs()
                errors_sorted, perm = torch.sort(errors.view(-1), descending=True)
                fg_sorted = fg.view(-1)[perm]
                grad = self.lovasz_grad(fg_sorted)
                losses.append((errors_sorted * grad).sum())
        if len(losses) == 0:
            return torch.tensor(0.0, device=pred.device)
        return sum(losses) / len(losses)
    
    def lovasz_grad(self, gt_sorted):
        n = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / (union + 1e-6)
        if n > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

class SceneFlowLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_flow, target_flow, mask=None):
        diff = (pred_flow - target_flow).abs()
        if mask is not None:
            diff = diff * mask.unsqueeze(1)
            return diff.sum() / (mask.sum() * 3 + 1e-6)
        return diff.mean()

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_softmax_flat(logits, target, ignore_index=255):
    """logits: [M, C]，target: [M] → Lovász-Softmax（按出现过的类别取平均）。"""
    valid = target != ignore_index
    if valid.sum() == 0:
        return logits.sum() * 0.0
    prob = F.softmax(logits[valid].float(), dim=1)
    target = target[valid]
    losses = []
    for c in range(prob.shape[1]):
        fg = (target == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - prob[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        losses.append((errors_sorted * _lovasz_grad(fg[perm])).sum())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-6)
    if len(gt_sorted) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


class OccupancyLoss(nn.Module):
    """语义损失：CE + lovasz_weight × Lovász-Softmax。接受稠密 [B,C,X,Y,Z] 或点形式 [B,P,C]。

    Lovász 需要对每个类别的误差做全量排序并保留反向所需的排序结果，在 512 万体素上
    单帧要占数 GB 显存。超过 lovasz_max_points 时子采样：保留全部非空体素（小类别全保留），
    空体素（0: free）随机采样补足。Lovász 是基于排序的 IoU 替代，子采样是常用近似。
    """

    def __init__(self, num_classes=18, ignore_index=255, lovasz_weight=0.5, lovasz_max_points=0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lovasz_weight = lovasz_weight
        self.lovasz_max_points = lovasz_max_points

    def _lovasz_subsample(self, pred_flat, target_flat):
        M = target_flat.numel()
        if not self.lovasz_max_points or M <= self.lovasz_max_points:
            return pred_flat, target_flat
        occ = torch.nonzero((target_flat != 0) & (target_flat != self.ignore_index), as_tuple=False).squeeze(1)
        if occ.numel() >= self.lovasz_max_points:
            occ = occ[torch.randperm(occ.numel(), device=occ.device)[:self.lovasz_max_points]]
            idx = occ
        else:
            free = torch.nonzero(target_flat == 0, as_tuple=False).squeeze(1)
            n_free = self.lovasz_max_points - occ.numel()
            free = free[torch.randperm(free.numel(), device=free.device)[:n_free]]
            idx = torch.cat([occ, free])
        return pred_flat[idx], target_flat[idx]

    def forward(self, pred, target):
        if pred.dim() == 5:                                   # [B, C, X, Y, Z]
            pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, pred.shape[1])
        else:                                                 # [B, P, C]
            pred_flat = pred.reshape(-1, pred.shape[-1])
        target_flat = target.reshape(-1)
        valid = target_flat != self.ignore_index
        if valid.sum() == 0:
            zero = pred.sum() * 0.0
            return {'total': zero, 'ce': zero, 'lovasz': zero}
        ce = F.cross_entropy(pred_flat[valid].float(), target_flat[valid])
        lov = lovasz_softmax_flat(*self._lovasz_subsample(pred_flat, target_flat), self.ignore_index)
        return {'total': ce + self.lovasz_weight * lov, 'ce': ce, 'lovasz': lov}


def depth_bin_edges(config, device):
    return torch.exp(torch.linspace(math.log(config.depth_min), math.log(config.depth_max),
                                    config.depth_bins + 1, device=device))


def depth_to_bins(depth, config):
    """depth [..] 米（0 = 无效）→ (bin 索引, 有效掩码)。对数均匀分 bin。"""
    edges = depth_bin_edges(config, depth.device)
    idx = torch.bucketize(depth, edges[1:-1])
    valid = (depth >= config.depth_min) & (depth < config.depth_max)
    return idx, valid


def coarse_occupancy_target(voxels, config, ignore_index=255):
    """[B, X, Y, Z] 语义真值 → ([B, cx, cy, cz] 粗格是否含非空体素, 同形状 有效掩码)。

    粗格里只要有一个观测到的非空体素就是占用；全部体素都是 ignore（相机没观测到）的粗格无效、不监督——
    否则 ignore 会被当成空，粗占用头学到"看不见的地方都是空的"。"""
    occ = ((voxels != 0) & (voxels != ignore_index)).float().unsqueeze(1)
    seen = (voxels != ignore_index).float().unsqueeze(1)
    k = tuple(v // c for v, c in zip(config.voxel_size, config.coarse_size))
    target = F.max_pool3d(occ, kernel_size=k, stride=k).squeeze(1)
    valid = F.max_pool3d(seen, kernel_size=k, stride=k).squeeze(1) > 0
    return target, valid


def sample_implicit_points(voxels, config, num_points, ignore_index=255):
    """隐式头训练采样：occupied_ratio 比例来自非空体素，其余从非 ignore 体素里均匀随机
    （采到 ignore 的点不产生损失，白白浪费采样预算）。

    返回 flat_idx [B, P]（[X,Y,Z] 行优先索引）、points [B, P, 3]（grid_sample 归一化坐标）。
    """
    B = voxels.shape[0]
    X, Y, Z = config.voxel_size
    flat = voxels.view(B, -1)
    n_occ = int(num_points * config.implicit_occupied_ratio)
    idx_list = []
    for b in range(B):
        occ_idx = torch.nonzero((flat[b] != 0) & (flat[b] != ignore_index), as_tuple=False).squeeze(1)
        if occ_idx.numel() > 0:
            pick = occ_idx[torch.randint(0, occ_idx.numel(), (n_occ,), device=voxels.device)]
        else:
            pick = torch.randint(0, flat.shape[1], (n_occ,), device=voxels.device)
        seen_idx = torch.nonzero(flat[b] != ignore_index, as_tuple=False).squeeze(1)
        if 0 < seen_idx.numel() < flat.shape[1]:
            rnd = seen_idx[torch.randint(0, seen_idx.numel(), (num_points - n_occ,), device=voxels.device)]
        else:
            rnd = torch.randint(0, flat.shape[1], (num_points - n_occ,), device=voxels.device)
        idx_list.append(torch.cat([pick, rnd]))
    flat_idx = torch.stack(idx_list)
    ix = flat_idx // (Y * Z)
    iy = (flat_idx // Z) % Y
    iz = flat_idx % Z
    pts = torch.stack([(ix.float() + 0.5) * 2.0 / X - 1.0,
                       (iy.float() + 0.5) * 2.0 / Y - 1.0,
                       (iz.float() + 0.5) * 2.0 / Z - 1.0], dim=-1)
    return flat_idx, pts


class E2ELoss(nn.Module):
    """总损失 = 语义 + depth_w·深度 + flow_w·流场 + coarse_w·粗占用（各项按配置/数据可用性启用）。

    targets 字典（单帧）:
      voxels [B,X,Y,Z] long；depth [B,N,h,w]（米，0 无效）；flow [B,3,X,Y,Z]；flow_mask [B,X,Y,Z]；
      has_depth / has_flow [B] bool；隐式头时还有 point_idx [B,P]
    """

    def __init__(self, config, ignore_index=255):
        super().__init__()
        self.config = config
        self.ignore_index = ignore_index
        self.semantic = OccupancyLoss(config.num_classes, ignore_index,
                                      lovasz_max_points=config.lovasz_max_points)

    def forward(self, outputs, targets):
        cfg = self.config
        voxels = targets['voxels']
        losses = {}

        if 'semantic_points' in outputs:
            idx = targets['point_idx']
            sem_target = voxels.view(voxels.shape[0], -1).gather(1, idx)
            sem = self.semantic(outputs['semantic_points'], sem_target)
        else:
            sem = self.semantic(outputs['semantic'], voxels)
        losses.update(sem)
        total = sem['total']

        if cfg.use_depth_aux and outputs.get('depth_logits') is not None and 'depth' in targets:
            logits = outputs['depth_logits']                               # [B, N, D, h, w]
            bins, valid = depth_to_bins(targets['depth'], cfg)
            valid = valid & targets['has_depth'].view(-1, 1, 1, 1)
            if valid.any():
                lg = logits.permute(0, 1, 3, 4, 2)[valid].float()
                d_loss = F.cross_entropy(lg, bins[valid])
            else:
                d_loss = logits.sum() * 0.0
            losses['depth'] = d_loss
            total = total + cfg.depth_loss_weight * d_loss

        if cfg.use_flow_head and 'flow' in targets:
            mask = (targets['flow_mask'].bool() & targets['has_flow'].view(-1, 1, 1, 1)
                    & (voxels != self.ignore_index))
            if 'flow_points' in outputs and outputs['flow_points'] is not None:
                idx = targets['point_idx']
                B = voxels.shape[0]
                pred = outputs['flow_points']                               # [B, P, 3]
                tgt = targets['flow'].view(B, 3, -1).gather(2, idx.unsqueeze(1).expand(-1, 3, -1)).transpose(1, 2)
                m = mask.view(B, -1).gather(1, idx)
            elif outputs.get('flow') is not None:
                pred = outputs['flow'].permute(0, 2, 3, 4, 1)               # [B, X, Y, Z, 3]
                tgt = targets['flow'].permute(0, 2, 3, 4, 1)
                m = mask
            else:
                pred = None
            if pred is not None:
                if m.any():
                    f_loss = F.smooth_l1_loss(pred[m].float(), tgt[m].float(), beta=0.1)
                else:
                    f_loss = pred.sum() * 0.0
                losses['flow'] = f_loss
                total = total + cfg.flow_loss_weight * f_loss

        if outputs.get('coarse_occ_logits') is not None and cfg.coarse_occ_loss_weight > 0:
            occ_t, occ_valid = coarse_occupancy_target(voxels, cfg, self.ignore_index)
            if occ_valid.any():
                c_loss = F.binary_cross_entropy_with_logits(outputs['coarse_occ_logits'].float()[occ_valid],
                                                            occ_t[occ_valid])
            else:
                c_loss = outputs['coarse_occ_logits'].sum() * 0.0
            losses['coarse_occ'] = c_loss
            total = total + cfg.coarse_occ_loss_weight * c_loss

        losses['total'] = total
        return losses


@torch.no_grad()
def update_confusion(confusion, pred, target, num_classes, ignore_index=255):
    """pred/target: 任意同形状整型张量；confusion: [C, C]（行=真值，列=预测）。"""
    valid = target != ignore_index
    t = target[valid].long()
    p = pred[valid].long()
    confusion += torch.bincount(t * num_classes + p, minlength=num_classes * num_classes).view(
        num_classes, num_classes).to(confusion.dtype)
    return confusion


def miou_from_confusion(confusion, exclude=(0,)):
    """返回 (mIoU, 每类 IoU)。默认不把 0:free 计入 mIoU（与 Occ3D 等基准一致）。"""
    inter = confusion.diag().float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - inter
    iou = inter / union.clamp(min=1)
    present = confusion.sum(1) > 0
    keep = present.clone()
    for c in exclude:
        keep[c] = False
    miou = iou[keep].mean().item() if keep.any() else 0.0
    return miou, iou

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import E2EOccConfig, upgrade_config
from e2e_occ_net import build_model
from loss import E2ELoss, sample_implicit_points, update_confusion, miou_from_confusion

CLASS_NAMES = ['free', 'barrier', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'pedestrian',
               'traffic_cone', 'trailer', 'truck', 'driveable', 'other_flat', 'sidewalk', 'terrain',
               'manmade', 'vegetation', 'general_obj']
from dataset import get_dataloader

# conda activate deepsys
# python e2e_occ/train.py --data_root <dataset_dir> --batch_size 1 --epochs 100 --amp --grad_accum 4

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def frame_targets(batch, t, device, config, point_idx=None):
    tg = {'voxels': batch['voxels'][:, t].to(device, non_blocking=True).long()}
    if 'depth' in batch:
        tg['depth'] = batch['depth'][:, t].to(device, non_blocking=True)
        tg['has_depth'] = batch['has_depth'][:, t].to(device)
    if 'flow' in batch:
        tg['flow'] = batch['flow'][:, t].to(device, non_blocking=True).float()
        tg['flow_mask'] = batch['flow_mask'][:, t].to(device, non_blocking=True)
        tg['has_flow'] = batch['has_flow'][:, t].to(device)
    if point_idx is not None:
        tg['point_idx'] = point_idx
    return tg


def run_sequence(model, batch, criterion, config, device, amp_dtype, use_amp, train=True,
                 tbptt=2, grad_scale=1.0, scaler=None, confusion=None):
    """逐帧前向一个序列（T=1 时即单帧）；训练时按 tbptt 帧一段做截断反向传播。

    返回各损失项的序列平均值（python float 字典）。
    """
    images = batch['images'].to(device, non_blocking=True)            # [B, T, N, 1, H, W]
    intrinsics = batch['intrinsics'].to(device)
    extrinsics = batch['extrinsics'].to(device)
    ego_motion_all = batch['ego_motion'].to(device)
    B, T = images.shape[:2]
    if 'time_delta' in batch:                                          # 数据集按场景给出 (抽帧采集)
        time_delta_all = batch['time_delta'].to(device).float()        # [B, T]
    else:
        time_delta_all = torch.full((B, T), config.frame_dt, device=device)

    memory = None
    logs = {}
    # 时间加权：越靠后的帧可利用的历史越多，权重越高（鼓励网络利用时序信息）。
    # 按整个序列的权重和归一化（而不是每段各自归一化），否则 tbptt=1 时每段只有一帧，
    # loss·w/w 会把加权完全抵消。
    weights = [1.0 + (t / max(1, T - 1)) for t in range(T)]
    w_sum = sum(weights)
    chunk_loss = 0.0
    for t in range(T):
        if train and t % tbptt == 0 and memory is not None:
            memory = memory.detach()
        ego_motion = ego_motion_all[:, t] if t > 0 else None

        point_idx, query_points = None, None
        if train and config.head_type == 'implicit':
            vox = batch['voxels'][:, t].to(device).long()
            point_idx, query_points = sample_implicit_points(vox, config, config.implicit_train_points)

        with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            out = model(images[:, t], intrinsics, extrinsics, memory=memory, ego_motion=ego_motion,
                        time_delta=time_delta_all[:, t] if t > 0 else None, query_points=query_points)
            tg = frame_targets(batch, t, device, config, point_idx)
            losses = criterion(out, tg)

        memory = out['memory']
        for k, v in losses.items():
            logs[k] = logs.get(k, 0.0) + float(v.detach()) / T

        if confusion is not None and 'semantic' in out:
            pred = out['semantic'].argmax(1)
            update_confusion(confusion, pred, tg['voxels'], config.num_classes)

        if train:
            chunk_loss = chunk_loss + losses['total'] * weights[t]
            if (t + 1) % tbptt == 0 or t == T - 1:
                scaler.scale(chunk_loss / (w_sum * grad_scale)).backward()
                chunk_loss = 0.0
    return logs


def optimizer_step(model, optimizer, scaler, scheduler):
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def train_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, epoch, config, args, amp_dtype):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals, n = {}, 0
    pending = 0                                   # 已累积梯度、还没 step 的 batch 数
    start = time.time()
    for i, batch in enumerate(loader):
        logs = run_sequence(model, batch, criterion, config, device, amp_dtype, args.amp, train=True,
                            tbptt=args.tbptt, grad_scale=args.grad_accum, scaler=scaler)
        pending += 1
        if pending == args.grad_accum:
            optimizer_step(model, optimizer, scaler, scheduler)
            pending = 0
        for k, v in logs.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
        msg = ' '.join(f'{k}={v:.4f}' for k, v in logs.items())
        print(f'Epoch {epoch} [{i + 1}/{len(loader)}] {msg} Time: {time.time() - start:.1f}s', flush=True)
        if args.max_steps and i + 1 >= args.max_steps:
            break
    if pending:
        # epoch 末尾凑不满 grad_accum 的 batch：之前直接在下个 epoch 开头 zero_grad 丢掉 (白算)。
        # 梯度按 grad_accum 缩放过，这里补回实际 batch 数的比例再 step
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(args.grad_accum / pending)
        optimizer_step(model, optimizer, scaler, scheduler)
    return {k: v / max(1, n) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, config, args, amp_dtype):
    model.eval()
    totals, n = {}, 0
    confusion = torch.zeros(config.num_classes, config.num_classes, dtype=torch.long, device=device)
    for i, batch in enumerate(loader):
        logs = run_sequence(model, batch, criterion, config, device, amp_dtype, args.amp, train=False,
                            confusion=confusion)
        for k, v in logs.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
        if args.max_steps and i + 1 >= args.max_steps:
            break
    torch.cuda.empty_cache()
    miou, iou = miou_from_confusion(confusion)
    out = {k: v / max(1, n) for k, v in totals.items()}
    out['miou'] = miou
    present = (confusion.sum(1) > 0).tolist()
    # 几何 IoU (占用/空，不分类别；Occ3D 常用的第二个指标)，只在观测体素上 (confusion 已排除 255)
    occ_tp = confusion[1:, 1:].sum().item()
    occ_union = confusion.sum().item() - confusion[0, 0].item()
    out['geo_iou'] = occ_tp / max(1, occ_union)
    out['per_class'] = ' '.join(f'{CLASS_NAMES[c]}={iou[c].item():.3f}' for c in range(1, config.num_classes)
                                if present[c])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=0, help='Windows 下默认 0，避免死锁')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained_backbone', type=str, default=None,
                        help='pretrain_mae.py 产出的图像骨干权重（只加载 patch_embed/encoder 中形状匹配的参数）')
    parser.add_argument('--amp', action='store_true', help='混合精度训练')
    parser.add_argument('--amp_dtype', choices=['fp16', 'bf16'], default='bf16',
                        help='bf16 动态范围大，无需 loss scaling（4090 支持）')
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--tbptt', type=int, default=1,
                        help='时序截断反向传播的段长（帧）。1 = 每帧单独反向、记忆逐帧 detach（StreamPETR 式流式训练，'
                             '显存与单帧相同）；2 及以上让梯度穿过记忆，显存按帧数线性增长')
    parser.add_argument('--compile', action='store_true', help='torch.compile（Windows 上 Triton 不稳定，仅供尝试）')
    parser.add_argument('--max_steps', type=int, default=0, help='每个 epoch 最多跑多少个 batch（冒烟测试用）')
    parser.add_argument('--seed', type=int, default=0, help='随机种子 (样本顺序、隐式头采样点、参数初始化)')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='覆盖配置字段，例如 --set use_sparse_fine=False head_type=implicit')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16

    config = E2EOccConfig()
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if 'config' in ckpt:
            config = upgrade_config(ckpt['config'])
    for kv in args.set:
        k, v = kv.split('=', 1)
        cur = getattr(config, k)
        if isinstance(cur, bool):
            v = v.lower() in ('1', 'true', 'yes')
        elif isinstance(cur, tuple):
            v = tuple(type(cur[0])(x) for x in v.split(','))
        else:
            v = type(cur)(v)
        setattr(config, k, v)

    model = build_model(config).to(device)
    if args.pretrained_backbone:
        sd = torch.load(args.pretrained_backbone, map_location=device, weights_only=False)
        sd = sd.get('backbone', sd)
        own = model.state_dict()
        loadable = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(loadable, strict=False)
        print(f'已加载预训练骨干参数 {len(loadable)}/{len(sd)} 个')

    criterion = E2ELoss(config)
    # 偏置、归一化层、可学习位置/查询嵌入 (1 维参数) 不做 weight decay：对它们衰减只会把尺度往 0 拉
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay},
                                   {'params': no_decay, 'weight_decay': 0.0}], lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and amp_dtype == torch.float16))

    train_loader = get_dataloader(args.data_root, 'train', args.batch_size, args.num_workers, config)
    val_loader = get_dataloader(args.data_root, 'val', args.batch_size, args.num_workers, config)

    max_batches = min(len(train_loader), args.max_steps) if args.max_steps else len(train_loader)
    steps_per_epoch = max(1, math.ceil(max_batches / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    # 预热不超过总步数的 1/5：小数据集上默认 500 步预热可能比整个训练还长，学习率永远升不上去
    warmup = min(args.warmup_steps, max(1, total_steps // 5))
    if warmup < args.warmup_steps:
        print(f'预热步数 {args.warmup_steps} 超过总优化步数 {total_steps} 的 1/5，改为 {warmup}')

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_miou = -1.0
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        # 之前用 ckpt['miou']：从 last_model 恢复时那是最后一个 epoch 的分数而不是历史最好，
        # 之后更差的模型也可能覆盖 best_model.pth
        best_miou = ckpt.get('best_miou', ckpt.get('miou', -1.0))
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        else:                                     # 旧 checkpoint：按 epoch 推算步数
            for _ in range(start_epoch * steps_per_epoch):
                scheduler.step()
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        print(f'从 epoch {start_epoch - 1} 恢复 (历史最好 mIoU {best_miou:.4f})')

    train_model = torch.compile(model) if args.compile else model
    print(f'模型参数量: {model.get_num_params() / 1e6:.2f}M | 样本: train {len(train_loader.dataset)} '
          f'val {len(val_loader.dataset)} | AMP={args.amp}({args.amp_dtype}) T={config.temporal_frames} '
          f'TBPTT={args.tbptt} head={config.head_type} sparse={config.use_sparse_fine}', flush=True)

    for epoch in range(start_epoch, args.epochs):
        tr = train_epoch(train_model, train_loader, criterion, optimizer, scaler, scheduler, device, epoch,
                         config, args, amp_dtype)

        va = validate(model, val_loader, criterion, device, config, args, amp_dtype)
        print(f"Epoch {epoch}: train_total={tr.get('total', 0):.4f} val_total={va.get('total', 0):.4f} "
              f"val_mIoU={va['miou']:.4f} val_geoIoU={va['geo_iou']:.4f} lr={optimizer.param_groups[0]['lr']:.2e}",
              flush=True)
        print(f"  per-class IoU: {va['per_class']}", flush=True)

        is_best = va['miou'] > best_miou
        best_miou = max(best_miou, va['miou'])
        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                 'loss': va.get('total', 0.0), 'miou': va['miou'], 'best_miou': best_miou, 'config': config}
        torch.save(state, os.path.join(args.output_dir, 'last_model.pth'))
        if is_best:
            torch.save(state, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'Saved best model at epoch {epoch} (mIoU {best_miou:.4f})', flush=True)


if __name__ == '__main__':
    main()

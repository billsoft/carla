"""
OccNetV3 时序训练脚本 (统一版)

支持三种时序训练模式:
============================================================

1. Memory Cell (默认, 推荐)
   - 原理: ConvGRU 压缩时序信息到单个 memory state
   - 显存: 无额外增长 (每帧独立 backward)
   - 效果: 接近完整时序融合 (~95% 精度)
   - 参考: BEVFormer v2, StreamPETR, VideoBEV

2. Gradient Accumulation (梯度累积)
   - 原理: 每帧独立 backward, 累积梯度后更新
   - 显存: 无额外增长
   - 效果: 等效于更大 batch size

3. Classic TBPTT (经典, 高显存)
   - 原理: 累积多帧 loss 后一次性 backward
   - 显存: 随帧数线性增长 (~12GB for 3帧)
   - 效果: 梯度可回传到历史帧

使用方法:
    # Memory Cell (推荐)
    python train_tbptt.py --dataset /path/to/data --mode memory_cell --amp

    # Gradient Accumulation
    python train_tbptt.py --dataset /path/to/data --mode grad_accum --window 3 --amp

    # Classic TBPTT (高显存)
    python train_tbptt.py --dataset /path/to/data --mode classic --window 3 --amp
"""

import torch
import torch.nn as nn
import os
import time
import argparse
from torch.amp import GradScaler, autocast
from configs.default import config
from models import build_model
from losses.losses import OccLoss
from data.dataset import build_dataloader, FP16DataPrefetcher


# ==================== Memory Cell 训练 ====================

def train_one_epoch_memory_cell(model, loader, optimizer, scaler, loss_fn, epoch, cfg):
    """
    Memory Cell 训练 - 显存友好版 (推荐)

    每帧独立 backward, 时序信息通过 memory 值传递
    """
    model.train()
    total_loss = 0
    device = next(model.parameters()).device

    print(f"  [Memory Cell] 显存友好训练")
    prefetcher = FP16DataPrefetcher(loader, device)
    batch = prefetcher.next()

    step = 0
    start_time = time.time()
    last_scene_id = None

    while batch is not None:
        data_time = time.time() - start_time

        # 场景切换检测
        scene_id = batch.get('scene_id', None)
        timestamp = batch.get('timestamp', None)

        if scene_id is not None and last_scene_id is not None:
            current_scene = scene_id[0] if isinstance(scene_id, (list, tuple)) else scene_id
            if current_scene != last_scene_id:
                print(f"  [Scene Switch] {last_scene_id} -> {current_scene}")
                model.reset_temporal()

        if scene_id is not None:
            last_scene_id = scene_id[0] if isinstance(scene_id, (list, tuple)) else scene_id

        # Forward
        optimizer.zero_grad()

        with autocast('cuda', enabled=cfg.use_amp):
            outputs = model(
                batch['images'],
                batch.get('ego_motion'),
                batch.get('ego_pose'),
                timestamp=timestamp,
                scene_id=scene_id,
                intrinsics=batch.get('intrinsics'),
                extrinsics=batch.get('extrinsics'),
            )
            losses = loss_fn(outputs, batch)
            loss = losses['total']

        # NaN 检查
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Loss is {loss.item()} at Step {step}, skipping.")
            batch = prefetcher.next()
            step += 1
            start_time = time.time()
            continue

        # Backward
        if cfg.use_amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        total_loss += loss.item()
        batch_time = time.time() - start_time

        # 日志
        if step % cfg.log_interval == 0:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            dist_loss = losses.get('distance', 0)
            if isinstance(dist_loss, torch.Tensor):
                dist_loss = dist_loss.item()
            depth_loss = losses.get('depth', 0)
            if isinstance(depth_loss, torch.Tensor):
                depth_loss = depth_loss.item()

            print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f} "
                  f"Focal: {losses.get('focal', 0):.4f} Dice: {losses.get('dice', 0):.4f} "
                  f"Dist: {dist_loss:.4f} Depth: {depth_loss:.4f} "
                  f"GPU: {mem:.2f}GB", flush=True)

        batch = prefetcher.next()
        step += 1
        start_time = time.time()

    return total_loss / max(step, 1)


# ==================== Gradient Accumulation 训练 ====================

def train_one_epoch_grad_accum(model, loader, optimizer, scaler, loss_fn, epoch, cfg, window=3):
    """
    Gradient Accumulation 训练

    每帧独立 backward, 累积梯度后更新
    等效于 window 倍的 batch size
    """
    model.train()
    total_loss = 0
    device = next(model.parameters()).device

    print(f"  [Grad Accum] 窗口大小: {window}")
    prefetcher = FP16DataPrefetcher(loader, device)
    batch = prefetcher.next()

    step = 0
    window_step = 0
    start_time = time.time()
    last_scene_id = None

    optimizer.zero_grad()

    while batch is not None:
        data_time = time.time() - start_time

        # 场景切换检测
        scene_id = batch.get('scene_id', None)
        timestamp = batch.get('timestamp', None)

        scene_changed = False
        if scene_id is not None and last_scene_id is not None:
            current_scene = scene_id[0] if isinstance(scene_id, (list, tuple)) else scene_id
            if current_scene != last_scene_id:
                scene_changed = True
                print(f"  [Scene Switch] {last_scene_id} -> {current_scene}")

        if scene_id is not None:
            last_scene_id = scene_id[0] if isinstance(scene_id, (list, tuple)) else scene_id

        # 场景切换时完成当前窗口
        if scene_changed and window_step > 0:
            if cfg.use_amp:
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            optimizer.zero_grad()
            model.reset_temporal()
            window_step = 0

        # Forward
        with autocast('cuda', enabled=cfg.use_amp):
            outputs = model(
                batch['images'],
                batch.get('ego_motion'),
                batch.get('ego_pose'),
                timestamp=timestamp,
                scene_id=scene_id,
                intrinsics=batch.get('intrinsics'),
                extrinsics=batch.get('extrinsics'),
            )
            losses = loss_fn(outputs, batch)
            loss = losses['total'] / window  # 平均梯度

        # NaN 检查
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Loss is {loss.item() * window} at Step {step}, skipping.")
            batch = prefetcher.next()
            step += 1
            start_time = time.time()
            continue

        # Backward (累积梯度)
        if cfg.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * window
        window_step += 1

        # 窗口完成时更新
        if window_step >= window:
            if cfg.use_amp:
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            optimizer.zero_grad()

            # Detach temporal history
            if hasattr(model.temporal, 'detach_memory'):
                model.temporal.detach_memory()
            elif hasattr(model.temporal, 'detach_history'):
                model.temporal.detach_history()

            window_step = 0

        batch_time = time.time() - start_time

        # 日志
        if step % cfg.log_interval == 0:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            dist_loss = losses.get('distance', 0)
            if isinstance(dist_loss, torch.Tensor):
                dist_loss = dist_loss.item()
            depth_loss = losses.get('depth', 0)
            if isinstance(depth_loss, torch.Tensor):
                depth_loss = depth_loss.item()

            print(f"Epoch {epoch} Step {step} W:{window_step}/{window} "
                  f"Loss: {loss.item() * window:.4f} "
                  f"Focal: {losses.get('focal', 0):.4f} Dice: {losses.get('dice', 0):.4f} "
                  f"Dist: {dist_loss:.4f} Depth: {depth_loss:.4f} "
                  f"GPU: {mem:.2f}GB", flush=True)

        batch = prefetcher.next()
        step += 1
        start_time = time.time()

    # 处理最后不完整的窗口
    if window_step > 0:
        if cfg.use_amp:
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(step, 1)


# ==================== Classic TBPTT 训练 ====================

def train_one_epoch_classic_tbptt(model, loader, optimizer, scaler, loss_fn, epoch, cfg, window=3):
    """
    Classic TBPTT 训练 - 高显存版

    累积多帧 loss 后一次性 backward
    梯度可回传到历史帧
    """
    model.train()
    total_loss = 0
    device = next(model.parameters()).device

    print(f"  [Classic TBPTT] 窗口大小: {window} (高显存警告!)")
    prefetcher = FP16DataPrefetcher(loader, device)
    batch = prefetcher.next()

    step = 0
    window_step = 0
    accumulated_loss = torch.tensor(0.0, device=device)
    start_time = time.time()
    last_scene_id = None

    while batch is not None:
        data_time = time.time() - start_time

        # 场景切换检测
        scene_id = batch.get('scene_id', None)
        timestamp = batch.get('timestamp', None)

        scene_changed = False
        if scene_id is not None and last_scene_id is not None:
            current_scene = scene_id[0] if isinstance(scene_id, (list, tuple)) else scene_id
            if current_scene != last_scene_id:
                scene_changed = True
                print(f"  [Scene Switch] {last_scene_id} -> {current_scene}")

        if scene_id is not None:
            last_scene_id = scene_id[0] if isinstance(scene_id, (list, tuple)) else scene_id

        # 场景切换时处理累积的 loss
        if scene_changed and window_step > 0 and accumulated_loss.requires_grad:
            if cfg.use_amp:
                scaler.scale(accumulated_loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                accumulated_loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            optimizer.zero_grad()
            model.reset_temporal()
            accumulated_loss = torch.tensor(0.0, device=device)
            window_step = 0

        # Forward
        with autocast('cuda', enabled=cfg.use_amp):
            outputs = model(
                batch['images'],
                batch.get('ego_motion'),
                batch.get('ego_pose'),
                timestamp=timestamp,
                scene_id=scene_id,
                intrinsics=batch.get('intrinsics'),
                extrinsics=batch.get('extrinsics'),
            )
            losses = loss_fn(outputs, batch)
            loss = losses['total']

        # NaN 检查
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Loss is {loss.item()} at Step {step}, skipping.")
            batch = prefetcher.next()
            step += 1
            start_time = time.time()
            continue

        # 累积 loss (保留计算图!)
        accumulated_loss = accumulated_loss + loss
        window_step += 1
        total_loss += loss.item()

        # 窗口完成时 backward
        if window_step >= window:
            if cfg.use_amp:
                scaler.scale(accumulated_loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                accumulated_loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            optimizer.zero_grad()

            # Detach temporal history
            if hasattr(model.temporal, 'detach_history'):
                model.temporal.detach_history()
            elif hasattr(model.temporal, 'detach_memory'):
                model.temporal.detach_memory()

            accumulated_loss = torch.tensor(0.0, device=device)
            window_step = 0

        batch_time = time.time() - start_time

        # 日志
        if step % cfg.log_interval == 0:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            dist_loss = losses.get('distance', 0)
            if isinstance(dist_loss, torch.Tensor):
                dist_loss = dist_loss.item()
            depth_loss = losses.get('depth', 0)
            if isinstance(depth_loss, torch.Tensor):
                depth_loss = depth_loss.item()

            print(f"Epoch {epoch} Step {step} W:{window_step}/{window} "
                  f"Loss: {loss.item():.4f} "
                  f"Focal: {losses.get('focal', 0):.4f} Dice: {losses.get('dice', 0):.4f} "
                  f"Dist: {dist_loss:.4f} Depth: {depth_loss:.4f} "
                  f"GPU: {mem:.2f}GB", flush=True)

        batch = prefetcher.next()
        step += 1
        start_time = time.time()

    # 处理最后不完整的窗口
    if window_step > 0 and accumulated_loss.requires_grad:
        if cfg.use_amp:
            scaler.scale(accumulated_loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            accumulated_loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(step, 1)


# ==================== 验证 ====================

@torch.no_grad()
def validate(model, loader, loss_fn, cfg):
    """验证"""
    model.eval()
    total_loss = 0
    device = next(model.parameters()).device

    for batch in loader:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        with autocast('cuda', enabled=cfg.use_amp):
            outputs = model(batch['images'])
            losses = loss_fn(outputs, batch)

        total_loss += losses['total'].item()

    return total_loss / len(loader)


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Train OccNetV3 with Temporal Fusion')

    # 数据
    parser.add_argument('--dataset', type=str, default='./data', help='数据集根目录')

    # 时序训练模式
    parser.add_argument('--mode', type=str, default='memory_cell',
                        choices=['memory_cell', 'grad_accum', 'classic'],
                        help='时序训练模式: memory_cell(推荐), grad_accum, classic')
    parser.add_argument('--window', type=int, default=3,
                        help='窗口大小 (grad_accum/classic 模式使用)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--amp', action='store_true', default=None)
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.add_argument('--grad-clip', type=float, default=None)
    parser.add_argument('--num-workers', type=int, default=None)

    # 其他
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log-interval', type=int, default=None)

    args = parser.parse_args()

    # 更新 config
    config.data_dir = args.dataset

    # 根据模式配置时序融合
    if args.mode == 'memory_cell':
        config.use_memory_cell = True
        config.use_coarse_only_tbptt = False
    else:
        config.use_memory_cell = False
        config.use_coarse_only_tbptt = False

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.max_epochs = args.epochs
    if args.amp is not None:
        config.use_amp = args.amp
    if args.lr:
        config.lr = args.lr
    if args.grad_clip:
        config.grad_clip = args.grad_clip
    if args.num_workers:
        config.num_workers = args.num_workers
    if args.save_dir:
        config.save_dir = args.save_dir
    else:
        config.save_dir = f'./checkpoints_{args.mode}'
    if args.log_interval:
        config.log_interval = args.log_interval

    os.makedirs(config.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型
    model = build_model(config).to(device)
    loss_fn = OccLoss(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    scaler = GradScaler('cuda', enabled=config.use_amp)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    train_loader = build_dataloader(config, 'train')
    val_loader = build_dataloader(config, 'val')

    # 打印配置
    print("=" * 60)
    print("OccNetV3 Temporal Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"AMP: {config.use_amp}")
    print(f"Mode: {args.mode}")
    if args.mode != 'memory_cell':
        print(f"Window: {args.window}")
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print("=" * 60)

    # 模式说明
    print("\n训练模式说明:")
    if args.mode == 'memory_cell':
        print("  [Memory Cell] (推荐)")
        print("  - ConvGRU 压缩时序信息到单个 memory state")
        print("  - 显存: 无额外增长")
        print("  - 参考: BEVFormer v2, StreamPETR")
    elif args.mode == 'grad_accum':
        print("  [Gradient Accumulation]")
        print(f"  - 累积 {args.window} 帧梯度后更新")
        print("  - 显存: 无额外增长")
        print("  - 等效于更大 batch size")
    else:
        print("  [Classic TBPTT]")
        print(f"  - 累积 {args.window} 帧 loss 后 backward")
        print("  - 显存: 随帧数线性增长!")
        print("  - 如 OOM, 请切换到 memory_cell 模式")
    print()

    best_loss = float('inf')

    for epoch in range(start_epoch, config.max_epochs):
        print(f"\nStarting epoch {epoch}...", flush=True)

        # 学习率预热
        if epoch < config.warmup_epochs:
            lr_scale = (epoch + 1) / config.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = config.lr * lr_scale

        # 重置时序状态
        model.reset_temporal()

        # 训练
        if args.mode == 'memory_cell':
            train_loss = train_one_epoch_memory_cell(
                model, train_loader, optimizer, scaler, loss_fn, epoch, config
            )
        elif args.mode == 'grad_accum':
            train_loss = train_one_epoch_grad_accum(
                model, train_loader, optimizer, scaler, loss_fn, epoch, config, args.window
            )
        else:
            train_loss = train_one_epoch_classic_tbptt(
                model, train_loader, optimizer, scaler, loss_fn, epoch, config, args.window
            )

        if epoch >= config.warmup_epochs:
            scheduler.step()

        print(f"Epoch {epoch} Train Loss: {train_loss:.4f}", flush=True)

        # 验证
        if (epoch + 1) % config.eval_interval == 0:
            model.reset_temporal()
            val_loss = validate(model, val_loader, loss_fn, config)
            print(f"Epoch {epoch} Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': val_loss,
                    'mode': args.mode,
                }, os.path.join(config.save_dir, 'best.pth'))
                print(f"  Best model saved!")

        # 定期保存
        if (epoch + 1) % config.save_interval == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'mode': args.mode,
            }, os.path.join(config.save_dir, f'epoch_{epoch}.pth'))

    print("\nTraining complete!")


if __name__ == '__main__':
    main()

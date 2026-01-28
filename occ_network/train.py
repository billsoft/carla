"""
OccNetV3 训练脚本 (改进版 V2)

功能:
1. 数据加载与预处理 (支持 FP16 预取)
2. 混合精度训练 (AMP)
3. 多任务损失计算 (语义 + 几何 + 深度)
4. 梯度裁剪与优化器调度
5. 模型保存与验证
6. **[V2] 场景边界检测**: 自动检测场景切换并 reset 时序
7. **[V2] TBPTT**: 时序融合的截断反向传播
8. **[V2] 边缘感知深度损失**: 不在边缘处强制深度平滑

训练策略:
- **Optimizer**: AdamW (权重衰减 0.01)
- **Scheduler**: CosineAnnealingLR (余弦退火)
- **Loss**:
  - Focal Loss: 处理类别不平衡 (alpha=0.25, gamma=2.0)
  - Distance-Aware Loss: 近距离物体权重更高
  - Depth Supervision: 辅助深度监督 (边缘感知 L1 Log Loss)
- **Gradient Clipping**: 防止梯度爆炸 (阈值 1.0)
- **AMP**: 自动混合精度，节省显存并加速
"""
import torch
import torch.nn as nn
import os
import time
import argparse
from torch.cuda.amp import GradScaler, autocast
from configs.default import config
from models import build_model
from losses.losses import OccLoss
from data.dataset import build_dataloader, FP16DataPrefetcher

def train_one_epoch(model, loader, optimizer, scaler, loss_fn, epoch, config):
    """
    训练一个 epoch

    改进 V2:
    - 支持场景边界检测 (自动 reset 时序)
    - 支持时间戳传递 (用于时空编码)
    """
    model.train()
    total_loss = 0
    device = next(model.parameters()).device
    print(f"  Creating prefetcher...", flush=True)
    prefetcher = FP16DataPrefetcher(loader, device)
    print(f"  Getting first batch...", flush=True)
    batch = prefetcher.next()
    print(f"  Got first batch, starting training loop...", flush=True)
    step = 0
    start_time = time.time()
    last_scene_id = None

    while batch is not None:
        # 记录数据加载完成时间
        data_time = time.time() - start_time

        # [V2] 获取场景信息 (用于场景边界检测)
        scene_id = batch.get('scene_id', None)
        timestamp = batch.get('timestamp', None)

        # [V2] 场景切换检测 - 在 batch 级别也进行检查
        if scene_id is not None and last_scene_id is not None:
            # 检查 batch 中是否有场景切换
            if isinstance(scene_id, (list, tuple)):
                # 如果是 batch 形式，检查第一个样本
                current_scene = scene_id[0] if len(scene_id) > 0 else None
            else:
                current_scene = scene_id

            if current_scene != last_scene_id:
                model.reset_temporal()
                print(f"  [Scene Switch] {last_scene_id} -> {current_scene}, reset temporal")

            last_scene_id = current_scene
        elif scene_id is not None:
            if isinstance(scene_id, (list, tuple)):
                last_scene_id = scene_id[0] if len(scene_id) > 0 else None
            else:
                last_scene_id = scene_id

        optimizer.zero_grad()
        with autocast(enabled=config.use_amp):
            # [V2] 传递时间戳和场景ID
            outputs = model(
                batch['images'],
                batch.get('ego_motion'),
                batch.get('ego_pose'),
                timestamp=timestamp,
                scene_id=scene_id,
                intrinsics=batch.get('intrinsics'),
                extrinsics=batch.get('extrinsics')
            )
            losses = loss_fn(outputs, batch)
            loss = losses['total']

        # NaN 检查
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Loss is {loss.item()} at Epoch {epoch} Step {step}, skipping batch.")
            optimizer.zero_grad()
            batch = prefetcher.next()
            step += 1
            start_time = time.time()
            continue

        if config.use_amp:
            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        total_loss += loss.item()

        # 计算总时间
        batch_time = time.time() - start_time

        if step % config.log_interval == 0:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            # 增加时间统计和距离损失
            dist_loss = losses.get('distance', 0)
            if isinstance(dist_loss, torch.Tensor):
                dist_loss = dist_loss.item()
            # 深度损失
            depth_loss = losses.get('depth', 0)
            if isinstance(depth_loss, torch.Tensor):
                depth_loss = depth_loss.item()
            print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f} Focal: {losses.get('focal', 0):.4f} Dice: {losses.get('dice', 0):.4f} Dist: {dist_loss:.4f} Depth: {depth_loss:.4f} GPU Mem: {mem:.2f}GB Data: {data_time:.3f}s Batch: {batch_time:.3f}s", flush=True)

        batch = prefetcher.next()
        step += 1
        start_time = time.time()
    return total_loss / max(step, 1)

@torch.no_grad()
def validate(model, loader, loss_fn, config):
    model.eval()
    total_loss = 0
    device = next(model.parameters()).device
    for batch in loader:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        with autocast(enabled=config.use_amp):
            outputs = model(batch['images'])
            losses = loss_fn(outputs, batch)
        total_loss += losses['total'].item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description='Train OccNetV3')
    # 数据
    parser.add_argument('--dataset', type=str, default='./data', help='数据集根目录')
    # 训练
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=None, help='批量大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--amp', action='store_true', default=None, help='启用混合精度训练')
    parser.add_argument('--no-amp', dest='amp', action='store_false', help='禁用混合精度训练')
    parser.add_argument('--grad-clip', type=float, default=None, help='梯度裁剪阈值')
    parser.add_argument('--num-workers', type=int, default=None)
    # 其他参数
    parser.add_argument('--save-dir', type=str, default=None, help='保存目录')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log-interval', type=int, default=None)
    args = parser.parse_args()

    # 更新config
    config.data_dir = args.dataset
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
    if args.log_interval:
        config.log_interval = args.log_interval
    os.makedirs(config.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    loss_fn = OccLoss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    scaler = GradScaler(enabled=config.use_amp)
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    train_loader = build_dataloader(config, 'train')
    val_loader = build_dataloader(config, 'val')
    print(f"Training on {device}")
    print(f"AMP: {config.use_amp}, Checkpoint: {config.use_checkpoint}")
    print(f"Coarse-to-Fine: {config.use_coarse_to_fine}, Sparse: {config.use_sparse}")
    print(f"Positional Encoding: Unified Spherical (Equidistant Projection)")
    print(f"[V2] Depth-Aware Fusion: {getattr(config, 'use_depth_aware_fusion', True)}")
    print(f"[V2] TBPTT Steps: {getattr(config, 'tbptt_steps', 3)}")
    print(f"[V2] Dynamic Motion: {getattr(config, 'use_dynamic_motion', True)}")
    print(f"Train loader size: {len(train_loader)}, Val loader size: {len(val_loader)}", flush=True)
    best_loss = float('inf')
    for epoch in range(start_epoch, config.max_epochs):
        print(f"Starting epoch {epoch}...", flush=True)
        if epoch < config.warmup_epochs:
            lr_scale = (epoch + 1) / config.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = config.lr * lr_scale
        model.reset_temporal()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, epoch, config)
        if epoch >= config.warmup_epochs:
            scheduler.step()
        print(f"Epoch {epoch} Train Loss: {train_loss:.4f}", flush=True)
        if (epoch + 1) % config.eval_interval == 0:
            model.reset_temporal()
            val_loss = validate(model, val_loader, loss_fn, config)
            print(f"Epoch {epoch} Val Loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'loss': val_loss}, os.path.join(config.save_dir, 'best.pth'))
        if (epoch + 1) % config.save_interval == 0:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, os.path.join(config.save_dir, f'epoch_{epoch}.pth'))
    print("Training complete!")

if __name__ == '__main__':
    main()

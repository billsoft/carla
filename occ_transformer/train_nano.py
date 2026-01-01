#!/usr/bin/env python3
"""
Nano 版本 Occupancy Network 训练脚本

专为低显存环境优化 (<1.2GB)
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models import TransformerOccNetNano
from data.carla_dataset_bayer import build_dataloader
from utils.loss import MaskedWeightedCELoss, get_default_class_weights


def setup_logging(save_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger('NanoOccNet')
    logger.setLevel(logging.INFO)

    # 清除已有处理器
    logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(save_dir, 'train.log'))
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train Nano Occupancy Network')

    # 数据
    parser.add_argument('--dataset', type=str, required=True, help='数据集根目录')

    # 训练
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=4)

    # 模型配置
    parser.add_argument('--img-size', type=int, nargs=2, default=[960, 1280], help='图像尺寸 H W')

    # 训练技巧
    parser.add_argument('--amp', action='store_true', help='混合精度训练')
    parser.add_argument('--grad-clip', type=float, default=5.0, help='梯度裁剪阈值')

    # 保存
    parser.add_argument('--save-dir', type=str, default='outputs/nano_occ')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log-interval', type=int, default=20)

    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, epoch, args, logger):
    """训练一个 epoch"""
    model.train()

    total_loss = 0
    num_batches = len(dataloader)

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        images = batch['images'].to(args.device)
        occupancy = batch['occupancy'].to(args.device)

        # Mask (Deprecated) - 强制全 True
        mask = torch.ones_like(occupancy, dtype=torch.bool, device=args.device)

        # 混合精度前向传播
        with autocast(enabled=args.amp):
            outputs = model(images)
            # 使用 MaskedWeightedCELoss
            loss = criterion(outputs, occupancy.long(), mask)

        # ✅ NaN 检测: 如果 loss 是 NaN/Inf,跳过此 batch
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f'Epoch [{epoch}][{batch_idx+1}/{num_batches}] Loss is NaN/Inf, skipping batch...')
            optimizer.zero_grad()
            continue

        # 反向传播
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            
            # ✅ 梯度检测: unscale 后检查梯度是否正常
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 如果梯度爆炸,跳过此 batch
            if torch.isnan(grad_norm) or grad_norm > 100.0:
                logger.warning(f'Epoch [{epoch}][{batch_idx+1}/{num_batches}] Gradient exploded (norm={grad_norm:.2f}), skipping batch...')
                optimizer.zero_grad()
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        # 日志
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            lr = optimizer.param_groups[0]['lr']

            # 显存使用
            if args.device == 'cuda':
                mem = torch.cuda.max_memory_allocated() / 1e9
                mem_str = f'Mem: {mem:.2f}GB'
            else:
                mem_str = ''

            logger.info(
                f'Epoch [{epoch}][{batch_idx+1}/{num_batches}] '
                f'Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) '
                f'LR: {lr:.2e} {mem_str}'
            )

    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    logger.info(f'Epoch [{epoch}] done in {epoch_time:.1f}s, Loss: {avg_loss:.4f}')

    return avg_loss


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
    }, path)


def main():
    args = parse_args()

    # 保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)

    logger = setup_logging(save_dir)
    logger.info(f'Arguments: {args}')

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    if device.type == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name()}')
        logger.info(f'Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # 数据加载
    logger.info('Loading data...')

    img_size = tuple(args.img_size)  # (H, W)

    train_loader = build_dataloader(
        dataset_root=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=img_size,
        augment=True,
        shuffle=True,
    )

    logger.info(f'Train: {len(train_loader.dataset)} samples')

    # 构建 Nano Network
    logger.info('Building Nano Occupancy Network...')
    model = TransformerOccNetNano(
        num_cameras=8,
        img_size=img_size,
        embed_dim=128,
        bev_size=(50, 50),
        num_height_levels=4,
        output_grid_size=(200, 200, 16),
        use_checkpoint=True # 开启 Gradient Checkpointing
    ).to(device)

    # 参数统计
    params_summary = model.get_params_summary()
    logger.info(f'Parameters Summary:')
    for name, value in params_summary.items():
        logger.info(f'  {name}: {value:.2f}M')

    # 损失函数
    class_weights = get_default_class_weights()
    criterion = MaskedWeightedCELoss(class_weights=class_weights).to(device)
    logger.info(f"Using MaskedWeightedCELoss with class weights.")

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 学习率调度
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # 混合精度
    scaler = GradScaler(enabled=args.amp)

    # 恢复训练
    start_epoch = 0
    if args.resume:
        logger.info(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt['scheduler_state_dict']:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if args.amp and ckpt['scaler_state_dict']:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    # 训练循环
    logger.info('Starting training...')

    for epoch in range(start_epoch, args.epochs):
        # 重置显存统计
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            epoch, args, logger
        )

        # 保存检查点
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch,
            os.path.join(save_dir, f'epoch_{epoch:03d}.pth')
        )

    logger.info(f'Training done. Checkpoints saved to {save_dir}')


if __name__ == '__main__':
    main()

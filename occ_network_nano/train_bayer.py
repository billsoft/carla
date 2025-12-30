#!/usr/bin/env python3
"""
Bayer RAW Occupancy Network 训练脚本

python occ_network_nano/train_bayer.py --dataset dataset_10k --batch-size 2 --epochs 50 --device cuda --amp

专为单通道 Bayer RGGB 输入优化，数据量降低 66%。

用法:
    python train_bayer.py --dataset ../dataset_10k --epochs 50 --batch-size 4
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

from models import build_bayer_occ_net
from data.carla_dataset_bayer import build_dataloader
from utils.loss import MaskedWeightedCELoss, get_default_class_weights


def setup_logging(save_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger('BayerOccNet')
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
    parser = argparse.ArgumentParser(description='Train Bayer RAW Occupancy Network')

    # 数据
    parser.add_argument('--dataset', type=str, required=True, help='数据集根目录')

    # 训练
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--num-workers', type=int, default=4)

    # 模型配置
    parser.add_argument('--img-size', type=int, nargs=2, default=[384, 640], help='图像尺寸 H W')
    parser.add_argument('--width-mult', type=float, default=1.0, help='Backbone 宽度乘数')

    # 训练技巧
    parser.add_argument('--amp', action='store_true', help='混合精度训练')
    parser.add_argument('--grad-clip', type=float, default=35.0)

    # 保存
    parser.add_argument('--save-dir', type=str, default='outputs/bayer_raw')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log-interval', type=int, default=20)

    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


# 使用完整的 BayerOccNet 网络
# 包括: Backbone → FPN → View Transformer → BEV Encoder → Occ Decoder


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, epoch, args, logger):
    """训练一个 epoch"""
    model.train()

    total_loss = 0
    num_batches = len(dataloader)

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        images = batch['images'].to(args.device)
        occupancy = batch['occupancy'].to(args.device)
        mask = batch['mask'].to(args.device)

        # 混合精度前向传播
        with autocast(enabled=args.amp):
            outputs = model(images)
            # 使用 MaskedWeightedCELoss
            loss = criterion(outputs, occupancy.long(), mask)

        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

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
                mem_str = f'Mem: {mem:.1f}GB'
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
        'scaler_state_dict': scaler.state_dict(),
    }, path)


def main():
    args = parse_args()

    # 添加 num_classes 参数
    args.num_classes = 18

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

    # 构建完整的 Bayer Occupancy Network
    logger.info('Building complete Bayer Occupancy Network...')
    model = build_bayer_occ_net(
        num_classes=18,
        grid_size=(200, 200, 16),
        img_size=tuple(args.img_size),
        backbone_width_mult=args.width_mult,
        fpn_channels=128,
        bev_size=(100, 100),
        hidden_channels=64
    ).to(device)

    # 参数统计
    params_summary = model.get_params_summary()
    logger.info(f'Parameters Summary:')
    logger.info(f'  Backbone:         {params_summary["backbone"]:.2f}M')
    logger.info(f'  FPN:              {params_summary["fpn"]:.2f}M')
    logger.info(f'  View Transformer: {params_summary["view_transformer"]:.2f}M')
    logger.info(f'  BEV Encoder:      {params_summary["bev_encoder"]:.2f}M')
    logger.info(f'  Occ Decoder:      {params_summary["occ_decoder"]:.2f}M')
    logger.info(f'  Total:            {params_summary["total"]:.2f}M')

    # 损失函数
    # 使用带权重和掩码的损失函数
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

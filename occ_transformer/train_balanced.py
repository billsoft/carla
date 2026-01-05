#!/usr/bin/env python3
"""
Balanced 版本 Occupancy Network 训练脚本

平衡精度和显存，支持 BS=2
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

from models import TransformerOccNetBalanced
from data.carla_dataset_bayer import build_dataloader
from utils.loss import (
    MaskedWeightedCELoss, 
    FocalLoss, 
    LovaszSoftmaxLoss, 
    CombinedLoss, 
    OHEMLoss,
    DistanceAwareLoss,
    get_default_class_weights,
    get_moving_class_weights
)




def setup_logging(save_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger('BalancedOccNet')
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
    parser = argparse.ArgumentParser(description='Train Balanced Occupancy Network')

    # 数据
    parser.add_argument('--dataset', type=str, default='dataset_10k', help='数据集根目录')

    # 训练
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=4)

    # 模型配置
    parser.add_argument('--img-size', type=int, nargs=2, default=[960, 1280], help='图像尺寸 H W')
    parser.add_argument('--embed-dim', type=int, default=192, help='Embedding 维度')
    
    # 训练技巧
    parser.add_argument('--loss-type', type=str, default='ce', 
                       help='损失函数类型: ce, focal, lovasz, combined, ohem, distance, distance_focal')
    parser.add_argument('--gamma', type=float, default=2.5, help='Focal Loss gamma (default: 2.5)')
    parser.add_argument('--class-weight-mode', type=str, default='default',
                       choices=['default', 'moving'],
                       help='类别权重模式: default (基础), moving (针对移动物体增强)')
    parser.add_argument('--amp', action='store_true', default=True, help='混合精度训练')
    parser.add_argument('--grad-clip', type=float, default=5.0, help='梯度裁剪阈值')
    parser.add_argument('--use-checkpoint', action='store_true', default=True, help='使用梯度检查点')

    # 保存
    parser.add_argument('--save-dir', type=str, default='outputs/balanced_occ')
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
        # mask = batch['mask'].to(args.device)
        mask = torch.ones_like(occupancy, dtype=torch.bool, device=args.device)

        # 混合精度前向传播
        with autocast(enabled=args.amp):
            outputs = model(images)
            # 使用 MaskedWeightedCELoss
            loss = criterion(outputs, occupancy.long(), mask)

        # ✅ NaN 检测
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f'Epoch [{epoch}][{batch_idx+1}/{num_batches}] Loss is NaN/Inf, skipping batch...')
            optimizer.zero_grad()
            continue

        # 反向传播
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            
            # ✅ 梯度检测
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

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
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
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
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning('⚠️  CUDA requested but not available! Fallback to CPU.')
        device = torch.device('cpu')
    else:
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
        shuffle=True
    )
    
    logger.info(f'Data loaded. Steps per epoch: {len(train_loader)}')

    # 模型构建
    logger.info('Building Balanced-Pro model...')
    # Balanced-Pro 优化版配置 (8GB显存优化)：
    # embed_dim=256, encoder_layers=10, decoder_layers=6, 
    # bev_size=(100, 100), num_height_levels=16, num_deform_points=8
    model = TransformerOccNetBalanced(
        num_cameras=8,
        img_size=img_size,
        embed_dim=256,         # Fixed 256
        encoder_layers=10,     # 5 -> 10
        decoder_layers=6,      # 4 -> 6
        num_heads=8,
        bev_size=(100, 100),   # 50x50 -> 100x100 (分辨率提升)
        num_height_levels=16,  # 8 -> 16 (高度精度提升)
        num_deform_points=8,   # 6 -> 8
        output_grid_size=(200, 200, 16),
        use_checkpoint=args.use_checkpoint
    ).to(device)
    
    # 打印参数量
    params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model params: {params/1e6:.2f}M')

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 混合精度
    scaler = GradScaler(enabled=args.amp)

    # 恢复训练逻辑
    start_epoch = 1
    last_epoch = -1  # 用于 scheduler 的 last_epoch
    
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f'Loading checkpoint from {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 1. 无论何种情况，都加载模型权重 (保证权重不丢失)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model weights loaded.")

            ckpt_epoch = checkpoint['epoch']
            
            # 2. 判断是 "断点续训" 还是 "基于旧权重重新训练/微调"
            if ckpt_epoch < args.epochs:
                # 情况 A: 断点续训 (如从 epoch 10 恢复，目标 50; 或从 50 恢复，目标 100)
                logger.info(f"Resuming training context from epoch {ckpt_epoch} to {args.epochs}...")
                
                start_epoch = ckpt_epoch + 1
                
                # 加载优化器状态 (保持动量)
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("Optimizer state loaded.")
                
                # 加载 Scaler
                if 'scaler_state_dict' in checkpoint and checkpoint.get('scaler_state_dict'):
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    logger.info("Scaler state loaded.")
                    
                # 计算 last_epoch 以恢复调度器进度
                # 注意: last_epoch 是步数 (steps)，不是 epoch 数
                # -1 表示从头开始，否则应为 (start_epoch - 1) * steps_per_epoch - 1
                last_epoch = (start_epoch - 1) * len(train_loader) - 1
                
            else:
                # 情况 B: 旧训练已完成 (或 checkpoint epoch >= args.epochs)
                # 用户希望基于这些权重重新开始训练 (Fine-tuning 或 Restart)
                logger.info(f"Checkpoint epoch ({ckpt_epoch}) >= Target epochs ({args.epochs}).")
                logger.info("Starting FRESH training (Epoch 1) using loaded weights.")
                
                start_epoch = 1
                last_epoch = -1
                # 不加载优化器和 Scaler，重置为初始状态
        else:
            logger.warning(f"Checkpoint file not found: {args.resume}. Training from scratch.")

    # 学习率调度 (在恢复逻辑之后初始化，以支持正确的 last_epoch)
    # OneCycleLR 需要知道总步数和当前步数 (last_epoch)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3,
        last_epoch=last_epoch 
    )
    logger.info(f"Scheduler initialized. Start Epoch: {start_epoch}, Last Step: {last_epoch}")

    # 损失函数
    # 获取类别权重
    if args.class_weight_mode == 'moving':
        logger.info("Using Moving Objects Enhanced Class Weights (Aggressive V2)")
        class_weights = get_moving_class_weights()
    else:
        logger.info("Using Default Class Weights (Aggressive V2)")
        class_weights = get_default_class_weights()
        
    class_weights = torch.tensor(class_weights).float().to(device)
    
    if args.loss_type == 'ce':
        criterion = MaskedWeightedCELoss(class_weights=class_weights)
    elif args.loss_type == 'focal':
        criterion = FocalLoss(class_weights=class_weights, gamma=args.gamma)
    elif args.loss_type == 'lovasz':
        criterion = LovaszSoftmaxLoss()
    elif args.loss_type == 'combined':
        criterion = CombinedLoss(
            ce_weight=1.0, 
            lovasz_weight=1.0, 
            class_weights=class_weights
        )
    elif args.loss_type == 'ohem':
        criterion = OHEMLoss(class_weights=class_weights, thresh=0.7)
    elif args.loss_type == 'distance':
        criterion = DistanceAwareLoss(class_weights=class_weights)
    elif args.loss_type == 'distance_focal':
        logger.info("Using Distance Aware + Focal Loss")
        criterion = DistanceAwareLoss(class_weights=class_weights)
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
        
    logger.info(f"Using Loss Function: {args.loss_type}")

    # 训练循环
    logger.info('Start training...')
    
    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, epoch, args, logger
        )

        # 保存检查点
        if epoch % 5 == 0 or epoch == args.epochs:
            save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, save_path)
            logger.info(f'Saved checkpoint to {save_path}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, save_path)
            logger.info(f'Saved best model with loss {best_loss:.4f}')

    logger.info('Training completed.')


if __name__ == '__main__':
    main()

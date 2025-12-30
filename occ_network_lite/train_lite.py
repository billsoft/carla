#!/usr/bin/env python3
# train_lite.py
"""
轻量级 Occupancy Network 训练脚本

针对显存受限场景优化:
- MobileNetV2 backbone
- 降低分辨率和特征维度
- 支持 8GB 显存 GPU

用法:
    python train_lite.py --data_root ../dataset_output --epochs 24 --batch_size 2
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.occ_network_lite import OccupancyNetworkLite, LiteConfig, build_lite_model
from datasets.carla_occ_dataset import build_dataloader
from losses.occ_loss import CombinedOccLoss
from utils.metrics import OccupancyMetrics


def setup_logging(save_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(save_dir, exist_ok=True)
    
    logger = logging.getLogger('OccNetLite')
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
    parser = argparse.ArgumentParser(description='Train Lite Occupancy Network')
    
    # 数据
    parser.add_argument('--data_root', type=str, required=True)
    
    # 训练
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 模型配置
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--bev_size', type=int, default=100)
    parser.add_argument('--num_heights', type=int, default=8)
    parser.add_argument('--img_h', type=int, default=256)
    parser.add_argument('--img_w', type=int, default=448)
    
    # 训练技巧
    parser.add_argument('--amp', action='store_true', help='混合精度')
    parser.add_argument('--grad_clip', type=float, default=35.0)
    parser.add_argument('--accumulate', type=int, default=1, help='梯度累积步数')
    
    # 保存
    parser.add_argument('--save_dir', type=str, default='checkpoints_lite')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--val_interval', type=int, default=1)
    
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


def train_one_epoch(
    model, dataloader, criterion, optimizer, scheduler, scaler,
    epoch, args, logger, accumulate_steps=1
):
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0
    num_batches = len(dataloader)
    
    optimizer.zero_grad()
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['images'].to(args.device)
        occupancy = batch['occupancy'].to(args.device)
        mask = batch['mask'].to(args.device)
        extrinsics = batch['extrinsics'].to(args.device)

        # 混合精度前向传播
        with autocast(enabled=args.amp):
            outputs = model(images, extrinsics=extrinsics)  # 传递相机外参
            occ_logits = outputs['occ_logits']
            loss, loss_dict = criterion(occ_logits, occupancy, mask)
            loss = loss / accumulate_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积
        if (batch_idx + 1) % accumulate_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
        
        total_loss += loss.item() * accumulate_steps
        
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
                f'Loss: {loss.item()*accumulate_steps:.4f} (avg: {avg_loss:.4f}) '
                f'LR: {lr:.2e} {mem_str}'
            )
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    logger.info(f'Epoch [{epoch}] done in {epoch_time:.1f}s, Loss: {avg_loss:.4f}')
    
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, criterion, args, logger):
    """验证"""
    model.eval()
    
    total_loss = 0
    metrics = OccupancyMetrics(num_classes=18)
    
    for batch in dataloader:
        images = batch['images'].to(args.device)
        occupancy = batch['occupancy'].to(args.device)
        mask = batch['mask'].to(args.device)
        extrinsics = batch['extrinsics'].to(args.device)

        with autocast(enabled=args.amp):
            outputs = model(images, extrinsics=extrinsics)  # 传递相机外参
            occ_logits = outputs['occ_logits']
            loss, _ = criterion(occ_logits, occupancy, mask)
        
        total_loss += loss.item()
        
        occ_pred = occ_logits.argmax(dim=1)
        metrics.update(occ_pred, occupancy, mask)
    
    avg_loss = total_loss / len(dataloader)
    results = metrics.compute()
    
    logger.info(f'Validation: Loss={avg_loss:.4f}, mIoU={results["miou"]:.4f}, Acc={results["overall_acc"]:.4f}')
    
    return avg_loss, results['miou']


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_miou, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
        'best_miou': best_miou,
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
    
    # 使用轻量级配置的网格大小
    grid_size = (args.bev_size, args.bev_size, args.num_heights)
    img_size = (args.img_h, args.img_w)
    
    train_loader = build_dataloader(
        data_root=args.data_root,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=img_size,
        grid_size=grid_size,
        augment=True,
    )
    
    val_loader = build_dataloader(
        data_root=args.data_root,
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=img_size,
        grid_size=grid_size,
        augment=False,
    )
    
    logger.info(f'Train: {len(train_loader.dataset)} samples')
    logger.info(f'Val: {len(val_loader.dataset)} samples')
    
    # 构建轻量级模型
    logger.info('Building lite model...')
    model = OccupancyNetworkLite(
        num_cameras=8,
        img_size=img_size,
        embed_dim=args.embed_dim,
        bev_h=args.bev_size,
        bev_w=args.bev_size,
        num_classes=18,
        num_heights=args.num_heights,
    ).to(device)
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable')
    
    # 损失函数
    criterion = CombinedOccLoss(num_classes=18).to(device)
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度
    total_steps = args.epochs * len(train_loader) // args.accumulate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # 混合精度
    scaler = GradScaler(enabled=args.amp)
    
    # 恢复训练
    start_epoch = 0
    best_miou = 0.0
    
    if args.resume:
        logger.info(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt['scheduler_state_dict']:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_miou = ckpt.get('best_miou', 0.0)
    
    # 训练循环
    logger.info('Starting training...')
    
    for epoch in range(start_epoch, args.epochs):
        # 重置显存统计
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            epoch, args, logger, args.accumulate
        )
        
        # 验证
        if (epoch + 1) % args.val_interval == 0:
            val_loss, val_miou = validate(model, val_loader, criterion, args, logger)
            
            if val_miou > best_miou:
                best_miou = val_miou
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, best_miou,
                    os.path.join(save_dir, 'best.pth')
                )
                logger.info(f'New best: mIoU={best_miou:.4f}')
        
        # 保存最新
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch, best_miou,
            os.path.join(save_dir, 'last.pth')
        )
    
    logger.info(f'Training done. Best mIoU: {best_miou:.4f}')


if __name__ == '__main__':
    main()

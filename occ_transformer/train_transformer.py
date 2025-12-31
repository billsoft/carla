#!/usr/bin/env python3
# train_transformer.py
"""
Transformer Occupancy Network 训练脚本

用法:
    python train_transformer.py --dataset /path/to/dataset --epochs 100
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

sys.path.insert(0, str(Path(__file__).parent))

from models.transformer_occ import build_unified_transformer_occ
from data.carla_dataset_bayer import build_dataloader
from utils.loss import MaskedWeightedCELoss, get_default_class_weights


def parse_args():
    parser = argparse.ArgumentParser(description='Transformer OccNet Training')
    
    # 数据
    parser.add_argument('--dataset', type=str, required=True, help='数据集根目录')
    parser.add_argument('--img-size', type=int, nargs=2, default=[960, 1280], help='图像尺寸 H W')
    
    # 模型
    parser.add_argument('--model-type', type=str, default='lite', choices=['standard', 'lite'])
    parser.add_argument('--embed-dim', type=int, default=256, help='嵌入维度')
    parser.add_argument('--encoder-layers', type=int, default=4, help='编码器层数')
    parser.add_argument('--decoder-layers', type=int, default=2, help='解码器层数')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch 大小')
    parser.add_argument('--num-classes', type=int, default=18, help='类别数')
    
    # 训练
    parser.add_argument('--batch-size', type=int, default=1, help='Batch 大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup 轮数')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--amp', action='store_true', default=True, help='混合精度训练')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载工作进程数')
    parser.add_argument('--log-interval', type=int, default=20, help='日志间隔')
    parser.add_argument('--save-dir', type=str, default='checkpoints_transformer', help='保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点')
    
    return parser.parse_args()


def setup_logging(save_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(save_dir, exist_ok=True)
    
    logger = logging.getLogger('TransformerOccNet')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(os.path.join(save_dir, 'train.log'))
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def get_lr_scheduler(optimizer, args, steps_per_epoch):
    """获取学习率调度器"""
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            # 线性 warmup
            return step / warmup_steps
        else:
            # 余弦退火
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model, dataloader, criterion, optimizer, scheduler, scaler,
    epoch, args, logger
):
    """训练一个 epoch"""
    model.train()
    
    # BatchNorm 处理
    if args.batch_size == 1:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
                
    total_loss = 0
    num_batches = len(dataloader)
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['images'].to(args.device)
        occupancy = batch['occupancy'].to(args.device)
        mask = batch['mask'].to(args.device)
        
        # 前向传播
        with autocast(enabled=args.amp):
            outputs = model(images)
            loss = criterion(outputs, occupancy.long(), mask)
            
        # 检查 NaN
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f'Epoch [{epoch}][{batch_idx+1}/{num_batches}] Loss is NaN/Inf, skipping...')
            optimizer.zero_grad()
            continue
            
        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # 梯度裁剪
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        if torch.isnan(grad_norm) or grad_norm > 100.0:
            logger.warning(f'Epoch [{epoch}][{batch_idx+1}/{num_batches}] Gradient exploded, skipping...')
            optimizer.zero_grad()
            scaler.update()
            continue
            
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item()
        
        # 日志
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            lr = optimizer.param_groups[0]['lr']
            mem_str = ''
            if args.device == 'cuda':
                mem = torch.cuda.max_memory_allocated() / 1e9
                mem_str = f'Mem: {mem:.1f}GB'
                
            logger.info(
                f'Epoch [{epoch}][{batch_idx+1}/{num_batches}] '
                f'Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) '
                f'LR: {lr:.2e} {mem_str}'
            )
            
    epoch_time = time.time() - start_time
    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f'Epoch [{epoch}] done in {epoch_time:.1f}s, Loss: {avg_loss:.4f}')
    
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, path):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
    }, path)


def main():
    args = parse_args()
    
    # 设置
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    logger = setup_logging(save_dir)
    
    logger.info("=" * 70)
    logger.info("Transformer Occupancy Network Training".center(70))
    logger.info("=" * 70)
    logger.info(f'\nArguments: {args}')
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    if device.type == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name()}')
        logger.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
        
    # 数据
    logger.info('\nLoading data...')
    img_size = tuple(args.img_size)
    train_loader = build_dataloader(
        dataset_root=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=img_size,
        augment=True,
        shuffle=True
    )
    logger.info(f'Train: {len(train_loader.dataset)} samples')
    
    # 模型
    logger.info(f'\nBuilding {args.model_type} Transformer OccNet...')
    
    if args.model_type == 'lite':
        model = build_unified_transformer_occ(
            lite=True,
            num_classes=args.num_classes,
            img_size=img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers
        ).to(device)
    else:
        model = build_unified_transformer_occ(
            lite=False,
            num_classes=args.num_classes,
            img_size=img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers
        ).to(device)
        
    # 参数统计
    params = model.get_params_summary()
    logger.info('Parameters:')
    for name, value in params.items():
        logger.info(f'  {name}: {value:.2f}M')
        
    # 损失函数
    class_weights = get_default_class_weights()
    criterion = MaskedWeightedCELoss(class_weights=class_weights).to(device)
    logger.info('Loss: MaskedWeightedCELoss')
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = get_lr_scheduler(optimizer, args, len(train_loader))
    
    # AMP
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
        
    # 训练
    logger.info('\nStarting training...')
    logger.info("=" * 70)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            
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
        
        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch,
                os.path.join(save_dir, 'best.pth')
            )
            logger.info(f'New best loss: {best_loss:.4f}')
            
    logger.info("=" * 70)
    logger.info(f'Training done. Checkpoints saved to {save_dir}')
    logger.info(f'Best loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()

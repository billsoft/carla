#!/usr/bin/env python3
# train.py
"""
Occupancy Network 训练脚本

用法:
    python train.py --data_root /path/to/dataset --epochs 24
    python train.py --data_root /path/to/dataset --resume checkpoints/last.pth
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
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.default_config import get_config, Config
from models.occ_network import OccupancyNetwork, build_occ_network
from datasets.carla_occ_dataset import build_dataloader
from losses.occ_loss import CombinedOccLoss, build_loss
from utils.metrics import OccupancyMetrics


def setup_logging(save_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(save_dir, exist_ok=True)
    
    logger = logging.getLogger('OccNet')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(os.path.join(save_dir, 'train.log'))
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Occupancy Network')
    
    # 数据
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=24,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101'],
                        help='Backbone 类型')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='特征维度')
    parser.add_argument('--bev_size', type=int, default=200,
                        help='BEV 网格大小')
    parser.add_argument('--num_heights', type=int, default=16,
                        help='高度层数')
    
    # 训练技巧
    parser.add_argument('--amp', action='store_true',
                        help='使用混合精度训练')
    parser.add_argument('--grad_clip', type=float, default=35.0,
                        help='梯度裁剪')
    
    # 保存和恢复
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    
    # 日志
    parser.add_argument('--log_interval', type=int, default=50,
                        help='日志打印间隔')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='验证间隔（epoch）')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    
    return parser.parse_args()


def build_model(args) -> nn.Module:
    """构建模型"""
    model = OccupancyNetwork(
        num_cameras=8,
        img_size=(384, 640),
        backbone_type=args.backbone,
        backbone_pretrained=True,
        embed_dim=args.embed_dim,
        num_heads=8,
        num_transformer_layers=6,
        bev_h=args.bev_size,
        bev_w=args.bev_size,
        num_classes=18,
        num_heights=args.num_heights,
        full_grid_size=(500, 500, 40),
    )
    
    return model


def build_optimizer(model: nn.Module, args) -> optim.Optimizer:
    """构建优化器"""
    # 分组参数：backbone 使用较小学习率
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': other_params, 'lr': args.lr},
    ]
    
    optimizer = optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    return optimizer


def build_scheduler(optimizer: optim.Optimizer, args, steps_per_epoch: int):
    """构建学习率调度器"""
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = steps_per_epoch  # 1 epoch warmup
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return scheduler


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    args,
    logger: logging.Logger,
    writer: SummaryWriter,
):
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0
    total_ce_loss = 0
    total_lovasz_loss = 0
    num_batches = len(dataloader)
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # 数据移到 GPU
        images = batch['images'].to(args.device)
        occupancy = batch['occupancy'].to(args.device)
        mask = batch['mask'].to(args.device)
        
        # 前向传播
        optimizer.zero_grad()
        
        if args.amp:
            with autocast():
                outputs = model(images)
                occ_logits = outputs['occ_logits']
                loss, loss_dict = criterion(occ_logits, occupancy, mask)
        else:
            outputs = model(images)
            occ_logits = outputs['occ_logits']
            loss, loss_dict = criterion(occ_logits, occupancy, mask)
        
        # 反向传播
        if args.amp:
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        total_ce_loss += loss_dict['ce_loss']
        total_lovasz_loss += loss_dict['lovasz_loss']
        
        # 日志
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            lr = optimizer.param_groups[1]['lr']
            
            logger.info(
                f'Epoch [{epoch}][{batch_idx + 1}/{num_batches}] '
                f'Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) '
                f'LR: {lr:.2e}'
            )
            
            # TensorBoard
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/ce_loss', loss_dict['ce_loss'], global_step)
            writer.add_scalar('train/lovasz_loss', loss_dict['lovasz_loss'], global_step)
            writer.add_scalar('train/lr', lr, global_step)
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    
    logger.info(
        f'Epoch [{epoch}] completed in {epoch_time:.1f}s, '
        f'Avg Loss: {avg_loss:.4f}'
    )
    
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    epoch: int,
    args,
    logger: logging.Logger,
    writer: SummaryWriter,
):
    """验证"""
    model.eval()
    
    total_loss = 0
    num_batches = len(dataloader)
    
    metrics = OccupancyMetrics(num_classes=18)
    
    for batch in dataloader:
        images = batch['images'].to(args.device)
        occupancy = batch['occupancy'].to(args.device)
        mask = batch['mask'].to(args.device)
        
        # 前向传播
        outputs = model(images)
        occ_logits = outputs['occ_logits']
        
        loss, _ = criterion(occ_logits, occupancy, mask)
        total_loss += loss.item()
        
        # 预测
        occ_pred = occ_logits.argmax(dim=1)
        
        # 更新指标
        metrics.update(occ_pred, occupancy, mask)
    
    avg_loss = total_loss / num_batches
    results = metrics.compute()
    
    logger.info(f'\nValidation Epoch [{epoch}]:')
    logger.info(f'  Loss: {avg_loss:.4f}')
    logger.info(f'  mIoU: {results["miou"]:.4f}')
    logger.info(f'  Accuracy: {results["overall_acc"]:.4f}')
    
    # TensorBoard
    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/miou', results['miou'], epoch)
    writer.add_scalar('val/accuracy', results['overall_acc'], epoch)
    
    return avg_loss, results['miou']


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_miou: float,
    save_path: str,
):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_miou': best_miou,
    }
    
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    checkpoint_path: str,
):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('best_miou', 0.0)


def main():
    args = parse_args()
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(save_dir)
    logger.info(f'Arguments: {args}')
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 数据加载
    logger.info('Building dataloaders...')
    train_loader = build_dataloader(
        data_root=args.data_root,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(384, 640),
        grid_size=(args.bev_size, args.bev_size, args.num_heights),
        augment=True,
    )
    
    val_loader = build_dataloader(
        data_root=args.data_root,
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(384, 640),
        grid_size=(args.bev_size, args.bev_size, args.num_heights),
        augment=False,
    )
    
    logger.info(f'Train samples: {len(train_loader.dataset)}')
    logger.info(f'Val samples: {len(val_loader.dataset)}')
    
    # 构建模型
    logger.info('Building model...')
    model = build_model(args)
    model = model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {total_params / 1e6:.2f}M')
    logger.info(f'Trainable parameters: {trainable_params / 1e6:.2f}M')
    
    # 损失函数
    criterion = CombinedOccLoss(num_classes=18).to(device)
    
    # 优化器
    optimizer = build_optimizer(model, args)
    
    # 学习率调度
    scheduler = build_scheduler(optimizer, args, len(train_loader))
    
    # 混合精度
    scaler = GradScaler(enabled=args.amp)
    
    # 恢复训练
    start_epoch = 0
    best_miou = 0.0
    
    if args.resume:
        logger.info(f'Resuming from {args.resume}')
        start_epoch, best_miou = load_checkpoint(
            model, optimizer, scheduler, scaler, args.resume
        )
        start_epoch += 1
        logger.info(f'Resumed from epoch {start_epoch}, best mIoU: {best_miou:.4f}')
    
    # 训练循环
    logger.info('Starting training...')
    
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, epoch, args, logger, writer
        )
        
        # 验证
        if (epoch + 1) % args.val_interval == 0:
            val_loss, val_miou = validate(
                model, val_loader, criterion, epoch, args, logger, writer
            )
            
            # 保存最佳模型
            if val_miou > best_miou:
                best_miou = val_miou
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, best_miou,
                    os.path.join(save_dir, 'best.pth')
                )
                logger.info(f'New best model saved with mIoU: {best_miou:.4f}')
        
        # 定期保存
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch, best_miou,
            os.path.join(save_dir, 'last.pth')
        )
    
    logger.info(f'Training completed. Best mIoU: {best_miou:.4f}')
    writer.close()


if __name__ == '__main__':
    main()

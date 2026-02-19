import torch
import torch.nn as nn
import argparse
import os
import sys
import time
from config import E2EOccConfig
from e2e_occ_net import build_model
from loss import OccupancyLoss
from dataset import get_dataloader

# conda activate deepsys
# python d:\code\carla\e2e_occ\train.py --batch_size 1 --epochs 100 --amp --data_root d:\code\carla\dataset_10k_bak

# 禁用输出缓冲，确保每条日志立即显示（Windows下默认行缓冲会导致输出积压）
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, use_amp=False, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    start = time.time()
    optimizer.zero_grad()
    
    # TBPTT Settings
    TBPTT_CHUNK_SIZE = 2  # Truncate gradients every 2 steps
    
    for i, batch in enumerate(loader):
        images = batch['images'].to(device)
        voxels = batch['voxels'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        extrinsics = batch['extrinsics'].to(device)
        
        # Check for sequence data [B, T, N, C, H, W]
        is_sequence = images.dim() == 6
        
        loss_val = 0.0
        ce_loss_val = 0.0
        
        if is_sequence:
            B, T, N, C, H, W = images.shape
            memory = None
            seq_loss = 0.0
            seq_ce = 0.0

            # TBPTT 循环：每个 chunk 做一次明确的 backward，scaler 状态清晰
            for t_start in range(0, T, TBPTT_CHUNK_SIZE):
                t_end = min(t_start + TBPTT_CHUNK_SIZE, T)

                # 截断梯度历史
                if memory is not None:
                    memory = memory.detach()

                chunk_loss = torch.tensor(0.0, device=device)
                total_weight = 0.0

                for t in range(t_start, t_end):
                    img_t = images[:, t]      # [B, N, C, H, W]
                    vox_t = voxels[:, t]      # [B, X, Y, Z]
                    ext_t = extrinsics[:, t]  # [B, N, 4, 4]

                    ego_motion = None
                    if t > 0:
                        ext_prev = extrinsics[:, t-1]
                        # pose: [B, 4, 4], Camera→World (extrinsics 惯例)
                        # ego_motion = inv(C_t→W) @ (C_{t-1}→W) = C_{t-1}→C_t
                        # 语义：上一帧体素坐标系 → 当前帧体素坐标系
                        pose_t = ext_t[:, 0]
                        pose_prev = ext_prev[:, 0]
                        ego_motion = torch.linalg.inv(pose_t) @ pose_prev

                        # Log ego_motion for the first batch of each epoch
                        if i == 0 and t == 1:
                            print(f"\n--- Ego-motion check (epoch {epoch}) ---")
                            print(ego_motion[0].detach().cpu().numpy())
                            print("------------------------------------\n")

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs = model(img_t, intrinsics, ext_t, memory=memory, ego_motion=ego_motion)
                        loss_dict = criterion(outputs['semantic'], vox_t)

                        time_weight = 1.0 + (t / max(1, T - 1))
                        chunk_loss = chunk_loss + loss_dict['total'] * time_weight
                        total_weight += time_weight

                    seq_loss += loss_dict['total'].item()
                    seq_ce += loss_dict['ce'].item()
                    memory = outputs['memory']

                # 每个 chunk 统一 backward 一次，scaler 状态始终明确
                chunk_loss_norm = chunk_loss / (total_weight * grad_accum_steps)
                scaler.scale(chunk_loss_norm).backward()

            loss_val = seq_loss / T
            ce_loss_val = seq_ce / T

        else:
            # 单帧训练
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images, intrinsics, extrinsics)
                losses = criterion(outputs['semantic'], voxels)
                loss = losses['total'] / grad_accum_steps
                loss_val = losses['total'].item()
                ce_loss_val = losses['ce'].item()

            scaler.scale(loss).backward()

        # 梯度累积步进（两个分支的 scaler 状态此时均明确）
        if (i + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss_val * grad_accum_steps

        # 每步都打印，确保实时可见（小数据集时每10步一次可能长时间无输出）
        elapsed = time.time() - start
        print(f'Epoch {epoch} [{i+1}/{len(loader)}] Loss: {loss_val:.4f} CE: {ce_loss_val:.4f} Time: {elapsed:.1f}s', flush=True)
            
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            voxels = batch['voxels'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            
            is_sequence = images.dim() == 6
            
            if is_sequence:
                B, T, N, C, H, W = images.shape
                memory = None
                seq_loss = 0.0
                
                for t in range(T):
                    img_t = images[:, t]
                    vox_t = voxels[:, t]
                    ext_t = extrinsics[:, t]
                    
                    ego_motion = None
                    if t > 0:
                        ext_prev = extrinsics[:, t-1]
                        # ego_motion = inv(C_t→W) @ (C_{t-1}→W) = C_{t-1}→C_t
                        pose_t = ext_t[:, 0]
                        pose_prev = ext_prev[:, 0]
                        ego_motion = torch.linalg.inv(pose_t) @ pose_prev

                    outputs = model(img_t, intrinsics, ext_t, memory=memory, ego_motion=ego_motion)
                    loss_dict = criterion(outputs['semantic'], vox_t)
                    seq_loss += loss_dict['total'].item()
                    memory = outputs['memory']
                
                total_loss += seq_loss / T
            else:
                outputs = model(images, intrinsics, extrinsics)
                losses = criterion(outputs['semantic'], voxels)
                total_loss += losses['total'].item()
            
    # Clear cache after validation to free up memory for training
    torch.cuda.empty_cache()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers. Default 0 for Windows to avoid deadlocks.')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision training')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = E2EOccConfig()
    model = build_model(config).to(device)
    print(f'Model params: {model.get_num_params() / 1e6:.2f}M')
    
    criterion = OccupancyLoss(num_classes=config.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')
        
    train_loader = get_dataloader(args.data_root, 'train', args.batch_size, args.num_workers, config)
    val_loader = get_dataloader(args.data_root, 'val', args.batch_size, args.num_workers, config)
    
    best_loss = float('inf')
    
    print(f"Starting training with AMP={args.amp}, Grad Accum={args.grad_accum}", flush=True)
    print(f"Temporal Training: {config.use_temporal} (Frames={config.temporal_frames})", flush=True)
    
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, 
                               use_amp=args.amp, grad_accum_steps=args.grad_accum)
        
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}', flush=True)

        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'Saved best model at epoch {epoch}', flush=True)

if __name__ == '__main__':
    main()

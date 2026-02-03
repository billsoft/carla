import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import argparse
import os
import time
from config import E2EOccConfig
from e2e_occ_net import build_model
from loss import OccupancyLoss
from dataset import get_dataloader

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
            
            # TBPTT Loop: Process sequence in chunks
            for t_start in range(0, T, TBPTT_CHUNK_SIZE):
                t_end = min(t_start + TBPTT_CHUNK_SIZE, T)
                chunk_steps = t_end - t_start
                
                # Detach memory to truncate gradient history
                if memory is not None:
                    memory = memory.detach()
                
                chunk_loss = 0.0
                total_weight = 0.0
                
                # Process steps within chunk
                for t in range(t_start, t_end):
                    img_t = images[:, t]      # [B, N, C, H, W]
                    vox_t = voxels[:, t]      # [B, X, Y, Z]
                    ext_t = extrinsics[:, t]  # [B, N, 4, 4]
                    
                    # Calculate Ego-Motion: T_{t-1 -> t}
                    ego_motion = None
                    if t > 0:
                        ext_prev = extrinsics[:, t-1] 
                        pose_t = ext_t[:, 0] 
                        pose_prev = ext_prev[:, 0] 
                        ego_motion = torch.linalg.inv(pose_t) @ pose_prev
                        
                    with autocast(enabled=use_amp):
                        outputs = model(img_t, intrinsics, ext_t, memory=memory, ego_motion=ego_motion)
                        loss_dict = criterion(outputs['semantic'], vox_t)
                        
                        # Time-weighted Loss: Give more weight to later frames in sequence
                        # Weight grows linearly from 1.0 to 2.0 over the sequence
                        time_weight = 1.0 + (t / max(1, T - 1))
                        
                        step_loss = loss_dict['total']
                        
                        # Accumulate weighted loss
                        chunk_loss += step_loss * time_weight
                        total_weight += time_weight
                        
                        # Metrics logging (raw loss)
                        seq_loss += step_loss.item()
                        seq_ce += loss_dict['ce'].item()
                        
                        # Update memory for next step
                        memory = outputs['memory']
                
                # Backward for this chunk
                # Normalize by total weight instead of simple average
                loss_to_backprop = chunk_loss / (total_weight * grad_accum_steps)
                scaler.scale(loss_to_backprop).backward()
            
            # Average metrics over full sequence
            loss_val = seq_loss / T
            ce_loss_val = seq_ce / T
            
            # For gradient accumulation step logic below
            # We already backwarded, so 'loss' variable for step() check is just for logging/scaler logic?
            # Actually, the outer loop structure expects 'loss' to be defined for scaler.scale(loss).backward()
            # BUT we already did backward inside the TBPTT loop!
            # We need to restructure the outer loop to NOT backward again if is_sequence.
            
        else:
            # Standard single-frame training
            with autocast(enabled=use_amp):
                outputs = model(images, intrinsics, extrinsics)
                losses = criterion(outputs['semantic'], voxels)
                loss = losses['total'] / grad_accum_steps
                loss_val = losses['total'].item()
                ce_loss_val = losses['ce'].item()
            
            # Scale and Backward (Only for single frame case)
            scaler.scale(loss).backward()
        
        # Gradient Accumulation Step
        if (i + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Optional: Aggressive cache clearing for debug (Slow!)
            # torch.cuda.empty_cache() 
        
        total_loss += loss_val * grad_accum_steps # Restore scale for logging
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f'Epoch {epoch} [{i+1}/{len(loader)}] Loss: {loss_val:.4f} CE: {ce_loss_val:.4f} Time: {elapsed:.1f}s')
            
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
    parser.add_argument('--num_workers', type=int, default=4)
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
    scaler = GradScaler(enabled=args.amp)
    
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
    
    print(f"Starting training with AMP={args.amp}, Grad Accum={args.grad_accum}")
    print(f"Temporal Training: {config.use_temporal} (Frames={config.temporal_frames})")
    
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, 
                               use_amp=args.amp, grad_accum_steps=args.grad_accum)
        
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}')
        
        ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'config': config}
        torch.save(ckpt, os.path.join(args.output_dir, 'latest.pth'))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(ckpt, os.path.join(args.output_dir, 'best.pth'))

if __name__ == '__main__':
    main()

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
    
    for i, batch in enumerate(loader):
        images = batch['images'].to(device)
        voxels = batch['voxels'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        extrinsics = batch['extrinsics'].to(device)
        
        # Mixed Precision Context
        with autocast(enabled=use_amp):
            outputs = model(images, intrinsics, extrinsics)
            losses = criterion(outputs['semantic'], voxels)
            loss = losses['total'] / grad_accum_steps
        
        # Scale and Backward
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
        
        total_loss += loss.item() * grad_accum_steps
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f'Epoch {epoch} [{i+1}/{len(loader)}] Loss: {loss.item()*grad_accum_steps:.4f} CE: {losses["ce"].item():.4f} Time: {elapsed:.1f}s')
            
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

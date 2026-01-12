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
    model.train()
    total_loss = 0
    device = next(model.parameters()).device
    prefetcher = FP16DataPrefetcher(loader, device)
    batch = prefetcher.next()
    step = 0
    while batch is not None:
        optimizer.zero_grad()
        with autocast(enabled=config.use_amp):
            outputs = model(batch['images'], batch.get('ego_motion'), batch.get('ego_pose'))
            losses = loss_fn(outputs, batch)
            loss = losses['total']
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
        if step % config.log_interval == 0:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f} Focal: {losses.get('focal', 0):.4f} Dice: {losses.get('dice', 0):.4f} GPU Mem: {mem:.2f}GB")
        batch = prefetcher.next()
        step += 1
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    config.data_dir = args.data_dir
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
    best_loss = float('inf')
    for epoch in range(start_epoch, config.max_epochs):
        if epoch < config.warmup_epochs:
            lr_scale = (epoch + 1) / config.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = config.lr * lr_scale
        model.reset_temporal()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, epoch, config)
        if epoch >= config.warmup_epochs:
            scheduler.step()
        print(f"Epoch {epoch} Train Loss: {train_loss:.4f}")
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

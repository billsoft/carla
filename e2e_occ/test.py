import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import E2EOccConfig
from e2e_occ_net import build_model
from loss import OccupancyLoss

def test_forward():
    print("=" * 60)
    print("E2E-OccNet Test (400x400x32 output)")
    print("=" * 60)
    config = E2EOccConfig()
    print(f"\nConfig:")
    print(f"  Image size: {config.image_size}")
    print(f"  Feature size: {config.feat_size}")
    print(f"  Coarse size: {config.coarse_size} ({config.num_coarse_queries} queries)")
    print(f"  Fine size: {config.fine_size} ({config.num_fine_queries} queries)")
    print(f"  Voxel size: {config.voxel_size}")
    print(f"  Voxel resolution: {config.voxel_resolution}m")
    print(f"  Num classes: {config.num_classes}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    model = build_model(config).to(device)
    num_params = model.get_num_params()
    print(f"Parameters: {num_params / 1e6:.2f}M")
    batch_size = 1
    images = torch.randn(batch_size, config.num_cameras, 1, *config.image_size).to(device)
    print(f"\nInput shape: {images.shape}")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = model(images)
    logits = outputs['semantic']
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, config.num_classes, *config.voxel_size), f"Expected {(batch_size, config.num_classes, *config.voxel_size)}, got {logits.shape}"
    if device.type == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        print(f"Peak memory (inference): {mem:.2f} GB")
    target = torch.randint(0, config.num_classes, (batch_size, *config.voxel_size)).to(device)
    criterion = OccupancyLoss(num_classes=config.num_classes)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model.train()
    outputs = model(images)
    losses = criterion(outputs['semantic'], target)
    losses['total'].backward()
    if device.type == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        print(f"Peak memory (training): {mem:.2f} GB")
    print(f"\nLoss: {losses['total'].item():.4f}")
    print(f"CE Loss: {losses['ce'].item():.4f}")
    print(f"Lovasz Loss: {losses['lovasz'].item():.4f}")
    print("\n" + "=" * 60)
    print("Test PASSED!")
    print("=" * 60)

if __name__ == '__main__':
    test_forward()

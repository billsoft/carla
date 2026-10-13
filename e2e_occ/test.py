"""快速前向/反向测试（随机输入 + 真实相机标定）。完整自检见 verify_network.py。"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import E2EOccConfig
from e2e_occ_net import build_model
from loss import E2ELoss
from dataset import default_calibration


def test_forward():
    print('=' * 60 + '\nE2E-OccNet Test (400x400x32 output)\n' + '=' * 60)
    config = E2EOccConfig()
    print(f'  Image size: {config.image_size}, levels: {config.level_strides}')
    print(f'  Coarse: {config.coarse_size} ({config.num_coarse_queries}), Fine: {config.fine_size} '
          f'({config.num_fine_queries}, sparse={config.use_sparse_fine})')
    print(f'  Voxel: {config.voxel_size} @ {config.voxel_resolution}m, classes: {config.num_classes}, head: {config.head_type}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    print(f'Parameters: {model.get_num_params() / 1e6:.2f}M')

    K, T, _ = default_calibration(config.num_cameras, config.image_size)
    K, T = K.unsqueeze(0).to(device), T.unsqueeze(0).to(device)
    images = torch.rand(1, config.num_cameras, 1, *config.image_size, device=device)

    model.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        out = model(images, K, T)
    assert out['semantic'].shape == (1, config.num_classes, *config.voxel_size), out['semantic'].shape
    print(f"Output: semantic {tuple(out['semantic'].shape)}"
          + (f", flow {tuple(out['flow'].shape)}" if out['flow'] is not None else ''))
    if device.type == 'cuda':
        print(f'Peak memory (inference): {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')

    model.train()
    vox = torch.randint(0, config.num_classes, (1, *config.voxel_size), device=device)
    tg = {'voxels': vox}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        out = model(images, K, T)
        losses = E2ELoss(config)(out, tg)
    losses['total'].backward()
    if device.type == 'cuda':
        print(f'Peak memory (training, 1 frame): {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')
    print('Losses: ' + ', '.join(f'{k}={float(v):.4f}' for k, v in losses.items()))
    print('=' * 60 + '\nTest PASSED!\n' + '=' * 60)


if __name__ == '__main__':
    test_forward()

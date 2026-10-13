"""
图像骨干 MAE 式自监督预训练（掩码重建 Bayer RAW）。

思路（He et al., Masked Autoencoders）：随机遮掉 RAW 图上 mask_ratio 比例的 16×16 块（置零），
骨干（MultiCameraPatchEmbed + ImageEncoder）+ 一个线性解码器从 stride-16 特征重建被遮块的
原始像素，损失只在被遮块上计算（逐块归一化目标，norm_pix_loss）。CARLA 可以无限量采集
无标注 RAW，用它预训练骨干，再用 train.py --pretrained_backbone 加载后在 10k 标注样本上微调。

说明：本网络骨干是卷积 stem + 窗口注意力，不能像 ViT-MAE 那样直接丢弃被遮 token 节省计算，
这里用"置零 + 只在被遮块上算损失"的 SimMIM 式做法，计算量与正常前向相同。

用法:
  python e2e_occ/pretrain_mae.py --data_root <dataset_dir> --epochs 20 --amp
  python e2e_occ/train.py --data_root <dataset_dir> --pretrained_backbone checkpoints_mae/backbone.pth ...
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import E2EOccConfig
from raw_embed import MultiCameraPatchEmbed
from image_encoder import ImageEncoder
from dataset import get_dataloader

sys.stdout.reconfigure(line_buffering=True)


class MAEBackbone(nn.Module):
    def __init__(self, config, patch=16):
        super().__init__()
        self.config = config
        self.patch = patch
        self.patch_embed = MultiCameraPatchEmbed(config)
        self.encoder = ImageEncoder(config)
        self.decoder = nn.Sequential(
            nn.Conv2d(config.embed_dim, config.embed_dim, 1), nn.GELU(),
            nn.Conv2d(config.embed_dim, patch * patch, 1))

    def random_mask(self, B, N, H, W, ratio, device):
        h, w = H // self.patch, W // self.patch
        m = (torch.rand(B, N, 1, h, w, device=device) < ratio).float()
        return m, F.interpolate(m.flatten(0, 1), scale_factor=self.patch, mode='nearest').view(B, N, 1, H, W)

    def forward(self, images, intrinsics, extrinsics, mask_ratio=0.6):
        B, N, _, H, W = images.shape
        m_patch, m_pix = self.random_mask(B, N, H, W, mask_ratio, images.device)
        feats = self.patch_embed(images * (1.0 - m_pix))
        f16 = self.encoder(feats, intrinsics, extrinsics)['feats'][16]           # [B, N, C, h, w]
        pred = self.decoder(f16.flatten(0, 1))                                    # [B*N, p*p, h, w]
        target = F.unfold(images.flatten(0, 1), kernel_size=self.patch, stride=self.patch)  # [B*N, p*p, h*w]
        target = target.view_as(pred)
        mean = target.mean(1, keepdim=True)
        var = target.var(1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()
        loss = ((pred.float() - target) ** 2).mean(1, keepdim=True)              # [B*N, 1, h, w]
        m = m_patch.flatten(0, 1)
        return (loss * m).sum() / m.sum().clamp(min=1.0)

    def backbone_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if k.startswith(('patch_embed.', 'encoder.'))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--output_dir', default='./checkpoints_mae')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--mask_ratio', type=float, default=0.6)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--max_steps', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = E2EOccConfig()
    # 预训练只需要单帧图像：关掉时序/深度/流场的数据加载
    config.use_temporal = False
    config.use_depth_aux = False
    config.use_flow_head = False
    model = MAEBackbone(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    loader = get_dataloader(args.data_root, 'train', 1, 0, config)
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for i, batch in enumerate(loader):
            images = batch['images'][:, 0].to(device)
            K, T = batch['intrinsics'].to(device), batch['extrinsics'].to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp):
                loss = model(images, K, T, args.mask_ratio)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            print(f'MAE epoch {epoch} [{i + 1}/{len(loader)}] loss={loss.item():.4f} {time.time() - t0:.1f}s')
            if args.max_steps and i + 1 >= args.max_steps:
                break
        torch.save({'backbone': model.backbone_state_dict(), 'config': config, 'epoch': epoch},
                   os.path.join(args.output_dir, 'backbone.pth'))


if __name__ == '__main__':
    main()

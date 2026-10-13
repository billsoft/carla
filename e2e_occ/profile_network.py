"""逐阶段显存/耗时剖析（训练模式，单帧前向 + 反向）。

用法: python e2e_occ/profile_network.py [--set use_sparse_fine=False ...]
"""
import argparse
import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import E2EOccConfig
from e2e_occ_net import E2EOccNet
from loss import E2ELoss
from dataset import default_calibration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', nargs='*', default=[])
    args = ap.parse_args()
    config = E2EOccConfig()
    for kv in args.set:
        k, v = kv.split('=', 1)
        cur = getattr(config, k)
        setattr(config, k, (v.lower() in ('1', 'true')) if isinstance(cur, bool) else type(cur)(v))

    model = E2EOccNet(config).cuda().train()
    K, T, _ = default_calibration(config.num_cameras, config.image_size)
    K, T = K.unsqueeze(0).cuda(), T.unsqueeze(0).cuda()
    imgs = torch.rand(1, config.num_cameras, 1, *config.image_size, device='cuda')

    stats = []

    def mark(tag):
        torch.cuda.synchronize()
        stats.append((tag, time.time(), torch.cuda.memory_allocated() / 1024 ** 3,
                      torch.cuda.max_memory_allocated() / 1024 ** 3))

    torch.cuda.reset_peak_memory_stats()
    mark('start')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        feats = model.patch_embed(imgs); mark('patch_embed')
        enc = model.encoder(feats, K, T); mark('encoder')
        dec = model.decoder(enc['feats'], K, T); mark('decoder')
        out = model.head(dec['fine_feats']); mark('head')
        out.update(depth_logits=enc['depth_logits'], coarse_occ_logits=dec['coarse_occ_logits'])
        vox = torch.randint(0, config.num_classes, (1, *config.voxel_size), device='cuda')
        loss = E2ELoss(config)(out, {'voxels': vox})['total']; mark('loss')
    loss.backward(); mark('backward')

    print(f'{"阶段":<14s}{"耗时(s)":>10s}{"当前显存(GB)":>16s}{"峰值显存(GB)":>16s}')
    for (tag, t, cur, peak), (_, t_prev, _, _) in zip(stats[1:], stats[:-1]):
        print(f'{tag:<14s}{t - t_prev:>10.2f}{cur:>16.2f}{peak:>16.2f}')


if __name__ == '__main__':
    main()

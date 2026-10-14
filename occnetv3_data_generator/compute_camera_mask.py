"""
离线补算 / 重算数据集的相机可见性掩码 mask_camera/<sid>.npy (与采集时同一个函数：
processing/camera_visibility.py)。需要数据集里有 depth/<sid>/cam_i.npy 和 ue_frotator 约定的标定。

    python compute_camera_mask.py --dataset C:/carla_datasets/ue58_town10_200 [--overwrite] [--stats]

--stats 打印每类占用体素被相机观测到的比例，以及每列"最上面一个占用体素"(地表/物体顶面) 的观测率，
用来检查掩码有没有在远处地面系统性漏标 (掠射角问题)。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config.occupancy_config import X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION, OCCUPANCY_LABELS  # noqa: E402
from processing.camera_visibility import CameraVisibility, save_mask, load_mask  # noqa: E402


def top_surface(occ):
    """每列最上面一个占用体素 (X,Y,Z) bool。"""
    occ_any = occ > 0
    zs = np.arange(occ.shape[2])
    top = np.where(occ_any.any(2), (occ_any * zs).max(2), -1)
    return occ_any & (zs[None, None, :] == top[:, :, None])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--overwrite', action='store_true', help='已存在的 mask_camera 也重算')
    ap.add_argument('--stats', action='store_true', help='打印观测率统计')
    ap.add_argument('--every', type=int, default=1, help='只处理每 N 帧 (配合 --stats 快速抽查)')
    args = ap.parse_args()

    root = Path(args.dataset)
    vis = CameraVisibility.from_calibration(root / 'calibration', X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION)
    n_cam = len(vis.cameras)
    out_dir = root / 'mask_camera'
    out_dir.mkdir(exist_ok=True)
    sids = sorted(p.stem for p in (root / 'occupancy').glob('*.npy'))[::args.every]

    occ_n = np.zeros(18)
    occ_v = np.zeros(18)
    top_n = np.zeros(18)
    top_v = np.zeros(18)
    top_dist = np.zeros((2, 5))       # 顶面观测率随水平距离: 0-8,8-16,16-24,24-32,32+ m
    done = skipped = missing = 0
    t0 = time.time()
    cx = X_RANGE[0] + (np.arange(vis.shape[0]) + 0.5) * RESOLUTION
    cy = Y_RANGE[0] + (np.arange(vis.shape[1]) + 0.5) * RESOLUTION
    dist_bin = np.minimum((np.hypot(cx[:, None], cy[None, :]) // 8).astype(int), 4)   # (X, Y)
    for sid in sids:
        out = out_dir / f'{sid}.npy'
        depth_paths = [root / 'depth' / sid / f'cam_{i}.npy' for i in range(n_cam)]
        if not all(p.exists() for p in depth_paths):
            missing += 1
            continue
        if out.exists() and not args.overwrite and not args.stats:
            skipped += 1
            continue
        if out.exists() and not args.overwrite:
            mask = load_mask(out, vis.shape[2])
        else:
            mask = vis.compute([np.load(p) for p in depth_paths])
            save_mask(out, mask)
            done += 1
        if args.stats:
            occ = np.load(root / 'occupancy' / f'{sid}.npy')
            top = top_surface(occ)
            occ_n += np.bincount(occ.ravel(), minlength=18)[:18]
            occ_v += np.bincount(occ[mask], minlength=18)[:18]
            top_n += np.bincount(occ[top], minlength=18)[:18]
            top_v += np.bincount(occ[top & mask], minlength=18)[:18]
            tb = np.broadcast_to(dist_bin[:, :, None], occ.shape)
            top_dist[0] += np.bincount(tb[top], minlength=5)[:5]
            top_dist[1] += np.bincount(tb[top & mask], minlength=5)[:5]
            print(f'{sid}: 观测体素 {mask.mean() * 100:.1f}%, 占用体素被观测 '
                  f'{100 * mask[occ > 0].mean():.1f}%, 顶面被观测 {100 * mask[top].mean():.1f}%')
    print(f'完成: 新算 {done}, 跳过已有 {skipped}, 缺深度 {missing}, 耗时 {time.time() - t0:.0f}s')
    if args.stats and occ_n[1:].sum() > 0:
        print('\n类别              占用体素  被观测%   顶面体素  顶面被观测%')
        for k in range(1, 18):
            if occ_n[k]:
                print(f'{k:2d} {OCCUPANCY_LABELS[k]:20s} {int(occ_n[k]):9d}  {100 * occ_v[k] / occ_n[k]:6.1f}  '
                      f'{int(top_n[k]):9d}  {100 * top_v[k] / max(top_n[k], 1):6.1f}')
        print('顶面观测率 vs 水平距离: ' + ', '.join(
            f'{a}-{a + 8}m {100 * top_dist[1, i] / max(top_dist[0, i], 1):.1f}%'
            for i, a in enumerate(range(0, 40, 8))))


if __name__ == '__main__':
    main()

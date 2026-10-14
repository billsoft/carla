"""
推理脚本：对数据集逐帧流式推理（同一场景内携带时序记忆，换场景时重置），
结果按原数据集格式写到 output/occupancy/<sample_id>.npy，可直接用 dataset_viewer_v2 查看。

相机标定、RAW 位深归一化、ego_motion 全部复用 dataset.OccupancyDataset 的实现，
与训练完全一致（旧版推理脚本自己解析标定，硬编码 /4095，并把含世界位姿的外参喂给网络）。
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from config import E2EOccConfig
from e2e_occ_net import build_model
from dataset import OccupancyDataset, _SAMPLE_RE, default_calibration


class OccInference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = ckpt.get('config', E2EOccConfig())
        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        print(f'[Inference] Model loaded from {checkpoint_path} ({self.config.head_type} head)')

    @torch.no_grad()
    def predict(self, images, intrinsics, extrinsics, memory=None, ego_motion=None, frame_dt=None):
        """images [B,N,1,H,W]、intrinsics [B,N,3,3]、extrinsics [B,N,4,4]（Camera(cv)→Vehicle）
        返回 (pred [B,X,Y,Z] uint8 numpy, flow [B,3,X,Y,Z] 或 None, new_memory)"""
        cfg = self.config
        B = images.shape[0]
        dt = cfg.frame_dt if frame_dt is None else frame_dt
        td = torch.full((B,), dt, device=self.device) if memory is not None else None
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(self.device.type == 'cuda')):
            out = self.model(images.to(self.device), intrinsics.to(self.device), extrinsics.to(self.device),
                             memory=memory, ego_motion=None if ego_motion is None else ego_motion.to(self.device),
                             time_delta=td)
        pred = out['semantic'].argmax(dim=1).cpu().numpy().astype(np.uint8)
        flow = out['flow'].float().cpu().numpy() if out.get('flow') is not None else None
        return pred, flow, out['memory']

    def benchmark(self, num_runs=50, warmup=5):
        cfg = self.config
        K, T, _ = default_calibration(cfg.num_cameras, cfg.image_size)
        K, T = K.unsqueeze(0).to(self.device), T.unsqueeze(0).to(self.device)
        img = torch.rand(1, cfg.num_cameras, 1, *cfg.image_size, device=self.device)
        mem = None
        for _ in range(warmup):
            _, _, mem = self.predict(img, K, T, mem, torch.eye(4).unsqueeze(0) if mem is not None else None)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            _, _, mem = self.predict(img, K, T, mem, torch.eye(4).unsqueeze(0))
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        fps = num_runs / (time.time() - start)
        print(f'FPS: {fps:.2f}, Latency: {1000 / fps:.2f}ms（含 argmax 与回传 CPU）')
        return fps


def run_inference_on_dataset(checkpoint_path, data_root, output_dir, split='val', num_samples=100,
                             device='cuda', save_flow=False):
    engine = OccInference(checkpoint_path, device=device)
    cfg = E2EOccConfig(**{k: getattr(engine.config, k) for k in E2EOccConfig.__dataclass_fields__})
    # 推理逐帧读取，时序记忆由这里自己携带；不需要加载深度/流场真值
    cfg.use_temporal = False
    cfg.use_depth_aux = False
    cfg.use_flow_head = False
    ds = OccupancyDataset(data_root, split=split, config=cfg)
    samples = ds.samples[:num_samples]
    out_dir = Path(output_dir)
    (out_dir / 'occupancy').mkdir(parents=True, exist_ok=True)
    if save_flow:
        (out_dir / 'flow').mkdir(parents=True, exist_ok=True)
    print(f'[Inference] {data_root} [{split}] {len(samples)} 帧 → {output_dir}')

    memory, prev_pose, prev_key = None, None, None
    K = ds.intrinsics.unsqueeze(0)
    E = ds.extrinsics.unsqueeze(0)
    for i, sid in enumerate(samples):
        t0 = time.time()
        frame = ds._load_frame(sid)
        m = _SAMPLE_RE.match(sid)
        key = (m['scene'], int(m['frame'])) if m else None
        # 同一场景且帧号连续才延续记忆
        cont = (engine.config.use_temporal and prev_key is not None and key is not None
                and key[0] == prev_key[0] and key[1] == prev_key[1] + 1)
        ego_motion = None
        if cont:
            ego_motion = (torch.linalg.inv(frame['pose']) @ prev_pose).float().unsqueeze(0)
        else:
            memory = None
        pred, flow, memory = engine.predict(frame['images'].unsqueeze(0), K, E, memory, ego_motion,
                                            frame_dt=ds.frame_dt(sid))
        np.save(out_dir / 'occupancy' / f'{sid}.npy', pred[0])
        if save_flow and flow is not None:
            np.save(out_dir / 'flow' / f'{sid}.npy', flow[0].astype(np.float16))
        prev_pose, prev_key = frame['pose'], key
        print(f'[{i + 1:4d}/{len(samples)}] {sid} non_empty={int(np.count_nonzero(pred[0])):,} '
              f'{"(记忆延续)" if cont else "(新场景)"} {(time.time() - t0) * 1000:.0f}ms', flush=True)

    with open(out_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(samples))
    print(f'\n[Inference] 完成。查看: python dataset_viewer_v2/server.py --dataset {output_dir}')


def main():
    p = argparse.ArgumentParser(description='E2E Occupancy Inference')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_root', default='d:/code/carla/dataset_10k_bak')
    p.add_argument('--output', default='d:/code/carla/inference_results')
    p.add_argument('--split', default='val')
    p.add_argument('--num_samples', type=int, default=100)
    p.add_argument('--device', default='cuda')
    p.add_argument('--save_flow', action='store_true')
    p.add_argument('--benchmark', action='store_true')
    a = p.parse_args()
    if a.benchmark:
        OccInference(a.checkpoint, a.device).benchmark()
        return
    run_inference_on_dataset(a.checkpoint, a.data_root, a.output, a.split, a.num_samples, a.device, a.save_flow)


if __name__ == '__main__':
    main()

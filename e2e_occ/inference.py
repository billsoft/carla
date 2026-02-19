import torch
import numpy as np
import argparse
import os
import sys
import time
from pathlib import Path
from config import E2EOccConfig
from e2e_occ_net import build_model
from dataset import OccupancyDataset


class OccInference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = ckpt.get('config', E2EOccConfig())
        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        print(f'[Inference] Model loaded from {checkpoint_path}')
        print(f'[Inference] Device: {self.device}')

    @torch.no_grad()
    def predict_batch(self, images, intrinsics, extrinsics):
        """
        推理单帧（非时序）
        images:     [B, N, C, H, W]
        intrinsics: [B, N, 3, 3]
        extrinsics: [B, N, 4, 4]
        returns:    [B, X, Y, Z] uint8 numpy
        """
        images = images.to(self.device)
        intrinsics = intrinsics.to(self.device)
        extrinsics = extrinsics.to(self.device)

        with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
            outputs = self.model(images, intrinsics, extrinsics)

        logits = outputs['semantic']           # [B, 18, 400, 400, 32]
        pred = logits.argmax(dim=1)            # [B, 400, 400, 32]
        return pred.cpu().numpy().astype(np.uint8)

    def benchmark(self, num_runs=100, warmup=10):
        dummy_img = torch.randn(1, self.config.num_cameras, 1, *self.config.image_size).to(self.device)
        dummy_K   = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, self.config.num_cameras, 1, 1).to(self.device)
        dummy_E   = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, self.config.num_cameras, 1, 1).to(self.device)

        for _ in range(warmup):
            with torch.no_grad():
                self.model(dummy_img, dummy_K, dummy_E)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                self.model(dummy_img, dummy_K, dummy_E)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        fps = num_runs / elapsed
        print(f'FPS: {fps:.2f}, Latency: {1000/fps:.2f}ms')
        return fps


def run_inference_on_dataset(checkpoint_path, data_root, output_dir, num_samples=100, device='cuda'):
    """
    对 dataset_10k_bak 格式的数据集进行推理，输出保存到 output_dir/occupancy/<sample_id>.npy
    格式与原始数据集完全相同，可直接用 dataset_viewer_v2 查看。
    """
    engine = OccInference(checkpoint_path, device=device)
    config = engine.config

    # 使用单帧模式（不用时序，推理更简单）
    config_infer = E2EOccConfig()
    config_infer.use_temporal = False
    config_infer.temporal_frames = 1

    dataset = OccupancyDataset(data_root, split='train', config=config_infer)

    # 同时也加载 val/test
    all_samples = []
    for split in ['train', 'val', 'test']:
        txt = Path(data_root) / f'{split}.txt'
        if txt.exists():
            with open(txt) as f:
                all_samples += [l.strip() for l in f if l.strip()]
    # 去重
    all_samples = list(dict.fromkeys(all_samples))

    # 如果没有split文件，直接扫occupancy目录
    if not all_samples:
        occ_dir = Path(data_root) / 'occupancy'
        all_samples = sorted([f.stem for f in occ_dir.glob('*.npy')])

    total = min(num_samples, len(all_samples))
    print(f'[Inference] Dataset: {data_root}')
    print(f'[Inference] Total samples: {len(all_samples)}, running: {total}')

    # 输出目录结构与原数据集一致（viewer直接能读）
    out_occ_dir = Path(output_dir) / 'occupancy'
    out_occ_dir.mkdir(parents=True, exist_ok=True)

    # 同时写 train.txt 方便viewer加载
    written_ids = []

    for i, sample_id in enumerate(all_samples[:total]):
        t0 = time.time()

        # 加载图像
        images = []
        img_dir = Path(data_root) / 'images' / sample_id
        for cam_i in range(config.num_cameras):
            dng_path = img_dir / f'cam_{cam_i}.dng'
            npy_path = img_dir / f'cam_{cam_i}.npy'

            img = None
            if dng_path.exists():
                try:
                    import rawpy
                    with rawpy.imread(str(dng_path)) as raw:
                        arr = raw.raw_image_visible.astype(np.float32)
                        arr = arr / 4095.0
                        img = torch.from_numpy(arr[np.newaxis])   # [1, H, W]
                except Exception as e:
                    print(f'  [warn] DNG load failed cam_{cam_i}: {e}')

            if img is None and npy_path.exists():
                arr = np.load(npy_path).astype(np.float32)
                img = torch.from_numpy(arr)

            if img is None:
                img = torch.zeros(1, *config.image_size)

            images.append(img)

        images = torch.stack(images, dim=0).unsqueeze(0).float()  # [1, N, 1, H, W]

        # 加载相机参数
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, config.num_cameras, 1, 1)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, config.num_cameras, 1, 1)

        # 优先加载 ego_pose + calibration
        K_path = Path(data_root) / 'camera_params' / f'{sample_id}.npz'
        ego_path = Path(data_root) / 'ego_pose' / f'{sample_id}.npy'

        if K_path.exists():
            d = np.load(K_path, allow_pickle=False)
            intrinsics = torch.from_numpy(d['intrinsics'].astype(np.float32)).unsqueeze(0)
            extrinsics = torch.from_numpy(d['extrinsics'].astype(np.float32)).unsqueeze(0)
        elif ego_path.exists():
            calib_K = Path(data_root) / 'calibration' / 'intrinsics.json'
            calib_E = Path(data_root) / 'calibration' / 'extrinsics.json'
            if calib_K.exists() and calib_E.exists():
                import json
                with open(calib_K) as f:
                    kd = json.load(f)
                with open(calib_E) as f:
                    ed = json.load(f)
                ego_pose = np.load(str(ego_path)).astype(np.float32)
                K_list, E_list = [], []
                for ci in range(config.num_cameras):
                    ck = kd.get(f'cam_{ci}', {})
                    K = np.eye(3, dtype=np.float32)
                    K[0,0] = ck.get('fx', 800); K[1,1] = ck.get('fy', 800)
                    K[0,2] = ck.get('cx', 640); K[1,2] = ck.get('cy', 480)
                    K_list.append(K)
                    ce = ed.get(f'cam_{ci}', {})
                    R = np.array(ce.get('rotation_matrix', np.eye(3).tolist()), dtype=np.float32)
                    t = np.array(ce.get('translation', [0,0,0]), dtype=np.float32)
                    T_rel = np.eye(4, dtype=np.float32)
                    T_rel[:3,:3] = R; T_rel[:3,3] = t
                    T_abs = ego_pose @ T_rel
                    E_list.append(T_abs)
                intrinsics = torch.from_numpy(np.stack(K_list)).unsqueeze(0)
                extrinsics = torch.from_numpy(np.stack(E_list)).unsqueeze(0)

        # 推理
        pred = engine.predict_batch(images, intrinsics, extrinsics)  # [1, 400, 400, 32]
        pred_voxel = pred[0]  # [400, 400, 32]

        # 保存（格式与数据集一致）
        out_path = out_occ_dir / f'{sample_id}.npy'
        np.save(str(out_path), pred_voxel)
        written_ids.append(sample_id)

        elapsed = time.time() - t0
        non_empty = int(np.count_nonzero(pred_voxel))
        print(f'[{i+1:4d}/{total}] {sample_id}  non_empty={non_empty:,}  {elapsed*1000:.0f}ms', flush=True)

    # 写 train.txt 供 viewer 加载
    with open(Path(output_dir) / 'train.txt', 'w') as f:
        f.write('\n'.join(written_ids))

    print(f'\n[Inference] Done. Results saved to: {output_dir}')
    print(f'[Inference] Viewer command:')
    print(f'  python d:/code/carla/dataset_viewer_v2/server.py --dataset {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='E2E Occupancy Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--data_root',  type=str, default='d:/code/carla/dataset_10k_bak', help='数据集根目录')
    parser.add_argument('--output',     type=str, default='d:/code/carla/inference_results', help='推理结果输出目录')
    parser.add_argument('--num_samples',type=int, default=100, help='推理样本数量')
    parser.add_argument('--device',     type=str, default='cuda', help='cuda 或 cpu')
    parser.add_argument('--benchmark',  action='store_true', help='只跑性能测试')
    args = parser.parse_args()

    if args.benchmark:
        engine = OccInference(args.checkpoint, args.device)
        engine.benchmark()
        return

    run_inference_on_dataset(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output,
        num_samples=args.num_samples,
        device=args.device,
    )


if __name__ == '__main__':
    main()

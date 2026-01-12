import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class OccDataset(Dataset):
    def __init__(self, data_dir, split='train', config=None, use_fp16=True):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.config = config
        self.use_fp16 = use_fp16
        self.samples = self._load_samples()

    def _load_samples(self):
        split_file = os.path.join(self.data_dir, f'{self.split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        return []

    def __len__(self):
        return max(len(self.samples), 100)

    def __getitem__(self, idx):
        if len(self.samples) > 0 and idx < len(self.samples):
            return self._load_real_sample(idx)
        return self._generate_synthetic_sample(idx)

    def _load_real_sample(self, idx):
        sample_id = self.samples[idx]
        images = []
        for cam_id in range(self.config.num_cameras if self.config else 8):
            img_path = os.path.join(self.data_dir, 'images', sample_id, f'cam_{cam_id}.npy')
            if os.path.exists(img_path):
                img = np.load(img_path)
            else:
                img = np.random.randn(1, 960, 1280).astype(np.float32)
            images.append(img)
        images = np.stack(images, axis=0)
        occ_path = os.path.join(self.data_dir, 'occupancy', f'{sample_id}.npy')
        if os.path.exists(occ_path):
            occupancy = np.load(occ_path)
        else:
            occupancy = np.zeros((512, 512, 40), dtype=np.int64)
        flow_path = os.path.join(self.data_dir, 'flow', f'{sample_id}.npy')
        if os.path.exists(flow_path):
            flow = np.load(flow_path)
        else:
            flow = np.zeros((3, 512, 512, 40), dtype=np.float32)
        dtype = torch.float16 if self.use_fp16 else torch.float32
        return {'images': torch.from_numpy(images).to(dtype), 'semantic': torch.from_numpy(occupancy).long(), 'flow': torch.from_numpy(flow).to(dtype), 'flow_mask': torch.ones(512, 512, 40, dtype=torch.bool)}

    def _generate_synthetic_sample(self, idx):
        config = self.config
        num_cameras = config.num_cameras if config else 8
        image_size = config.image_size if config else (960, 1280)
        voxel_size = config.voxel_size if config else (512, 512, 40)
        num_classes = config.num_classes if config else 18
        dtype = torch.float16 if self.use_fp16 else torch.float32
        images = torch.randn(num_cameras, 1, *image_size, dtype=dtype)
        semantic = torch.zeros(*voxel_size, dtype=torch.long)
        np.random.seed(idx)
        num_objects = np.random.randint(5, 20)
        for _ in range(num_objects):
            cx, cy, cz = np.random.randint(50, voxel_size[0] - 50), np.random.randint(50, voxel_size[1] - 50), np.random.randint(5, voxel_size[2] - 5)
            sx, sy, sz = np.random.randint(5, 30), np.random.randint(5, 30), np.random.randint(2, 10)
            cls = np.random.randint(1, num_classes)
            x1, x2 = max(0, cx - sx // 2), min(voxel_size[0], cx + sx // 2)
            y1, y2 = max(0, cy - sy // 2), min(voxel_size[1], cy + sy // 2)
            z1, z2 = max(0, cz - sz // 2), min(voxel_size[2], cz + sz // 2)
            semantic[x1:x2, y1:y2, z1:z2] = cls
        ground_height = 2
        semantic[:, :, :ground_height] = 11
        flow = torch.zeros(3, *voxel_size, dtype=dtype)
        moving_mask = (semantic > 0) & (semantic < 11)
        if moving_mask.any():
            flow[0][moving_mask] = torch.randn(moving_mask.sum(), dtype=dtype) * 0.5
            flow[1][moving_mask] = torch.randn(moving_mask.sum(), dtype=dtype) * 0.5
        flow_mask = semantic > 0
        ego_motion = torch.eye(4, dtype=dtype)
        ego_pose = torch.eye(4, dtype=dtype)
        return {'images': images, 'semantic': semantic, 'flow': flow, 'flow_mask': flow_mask, 'ego_motion': ego_motion, 'ego_pose': ego_pose}

def collate_fn(batch):
    result = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            result[key] = [b[key] for b in batch]
    return result

def build_dataloader(config, split='train'):
    dataset = OccDataset(data_dir=getattr(config, 'data_dir', './data'), split=split, config=config, use_fp16=getattr(config, 'use_fp16_input', True))
    shuffle = split == 'train'
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=split == 'train')

class FP16DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            for key in self.next_batch:
                if isinstance(self.next_batch[key], torch.Tensor):
                    self.next_batch[key] = self.next_batch[key].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

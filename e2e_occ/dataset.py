import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import rawpy

class OccupancyDataset(Dataset):
    def __init__(self, data_root, split='train', config=None):
        self.data_root = data_root
        self.split = split
        self.config = config
        self.num_cameras = config.num_cameras if config else 8
        self.image_size = config.image_size if config else (960, 1280)
        self.voxel_size = config.voxel_size if config else (400, 400, 32)
        
        # Temporal settings
        self.sequence_length = 1
        if config and hasattr(config, 'use_temporal') and config.use_temporal:
            self.sequence_length = getattr(config, 'temporal_frames', 1)
        
        self.samples = self._load_samples()
        self.intrinsics_data, self.extrinsics_data = self._load_calibration()
        # 一次性构建并缓存，避免每次 __getitem__ 重复构建 tensor
        self._cached_intrinsics, self._cached_extrinsics = self._get_camera_matrices()
        
    def _load_samples(self):
        split_file = os.path.join(self.data_root, f'{self.split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        return []

    def _load_calibration(self):
        calib_dir = os.path.join(self.data_root, 'calibration')
        int_path = os.path.join(calib_dir, 'intrinsics.json')
        ext_path = os.path.join(calib_dir, 'extrinsics.json')
        
        if not os.path.exists(int_path) or not os.path.exists(ext_path):
            print("Warning: Calibration files not found, using defaults.")
            return None, None
            
        with open(int_path, 'r') as f:
            intrinsics = json.load(f)
        with open(ext_path, 'r') as f:
            extrinsics = json.load(f)
            
        return intrinsics, extrinsics

    def _get_camera_matrices(self):
        if self.intrinsics_data is None or self.extrinsics_data is None:
            return self._get_default_camera_params()
            
        K_list = []
        E_list = []
        
        for i in range(self.num_cameras):
            cam_name = f'cam_{i}'
            
            # Intrinsics
            cfg_int = self.intrinsics_data[cam_name]
            K = torch.eye(3)
            K[0, 0] = cfg_int['fx']
            K[1, 1] = cfg_int['fy']
            K[0, 2] = cfg_int['cx']
            K[1, 2] = cfg_int['cy']
            K_list.append(K)
            
            # Extrinsics
            # JSON has rotation matrix and translation
            # We need 4x4 matrix
            cfg_ext = self.extrinsics_data[cam_name]
            E = torch.eye(4)
            # Rotation matrix in json is list of lists
            R = torch.tensor(cfg_ext['rotation_matrix'])
            t = torch.tensor(cfg_ext['translation'])
            E[:3, :3] = R
            E[:3, 3] = t
            E_list.append(E)
            
        return torch.stack(K_list), torch.stack(E_list)

    def _load_dng_image(self, dng_path):
        try:
            with rawpy.imread(dng_path) as raw:
                # Load RAW data (12-bit)
                img = raw.raw_image_visible.astype(np.float32)
                # Normalize to [0, 1]
                img = img / 4095.0 # Assuming 12-bit
                # Add channel dim: [1, H, W]
                img = img[np.newaxis, :, :]
                return torch.from_numpy(img)
        except Exception as e:
            print(f"Error loading DNG {dng_path}: {e}")
            return torch.zeros(1, *self.image_size)

    def __len__(self):
        if self.sequence_length > 1:
            return max(0, len(self.samples) - self.sequence_length + 1)
        return len(self.samples)
    
    def _load_single_frame(self, idx):
        sample_id = self.samples[idx]
        
        # 1. Load Images
        images = []
        for i in range(self.num_cameras):
            # Try multiple paths
            dng_path = os.path.join(self.data_root, 'images', sample_id, f'cam_{i}.dng')
            if os.path.exists(dng_path):
                img = self._load_dng_image(dng_path)
            else:
                # Fallback or error
                img = torch.zeros(1, *self.image_size)
            images.append(img)
        images = torch.stack(images, dim=0).float() # [N, 1, H, W]
        
        # 2. Load Voxels (Occupancy)
        occ_path = os.path.join(self.data_root, 'occupancy', f'{sample_id}.npy')
        if os.path.exists(occ_path):
            voxels = np.load(occ_path)
            voxels = torch.from_numpy(voxels).long()
        else:
            voxels = torch.zeros(self.voxel_size, dtype=torch.long)
            
        return {
            'images': images,
            'voxels': voxels,
            'intrinsics': self._cached_intrinsics,
            'extrinsics': self._cached_extrinsics
        }

    def __getitem__(self, idx):
        if self.sequence_length == 1:
            return self._load_single_frame(idx)
        
        frames = []
        for t in range(self.sequence_length):
            frames.append(self._load_single_frame(idx + t))
        
        # Stack frames
        return {
            'images': torch.stack([f['images'] for f in frames]),      # [T, N, C, H, W]
            'voxels': torch.stack([f['voxels'] for f in frames]),      # [T, X, Y, Z]
            'intrinsics': frames[0]['intrinsics'],                     # [N, 3, 3] (Assume constant)
            'extrinsics': torch.stack([f['extrinsics'] for f in frames]), # [T, N, 4, 4]
        }

    def _get_default_camera_params(self):
        # Return dummy identity parameters if missing
        intrinsics = torch.eye(3).unsqueeze(0).repeat(self.num_cameras, 1, 1)
        intrinsics[:, 0, 0] = 800
        intrinsics[:, 1, 1] = 800
        intrinsics[:, 0, 2] = self.image_size[1] / 2
        intrinsics[:, 1, 2] = self.image_size[0] / 2
        
        extrinsics = torch.eye(4).unsqueeze(0).repeat(self.num_cameras, 1, 1)
        return intrinsics, extrinsics

def get_dataloader(data_root, split='train', batch_size=1, num_workers=4, config=None):
    dataset = OccupancyDataset(data_root, split, config)
    shuffle = split == 'train'
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=shuffle)

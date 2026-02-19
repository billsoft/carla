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

        # 逐帧相机参数目录（每帧包含绝对世界坐标系下的外参，用于 ego_motion 计算）
        self._camera_params_dir = os.path.join(data_root, 'camera_params')
        self._has_per_frame_params = os.path.isdir(self._camera_params_dir)

        # 静态标定备用（仅在逐帧 npz 不存在时使用）
        self._cached_intrinsics, self._cached_extrinsics = self._load_static_calibration()

    # ------------------------------------------------------------------
    # 样本列表加载
    # ------------------------------------------------------------------

    def _load_samples(self):
        split_file = os.path.join(self.data_root, f'{self.split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        return []

    # ------------------------------------------------------------------
    # 静态标定（备用，相机安装参数，所有帧相同）
    # ------------------------------------------------------------------

    def _load_static_calibration(self):
        """加载 calibration/ 目录下的静态相机安装参数（备用）。
        注意：这些参数是固定的相机安装外参，不含车辆位姿，
        不能用于跨帧 ego_motion 计算。
        """
        calib_dir = os.path.join(self.data_root, 'calibration')
        int_path = os.path.join(calib_dir, 'intrinsics.json')
        ext_path = os.path.join(calib_dir, 'extrinsics.json')

        if not os.path.exists(int_path) or not os.path.exists(ext_path):
            return self._get_default_camera_params()

        with open(int_path, 'r') as f:
            int_data = json.load(f)
        with open(ext_path, 'r') as f:
            ext_data = json.load(f)

        K_list, E_list = [], []
        for i in range(self.num_cameras):
            cam_name = f'cam_{i}'
            cfg_int = int_data[cam_name]
            K = torch.eye(3)
            K[0, 0] = cfg_int['fx']
            K[1, 1] = cfg_int['fy']
            K[0, 2] = cfg_int['cx']
            K[1, 2] = cfg_int['cy']
            K_list.append(K)

            cfg_ext = ext_data[cam_name]
            E = torch.eye(4)
            E[:3, :3] = torch.tensor(cfg_ext['rotation_matrix'])
            E[:3, 3] = torch.tensor(cfg_ext['translation'])
            E_list.append(E)

        return torch.stack(K_list), torch.stack(E_list)

    def _get_default_camera_params(self):
        intrinsics = torch.eye(3).unsqueeze(0).repeat(self.num_cameras, 1, 1)
        intrinsics[:, 0, 0] = 800
        intrinsics[:, 1, 1] = 800
        intrinsics[:, 0, 2] = self.image_size[1] / 2
        intrinsics[:, 1, 2] = self.image_size[0] / 2
        extrinsics = torch.eye(4).unsqueeze(0).repeat(self.num_cameras, 1, 1)
        return intrinsics, extrinsics

    # ------------------------------------------------------------------
    # 逐帧相机参数（含车辆绝对位姿，用于 ego_motion 计算）
    # ------------------------------------------------------------------

    def _load_per_frame_params(self, sample_id):
        """加载逐帧相机参数 npz，包含当前帧车辆绝对位姿。

        npz 格式（由数据采集器 dense_occupancy_collection 生成）：
            intrinsics: [N, 3, 3]  相机内参
            extrinsics: [N, 4, 4]  World→Camera 变换（含当前帧车辆绝对位姿）

        相邻帧的 extrinsics 差值即为 ego_motion，这是跨帧时序对齐的关键。
        """
        npz_path = os.path.join(self._camera_params_dir, f'{sample_id}.npz')
        if not os.path.exists(npz_path):
            return None, None

        data = np.load(npz_path, allow_pickle=False)
        intrinsics = torch.from_numpy(data['intrinsics'].astype(np.float32))  # [N, 3, 3]
        extrinsics = torch.from_numpy(data['extrinsics'].astype(np.float32))  # [N, 4, 4]
        return intrinsics, extrinsics

    def _get_frame_params(self, sample_id):
        """获取帧级相机参数，优先使用逐帧 npz（含绝对位姿），否则退回静态标定。"""
        if self._has_per_frame_params:
            K, E = self._load_per_frame_params(sample_id)
            if K is not None:
                return K, E
        # 静态标定备用（ego_motion 将退化为 Identity，时序对齐失效，但不会崩溃）
        return self._cached_intrinsics, self._cached_extrinsics

    # ------------------------------------------------------------------
    # 图像加载
    # ------------------------------------------------------------------

    def _load_dng_image(self, dng_path):
        try:
            with rawpy.imread(dng_path) as raw:
                img = raw.raw_image_visible.astype(np.float32)
                img = img / 4095.0  # 12-bit normalize to [0, 1]
                img = img[np.newaxis, :, :]  # [1, H, W]
                return torch.from_numpy(img)
        except Exception as e:
            print(f"Error loading DNG {dng_path}: {e}")
            return torch.zeros(1, *self.image_size)

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self):
        if self.sequence_length > 1:
            return max(0, len(self.samples) - self.sequence_length + 1)
        return len(self.samples)

    def _load_single_frame(self, idx):
        sample_id = self.samples[idx]

        # 1. Load Images [N, 1, H, W]
        images = []
        for i in range(self.num_cameras):
            dng_path = os.path.join(self.data_root, 'images', sample_id, f'cam_{i}.dng')
            if os.path.exists(dng_path):
                img = self._load_dng_image(dng_path)
            else:
                img = torch.zeros(1, *self.image_size)
            images.append(img)
        images = torch.stack(images, dim=0).float()  # [N, 1, H, W]

        # 2. Load Voxels [X, Y, Z]
        occ_path = os.path.join(self.data_root, 'occupancy', f'{sample_id}.npy')
        if os.path.exists(occ_path):
            voxels = torch.from_numpy(np.load(occ_path)).long()
        else:
            voxels = torch.zeros(self.voxel_size, dtype=torch.long)

        # 3. 相机参数：优先逐帧 npz（含绝对位姿），否则使用静态标定
        intrinsics, extrinsics = self._get_frame_params(sample_id)

        return {
            'images': images,           # [N, 1, H, W]
            'voxels': voxels,           # [X, Y, Z]
            'intrinsics': intrinsics,   # [N, 3, 3]
            'extrinsics': extrinsics,   # [N, 4, 4]，逐帧不同（含绝对位姿）
        }

    def __getitem__(self, idx):
        if self.sequence_length == 1:
            return self._load_single_frame(idx)

        frames = [self._load_single_frame(idx + t) for t in range(self.sequence_length)]

        # intrinsics 假定相机内参恒定（只取第 0 帧）
        # extrinsics 每帧不同（含车辆绝对位姿），[T, N, 4, 4] 用于 ego_motion 计算
        return {
            'images': torch.stack([f['images'] for f in frames]),         # [T, N, C, H, W]
            'voxels': torch.stack([f['voxels'] for f in frames]),         # [T, X, Y, Z]
            'intrinsics': frames[0]['intrinsics'],                        # [N, 3, 3]
            'extrinsics': torch.stack([f['extrinsics'] for f in frames]), # [T, N, 4, 4]
        }


def get_dataloader(data_root, split='train', batch_size=1, num_workers=4, config=None):
    dataset = OccupancyDataset(data_root, split, config)
    shuffle = split == 'train'
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=shuffle
    )

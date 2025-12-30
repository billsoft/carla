# datasets/carla_occ_dataset.py
"""
CARLA Dense Occupancy Dataset 数据加载器

加载:
- 8个相机的 RGB 图像
- 3D 体素标注 (occupancy)
- 可见性掩码 (mask)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import torchvision.transforms as T


class CARLAOccDataset(Dataset):
    """
    CARLA 占用网格数据集
    
    数据结构:
        dataset_root/
        ├── cameras/
        │   ├── cam_front_main/
        │   │   ├── 000000.png
        │   │   └── ...
        │   ├── cam_front_wide/
        │   └── ...
        └── occupancy/
            ├── 000000.npz
            └── ...
    """
    
    # 相机ID列表（固定顺序）
    CAMERA_IDS = [
        'cam_front_main',
        'cam_front_wide', 
        'cam_front_narrow',
        'cam_left_pillar',
        'cam_right_pillar',
        'cam_left_repeater',
        'cam_right_repeater',
        'cam_rear',
    ]
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        img_size: Tuple[int, int] = (384, 640),
        grid_size: Tuple[int, int, int] = (200, 200, 16),
        transform: Optional[Callable] = None,
        load_depth: bool = False,
    ):
        """
        Args:
            data_root: 数据集根目录
            split: 'train', 'val', 'test'
            img_size: 图像缩放尺寸 (H, W)
            grid_size: 体素网格尺寸 (用于下采样标注)
            transform: 图像变换
            load_depth: 是否加载深度图（可选）
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.grid_size = grid_size
        self.load_depth = load_depth
        
        # 图像变换
        self.transform = transform or self._default_transform()
        
        # 获取所有帧索引
        self.frame_indices = self._get_frame_indices()
        
        # 数据集分割
        self._split_dataset()
        
        print(f"Loaded {len(self.frame_indices)} frames for {split}")
        
    def _default_transform(self) -> T.Compose:
        """默认图像变换"""
        return T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
    def _get_frame_indices(self) -> List[int]:
        """获取所有帧索引"""
        occ_dir = self.data_root / 'occupancy'
        
        if not occ_dir.exists():
            raise ValueError(f"Occupancy directory not found: {occ_dir}")
            
        indices = sorted([
            int(f.stem) for f in occ_dir.glob('*.npz')
        ])
        
        return indices
    
    def _split_dataset(self):
        """划分数据集"""
        total = len(self.frame_indices)
        
        # 80% train, 10% val, 10% test
        train_end = int(total * 0.8)
        val_end = int(total * 0.9)
        
        if self.split == 'train':
            self.frame_indices = self.frame_indices[:train_end]
        elif self.split == 'val':
            self.frame_indices = self.frame_indices[train_end:val_end]
        elif self.split == 'test':
            self.frame_indices = self.frame_indices[val_end:]
        # else: 使用全部数据
        
    def __len__(self) -> int:
        return len(self.frame_indices)
    
    def _load_image(self, cam_id: str, frame_idx: int) -> Image.Image:
        """加载单张图像"""
        img_path = self.data_root / 'cameras' / cam_id / f'{frame_idx:06d}.png'
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        return Image.open(img_path).convert('RGB')
    
    def _load_occupancy(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """加载体素标注"""
        occ_path = self.data_root / 'occupancy' / f'{frame_idx:06d}.npz'

        if not occ_path.exists():
            raise FileNotFoundError(f"Occupancy file not found: {occ_path}")

        data = np.load(occ_path)

        occupancy = data['occupancy']  # (500, 500, 40) uint8
        mask = data['mask']            # (500, 500, 40) bool

        return {
            'occupancy': occupancy,
            'mask': mask,
            'x_range': data['x_range'],
            'y_range': data['y_range'],
            'z_range': data['z_range'],
            'resolution': data['resolution'],
        }

    def _load_camera_params(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """
        加载相机参数

        Returns:
            dict with:
                - 'intrinsics': [8, 3, 3] 内参矩阵
                - 'extrinsics': [8, 4, 4] 外参矩阵（世界 -> 相机）
        """
        cam_params_path = self.data_root / 'camera_params' / f'{frame_idx:06d}.npz'

        if not cam_params_path.exists():
            raise FileNotFoundError(f"Camera params not found: {cam_params_path}")

        data = np.load(cam_params_path, allow_pickle=True)

        return {
            'intrinsics': data['intrinsics'],  # [8, 3, 3]
            'extrinsics': data['extrinsics'],  # [8, 4, 4]
        }
    
    def _downsample_occupancy(
        self,
        occupancy: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        下采样体素标注
        
        从 (500, 500, 40) 下采样到目标尺寸 (如 200, 200, 16)
        使用最近邻或多数投票
        """
        src_shape = occupancy.shape
        tgt_shape = self.grid_size
        
        if src_shape == tgt_shape:
            return occupancy, mask
            
        # 计算下采样因子
        scale_x = src_shape[0] // tgt_shape[0]
        scale_y = src_shape[1] // tgt_shape[1]
        scale_z = src_shape[2] // tgt_shape[2]
        
        # 使用区域多数投票下采样
        occ_ds = np.zeros(tgt_shape, dtype=np.uint8)
        mask_ds = np.zeros(tgt_shape, dtype=bool)
        
        for i in range(tgt_shape[0]):
            for j in range(tgt_shape[1]):
                for k in range(tgt_shape[2]):
                    # 获取对应的源区域
                    i_start, i_end = i * scale_x, (i + 1) * scale_x
                    j_start, j_end = j * scale_y, (j + 1) * scale_y
                    k_start, k_end = k * scale_z, (k + 1) * scale_z
                    
                    region_occ = occupancy[i_start:i_end, j_start:j_end, k_start:k_end]
                    region_mask = mask[i_start:i_end, j_start:j_end, k_start:k_end]
                    
                    # 多数投票（只考虑可见区域）
                    if region_mask.any():
                        valid_occ = region_occ[region_mask]
                        # 使用 bincount 找众数
                        counts = np.bincount(valid_occ.flatten(), minlength=18)
                        occ_ds[i, j, k] = counts.argmax()
                        mask_ds[i, j, k] = True
                    else:
                        occ_ds[i, j, k] = 0  # free
                        mask_ds[i, j, k] = False
                        
        return occ_ds, mask_ds
    
    def _downsample_occupancy_fast(
        self,
        occupancy: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        快速下采样（使用 resize）
        
        精度略低但速度快很多
        """
        from scipy.ndimage import zoom
        
        src_shape = occupancy.shape
        tgt_shape = self.grid_size
        
        if src_shape == tgt_shape:
            return occupancy, mask
            
        # 计算缩放因子
        scale = [t / s for t, s in zip(tgt_shape, src_shape)]
        
        # 最近邻下采样
        occ_ds = zoom(occupancy, scale, order=0)  # order=0 = nearest
        mask_ds = zoom(mask.astype(np.float32), scale, order=0) > 0.5
        
        return occ_ds.astype(np.uint8), mask_ds
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            dict with:
                - 'images': [8, 3, H, W] 多相机图像
                - 'occupancy': [X, Y, Z] 体素标注
                - 'mask': [X, Y, Z] 可见性掩码
                - 'frame_idx': 帧索引
        """
        frame_idx = self.frame_indices[idx]
        
        # 1. 加载 8 张图像
        images = []
        for cam_id in self.CAMERA_IDS:
            img = self._load_image(cam_id, frame_idx)
            img_tensor = self.transform(img)
            images.append(img_tensor)

        images = torch.stack(images, dim=0)  # [8, 3, H, W]

        # 2. 加载体素标注
        occ_data = self._load_occupancy(frame_idx)
        occupancy = occ_data['occupancy']
        mask = occ_data['mask']

        # 3. 加载相机参数
        cam_params = self._load_camera_params(frame_idx)
        intrinsics = torch.from_numpy(cam_params['intrinsics']).float()  # [8, 3, 3]
        extrinsics = torch.from_numpy(cam_params['extrinsics']).float()  # [8, 4, 4]

        # 4. 下采样到训练尺寸
        occupancy, mask = self._downsample_occupancy_fast(occupancy, mask)

        # 5. 转为 tensor
        occupancy = torch.from_numpy(occupancy).long()
        mask = torch.from_numpy(mask).bool()

        return {
            'images': images,
            'occupancy': occupancy,
            'mask': mask,
            'intrinsics': intrinsics,  # 新增
            'extrinsics': extrinsics,  # 新增
            'frame_idx': frame_idx,
        }


class CARLAOccDatasetWithAugmentation(CARLAOccDataset):
    """
    带数据增强的数据集
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        img_size: Tuple[int, int] = (384, 640),
        grid_size: Tuple[int, int, int] = (200, 200, 16),
        augment: bool = True,
    ):
        # 不使用默认 transform
        super().__init__(data_root, split, img_size, grid_size, transform=None)
        
        self.augment = augment and (split == 'train')
        
        # 初始化增强随机种子（修复多进程 DataLoader 问题）
        self._aug_seed = 0
        
        # 基础变换
        self.base_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
        ])
        
        # 归一化
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # 颜色增强
        self.color_aug = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.1),
        ])
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_idx = self.frame_indices[idx]
        
        # 为每个样本生成新的增强种子
        if self.augment:
            self._aug_seed = np.random.randint(0, 2**31)
        
        # 加载图像
        images = []
        for cam_id in self.CAMERA_IDS:
            img = self._load_image(cam_id, frame_idx)
            
            # 数据增强（所有相机使用相同的增强参数）
            if self.augment:
                torch.manual_seed(self._aug_seed)
                np.random.seed(self._aug_seed % (2**32))
                img = self.color_aug(img)
                
            img_tensor = self.base_transform(img)
            img_tensor = self.normalize(img_tensor)
            images.append(img_tensor)
            
        images = torch.stack(images, dim=0)
        
        # 加载标注
        occ_data = self._load_occupancy(frame_idx)
        occupancy, mask = self._downsample_occupancy_fast(
            occ_data['occupancy'], occ_data['mask']
        )
        
        # 加载相机参数
        cam_params = self._load_camera_params(frame_idx)
        intrinsics = torch.from_numpy(cam_params['intrinsics']).float()
        extrinsics = torch.from_numpy(cam_params['extrinsics']).float()
        
        # 空间增强：水平翻转
        if self.augment and np.random.random() < 0.5:
            images = torch.flip(images, dims=[-1])  # 水平翻转图像
            occupancy = np.flip(occupancy, axis=1).copy()  # Y轴翻转体素
            mask = np.flip(mask, axis=1).copy()
            
            # 交换左右相机
            # cam_left_pillar <-> cam_right_pillar
            # cam_left_repeater <-> cam_right_repeater
            images[[3, 4]] = images[[4, 3]]
            images[[5, 6]] = images[[6, 5]]

            # 翻转外参（Y轴翻转）
            # 外参是 world -> camera，需要对 world 坐标系进行翻转
            # T_cam_world_new = T_cam_world @ T_flip
            # T_flip = diag(1, -1, 1, 1)
            flip_mat = torch.tensor([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
            
            extrinsics = torch.matmul(extrinsics, flip_mat)
            
            # 交换左右相机的外参和内参
            intrinsics[[3, 4]] = intrinsics[[4, 3]]
            intrinsics[[5, 6]] = intrinsics[[6, 5]]
            extrinsics[[3, 4]] = extrinsics[[4, 3]]
            extrinsics[[5, 6]] = extrinsics[[6, 5]]
        
        return {
            'images': images,
            'occupancy': torch.from_numpy(occupancy).long(),
            'mask': torch.from_numpy(mask).bool(),
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'frame_idx': frame_idx,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义 collate 函数
    """
    return {
        'images': torch.stack([b['images'] for b in batch]),
        'occupancy': torch.stack([b['occupancy'] for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'intrinsics': torch.stack([b['intrinsics'] for b in batch]),
        'extrinsics': torch.stack([b['extrinsics'] for b in batch]),
        'frame_idx': torch.tensor([b['frame_idx'] for b in batch]),
    }


def build_dataloader(
    data_root: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (384, 640),
    grid_size: Tuple[int, int, int] = (200, 200, 16),
    augment: bool = True,
) -> DataLoader:
    """
    构建数据加载器
    """
    if augment and split == 'train':
        dataset = CARLAOccDatasetWithAugmentation(
            data_root=data_root,
            split=split,
            img_size=img_size,
            grid_size=grid_size,
            augment=True,
        )
    else:
        dataset = CARLAOccDataset(
            data_root=data_root,
            split=split,
            img_size=img_size,
            grid_size=grid_size,
        )
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
    
    return dataloader


# datasets/__init__.py 内容
__all__ = ['CARLAOccDataset', 'CARLAOccDatasetWithAugmentation', 'build_dataloader', 'collate_fn']


# 测试代码
if __name__ == '__main__':
    print("Testing CARLA Occ Dataset...")
    
    # 创建模拟数据
    import tempfile
    import shutil
    
    # 创建临时目录
    tmp_dir = tempfile.mkdtemp()
    print(f"Created temp directory: {tmp_dir}")
    
    try:
        # 创建目录结构
        occ_dir = Path(tmp_dir) / 'occupancy'
        occ_dir.mkdir()
        
        for cam_id in CARLAOccDataset.CAMERA_IDS:
            cam_dir = Path(tmp_dir) / 'cameras' / cam_id
            cam_dir.mkdir(parents=True)
            
        # 创建模拟数据
        for i in range(10):
            # 模拟图像
            for cam_id in CARLAOccDataset.CAMERA_IDS:
                img = Image.new('RGB', (640, 480), color=(128, 128, 128))
                img.save(Path(tmp_dir) / 'cameras' / cam_id / f'{i:06d}.png')
                
            # 模拟体素标注
            occ = np.random.randint(0, 18, (500, 500, 40), dtype=np.uint8)
            mask = np.random.random((500, 500, 40)) > 0.5
            
            np.savez_compressed(
                occ_dir / f'{i:06d}.npz',
                occupancy=occ,
                mask=mask,
                x_range=np.array([-50.0, 50.0]),
                y_range=np.array([-50.0, 50.0]),
                z_range=np.array([-4.0, 4.0]),
                resolution=np.array([0.2]),
            )
            
        # 测试数据集
        dataset = CARLAOccDataset(
            data_root=tmp_dir,
            split='all',
            img_size=(384, 640),
            grid_size=(200, 200, 16),
        )
        
        print(f"\nDataset length: {len(dataset)}")
        
        # 获取样本
        sample = dataset[0]
        
        print(f"Images shape: {sample['images'].shape}")
        print(f"Occupancy shape: {sample['occupancy'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Mask ratio: {sample['mask'].float().mean():.2%}")
        
        # 测试 DataLoader
        dataloader = build_dataloader(
            tmp_dir, 'all', batch_size=2, num_workers=0,
            grid_size=(200, 200, 16),
        )
        
        batch = next(iter(dataloader))
        print(f"\nBatch images shape: {batch['images'].shape}")
        print(f"Batch occupancy shape: {batch['occupancy'].shape}")
        
        print("\n✓ All tests passed!")
        
    finally:
        # 清理
        shutil.rmtree(tmp_dir)
        print(f"Cleaned up temp directory")

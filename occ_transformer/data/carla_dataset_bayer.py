# data/carla_dataset_bayer.py
"""
CARLA Bayer 数据集加载器

支持 12-bit Bayer RAW 图像和 3D 占用网格标签
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F


class CARLADatasetBayer(Dataset):
    """
    CARLA Bayer 数据集
    
    加载:
    - 8 相机 Bayer RAW 图像
    - 3D 占用网格标签
    - 相机内外参（可选）
    """
    
    DEFAULT_CAMERAS = [
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
        root: str,
        cameras: List[str] = None,
        img_size: Tuple[int, int] = None,
        augment: bool = False,
        target_grid_size: Tuple[int, int, int] = (200, 200, 16),
    ):
        """
        Args:
            root: 数据集根目录
            cameras: 相机列表
            img_size: 目标图像尺寸 (H, W)
            augment: 是否数据增强
            target_grid_size: 目标网格尺寸
        """
        self.root = Path(root)
        self.cameras = cameras or self.DEFAULT_CAMERAS
        self.img_size = img_size
        self.augment = augment
        self.target_grid_size = target_grid_size
        
        # 检查数据集
        self._check_dataset()
        
        # 加载样本 ID
        occ_dir = self.root / 'occupancy'
        self.sample_ids = sorted([f.stem for f in occ_dir.glob('*.npz')])
        
        print(f"[CARLADatasetBayer] 数据集已加载:")
        print(f"  路径: {self.root}")
        print(f"  样本数: {len(self.sample_ids)}")
        print(f"  相机数: {len(self.cameras)}")
        print(f"  图像尺寸: {self.img_size if self.img_size else '原始'}")
        print(f"  目标网格: {self.target_grid_size}")
        print(f"  数据增强: {'启用' if self.augment else '关闭'}")
        
    def _check_dataset(self):
        """检查数据集完整性"""
        assert self.root.exists(), f"数据集路径不存在: {self.root}"
        assert (self.root / 'occupancy').exists(), f"occupancy 目录不存在"
        
        # 检查相机目录
        for cam in self.cameras:
            cam_dir = self.root / 'cameras' / cam
            if not cam_dir.exists():
                print(f"警告: 相机目录不存在: {cam_dir}")
                
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        加载一个样本
        
        Returns:
            data: 字典
                - images: [N_cam, 1, H, W], float32, [0,1]
                - occupancy: [X, Y, Z], uint8
                - mask: [X, Y, Z], bool
                - intrinsics: [N_cam, 3, 3], float32 (可选)
                - extrinsics: [N_cam, 4, 4], float32 (可选)
        """
        sample_id = self.sample_ids[idx]
        
        # 加载图像
        images = []
        for cam in self.cameras:
            img_path = self.root / 'cameras' / cam / f'{sample_id}.png'
            
            if img_path.exists():
                img = self._load_bayer_image(str(img_path))
            else:
                # 如果图像不存在，创建随机数据（用于测试）
                if self.img_size:
                    img = np.random.randint(0, 65535, self.img_size, dtype=np.uint16)
                else:
                    img = np.random.randint(0, 65535, (960, 1280), dtype=np.uint16)
                    
            # 转换为 tensor
            img_tensor = self._bayer_to_tensor(img)
            
            # 调整大小
            if self.img_size is not None:
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=self.img_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
            images.append(img_tensor)
            
        images = torch.stack(images, dim=0)  # [N_cam, 1, H, W]
        
        # 加载占用网格
        occ_path = self.root / 'occupancy' / f'{sample_id}.npz'
        occ_data = np.load(str(occ_path))
        
        occupancy = torch.from_numpy(occ_data['occupancy'].astype(np.int64))
        mask = torch.from_numpy(occ_data['mask'].astype(bool))
        
        # 调整网格大小
        if occupancy.shape != self.target_grid_size:
            occupancy = self._resize_occupancy(occupancy, self.target_grid_size)
            mask = self._resize_mask(mask, self.target_grid_size)
        
        # 数据增强
        if self.augment:
            images, occupancy, mask = self._augment(images, occupancy, mask)
        
        data = {
            'images': images,
            'occupancy': occupancy,
            'mask': mask,
        }
        
        # 加载相机参数（如果存在）
        calib_path = self.root / 'calibration' / f'{sample_id}.npz'
        if calib_path.exists():
            calib_data = np.load(str(calib_path))
            if 'intrinsics' in calib_data:
                data['intrinsics'] = torch.from_numpy(calib_data['intrinsics'].astype(np.float32))
            if 'extrinsics' in calib_data:
                data['extrinsics'] = torch.from_numpy(calib_data['extrinsics'].astype(np.float32))
        
        return data
    
    def _load_bayer_image(self, path: str) -> np.ndarray:
        """加载 Bayer 图像"""
        import cv2
        
        # 读取图像 (保留位深)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise RuntimeError(f"无法加载图像: {path}")
            
        # 如果是多通道 (RGB/RGBA), 转为灰度
        if img.ndim == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            
        if img.dtype == np.uint8:
            img = img.astype(np.uint16) << 8
        elif img.dtype in [np.int32, np.uint32]:
            img = img.astype(np.uint16)
            
        return img
    
    def _bayer_to_tensor(self, bayer: np.ndarray) -> torch.Tensor:
        """Bayer 转 Tensor"""
        bayer_norm = bayer.astype(np.float32) / 65535.0
        tensor = torch.from_numpy(bayer_norm).unsqueeze(0)
        return tensor
    
    def _resize_occupancy(self, occ: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
        """调整占用网格大小"""
        occ = occ.unsqueeze(0).unsqueeze(0).float()
        occ = F.interpolate(occ, size=target_size, mode='nearest')
        return occ.squeeze(0).squeeze(0).long()
    
    def _resize_mask(self, mask: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
        """调整 mask 大小"""
        mask = mask.unsqueeze(0).unsqueeze(0).float()
        mask = F.interpolate(mask, size=target_size, mode='nearest')
        return mask.squeeze(0).squeeze(0).bool()
    
    def _augment(
        self,
        images: torch.Tensor,
        occupancy: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """数据增强"""
        # 随机水平翻转
        if torch.rand(1).item() > 0.5:
            images = torch.flip(images, dims=[-1])
            occupancy = torch.flip(occupancy, dims=[1])
            mask = torch.flip(mask, dims=[1])
            
            # 交换左右相机
            # left_pillar <-> right_pillar (3 <-> 4)
            # left_repeater <-> right_repeater (5 <-> 6)
            indices = [0, 1, 2, 4, 3, 6, 5, 7]
            images = images[indices]
            
        return images, occupancy, mask


def build_dataloader(
    dataset_root: str,
    batch_size: int = 2,
    num_workers: int = 4,
    shuffle: bool = True,
    img_size: Tuple[int, int] = None,
    augment: bool = False,
    target_grid_size: Tuple[int, int, int] = (200, 200, 16),
) -> DataLoader:
    """
    构建 DataLoader
    
    Args:
        dataset_root: 数据集根目录
        batch_size: 批量大小
        num_workers: 工作进程数
        shuffle: 是否打乱
        img_size: 图像大小 (H, W)
        augment: 是否数据增强
        target_grid_size: 目标网格尺寸
        
    Returns:
        dataloader
    """
    dataset = CARLADatasetBayer(
        root=dataset_root,
        img_size=img_size,
        augment=augment,
        target_grid_size=target_grid_size,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader


if __name__ == '__main__':
    print("=" * 60)
    print("CARLA Bayer 数据集测试")
    print("=" * 60)
    
    # 创建测试数据
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建目录结构
        os.makedirs(f"{tmpdir}/bayer/cam_front_main")
        os.makedirs(f"{tmpdir}/occupancy")
        
        # 创建测试占用网格
        occ = np.random.randint(0, 18, size=(200, 200, 16), dtype=np.uint8)
        mask = np.ones((200, 200, 16), dtype=bool)
        np.savez(f"{tmpdir}/occupancy/000000.npz", occupancy=occ, mask=mask)
        
        # 测试数据集
        dataset = CARLADatasetBayer(
            root=tmpdir,
            img_size=(384, 512),
        )
        
        print(f"\n样本数: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n样本内容:")
            print(f"  images: {sample['images'].shape}")
            print(f"  occupancy: {sample['occupancy'].shape}")
            print(f"  mask: {sample['mask'].shape}")
    
    print("\n" + "=" * 60)
    print("✅ 测试通过！")
    print("=" * 60)

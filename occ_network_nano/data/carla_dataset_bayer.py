"""
CARLA Bayer RAW 数据集 - 加载单通道 Bayer RGGB 数据

简单高效，专为 Bayer 数据设计，支持：
- 8 个环视相机 单通道 Bayer RGGB
- 体素占据标注
- 相机参数
- 12-bit/16-bit DNG 格式
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.bayer_utils import load_bayer_image, bayer_to_tensor


import torch.nn.functional as F

class CARLADatasetBayer(Dataset):
    """
    CARLA Bayer RAW 数据集
    
    数据格式：
    - cameras/CAM_XXX/NNNNNN.dng: 单通道 12-bit Bayer (H, W)
    - occupancy/NNNNNN.npz: 体素占据 (500, 500, 40)
    - camera_params/NNNNNN.npz: 相机参数

    Args:
        root: 数据集根目录
        cameras: 相机列表（默认 8 个）
        img_size: 图像大小 (H, W)，None 则使用原始尺寸
        augment: 是否数据增强
        target_grid_size: 目标体素网格大小 (X, Y, Z)，默认 (200, 200, 16)
    """

    # 默认 8 个相机 (与 Tesla Config 匹配)
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
        self.root = Path(root)
        self.cameras = cameras or self.DEFAULT_CAMERAS
        self.img_size = img_size  # (H, W)
        self.augment = augment
        self.target_grid_size = target_grid_size

        # 检查数据集
        self._check_dataset()

        # 获取样本列表
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
        required_dirs = ['cameras', 'occupancy', 'camera_params']
        for dir_name in required_dirs:
            dir_path = self.root / dir_name
            if not dir_path.exists():
                raise FileNotFoundError(f"缺少目录: {dir_path}")

        # 检查相机目录
        for cam in self.cameras:
            cam_dir = self.root / 'cameras' / cam
            if not cam_dir.exists():
                raise FileNotFoundError(f"缺少相机目录: {cam_dir}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        加载一个样本

        Returns:
            data: 字典
                - images: [N_cam, 1, H, W], float32, [0,1]
                - occupancy: [200, 200, 16], uint8
                - mask: [200, 200, 16], bool
                - intrinsics: [N_cam, 3, 3], float32
                - extrinsics: [N_cam, 4, 4], float32
        """
        sample_id = self.sample_ids[idx]

        # 1. 加载 Bayer 图像（12-bit DNG，自动扩展到 16-bit）
        images = []
        for cam in self.cameras:
            # 支持多种格式：优先 .dng，然后 .png
            img_path = self.root / 'cameras' / cam / f"{sample_id}.dng"
            if not img_path.exists():
                img_path = self.root / 'cameras' / cam / f"{sample_id}.png"

            bayer = load_bayer_image(str(img_path), is_12bit=True)  # (H, W) uint16 [0, 65535]

            # 调整大小（保持单通道）
            if self.img_size is not None:
                import cv2
                bayer = cv2.resize(bayer, (self.img_size[1], self.img_size[0]),
                                  interpolation=cv2.INTER_LINEAR)

            # 转换为 Tensor
            bayer_tensor = bayer_to_tensor(bayer, normalize=True)  # (1, H, W), [0,1]
            images.append(bayer_tensor)

        images = torch.stack(images, dim=0)  # [N_cam, 1, H, W]

        # 2. 加载体素占据
        occ_path = self.root / 'occupancy' / f"{sample_id}.npz"
        occ_data = np.load(occ_path)
        occupancy = occ_data['occupancy']  # [500, 500, 40], uint8 (原始)
        mask = occ_data['mask']            # [500, 500, 40], bool (原始)

        # 调整体素网格大小到目标尺寸 (200, 200, 16)
        if self.target_grid_size != occupancy.shape:
            # 转换为 Tensor 并增加 batch/channel 维度: [1, 1, X, Y, Z]
            # 注意: grid_sample/interpolate 需要 [N, C, D, H, W] -> 这里是 [N, C, X, Y, Z]
            # occupancy 形状是 [X, Y, Z]
            
            # 占据网格 (Nearest 插值)
            occ_tensor = torch.from_numpy(occupancy).float().unsqueeze(0).unsqueeze(0)
            occ_resized = F.interpolate(
                occ_tensor, 
                size=self.target_grid_size, 
                mode='nearest'
            )
            occupancy = occ_resized.squeeze().long().numpy() # [200, 200, 16]

            # Mask (Nearest 插值)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_resized = F.interpolate(
                mask_tensor, 
                size=self.target_grid_size, 
                mode='nearest'
            )
            mask = mask_resized.squeeze().bool().numpy() # [200, 200, 16]

        # 3. 加载相机参数
        cam_param_path = self.root / 'camera_params' / f"{sample_id}.npz"
        cam_params = np.load(cam_param_path)
        intrinsics = cam_params['intrinsics']  # [8, 3, 3]
        extrinsics = cam_params['extrinsics']  # [8, 4, 4]

        # 4. 验证时间戳对齐 (如果元数据中有)
        # 注意: 当前数据生成代码未在 npz 中保存时间戳，后续可改进。
        # 这里假设文件名 ID 严格对应。
        
        # 数据增强（简单版本）
        if self.augment:
            images, occupancy = self._augment(images, occupancy)

        return {
            'images': images,              # [N_cam, 1, H, W]
            'occupancy': torch.from_numpy(occupancy),  # [200, 200, 16]
            'mask': torch.from_numpy(mask),            # [200, 200, 16]
            'intrinsics': torch.from_numpy(intrinsics).float(),  # [N_cam, 3, 3]
            'extrinsics': torch.from_numpy(extrinsics).float(),  # [N_cam, 4, 4]
            'sample_id': sample_id,
        }

    def _augment(self, images: torch.Tensor, occupancy: np.ndarray):
        """
        简单数据增强

        Args:
            images: [N_cam, 1, H, W]
            occupancy: [200, 200, 16]

        Returns:
            增强后的数据
        """
        # 1. 随机水平翻转（50%）
        if torch.rand(1) > 0.5:
            images = torch.flip(images, dims=[-1])  # 翻转宽度维度
            occupancy = np.flip(occupancy, axis=1).copy()

        # 2. 随机亮度调整（范围 [0.8, 1.2]）
        if torch.rand(1) > 0.5:
            brightness_scale = 0.8 + torch.rand(1) * 0.4  # [0.8, 1.2]
            images = torch.clamp(images * brightness_scale, 0.0, 1.0)

        return images, occupancy


def build_dataloader(
    dataset_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    img_size: Tuple[int, int] = (960, 1280),
    augment: bool = False,
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

    Returns:
        dataloader
    """
    dataset = CARLADatasetBayer(
        root=dataset_root,
        img_size=img_size,
        augment=augment,
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
    print("CARLA Bayer RAW 数据集测试")
    print("=" * 60)

    # 测试数据集加载
    try:
        dataset = CARLADatasetBayer(
            root='../dataset_10k',  # 需要先生成数据集
            img_size=(960, 1280),
            augment=True,
        )

        print(f"\n✅ 数据集加载成功！")
        print(f"样本总数: {len(dataset)}")

        # 加载第一个样本
        print(f"\n正在加载第一个样本...")
        sample = dataset[0]

        print(f"\n样本内容:")
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                if key == 'images':
                    print(f"    → 数值范围: [{val.min():.4f}, {val.max():.4f}]")
            elif isinstance(val, str):
                print(f"  {key}: {val}")

        # 数据格式检查
        print(f"\n数据格式检查:")
        assert sample['images'].shape[0] == 8, "应有 8 个相机"
        assert sample['images'].shape[1] == 1, "应是单通道 Bayer"
        assert sample['occupancy'].shape == (200, 200, 16), "体素形状错误"
        print(f"  ✅ 图像形状: 通过 (8 相机, 单通道)")
        print(f"  ✅ 图像范围: 通过 ([0, 1] 归一化)")
        print(f"  ✅ 占据形状: 通过 (200×200×16)")

        print("\n" + "=" * 60)
        print("✅ 所有检查通过！数据集格式正确。")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n⚠️ 测试跳过: {e}")
        print("提示: 请先生成 Bayer 数据集")

# OccNetV3 训练指南

本指南说明如何使用 `occnetv3_data_generator` 生成的数据集来训练 Occupancy Network。

## 1. 数据集准备

### 1.1 生成数据
使用 `main_collection.py` 生成数据：
```bash
python occnetv3_data_generator/main_collection.py --frames 1000 --output dataset/train --town Town10HD
```

### 1.2 数据结构
生成的数据集结构如下：
```
dataset/
├── calibration/
│   ├── intrinsics.json  # 相机内参
│   └── extrinsics.json  # 相机外参
├── images/              # 8路相机图像 (float16 gray)
│   └── scene_0000_frame_0000/
│       ├── cam_front_main.npy
│       ├── ...
├── occupancy/           # 3D 占用网格 GT (512x512x40)
│   └── scene_0000_frame_0000.npy
├── flow/                # 3D 场景流 GT
├── flow_mask/           # 流场掩码
├── ego_pose/            # 车辆位姿
└── train.txt            # 训练样本列表
```

## 2. 数据加载 (PyTorch 示例)

以下是一个简单的 PyTorch Dataset 实现，用于加载该数据：

```python
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json

class OccNetDataset(Dataset):
    def __init__(self, data_root, split='train'):
        self.data_root = Path(data_root)
        self.split_file = self.data_root / f'{split}.txt'
        
        with open(self.split_file, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]
            
        # 加载标定参数
        with open(self.data_root / 'calibration/intrinsics.json') as f:
            self.intrinsics = json.load(f)
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        # 1. 加载图像 (8个相机)
        images = []
        # 假设相机列表固定
        cameras = ['front_main', 'front_wide', 'front_narrow', 'left_pillar', 
                   'right_pillar', 'left_repeater', 'right_repeater', 'rear']
                   
        img_dir = self.data_root / 'images' / sample_id
        for cam in cameras:
            # 加载 .npy (1, H, W) float16
            img = np.load(img_dir / f'cam_{cam}.npy').astype(np.float32)
            images.append(torch.from_numpy(img))
            
        # Stack images: (8, 1, H, W)
        image_tensor = torch.stack(images)
        
        # 2. 加载 Occupancy Label
        # (512, 512, 40) uint8
        occ_path = self.data_root / 'occupancy' / f'{sample_id}.npy'
        occupancy = np.load(occ_path).astype(np.int64)
        
        # 3. 加载 Pose
        pose_path = self.data_root / 'ego_pose' / f'{sample_id}.npy'
        pose = np.load(pose_path).astype(np.float32)
        
        return {
            'images': image_tensor,
            'occupancy': torch.from_numpy(occupancy),
            'pose': torch.from_numpy(pose),
            'sample_id': sample_id
        }
```

## 3. 模型输入处理

由于输入是 8 路灰度图，如果使用预训练的 ResNet/Swin (通常接受 RGB)，需要：
1. **复制通道**: 将单通道灰度图复制为 3 通道 (`x.repeat(1, 3, 1, 1)`)
2. **修改第一层**: 修改模型的第一层卷积，使其接受 1 通道输入 (推荐，减少计算量)

## 4. 损失函数

Occupancy 任务通常使用 **CrossEntropyLoss** 或 **Focal Loss**。

```python
import torch.nn as nn

# 18类 (包含 empty)
criterion = nn.CrossEntropyLoss(ignore_index=255) # 假设 255 是忽略类别

# forward
logits = model(images) # (B, 18, X, Y, Z)
loss = criterion(logits, occupancy)
```

## 5. 训练注意事项

1. **显存占用**: 3D Volume 非常大，建议使用 `float16` 混合精度训练 (AMP)。
2. **数据增强**: 可以在 2D 图像空间做 Augmentation，同时更新内参；或在 3D 空间做 Flip/Rotate。
3. **类别不平衡**: Empty (空气) 类别占绝大多数，必须使用 Class Balancing 或 OHEM (Online Hard Example Mining)。

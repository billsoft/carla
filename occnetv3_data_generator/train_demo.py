
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import os

class OccNetDataset(Dataset):
    """
    OccNetV3 数据集加载器示例
    """
    def __init__(self, data_root, split='train'):
        self.data_root = Path(data_root)
        self.split = split
        
        # 简单起见，这里直接遍历目录，实际应读取 split 文件
        self.samples = sorted([p.stem for p in (self.data_root / 'occupancy').glob('*.npy')])
        
        # 加载标定参数 (示例)
        self.intrinsics = {}
        if (self.data_root / 'calibration/intrinsics.json').exists():
            with open(self.data_root / 'calibration/intrinsics.json') as f:
                self.intrinsics = json.load(f)

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        # 1. 加载图像 (8个相机)
        # 假设相机列表固定
        cameras = ['front_main', 'front_wide', 'front_narrow', 'left_pillar', 
                   'right_pillar', 'left_repeater', 'right_repeater', 'rear']
        
        images = []
        # 注意: 图片在 images/<sample_id>/cam_<id>.npy
        img_dir = self.data_root / 'images' / sample_id
        
        for cam in cameras:
            # 加载 .npy (1, H, W) float16
            # 实际训练中可能需要 Resize 到更小尺寸 (e.g. 256x704)
            img_path = img_dir / f'cam_{cam}.npy'
            if img_path.exists():
                img = np.load(img_path).astype(np.float32)
            else:
                # Fallback: 全0
                img = np.zeros((1, 960, 1280), dtype=np.float32)
                
            images.append(torch.from_numpy(img))
            
        # Stack images: (B, N, C, H, W) -> 这里是 (8, 1, 960, 1280)
        image_tensor = torch.stack(images)
        
        # 2. 加载 Occupancy Label
        # (512, 512, 40) uint8 -> (512, 512, 40) long
        occ_path = self.data_root / 'occupancy' / f'{sample_id}.npy'
        occupancy = np.load(occ_path).astype(np.int64)
        
        return {
            'images': image_tensor,
            'occupancy': torch.from_numpy(occupancy),
            'sample_id': sample_id
        }

class SimpleOccNet(nn.Module):
    """
    简单的 Occupancy Network 示例模型
    """
    def __init__(self, num_classes=18):
        super().__init__()
        # 假设输入特征提取后被投影到 BEV/Voxel
        # 这里仅演示输出头
        self.voxel_head = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # x: images (B, 8, 1, H, W)
        # 这里省略复杂的 2D->3D 变换
        # 假设我们直接生成一个随机的 Voxel Feature (B, 64, 128, 128, 10)
        # 然后上采样到 (512, 512, 40)
        
        B = x.shape[0]
        dummy_voxel_feat = torch.randn(B, 64, 128, 128, 10).to(x.device)
        
        out = self.voxel_head(dummy_voxel_feat)
        
        # 上采样到目标尺寸
        out = torch.nn.functional.interpolate(out, size=(512, 512, 40), mode='trilinear', align_corners=False)
        return out

def train():
    # 配置
    DATA_ROOT = r'D:\code\carla\dataset_10k_bak' # 你的数据目录
    BATCH_SIZE = 1
    LR = 1e-4
    EPOCHS = 2
    
    if not os.path.exists(DATA_ROOT):
        print(f"数据目录不存在: {DATA_ROOT}")
        return

    # 数据集
    dataset = OccNetDataset(DATA_ROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 模型
    model = SimpleOccNet(num_classes=18).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 假设0是空闲空间，如果不想计算loss的话
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print("开始训练...")
    model.train()
    
    for epoch in range(EPOCHS):
        for i, batch in enumerate(loader):
            images = batch['images'].cuda()
            occupancy = batch['occupancy'].cuda() # (B, 512, 512, 40)
            
            optimizer.zero_grad()
            
            # Forward
            logits = model(images) # (B, 18, 512, 512, 40)
            
            # Loss
            loss = criterion(logits, occupancy)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train()

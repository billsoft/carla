import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        
        # Classification First Approach (Low Memory)
        # Input: [B, dim, 80, 80, 16]
        
        # 1. Reduce channels first
        self.conv1 = nn.Sequential(
            nn.Conv3d(dim, dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.GELU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(dim // 4),
            nn.GELU(),
        )
        
        # 2. Classify at low resolution
        # [B, 64, 80, 80, 16] -> [B, 18, 80, 80, 16]
        self.cls_head = nn.Conv3d(dim // 4, config.num_classes, kernel_size=1)
        
        # 3. 上采样精化层（两步：80->200->400，每步插值后卷积残差精化）
        self.refine1 = nn.Sequential(
            nn.Conv3d(config.num_classes, config.num_classes, kernel_size=3, padding=1),
            nn.BatchNorm3d(config.num_classes),
            nn.ReLU(inplace=True)
        )
        # refine2 在最终 logits 层，不加激活函数，加 BN 稳定训练
        self.refine2 = nn.Sequential(
            nn.Conv3d(config.num_classes, config.num_classes, kernel_size=3, padding=1),
            nn.BatchNorm3d(config.num_classes),
        )

    def forward(self, x):
        # x: [B, fx, fy, fz, C] (from OccDecoder)
        # Permute to [B, C, fx, fy, fz] for Conv3d/Interpolate
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # 1. 降维
        x = self.conv1(x)  # -> [B, 128, 80, 80, 16]
        x = self.conv2(x)  # -> [B, 64, 80, 80, 16]

        # 2. 低分辨率分类
        logits_small = self.cls_head(x)  # -> [B, 18, 80, 80, 16]

        # 3. 两步上采样 + 可学习精化（尺寸从 config.voxel_size 派生，修改 config 自动生效）
        vx, vy, vz = self.config.voxel_size          # e.g. (400, 400, 32)
        mid_size = (vx // 2, vy // 2, vz)            # e.g. (200, 200, 32)

        # Step 1: fine_size -> mid_size
        logits_mid = F.interpolate(logits_small, size=mid_size, mode='trilinear', align_corners=False)
        logits_mid = self.refine1(logits_mid) + logits_mid

        # Step 2: mid_size -> voxel_size
        logits_final = F.interpolate(logits_mid, size=(vx, vy, vz), mode='trilinear', align_corners=False)
        logits_final = logits_final + self.refine2(logits_final)

        return logits_final


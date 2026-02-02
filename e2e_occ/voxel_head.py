import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        
        # Simplified Head: Trilinear Interpolate + Conv
        # Reduced channels progressively
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
        self.refine = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(dim // 4),
            nn.GELU(),
        )
        self.cls_head = nn.Conv3d(dim // 4, config.num_classes, kernel_size=1)
    
    def forward(self, x):
        # x: [B, C, fx, fy, fz]
        # First upsample block
        # Target size? 
        # Fine: 80x80x12 -> 400x400x32
        # Scale: 5x, 5x, 2.6x?
        # Just interpolate directly to final size is too aggressive?
        # Maybe two steps or direct.
        # XuGong suggested: interpolate to voxel_size directly.
        
        target_size = self.config.voxel_size # (400, 400, 32)
        
        # 1. Project channels first (Heavy computation if done on 400^3)
        # So reduce channels on small resolution
        x = self.conv1(x) # dim -> dim/2
        x = self.conv2(x) # dim/2 -> dim/4
        
        # 2. Upsample to final resolution
        x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)
        
        # 3. Light Refine
        x = self.refine(x)
        
        # 4. Classify
        x = self.cls_head(x)
        
        return x

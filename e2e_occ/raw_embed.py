import torch
import torch.nn as nn
import torch.nn.functional as F

class RAWPatchEmbed(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Replace manual RGGB unpacking with learnable convolution
        # Input: 1 channel (RAW), Output: 4 channels (RGGB-like features)
        # Kernel: 2x2, Stride: 2 -> Halves H/W, quadruples channels
        self.rggb_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        
        self.stem = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
        )
    
    def forward(self, x):
        B, N, C, H, W = x.shape
        # Flatten batch and camera dimensions for convolution
        x = x.view(B * N, C, H, W)
        
        # Apply learnable RGGB downsampling
        x = self.rggb_conv(x)
        
        # Apply Stem
        x = self.stem(x)
        
        _, D, Hf, Wf = x.shape
        return x.view(B, N, D, Hf, Wf)

class MultiCameraPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = RAWPatchEmbed(embed_dim=config.embed_dim)
        self.norm = nn.LayerNorm(config.embed_dim)
    
    def forward(self, images):
        B, N, C, H, W = images.shape
        feats = self.patch_embed(images)
        feats = feats.permute(0, 1, 3, 4, 2)
        feats = self.norm(feats)
        feats = feats.permute(0, 1, 4, 2, 3)
        return feats

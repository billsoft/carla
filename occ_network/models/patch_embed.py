import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(960, 1280), patch_size=16, in_channels=1, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)

class HybridPatchEmbed(nn.Module):
    def __init__(self, img_size=(960, 1280), patch_size=16, in_channels=1, embed_dim=192):
        super().__init__()
        # Bayer RAW Stem: 4-stage downsampling to reach stride 16
        # Stage 1: RGGB Merge (Stride 2)
        # Input: [B, 1, H, W] -> Output: [B, 32, H/2, W/2]
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        
        # Stage 2: Stride 4
        # Input: [B, 32, H/2, W/2] -> Output: [B, 64, H/4, W/4]
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # Stage 3: Stride 8
        # Input: [B, 64, H/4, W/4] -> Output: [B, 128, H/8, W/8]
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        # Stage 4: Projection (Stride 16)
        # Input: [B, 128, H/8, W/8] -> Output: [B, embed_dim, H/16, W/16]
        self.proj = nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)

class MultiCameraPatchEmbed(nn.Module):
    def __init__(self, img_size=(960, 1280), patch_size=16, in_channels=1, embed_dim=192, num_cameras=8):
        super().__init__()
        self.num_cameras = num_cameras
        self.patch_embed = HybridPatchEmbed(img_size, patch_size, in_channels, embed_dim)

    def forward(self, x):
        B, N, C, H, W = x.shape
        all_tokens, spatial_shape = [], None
        for cam_idx in range(N):
            tokens, shape = self.patch_embed(x[:, cam_idx])
            all_tokens.append(tokens)
            if spatial_shape is None:
                spatial_shape = shape
        return all_tokens, spatial_shape

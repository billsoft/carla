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
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
        )
        proj_patch = patch_size // 4
        self.proj = nn.Conv2d(64, embed_dim, kernel_size=proj_patch, stride=proj_patch)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.stem(x)
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

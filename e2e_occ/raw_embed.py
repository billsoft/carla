import torch
import torch.nn as nn
import torch.nn.functional as F

class RGGBUnpack(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        r = x[:, :, 0::2, 0::2]
        g1 = x[:, :, 0::2, 1::2]
        g2 = x[:, :, 1::2, 0::2]
        b = x[:, :, 1::2, 1::2]
        out = torch.cat([r, g1, g2, b], dim=1)
        _, C4, H2, W2 = out.shape
        return out.view(B, N, C4, H2, W2)

class RAWPatchEmbed(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.rggb_unpack = RGGBUnpack()
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
        x = self.rggb_unpack(x)
        _, _, C4, H2, W2 = x.shape
        x = x.view(B * N, C4, H2, W2)
        x = self.stem(x)
        _, D, Hf, Wf = x.shape
        return x.view(B, N, D, Hf, Wf)

class MultiCameraPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = RAWPatchEmbed(embed_dim=config.embed_dim)
        # Removed redundant Camera Embedding (covered by Ray Encoding)
        # self.camera_embed = nn.Embedding(config.num_cameras, config.embed_dim)
        self.norm = nn.LayerNorm(config.embed_dim)
    
    def forward(self, images):
        B, N, C, H, W = images.shape
        feats = self.patch_embed(images)
        _, _, D, Hf, Wf = feats.shape
        # camera_ids = torch.arange(N, device=images.device)
        # cam_embed = self.camera_embed(camera_ids).view(1, N, D, 1, 1)
        # feats = feats + cam_embed
        feats = feats.permute(0, 1, 3, 4, 2)
        feats = self.norm(feats)
        feats = feats.permute(0, 1, 4, 2, 3)
        return feats

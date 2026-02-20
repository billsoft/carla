import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
try:
    # 优先相对导入（作为包使用时）
    from .position_encoding import RayDirectionEncoding
except (ImportError, ValueError):
    # 回退到绝对导入（作为独立脚本运行时）
    from position_encoding import RayDirectionEncoding

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, H, W, C = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        qkv = self.qkv(x).reshape(-1, ws * ws, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        x = self.proj(x)
        x = x.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        return x[:, :H, :W, :]

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, window_size=7, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.use_ray_encoding:
            self.ray_embed = RayDirectionEncoding(config.embed_dim, config.feat_size, config.num_cameras)
            
        self.blocks = nn.ModuleList([
            EncoderBlock(config.embed_dim, config.num_heads, dropout=config.dropout)
            for _ in range(config.encoder_layers)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Disable checkpoint for Encoder as per plan
        self.use_checkpoint = False
    
    def _process_single_camera(self, x, rays=None):
        x = x.squeeze(1)
        if rays is not None:
            x = x + rays.squeeze(1)
            
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x, intrinsics=None, extrinsics=None):
        B, N, C, H, W = x.shape
        ray_feat = None
        if self.config.use_ray_encoding and intrinsics is not None:
            ray_feat = self.ray_embed(x, intrinsics, extrinsics)
            ray_feat = ray_feat.permute(0, 1, 3, 4, 2)
            
        x = x.permute(0, 1, 3, 4, 2)
        
        outs = []
        for i in range(N):
            x_cam = x[:, i:i+1] 
            ray_cam = ray_feat[:, i:i+1] if ray_feat is not None else None
            
            if self.use_checkpoint and self.training:
                out_cam = checkpoint(self._process_single_camera, x_cam, ray_cam, use_reentrant=False)
            else:
                out_cam = self._process_single_camera(x_cam, ray_cam)
                
            outs.append(out_cam)
            
        x = torch.stack(outs, dim=1)
        x = self.norm(x)
        x = x.permute(0, 1, 4, 2, 3)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from position_encoding import SineCosinePositionEncoding2D, RayDirectionEncoding

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
        self.pos_embed = SineCosinePositionEncoding2D(config.embed_dim)
        self.ray_embed = RayDirectionEncoding(config.embed_dim, config.feat_size, config.num_cameras)
        self.blocks = nn.ModuleList([
            EncoderBlock(config.embed_dim, config.num_heads, dropout=config.dropout)
            for _ in range(config.encoder_layers)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Checkpoint flag
        self.use_checkpoint = True
    
    def _process_single_camera(self, x, pos, rays=None):
        # x: [B, 1, H, W, C] -> [B, H, W, C]
        x = x.squeeze(1)
        
        # Add PE
        x = x + pos.unsqueeze(0)
        
        # Add Ray PE
        if rays is not None:
            # rays: [B, 1, H, W, C] -> [B, H, W, C]
            x = x + rays.squeeze(1)
            
        # Blocks
        for block in self.blocks:
            x = block(x)
            
        return x

    def forward(self, x, intrinsics=None, extrinsics=None):
        B, N, C, H, W = x.shape
        
        # Prepare 2D PE (Shared)
        pos = self.pos_embed(H, W, x.device) 
        if pos.dim() == 2:
            pos = pos.view(H, W, C)
            
        # Prepare Ray PE (Batched computation is light)
        ray_feat = None
        if intrinsics is not None:
            # [B, N, C, H, W]
            ray_feat = self.ray_embed(x, intrinsics, extrinsics)
            ray_feat = ray_feat.permute(0, 1, 3, 4, 2) # [B, N, H, W, C]
            
        x = x.permute(0, 1, 3, 4, 2) # [B, N, H, W, C]
        
        # Serial Processing Loop with Checkpointing
        outs = []
        for i in range(N):
            x_cam = x[:, i:i+1] # Keep dim for consistency or slice?
            # x_cam: [B, 1, H, W, C]
            
            ray_cam = ray_feat[:, i:i+1] if ray_feat is not None else None
            
            if self.use_checkpoint and self.training:
                # Checkpoint requires inputs to have grad. 
                # x has grad. pos usually doesn't (buffer). ray_cam has grad?
                # ray_feat comes from ray_embed(x...), so yes.
                
                # Note: checkpointing simple functions might add overhead, 
                # but here _process_single_camera contains multiple Transformer blocks.
                out_cam = checkpoint(self._process_single_camera, x_cam, pos, ray_cam, use_reentrant=False)
            else:
                out_cam = self._process_single_camera(x_cam, pos, ray_cam)
                
            outs.append(out_cam)
            
        x = torch.stack(outs, dim=1) # [B, N, H, W, C]
        
        x = self.norm(x)
        x = x.permute(0, 1, 4, 2, 3) # [B, N, C, H, W]
        return x

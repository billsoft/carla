import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, List

class SineCosinePositionEncoding2D(nn.Module):
    def __init__(self, dim, temperature=10000):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
    
    def forward(self, h, w, device):
        y_pos = torch.arange(h, device=device).float().unsqueeze(1).repeat(1, w)
        x_pos = torch.arange(w, device=device).float().unsqueeze(0).repeat(h, 1)
        dim_t = torch.arange(self.dim // 4, device=device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.dim // 4))
        pos_x = x_pos.unsqueeze(-1) / dim_t
        pos_y = y_pos.unsqueeze(-1) / dim_t
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)
        return torch.cat([pos_x, pos_y], dim=-1)

class SineCosinePositionEncoding3D(nn.Module):
    def __init__(self, dim, temperature=10000):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
    
    def forward(self, x, y, z, device):
        d = self.dim // 6
        dim_t = torch.arange(d, device=device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / d)
        xs = torch.arange(x, device=device).float()
        ys = torch.arange(y, device=device).float()
        zs = torch.arange(z, device=device).float()
        pos_x = xs.view(-1, 1, 1, 1) / dim_t.view(1, 1, 1, -1)
        pos_y = ys.view(1, -1, 1, 1) / dim_t.view(1, 1, 1, -1)
        pos_z = zs.view(1, 1, -1, 1) / dim_t.view(1, 1, 1, -1)
        px = torch.stack([pos_x.sin(), pos_x.cos()], -1).flatten(-2)
        py = torch.stack([pos_y.sin(), pos_y.cos()], -1).flatten(-2)
        pz = torch.stack([pos_z.sin(), pos_z.cos()], -1).flatten(-2)
        px = px.expand(x, y, z, -1)
        py = py.expand(x, y, z, -1)
        pz = pz.expand(x, y, z, -1)
        pe = torch.cat([px, py, pz], dim=-1)
        if pe.shape[-1] < self.dim:
            pe = F.pad(pe, (0, self.dim - pe.shape[-1]))
        return pe.view(-1, self.dim)

class LearnedCameraEmbedding(nn.Module):
    def __init__(self, num_cameras, dim):
        super().__init__()
        self.embed = nn.Embedding(num_cameras, dim)
    
    def forward(self, camera_ids):
        return self.embed(camera_ids)

class RayDirectionEncoding(nn.Module):
    """
    射线方向编码 (Restored)
    """

    def __init__(
        self,
        dim: int,
        image_size: Tuple[int, int],
        num_cameras: int = 8,
        num_freqs: int = 10,
    ):
        super().__init__()
        self.dim = dim
        self.image_size = image_size
        self.num_freqs = num_freqs
        self.num_cameras = num_cameras

        # Input: 3 (x, y, z)
        # Encoded: 3 + 3 * 2 * num_freqs
        self.input_dim = 3 + 3 * 2 * num_freqs
        
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
    
    def _sinusoidal_encoding(self, x: torch.Tensor) -> torch.Tensor:
        freq_bands = 2.0 ** torch.linspace(0, self.num_freqs - 1, self.num_freqs, device=x.device)
        x_expanded = x.unsqueeze(-1) * freq_bands
        sin_x = torch.sin(x_expanded * torch.pi)
        cos_x = torch.cos(x_expanded * torch.pi)
        embeddings = torch.stack([sin_x, cos_x], dim=-1)
        embeddings = embeddings.flatten(start_dim=-3)
        return torch.cat([x, embeddings], dim=-1)

    def get_rays_from_params(self, intrinsics, extrinsics, H, W):
        """
        intrinsics: [B, N, 3, 3]
        extrinsics: [B, N, 4, 4]
        """
        B, N, _, _ = intrinsics.shape
        device = intrinsics.device
        
        y, x = torch.meshgrid(
            torch.linspace(0, H-1, H, device=device),
            torch.linspace(0, W-1, W, device=device),
            indexing='ij'
        )
        pixel_coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).unsqueeze(0)
        pixel_coords = pixel_coords.expand(B, N, -1, -1, -1)
        
        inv_K = torch.inverse(intrinsics).unsqueeze(2).unsqueeze(2)
        cam_coords = torch.matmul(inv_K, pixel_coords.unsqueeze(-1)).squeeze(-1)
        
        cam_dirs = cam_coords / cam_coords.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        
        R = extrinsics[..., :3, :3].unsqueeze(2).unsqueeze(2)
        world_dirs = torch.matmul(R, cam_dirs.unsqueeze(-1)).squeeze(-1)
        
        return world_dirs

    def forward(self, x, intrinsics=None, extrinsics=None):
        B, N, C, H, W = x.shape
        
        if intrinsics is None:
            return torch.zeros(B, N, H, W, self.dim, device=x.device)
            
        rays = self.get_rays_from_params(intrinsics, extrinsics, H, W)
        encoded_rays = self._sinusoidal_encoding(rays)
        encoded = self.proj(encoded_rays) # [B, N, H, W, dim]
        
        return encoded.permute(0, 1, 4, 2, 3) # [B, N, dim, H, W]

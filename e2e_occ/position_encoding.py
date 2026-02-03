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
        
        Updated to use equidistant projection model: theta = r / f
        to match d:/code/carla/occ_network/models/position_encoding.py
        """
        B, N, _, _ = intrinsics.shape
        device = intrinsics.device
        
        # 1. Pixel Coordinates
        y, x = torch.meshgrid(
            torch.linspace(0, H-1, H, device=device),
            torch.linspace(0, W-1, W, device=device),
            indexing='ij'
        )
        # Shift to center
        cx = W / 2.0
        cy = H / 2.0
        dx = x - cx
        dy = y - cy
        
        # 2. Radius in image plane
        r = torch.sqrt(dx**2 + dy**2)
        phi = torch.atan2(dy, dx)
        
        # 3. Equidistant Projection: theta = r / f
        # We need focal length f. Intrinsics[0,0] is fx.
        # Assuming fx approx fy approx f
        # [B, N, 1, 1]
        f = intrinsics[..., 0, 0].unsqueeze(-1).unsqueeze(-1) 
        
        theta = r.unsqueeze(0).unsqueeze(0) / f
        
        # 4. Spherical to Cartesian (Camera Frame)
        # Z is forward, X right, Y down
        # sin(theta) is the radial component
        sin_theta = torch.sin(theta)
        cam_z = torch.cos(theta)
        cam_x = sin_theta * torch.cos(phi.unsqueeze(0).unsqueeze(0))
        cam_y = sin_theta * torch.sin(phi.unsqueeze(0).unsqueeze(0))
        
        cam_dirs = torch.stack([cam_x, cam_y, cam_z], dim=-1) # [B, N, H, W, 3]
        
        # 5. Camera to World
        # cam_dirs is [B, N, H, W, 3]
        # R is [B, N, 3, 3]
        R = extrinsics[..., :3, :3]
        
        # Rotate: (R @ dir^T)^T = dir @ R^T
        # [B, N, H, W, 3] @ [B, N, 3, 3]^T
        world_dirs = torch.einsum('bnij,bnhwj->bnhwi', R, cam_dirs)
        
        return world_dirs

    def forward(self, x, intrinsics=None, extrinsics=None):
        B, N, C, H, W = x.shape
        
        if intrinsics is None:
            return torch.zeros(B, N, H, W, self.dim, device=x.device)
            
        rays = self.get_rays_from_params(intrinsics, extrinsics, H, W)
        encoded_rays = self._sinusoidal_encoding(rays)
        encoded = self.proj(encoded_rays) # [B, N, H, W, dim]
        
        return encoded.permute(0, 1, 4, 2, 3) # [B, N, dim, H, W]

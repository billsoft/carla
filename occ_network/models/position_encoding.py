import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Dict

class PositionEncoding2D(nn.Module):
    def __init__(self, dim: int, h: int, w: int, temperature: float = 10000):
        super().__init__()
        half_dim = dim // 2
        y_embed = torch.arange(h, dtype=torch.float32).unsqueeze(1).repeat(1, w) / h
        x_embed = torch.arange(w, dtype=torch.float32).unsqueeze(0).repeat(h, 1) / w
        dim_t = temperature ** (2 * (torch.arange(half_dim, dtype=torch.float32) // 2) / half_dim)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3).flatten(2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3).flatten(2)
        self.register_buffer('pos', torch.cat([pos_y, pos_x], dim=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, H, W = x.shape
            return x + self.pos[:H, :W].permute(2, 0, 1).unsqueeze(0)
        return x

class CameraRoPE(nn.Module):
    def __init__(self, dim: int, temperature: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer('inv_freq', 1.0 / (temperature ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, x: torch.Tensor, yaw_angles: torch.Tensor) -> torch.Tensor:
        theta = yaw_angles.unsqueeze(-1) * self.inv_freq
        theta = torch.cat([theta, theta], dim=-1)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
        return torch.stack([x1 * cos_t[..., ::2] - x2 * sin_t[..., ::2], x1 * sin_t[..., 1::2] + x2 * cos_t[..., 1::2]], dim=-1).reshape(*x.shape)

class HyperbolicFOVEncoding(nn.Module):
    def __init__(self, dim: int, fov_list: list, ref_fov: float = 70.0, temperature: float = 10000.0):
        super().__init__()
        phis = torch.tensor([math.asinh(math.sqrt(f / ref_fov) - 1) for f in fov_list], dtype=torch.float32)
        self.register_buffer('phis', phis)
        self.register_buffer('inv_freq', 1.0 / (temperature ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, x: torch.Tensor, camera_ids: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        phi = self.phis[camera_ids].unsqueeze(-1) * self.inv_freq
        cosh_p, sinh_p = torch.cosh(phi), torch.sinh(phi)
        x_pairs = x.view(B, N, -1, 2)
        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
        return torch.stack([x1 * cosh_p + x2 * sinh_p, x1 * sinh_p + x2 * cosh_p], dim=-1).view(B, N, d)

class MultiCameraPositionEncoding(nn.Module):
    def __init__(self, dim: int, num_cameras: int, image_size: Tuple[int, int], camera_configs: Dict, patch_size: int = 16):
        super().__init__()
        self.dim = dim
        yaw_angles, fov_list = [], []
        for cam_name in sorted(camera_configs.keys(), key=lambda x: camera_configs[x]['id']):
            cfg = camera_configs[cam_name]
            yaw_angles.append(cfg['rotation'][2] * math.pi / 180.0)
            fov_list.append(cfg['fov'])
        self.register_buffer('yaw_angles', torch.tensor(yaw_angles, dtype=torch.float32))
        feat_h, feat_w = image_size[0] // patch_size, image_size[1] // patch_size
        self.pixel_pe = PositionEncoding2D(dim, feat_h, feat_w)
        self.camera_rope = CameraRoPE(dim)
        self.fov_hyperbolic = HyperbolicFOVEncoding(dim, fov_list)

    def add_pixel_pe(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixel_pe(x)

    def encode_qk_single_camera(self, q: torch.Tensor, k: torch.Tensor, camera_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, d = q.shape
        cam_ids = torch.full((B, N), camera_id, dtype=torch.long, device=q.device)
        yaw = torch.full((B, N), self.yaw_angles[camera_id].item(), device=q.device)
        q = self.fov_hyperbolic(q, cam_ids)
        k = self.fov_hyperbolic(k, cam_ids)
        q = self.camera_rope(q, yaw)
        k = self.camera_rope(k, yaw)
        return q, k

class CameraPositionEncoding(nn.Module):
    def __init__(self, dim, num_cameras, image_size, camera_configs, patch_size=16):
        super().__init__()
        self.encoder = MultiCameraPositionEncoding(dim, num_cameras, image_size, camera_configs, patch_size)

    def forward(self, x):
        return self.encoder.add_pixel_pe(x)

    def encode_qk_single_camera(self, q, k, camera_id):
        return self.encoder.encode_qk_single_camera(q, k, camera_id)

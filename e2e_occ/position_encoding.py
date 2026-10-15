import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .geometry import equidistant_unproject, level_pixel_to_image, no_autocast
except ImportError:
    from geometry import equidistant_unproject, level_pixel_to_image, no_autocast


class SineCosinePositionEncoding3D(nn.Module):
    """3D 体素网格的正弦位置编码（输入为网格整数索引，加到查询内容向量上）。"""

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


def pixel_rays_vehicle(intrinsics, extrinsics, H, W, stride):
    """计算 stride 层级特征图每个像素的入射射线方向（车辆坐标系单位向量）。

    intrinsics: [B, N, 3, 3]，按原图分辨率标定的等距投影内参（fx, fy, cx, cy）
    extrinsics: [B, N, 4, 4]，Camera(OpenCV)→Vehicle
    返回 [B, N, H, W, 3]

    这是 deformable_attention.get_reference_points（正投影）的精确逆运算，二者都通过
    geometry.py 的同一套等距投影公式实现。
    """
    device = intrinsics.device
    with no_autocast(device):                                         # 几何一律 FP32，见 geometry.py
        intrinsics, extrinsics = intrinsics.float(), extrinsics.float()
        ys = level_pixel_to_image(torch.arange(H, device=device, dtype=torch.float32), stride)
        xs = level_pixel_to_image(torch.arange(W, device=device, dtype=torch.float32), stride)
        v, u = torch.meshgrid(ys, xs, indexing='ij')                # [H, W]，原图像素坐标
        fx = intrinsics[..., 0, 0, None, None]
        fy = intrinsics[..., 1, 1, None, None]
        cx = intrinsics[..., 0, 2, None, None]
        cy = intrinsics[..., 1, 2, None, None]
        dirs_cam = equidistant_unproject(u, v, fx, fy, cx, cy)      # [B, N, H, W, 3]
        R = extrinsics[..., :3, :3]
        return torch.einsum('bnij,bnhwj->bnhwi', R, dirs_cam)


class RayDirectionEncoding(nn.Module):
    """射线方向编码：特征像素 → 车辆系射线方向 → NeRF 式正弦编码 → MLP。

    多尺度时各层级共享同一个 MLP（方向语义与层级无关）。
    """

    def __init__(self, dim: int, num_freqs: int = 10):
        super().__init__()
        self.dim = dim
        self.num_freqs = num_freqs
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

    def forward(self, intrinsics, extrinsics, H, W, stride):
        """返回 [B, N, dim, H, W]"""
        rays = pixel_rays_vehicle(intrinsics, extrinsics, H, W, stride)
        with no_autocast(rays.device):                                # 最高频 2^9·π：FP32 算完再进 MLP
            enc = self._sinusoidal_encoding(rays.float())
        encoded = self.proj(enc)
        return encoded.permute(0, 1, 4, 2, 3)

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Dict, List

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

class RayDirectionEncoding(nn.Module):
    """
    射线方向编码

    将每个像素的 3D 射线方向编码为特征向量
    这帮助模型理解不同相机像素在 3D 空间中"指向"的方向
    """

    def __init__(
        self,
        dim: int,
        image_size: Tuple[int, int],
        camera_configs: Dict,
        patch_size: int = 16,
        temperature: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size

        # 预计算每个相机的射线方向
        for cam_name, cfg in camera_configs.items():
            cam_id = cfg['id']
            rays = self._compute_ray_directions(
                cfg['fov'],
                cfg['rotation'],
                image_size,
                patch_size
            )
            # rays: [H_patches, W_patches, 3]
            self.register_buffer(f'rays_{cam_id}', rays)

        # 正弦编码频率
        inv_freq = 1.0 / (temperature ** (torch.arange(0, dim, 6).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def _compute_ray_directions(
        self,
        fov: float,
        rotation: List[float],
        image_size: Tuple[int, int],
        patch_size: int
    ) -> torch.Tensor:
        """
        计算 patch 中心点的射线方向

        Returns:
            rays: [H_patches, W_patches, 3] 归一化射线方向
        """
        H, W = image_size
        H_p, W_p = H // patch_size, W // patch_size

        # 计算内参
        fx = W / (2 * math.tan(math.radians(fov / 2)))
        fy = fx  # 假设正方形像素
        cx, cy = W / 2, H / 2

        # Patch 中心坐标
        u = torch.linspace(patch_size/2, W - patch_size/2, W_p)
        v = torch.linspace(patch_size/2, H - patch_size/2, H_p)
        vv, uu = torch.meshgrid(v, u, indexing='ij')  # [H_p, W_p]

        # 相机坐标系下的射线方向
        dx = (uu - cx) / fx
        dy = (vv - cy) / fy
        dz = torch.ones_like(dx)

        rays_cam = torch.stack([dx, dy, dz], dim=-1)  # [H_p, W_p, 3]

        # 归一化
        rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True)

        # 转换到车辆坐标系
        R = self._rotation_matrix(rotation)  # [3, 3]
        rays_world = torch.einsum('ij,hwj->hwi', R, rays_cam)

        return rays_world

    def _rotation_matrix(self, rotation: List[float]) -> torch.Tensor:
        """
        从欧拉角计算旋转矩阵
        rotation: [pitch, roll, yaw] in degrees
        """
        pitch, roll, yaw = [math.radians(r) for r in rotation]

        # Rz @ Ry @ Rx
        Rx = torch.tensor([
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch), math.cos(pitch)]
        ], dtype=torch.float32)

        Ry = torch.tensor([
            [math.cos(roll), 0, math.sin(roll)],
            [0, 1, 0],
            [-math.sin(roll), 0, math.cos(roll)]
        ], dtype=torch.float32)

        Rz = torch.tensor([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        return Rz @ Ry @ Rx

    def _sinusoidal_encode(self, rays: torch.Tensor) -> torch.Tensor:
        """
        正弦编码射线方向

        rays: [..., 3] 射线方向向量
        returns: [..., dim] 编码后特征
        """
        shape = rays.shape[:-1]
        rays_flat = rays.view(-1, 3)  # [*, 3]

        encodings = []
        for i in range(3):  # x, y, z
            coord = rays_flat[:, i:i+1]  # [*, 1]
            freq = coord * self.inv_freq  # [*, dim//6]
            enc = torch.cat([freq.sin(), freq.cos()], dim=-1)  # [*, dim//3]
            encodings.append(enc)

        encoded = torch.cat(encodings, dim=-1)  # [*, dim]

        # 调整到目标维度
        if encoded.shape[-1] > self.dim:
            encoded = encoded[..., :self.dim]
        elif encoded.shape[-1] < self.dim:
            padding = torch.zeros(*encoded.shape[:-1], self.dim - encoded.shape[-1],
                                  device=encoded.device, dtype=encoded.dtype)
            encoded = torch.cat([encoded, padding], dim=-1)

        return encoded.view(*shape, self.dim)

    def forward(self, camera_id: int, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        获取指定相机的射线编码

        Args:
            camera_id: 相机ID
            batch_size: batch大小
            device: 目标设备

        Returns:
            ray_encoding: [B, H_p * W_p, dim]
        """
        rays = getattr(self, f'rays_{camera_id}')  # [H_p, W_p, 3]
        H_p, W_p, _ = rays.shape

        # 编码
        encoded = self._sinusoidal_encode(rays)  # [H_p, W_p, dim]

        # Flatten 并 batch expand
        encoded = encoded.view(H_p * W_p, self.dim)  # [N, dim]
        encoded = encoded.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, dim]

        if device is not None:
            encoded = encoded.to(device)

        return encoded


class CameraPositionEncoding(nn.Module):
    def __init__(self, dim, num_cameras, image_size, camera_configs, patch_size=16, use_ray_encoding=True):
        super().__init__()
        self.encoder = MultiCameraPositionEncoding(dim, num_cameras, image_size, camera_configs, patch_size)
        self.use_ray_encoding = use_ray_encoding
        if use_ray_encoding:
            self.ray_encoder = RayDirectionEncoding(dim, image_size, camera_configs, patch_size)

    def forward(self, x):
        return self.encoder.add_pixel_pe(x)

    def encode_qk_single_camera(self, q, k, camera_id):
        return self.encoder.encode_qk_single_camera(q, k, camera_id)

    def get_ray_encoding(self, camera_id: int, batch_size: int, device: torch.device = None) -> Optional[torch.Tensor]:
        """获取射线方向编码"""
        if self.use_ray_encoding:
            return self.ray_encoder(camera_id, batch_size, device)
        return None

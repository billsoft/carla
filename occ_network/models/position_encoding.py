import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Dict, List

# ============================================================================
# RayDirectionEncoding 增强版
# ============================================================================
# 核心改进:
# 1. 支持多种投影模型 (Pinhole, Equidistant, Stereographic)
# 2. 移除 HyperbolicFOVEncoding (冗余)
# 3. 统一输出归一化笛卡尔坐标 (x, y, z)
# ============================================================================


class RayDirectionEncoding(nn.Module):
    """
    射线方向编码 (支持多种投影模型)
    
    将每个像素映射为 3D 空间中的单位射线方向向量。
    支持:
    - Pinhole: 标准相机 (FOV < 100°)
    - Equidistant: 鱼眼相机 (FOV > 100°)
    - Stereographic: 超广角
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
            # 默认使用 pinhole，除非配置指定
            projection = cfg.get('projection', 'pinhole')
            
            rays = self._compute_ray_directions(
                cfg['fov'],
                cfg['rotation'],
                image_size,
                patch_size,
                projection
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
        patch_size: int,
        projection: str
    ) -> torch.Tensor:
        """
        计算 patch 中心点的射线方向 (支持多种投影)
        """
        H, W = image_size
        H_p, W_p = H // patch_size, W // patch_size
        
        # Patch 中心坐标 (图像坐标系)
        # u: 右 (x), v: 下 (y)
        u = torch.linspace(patch_size/2, W - patch_size/2, W_p)
        v = torch.linspace(patch_size/2, H - patch_size/2, H_p)
        vv, uu = torch.meshgrid(v, u, indexing='ij')  # [H_p, W_p]
        
        cx, cy = W / 2, H / 2
        
        # 1. 计算像素到光心的图像平面距离 r
        # dx, dy 是图像平面上的坐标 (像素单位)
        dx = uu - cx
        dy = vv - cy
        r = torch.sqrt(dx**2 + dy**2)
        
        # 图像平面上的方位角 phi (atan2(y, x))
        # 注意: 图像坐标系 y 是向下的
        phi_img = torch.atan2(dy, dx)
        
        # 2. 根据投影模型计算入射角 theta (射线与光轴的夹角)
        fov_rad = math.radians(fov)
        
        if projection == 'pinhole':
            # r = f * tan(theta)  =>  theta = atan(r / f)
            # f = W / (2 * tan(FOV/2))
            f = W / (2 * math.tan(fov_rad / 2))
            theta = torch.atan(r / f)
            
        elif projection == 'equidistant':
            # r = f * theta  =>  theta = r / f
            # 这里的 f 是比例系数，通常取 f = W / FOV_rad (假设 FOV 对应图像宽度)
            # 或者更严格地，如果是对角线FOV... 这里简化假设水平FOV对应宽度
            f = W / fov_rad
            theta = r / f
            
        elif projection == 'stereographic':
            # r = 2 * f * tan(theta / 2)
            # f = W / (4 * tan(FOV/4))
            f = W / (4 * math.tan(fov_rad / 4))
            theta = 2 * torch.atan(r / (2 * f))
            
        else:
            raise ValueError(f"Unknown projection type: {projection}")
            
        # 3. 转换为相机坐标系下的 3D 向量 (x, y, z)
        # 假设: Z轴向前 (光轴), X轴向右, Y轴向下
        # ray_z = cos(theta)
        # ray_x = sin(theta) * cos(phi_img)
        # ray_y = sin(theta) * sin(phi_img)
        
        ray_z = torch.cos(theta)
        sin_theta = torch.sin(theta)
        ray_x = sin_theta * torch.cos(phi_img)
        ray_y = sin_theta * torch.sin(phi_img)
        
        rays_cam = torch.stack([ray_x, ray_y, ray_z], dim=-1) # [H_p, W_p, 3]
        
        # 归一化 (理论上已经是单位向量，但为了数值稳定性)
        rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True)
        
        # 4. 转换到世界/车辆坐标系
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
        rays = getattr(self, f'rays_{camera_id}')  # [H_p, W_p, 3]
        H_p, W_p, _ = rays.shape

        encoded = self._sinusoidal_encode(rays)  # [H_p, W_p, dim]
        encoded = encoded.view(H_p * W_p, self.dim)  # [N, dim]
        encoded = encoded.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, dim]

        if device is not None:
            encoded = encoded.to(device)

        return encoded


class CameraPositionEncoding(nn.Module):
    """
    相机位置编码主类 (简化版)
    
    仅保留 RayDirectionEncoding，移除冗余的 HyperbolicFOVEncoding。
    """

    def __init__(self, dim, num_cameras, image_size, camera_configs, patch_size=16, use_ray_encoding=True):
        super().__init__()
        self.use_ray_encoding = use_ray_encoding
        if use_ray_encoding:
            self.ray_encoder = RayDirectionEncoding(dim, image_size, camera_configs, patch_size)
            
    def forward(self, x):
        # 不再添加任何 pixel-wise PE (如原来的 add_pixel_pe)
        # 位置信息完全由 cross-attention 中的 ray embedding 提供
        return x

    def encode_qk_single_camera(self, q, k, camera_id):
        # 移除 FOV 双曲编码
        # 这里直接返回原值，保留接口兼容性
        return q, k

    def get_ray_encoding(self, camera_id: int, batch_size: int, device: torch.device = None) -> Optional[torch.Tensor]:
        if self.use_ray_encoding:
            return self.ray_encoder(camera_id, batch_size, device)
        return None

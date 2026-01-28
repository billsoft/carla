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
    射线方向编码 (统一等距投影模型)

    将每个像素映射为 3D 空间中的单位射线方向向量。

    核心思想:
    - 所有相机统一使用等距投影 (equidistant): θ = r / f
    - 不同 FOV 的相机 = 在同一个球面上截取不同大小的区域
    - 小 FOV (如 35°) = 球面上的小区域 (长焦)
    - 大 FOV (如 120°) = 球面上的大区域 (广角)

    优势:
    - 概念统一，无需区分 pinhole/fisheye
    - 代码简洁，无需配置 projection 参数
    - 小角度时误差可由网络学习补偿
    """

    def __init__(
        self,
        dim: int,
        image_size: Tuple[int, int],
        camera_configs: Dict,
        patch_size: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size

        # 预计算每个相机的射线方向 (统一使用等距投影)
        for cam_name, cfg in camera_configs.items():
            cam_id = cfg['id']
            rays = self._compute_ray_directions(
                cfg['fov'],
                cfg['rotation'],
                image_size,
                patch_size
            )
            self.register_buffer(f'rays_{cam_id}', rays)

        # 线性映射层：将 3D 射线向量映射到特征维度 dim
        self.proj = nn.Linear(3, dim)

    def _compute_ray_directions(
        self,
        fov: float,
        rotation: List[float],
        image_size: Tuple[int, int],
        patch_size: int,
        projection: str = 'equidistant'  # 保留参数但不再使用
    ) -> torch.Tensor:
        """
        计算 patch 中心点的射线方向 (统一使用等距投影)

        设计决策:
        - 统一使用 equidistant (等距投影): θ = r / f
        - 所有相机都视为"球面上截取不同区域"
        - FOV 决定截取的球面范围大小
        - 小角度时 equidistant ≈ pinhole，误差可由网络学习补偿

        数学原理:
        - 等距投影: r = f * θ  =>  θ = r / f
        - f = (W/2) / (FOV_rad/2) = W / FOV_rad
        - 像素距离 r 线性映射到入射角 θ
        """
        H, W = image_size
        H_p, W_p = H // patch_size, W // patch_size

        # Patch 中心坐标 (图像坐标系)
        u = torch.linspace(patch_size/2, W - patch_size/2, W_p)
        v = torch.linspace(patch_size/2, H - patch_size/2, H_p)
        vv, uu = torch.meshgrid(v, u, indexing='ij')  # [H_p, W_p]

        cx, cy = W / 2, H / 2

        # 1. 计算像素到光心的图像平面距离 r
        dx = uu - cx
        dy = vv - cy
        r = torch.sqrt(dx**2 + dy**2)

        # 图像平面上的方位角 phi
        phi_img = torch.atan2(dy, dx)

        # 2. 统一使用等距投影计算入射角 theta
        # θ = r / f，其中 f = W / FOV_rad
        fov_rad = math.radians(fov)
        f = W / fov_rad
        theta = r / f

        # 3. 球面坐标 → 笛卡尔射线方向
        # 相机坐标系: Z轴向前(光轴), X轴向右, Y轴向下
        ray_z = torch.cos(theta)
        sin_theta = torch.sin(theta)
        ray_x = sin_theta * torch.cos(phi_img)
        ray_y = sin_theta * torch.sin(phi_img)

        rays_cam = torch.stack([ray_x, ray_y, ray_z], dim=-1)  # [H_p, W_p, 3]

        # 归一化
        rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # 4. 转换到世界/车辆坐标系
        R = self._rotation_matrix(rotation)
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

    def forward(self, camera_id: int, batch_size: int, device: torch.device = None) -> torch.Tensor:
        rays = getattr(self, f'rays_{camera_id}')  # [H_p, W_p, 3]
        H_p, W_p, _ = rays.shape

        # 直接将 3D 射线向量投影到特征维度
        encoded = self.proj(rays)  # [H_p, W_p, dim]
        
        encoded = encoded.view(H_p * W_p, self.dim)  # [N, dim]
        encoded = encoded.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, dim]

        if device is not None:
            encoded = encoded.to(device)

        return encoded


class CameraPositionEncoding(nn.Module):
    """
    相机位置编码主类 (简化版)

    位置编码架构:
    - RayDirectionEncoding: 射线方向编码 (支持 pinhole/equidistant/stereographic)

    已移除:
    - HyperbolicFOVEncoding: FOV 信息已在射线方向中编码
    - encode_qk_single_camera: 不再需要对 Q/K 做额外变换

    详见: occ_network/球面位置编码.md
    """

    def __init__(self, dim, num_cameras, image_size, camera_configs, patch_size=16):
        super().__init__()
        self.ray_encoder = RayDirectionEncoding(dim, image_size, camera_configs, patch_size)

    def get_ray_encoding(self, camera_id: int, batch_size: int, device: torch.device = None) -> Optional[torch.Tensor]:
        """
        获取指定相机的射线方向编码

        Args:
            camera_id: 相机 ID (0-7)
            batch_size: batch 大小
            device: 目标设备

        Returns:
            ray_encoding: [B, N, dim] 射线方向编码
        """
        return self.ray_encoder(camera_id, batch_size, device)

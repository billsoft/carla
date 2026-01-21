import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Dict, List

# ============================================================================
# CameraRoPE 已移除
# ============================================================================
# 原因: RayDirectionEncoding 已包含完整 6-DoF 旋转信息 (Pitch, Roll, Yaw)
# CameraRoPE 仅编码 Yaw，是 RayDirection 的子集，属于冗余编码
# 详见: occ_network/位置编码优化方案.md
# ============================================================================


class HyperbolicFOVEncoding(nn.Module):
    """
    双曲 FOV 编码

    对不同视场角的相机特征进行双曲缩放:
    - 广角相机 (FOV > 70°): 特征向量"膨胀"
    - 长焦相机 (FOV < 70°): 特征向量"收缩"

    这编码了内参差异，与 RayDirectionEncoding 的外参编码互补
    """

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
    """
    多相机位置编码 (简化版)

    仅使用 HyperbolicFOVEncoding 进行视场角差异编码
    3D 方向编码交给 RayDirectionEncoding 处理

    移除内容:
    - CameraRoPE: Yaw 编码已由 RayDirectionEncoding 包含
    - yaw_angles buffer: 不再需要
    """

    def __init__(self, dim: int, num_cameras: int, image_size: Tuple[int, int], camera_configs: Dict, patch_size: int = 16):
        super().__init__()
        self.dim = dim

        # 提取 FOV 列表 (仅保留 FOV 信息)
        fov_list = []
        for cam_name in sorted(camera_configs.keys(), key=lambda x: camera_configs[x]['id']):
            cfg = camera_configs[cam_name]
            fov_list.append(cfg['fov'])

        # 仅保留 FOV 双曲编码 (内参差异)
        self.fov_hyperbolic = HyperbolicFOVEncoding(dim, fov_list)

    def add_pixel_pe(self, x: torch.Tensor) -> torch.Tensor:
        """相对位置编码方案: 不修改输入特征"""
        return x

    def encode_qk_single_camera(self, q: torch.Tensor, k: torch.Tensor, camera_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对单个相机的 Q, K 应用 FOV 编码

        移除: CameraRoPE 的 Yaw 旋转 (由 RayDirectionEncoding 提供)
        """
        B, N, d = q.shape
        cam_ids = torch.full((B, N), camera_id, dtype=torch.long, device=q.device)

        # 仅应用 FOV 双曲编码
        q = self.fov_hyperbolic(q, cam_ids)
        k = self.fov_hyperbolic(k, cam_ids)

        return q, k


class RayDirectionEncoding(nn.Module):
    """
    射线方向编码 (6-DoF 几何先验)

    将每个像素的 3D 射线方向编码为特征向量
    这帮助模型理解不同相机像素在 3D 空间中"指向"的方向

    包含完整的几何信息:
    - 内参: FOV, 光心位置
    - 外参: Pitch, Roll, Yaw (通过旋转矩阵)

    注意: 这是唯一编码 Yaw 的模块 (CameraRoPE 已移除以避免冗余)
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

        # 转换到车辆坐标系 (包含完整的 Pitch, Roll, Yaw 信息)
        R = self._rotation_matrix(rotation)  # [3, 3]
        rays_world = torch.einsum('ij,hwj->hwi', R, rays_cam)

        return rays_world

    def _rotation_matrix(self, rotation: List[float]) -> torch.Tensor:
        """
        从欧拉角计算旋转矩阵
        rotation: [pitch, roll, yaw] in degrees

        这是 6-DoF 几何信息的核心:
        - Pitch: 俯仰 (减速带, 上下坡)
        - Roll: 翻滚 (弯道倾斜)
        - Yaw: 偏航 (相机朝向) <- 唯一的 Yaw 编码位置
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
    """
    相机位置编码主类

    组合多种编码:
    1. HyperbolicFOVEncoding: 视场角差异 (内参)
    2. RayDirectionEncoding: 3D 射线方向 (内参 + 6-DoF 外参)

    已移除:
    - CameraRoPE: Yaw 冗余编码
    """

    def __init__(self, dim, num_cameras, image_size, camera_configs, patch_size=16, use_ray_encoding=True):
        super().__init__()
        self.encoder = MultiCameraPositionEncoding(dim, num_cameras, image_size, camera_configs, patch_size)
        self.use_ray_encoding = use_ray_encoding
        if use_ray_encoding:
            self.ray_encoder = RayDirectionEncoding(dim, image_size, camera_configs, patch_size)

    def forward(self, x):
        return self.encoder.add_pixel_pe(x)

    def encode_qk_single_camera(self, q, k, camera_id):
        """对 Q, K 应用 FOV 编码 (无 RoPE)"""
        return self.encoder.encode_qk_single_camera(q, k, camera_id)

    def get_ray_encoding(self, camera_id: int, batch_size: int, device: torch.device = None) -> Optional[torch.Tensor]:
        """获取射线方向编码 (6-DoF 几何先验)"""
        if self.use_ray_encoding:
            return self.ray_encoder(camera_id, batch_size, device)
        return None


# 兼容性别名 (保持向后兼容)
# 注意: 旧代码可能直接使用 MultiCameraPositionEncoding
# CameraPositionEncoding = MultiCameraPositionEncoding  # 已注释，使用完整版 CameraPositionEncoding

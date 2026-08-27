import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

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

def rescale_focal_to_feature_map(intrinsics, H, W):
    """
    intrinsics: [B, N, 3, 3]，等距投影相机的内参（各向同性，fx=fy=f），按"原始图像分辨率"标定
                (对应 occnetv3_data_generator camera_manager.py::get_intrinsics 的等距分支：
                f = (height/2) / (vertical_fov_rad/2)，与 CARLA CameraModelUtil::ComputeDistance
                的 Equidistant 分支一致)
    H, W: 本次调用对应的（可能降采样的）特征图高宽

    等距/针孔都需要这一步：intrinsics 是按原图分辨率标定的，但射线编码 / 可变形注意力的
    参考点投影都在下采样后的特征图分辨率上做逐像素运算，直接用原图焦距会让像素偏移量
    (特征图尺度，量级几十) 除以原图焦距 (原图尺度，量级几百上千)，把入射角压缩到几乎全部
    指向正前方。这里用 intrinsics 自带的主点 (cx_orig, cy_orig) 和实际的 (H, W) 推出降采样
    比例，把焦距换算到特征图像素单位。

    返回 f: [B, N]（无额外的尾随单位维度，调用方按自己的广播需求 unsqueeze）。
    """
    cx_orig = intrinsics[..., 0, 2]  # [B, N]
    cy_orig = intrinsics[..., 1, 2]  # [B, N]
    scale_x = (W / 2.0) / cx_orig.clamp_min(1e-6)
    scale_y = (H / 2.0) / cy_orig.clamp_min(1e-6)
    # 等距投影各向同性，scale_x 理论上等于 scale_y；取平均以稳健应对 H/W 取整带来的误差
    scale = (scale_x + scale_y) * 0.5
    f = intrinsics[..., 0, 0] * scale  # [B, N]
    return f


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
        intrinsics: [B, N, 3, 3]，焦距/主点是相对"原始图像分辨率"标定的（等距投影，fx=fy=f）
        extrinsics: [B, N, 4, 4]
        H, W: 本次调用对应的特征图高宽（可能是原始图像的降采样版本）

        等距投影（equidistant）反投影模型：theta = r/f，phi = atan2(dy, dx)，
        与 CARLA sensor.camera.rgb_fisheye（camera_model=equidistant）采集时用的投影模型
        完全一致（Unreal/.../Util/CameraModelUtil.cpp 的 ComputeDistance/ComputeAngle
        Equidistant 分支：F=(Height/2)/(FOV/2), theta=Distance），焦距换算见
        rescale_focal_to_feature_map。
        """
        device = intrinsics.device

        # 1. 特征图像素坐标（以主点为原点）
        y, x = torch.meshgrid(
            torch.linspace(0, H-1, H, device=device),
            torch.linspace(0, W-1, W, device=device),
            indexing='ij'
        )
        cx = W / 2.0
        cy = H / 2.0
        dx = x - cx
        dy = y - cy

        # 2. 焦距换算到特征图像素单位（原图→特征图，见函数说明）
        f = rescale_focal_to_feature_map(intrinsics, H, W).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]

        # 3. 等距投影反投影：theta = r/f
        r = torch.sqrt(dx ** 2 + dy ** 2).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        phi = torch.atan2(dy, dx).unsqueeze(0).unsqueeze(0)

        theta = r / f  # [B, N, H, W]

        sin_theta = torch.sin(theta)
        cam_z = torch.cos(theta)
        cam_x = sin_theta * torch.cos(phi)
        cam_y = sin_theta * torch.sin(phi)

        cam_dirs = torch.stack([cam_x, cam_y, cam_z], dim=-1)  # [B, N, H, W, 3]

        # 4. Camera to World
        R = extrinsics[..., :3, :3]
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

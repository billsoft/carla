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

# 原始标定分辨率：目前项目里唯一实际使用的 image_size（E2EOccConfig.image_size 默认值，
# dataset.py::_get_default_camera_params 也硬编码同一个值）。rescale_focal_to_feature_map
# 用它把 intrinsics 里按原图标定的 fx/cx/cy 换算到调用方实际传入的目标 H,W（特征图分辨率，
# 可能小于原图，也可能是 verify_network.py 里测试用的任意合成分辨率）。如果以后
# E2EOccConfig.image_size 真的改了默认值，这里要同步改。
_CALIBRATED_IMAGE_SIZE = (960, 1280)  # (H_orig, W_orig)


def rescale_focal_to_feature_map(intrinsics, H, W):
    """
    intrinsics: [B, N, 3, 3]，等距投影相机的内参（各向同性，fx=fy=f；cx,cy 为主点像素坐标，
                不假设居中），按 _CALIBRATED_IMAGE_SIZE 标定
                (对应 occnetv3_data_generator camera_manager.py::get_intrinsics 的等距分支：
                f = (height/2) / (vertical_fov_rad/2)，与 CARLA CameraModelUtil::ComputeDistance
                的 Equidistant 分支一致)
    H, W: 本次调用对应的（可能降采样的）特征图高宽

    等距/针孔都需要这一步：intrinsics 是按原图分辨率标定的，但射线编码 / 可变形注意力的
    参考点投影都在下采样后的特征图分辨率上做逐像素运算，直接用原图焦距/主点会让像素坐标
    量纲对不上。之前的写法从 cx_orig 反推降采样比例（隐含假设 cx_orig 恰好等于原图宽度
    一半），主点非居中的真实标定数据会让这个假设失效、连降采样比例都跟着算错；现在直接用
    已知的原始标定分辨率算比例，cx/cy 按各自轴的比例换算后原样返回（不再假设居中）。

    返回 (f, cx, cy)，均已换算到目标 H,W 的像素单位，形状 [B, N]（无额外的尾随单位维度，
    调用方按自己的广播需求 unsqueeze）。
    """
    H_orig, W_orig = _CALIBRATED_IMAGE_SIZE
    scale_x = W / W_orig
    scale_y = H / H_orig
    # 等距投影各向同性，scale_x 理论上等于 scale_y；取平均以稳健应对 H/W 取整带来的误差
    scale = (scale_x + scale_y) * 0.5
    f = intrinsics[..., 0, 0] * scale        # [B, N]
    cx = intrinsics[..., 0, 2] * scale_x     # [B, N]
    cy = intrinsics[..., 1, 2] * scale_y     # [B, N]
    return f, cx, cy


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

        # 1. 特征图像素坐标（尚未减去主点，主点随相机/batch变化，下面按真实 cx/cy 广播减）
        y, x = torch.meshgrid(
            torch.linspace(0, H-1, H, device=device),
            torch.linspace(0, W-1, W, device=device),
            indexing='ij'
        )
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        y = y.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # 2. 焦距/主点换算到特征图像素单位（原图→特征图，见函数说明）
        f, cx, cy = rescale_focal_to_feature_map(intrinsics, H, W)  # 各 [B, N]
        f = f.unsqueeze(-1).unsqueeze(-1)    # [B, N, 1, 1]
        cx = cx.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
        cy = cy.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]

        dx = x - cx  # [B, N, H, W]
        dy = y - cy  # [B, N, H, W]

        # 3. 等距投影反投影：theta = r/f
        r = torch.sqrt(dx ** 2 + dy ** 2)
        phi = torch.atan2(dy, dx)

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

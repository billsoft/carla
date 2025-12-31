# models/transformer_occ/position_encoding.py
"""
位置编码模块

1. CameraPositionEncoding: 相机位置编码（射线方向 + 相机位置）
2. SpatialPositionEncoding: 2D 空间位置编码
3. VoxelPositionEncoding: 3D 体素位置编码
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class SinusoidalPositionEncoding(nn.Module):
    """
    正弦位置编码 (经典 Transformer 位置编码)
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    
    def __init__(self, embed_dim: int, max_len: int = 10000):
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] 输入序列
        Returns:
            pe: [1, N, D] 位置编码
        """
        return self.pe[:x.size(1)].unsqueeze(0)


class LearnablePositionEncoding(nn.Module):
    """可学习的位置编码"""
    
    def __init__(self, num_positions: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_embed[:, :x.size(1)]


class Spatial2DPositionEncoding(nn.Module):
    """
    2D 空间位置编码
    
    为图像 patches 生成 (x, y) 位置编码
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int],
        embed_dim: int,
        learnable: bool = True
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        H, W = grid_size
        
        if learnable:
            # 可学习的位置编码
            self.row_embed = nn.Parameter(torch.zeros(H, embed_dim // 2))
            self.col_embed = nn.Parameter(torch.zeros(W, embed_dim // 2))
            nn.init.trunc_normal_(self.row_embed, std=0.02)
            nn.init.trunc_normal_(self.col_embed, std=0.02)
        else:
            # 正弦位置编码
            row_embed = self._make_sinusoidal(H, embed_dim // 2)
            col_embed = self._make_sinusoidal(W, embed_dim // 2)
            self.register_buffer('row_embed', row_embed)
            self.register_buffer('col_embed', col_embed)
            
    def _make_sinusoidal(self, length: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:dim//2])
        return pe
    
    def forward(self) -> torch.Tensor:
        """
        Returns:
            pos_embed: [H*W, D] 2D 位置编码
        """
        H, W = self.grid_size
        
        # 扩展为网格
        row_embed = self.row_embed.unsqueeze(1).expand(H, W, -1)  # [H, W, D/2]
        col_embed = self.col_embed.unsqueeze(0).expand(H, W, -1)  # [H, W, D/2]
        
        # 拼接
        pos_embed = torch.cat([row_embed, col_embed], dim=-1)  # [H, W, D]
        pos_embed = pos_embed.flatten(0, 1)  # [H*W, D]
        
        return pos_embed


class CameraPositionEncoding(nn.Module):
    """
    相机位置编码
    
    编码信息:
    - 像素坐标 (u, v)
    - 射线方向 (dx, dy, dz)
    - 相机位置 (cx, cy, cz)
    - 相机 ID
    
    这是"位置编码 = 相机参数"的核心实现
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_cameras: int = 8,
        grid_size: Tuple[int, int] = (60, 80),
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.grid_size = grid_size
        
        # 输入维度: u, v, ray_dir(3), cam_pos(3) = 8
        # 加上相机 ID embedding
        input_dim = 8
        
        # MLP 编码器
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # 相机 ID embedding (备用)
        self.camera_embed = nn.Embedding(num_cameras, embed_dim)
        
        # 默认相机参数（如果没有提供）
        self._init_default_cameras()
        
    def _init_default_cameras(self):
        """初始化默认的 8 相机配置（类似特斯拉布局）"""
        # 相机位置 (相对于车辆中心)
        default_positions = torch.tensor([
            [2.0, 0.0, 1.5],    # front_main
            [2.0, 0.0, 1.5],    # front_wide
            [2.0, 0.0, 1.5],    # front_narrow
            [1.0, -1.0, 1.2],   # left_pillar
            [1.0, 1.0, 1.2],    # right_pillar
            [-0.5, -1.0, 1.0],  # left_repeater
            [-0.5, 1.0, 1.0],   # right_repeater
            [-2.0, 0.0, 1.5],   # rear
        ], dtype=torch.float32)
        
        # 相机朝向 (yaw 角度，0=前方)
        default_yaws = torch.tensor([
            0.0,    # front_main
            0.0,    # front_wide
            0.0,    # front_narrow
            -60.0,  # left_pillar
            60.0,   # right_pillar
            -120.0, # left_repeater
            120.0,  # right_repeater
            180.0,  # rear
        ], dtype=torch.float32) * (math.pi / 180.0)  # 转弧度
        
        self.register_buffer('default_positions', default_positions)
        self.register_buffer('default_yaws', default_yaws)
        
    def _compute_ray_directions(
        self,
        grid_size: Tuple[int, int],
        camera_idx: int,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算每个 patch 的射线方向
        
        Returns:
            uv: [H*W, 2] 归一化像素坐标
            ray_dirs: [H*W, 3] 射线方向（单位向量）
            cam_pos: [3] 相机位置
        """
        H, W = grid_size
        
        # 生成归一化像素坐标 [-1, 1]
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        uv = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [H*W, 2]
        
        if intrinsics is not None and extrinsics is not None:
            # 使用真实相机参数
            
            # 1. 像素坐标反投影到相机坐标系 (z=1)
            # intrinsics: [B, N_cam, 3, 3] -> 这里我们只取第一个样本的第 camera_idx 个相机的参数
            # 假设所有样本的相机参数相同，或者我们通过 batch_size 扩展处理
            # 这里的输入 intrinsics 可能是 [3, 3] (单个相机)
            
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            
            # ray_cam = K⁻¹ * [u, v, 1]
            # u, v 已经是归一化到 [-1, 1] 的，需要转回像素坐标
            # u_pix = (u + 1) * W / 2
            # v_pix = (v + 1) * H / 2
            # ray_x = (u_pix - cx) / fx
            # ray_y = (v_pix - cy) / fy
            
            # 简化计算：
            # ray_x = ((uv[:, 0] + 1) * W / 2 - cx) / fx
            # ray_y = ((uv[:, 1] + 1) * H / 2 - cy) / fy
            
            ray_x = ((uv[:, 0] + 1) * 0.5 * W - cx) / fx
            ray_y = ((uv[:, 1] + 1) * 0.5 * H - cy) / fy
            ray_z = torch.ones_like(ray_x)
            
            ray_cam = torch.stack([ray_x, ray_y, ray_z], dim=-1) # [N, 3]
            
            # 2. 旋转到世界坐标系
            # ray_world = R * ray_cam
            R = extrinsics[:3, :3] # [3, 3]
            t = extrinsics[:3, 3]  # [3]
            
            # 注意: 这里的 R 通常是 World -> Camera 的旋转矩阵
            # 所以 Camera -> World 需要 R^T (或 R⁻¹)
            # ray_world = R^T * ray_cam
            # 向量乘法: ray_cam @ R
            
            ray_dirs = ray_cam @ R
            ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
            
            # 3. 相机位置
            # cam_pos = -R^T * t
            cam_pos = -R.T @ t
        else:
            # 使用默认相机参数
            yaw = self.default_yaws[camera_idx]
            
            # 简化的射线方向计算
            # 假设相机 FOV 约 70 度
            fov_factor = 0.7  # tan(35°) ≈ 0.7
            
            # 局部坐标系的射线
            ray_x_local = uv[:, 0] * fov_factor
            ray_y_local = uv[:, 1] * fov_factor * 0.75  # 假设 4:3 传感器
            ray_z_local = torch.ones_like(ray_x_local)
            
            # 旋转到世界坐标系
            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)
            
            ray_x = ray_z_local * cos_yaw - ray_x_local * sin_yaw
            ray_y = ray_z_local * sin_yaw + ray_x_local * cos_yaw
            ray_z = ray_y_local  # 高度方向
            
            ray_dirs = torch.stack([ray_x, ray_y, ray_z], dim=-1)
            ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
            
            cam_pos = self.default_positions[camera_idx]
            
        return uv, ray_dirs, cam_pos
    
    def forward(
        self,
        camera_idx: int,
        batch_size: int,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        生成相机位置编码
        
        Args:
            camera_idx: 相机索引
            batch_size: batch 大小
            intrinsics: [3, 3] 相机内参（可选）
            extrinsics: [4, 4] 相机外参（可选）
            device: 设备
            
        Returns:
            camera_pe: [B, N_patches, D] 相机位置编码
        """
        if device is None:
            device = next(self.parameters()).device
            
        # 计算射线方向
        uv, ray_dirs, cam_pos = self._compute_ray_directions(
            self.grid_size, camera_idx, intrinsics, extrinsics, device
        )
        
        N = uv.shape[0]  # H * W
        
        # 扩展相机位置
        cam_pos_expanded = cam_pos.unsqueeze(0).expand(N, -1)  # [N, 3]
        
        # 拼接所有信息
        features = torch.cat([
            uv,              # [N, 2] 像素坐标
            ray_dirs,        # [N, 3] 射线方向
            cam_pos_expanded # [N, 3] 相机位置
        ], dim=-1)  # [N, 8]
        
        # MLP 编码
        camera_pe = self.mlp(features)  # [N, D]
        
        # 扩展 batch 维度
        camera_pe = camera_pe.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, D]
        
        return camera_pe


class Voxel3DPositionEncoding(nn.Module):
    """
    3D 体素位置编码
    
    为每个体素生成 (x, y, z) 位置编码
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (50, 50, 8),
        embed_dim: int = 256,
        learnable: bool = True,
        x_range: Tuple[float, float] = (-25.0, 25.0),
        y_range: Tuple[float, float] = (-25.0, 25.0),
        z_range: Tuple[float, float] = (-2.0, 6.0),
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        X, Y, Z = grid_size
        num_voxels = X * Y * Z
        
        if learnable:
            # 可学习的 3D 位置编码
            self.x_embed = nn.Parameter(torch.zeros(X, embed_dim // 3))
            self.y_embed = nn.Parameter(torch.zeros(Y, embed_dim // 3))
            self.z_embed = nn.Parameter(torch.zeros(Z, embed_dim - 2 * (embed_dim // 3)))
            nn.init.trunc_normal_(self.x_embed, std=0.02)
            nn.init.trunc_normal_(self.y_embed, std=0.02)
            nn.init.trunc_normal_(self.z_embed, std=0.02)
            self.learnable = True
        else:
            # 正弦位置编码
            self.learnable = False
            # 预计算 3D 坐标
            x = torch.linspace(x_range[0], x_range[1], X)
            y = torch.linspace(y_range[0], y_range[1], Y)
            z = torch.linspace(z_range[0], z_range[1], Z)
            
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # [X*Y*Z, 3]
            
            # 正弦编码
            pos_embed = self._make_3d_sinusoidal(coords, embed_dim)
            self.register_buffer('pos_embed', pos_embed)
            
    def _make_3d_sinusoidal(self, coords: torch.Tensor, dim: int) -> torch.Tensor:
        """生成 3D 正弦位置编码"""
        N = coords.shape[0]
        pe = torch.zeros(N, dim)
        
        # 每个坐标轴分配 dim/3 的维度
        dim_per_axis = dim // 3
        
        for axis in range(3):
            position = coords[:, axis].unsqueeze(1)  # [N, 1]
            div_term = torch.exp(
                torch.arange(0, dim_per_axis, 2).float() * (-math.log(10000.0) / dim_per_axis)
            )
            
            start_idx = axis * dim_per_axis
            pe[:, start_idx:start_idx+dim_per_axis:2] = torch.sin(position * div_term)
            if dim_per_axis > 1:
                pe[:, start_idx+1:start_idx+dim_per_axis:2] = torch.cos(position * div_term[:dim_per_axis//2])
                
        return pe
    
    def forward(self) -> torch.Tensor:
        """
        Returns:
            pos_embed: [X*Y*Z, D] 3D 位置编码
        """
        if self.learnable:
            X, Y, Z = self.grid_size
            
            # 扩展为 3D 网格
            x_embed = self.x_embed.view(X, 1, 1, -1).expand(X, Y, Z, -1)
            y_embed = self.y_embed.view(1, Y, 1, -1).expand(X, Y, Z, -1)
            z_embed = self.z_embed.view(1, 1, Z, -1).expand(X, Y, Z, -1)
            
            # 拼接
            pos_embed = torch.cat([x_embed, y_embed, z_embed], dim=-1)
            pos_embed = pos_embed.flatten(0, 2)  # [X*Y*Z, D]
            
            return pos_embed
        else:
            return self.pos_embed


if __name__ == '__main__':
    print("=" * 60)
    print("位置编码模块测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试 2D 空间位置编码
    print("\n[1] 2D 空间位置编码:")
    spatial_pe = Spatial2DPositionEncoding(grid_size=(60, 80), embed_dim=256)
    pe_2d = spatial_pe()
    print(f"  Grid Size: (60, 80)")
    print(f"  输出: {pe_2d.shape}")
    
    # 测试相机位置编码
    print("\n[2] 相机位置编码:")
    camera_pe = CameraPositionEncoding(
        embed_dim=256,
        num_cameras=8,
        grid_size=(60, 80)
    ).to(device)
    
    pe_cam = camera_pe(camera_idx=0, batch_size=2, device=device)
    print(f"  Camera: 0")
    print(f"  输出: {pe_cam.shape}")
    
    # 测试 3D 体素位置编码
    print("\n[3] 3D 体素位置编码:")
    voxel_pe = Voxel3DPositionEncoding(
        grid_size=(50, 50, 8),
        embed_dim=256,
        learnable=True
    )
    
    pe_3d = voxel_pe()
    print(f"  Grid Size: (50, 50, 8)")
    print(f"  输出: {pe_3d.shape}")
    print(f"  Num Voxels: {50*50*8}")
    
    # 参数量
    total_params = sum(p.numel() for p in camera_pe.parameters())
    print(f"\n相机 PE 参数量: {total_params/1e3:.1f}K")
    
    total_params = sum(p.numel() for p in voxel_pe.parameters())
    print(f"体素 PE 参数量: {total_params/1e3:.1f}K")
    
    print("\n" + "=" * 60)
    print("✅ 测试通过！")
    print("=" * 60)

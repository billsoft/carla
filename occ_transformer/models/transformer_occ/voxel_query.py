# models/transformer_occ/voxel_query.py
"""
3D Voxel Query 模块

可学习的体素查询向量，用于从图像特征中提取 3D 信息
类比 DETR 的 object queries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class Voxel3DPositionEncodingLocal(nn.Module):
    """3D 体素位置编码（本地版本，避免循环导入）"""
    
    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (50, 50, 8),
        embed_dim: int = 256,
        learnable: bool = True,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        X, Y, Z = grid_size
        
        if learnable:
            self.x_embed = nn.Parameter(torch.zeros(X, embed_dim // 3))
            self.y_embed = nn.Parameter(torch.zeros(Y, embed_dim // 3))
            self.z_embed = nn.Parameter(torch.zeros(Z, embed_dim - 2 * (embed_dim // 3)))
            nn.init.trunc_normal_(self.x_embed, std=0.02)
            nn.init.trunc_normal_(self.y_embed, std=0.02)
            nn.init.trunc_normal_(self.z_embed, std=0.02)
            self.learnable = True
        else:
            self.learnable = False
            x = torch.linspace(-1, 1, X)
            y = torch.linspace(-1, 1, Y)
            z = torch.linspace(-1, 1, Z)
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
            pos_embed = self._make_3d_sinusoidal(coords, embed_dim)
            self.register_buffer('pos_embed', pos_embed)
            
    def _make_3d_sinusoidal(self, coords: torch.Tensor, dim: int) -> torch.Tensor:
        N = coords.shape[0]
        pe = torch.zeros(N, dim)
        dim_per_axis = dim // 3
        
        for axis in range(3):
            position = coords[:, axis].unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, dim_per_axis, 2).float() * (-math.log(10000.0) / dim_per_axis)
            )
            start_idx = axis * dim_per_axis
            pe[:, start_idx:start_idx+dim_per_axis:2] = torch.sin(position * div_term)
            if dim_per_axis > 1:
                pe[:, start_idx+1:start_idx+dim_per_axis:2] = torch.cos(position * div_term[:dim_per_axis//2])
        return pe
    
    def forward(self) -> torch.Tensor:
        if self.learnable:
            X, Y, Z = self.grid_size
            x_embed = self.x_embed.view(X, 1, 1, -1).expand(X, Y, Z, -1)
            y_embed = self.y_embed.view(1, Y, 1, -1).expand(X, Y, Z, -1)
            z_embed = self.z_embed.view(1, 1, Z, -1).expand(X, Y, Z, -1)
            pos_embed = torch.cat([x_embed, y_embed, z_embed], dim=-1)
            return pos_embed.flatten(0, 2)
        else:
            return self.pos_embed


class VoxelQueries(nn.Module):
    """
    可学习的 3D 体素查询
    
    每个体素位置有一个可学习的查询向量
    查询向量 + 3D 位置编码 → 最终的查询
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (50, 50, 8),
        embed_dim: int = 256,
        x_range: Tuple[float, float] = (-25.0, 25.0),
        y_range: Tuple[float, float] = (-25.0, 25.0),
        z_range: Tuple[float, float] = (-2.0, 6.0),
        learnable_pe: bool = True,
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.num_voxels = grid_size[0] * grid_size[1] * grid_size[2]
        
        # 可学习的查询向量
        self.query_embed = nn.Parameter(torch.zeros(self.num_voxels, embed_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        
        # 3D 位置编码
        self.pos_embed = Voxel3DPositionEncodingLocal(
            grid_size=grid_size,
            embed_dim=embed_dim,
            learnable=learnable_pe,
        )
        
        # 存储网格坐标（用于参考点）
        self._init_reference_points(grid_size)
        
    def _init_reference_points(self, grid_size: Tuple[int, int, int]):
        X, Y, Z = grid_size
        x = torch.linspace(0, 1, X)
        y = torch.linspace(0, 1, Y)
        z = torch.linspace(0, 1, Z)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        reference_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
        self.register_buffer('reference_points_3d', reference_points)
        reference_points_2d = reference_points[:, :2]
        self.register_buffer('reference_points_2d', reference_points_2d)
        
    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_embed = self.pos_embed()
        queries = self.query_embed + pos_embed
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        query_embed = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = self.reference_points_2d.unsqueeze(0).expand(batch_size, -1, -1)
        return queries, query_embed, reference_points
    
    def get_3d_reference_points(self, batch_size: int) -> torch.Tensor:
        return self.reference_points_3d.unsqueeze(0).expand(batch_size, -1, -1)


class HierarchicalVoxelQueries(nn.Module):
    """
    分层体素查询 - 先低分辨率预测，再上采样
    """
    
    def __init__(
        self,
        query_grid_size: Tuple[int, int, int] = (50, 50, 8),
        output_grid_size: Tuple[int, int, int] = (200, 200, 16),
        embed_dim: int = 256,
        x_range: Tuple[float, float] = (-25.0, 25.0),
        y_range: Tuple[float, float] = (-25.0, 25.0),
        z_range: Tuple[float, float] = (-2.0, 6.0),
    ):
        super().__init__()
        
        self.query_grid_size = query_grid_size
        self.output_grid_size = output_grid_size
        
        self.voxel_queries = VoxelQueries(
            grid_size=query_grid_size,
            embed_dim=embed_dim,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm3d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(embed_dim // 2, embed_dim, kernel_size=(2, 2, 1), stride=(2, 2, 1)),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.voxel_queries(batch_size)
    
    def upsample_features(self, features: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        X, Y, Z = self.query_grid_size
        features = features.view(B, X, Y, Z, -1).permute(0, 4, 1, 2, 3)
        return self.upsample(features)


class BEVQueries(nn.Module):
    """
    BEV (Bird's Eye View) 查询
    
    只在 XY 平面查询，然后扩展到 Z 维度
    比完整的 3D 查询更高效
    """
    
    def __init__(
        self,
        bev_size: Tuple[int, int] = (100, 100),
        num_height_levels: int = 8,
        embed_dim: int = 256,
        x_range: Tuple[float, float] = (-25.0, 25.0),
        y_range: Tuple[float, float] = (-25.0, 25.0),
    ):
        super().__init__()
        
        self.bev_size = bev_size
        self.num_height_levels = num_height_levels
        self.embed_dim = embed_dim
        self.num_queries = bev_size[0] * bev_size[1]
        
        # BEV 查询嵌入
        self.query_embed = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        
        # BEV 位置编码
        H, W = bev_size
        x = torch.linspace(0, 1, H)
        y = torch.linspace(0, 1, W)
        
        # 几何约束优化：中心加密分布
        x = torch.sigmoid((x - 0.5) * 4)
        
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        reference_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        self.register_buffer('reference_points', reference_points)
        
        # 2D 位置编码 MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        
        # 高度扩展 MLP
        self.height_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim * num_height_levels),
        )
        
    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_embed = self.pos_mlp(self.reference_points)
        queries = self.query_embed + pos_embed
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        query_embed = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = self.reference_points.unsqueeze(0).expand(batch_size, -1, -1)
        return queries, query_embed, reference_points
    
    def expand_to_3d(self, bev_features: torch.Tensor) -> torch.Tensor:
        B, N, D = bev_features.shape
        H, W = self.bev_size
        Z = self.num_height_levels
        features_3d = self.height_mlp(bev_features)
        features_3d = features_3d.view(B, H, W, D, Z)
        features_3d = features_3d.permute(0, 3, 1, 2, 4)
        return features_3d


if __name__ == '__main__':
    print("=" * 60)
    print("Voxel Query 模块测试")
    print("=" * 60)
    
    # 测试 VoxelQueries
    print("\n[1] VoxelQueries:")
    vq = VoxelQueries(grid_size=(50, 50, 8), embed_dim=256)
    queries, query_embed, ref_points = vq(batch_size=2)
    print(f"  Queries: {queries.shape}")
    print(f"  Reference Points: {ref_points.shape}")
    print(f"  Num Voxels: {vq.num_voxels}")
    
    # 测试 BEVQueries
    print("\n[2] BEVQueries:")
    bq = BEVQueries(bev_size=(100, 100), num_height_levels=8, embed_dim=256)
    queries, _, ref_points = bq(batch_size=2)
    print(f"  Queries: {queries.shape}")
    print(f"  Reference Points: {ref_points.shape}")
    
    print("\n✅ 测试通过！")

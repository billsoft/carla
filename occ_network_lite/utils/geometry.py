# utils/geometry.py
"""
几何工具函数

包含:
- 坐标系变换
- 相机投影
- 体素网格操作
"""

import torch
import numpy as np
from typing import Tuple, Optional


def create_meshgrid(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    resolution: float,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    创建 3D 网格坐标
    
    Args:
        x_range: X 轴范围 (min, max)
        y_range: Y 轴范围 (min, max)
        z_range: Z 轴范围 (min, max)
        resolution: 网格分辨率
        device: 目标设备
        
    Returns:
        grid: [X, Y, Z, 3] 网格坐标，每个位置是 (x, y, z)
    """
    # 计算网格尺寸
    nx = int((x_range[1] - x_range[0]) / resolution)
    ny = int((y_range[1] - y_range[0]) / resolution)
    nz = int((z_range[1] - z_range[0]) / resolution)
    
    # 创建坐标轴（体素中心）
    x = torch.linspace(
        x_range[0] + resolution / 2,
        x_range[1] - resolution / 2,
        nx,
        device=device,
    )
    y = torch.linspace(
        y_range[0] + resolution / 2,
        y_range[1] - resolution / 2,
        ny,
        device=device,
    )
    z = torch.linspace(
        z_range[0] + resolution / 2,
        z_range[1] - resolution / 2,
        nz,
        device=device,
    )
    
    # 创建 3D 网格
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.stack([xx, yy, zz], dim=-1)  # [X, Y, Z, 3]
    
    return grid


def get_reference_points(
    bev_h: int,
    bev_w: int,
    x_range: Tuple[float, float] = (-50.0, 50.0),
    y_range: Tuple[float, float] = (-50.0, 50.0),
    num_heights: int = 4,
    z_range: Tuple[float, float] = (-4.0, 4.0),
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    获取 BEV 参考点（用于 Deformable Attention）
    
    Args:
        bev_h: BEV 高度
        bev_w: BEV 宽度
        x_range: X 范围
        y_range: Y 范围
        num_heights: 采样的高度层数
        z_range: Z 范围
        device: 设备
        
    Returns:
        ref_points: [bev_h * bev_w, num_heights, 3] 参考点坐标
    """
    # BEV 网格坐标
    x = torch.linspace(x_range[0], x_range[1], bev_w, device=device)
    y = torch.linspace(y_range[0], y_range[1], bev_h, device=device)
    
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    
    # 展平
    xx = xx.flatten()  # [H*W]
    yy = yy.flatten()  # [H*W]
    
    # 多个高度采样
    z = torch.linspace(z_range[0], z_range[1], num_heights, device=device)
    
    # 扩展到所有高度
    num_points = bev_h * bev_w
    xx = xx[:, None].expand(-1, num_heights)  # [H*W, num_heights]
    yy = yy[:, None].expand(-1, num_heights)
    zz = z[None, :].expand(num_points, -1)
    
    ref_points = torch.stack([xx, yy, zz], dim=-1)  # [H*W, num_heights, 3]
    
    return ref_points


def project_points_to_image(
    points_3d: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    image_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 3D 点投影到图像平面
    
    Args:
        points_3d: [N, 3] 3D 点坐标（车身坐标系）
        intrinsic: [3, 3] 相机内参
        extrinsic: [4, 4] 相机外参（车身→相机）
        image_size: (H, W) 图像尺寸
        
    Returns:
        uv: [N, 2] 图像坐标
        valid: [N] 有效性掩码（在图像范围内且深度为正）
    """
    N = points_3d.shape[0]
    device = points_3d.device
    
    # 转为齐次坐标
    points_homo = torch.cat([
        points_3d,
        torch.ones(N, 1, device=device),
    ], dim=1)  # [N, 4]
    
    # 车身坐标系 → 相机坐标系
    points_cam = (extrinsic @ points_homo.T).T  # [N, 4]
    points_cam = points_cam[:, :3]  # [N, 3]
    
    # 提取深度
    depth = points_cam[:, 2]
    
    # 投影到图像平面
    points_img = (intrinsic @ points_cam.T).T  # [N, 3]
    
    # 归一化
    u = points_img[:, 0] / (points_img[:, 2] + 1e-6)
    v = points_img[:, 1] / (points_img[:, 2] + 1e-6)
    
    uv = torch.stack([u, v], dim=1)  # [N, 2]
    
    # 有效性检查
    H, W = image_size
    valid = (
        (depth > 0) &
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H)
    )
    
    return uv, valid


def transform_points(
    points: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    """
    对点云应用变换矩阵
    
    Args:
        points: [N, 3] 或 [B, N, 3] 点坐标
        transform: [4, 4] 或 [B, 4, 4] 变换矩阵
        
    Returns:
        transformed: 变换后的点坐标
    """
    if points.dim() == 2:
        # [N, 3] -> [N, 4]
        points_homo = torch.cat([
            points,
            torch.ones(points.shape[0], 1, device=points.device),
        ], dim=1)
        
        # 变换
        transformed = (transform @ points_homo.T).T[:, :3]
        
    elif points.dim() == 3:
        # [B, N, 3]
        B, N, _ = points.shape
        
        points_homo = torch.cat([
            points,
            torch.ones(B, N, 1, device=points.device),
        ], dim=2)  # [B, N, 4]
        
        # 批量矩阵乘法
        transformed = torch.bmm(
            points_homo,
            transform.transpose(-1, -2)
        )[:, :, :3]  # [B, N, 3]
        
    else:
        raise ValueError(f"Unsupported points dimension: {points.dim()}")
        
    return transformed


def voxel_to_world(
    voxel_idx: torch.Tensor,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    grid_size: Tuple[int, int, int],
) -> torch.Tensor:
    """
    体素索引转世界坐标
    
    Args:
        voxel_idx: [N, 3] 体素索引 (i, j, k)
        x_range: X 范围
        y_range: Y 范围
        z_range: Z 范围
        grid_size: 网格尺寸 (nx, ny, nz)
        
    Returns:
        world_coords: [N, 3] 世界坐标 (x, y, z)
    """
    resolution_x = (x_range[1] - x_range[0]) / grid_size[0]
    resolution_y = (y_range[1] - y_range[0]) / grid_size[1]
    resolution_z = (z_range[1] - z_range[0]) / grid_size[2]
    
    x = x_range[0] + (voxel_idx[:, 0] + 0.5) * resolution_x
    y = y_range[0] + (voxel_idx[:, 1] + 0.5) * resolution_y
    z = z_range[0] + (voxel_idx[:, 2] + 0.5) * resolution_z
    
    return torch.stack([x, y, z], dim=1)


def world_to_voxel(
    world_coords: torch.Tensor,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    grid_size: Tuple[int, int, int],
) -> torch.Tensor:
    """
    世界坐标转体素索引
    
    Args:
        world_coords: [N, 3] 世界坐标 (x, y, z)
        x_range: X 范围
        y_range: Y 范围
        z_range: Z 范围
        grid_size: 网格尺寸 (nx, ny, nz)
        
    Returns:
        voxel_idx: [N, 3] 体素索引 (i, j, k)，int64
    """
    resolution_x = (x_range[1] - x_range[0]) / grid_size[0]
    resolution_y = (y_range[1] - y_range[0]) / grid_size[1]
    resolution_z = (z_range[1] - z_range[0]) / grid_size[2]
    
    i = ((world_coords[:, 0] - x_range[0]) / resolution_x).long()
    j = ((world_coords[:, 1] - y_range[0]) / resolution_y).long()
    k = ((world_coords[:, 2] - z_range[0]) / resolution_z).long()
    
    # 限制在有效范围内
    i = torch.clamp(i, 0, grid_size[0] - 1)
    j = torch.clamp(j, 0, grid_size[1] - 1)
    k = torch.clamp(k, 0, grid_size[2] - 1)
    
    return torch.stack([i, j, k], dim=1)


def compute_frustum_bounds(
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    image_size: Tuple[int, int],
    depth_range: Tuple[float, float] = (0.5, 100.0),
) -> torch.Tensor:
    """
    计算相机视锥体边界
    
    Args:
        intrinsic: [3, 3] 相机内参
        extrinsic: [4, 4] 相机外参
        image_size: (H, W) 图像尺寸
        depth_range: 深度范围
        
    Returns:
        corners: [8, 3] 视锥体 8 个角点坐标
    """
    H, W = image_size
    d_min, d_max = depth_range
    
    # 图像 4 个角点
    corners_2d = torch.tensor([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H],
    ], dtype=torch.float32)
    
    # 反投影到相机坐标系
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    corners_3d = []
    for d in [d_min, d_max]:
        for uv in corners_2d:
            u, v = uv
            x = (u - cx) * d / fx
            y = (v - cy) * d / fy
            z = d
            corners_3d.append([x, y, z])
            
    corners_3d = torch.tensor(corners_3d, dtype=torch.float32)
    
    # 相机坐标系 → 车身坐标系
    extrinsic_inv = torch.inverse(extrinsic)
    corners_world = transform_points(corners_3d, extrinsic_inv)
    
    return corners_world


# 测试代码
if __name__ == '__main__':
    print("Testing geometry utilities...")
    
    device = torch.device('cpu')
    
    # 1. 测试 meshgrid
    print("\n1. Testing create_meshgrid...")
    grid = create_meshgrid(
        x_range=(-10, 10),
        y_range=(-10, 10),
        z_range=(-2, 2),
        resolution=1.0,
        device=device,
    )
    print(f"   Grid shape: {grid.shape}")  # [20, 20, 4, 3]
    
    # 2. 测试参考点
    print("\n2. Testing get_reference_points...")
    ref_points = get_reference_points(
        bev_h=50, bev_w=50,
        num_heights=4,
        device=device,
    )
    print(f"   Reference points shape: {ref_points.shape}")  # [2500, 4, 3]
    
    # 3. 测试点投影
    print("\n3. Testing project_points_to_image...")
    points_3d = torch.tensor([
        [10, 0, 0],
        [20, 5, 0],
        [-5, 0, 1],
    ], dtype=torch.float32)
    
    intrinsic = torch.tensor([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1],
    ], dtype=torch.float32)
    
    extrinsic = torch.eye(4)
    
    uv, valid = project_points_to_image(
        points_3d, intrinsic, extrinsic, (480, 640)
    )
    print(f"   UV coordinates: {uv}")
    print(f"   Valid mask: {valid}")
    
    # 4. 测试坐标转换
    print("\n4. Testing coordinate conversion...")
    voxel_idx = torch.tensor([[100, 100, 10], [0, 0, 0]])
    world = voxel_to_world(
        voxel_idx,
        x_range=(-50, 50),
        y_range=(-50, 50),
        z_range=(-4, 4),
        grid_size=(200, 200, 16),
    )
    print(f"   Voxel {voxel_idx[0].tolist()} -> World {world[0].tolist()}")
    
    # 反向转换
    voxel_back = world_to_voxel(
        world,
        x_range=(-50, 50),
        y_range=(-50, 50),
        z_range=(-4, 4),
        grid_size=(200, 200, 16),
    )
    print(f"   World {world[0].tolist()} -> Voxel {voxel_back[0].tolist()}")
    
    print("\n✓ All tests passed!")

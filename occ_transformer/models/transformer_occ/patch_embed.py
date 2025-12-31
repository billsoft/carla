# models/transformer_occ/patch_embed.py
"""
Bayer Patch Embedding 模块

将 8 个相机的 Bayer RAW 图像转换为 patch 序列

输入: [B, 8, 1, H, W] - 8相机 12-bit Bayer
输出: [B, N_total, D] - 所有相机的 patch 序列
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BayerPatchEmbed(nn.Module):
    """
    Bayer Patch Embedding
    
    流程:
    1. PixelUnshuffle(2): [1, H, W] → [4, H/2, W/2] RGGB 分离
    2. Patch Conv: [4, H/2, W/2] → [D, H/patch, W/patch]
    3. Flatten: [D, H', W'] → [H'×W', D]
    
    Args:
        img_size: 原始图像尺寸 (H, W)
        patch_size: patch 大小（在 PixelUnshuffle 之后）
        in_channels: 输入通道数（Bayer=1）
        embed_dim: 嵌入维度
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (960, 1280),
        patch_size: int = 8,
        in_channels: int = 1,
        embed_dim: int = 256,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # PixelUnshuffle 后的尺寸
        self.unshuffle_size = (img_size[0] // 2, img_size[1] // 2)
        
        # Patch 后的网格尺寸
        self.grid_size = (
            self.unshuffle_size[0] // patch_size,
            self.unshuffle_size[1] // patch_size
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # 1. RGGB 分离
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        # [B, 1, H, W] → [B, 4, H/2, W/2]
        
        # 2. Patch Embedding (使用卷积实现)
        self.proj = nn.Conv2d(
            in_channels=4,  # RGGB 4 通道
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )
        
        # 3. LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        # 使用 truncated normal 初始化
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W] 单相机 Bayer 图像
            
        Returns:
            patches: [B, N_patches, D] patch 序列
        """
        B, C, H, W = x.shape
        assert C == 1, f"Expected 1 channel Bayer, got {C}"
        
        # 1. RGGB 分离
        x = self.pixel_unshuffle(x)  # [B, 4, H/2, W/2]
        
        # 2. Patch 卷积
        x = self.proj(x)  # [B, D, H', W']
        
        # 3. Flatten 为序列
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, D]
        
        # 4. LayerNorm
        x = self.norm(x)
        
        return x
    
    def get_output_size(self) -> Tuple[int, int, int]:
        """返回输出尺寸 (N_patches, grid_H, grid_W)"""
        return (self.num_patches, self.grid_size[0], self.grid_size[1])


class MultiCameraPatchEmbed(nn.Module):
    """
    多相机 Patch Embedding
    
    处理 8 个相机，共享 embedding 权重
    
    Args:
        num_cameras: 相机数量
        img_size: 图像尺寸
        patch_size: patch 大小
        embed_dim: 嵌入维度
    """
    
    def __init__(
        self,
        num_cameras: int = 8,
        img_size: Tuple[int, int] = (960, 1280),
        patch_size: int = 16,  # 推荐默认值 16
        embed_dim: int = 256,
    ):
        super().__init__()
        
        self.num_cameras = num_cameras
        self.embed_dim = embed_dim
        
        # 共享的 patch embedding
        self.patch_embed = BayerPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=embed_dim
        )
        
        # 相机 ID embedding
        self.camera_embed = nn.Embedding(num_cameras, embed_dim)
        
        # 输出信息
        self.num_patches_per_cam = self.patch_embed.num_patches
        self.total_patches = num_cameras * self.num_patches_per_cam
        self.grid_size = self.patch_embed.grid_size
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N_cam, 1, H, W] 多相机 Bayer 图像
            
        Returns:
            patches: [B, N_total, D] 所有相机的 patch 序列
            camera_ids: [B, N_total] 每个 patch 的相机 ID
        """
        B, N_cam, C, H, W = x.shape
        assert N_cam == self.num_cameras, f"Expected {self.num_cameras} cameras, got {N_cam}"
        
        all_patches = []
        all_camera_ids = []
        
        for cam_idx in range(N_cam):
            # 提取单相机图像
            cam_img = x[:, cam_idx]  # [B, 1, H, W]
            
            # Patch embedding
            patches = self.patch_embed(cam_img)  # [B, N_patches, D]
            
            # 添加相机 embedding
            cam_embed = self.camera_embed(
                torch.full((B, self.num_patches_per_cam), cam_idx, device=x.device, dtype=torch.long)
            )  # [B, N_patches, D]
            patches = patches + cam_embed
            
            all_patches.append(patches)
            
            # 记录相机 ID
            cam_ids = torch.full((B, self.num_patches_per_cam), cam_idx, device=x.device, dtype=torch.long)
            all_camera_ids.append(cam_ids)
        
        # 拼接所有相机
        patches = torch.cat(all_patches, dim=1)  # [B, N_total, D]
        camera_ids = torch.cat(all_camera_ids, dim=1)  # [B, N_total]
        
        return patches, camera_ids


if __name__ == '__main__':
    print("=" * 60)
    print("Bayer Patch Embedding 测试")
    print("=" * 60)
    
    # 测试单相机
    print("\n[1] 单相机 Patch Embedding:")
    patch_embed = BayerPatchEmbed(
        img_size=(960, 1280),
        patch_size=8,
        embed_dim=256
    )
    
    x = torch.randn(2, 1, 960, 1280)
    patches = patch_embed(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {patches.shape}")
    print(f"  Grid Size: {patch_embed.grid_size}")
    print(f"  Num Patches: {patch_embed.num_patches}")
    
    # 测试多相机
    print("\n[2] 多相机 Patch Embedding:")
    multi_embed = MultiCameraPatchEmbed(
        num_cameras=8,
        img_size=(960, 1280),
        patch_size=8,
        embed_dim=256
    )
    
    x_multi = torch.randn(2, 8, 1, 960, 1280)
    patches_multi, cam_ids = multi_embed(x_multi)
    print(f"  输入: {x_multi.shape}")
    print(f"  输出 patches: {patches_multi.shape}")
    print(f"  输出 camera_ids: {cam_ids.shape}")
    print(f"  每相机 patches: {multi_embed.num_patches_per_cam}")
    print(f"  总 patches: {multi_embed.total_patches}")
    
    # 参数量
    total_params = sum(p.numel() for p in multi_embed.parameters())
    print(f"\n参数量: {total_params/1e6:.2f}M")
    
    print("\n" + "=" * 60)
    print("✅ 测试通过！")
    print("=" * 60)

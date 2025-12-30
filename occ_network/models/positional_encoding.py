# models/positional_encoding.py
"""
位置编码模块

包含三种位置编码:
1. 2D 正弦位置编码 - 编码像素在图像中的位置
2. 相机 ID 嵌入 - 区分不同相机
3. 相机位姿编码 - 编码相机在车身坐标系的位置和朝向
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class SinusoidalPositionEncoding2D(nn.Module):
    """
    2D 正弦位置编码
    
    使用不同频率的 sin/cos 函数编码 (u, v) 坐标
    可泛化到任意分辨率
    """
    
    def __init__(
        self,
        embed_dim: int,
        temperature: float = 10000.0,
        normalize: bool = True,
    ):
        """
        Args:
            embed_dim: 输出嵌入维度
            temperature: 频率缩放温度
            normalize: 是否归一化坐标到 [0, 1]
        """
        super().__init__()
        
        assert embed_dim % 4 == 0, "embed_dim 必须是 4 的倍数"
        
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.normalize = normalize
        self.dim_per_axis = embed_dim // 2  # 每个轴用一半通道
        
    def forward(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        生成 2D 位置编码
        
        Args:
            height: 特征图高度
            width: 特征图宽度
            device: 目标设备
            
        Returns:
            pos_embed: [H, W, embed_dim] 位置编码
        """
        # 创建坐标网格
        y_coords = torch.arange(height, device=device, dtype=torch.float32)
        x_coords = torch.arange(width, device=device, dtype=torch.float32)
        
        # 可选归一化
        if self.normalize:
            y_coords = y_coords / (height - 1 + 1e-6)  # [0, 1]
            x_coords = x_coords / (width - 1 + 1e-6)
        
        # 频率分母
        dim_t = torch.arange(self.dim_per_axis // 2, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * dim_t / self.dim_per_axis)
        
        # X 方向编码: [W] -> [W, dim/4] -> [W, dim/2]
        pos_x = x_coords[:, None] / dim_t[None, :]
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
        
        # Y 方向编码: [H] -> [H, dim/4] -> [H, dim/2]
        pos_y = y_coords[:, None] / dim_t[None, :]
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)
        
        # 广播并拼接: [H, W, dim]
        pos_y = pos_y[:, None, :].expand(height, width, -1)
        pos_x = pos_x[None, :, :].expand(height, width, -1)
        
        pos_embed = torch.cat([pos_y, pos_x], dim=-1)
        
        return pos_embed


class CameraIDEmbedding(nn.Module):
    """
    相机 ID 可学习嵌入
    
    每个相机学习一个独特的嵌入向量，用于区分不同相机的特性
    """
    
    def __init__(self, num_cameras: int, embed_dim: int):
        super().__init__()
        
        self.num_cameras = num_cameras
        self.embed_dim = embed_dim
        
        self.embed = nn.Embedding(num_cameras, embed_dim)
        
        # 初始化
        nn.init.normal_(self.embed.weight, std=0.02)
        
    def forward(self, camera_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取相机嵌入
        
        Args:
            camera_ids: [N] 相机 ID，如果为 None 则返回所有相机的嵌入
            
        Returns:
            embeddings: [N, embed_dim] 或 [num_cameras, embed_dim]
        """
        if camera_ids is None:
            camera_ids = torch.arange(self.num_cameras, device=self.embed.weight.device)
            
        return self.embed(camera_ids)


class CameraPoseEncoding(nn.Module):
    """
    相机位姿 MLP 编码
    
    将相机外参矩阵（4x4）编码为嵌入向量
    外参包含旋转（3x3）和平移（3x1）
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 外参有效维度: 旋转 9 + 平移 3 = 12
        input_dim = 12
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, extrinsics: torch.Tensor) -> torch.Tensor:
        """
        编码相机位姿
        
        Args:
            extrinsics: [N, 4, 4] 相机外参矩阵
            
        Returns:
            pose_embed: [N, embed_dim] 位姿嵌入
        """
        # 提取旋转和平移
        rotation = extrinsics[:, :3, :3]  # [N, 3, 3]
        translation = extrinsics[:, :3, 3]  # [N, 3]
        
        # 展平并拼接
        rotation_flat = rotation.flatten(1)  # [N, 9]
        pose_vector = torch.cat([rotation_flat, translation], dim=1)  # [N, 12]
        
        # MLP 编码
        pose_embed = self.mlp(pose_vector)
        
        return pose_embed


class LearnableBEVPositionEncoding(nn.Module):
    """
    可学习的 BEV 位置编码
    
    用于 BEV Query 的位置编码
    """
    
    def __init__(self, bev_h: int, bev_w: int, embed_dim: int):
        super().__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dim = embed_dim
        
        # 可学习的位置编码
        self.row_embed = nn.Embedding(bev_h, embed_dim // 2)
        self.col_embed = nn.Embedding(bev_w, embed_dim // 2)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        
    def forward(self, device: torch.device) -> torch.Tensor:
        """
        生成 BEV 位置编码
        
        Returns:
            pos_embed: [bev_h, bev_w, embed_dim]
        """
        # 行列索引
        rows = torch.arange(self.bev_h, device=device)
        cols = torch.arange(self.bev_w, device=device)
        
        # 获取嵌入
        row_embed = self.row_embed(rows)  # [H, dim/2]
        col_embed = self.col_embed(cols)  # [W, dim/2]
        
        # 广播并拼接
        row_embed = row_embed[:, None, :].expand(-1, self.bev_w, -1)  # [H, W, dim/2]
        col_embed = col_embed[None, :, :].expand(self.bev_h, -1, -1)  # [H, W, dim/2]
        
        pos_embed = torch.cat([row_embed, col_embed], dim=-1)  # [H, W, dim]
        
        return pos_embed


class PositionalEncoder(nn.Module):
    """
    完整的位置编码器
    
    组合 2D 正弦编码 + 相机 ID 嵌入 + 相机位姿编码
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_cameras: int = 8,
        bev_h: int = 200,
        bev_w: int = 200,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.bev_h = bev_h
        self.bev_w = bev_w
        
        # 三个组件
        self.pe_2d = SinusoidalPositionEncoding2D(embed_dim)
        self.cam_embed = CameraIDEmbedding(num_cameras, embed_dim)
        self.pose_enc = CameraPoseEncoding(embed_dim)
        
        # BEV 位置编码
        self.bev_pos_enc = LearnableBEVPositionEncoding(bev_h, bev_w, embed_dim)
        
        # 融合投影
        self.fusion_proj = nn.Linear(embed_dim, embed_dim)
        
    def get_image_pos_encoding(
        self,
        height: int,
        width: int,
        extrinsics: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        获取图像特征的位置编码
        
        Args:
            height: 特征图高度
            width: 特征图宽度
            extrinsics: [num_cameras, 4, 4] 相机外参
            device: 目标设备
            
        Returns:
            pos_embed: [num_cameras, H, W, embed_dim]
        """
        # 1. 2D 位置编码: [H, W, dim]
        pe_2d = self.pe_2d(height, width, device)
        
        # 2. 相机 ID 嵌入: [num_cameras, dim]
        cam_embed = self.cam_embed()
        
        # 3. 位姿编码: [num_cameras, dim]
        pose_embed = self.pose_enc(extrinsics)
        
        # 4. 广播并组合
        # pe_2d: [H, W, dim] -> [1, H, W, dim] -> [num_cameras, H, W, dim]
        pe_2d = pe_2d.unsqueeze(0).expand(self.num_cameras, -1, -1, -1)
        
        # cam_embed: [num_cameras, dim] -> [num_cameras, 1, 1, dim] -> [num_cameras, H, W, dim]
        cam_embed = cam_embed[:, None, None, :].expand(-1, height, width, -1)
        
        # pose_embed: [num_cameras, dim] -> [num_cameras, 1, 1, dim] -> [num_cameras, H, W, dim]
        pose_embed = pose_embed[:, None, None, :].expand(-1, height, width, -1)
        
        # 5. 逐元素相加
        pos_embed = pe_2d + cam_embed + pose_embed
        
        # 6. 融合投影
        pos_embed = self.fusion_proj(pos_embed)
        
        return pos_embed
    
    def get_bev_pos_encoding(self, device: torch.device) -> torch.Tensor:
        """
        获取 BEV Query 的位置编码
        
        Returns:
            pos_embed: [bev_h * bev_w, embed_dim]
        """
        pos_embed = self.bev_pos_enc(device)  # [H, W, dim]
        pos_embed = pos_embed.flatten(0, 1)   # [H*W, dim]
        return pos_embed


# 测试代码
if __name__ == '__main__':
    print("Testing Positional Encoding modules...")
    
    device = torch.device('cpu')
    embed_dim = 256
    num_cameras = 8
    height, width = 48, 80
    
    # 1. 测试 2D 正弦编码
    print("\n1. Testing 2D Sinusoidal Encoding...")
    pe_2d = SinusoidalPositionEncoding2D(embed_dim)
    pos_2d = pe_2d(height, width, device)
    print(f"   Output shape: {pos_2d.shape}")  # [48, 80, 256]
    
    # 2. 测试相机 ID 嵌入
    print("\n2. Testing Camera ID Embedding...")
    cam_embed = CameraIDEmbedding(num_cameras, embed_dim)
    cam_emb = cam_embed()
    print(f"   Output shape: {cam_emb.shape}")  # [8, 256]
    
    # 3. 测试位姿编码
    print("\n3. Testing Camera Pose Encoding...")
    pose_enc = CameraPoseEncoding(embed_dim)
    extrinsics = torch.eye(4).unsqueeze(0).expand(num_cameras, -1, -1)
    pose_emb = pose_enc(extrinsics)
    print(f"   Output shape: {pose_emb.shape}")  # [8, 256]
    
    # 4. 测试完整编码器
    print("\n4. Testing Full Positional Encoder...")
    encoder = PositionalEncoder(embed_dim, num_cameras, bev_h=200, bev_w=200)
    
    img_pos = encoder.get_image_pos_encoding(height, width, extrinsics, device)
    print(f"   Image pos encoding shape: {img_pos.shape}")  # [8, 48, 80, 256]
    
    bev_pos = encoder.get_bev_pos_encoding(device)
    print(f"   BEV pos encoding shape: {bev_pos.shape}")  # [40000, 256]
    
    print("\n✓ All tests passed!")

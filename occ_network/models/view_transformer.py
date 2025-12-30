# models/view_transformer.py
"""
View Transformer 模块

将多相机 2D 图像特征变换为 BEV (Bird's Eye View) 特征

核心机制:
1. BEV Query: 可学习的查询向量，代表 BEV 网格上的每个位置
2. Cross Attention: Query 从图像特征中聚合信息
3. 位置编码: 帮助网络理解像素与 3D 空间的对应关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力
    
    Query 来自 BEV，Key/Value 来自图像特征
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
    def forward(
        self,
        query: torch.Tensor,          # [B, N_q, C]
        key: torch.Tensor,            # [B, N_kv, C]
        value: torch.Tensor,          # [B, N_kv, C]
        query_pos: Optional[torch.Tensor] = None,  # [B, N_q, C] 或 [N_q, C]
        key_pos: Optional[torch.Tensor] = None,    # [B, N_kv, C] 或 [N_kv, C]
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Returns:
            output: [B, N_q, C]
        """
        B, N_q, C = query.shape
        _, N_kv, _ = key.shape
        
        # 添加位置编码
        if query_pos is not None:
            if query_pos.dim() == 2:
                query_pos = query_pos.unsqueeze(0)
            query = query + query_pos
            
        if key_pos is not None:
            if key_pos.dim() == 2:
                key_pos = key_pos.unsqueeze(0)
            key = key + key_pos
        
        # 线性投影
        q = self.q_proj(query)  # [B, N_q, C]
        k = self.k_proj(key)    # [B, N_kv, C]
        v = self.v_proj(value)  # [B, N_kv, C]
        
        # 重塑为多头格式: [B, N, C] -> [B, num_heads, N, head_dim]
        q = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [B, num_heads, N_q, N_kv]
        
        # 可选的注意力掩码
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        # Softmax 归一化
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 加权聚合
        output = torch.matmul(attn, v)
        # [B, num_heads, N_q, head_dim]
        
        # 重塑回: [B, N_q, C]
        output = output.transpose(1, 2).contiguous().view(B, N_q, C)
        
        # 输出投影
        output = self.out_proj(output)
        
        return output


class TransformerDecoderLayer(nn.Module):
    """
    Transformer 解码器层
    
    包含:
    1. Self-Attention (BEV 位置之间的信息交换)
    2. Cross-Attention (从图像特征聚合信息)
    3. FFN (非线性变换)
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = 'relu',
    ):
        super().__init__()
        
        # Self Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross Attention
        self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,              # [B, N_q, C]
        key: torch.Tensor,                # [B, N_kv, C]
        value: torch.Tensor,              # [B, N_kv, C]
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播"""
        
        # 1. Self Attention
        q = k = query + query_pos if query_pos is not None else query
        self_attn_out, _ = self.self_attn(q, k, query)
        query = query + self.dropout1(self_attn_out)
        query = self.norm1(query)
        
        # 2. Cross Attention
        cross_attn_out = self.cross_attn(
            query, key, value,
            query_pos=query_pos,
            key_pos=key_pos,
        )
        query = query + self.dropout2(cross_attn_out)
        query = self.norm2(query)
        
        # 3. FFN
        ffn_out = self.ffn(query)
        query = query + self.dropout3(ffn_out)
        query = self.norm3(query)
        
        return query


class BEVQueryGenerator(nn.Module):
    """
    BEV Query 生成器
    
    生成可学习的 BEV Query 向量
    """
    
    def __init__(
        self,
        bev_h: int = 200,
        bev_w: int = 200,
        embed_dim: int = 256,
    ):
        super().__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dim = embed_dim
        
        # 可学习的 Query
        self.query_embed = nn.Parameter(
            torch.randn(bev_h * bev_w, embed_dim) * 0.02
        )
        
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        生成 BEV Query
        
        Args:
            batch_size: batch 大小
            
        Returns:
            query: [B, bev_h * bev_w, embed_dim]
        """
        query = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        return query


class ViewTransformer(nn.Module):
    """
    View Transformer
    
    将多相机图像特征变换为 BEV 特征
    
    数据流:
    1. 图像特征 [B, 8, C, H, W] -> 展平 -> [B, 8*H*W, C]
    2. BEV Query [B, bev_h*bev_w, C]
    3. Cross Attention: Query 从图像中聚合信息
    4. 重塑为 BEV: [B, C, bev_h, bev_w]
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        bev_h: int = 200,
        bev_w: int = 200,
        num_cameras: int = 8,
        feature_h: int = 48,
        feature_w: int = 80,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cameras = num_cameras
        self.feature_h = feature_h
        self.feature_w = feature_w
        
        # BEV Query 生成器
        self.query_generator = BEVQueryGenerator(bev_h, bev_w, embed_dim)
        
        # 可学习的 BEV 位置编码
        self.bev_pos_embed = nn.Parameter(
            torch.randn(bev_h * bev_w, embed_dim) * 0.02
        )
        
        # Transformer 解码器层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(
        self,
        img_features: torch.Tensor,  # [B, num_cameras, C, H, W]
        img_pos_embed: torch.Tensor,  # [num_cameras, H, W, C]
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            img_features: 多相机图像特征 [B, num_cameras, C, H, W]
            img_pos_embed: 图像位置编码 [num_cameras, H, W, C]
            
        Returns:
            bev_features: BEV 特征 [B, C, bev_h, bev_w]
        """
        B = img_features.shape[0]
        device = img_features.device
        
        # 1. 准备图像 Key-Value
        # [B, num_cameras, C, H, W] -> [B, num_cameras, H, W, C]
        img_feat = img_features.permute(0, 1, 3, 4, 2)
        
        # 展平: [B, num_cameras * H * W, C]
        img_feat_flat = img_feat.flatten(1, 3)
        
        # 位置编码展平: [num_cameras, H, W, C] -> [num_cameras * H * W, C]
        img_pos_flat = img_pos_embed.flatten(0, 2)
        
        # 2. 生成 BEV Query
        query = self.query_generator(B)  # [B, bev_h * bev_w, C]
        
        # BEV 位置编码
        bev_pos = self.bev_pos_embed  # [bev_h * bev_w, C]
        
        # 3. Transformer 解码
        output = query
        for layer in self.layers:
            output = layer(
                query=output,
                key=img_feat_flat,
                value=img_feat_flat,
                query_pos=bev_pos,
                key_pos=img_pos_flat,
            )
        
        # 4. 重塑为 BEV 特征图
        # [B, bev_h * bev_w, C] -> [B, bev_h, bev_w, C] -> [B, C, bev_h, bev_w]
        bev_features = output.view(B, self.bev_h, self.bev_w, self.embed_dim)
        bev_features = bev_features.permute(0, 3, 1, 2).contiguous()
        
        # 5. 输出投影
        bev_features = self.output_proj(bev_features)
        
        return bev_features


# EfficientViewTransformer 已删除
# 原因: 未实现真正的 Deformable Attention，实际是标准 Attention
# 如需 Deformable Attention，请使用专用库如 MultiScaleDeformableAttention


# 测试代码
if __name__ == '__main__':
    print("Testing View Transformer...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 配置
    B = 2
    num_cameras = 8
    embed_dim = 256
    feature_h, feature_w = 48, 80
    bev_h, bev_w = 200, 200
    
    # 创建模块
    view_transformer = ViewTransformer(
        embed_dim=embed_dim,
        num_heads=8,
        num_layers=2,  # 减少层数用于测试
        bev_h=bev_h,
        bev_w=bev_w,
        num_cameras=num_cameras,
        feature_h=feature_h,
        feature_w=feature_w,
    ).to(device)
    
    # 模拟输入
    img_features = torch.randn(B, num_cameras, embed_dim, feature_h, feature_w).to(device)
    img_pos_embed = torch.randn(num_cameras, feature_h, feature_w, embed_dim).to(device)
    
    # 前向传播
    print(f"\nInput shapes:")
    print(f"  img_features: {img_features.shape}")
    print(f"  img_pos_embed: {img_pos_embed.shape}")
    
    with torch.no_grad():
        bev_features = view_transformer(img_features, img_pos_embed)
    
    print(f"\nOutput shape:")
    print(f"  bev_features: {bev_features.shape}")  # [2, 256, 200, 200]
    
    # 统计参数量
    total_params = sum(p.numel() for p in view_transformer.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    
    print("\n✓ Test passed!")

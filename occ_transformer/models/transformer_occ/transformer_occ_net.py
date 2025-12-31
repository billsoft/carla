# models/transformer_occ/transformer_occ_net.py
"""
统一 Transformer Occupancy Network

将多视角 2D 图像"翻译"为 3D 体素占用网格
核心思想：图像到体素 = 序列到序列翻译

输入: [B, 8, 1, H, W] - 8 相机 12-bit Bayer RAW
输出: [B, num_classes, X, Y, Z] - 3D 占用网格 logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

try:
    from models.transformer_occ.patch_embed import MultiCameraPatchEmbed
    from models.transformer_occ.position_encoding import CameraPositionEncoding, Spatial2DPositionEncoding
    from models.transformer_occ.encoder import TransformerEncoder, MultiCameraEncoder
    from models.transformer_occ.voxel_query import VoxelQueries, BEVQueries
    from models.transformer_occ.decoder import VoxelDecoder, SimplifiedDecoder
except ImportError:
    # 如果作为模块导入时
    from .patch_embed import MultiCameraPatchEmbed
    from .position_encoding import CameraPositionEncoding, Spatial2DPositionEncoding
    from .encoder import TransformerEncoder, MultiCameraEncoder
    from .voxel_query import VoxelQueries, BEVQueries
    from .decoder import VoxelDecoder, SimplifiedDecoder


class TransformerOccNet(nn.Module):
    """
    统一 Transformer Occupancy Network
    
    架构:
    1. Patch Embedding: Bayer → Patches
    2. Position Encoding: Camera PE + Spatial PE
    3. Encoder: Transformer 编码图像
    4. Voxel Queries: 可学习的 3D 查询
    5. Decoder: Transformer 解码体素
    6. Head: 预测占用类别
    
    Args:
        num_cameras: 相机数量
        img_size: 图像尺寸 (H, W)
        patch_size: Patch 大小
        embed_dim: 嵌入维度
        encoder_layers: 编码器层数
        decoder_layers: 解码器层数
        num_heads: 注意力头数
        ffn_dim: FFN 隐藏维度
        query_grid_size: 查询网格大小 (X, Y, Z)
        output_grid_size: 输出网格大小
        num_classes: 类别数
        dropout: Dropout 率
        use_deformable_attn: 是否使用可变形注意力
    """
    
    def __init__(
        self,
        num_cameras: int = 8,
        img_size: Tuple[int, int] = (960, 1280),
        patch_size: int = 16,  # 推荐默认值 16
        embed_dim: int = 256,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        query_grid_size: Tuple[int, int, int] = (50, 50, 8),
        output_grid_size: Tuple[int, int, int] = (200, 200, 16),
        num_classes: int = 18,
        dropout: float = 0.0,
        use_deformable_attn: bool = True,
        x_range: Tuple[float, float] = (-25.0, 25.0),
        y_range: Tuple[float, float] = (-25.0, 25.0),
        z_range: Tuple[float, float] = (-2.0, 6.0),
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.num_cameras = num_cameras
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.query_grid_size = query_grid_size
        self.output_grid_size = output_grid_size
        self.num_classes = num_classes
        
        # ==================== Patch Embedding ====================
        self.patch_embed = MultiCameraPatchEmbed(
            num_cameras=num_cameras,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        # 计算特征尺寸
        self.grid_h = self.patch_embed.grid_size[0]
        self.grid_w = self.patch_embed.grid_size[1]
        self.num_patches_per_cam = self.patch_embed.num_patches_per_cam
        self.total_patches = self.patch_embed.total_patches
        
        # ==================== Position Encoding ====================
        # 2D 空间位置编码
        self.spatial_pe = Spatial2DPositionEncoding(
            grid_size=self.patch_embed.grid_size,
            embed_dim=embed_dim,
            learnable=True
        )
        
        # 相机位置编码
        self.camera_pe = CameraPositionEncoding(
            embed_dim=embed_dim,
            num_cameras=num_cameras,
            grid_size=self.patch_embed.grid_size
        )
        
        # ==================== Encoder ====================
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_window_attn=True,  # 使用 use_window_attn 替代 attention_type
            window_size=8,
            use_checkpoint=use_checkpoint
        )
        
        # ==================== Voxel Queries ====================
        self.voxel_queries = VoxelQueries(
            grid_size=query_grid_size,
            embed_dim=embed_dim,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            learnable_pe=True
        )
        
        # ==================== Decoder ====================
        self.decoder = VoxelDecoder(
            embed_dim=embed_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_deformable_attn=use_deformable_attn,
            num_deform_points=4,
            query_grid_size=query_grid_size,
            output_grid_size=output_grid_size,
            num_classes=num_classes
        )
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(
        self,
        images: torch.Tensor,
        camera_intrinsics: Optional[torch.Tensor] = None,
        camera_extrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: [B, N_cam, 1, H, W] 多相机 Bayer 图像
            camera_intrinsics: [B, N_cam, 3, 3] 相机内参（可选）
            camera_extrinsics: [B, N_cam, 4, 4] 相机外参（可选）
            
        Returns:
            occ_logits: [B, num_classes, X, Y, Z] 占用 logits
        """
        B, N_cam, C, H, W = images.shape
        device = images.device
        
        # ==================== 1. Patch Embedding ====================
        patches, camera_ids = self.patch_embed(images)  # [B, N_total, D]
        
        # ==================== 2. Position Encoding ====================
        # 2D 空间位置编码
        spatial_pe = self.spatial_pe()  # [H'*W', D]
        
        # 为每个相机添加空间位置编码
        all_pe = []
        for cam_idx in range(N_cam):
            # 获取相机位置编码
            cam_pe = self.camera_pe(
                camera_idx=cam_idx,
                batch_size=B,
                device=device
            )  # [B, N_per_cam, D]
            
            # 组合空间 PE 和相机 PE
            combined_pe = spatial_pe.unsqueeze(0).expand(B, -1, -1) + cam_pe
            all_pe.append(combined_pe)
            
        position_embed = torch.cat(all_pe, dim=1)  # [B, N_total, D]
        
        # 添加位置编码
        patches = patches + position_embed
        
        # ==================== 3. Encode ====================
        # 使用窗口注意力编码
        # 将所有相机视为一个大图像处理
        # MultiCameraEncoder 期望传入单相机的 H 和 W，但这里我们使用的是 TransformerEncoder
        # TransformerEncoder 对 H 和 W 的处理是直接用于 Window Attention 的 resize
        # 由于我们这里是将所有相机拼接在一起（维度1），所以这里的 H 和 W 应该是总的 H 和 W
        
        # 修正：TransformerEncoder 是将输入视为 [B, N_total, D]
        # 如果使用 Window Attention，它需要知道 N_total 对应的 2D 形状
        # 对于 8 个相机，我们可以视为 H x (W * 8) 或者 (H * 8) x W
        # 这里我们选择 H x (W * 8)
        
        total_H = self.grid_h
        total_W = self.grid_w * N_cam
        
        encoded = self.encoder(patches, H=total_H, W=total_W)  # [B, N_total, D]
        
        # ==================== 4. Voxel Queries ====================
        queries, query_pos, reference_points = self.voxel_queries(B)
        # queries: [B, num_voxels, D]
        # reference_points: [B, num_voxels, 2]
        
        # ==================== 5. Decode ====================
        # 准备空间形状信息
        memory_spatial_shapes = torch.tensor(
            [[total_H, total_W]], 
            device=device
        )
        
        occ_logits = self.decoder(
            query=queries,
            memory=encoded,
            query_pos=query_pos,
            reference_points=reference_points,
            memory_spatial_shapes=memory_spatial_shapes
        )  # [B, num_classes, X, Y, Z]
        
        return occ_logits
    
    def get_params_summary(self) -> Dict[str, float]:
        """获取参数量统计"""
        def count_params(module):
            return sum(p.numel() for p in module.parameters()) / 1e6
            
        return {
            'patch_embed': count_params(self.patch_embed),
            'position_encoding': count_params(self.spatial_pe) + count_params(self.camera_pe),
            'encoder': count_params(self.encoder),
            'voxel_queries': count_params(self.voxel_queries),
            'decoder': count_params(self.decoder),
            'total': count_params(self)
        }


class TransformerOccNetLite(nn.Module):
    """
    轻量级版本
    
    使用 BEV Queries 而非完整的 3D Queries
    更高效，适合部署
    """
    
    def __init__(
        self,
        num_cameras: int = 8,
        img_size: Tuple[int, int] = (960, 1280),
        patch_size: int = 16,  # 更大的 patch 减少序列长度
        embed_dim: int = 256,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        num_heads: int = 8,
        bev_size: Tuple[int, int] = (100, 100),
        num_height_levels: int = 16,
        output_grid_size: Tuple[int, int, int] = (200, 200, 16),
        num_classes: int = 18,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.num_cameras = num_cameras
        self.bev_size = bev_size
        self.output_grid_size = output_grid_size
        
        # Patch Embedding
        self.patch_embed = MultiCameraPatchEmbed(
            num_cameras=num_cameras,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        self.grid_h = self.patch_embed.grid_size[0]
        self.grid_w = self.patch_embed.grid_size[1]
        
        # Position Encoding
        self.spatial_pe = Spatial2DPositionEncoding(
            grid_size=self.patch_embed.grid_size,
            embed_dim=embed_dim
        )
        
        self.camera_pe = CameraPositionEncoding(
            embed_dim=embed_dim,
            num_cameras=num_cameras,
            grid_size=self.patch_embed.grid_size
        )
        
        # Encoder (轻量)
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            ffn_dim=embed_dim * 4,
            dropout=dropout,
            use_window_attn=True,
            window_size=8,
            use_checkpoint=use_checkpoint
        )
        
        # BEV Queries
        self.bev_queries = BEVQueries(
            bev_size=bev_size,
            num_height_levels=num_height_levels,
            embed_dim=embed_dim
        )
        
        # 简化 Decoder
        self.decoder = SimplifiedDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            query_grid_size=(bev_size[0], bev_size[1], num_height_levels),
            output_grid_size=output_grid_size,
            num_classes=num_classes,
            use_deformable=True
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N_cam = images.shape[:2]
        device = images.device
        
        # 1. Patch Embedding
        patches, _ = self.patch_embed(images)
        
        # 2. Position Encoding
        spatial_pe = self.spatial_pe()
        all_pe = []
        for cam_idx in range(N_cam):
            cam_pe = self.camera_pe(cam_idx, B, device=device)
            combined_pe = spatial_pe.unsqueeze(0).expand(B, -1, -1) + cam_pe
            all_pe.append(combined_pe)
        position_embed = torch.cat(all_pe, dim=1)
        patches = patches + position_embed
        
        # 3. Encode
        total_W = self.grid_w * N_cam
        encoded = self.encoder(patches, H=self.grid_h, W=total_W)
        
        # 4. BEV Queries
        queries, _, ref_points = self.bev_queries(B)
        
        # 5. 将 BEV queries 扩展到 3D
        bev_features = self.decoder.cross_attn(
            queries, ref_points, encoded,
            torch.tensor([[self.grid_h, total_W]], device=device)
        )
        bev_features = self.decoder.norm(bev_features + queries)
        
        # 扩展到 3D
        features_3d = self.bev_queries.expand_to_3d(bev_features)
        
        # 6. 3D 卷积和预测
        features_3d = self.decoder.conv3d(features_3d) + features_3d
        occ_logits = self.decoder.head(features_3d)
        
        return occ_logits
    
    def get_params_summary(self) -> Dict[str, float]:
        def count_params(module):
            return sum(p.numel() for p in module.parameters()) / 1e6
        return {
            'patch_embed': count_params(self.patch_embed),
            'encoder': count_params(self.encoder),
            'bev_queries': count_params(self.bev_queries),
            'decoder': count_params(self.decoder),
            'total': count_params(self)
        }


class TransformerOccNetMini(nn.Module):
    """
    Mini 版本 (介于 Nano 和 Lite 之间)
    
    配置:
    - BEV: 50x50
    - Embed Dim: 192
    - Encoder: 4层
    - Decoder: 4层
    """
    
    def __init__(
        self,
        num_cameras: int = 8,
        img_size: Tuple[int, int] = (960, 1280),
        patch_size: int = 16,
        embed_dim: int = 192,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        num_heads: int = 6,
        bev_size: Tuple[int, int] = (50, 50),
        num_height_levels: int = 16,
        output_grid_size: Tuple[int, int, int] = (200, 200, 16),
        num_classes: int = 18,
        dropout: float = 0.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        
        self.num_cameras = num_cameras
        self.bev_size = bev_size
        self.output_grid_size = output_grid_size
        
        # Patch Embedding
        self.patch_embed = MultiCameraPatchEmbed(
            num_cameras=num_cameras,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        self.grid_h = self.patch_embed.grid_size[0]
        self.grid_w = self.patch_embed.grid_size[1]
        
        # Position Encoding
        self.spatial_pe = Spatial2DPositionEncoding(
            grid_size=self.patch_embed.grid_size,
            embed_dim=embed_dim
        )
        
        self.camera_pe = CameraPositionEncoding(
            embed_dim=embed_dim,
            num_cameras=num_cameras,
            grid_size=self.patch_embed.grid_size
        )
        
        # Encoder
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            ffn_dim=embed_dim * 4,
            dropout=dropout,
            use_window_attn=True,
            window_size=8,
            use_checkpoint=use_checkpoint
        )
        
        # BEV Queries
        self.bev_queries = BEVQueries(
            bev_size=bev_size,
            num_height_levels=num_height_levels,
            embed_dim=embed_dim
        )
        
        # Decoder (使用 SimplifiedDecoder 但参数更强)
        self.decoder = SimplifiedDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            query_grid_size=(bev_size[0], bev_size[1], num_height_levels),
            output_grid_size=output_grid_size,
            num_classes=num_classes,
            use_deformable=True
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N_cam = images.shape[:2]
        device = images.device
        
        # 1. Patch Embedding
        patches, _ = self.patch_embed(images)
        
        # 2. Position Encoding
        spatial_pe = self.spatial_pe()
        all_pe = []
        for cam_idx in range(N_cam):
            cam_pe = self.camera_pe(cam_idx, B, device=device)
            combined_pe = spatial_pe.unsqueeze(0).expand(B, -1, -1) + cam_pe
            all_pe.append(combined_pe)
        position_embed = torch.cat(all_pe, dim=1)
        patches = patches + position_embed
        
        # 3. Encode
        total_W = self.grid_w * N_cam
        encoded = self.encoder(patches, H=self.grid_h, W=total_W)
        
        # 4. BEV Queries
        queries, _, ref_points = self.bev_queries(B)
        
        # 5. Decode
        bev_features = self.decoder.cross_attn(
            queries, ref_points, encoded,
            torch.tensor([[self.grid_h, total_W]], device=device)
        )
        bev_features = self.decoder.norm(bev_features + queries)
        
        # 扩展到 3D
        features_3d = self.bev_queries.expand_to_3d(bev_features)
        
        # 6. Predict
        features_3d = self.decoder.conv3d(features_3d) + features_3d
        occ_logits = self.decoder.head(features_3d)
        
        return occ_logits
    
    def get_params_summary(self) -> Dict[str, float]:
        def count_params(module):
            return sum(p.numel() for p in module.parameters()) / 1e6
        return {
            'patch_embed': count_params(self.patch_embed),
            'encoder': count_params(self.encoder),
            'bev_queries': count_params(self.bev_queries),
            'decoder': count_params(self.decoder),
            'total': count_params(self)
        }


def build_transformer_occ_net(
    model_type: str = 'standard',
    num_classes: int = 18,
    img_size: Tuple[int, int] = (960, 1280),
    output_grid_size: Tuple[int, int, int] = (200, 200, 16),
    **kwargs
) -> nn.Module:
    """
    构建 Transformer Occupancy Network
    
    Args:
        model_type: 'standard' 或 'lite'
        num_classes: 类别数
        img_size: 图像尺寸
        output_grid_size: 输出网格大小
        **kwargs: 其他参数
    """
    if model_type == 'standard':
        return TransformerOccNet(
            num_classes=num_classes,
            img_size=img_size,
            output_grid_size=output_grid_size,
            **kwargs
        )
    elif model_type == 'lite':
        return TransformerOccNetLite(
            num_classes=num_classes,
            img_size=img_size,
            output_grid_size=output_grid_size,
            **kwargs
        )
    elif model_type == 'mini':
        return TransformerOccNetMini(
            num_classes=num_classes,
            img_size=img_size,
            output_grid_size=output_grid_size,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    print("=" * 70)
    print("Transformer Occupancy Network 测试".center(70))
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # 测试标准版
    print("\n" + "-" * 70)
    print("[1] 标准版 TransformerOccNet")
    print("-" * 70)
    
    model = TransformerOccNet(
        num_cameras=8,
        img_size=(960, 1280),
        patch_size=8,
        embed_dim=256,
        encoder_layers=6,
        decoder_layers=6,
        query_grid_size=(50, 50, 8),
        output_grid_size=(200, 200, 16),
        num_classes=18
    ).to(device)
    
    # 打印参数
    params = model.get_params_summary()
    print("\n参数量统计:")
    for name, value in params.items():
        print(f"  {name}: {value:.2f}M")
        
    # 测试前向传播
    print("\n前向传播测试:")
    images = torch.randn(1, 8, 1, 960, 1280, device=device)
    print(f"  输入: {images.shape}")
    
    with torch.no_grad():
        occ_logits = model(images)
    print(f"  输出: {occ_logits.shape}")
    print(f"  预期: [1, 18, 200, 200, 16]")
    
    # 测试轻量版
    print("\n" + "-" * 70)
    print("[2] 轻量版 TransformerOccNetLite")
    print("-" * 70)
    
    model_lite = TransformerOccNetLite(
        num_cameras=8,
        img_size=(960, 1280),
        patch_size=16,
        embed_dim=256,
        encoder_layers=4,
        decoder_layers=2,
        bev_size=(100, 100),
        output_grid_size=(200, 200, 16),
        num_classes=18
    ).to(device)
    
    params_lite = model_lite.get_params_summary()
    print("\n参数量统计:")
    for name, value in params_lite.items():
        print(f"  {name}: {value:.2f}M")
        
    print("\n前向传播测试:")
    with torch.no_grad():
        occ_logits_lite = model_lite(images)
    print(f"  输入: {images.shape}")
    print(f"  输出: {occ_logits_lite.shape}")
    
    # 显存测试
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(images)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n显存占用 (标准版, BS=1): {peak_mem:.2f} GB")
        
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model_lite(images)
        peak_mem_lite = torch.cuda.max_memory_allocated() / 1e9
        print(f"显存占用 (轻量版, BS=1): {peak_mem_lite:.2f} GB")
    
    print("\n" + "=" * 70)
    print("✅ 测试通过！".center(70))
    print("=" * 70)

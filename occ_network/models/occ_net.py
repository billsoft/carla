"""
OccNetV3 - 3D Occupancy Prediction Network (改进版)

改进点:
1. 深度感知融合: 深度参与 2D→3D 重建 (Lift-Splat 风格)
2. 时序融合 TBPTT: 近期帧保留梯度，场景边界自动检测
3. 多尺度 BEV 解码: 大/中/小物体分别处理
4. 动态物体运动估计: 处理非自车运动
5. 边缘感知深度监督: 不在边缘处强制深度平滑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .patch_embed import MultiCameraPatchEmbed
from .encoder import MultiCameraEncoder
from .decoder import BEVDecoder, MultiScaleBEVDecoder, CoarseHeightExpansion, LightweightUpsampler, DepthPredictionHead
from .temporal import LightweightTemporalFusion
from .heads import MultiTaskHead, CoarseToFineHead
from .position_encoding import CameraPositionEncoding
from .sparse_modules import AdaptiveSparseProcessor, SPCONV_AVAILABLE, TORCHSPARSE_AVAILABLE
from .depth_to_3d import DepthAwareFusion, LiftSplatModule

class OccNetV3(nn.Module):
    """
    OccNetV3 - 3D Occupancy Prediction Network (改进版)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_fp16_input = getattr(config, 'use_fp16_input', True)

        # ==================== 特征提取 ====================
        self.patch_embed = MultiCameraPatchEmbed(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            num_cameras=config.num_cameras
        )
        self.camera_pe = CameraPositionEncoding(
            dim=config.embed_dim,
            num_cameras=config.num_cameras,
            image_size=config.image_size,
            camera_configs=config.cameras,
            patch_size=config.patch_size
        )
        self.encoder = MultiCameraEncoder(
            dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_encoder_layers,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            drop=config.drop_rate,
            attn_drop=config.attn_drop_rate,
            use_checkpoint=config.use_checkpoint
        )

        # ==================== 深度感知融合 (改进) ====================
        self.use_depth_aware_fusion = getattr(config, 'use_depth_aware_fusion', True)
        self.use_depth_supervision = getattr(config, 'use_depth_supervision', True)

        if self.use_depth_aware_fusion:
            # 使用深度感知融合 (Lift-Splat 风格)
            self.depth_fusion = LiftSplatModule(
                in_channels=config.embed_dim,
                out_channels=config.embed_dim,
                num_depth_bins=getattr(config, 'num_depth_bins', 64),
                depth_range=getattr(config, 'depth_range', (0.5, 80.0)),
                bev_size=config.bev_size,
                pc_range=config.pc_range,
                num_cameras=config.num_cameras,
                image_size=config.image_size,
                patch_size=config.patch_size,
            )
        else:
            # 原始简单融合
            self.fusion_proj = nn.Linear(config.embed_dim * config.num_cameras, config.embed_dim)

            # 深度预测头 (用于深度监督)
            if self.use_depth_supervision:
                feat_h = config.image_size[0] // config.patch_size
                feat_w = config.image_size[1] // config.patch_size
                self.depth_head = DepthPredictionHead(
                    in_channels=config.embed_dim,
                    num_depth_bins=getattr(config, 'num_depth_bins', 64),
                    feature_size=(feat_h, feat_w),
                    depth_range=getattr(config, 'depth_range', (0.5, 80.0))
                )

        # ==================== BEV 解码 (改进: 支持多尺度) ====================
        self.use_multi_scale_bev = getattr(config, 'use_multi_scale_bev', False)

        if self.use_multi_scale_bev:
            self.decoder = MultiScaleBEVDecoder(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                num_layers=config.num_decoder_layers,
                bev_h=config.bev_size[0],
                bev_w=config.bev_size[1],
                num_points=config.num_points,
                mlp_ratio=config.mlp_ratio,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                use_checkpoint=config.use_checkpoint,
                scales=(0.25, 0.5, 1.0),  # 多尺度
            )
        else:
            self.decoder = BEVDecoder(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                num_layers=config.num_decoder_layers,
                bev_h=config.bev_size[0],
                bev_w=config.bev_size[1],
                num_points=config.num_points,
                mlp_ratio=config.mlp_ratio,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                use_checkpoint=config.use_checkpoint
            )

        # ==================== 时序融合 (改进: TBPTT + 场景检测) ====================
        self.temporal = LightweightTemporalFusion(
            dim=config.embed_dim,
            num_frames=config.num_frames,
            bev_h=config.bev_size[0],
            bev_w=config.bev_size[1],
            pc_range=config.pc_range,
            tbptt_steps=getattr(config, 'tbptt_steps', 3),
            use_dynamic_motion=getattr(config, 'use_dynamic_motion', True),
            use_st_encoding=getattr(config, 'use_st_encoding', True),
        )

        # ==================== 3D 重建 ====================
        num_heights = config.voxel_size[2] // 4
        self.height_expand = CoarseHeightExpansion(config.embed_dim, num_heights)
        self.upsampler = LightweightUpsampler(
            in_channels=config.embed_dim,
            out_channels=config.embed_dim // 2,
            target_size=tuple(config.voxel_size),
            use_checkpoint=config.use_checkpoint
        )

        # ==================== 预测头 ====================
        self.use_coarse_to_fine = getattr(config, 'use_coarse_to_fine', True)
        self.use_sparse = getattr(config, 'use_sparse', False) and (SPCONV_AVAILABLE or TORCHSPARSE_AVAILABLE)

        if self.use_coarse_to_fine:
            self.head = CoarseToFineHead(
                in_channels=config.embed_dim // 2,
                num_classes=config.num_classes,
                coarse_size=tuple(config.coarse_voxel_size),
                fine_size=tuple(config.voxel_size),
                use_flow=config.use_flow,
                chunk_size_z=config.chunk_size_z
            )
        else:
            self.head = MultiTaskHead(
                in_channels=config.embed_dim // 2,
                num_classes=config.num_classes,
                use_flow=config.use_flow,
                chunk_size_z=config.chunk_size_z
            )

        if self.use_sparse:
            self.sparse_refine = AdaptiveSparseProcessor(
                in_channels=config.num_classes,
                num_classes=config.num_classes,
                hidden_channels=32,
                sparsity_threshold=config.sparsity_threshold
            )

    def reset_temporal(self):
        """重置时序历史 (场景切换时调用)"""
        self.temporal.reset()

    def forward(self, images, ego_motion=None, ego_pose=None,
                timestamp=None, scene_id=None,
                intrinsics=None, extrinsics=None,
                tbptt_encoder_detach=False):
        """
        前向传播

        Args:
            images: [B, N, C, H, W] 多相机图像
            ego_motion: [B, 4, 4] 自车运动矩阵 (可选)
            ego_pose: [B, 4, 4] 当前帧世界位姿 (可选)
            timestamp: float 当前帧时间戳 (用于时序编码)
            scene_id: str 场景ID (用于检测场景切换)
            intrinsics: [B, N, 3, 3] 相机内参 (用于 LiftSplat)
            extrinsics: [B, N, 4, 4] 相机外参 (用于 LiftSplat)
            tbptt_encoder_detach: bool 是否分离 Encoder 的梯度 (分阶段 TBPTT)
        """
        B, N, C, H, W = images.shape
        device = images.device

        # ==================== 特征提取 ====================
        camera_tokens, spatial_shape = self.patch_embed(images)
        encoded_tokens = self.encoder(camera_tokens, spatial_shape, self.camera_pe)
        feat_h, feat_w = spatial_shape

        # ==================== 深度感知融合 (改进) ====================
        if self.use_depth_aware_fusion:
            # 使用深度感知融合 (深度参与 2D→3D 重建)
            features = torch.stack([t.transpose(1, 2).reshape(B, -1, feat_h, feat_w) for t in encoded_tokens], dim=1)
            
            fused_tokens, depth_logits, depth_pred = self.depth_fusion(
                features,
                camera_intrinsics=intrinsics,
                camera_extrinsics=extrinsics
            )
            
            # 🔑 分阶段 TBPTT: 分离 Encoder 梯度
            if tbptt_encoder_detach and self.training:
                fused_tokens = fused_tokens.detach().requires_grad_(True)
            
            fused_tokens = fused_tokens.flatten(2).transpose(1, 2) # [B, L, C]
            
            bev_h, bev_w = self.config.bev_size
            spatial_shapes = torch.tensor([[bev_h, bev_w]], device=device)
             
        else:
            # 原始简单融合
            if self.use_depth_supervision:
                depth_features = []
                for cam_idx, cam_tokens in enumerate(encoded_tokens):
                    cam_feat = cam_tokens.transpose(1, 2).reshape(B, -1, feat_h, feat_w)
                    depth_features.append(cam_feat)
                depth_features = torch.stack(depth_features, dim=1)
                depth_logits, depth_pred = self.depth_head(depth_features)
            else:
                depth_logits, depth_pred = None, None

            all_tokens = torch.cat(encoded_tokens, dim=-1)
            fused_tokens = self.fusion_proj(all_tokens)
            
            # 🔑 分阶段 TBPTT: 分离 Encoder 梯度
            if tbptt_encoder_detach and self.training:
                fused_tokens = fused_tokens.detach().requires_grad_(True)
                
            spatial_shapes = torch.tensor([[feat_h, feat_w]], device=device)

        # ==================== BEV 解码 ====================
        bev_features = self.decoder(fused_tokens, spatial_shapes)

        # ==================== 时序融合 (改进: TBPTT + 场景检测) ====================
        bev_features = self.temporal(
            bev_features, ego_motion, ego_pose,
            timestamp=timestamp, scene_id=scene_id
        )

        # ==================== 3D 重建 ====================
        voxel_features = self.height_expand(bev_features)
        voxel_features = self.upsampler(voxel_features)

        # ==================== 预测头 ====================
        outputs = self.head(voxel_features)

        # 稀疏优化 (可选)
        if self.use_sparse and 'coarse_semantic' in outputs:
            coarse_pred = outputs['coarse_semantic']
            coarse_up = F.interpolate(
                coarse_pred,
                size=tuple(self.config.voxel_size),
                mode='trilinear',
                align_corners=False
            )
            sparse_refined = self.sparse_refine(outputs['semantic'], coarse_up)
            outputs['semantic'] = sparse_refined

        # ==================== 添加深度输出 (用于监督) ====================
        if self.use_depth_supervision or self.use_depth_aware_fusion:
            if depth_logits is not None:
                outputs['depth_logits'] = depth_logits  # [B, N, D, H, W]
            if depth_pred is not None:
                outputs['depth_pred'] = depth_pred      # [B, N, H, W]

        return outputs

    @torch.no_grad()
    def inference(self, images):
        if self.use_fp16_input and images.dtype == torch.float32:
            images = images.half()
        outputs = self.forward(images)
        semantic_logits = outputs['semantic']
        semantic_pred = semantic_logits.argmax(dim=1)
        return {'pred': semantic_pred, 'logits': semantic_logits, 'flow': outputs.get('flow')}

def build_model(config):
    """
    构建 OccNetV3 模型并打印配置信息
    """
    model = OccNetV3(config)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"OccNetV3 Model Configuration")
    print(f"{'='*60}")
    print(f"Total parameters: {num_params / 1e6:.2f}M")
    print(f"Trainable parameters: {num_trainable / 1e6:.2f}M")

    # Sparse 状态
    sparse_status = 'Disabled'
    if getattr(config, 'use_sparse', False):
        if SPCONV_AVAILABLE:
            sparse_status = 'Enabled (spconv)'
        elif TORCHSPARSE_AVAILABLE:
            sparse_status = 'Enabled (torchsparse)'
        else:
            sparse_status = 'Disabled (No backend found)'
    print(f"Sparse convolution: {sparse_status}")

    # 其他配置
    print(f"Coarse-to-Fine: {'Enabled' if getattr(config, 'use_coarse_to_fine', True) else 'Disabled'}")
    print(f"Flash Attention: {'Available' if hasattr(F, 'scaled_dot_product_attention') else 'Not available'}")

    # 改进功能状态
    print(f"\n[Improvements V2]")
    print(f"  Depth-Aware Fusion: {'Enabled' if getattr(config, 'use_depth_aware_fusion', True) else 'Disabled'}")
    print(f"  Depth Supervision: {'Enabled' if getattr(config, 'use_depth_supervision', True) else 'Disabled'}")
    print(f"  Multi-Scale BEV: {'Enabled' if getattr(config, 'use_multi_scale_bev', False) else 'Disabled'}")
    print(f"  TBPTT (Temporal): {getattr(config, 'tbptt_steps', 3)} steps")
    print(f"  Dynamic Motion Est: {'Enabled' if getattr(config, 'use_dynamic_motion', True) else 'Disabled'}")
    print(f"  Spatio-Temporal Enc: {'Enabled' if getattr(config, 'use_st_encoding', True) else 'Disabled'}")
    print(f"{'='*60}\n")

    return model

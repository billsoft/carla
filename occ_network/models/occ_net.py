import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .patch_embed import MultiCameraPatchEmbed
from .encoder import MultiCameraEncoder
from .decoder import BEVDecoder, CoarseHeightExpansion, LightweightUpsampler
from .temporal import LightweightTemporalFusion
from .heads import MultiTaskHead, CoarseToFineHead
from .position_encoding import CameraPositionEncoding
from .sparse_modules import AdaptiveSparseProcessor, SPCONV_AVAILABLE, TORCHSPARSE_AVAILABLE

class OccNetV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_fp16_input = getattr(config, 'use_fp16_input', True)
        self.patch_embed = MultiCameraPatchEmbed(img_size=config.image_size, patch_size=config.patch_size, in_channels=config.in_channels, embed_dim=config.embed_dim, num_cameras=config.num_cameras)
        self.camera_pe = CameraPositionEncoding(dim=config.embed_dim, num_cameras=config.num_cameras, image_size=config.image_size, camera_configs=config.cameras, patch_size=config.patch_size)
        self.encoder = MultiCameraEncoder(dim=config.embed_dim, num_heads=config.num_heads, num_layers=config.num_encoder_layers, window_size=config.window_size, mlp_ratio=config.mlp_ratio, drop=config.drop_rate, attn_drop=config.attn_drop_rate, use_checkpoint=config.use_checkpoint, use_rope_fov=True)
        self.fusion_proj = nn.Linear(config.embed_dim * config.num_cameras, config.embed_dim)
        self.decoder = BEVDecoder(dim=config.embed_dim, num_heads=config.num_heads, num_layers=config.num_decoder_layers, bev_h=config.bev_size[0], bev_w=config.bev_size[1], num_points=config.num_points, mlp_ratio=config.mlp_ratio, drop=config.drop_rate, attn_drop=config.attn_drop_rate, use_checkpoint=config.use_checkpoint)
        self.temporal = LightweightTemporalFusion(dim=config.embed_dim, num_frames=config.num_frames, bev_h=config.bev_size[0], bev_w=config.bev_size[1], pc_range=config.pc_range)
        num_heights = config.voxel_size[2] // 4
        self.height_expand = CoarseHeightExpansion(config.embed_dim, num_heights)
        self.upsampler = LightweightUpsampler(in_channels=config.embed_dim, out_channels=config.embed_dim // 2, target_size=tuple(config.voxel_size), use_checkpoint=config.use_checkpoint)
        self.use_coarse_to_fine = getattr(config, 'use_coarse_to_fine', True)
        
        # Sparse Backend Logic
        self.use_sparse = getattr(config, 'use_sparse', False) and (SPCONV_AVAILABLE or TORCHSPARSE_AVAILABLE)
        
        if self.use_coarse_to_fine:
            self.head = CoarseToFineHead(in_channels=config.embed_dim // 2, num_classes=config.num_classes, coarse_size=tuple(config.coarse_voxel_size), fine_size=tuple(config.voxel_size), use_flow=config.use_flow, chunk_size_z=config.chunk_size_z)
        else:
            self.head = MultiTaskHead(in_channels=config.embed_dim // 2, num_classes=config.num_classes, use_flow=config.use_flow, chunk_size_z=config.chunk_size_z)
        if self.use_sparse:
            self.sparse_refine = AdaptiveSparseProcessor(in_channels=config.num_classes, num_classes=config.num_classes, hidden_channels=32, sparsity_threshold=config.sparsity_threshold)

    def reset_temporal(self):
        self.temporal.reset()

    def forward(self, images, ego_motion=None, ego_pose=None):
        B, N, C, H, W = images.shape
        camera_tokens, spatial_shape = self.patch_embed(images)
        encoded_tokens = self.encoder(camera_tokens, spatial_shape, self.camera_pe)
        feat_h, feat_w = spatial_shape
        all_tokens = torch.cat(encoded_tokens, dim=-1)
        fused_tokens = self.fusion_proj(all_tokens)
        spatial_shapes = torch.tensor([[feat_h, feat_w]], device=images.device)
        bev_features = self.decoder(fused_tokens, spatial_shapes)
        bev_features = self.temporal(bev_features, ego_motion, ego_pose)
        voxel_features = self.height_expand(bev_features)
        voxel_features = self.upsampler(voxel_features)
        outputs = self.head(voxel_features)
        if self.use_sparse and 'coarse_semantic' in outputs:
            coarse_pred = outputs['coarse_semantic']
            coarse_up = F.interpolate(coarse_pred, size=tuple(self.config.voxel_size), mode='trilinear', align_corners=False)
            sparse_refined = self.sparse_refine(outputs['semantic'], coarse_up)
            outputs['semantic'] = sparse_refined
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
    model = OccNetV3(config)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params / 1e6:.2f}M")
    print(f"Trainable parameters: {num_trainable / 1e6:.2f}M")
    
    sparse_status = 'Disabled'
    if config.use_sparse:
        if SPCONV_AVAILABLE:
            sparse_status = 'Enabled (spconv)'
        elif TORCHSPARSE_AVAILABLE:
            sparse_status = 'Enabled (torchsparse)'
        else:
            sparse_status = 'Disabled (No backend found)'
            
    print(f"Sparse convolution: {sparse_status}")
    print(f"Coarse-to-Fine: {'Enabled' if config.use_coarse_to_fine else 'Disabled'}")
    print(f"Flash Attention: {'Available' if hasattr(F, 'scaled_dot_product_attention') else 'Not available'}")
    return model

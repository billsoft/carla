import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from position_encoding import SineCosinePositionEncoding3D
from deformable_attention import DeformableDecoderLayer

class OccupancyDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.coarse_query = nn.Parameter(torch.randn(1, config.num_coarse_queries, config.embed_dim) * 0.02)
        self.pos_3d = SineCosinePositionEncoding3D(config.embed_dim)
        
        self.coarse_layers = nn.ModuleList([
            DeformableDecoderLayer(
                config.embed_dim, config.num_heads, config.num_cameras,
                config.num_sample_points, dropout=config.dropout,
                use_self_attn=config.use_self_attention # Use config switch
            ) for _ in range(config.decoder_layers)
        ])
        
        self.coarse_to_fine = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.GELU(),
            nn.Linear(config.embed_dim * 2, config.embed_dim),
        )
        
        self.fine_layers = nn.ModuleList([
            DeformableDecoderLayer(
                config.embed_dim, config.num_heads, config.num_cameras,
                config.num_sample_points, dropout=config.dropout,
                use_self_attn=False # Disable self-attn for fine queries to save memory (Too heavy for 160k)
            ) for _ in range(config.decoder_layers)
        ])
        
        self.register_buffer('coarse_ref', self._create_reference_points(config.coarse_size))
        self.register_buffer('fine_ref', self._create_reference_points(config.fine_size))
        
        # Checkpoint Strategy
        self.checkpoint_coarse = False
        self.checkpoint_fine = True
    
    def _create_reference_points(self, size):
        x = torch.linspace(0, 1, size[0])
        y = torch.linspace(0, 1, size[1])
        z = torch.linspace(0, 1, size[2])
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        ref = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return ref.view(-1, 3)
    
    def forward(self, image_feats, intrinsics=None, extrinsics=None, memory=None, ego_motion=None):
        B = image_feats.shape[0]
        device = image_feats.device
        cx, cy, cz = self.config.coarse_size
        
        # Coarse Stage
        coarse_pos = self.pos_3d(cx, cy, cz, device)
        query = self.coarse_query.expand(B, -1, -1) + coarse_pos.unsqueeze(0)
        ref = self.coarse_ref.unsqueeze(0).expand(B, -1, -1)
        
        for layer in self.coarse_layers:
            if self.checkpoint_coarse and self.training:
                query = checkpoint(layer, query, ref, image_feats, intrinsics, extrinsics, use_reentrant=False)
            else:
                query = layer(query, ref, image_feats, intrinsics, extrinsics)
            
        coarse_feats = query.view(B, cx, cy, cz, -1).permute(0, 4, 1, 2, 3)
        
        # Temporal Fusion (BEV Space)
        new_memory = None
        if self.config.use_temporal and hasattr(self, 'temporal_fusion'):
             # [B, C, X, Y, Z] -> [B, X*Y*Z, C]
             coarse_flat = coarse_feats.permute(0, 2, 3, 4, 1).flatten(1, 3)
             # Pass ego_motion and spatial shape for alignment
             fused_flat, new_memory = self.temporal_fusion(
                 coarse_flat, 
                 memory, 
                 ego_motion=ego_motion, 
                 spatial_shape=(cx, cy, cz)
             )
             # Reshape back [B, Q, C] -> [B, X, Y, Z, C] -> [B, C, X, Y, Z]
             coarse_feats = fused_flat.view(B, cx, cy, cz, self.config.embed_dim).permute(0, 4, 1, 2, 3)
        
        # Fine Stage Setup
        fx, fy, fz = self.config.fine_size
        fine_feats = F.interpolate(coarse_feats, size=(fx, fy, fz), mode='trilinear', align_corners=False)
        fine_feats = fine_feats.permute(0, 2, 3, 4, 1).reshape(B, -1, self.config.embed_dim)
        fine_feats = self.coarse_to_fine(fine_feats)
        
        fine_pos = self.pos_3d(fx, fy, fz, device)
        query = fine_feats + fine_pos.unsqueeze(0)
        ref = self.fine_ref.unsqueeze(0).expand(B, -1, -1)
        
        # Fine Layers with Checkpointing
        for layer in self.fine_layers:
            if self.checkpoint_fine and self.training:
                query = checkpoint(layer, query, ref, image_feats, intrinsics, extrinsics, use_reentrant=False)
            else:
                query = layer(query, ref, image_feats, intrinsics, extrinsics)
                
        output = query.view(B, fx, fy, fz, -1).permute(0, 4, 1, 2, 3)
        return output, new_memory

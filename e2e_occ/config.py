from dataclasses import dataclass
from typing import Tuple

@dataclass
class E2EOccConfig:
    num_cameras: int = 8
    image_size: Tuple[int, int] = (960, 1280)
    raw_channels: int = 1
    
    # Balanced Capacity (Target 18-20GB)
    embed_dim: int = 256                    # 384 -> 256
    num_heads: int = 8
    encoder_layers: int = 4                 # 6 -> 4
    decoder_layers: int = 2                 # 3 -> 2
    
    # Optimized Resolution
    coarse_size: Tuple[int, int, int] = (25, 25, 8)        # 20K -> 5K queries (Fixes OOM in Self-Attn)
    fine_size: Tuple[int, int, int] = (80, 80, 16)         # 102.4K queries
    voxel_size: Tuple[int, int, int] = (400, 400, 32)
    
    num_classes: int = 18
    num_sample_points: int = 4
    dropout: float = 0.1
    
    # Feature Switches
    use_ray_encoding: bool = True
    use_self_attention: bool = True         # Coarse only
    
    # Temporal Fusion Settings
    use_temporal: bool = True               # Enable temporal fusion
    temporal_frames: int = 2                # Number of frames in sequence
    memory_dim: int = 256                   # Should match embed_dim
    
    # feat_size is property
    voxel_range: Tuple[float, ...] = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)
    voxel_resolution: float = 0.2
    
    @property
    def feat_size(self) -> Tuple[int, int]:
        return (self.image_size[0] // 16, self.image_size[1] // 16)
    
    @property
    def num_coarse_queries(self) -> int:
        return self.coarse_size[0] * self.coarse_size[1] * self.coarse_size[2]
    
    @property
    def num_fine_queries(self) -> int:
        return self.fine_size[0] * self.fine_size[1] * self.fine_size[2]

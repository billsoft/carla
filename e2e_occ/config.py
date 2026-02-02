from dataclasses import dataclass
from typing import Tuple

@dataclass
class E2EOccConfig:
    num_cameras: int = 8
    image_size: Tuple[int, int] = (960, 1280)
    raw_channels: int = 1
    embed_dim: int = 256
    num_heads: int = 4  # Reduced from 8
    encoder_layers: int = 3 # Reduced from 4
    decoder_layers: int = 2 # Reduced from 3
    coarse_size: Tuple[int, int, int] = (50, 50, 8)
    fine_size: Tuple[int, int, int] = (80, 80, 12) # Reduced from (100, 100, 16)
    voxel_size: Tuple[int, int, int] = (400, 400, 32)
    num_classes: int = 18
    num_sample_points: int = 2 # Reduced from 4
    dropout: float = 0.1
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

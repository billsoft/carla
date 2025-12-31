# models/transformer_occ/__init__.py
"""
统一 Transformer Occupancy Network 模块

核心思想: 图像到体素 = 序列到序列翻译问题
相机参数 = 位置编码（几何先验）
"""

from .patch_embed import BayerPatchEmbed, MultiCameraPatchEmbed
from .position_encoding import (
    SinusoidalPositionEncoding,
    LearnablePositionEncoding,
    Spatial2DPositionEncoding,
    CameraPositionEncoding,
    Voxel3DPositionEncoding
)
from .attention import (
    MultiHeadAttention,
    WindowAttention,
    DeformableAttention,
    EfficientAttention
)
from .encoder import (
    FeedForward,
    EncoderLayer,
    TransformerEncoder,
    HierarchicalEncoder,
    MultiCameraEncoder
)
from .voxel_query import (
    VoxelQueries,
    HierarchicalVoxelQueries,
    BEVQueries
)
from .decoder import (
    DecoderLayer,
    TransformerDecoder,
    VoxelDecoder,
    SimplifiedDecoder,
    BalancedDecoder
)
from .transformer_occ_net import (
    TransformerOccNet,
    TransformerOccNetLite,
    TransformerOccNetMini,
    build_transformer_occ_net
)
from .transformer_occ_nano_net import TransformerOccNetNano
from .transformer_occ_balanced_net import TransformerOccNetBalanced

__all__ = [
    # Patch Embedding
    'BayerPatchEmbed',
    'MultiCameraPatchEmbed',
    
    # Position Encoding
    'SinusoidalPositionEncoding',
    'LearnablePositionEncoding',
    'Spatial2DPositionEncoding',
    'CameraPositionEncoding',
    'Voxel3DPositionEncoding',
    
    # Attention
    'MultiHeadAttention',
    'WindowAttention',
    'DeformableAttention',
    'EfficientAttention',
    
    # Encoder
    'FeedForward',
    'EncoderLayer',
    'TransformerEncoder',
    'HierarchicalEncoder',
    'MultiCameraEncoder',
    
    # Voxel Queries
    'VoxelQueries',
    'HierarchicalVoxelQueries',
    'BEVQueries',
    
    # Decoder
    'DecoderLayer',
    'TransformerDecoder',
    'VoxelDecoder',
    'SimplifiedDecoder',
    'BalancedDecoder',
    
    # Main Network
    'TransformerOccNet',
    'TransformerOccNetLite',
    'TransformerOccNetMini',
    'TransformerOccNetNano',
    'TransformerOccNetBalanced',
    'build_transformer_occ_net',
]

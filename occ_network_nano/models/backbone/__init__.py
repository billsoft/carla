"""
Backbone 模块

支持的 Backbone:
- BayerMobileNetV2: 单通道 Bayer RGGB 输入
"""

from .mobilenet_v2_bayer import BayerMobileNetV2, build_bayer_mobilenetv2

__all__ = [
    'BayerMobileNetV2',
    'build_bayer_mobilenetv2',
]

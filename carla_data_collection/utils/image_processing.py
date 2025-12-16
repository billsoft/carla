"""
图像处理工具 - 12-bit RAW 转换
"""

import numpy as np


def convert_to_12bit_raw(bgra_image: np.ndarray) -> np.ndarray:
    """
    将 CARLA 8-bit BGRA 转换为 12-bit RAW

    Args:
        bgra_image: (H, W, 4) uint8, 范围 [0, 255]

    Returns:
        raw_image: (H, W, 3) uint16, 范围 [0, 4095]
    """
    # 1. 提取 RGB 通道
    rgb = bgra_image[:, :, :3]  # (H, W, 3)
    rgb = rgb[:, :, ::-1]  # BGR → RGB

    # 2. 转换为 float32
    rgb_float = rgb.astype(np.float32) / 255.0  # [0, 1]

    # 3. 移除 Gamma 校正(假设 gamma=2.2)
    rgb_linear = np.power(rgb_float, 2.2)

    # 4. 扩展到 12-bit 范围
    rgb_12bit = (rgb_linear * 4095.0).astype(np.uint16)

    # 5. Clip 到有效范围
    rgb_12bit = np.clip(rgb_12bit, 0, 4095)

    return rgb_12bit


def visualize_12bit_image(image_12bit: np.ndarray) -> np.ndarray:
    """
    将 12-bit 图像转换为可显示的 8-bit

    Args:
        image_12bit: (H, W, 3) uint16, 范围 [0, 4095]

    Returns:
        image_8bit: (H, W, 3) uint8, 范围 [0, 255]
    """
    # 简单线性映射
    image_8bit = (image_12bit / 4095.0 * 255.0).astype(np.uint8)
    return image_8bit

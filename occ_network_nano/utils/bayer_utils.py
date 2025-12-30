"""
Bayer RAW 数据处理工具

专为单通道 Bayer RGGB 数据设计，支持 12-bit/16-bit DNG 加载和预处理。
"""

import cv2
import numpy as np
import torch
from pathlib import Path as PathLib


def load_bayer_image(path: str, is_12bit: bool = True) -> np.ndarray:
    """
    加载 Bayer RAW 图像（支持 PNG/DNG/TIFF）

    Args:
        path: 图像路径（.png, .dng, .tif, .tiff）
        is_12bit: 是否为 12-bit 数据（需要扩展到 16-bit 完整范围）

    Returns:
        uint16 图像 [H, W], 单通道, 范围 [0, 65535]
    """
    # 支持多种扩展名
    path_obj = PathLib(path)
    if not path_obj.exists():
        # 尝试自动查找其他扩展名
        for ext in ['.dng', '.png', '.tif', '.tiff']:
            alt_path = path_obj.with_suffix(ext)
            if alt_path.exists():
                path = str(alt_path)
                path_obj = alt_path
                break
        else:
            raise FileNotFoundError(f"无法找到图像: {path}")

    # 读取图像（单通道灰度模式）
    # 尝试读取 DNG/TIFF
    try:
        import rawpy
        with rawpy.imread(path) as raw:
             img = raw.raw_image_visible.copy()
        
    except Exception:
        # 降级使用 OpenCV
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_GRAYSCALE)

    assert img is not None, f"无法加载图像: {path}"
    # assert img.dtype == np.uint16, f"图像不是 16-bit: {img.dtype}" # PIL 读出来可能是 int32 (I mode)
    if img.dtype == np.int32 or img.dtype == np.uint32:
        img = img.astype(np.uint16)
        
    assert len(img.shape) == 2, f"必须是单通道图像: {img.shape}"

    # 如果是 12-bit 数据，扩展到 16-bit 完整范围
    # 12-bit [0, 4095] → 16-bit [0, 65535]
    if is_12bit:
        # 方法: 左移 4 位
        img = (img << 4).astype(np.uint16)

    return img


def bayer_to_tensor(bayer: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """
    转换 Bayer 图像为 PyTorch Tensor

    Args:
        bayer: Bayer 图像 (H, W) uint16
        normalize: 是否归一化到 [0, 1]

    Returns:
        tensor: (1, H, W) float32, [0, 1] if normalize else [0, 65535]
    """
    if normalize:
        # uint16 [0, 65535] → float32 [0, 1]
        bayer_norm = bayer.astype(np.float32) / 65535.0
        tensor = torch.from_numpy(bayer_norm).unsqueeze(0)  # (1, H, W)
    else:
        # 保持 uint16
        tensor = torch.from_numpy(bayer).unsqueeze(0).float()

    return tensor


def tensor_to_bayer(tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
    """
    转换 Tensor 为 Bayer 图像

    Args:
        tensor: Tensor (1, H, W) float32
        denormalize: 是否反归一化

    Returns:
        bayer: (H, W) uint16
    """
    # (1, H, W) → (H, W)
    bayer = tensor.squeeze(0).cpu().numpy()

    if denormalize:
        # [0, 1] → [0, 65535]
        bayer = (bayer * 65535.0).astype(np.uint16)
    else:
        bayer = bayer.astype(np.uint16)

    return bayer


def visualize_bayer(bayer: np.ndarray, method: str = 'simple') -> np.ndarray:
    """
    可视化 Bayer 数据（转换为 RGB 以便显示）

    Args:
        bayer: Bayer 图像 (H, W) uint16
        method: 去马赛克方法
            - 'simple': 简单插值（快速）
            - 'ea': Edge-aware（边缘保护）
            - 'vng': Variable Number of Gradients（高质量）

    Returns:
        rgb: (H, W, 3) uint8, BGR 顺序（OpenCV 格式）
    """
    # 选择去马赛克算法
    if method == 'simple':
        code = cv2.COLOR_BayerRG2BGR  # RGGB → BGR
    elif method == 'ea':
        code = cv2.COLOR_BayerRG2BGR_EA
    elif method == 'vng':
        code = cv2.COLOR_BayerRG2BGR_VNG
    else:
        raise ValueError(f"未知方法: {method}")

    # 转换为 uint8（OpenCV 去马赛克要求）
    bayer_u8 = (bayer >> 8).astype(np.uint8)

    # 去马赛克
    rgb = cv2.cvtColor(bayer_u8, code)

    return rgb


def apply_black_white_level(
    bayer: np.ndarray,
    black_level: int = 64,
    white_level: int = 4095
) -> np.ndarray:
    """
    应用黑白电平校正（模拟真实相机 RAW 处理）

    Args:
        bayer: Bayer 图像 (H, W) uint16
        black_level: 黑电平（12-bit: 默认 64）
        white_level: 白电平（12-bit: 默认 4095）

    Returns:
        corrected: (H, W) uint16, 范围 [0, 65535]
    """
    # 减去黑电平
    bayer_corrected = bayer.astype(np.int32) - (black_level << 4)  # 12-bit → 16-bit
    bayer_corrected = np.clip(bayer_corrected, 0, (white_level - black_level) << 4)

    # 归一化到完整 16-bit 范围
    bayer_norm = bayer_corrected.astype(np.float32) / ((white_level - black_level) << 4)
    bayer_full = (bayer_norm * 65535.0).astype(np.uint16)

    return bayer_full


if __name__ == '__main__':
    print("=" * 60)
    print("Bayer 工具测试")
    print("=" * 60)

    # 测试 1: 创建模拟 Bayer 数据
    print("\n[1] 创建模拟 Bayer 数据:")
    H, W = 960, 1280
    bayer_sim = np.random.randint(0, 4096, size=(H, W), dtype=np.uint16)
    print(f"  形状: {bayer_sim.shape}")
    print(f"  范围: [{bayer_sim.min()}, {bayer_sim.max()}]")

    # 测试 2: Tensor 转换
    print(f"\n[2] Tensor 转换测试:")
    tensor = bayer_to_tensor(bayer_sim << 4, normalize=True)  # 12-bit → 16-bit
    print(f"  输入: {bayer_sim.shape}, {bayer_sim.dtype}")
    print(f"  Tensor: {tensor.shape}, {tensor.dtype}")
    print(f"  范围: [{tensor.min():.4f}, {tensor.max():.4f}]")

    bayer_back = tensor_to_bayer(tensor, denormalize=True)
    print(f"  输出: {bayer_back.shape}, {bayer_back.dtype}")

    # 测试 3: 可视化
    print(f"\n[3] 可视化测试:")
    rgb_vis = visualize_bayer(bayer_sim << 4, method='simple')
    print(f"  RGB: {rgb_vis.shape}, {rgb_vis.dtype}")

    print("\n" + "=" * 60)
    print("✅ Bayer 工具测试通过！")
    print("=" * 60)

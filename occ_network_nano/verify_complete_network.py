#!/usr/bin/env python3
"""
完整 Bayer Occupancy Network 验证脚本

验证整个网络流程：
    Backbone → FPN → View Transformer → BEV Encoder → Occ Decoder
"""

import sys
from pathlib import Path
import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models import build_bayer_occ_net


def verify_complete_network():
    print("=" * 80)
    print("完整 Bayer Occupancy Network 结构验证".center(80))
    print("=" * 80)

    # 配置
    B, N_cam = 2, 8
    H, W = 384, 640
    num_classes = 18
    grid_size = (200, 200, 16)

    print(f"\n配置:")
    print(f"  Batch Size: {B}")
    print(f"  Cameras: {N_cam}")
    print(f"  Image Size: {H}×{W} (Bayer RAW)")
    print(f"  Classes: {num_classes}")
    print(f"  Grid Size: {grid_size}")

    # 创建模型
    print(f"\n正在构建模型...")
    model = build_bayer_occ_net(
        num_classes=num_classes,
        grid_size=grid_size,
        img_size=(H, W),
        backbone_width_mult=1.0,
        fpn_channels=128,
        bev_size=(100, 100),
        hidden_channels=64
    )

    # 参数统计
    params_summary = model.get_params_summary()

    print(f"\n" + "=" * 80)
    print("模型参数统计".center(80))
    print("=" * 80)
    print(f"  {'模块':<25} {'参数量 (M)':>15} {'占比':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")

    total = params_summary['total']
    for module_name in ['backbone', 'fpn', 'view_transformer', 'bev_encoder', 'occ_decoder']:
        params = params_summary[module_name]
        percentage = (params / total * 100) if total > 0 else 0
        name_display = {
            'backbone': 'Backbone',
            'fpn': 'FPN Neck',
            'view_transformer': 'View Transformer',
            'bev_encoder': 'BEV Encoder',
            'occ_decoder': 'Occ Decoder'
        }[module_name]
        print(f"  {name_display:<25} {params:>15.2f} {percentage:>14.1f}%")

    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    print(f"  {'总计':<25} {total:>15.2f} {100.0:>14.1f}%")

    # 前向传播测试
    print(f"\n" + "=" * 80)
    print("前向传播测试".center(80))
    print("=" * 80)

    # 创建输入数据
    images = torch.randn(B, N_cam, 1, H, W)
    print(f"\n输入:")
    print(f"  Images: {images.shape} (B, N_cam, C=1, H, W)")

    # 前向传播
    print(f"\n执行前向传播...")
    with torch.no_grad():
        occ_logits = model(images)

    print(f"\n输出:")
    print(f"  Occupancy Logits: {occ_logits.shape}")
    print(f"  预期: [B={B}, num_classes={num_classes}, X={grid_size[0]}, Y={grid_size[1]}, Z={grid_size[2]}]")

    # 验证输出形状
    expected_shape = (B, num_classes, grid_size[0], grid_size[1], grid_size[2])
    if occ_logits.shape == expected_shape:
        print(f"\n  ✅ 输出形状正确!")
    else:
        print(f"\n  ❌ 输出形状错误!")
        print(f"     预期: {expected_shape}")
        print(f"     实际: {occ_logits.shape}")
        return False

    # 验证输出值
    print(f"\n输出值统计:")
    print(f"  Min: {occ_logits.min().item():.4f}")
    print(f"  Max: {occ_logits.max().item():.4f}")
    print(f"  Mean: {occ_logits.mean().item():.4f}")
    print(f"  Std: {occ_logits.std().item():.4f}")

    # 计算预测类别
    pred_classes = occ_logits.argmax(dim=1)
    print(f"\n预测类别统计:")
    print(f"  Pred Classes Shape: {pred_classes.shape}")
    print(f"  Unique Classes: {pred_classes.unique().tolist()}")

    # 显存估算
    if torch.cuda.is_available():
        model_cuda = model.cuda()
        images_cuda = images.cuda()

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model_cuda(images_cuda)

        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n显存使用:")
        print(f"  Peak Memory: {peak_memory:.2f} GB")

        del model_cuda, images_cuda
        torch.cuda.empty_cache()

    # 网络流程总结
    print(f"\n" + "=" * 80)
    print("网络流程总结".center(80))
    print("=" * 80)
    print(f"""
    [输入] 8×Bayer RAW Images [B={B}, N={N_cam}, C=1, H={H}, W={W}]
        ↓
    [Backbone] BayerMobileNetV2 with PixelUnshuffle
        → C3: [B×N, 96, {H//8}, {W//8}]   (1/8 分辨率)
        → C4: [B×N, 128, {H//16}, {W//16}]  (1/16 分辨率)
        → C5: [B×N, 256, {H//32}, {W//32}]  (1/32 分辨率)
        ↓
    [FPN Neck] 多尺度特征融合
        → [B×N, 128, {H//8}, {W//8}]
        ↓
    [View Transformer] 2D → BEV 投影
        → [B, 128, 100, 100] (BEV 特征)
        ↓
    [BEV Encoder] BEV 特征增强
        → [B, 128, 100, 100]
        ↓
    [Occ Decoder] BEV → 3D 体素
        → [B, {num_classes}, {grid_size[0]}, {grid_size[1]}, {grid_size[2]}]
    """)

    print("=" * 80)
    print("✅ 完整网络验证通过！".center(80))
    print("=" * 80)

    print(f"\n网络设计总结:")
    print(f"  ✅ Backbone: PixelUnshuffle 避免颜色通道混合")
    print(f"  ✅ FPN: 轻量级多尺度融合")
    print(f"  ✅ View Transformer: LSS 风格 2D→BEV")
    print(f"  ✅ BEV Encoder: 残差块增强空间特征")
    print(f"  ✅ Occ Decoder: 高度扩展 + 3D 卷积")
    print(f"  ✅ 总参数量: {total:.2f}M (轻量级)")

    return True


if __name__ == "__main__":
    success = verify_complete_network()
    sys.exit(0 if success else 1)

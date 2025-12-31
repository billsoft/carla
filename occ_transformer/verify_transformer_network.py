#!/usr/bin/env python3
# verify_transformer_network.py
"""
Transformer Occupancy Network 完整性验证

验证网络结构、前向传播、参数量
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from models.transformer_occ import (
    build_transformer_occ_net,
    TransformerOccNet,
    TransformerOccNetLite
)


def verify_network(model_type='lite', device='cuda'):
    """验证网络"""
    print("=" * 70)
    print(f"验证 {model_type.upper()} Transformer OccNet".center(70))
    print("=" * 70)
    
    # 配置
    B, N_cam = 1, 8
    H, W = 960, 1280
    num_classes = 18
    output_grid = (200, 200, 16)
    
    print(f"\n配置:")
    print(f"  Batch Size: {B}")
    print(f"  Cameras: {N_cam}")
    print(f"  Image Size: {H}×{W} (Bayer RAW)")
    print(f"  Classes: {num_classes}")
    print(f"  Output Grid: {output_grid}")
    
    # 构建模型
    print(f"\n构建模型...")
    
    if model_type == 'lite':
        model = TransformerOccNetLite(
            num_cameras=N_cam,
            img_size=(H, W),
            patch_size=16,
            embed_dim=256,
            encoder_layers=4,
            decoder_layers=2,
            bev_size=(100, 100),
            output_grid_size=output_grid,
            num_classes=num_classes
        )
    else:
        model = TransformerOccNet(
            num_cameras=N_cam,
            img_size=(H, W),
            patch_size=8,
            embed_dim=256,
            encoder_layers=6,
            decoder_layers=6,
            query_grid_size=(50, 50, 8),
            output_grid_size=output_grid,
            num_classes=num_classes
        )
        
    model = model.to(device)
    
    # 参数统计
    params = model.get_params_summary()
    print(f"\n" + "=" * 70)
    print("参数量统计".center(70))
    print("=" * 70)
    print(f"  {'模块':<25} {'参数量 (M)':>15} {'占比':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    
    total = params['total']
    for name, value in params.items():
        if name != 'total':
            pct = value / total * 100 if total > 0 else 0
            print(f"  {name:<25} {value:>15.2f} {pct:>14.1f}%")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    print(f"  {'Total':<25} {total:>15.2f} {100.0:>14.1f}%")
    
    # 前向传播测试
    print(f"\n" + "=" * 70)
    print("前向传播测试".center(70))
    print("=" * 70)
    
    images = torch.randn(B, N_cam, 1, H, W, device=device)
    print(f"\n输入: {images.shape}")
    
    # 测试
    model.eval()
    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            
        occ_logits = model(images)
        
        if device == 'cuda':
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            
    print(f"输出: {occ_logits.shape}")
    expected = (B, num_classes, *output_grid)
    print(f"预期: {expected}")
    
    # 验证形状
    if occ_logits.shape == expected:
        print(f"\n✅ 输出形状正确!")
    else:
        print(f"\n❌ 输出形状错误!")
        return False
        
    # 输出统计
    print(f"\n输出值统计:")
    print(f"  Min: {occ_logits.min().item():.4f}")
    print(f"  Max: {occ_logits.max().item():.4f}")
    print(f"  Mean: {occ_logits.mean().item():.4f}")
    print(f"  Std: {occ_logits.std().item():.4f}")
    
    # 预测类别
    pred_classes = occ_logits.argmax(dim=1)
    print(f"\n预测:")
    print(f"  Shape: {pred_classes.shape}")
    print(f"  Unique: {pred_classes.unique().tolist()[:10]}...")
    
    # 显存
    if device == 'cuda':
        print(f"\n显存占用: {peak_mem:.2f} GB")
        
    return True


def verify_all():
    """验证所有版本"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}\n")
    
    results = {}
    
    # 验证轻量版
    print("\n" + "#" * 70)
    results['lite'] = verify_network('lite', device)
    
    # 验证标准版 (如果显存足够)
    if device == 'cuda':
        torch.cuda.empty_cache()
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        if free_mem > 8e9:  # 需要约 8GB
            print("\n" + "#" * 70)
            results['standard'] = verify_network('standard', device)
        else:
            print("\n⚠️ 显存不足，跳过标准版测试")
            results['standard'] = None
    
    # 总结
    print("\n" + "=" * 70)
    print("验证总结".center(70))
    print("=" * 70)
    
    for name, result in results.items():
        status = "✅ 通过" if result else ("⏭️ 跳过" if result is None else "❌ 失败")
        print(f"  {name}: {status}")
        
    return all(r is not False for r in results.values())


def print_architecture_summary():
    """打印架构总结"""
    print("\n" + "=" * 70)
    print("Transformer OccNet 架构总结".center(70))
    print("=" * 70)
    
    summary = """
    ┌─────────────────────────────────────────────────────────────────┐
    │              Transformer Occupancy Network                      │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  输入: [B, 8, 1, 960, 1280] - 8相机 Bayer RAW                  │
    │                                                                 │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │ 1. Patch Embedding                                      │   │
    │  │    PixelUnshuffle(2) → Conv → [B, N_patches, D]        │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                           ↓                                     │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │ 2. Position Encoding                                    │   │
    │  │    Spatial PE + Camera PE (射线方向, 相机位置)          │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                           ↓                                     │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │ 3. Transformer Encoder                                  │   │
    │  │    窗口注意力 × L 层                                    │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                           ↓                                     │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │ 4. Voxel Queries                                        │   │
    │  │    可学习查询 + 3D 位置编码                             │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                           ↓                                     │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │ 5. Transformer Decoder                                  │   │
    │  │    Self-Attn + Cross-Attn (可变形) × L 层               │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                           ↓                                     │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │ 6. 3D Upsample + Head                                   │   │
    │  │    上采样 + Conv3D → [B, C, X, Y, Z]                    │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  输出: [B, 18, 200, 200, 16] - 3D 占用网格                    │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    版本对比:
    ┌────────────┬──────────────┬──────────────┐
    │    配置    │   Standard   │     Lite     │
    ├────────────┼──────────────┼──────────────┤
    │ Patch Size │      8       │      16      │
    │ Encoder    │    6 层      │     4 层     │
    │ Decoder    │    6 层      │     2 层     │
    │ Query      │  3D Voxel    │   BEV→3D     │
    │ 参数量     │   ~30M       │    ~15M      │
    │ 显存       │   ~8GB       │    ~4GB      │
    └────────────┴──────────────┴──────────────┘
    """
    print(summary)


if __name__ == '__main__':
    print_architecture_summary()
    success = verify_all()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ 所有验证通过！".center(70))
    else:
        print("❌ 部分验证失败".center(70))
    print("=" * 70)
    
    sys.exit(0 if success else 1)

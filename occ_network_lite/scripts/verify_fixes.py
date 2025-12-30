#!/usr/bin/env python3
"""
验证 occ_network_nano 修复是否成功

检查项:
1. 数据集能否正确加载相机参数
2. 模型能否接收相机参数
3. 完整的前向传播是否成功
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from datasets.carla_occ_dataset import CARLAOccDataset, build_dataloader
from models.occ_network_lite import OccupancyNetworkLite


def check_dataset(data_root: str):
    """检查数据集加载"""
    print("=" * 80)
    print("检查 1: 数据集加载相机参数")
    print("=" * 80)

    try:
        dataset = CARLAOccDataset(
            data_root=data_root,
            split='train',
            img_size=(256, 448),
            grid_size=(100, 100, 8),
            augment=False
        )

        print(f"✅ 数据集初始化成功")
        print(f"   样本数: {len(dataset)}")

        # 加载第一个样本
        sample = dataset[0]

        print(f"\n✅ 样本加载成功")
        print(f"   images: {sample['images'].shape}")
        print(f"   occupancy: {sample['occupancy'].shape}")
        print(f"   mask: {sample['mask'].shape}")
        print(f"   intrinsics: {sample['intrinsics'].shape}")
        print(f"   extrinsics: {sample['extrinsics'].shape}")

        # 验证相机参数形状
        assert sample['intrinsics'].shape == (8, 3, 3), "内参形状错误"
        assert sample['extrinsics'].shape == (8, 4, 4), "外参形状错误"

        print(f"\n✅ 相机参数形状正确")

        # 验证内参矩阵合理性 (焦距应该大于0)
        fx = sample['intrinsics'][0, 0, 0].item()
        fy = sample['intrinsics'][0, 1, 1].item()
        print(f"\n   前主摄内参: fx={fx:.2f}, fy={fy:.2f}")

        assert fx > 0 and fy > 0, "焦距应该大于0"
        print(f"✅ 内参矩阵合理")

        # 验证外参矩阵正交性 (旋转部分)
        R = sample['extrinsics'][0, :3, :3].numpy()
        should_be_eye = R @ R.T
        is_orthogonal = np.allclose(should_be_eye, np.eye(3), atol=1e-5)

        if is_orthogonal:
            print(f"✅ 外参旋转矩阵正交 (R @ R^T ≈ I)")
        else:
            print(f"⚠️  外参旋转矩阵非正交 (误差: {np.max(np.abs(should_be_eye - np.eye(3))):.6f})")

        return True

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_forward(data_root: str):
    """检查模型前向传播"""
    print("\n" + "=" * 80)
    print("检查 2: 模型前向传播")
    print("=" * 80)

    try:
        # 构建数据加载器
        dataloader = build_dataloader(
            data_root=data_root,
            split='train',
            batch_size=1,
            num_workers=0,
            img_size=(256, 448),
            grid_size=(100, 100, 8),
            augment=False
        )

        print(f"✅ 数据加载器构建成功")

        # 构建轻量级模型
        model = OccupancyNetworkLite(
            num_cameras=8,
            img_size=(256, 448),
            embed_dim=128,
            bev_h=100,
            bev_w=100,
            num_classes=18,
            num_heights=8,
        )

        print(f"✅ 模型构建成功")

        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   参数量: {total_params/1e6:.2f}M")

        # 获取一个 batch
        batch = next(iter(dataloader))

        print(f"\n✅ Batch 加载成功")
        print(f"   images: {batch['images'].shape}")
        print(f"   extrinsics: {batch['extrinsics'].shape}")

        # 前向传播 (不传递 extrinsics, 使用模型内部默认值)
        model.eval()
        with torch.no_grad():
            outputs = model(batch['images'])

        print(f"\n✅ 前向传播成功 (未传递 extrinsics)")
        print(f"   occ_logits: {outputs['occ_logits'].shape}")

        # 前向传播 (传递 extrinsics)
        with torch.no_grad():
            outputs = model(batch['images'], extrinsics=batch['extrinsics'])

        print(f"\n✅ 前向传播成功 (传递 extrinsics)")
        print(f"   occ_logits: {outputs['occ_logits'].shape}")

        # 验证输出形状
        expected_shape = (1, 18, 100, 100, 8)
        assert outputs['occ_logits'].shape == expected_shape, f"输出形状错误: {outputs['occ_logits'].shape}"

        print(f"✅ 输出形状正确: {expected_shape}")

        return True

    except Exception as e:
        print(f"❌ 模型前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 数据集路径
    data_root = Path(__file__).parent.parent.parent / 'dataset_output'

    if not data_root.exists():
        print(f"❌ 数据集不存在: {data_root}")
        print(f"\n请先运行数据采集:")
        print(f"  python dense_occupancy_collection/main_data_collection.py --frames 100")
        return

    # 检查 camera_params 目录
    camera_params_dir = data_root / 'camera_params'
    if not camera_params_dir.exists():
        print(f"❌ camera_params/ 目录不存在: {camera_params_dir}")
        print(f"\n这是旧数据集，必须重新采集:")
        print(f"  python dense_occupancy_collection/main_data_collection.py --frames 100")
        return

    print(f"数据集路径: {data_root}")
    print(f"camera_params 目录: {camera_params_dir}")
    print(f"参数文件数: {len(list(camera_params_dir.glob('*.npz')))}\n")

    # 检查数据集
    dataset_ok = check_dataset(str(data_root))

    # 检查模型
    model_ok = check_model_forward(str(data_root))

    # 总结
    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)

    if dataset_ok and model_ok:
        print("✅ 所有检查通过! occ_network_nano 修复成功")
        print("\n可以开始训练:")
        print("  cd occ_network_nano")
        print("  python train_lite.py --data_root ../dataset_output --epochs 10 --batch_size 2 --amp")
    else:
        print("❌ 部分检查失败，请查看上面的错误信息")


if __name__ == '__main__':
    main()

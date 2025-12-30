#!/usr/bin/env python3
"""
全面诊断 Class 0 预测问题
"""

import numpy as np
import torch
from pathlib import Path
import sys

def check_dataset_class0():
    """检查数据集中 class 0 的分布"""
    print("=" * 80)
    print("1. 数据集 Class 0 检查")
    print("=" * 80)

    dataset_path = Path(r'd:\code\carla\dataset_10k\occupancy')
    files = list(dataset_path.glob('*.npz'))[:20]

    class_counts = {}

    for file_path in files:
        data = np.load(file_path)
        occupancy = data['occupancy']
        mask = data['mask']

        # 只统计有效区域
        valid_occ = occupancy[mask]
        unique, counts = np.unique(valid_occ, return_counts=True)

        for cls, cnt in zip(unique, counts):
            class_counts[int(cls)] = class_counts.get(int(cls), 0) + int(cnt)

    total = sum(class_counts.values())

    print(f"\n前 {len(files)} 个文件的类别分布:")
    for cls in sorted(class_counts.keys())[:5]:  # 只显示前5个类
        count = class_counts[cls]
        pct = count / total * 100
        print(f"  Class {cls}: {count:,} ({pct:.2f}%)")

    has_class0 = 0 in class_counts
    print(f"\n✅ Class 0 在数据集中存在: {has_class0}")

    if has_class0:
        class0_pct = class_counts[0] / total * 100
        print(f"   Class 0 占比: {class0_pct:.2f}%")

    return has_class0

def check_loss_function():
    """检查损失函数配置"""
    print("\n" + "=" * 80)
    print("2. 损失函数配置检查")
    print("=" * 80)

    sys.path.insert(0, str(Path(__file__).parent / 'occ_network_nano'))
    from utils.loss import get_default_class_weights

    weights = get_default_class_weights()

    print(f"\nClass 权重配置:")
    for i in range(min(5, len(weights))):
        print(f"  Class {i}: weight={weights[i]}")

    class0_weight = weights[0]
    print(f"\n⚠️ Class 0 权重: {class0_weight}")

    if class0_weight < 0.5:
        print(f"   警告: Class 0 权重过低 ({class0_weight}), 可能导致网络忽略此类!")
        print(f"   建议: 提高到至少 0.5 或 1.0")
        return False
    else:
        print(f"   ✅ Class 0 权重合理")
        return True

def check_model_output():
    """检查模型输出层配置"""
    print("\n" + "=" * 80)
    print("3. 模型输出层检查")
    print("=" * 80)

    sys.path.insert(0, str(Path(__file__).parent / 'occ_network_nano'))
    from models import build_bayer_occ_net

    model = build_bayer_occ_net(
        num_classes=18,
        grid_size=(200, 200, 16),
        img_size=(384, 640),
        backbone_width_mult=1.0,
        fpn_channels=128,
        bev_size=(100, 100),
        hidden_channels=64
    )

    # 检查最后一层
    print(f"\n检查 Occ Decoder 输出层:")
    print(f"  网络结构: {type(model.occ_decoder).__name__}")

    # 查找分类头
    if hasattr(model.occ_decoder, 'cls_head'):
        cls_head = model.occ_decoder.cls_head
        print(f"  分类头: {cls_head}")

        # 检查输出通道数
        if hasattr(cls_head, 'out_channels'):
            print(f"  输出通道数: {cls_head.out_channels}")

        # 检查是否有 bias
        if hasattr(cls_head, 'bias') and cls_head.bias is not None:
            bias = cls_head.bias.data
            print(f"  Bias 统计:")
            print(f"    Shape: {bias.shape}")
            print(f"    Class 0 bias: {bias[0].item():.4f}")
            print(f"    Mean bias: {bias.mean().item():.4f}")
            print(f"    Std bias: {bias.std().item():.4f}")

            # 检查 bias 是否异常
            if bias[0] < bias.mean() - 2 * bias.std():
                print(f"  ⚠️ Class 0 的 bias 显著低于其他类!")
                return False

    # 创建虚拟输入测试
    print(f"\n测试模型输出分布:")
    images = torch.randn(1, 8, 1, 384, 640)

    with torch.no_grad():
        outputs = model(images)  # [1, 18, 200, 200, 16]

    # 统计每个类别的平均 logit
    mean_logits = outputs.mean(dim=[0, 2, 3, 4])  # [18]

    print(f"  各类别平均 logit (随机初始化):")
    for i in range(min(5, len(mean_logits))):
        print(f"    Class {i}: {mean_logits[i].item():.4f}")

    # 检查 class 0 是否异常低
    class0_logit = mean_logits[0].item()
    mean_all = mean_logits.mean().item()
    std_all = mean_logits.std().item()

    print(f"\n  统计:")
    print(f"    Mean logit: {mean_all:.4f}")
    print(f"    Std logit: {std_all:.4f}")
    print(f"    Class 0 logit: {class0_logit:.4f}")

    if class0_logit < mean_all - std_all:
        print(f"  ⚠️ Class 0 logit 偏低!")
        return False
    else:
        print(f"  ✅ Logit 分布正常")
        return True

def check_inference_results():
    """检查推理结果"""
    print("\n" + "=" * 80)
    print("4. 推理结果检查")
    print("=" * 80)

    inf_path = Path(r'd:\code\carla\inference_results')

    if not inf_path.exists():
        print(f"  ⚠️ 推理结果目录不存在: {inf_path}")
        return False

    files = list(inf_path.glob('*.npz'))[:10]

    if len(files) == 0:
        print(f"  ⚠️ 没有找到推理结果文件")
        return False

    print(f"\n检查前 {len(files)} 个推理结果:")

    has_class0 = False

    for file_path in files:
        data = np.load(file_path)
        occupancy = data['occupancy']

        unique = np.unique(occupancy)

        if 0 in unique:
            has_class0 = True
            break

    print(f"  推理结果中包含 Class 0: {has_class0}")

    if not has_class0:
        print(f"  ❌ 所有推理结果都没有 Class 0!")

        # 显示第一个文件的类别分布
        first_file = files[0]
        data = np.load(first_file)
        occupancy = data['occupancy']
        unique, counts = np.unique(occupancy, return_counts=True)

        print(f"\n  第一个文件 ({first_file.name}) 的类别:")
        for cls, cnt in zip(unique, counts):
            pct = cnt / occupancy.size * 100
            print(f"    Class {cls}: {cnt:,} ({pct:.1f}%)")

    return has_class0

def main():
    print("=" * 80)
    print("Class 0 预测问题全面诊断".center(80))
    print("=" * 80)

    results = {}

    # 1. 检查数据集
    results['dataset_has_class0'] = check_dataset_class0()

    # 2. 检查损失函数
    results['loss_weight_ok'] = check_loss_function()

    # 3. 检查模型
    results['model_output_ok'] = check_model_output()

    # 4. 检查推理结果
    results['inference_has_class0'] = check_inference_results()

    # 总结
    print("\n" + "=" * 80)
    print("诊断总结".center(80))
    print("=" * 80)

    for key, value in results.items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}: {value}")

    print("\n" + "=" * 80)

    # 给出建议
    if not results['loss_weight_ok']:
        print("\n🔧 建议修复:")
        print("  1. 提高 Class 0 权重从 0.1 到至少 1.0")
        print("     修改文件: occ_network_nano/utils/loss.py:94")
        print("     改为: 1.0,  # 0: Free/Unlabeled")
        print("  2. 重新训练模型")

    if results['dataset_has_class0'] and not results['inference_has_class0']:
        print("\n⚠️ 问题确认:")
        print("  数据集有 Class 0, 但推理结果没有 Class 0")
        print("  这说明网络训练有问题,最可能的原因:")
        print("    - Class 0 权重过低 (0.1)")
        print("    - 训练时 Class 0 被抑制")

if __name__ == "__main__":
    main()

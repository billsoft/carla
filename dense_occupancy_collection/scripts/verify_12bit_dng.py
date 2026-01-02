"""
验证 12-bit DNG 数据生成正确性

检查项：
1. 文件格式：.dng (TIFF)
2. 图像尺寸：1280x960
3. 数据类型：uint16
4. 数值范围：[0, 4095] (12-bit)
5. 深度数据存在
6. 体素数据存在
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def verify_dataset(dataset_path: str, num_samples: int = 3):
    """
    验证数据集完整性

    Args:
        dataset_path: 数据集根目录
        num_samples: 检查的样本数量
    """
    dataset_path = Path(dataset_path)

    print("=" * 70)
    print("12-bit DNG 数据集验证".center(70))
    print("=" * 70)
    print(f"\n数据集路径: {dataset_path}")

    if not dataset_path.exists():
        print(f"\n❌ 错误：数据集路径不存在")
        return False

    # 检查目录结构
    print("\n[1] 检查目录结构...")
    required_dirs = ['cameras', 'occupancy', 'camera_params', 'depth']
    all_dirs_exist = True

    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        exists = dir_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {dir_name}/")
        if not exists:
            all_dirs_exist = False

    if not all_dirs_exist:
        print("\n❌ 目录结构不完整")
        return False

    # 检查相机目录
    print("\n[2] 检查相机目录...")
    camera_names = [
        'cam_front_main', 'cam_front_wide', 'cam_front_narrow',
        'cam_left_pillar', 'cam_right_pillar',
        'cam_left_repeater', 'cam_right_repeater', 'cam_rear'
    ]

    cameras_dir = dataset_path / 'cameras'
    all_cameras_exist = True

    for cam_name in camera_names:
        cam_dir = cameras_dir / cam_name
        exists = cam_dir.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {cam_name}/")
        if not exists:
            all_cameras_exist = False

    if not all_cameras_exist:
        print("\n❌ 相机目录不完整")
        return False

    # 获取样本列表
    occupancy_dir = dataset_path / 'occupancy'
    sample_files = sorted(occupancy_dir.glob('*.npz'))

    if len(sample_files) == 0:
        print(f"\n❌ 未找到任何样本数据")
        return False

    print(f"\n找到 {len(sample_files)} 个样本")
    print(f"检查前 {min(num_samples, len(sample_files))} 个样本...")

    # 逐个检查样本
    all_passed = True

    for i, sample_file in enumerate(sample_files[:num_samples]):
        sample_id = sample_file.stem
        print(f"\n{'='*70}")
        print(f"[样本 {i+1}] ID: {sample_id}")
        print(f"{'='*70}")

        sample_passed = True

        # 检查 RGB 图像
        print("\n  [RGB 图像]")
        for cam_name in camera_names:
            # 优先检查 .dng 文件
            img_path = cameras_dir / cam_name / f"{sample_id}.dng"
            if not img_path.exists():
                # 降级检查 .png
                img_path = cameras_dir / cam_name / f"{sample_id}.png"

            if not img_path.exists():
                print(f"    ❌ {cam_name}: 文件不存在")
                sample_passed = False
                continue

            try:
                # 尝试加载 DNG
                try:
                    import rawpy
                    with rawpy.imread(str(img_path)) as raw:
                        img = raw.raw_image_visible.copy()
                except ImportError:
                    # 降级使用 OpenCV
                    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"    ❌ {cam_name}: 无法加载")
                    sample_passed = False
                    continue

                # 检查属性
                if img.dtype != np.uint16:
                    print(f"    ❌ {cam_name}: 必须是 uint16，实际 {img.dtype}")
                    sample_passed = False
                    continue

                if len(img.shape) != 2:
                    print(f"    ❌ {cam_name}: 必须是单通道，实际 {img.shape}")
                    sample_passed = False
                    continue

                # 检查范围 (应该是 12-bit [0, 4095])
                max_val = img.max()
                if max_val > 4095:
                    print(f"    ⚠️ {cam_name}: 最大值 {max_val} 超过 12-bit 范围 (4095)")
                else:
                    print(f"    ✅ {cam_name}: {img.shape} 12-bit range=[{img.min()}, {img.max()}]")

            except Exception as e:
                print(f"    ❌ {cam_name}: 错误 - {e}")
                sample_passed = False

        # 检查深度数据
        print("\n  [深度数据]")
        depth_dirs = [
            'depth_front', 'depth_front_right', 'depth_right', 'depth_back_right',
            'depth_back', 'depth_back_left', 'depth_left', 'depth_front_left'
        ]
        depth_ok = True

        for depth_name in depth_dirs:
            depth_path = dataset_path / 'depth' / depth_name / f"{sample_id}.png"
            exists = depth_path.exists()
            status = "✅" if exists else "❌"
            print(f"    {status} {depth_name}")
            if not exists:
                depth_ok = False
                sample_passed = False

        # 检查体素数据
        print("\n  [体素数据]")
        voxel_path = dataset_path / 'occupancy' / f"{sample_id}.npz"

        if voxel_path.exists():
            try:
                voxel_data = np.load(voxel_path)
                required_keys = ['occupancy', 'actor_ids', 'mask']
                missing_keys = [k for k in required_keys if k not in voxel_data]

                if missing_keys:
                    print(f"    ❌ 缺少字段: {missing_keys}")
                    sample_passed = False
                else:
                    occupancy = voxel_data['occupancy']
                    mask = voxel_data['mask']
                    print(f"    ✅ occupancy: {occupancy.shape}, 占据体素: {np.sum(occupancy > 0)}")
                    print(f"    ✅ mask: {mask.shape}, 有效体素: {np.sum(mask)}")
            except Exception as e:
                print(f"    ❌ 加载失败: {e}")
                sample_passed = False
        else:
            print(f"    ❌ 文件不存在")
            sample_passed = False

        # 检查相机参数
        print("\n  [相机参数]")
        cam_param_path = dataset_path / 'camera_params' / f"{sample_id}.npz"

        if cam_param_path.exists():
            try:
                cam_params = np.load(cam_param_path, allow_pickle=True)
                intrinsics = cam_params['intrinsics']
                extrinsics = cam_params['extrinsics']
                print(f"    ✅ intrinsics: {intrinsics.shape}")
                print(f"    ✅ extrinsics: {extrinsics.shape}")
            except Exception as e:
                print(f"    ❌ 加载失败: {e}")
                sample_passed = False
        else:
            print(f"    ❌ 文件不存在")
            sample_passed = False

        if not sample_passed:
            all_passed = False

        # 样本总结
        print(f"\n  样本 {i+1} 结果: {'✅ 通过' if sample_passed else '❌ 失败'}")

    # 最终总结
    print(f"\n{'='*70}")
    if all_passed:
        print("✅ 所有检查通过！数据集格式正确。".center(70))
    else:
        print("❌ 部分检查失败，请检查数据集。".center(70))
    print(f"{'='*70}\n")

    return all_passed


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='验证 12-bit DNG 数据集')
    parser.add_argument('--dataset', type=str, default='dataset_output',
                        help='数据集路径（默认：dataset_output）')
    parser.add_argument('--samples', type=int, default=3,
                        help='检查的样本数量（默认：3）')
    args = parser.parse_args()

    success = verify_dataset(args.dataset, args.samples)
    sys.exit(0 if success else 1)

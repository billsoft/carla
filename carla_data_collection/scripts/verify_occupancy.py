#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 Occupancy 体素数据格式和内容
"""

import numpy as np
from pathlib import Path
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from config.occupancy_config import OCCUPANCY_CLASS_NAMES
from config.camera_config import TESLA_CAMERA_CONFIGS

def verify_occupancy_file(npz_path):
    """
    验证单个 Occupancy 文件
    """
    print(f"\n{'='*80}")
    print(f"验证文件: {npz_path}")
    print(f"{'='*80}\n")

    # 加载数据
    data = np.load(npz_path)

    # 检查必需的字段
    required_keys = ['occupancy', 'mask', 'x_range', 'y_range', 'z_range', 'resolution', 'grid_size']
    print("1. 检查数据字段:")
    for key in required_keys:
        if key in data:
            print(f"   ✓ {key}")
        else:
            print(f"   ✗ {key} - 缺失!")
            return False

    # 获取数据
    occupancy = data['occupancy']
    mask = data['mask']
    x_range = data['x_range']
    y_range = data['y_range']
    z_range = data['z_range']
    resolution = data['resolution']
    grid_size = data['grid_size']

    # 验证形状
    print(f"\n2. 验证数据形状:")
    print(f"   Occupancy 形状: {occupancy.shape}")
    print(f"   期望网格尺寸: {grid_size}")

    if tuple(occupancy.shape) == tuple(grid_size):
        print(f"   ✓ 形状匹配")
    else:
        print(f"   ✗ 形状不匹配!")
        return False

    if occupancy.shape == mask.shape:
        print(f"   ✓ Mask 形状匹配")
    else:
        print(f"   ✗ Mask 形状不匹配!")
        return False

    # 验证空间范围
    print(f"\n3. 验证空间配置:")
    print(f"   X 范围: [{x_range[0]}, {x_range[1]}] 米")
    print(f"   Y 范围: [{y_range[0]}, {y_range[1]}] 米")
    print(f"   Z 范围: [{z_range[0]}, {z_range[1]}] 米")
    print(f"   分辨率: {resolution} 米/体素")

    # 计算理论网格尺寸
    expected_grid_x = int((x_range[1] - x_range[0]) / resolution)
    expected_grid_y = int((y_range[1] - y_range[0]) / resolution)
    expected_grid_z = int((z_range[1] - z_range[0]) / resolution)

    if (expected_grid_x == grid_size[0] and
        expected_grid_y == grid_size[1] and
        expected_grid_z == grid_size[2]):
        print(f"   ✓ 网格尺寸计算正确")
    else:
        print(f"   ✗ 网格尺寸计算错误!")
        print(f"   期望: ({expected_grid_x}, {expected_grid_y}, {expected_grid_z})")
        print(f"   实际: {tuple(grid_size)}")
        return False

    # 统计标签分布
    print(f"\n4. 体素统计:")
    print(f"   总体素数: {occupancy.size:,}")
    print(f"   有效观测: {np.sum(mask):,} ({np.sum(mask)/occupancy.size*100:.2f}%)")
    print(f"   非空体素: {np.sum(occupancy > 0):,} ({np.sum(occupancy > 0)/occupancy.size*100:.2f}%)")

    # 类别分布
    print(f"\n5. 语义类别分布:")
    unique_labels = np.unique(occupancy)
    print(f"   检测到 {len(unique_labels)} 个不同类别")

    for label in sorted(unique_labels):
        count = np.sum(occupancy == label)
        percentage = count / occupancy.size * 100
        class_name = OCCUPANCY_CLASS_NAMES[label] if label < len(OCCUPANCY_CLASS_NAMES) else "Unknown"
        print(f"   [{label:2d}] {class_name:15s}: {count:8,} ({percentage:5.2f}%)")

    # 验证数据类型
    print(f"\n6. 数据类型:")
    print(f"   Occupancy dtype: {occupancy.dtype} (期望: uint8)")
    print(f"   Mask dtype: {mask.dtype} (期望: bool)")

    if occupancy.dtype == np.uint8:
        print(f"   ✓ Occupancy 类型正确")
    else:
        print(f"   ⚠ Occupancy 类型异常")

    if mask.dtype == bool:
        print(f"   ✓ Mask 类型正确")
    else:
        print(f"   ⚠ Mask 类型异常")

    # 验证 Mask 和 Occupancy 的一致性
    print(f"\n7. 数据一致性:")
    # 检查: Mask为False的地方, Occupancy是否为0
    invalid_masked_voxels = np.sum((~mask) & (occupancy > 0))
    if invalid_masked_voxels == 0:
        print(f"   ✓ Mask 与 Occupancy 一致 (未观测区域值均为 0)")
    else:
        print(f"   ✗ 发现 {invalid_masked_voxels} 个未观测体素有非零值!")
        return False

    # 检查数据范围
    print(f"\n8. 数据范围:")
    print(f"   Occupancy 值范围: [{occupancy.min()}, {occupancy.max()}]")
    print(f"   期望范围: [0, {len(OCCUPANCY_CLASS_NAMES)-1}] (共 {len(OCCUPANCY_CLASS_NAMES)} 类)")

    if occupancy.max() < len(OCCUPANCY_CLASS_NAMES):
        print(f"   ✓ 标签值合法")
    else:
        print(f"   ✗ 存在非法标签值!")
        return False

    print(f"\n{'='*80}")
    print(f"✓ 验证通过!")
    print(f"{'='*80}\n")

    return True


def verify_cameras(dataset_dir, num_frames):
    """
    验证相机数据
    """
    cameras_dir = Path(dataset_dir) / "cameras"
    print(f"\n{'='*80}")
    print(f"验证相机数据")
    print(f"{'='*80}\n")

    if not cameras_dir.exists():
        print(f"✗ 相机目录不存在: {cameras_dir}")
        return False

    all_cameras_valid = True
    
    # 检查每个相机的文件夹
    for cam_config in TESLA_CAMERA_CONFIGS:
        cam_id = cam_config['id']
        cam_dir = cameras_dir / cam_id
        
        if not cam_dir.exists():
            print(f"✗ 缺失相机目录: {cam_id}")
            all_cameras_valid = False
            continue

        # 检查图片数量
        images = sorted(list(cam_dir.glob("*.png")) + list(cam_dir.glob("*.jpg")))
        if len(images) == num_frames:
            print(f"   ✓ {cam_id:<20}: {len(images)} 帧 (匹配)")
        else:
            print(f"   ✗ {cam_id:<20}: {len(images)} 帧 (期望 {num_frames})")
            all_cameras_valid = False

    return all_cameras_valid


def verify_calibration(dataset_dir):
    """
    验证标定文件
    """
    calib_path = Path(dataset_dir) / "calibration.json"
    print(f"\n{'='*80}")
    print(f"验证标定文件")
    print(f"{'='*80}\n")

    if not calib_path.exists():
        print(f"✗ 标定文件不存在: {calib_path}")
        return False

    try:
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        
        # 检查相机数量
        if 'cameras' not in calib_data:
             print(f"✗ 标定文件缺少 'cameras' 字段")
             return False
             
        calib_cams = calib_data['cameras']
        expected_cams = len(TESLA_CAMERA_CONFIGS)
        
        if len(calib_cams) == expected_cams:
            print(f"   ✓ 包含 {len(calib_cams)} 个相机的标定信息")
        else:
             print(f"   ⚠ 相机数量不匹配: {len(calib_cams)} (期望 {expected_cams})")
        
        # 检查内参和外参
        keys_valid = True
        for cam_id, cam_data in calib_cams.items():
            if 'intrinsics' not in cam_data or 'extrinsics' not in cam_data:
                print(f"   ✗ 相机 {cam_id} 缺少内参或外参")
                keys_valid = False
        
        if keys_valid:
             print(f"   ✓ 所有相机均包含内参和外参")
             
        return True

    except Exception as e:
        print(f"✗ 读取标定文件失败: {e}")
        return False


def verify_dataset(dataset_dir):
    """
    验证完整数据集
    """
    dataset_path = Path(dataset_dir)
    occupancy_dir = dataset_path / "occupancy"

    print(f"\n{'='*80}")
    print(f"验证数据集: {dataset_path}")
    print(f"{'='*80}\n")

    if not occupancy_dir.exists():
        print(f"✗ Occupancy 目录不存在: {occupancy_dir}")
        return False

    # 获取所有 .npz 文件
    npz_files = sorted(occupancy_dir.glob("*.npz"))

    if len(npz_files) == 0:
        print(f"✗ 未找到 .npz 文件")
        return False

    print(f"找到 {len(npz_files)} 个 Occupancy 文件\n")
    
    num_frames = len(npz_files)

    # 1. 验证相机数据
    cameras_valid = verify_cameras(dataset_dir, num_frames)
    
    # 2. 验证标定文件
    calib_valid = verify_calibration(dataset_dir)

    # 3. 验证每个 Occupancy 文件
    print(f"\n{'='*80}")
    print(f"验证 Occupancy 文件")
    print(f"{'='*80}")
    
    occupancy_valid = True
    for i, npz_file in enumerate(npz_files):
        print(f"\n[{i+1}/{len(npz_files)}] {npz_file.name}")
        if not verify_occupancy_file(npz_file):
            occupancy_valid = False
            print(f"   ✗ 验证失败!")
        else:
            print(f"   ✓ 验证成功")

    # 总结
    all_valid = cameras_valid and calib_valid and occupancy_valid

    print(f"\n{'='*80}")
    print(f"验证总结")
    print(f"{'='*80}")
    
    print(f"相机数据: {'✓ 通过' if cameras_valid else '✗ 失败'}")
    print(f"标定文件: {'✓ 通过' if calib_valid else '✗ 失败'}")
    print(f"体素数据: {'✓ 通过' if occupancy_valid else '✗ 失败'}")
    
    if all_valid:
        print(f"\n✓ 所有验证通过! 数据集完整。")
    else:
        print(f"\n✗ 数据集验证失败,请检查上述错误。")
    print(f"{'='*80}\n")

    return all_valid


if __name__ == '__main__':
    # 默认验证路径
    dataset_dir = Path(__file__).parent.parent.parent / "dataset_output" / "town10_test"

    if len(sys.argv) > 1:
        dataset_dir = Path(sys.argv[1])
    else:
        # 默认路径
        default_path = Path("d:/code/carla/dataset_output/town10_test")
        if default_path.exists():
            dataset_dir = default_path
        else:
            # 回退到相对路径
            dataset_dir = Path(__file__).parent.parent.parent / "dataset_output" / "town10_test"

    verify_dataset(dataset_dir)

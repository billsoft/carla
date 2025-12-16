"""
数据集验证脚本
检查采集的 HDF5 数据集的结构、内容和数量
"""

import sys
from pathlib import Path
import h5py
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def validate_dataset(h5_path):
    """
    验证 HDF5 数据集

    检查:
    1. 文件结构
    2. 数据类型和形状
    3. 数据范围和统计
    4. 数据完整性
    """
    print("="*80)
    print(f"数据集验证: {h5_path}")
    print("="*80)
    print()

    if not Path(h5_path).exists():
        print(f"❌ 错误: 文件不存在 {h5_path}")
        return False

    try:
        with h5py.File(h5_path, 'r') as f:
            # ========== 1. 基本信息 ==========
            print("【1】 基本信息")
            print("-" * 80)

            # 获取实际帧数
            num_frames = f['images'].shape[0]
            print(f"  总帧数: {num_frames}")

            file_size_mb = Path(h5_path).stat().st_size / (1024 * 1024)
            print(f"  文件大小: {file_size_mb:.2f} MB")

            # Occupancy 配置
            print(f"\n  Occupancy 配置:")
            print(f"    X 范围: {f.attrs['occupancy_x_range']} 米")
            print(f"    Y 范围: {f.attrs['occupancy_y_range']} 米")
            print(f"    Z 范围: {f.attrs['occupancy_z_range']} 米")
            print(f"    分辨率: {f.attrs['occupancy_resolution']} 米/体素")

            # ========== 2. 数据结构检查 ==========
            print(f"\n【2】 数据结构")
            print("-" * 80)

            required_datasets = {
                'images': (num_frames, 8, 960, 1280, 3),
                'occupancy': (num_frames, 200, 200, 16),
                'occupancy_mask': (num_frames, 200, 200, 16),
                'timestamps': (num_frames,),
                'frame_ids': (num_frames,),
                'vehicle_location': (num_frames, 3),
                'vehicle_rotation': (num_frames, 3),
                'vehicle_velocity': (num_frames, 3),
                'camera_intrinsics': (8, 3, 3),
                'camera_extrinsics': (8, 4, 4),
            }

            all_valid = True

            for name, expected_shape in required_datasets.items():
                if name not in f:
                    print(f"  ❌ 缺失数据集: {name}")
                    all_valid = False
                    continue

                dataset = f[name]
                actual_shape = dataset.shape
                dtype = dataset.dtype

                # 检查形状
                if actual_shape == expected_shape:
                    status = "✓"
                else:
                    status = "❌"
                    all_valid = False

                print(f"  {status} {name:<25} {str(actual_shape):<30} {dtype}")

            # ========== 3. 图像数据统计 ==========
            print(f"\n【3】 图像数据统计 (8 相机 RGB)")
            print("-" * 80)

            images = f['images']
            print(f"  形状: {images.shape}  (帧, 相机, H, W, C)")
            print(f"  数据类型: {images.dtype} (12-bit 存储为 uint16)")

            # 随机采样一帧检查
            if num_frames > 0:
                sample_idx = min(num_frames // 2, num_frames - 1)
                sample_image = images[sample_idx, 0]  # 第一个相机

                print(f"\n  采样检查 (帧 {sample_idx}, 相机 0):")
                print(f"    最小值: {sample_image.min()}")
                print(f"    最大值: {sample_image.max()}")
                print(f"    平均值: {sample_image.mean():.2f}")
                print(f"    标准差: {sample_image.std():.2f}")

                # 12-bit 应该在 [0, 4095] 范围
                if sample_image.max() <= 4095:
                    print(f"    ✓ 12-bit 范围正确")
                else:
                    print(f"    ❌ 警告: 超出 12-bit 范围")

            # ========== 4. Occupancy 数据统计 ==========
            print(f"\n【4】 Occupancy 数据统计")
            print("-" * 80)

            occupancy = f['occupancy']
            occupancy_mask = f['occupancy_mask']

            print(f"  Occupancy 形状: {occupancy.shape}  (帧, X, Y, Z)")
            print(f"  数据类型: {occupancy.dtype}")

            if num_frames > 0:
                sample_idx = min(num_frames // 2, num_frames - 1)
                sample_occ = occupancy[sample_idx]
                sample_mask = occupancy_mask[sample_idx]

                # 统计非空体素
                non_empty = np.sum(sample_occ > 0)
                total_voxels = np.prod(sample_occ.shape)
                occupancy_rate = 100.0 * non_empty / total_voxels

                print(f"\n  采样检查 (帧 {sample_idx}):")
                print(f"    总体素数: {total_voxels:,}")
                print(f"    非空体素: {non_empty:,}  ({occupancy_rate:.2f}%)")
                print(f"    有效观测: {np.sum(sample_mask):,}  ({100.0*np.sum(sample_mask)/total_voxels:.2f}%)")

                # 类别分布
                unique, counts = np.unique(sample_occ, return_counts=True)
                print(f"\n    类别分布:")
                for label, count in zip(unique, counts):
                    if label > 0:  # 跳过空类别
                        pct = 100.0 * count / total_voxels
                        print(f"      类别 {label:2d}: {count:8,} 体素 ({pct:5.2f}%)")

            # ========== 5. 车辆位姿统计 ==========
            print(f"\n【5】 车辆位姿统计")
            print("-" * 80)

            locations = f['vehicle_location'][:]
            velocities = f['vehicle_velocity'][:]

            print(f"  位置范围:")
            print(f"    X: {locations[:, 0].min():.2f} ~ {locations[:, 0].max():.2f} 米")
            print(f"    Y: {locations[:, 1].min():.2f} ~ {locations[:, 1].max():.2f} 米")
            print(f"    Z: {locations[:, 2].min():.2f} ~ {locations[:, 2].max():.2f} 米")

            speeds = np.linalg.norm(velocities, axis=1)  # m/s
            speeds_kmh = speeds * 3.6  # km/h

            print(f"\n  速度统计:")
            print(f"    平均: {speeds_kmh.mean():.2f} km/h")
            print(f"    最大: {speeds_kmh.max():.2f} km/h")
            print(f"    最小: {speeds_kmh.min():.2f} km/h")

            # ========== 6. 相机标定 ==========
            print(f"\n【6】 相机标定")
            print("-" * 80)

            camera_names = [name.decode() for name in f['camera_names'][:]]
            intrinsics = f['camera_intrinsics'][:]
            extrinsics = f['camera_extrinsics'][:]

            print(f"  相机数量: {len(camera_names)}")
            print(f"  相机列表: {', '.join(camera_names)}")

            print(f"\n  示例内参矩阵 (相机 0):")
            print(f"    {intrinsics[0]}")

            print(f"\n  示例外参矩阵 (相机 0):")
            print(f"    {extrinsics[0]}")

            # ========== 7. 时间戳连续性 ==========
            print(f"\n【7】 时间戳连续性")
            print("-" * 80)

            timestamps = f['timestamps'][:]
            frame_ids = f['frame_ids'][:]

            if len(timestamps) > 1:
                time_diffs = np.diff(timestamps)
                print(f"  帧间隔统计:")
                print(f"    平均: {time_diffs.mean()*1000:.2f} ms")
                print(f"    最大: {time_diffs.max()*1000:.2f} ms")
                print(f"    最小: {time_diffs.min()*1000:.2f} ms")
                print(f"    标准差: {time_diffs.std()*1000:.2f} ms")

                # 检查是否有跳帧
                frame_diffs = np.diff(frame_ids)
                skipped_frames = np.sum(frame_diffs > 1)
                if skipped_frames > 0:
                    print(f"    ⚠️  检测到 {skipped_frames} 次跳帧")
                else:
                    print(f"    ✓ 无跳帧")

            # ========== 最终结果 ==========
            print("\n" + "="*80)
            if all_valid:
                print("✓ 数据集验证通过!")
            else:
                print("❌ 数据集存在问题,请检查上述错误")
            print("="*80)

            return all_valid

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='验证 CARLA Occupancy 数据集')
    parser.add_argument('h5_file', type=str, help='HDF5 数据集文件路径')
    args = parser.parse_args()

    validate_dataset(args.h5_file)


if __name__ == '__main__':
    # 如果没有提供参数,查找最新的数据集文件
    if len(sys.argv) == 1:
        data_dir = Path('data/collected')
        if data_dir.exists():
            h5_files = sorted(data_dir.glob('*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
            if h5_files:
                print(f"未指定文件,使用最新数据集: {h5_files[0]}\n")
                validate_dataset(str(h5_files[0]))
            else:
                print("错误: data/collected/ 目录下没有找到 .h5 文件")
                print("用法: python validate_dataset.py <path_to_h5_file>")
        else:
            print("错误: data/collected/ 目录不存在")
            print("用法: python validate_dataset.py <path_to_h5_file>")
    else:
        main()

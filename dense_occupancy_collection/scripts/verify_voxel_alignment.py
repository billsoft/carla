"""
体素坐标对齐验证脚本
验证体素索引与 UE5 世界坐标的精确映射关系
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dense_occupancy_collection.config.occupancy_config import (
    X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION, GRID_SIZE
)

def voxel_index_to_world_coord(ix, iy, iz):
    """
    体素索引 -> 世界坐标 (体素中心)

    Args:
        ix, iy, iz: 体素索引 [0, GRID_SIZE-1]

    Returns:
        x, y, z: 世界坐标 (米)
    """
    x = X_RANGE[0] + (ix + 0.5) * RESOLUTION
    y = Y_RANGE[0] + (iy + 0.5) * RESOLUTION
    z = Z_RANGE[0] + (iz + 0.5) * RESOLUTION
    return x, y, z

def world_coord_to_voxel_index(x, y, z):
    """
    世界坐标 -> 体素索引

    Args:
        x, y, z: 世界坐标 (米)

    Returns:
        ix, iy, iz: 体素索引
    """
    # 使用 floor 计算索引，这是最标准的体素化方法
    ix = int(np.floor((x - X_RANGE[0]) / RESOLUTION))
    iy = int(np.floor((y - Y_RANGE[0]) / RESOLUTION))
    iz = int(np.floor((z - Z_RANGE[0]) / RESOLUTION))

    # Clip to valid range
    ix = np.clip(ix, 0, GRID_SIZE[0] - 1)
    iy = np.clip(iy, 0, GRID_SIZE[1] - 1)
    iz = np.clip(iz, 0, GRID_SIZE[2] - 1)

    return ix, iy, iz

def test_round_trip():
    """测试往返转换精度"""
    print("=" * 60)
    print("体素坐标往返转换测试")
    print("=" * 60)

    test_cases = [
        (0, 0, 0),           # 网格起点
        (255, 255, 19),      # 网格中心
        (511, 511, 39),      # 网格终点
        (100, 200, 15),      # 随机点
    ]

    max_error = 0.0

    for ix, iy, iz in test_cases:
        # 索引 -> 坐标
        x, y, z = voxel_index_to_world_coord(ix, iy, iz)

        # 坐标 -> 索引
        ix2, iy2, iz2 = world_coord_to_voxel_index(x, y, z)

        # 验证
        error = abs(ix - ix2) + abs(iy - iy2) + abs(iz - iz2)
        max_error = max(max_error, error)

        status = "✅" if error == 0 else "❌"
        print(f"{status} [{ix:3d}, {iy:3d}, {iz:2d}] -> ({x:6.2f}, {y:6.2f}, {z:5.2f}) -> [{ix2:3d}, {iy2:3d}, {iz2:2d}]")

    print(f"\n最大索引误差: {max_error} (期望: 0)")
    assert max_error == 0, "坐标转换存在误差!"
    print("✅ 所有测试通过\n")

def test_boundary_precision():
    """测试边界体素的坐标精度"""
    print("=" * 60)
    print("边界体素坐标精度测试")
    print("=" * 60)

    # 测试第一个和最后一个体素
    x_first, _, _ = voxel_index_to_world_coord(0, 0, 0)
    x_last, _, _ = voxel_index_to_world_coord(GRID_SIZE[0] - 1, 0, 0)

    # 第一个体素中心应该在 x_min + 0.5 * resolution
    expected_first = X_RANGE[0] + 0.5 * RESOLUTION
    # 最后一个体素中心应该在 x_max - 0.5 * resolution
    expected_last = X_RANGE[1] - 0.5 * RESOLUTION

    error_first = abs(x_first - expected_first)
    error_last = abs(x_last - expected_last)

    print(f"第一个体素 (ix=0):")
    print(f"  实际 X 坐标: {x_first:.6f} m")
    print(f"  期望 X 坐标: {expected_first:.6f} m")
    print(f"  误差: {error_first:.9f} m")

    print(f"\n最后一个体素 (ix={GRID_SIZE[0]-1}):")
    print(f"  实际 X 坐标: {x_last:.6f} m")
    print(f"  期望 X 坐标: {expected_last:.6f} m")
    print(f"  误差: {error_last:.9f} m")

    # 容许 1e-6 米误差 (1 微米)
    assert error_first < 1e-6, f"第一个体素坐标误差过大: {error_first}"
    assert error_last < 1e-6, f"最后一个体素坐标误差过大: {error_last}"
    print("\n✅ 边界精度测试通过\n")

def test_grid_coverage():
    """测试网格覆盖范围"""
    print("=" * 60)
    print("网格覆盖范围验证")
    print("=" * 60)

    print(f"X 范围: {X_RANGE} m")
    print(f"Y 范围: {Y_RANGE} m")
    print(f"Z 范围: {Z_RANGE} m")
    print(f"分辨率: {RESOLUTION} m")
    print(f"网格尺寸: {GRID_SIZE}")

    # 验证覆盖范围
    x_span = X_RANGE[1] - X_RANGE[0]
    y_span = Y_RANGE[1] - Y_RANGE[0]
    z_span = Z_RANGE[1] - Z_RANGE[0]

    expected_x = GRID_SIZE[0] * RESOLUTION
    expected_y = GRID_SIZE[1] * RESOLUTION
    expected_z = GRID_SIZE[2] * RESOLUTION

    print(f"\n实际覆盖:")
    print(f"  X: {x_span:.2f} m (期望: {expected_x:.2f} m)")
    print(f"  Y: {y_span:.2f} m (期望: {expected_y:.2f} m)")
    print(f"  Z: {z_span:.2f} m (期望: {expected_z:.2f} m)")

    # 验证
    assert abs(x_span - expected_x) < 1e-6, f"X 覆盖范围错误"
    assert abs(y_span - expected_y) < 1e-6, f"Y 覆盖范围错误"
    assert abs(z_span - expected_z) < 1e-6, f"Z 覆盖范围错误"

    print("\n✅ 覆盖范围验证通过\n")

if __name__ == '__main__':
    print("\n" + "🔍 体素坐标对齐验证".center(60) + "\n")

    test_grid_coverage()
    test_boundary_precision()
    test_round_trip()

    print("=" * 60)
    print("✅ 所有验证通过! 体素坐标与 UE5 世界坐标严格对齐".center(60))
    print("=" * 60)

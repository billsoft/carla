"""
验证全景反投影算法修正

对比修正前后的反投影结果，检查是否消除了体素间隙
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from processing.panorama_tools import PanoramaTools
from processing.dense_voxel_generator import DenseVoxelGenerator
from config.panorama_config import PANO_WIDTH, PANO_HEIGHT, CUBE_SIZE
from config.occupancy_config import X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION


def create_test_panorama():
    """
    创建一个测试用的全景深度图
    模拟一个均匀分布的场景
    """
    H, W = PANO_HEIGHT, PANO_WIDTH

    # 创建一个均匀深度的全景图（10米远的环形墙）
    depth_pano = np.full((H, W), 10.0, dtype=np.float32)

    # 语义标签：全部设为1（车辆）
    semantic_pano = np.ones((H, W), dtype=np.uint8)

    return depth_pano, semantic_pano


def analyze_point_cloud(points, labels):
    """分析点云的分布特性"""
    print("\n=== 点云分析 ===")
    print(f"总点数: {len(points)}")

    if len(points) == 0:
        return

    # 坐标范围
    print(f"X范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"Y范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"Z范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    # 计算点到原点的距离
    distances = np.linalg.norm(points, axis=1)
    print(f"距离范围: [{distances.min():.2f}, {distances.max():.2f}]")
    print(f"平均距离: {distances.mean():.2f}")
    print(f"距离标准差: {distances.std():.2f}")

    # 角度分布检查
    theta = np.arctan2(points[:, 1], points[:, 0])
    phi = np.arctan2(points[:, 2], np.sqrt(points[:, 0]**2 + points[:, 1]**2))

    print(f"θ (水平角) 范围: [{np.degrees(theta.min()):.1f}°, {np.degrees(theta.max()):.1f}°]")
    print(f"φ (垂直角) 范围: [{np.degrees(phi.min()):.1f}°, {np.degrees(phi.max()):.1f}°]")


def analyze_voxels(occupancy, mask):
    """分析体素的连续性"""
    print("\n=== 体素分析 ===")

    total = np.prod(occupancy.shape)
    observed = np.sum(mask)
    occupied = np.sum(occupancy > 0)

    print(f"总体素数: {total}")
    print(f"观测到的体素: {observed} ({observed/total*100:.2f}%)")
    print(f"占用体素: {occupied} ({occupied/observed*100:.2f}% of observed)")

    # 检查水平面的连续性
    # 选择中间高度层 (Z=0附近)
    z_center_idx = occupancy.shape[2] // 2
    horizontal_slice = mask[:, :, z_center_idx]

    print(f"\n水平切片 (Z={z_center_idx}) 分析:")
    print(f"该层有效体素: {np.sum(horizontal_slice)}")

    # 计算间隙数（相邻有效体素之间的空隙）
    gaps_x = 0
    gaps_y = 0

    for i in range(horizontal_slice.shape[0] - 1):
        for j in range(horizontal_slice.shape[1]):
            if horizontal_slice[i, j] and not horizontal_slice[i+1, j]:
                # 检查是否在下一个有效体素之前有间隙
                for k in range(i+2, min(i+10, horizontal_slice.shape[0])):
                    if horizontal_slice[k, j]:
                        gaps_x += 1
                        break

    for j in range(horizontal_slice.shape[1] - 1):
        for i in range(horizontal_slice.shape[0]):
            if horizontal_slice[i, j] and not horizontal_slice[i, j+1]:
                for k in range(j+2, min(j+10, horizontal_slice.shape[1])):
                    if horizontal_slice[i, k]:
                        gaps_y += 1
                        break

    print(f"X方向间隙数: {gaps_x}")
    print(f"Y方向间隙数: {gaps_y}")

    if gaps_x + gaps_y > 0:
        print("⚠ 检测到体素间隙！")
    else:
        print("✓ 体素连续无间隙")


def main():
    print("=" * 60)
    print("全景反投影算法验证")
    print("=" * 60)

    # 1. 创建测试数据
    print("\n创建测试全景图...")
    depth_pano, semantic_pano = create_test_panorama()
    print(f"全景图尺寸: {depth_pano.shape}")

    # 2. 初始化工具
    print("\n初始化处理工具...")
    pano_tools = PanoramaTools(PANO_WIDTH, PANO_HEIGHT, CUBE_SIZE)
    voxel_generator = DenseVoxelGenerator(X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION)

    # 3. 反投影
    print("\n执行反投影...")
    points, labels = pano_tools.unproject_panorama(depth_pano, semantic_pano, max_depth=100.0)
    analyze_point_cloud(points, labels)

    # 4. 体素化
    print("\n执行体素化...")
    occupancy, mask = voxel_generator.generate(points, labels)
    analyze_voxels(occupancy, mask)

    # 5. 理论验证
    print("\n=== 理论一致性检查 ===")
    # 对于均匀10米的环形墙，所有点应该在距离=10附近
    distances = np.linalg.norm(points, axis=1)
    max_deviation = np.abs(distances - 10.0).max()
    print(f"最大距离偏差: {max_deviation:.4f} 米")

    if max_deviation < 0.01:
        print("✓ 距离一致性良好")
    else:
        print(f"⚠ 距离偏差较大: {max_deviation:.4f} 米")

    print("\n验证完成！")


if __name__ == '__main__':
    main()

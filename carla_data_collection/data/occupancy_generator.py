"""
Occupancy 体素生成器
从激光雷达点云生成 3D 占据网格
"""

import numpy as np
from typing import Tuple, Dict

from config.occupancy_config import (
    OCCUPANCY_CONFIG,
    CARLA_TO_OCCUPANCY_LABEL_MAP,
    OCCUPANCY_CLASS_NAMES
)


class OccupancyGenerator:
    """
    将语义激光雷达点云转换为 3D Occupancy 体素网格
    """

    def __init__(self, config: Dict = None):
        """
        Args:
            config: Occupancy 配置字典 (默认使用 OCCUPANCY_CONFIG)
        """
        self.config = config or OCCUPANCY_CONFIG

        # 空间范围 (米)
        self.x_range = self.config['x_range']
        self.y_range = self.config['y_range']
        self.z_range = self.config['z_range']
        self.resolution = self.config['resolution']

        # 网格尺寸
        self.grid_size = [
            int((self.x_range[1] - self.x_range[0]) / self.resolution),
            int((self.y_range[1] - self.y_range[0]) / self.resolution),
            int((self.z_range[1] - self.z_range[0]) / self.resolution),
        ]

        print(f"[OccupancyGenerator] 初始化:")
        print(f"  空间范围: X[{self.x_range[0]}, {self.x_range[1]}], "
              f"Y[{self.y_range[0]}, {self.y_range[1]}], "
              f"Z[{self.z_range[0]}, {self.z_range[1]}]")
        print(f"  分辨率: {self.resolution}m")
        print(f"  网格尺寸: {self.grid_size}")

    def generate(self,
                 xyz_ego: np.ndarray,
                 semantic_tags: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        从车辆坐标系点云生成 Occupancy 网格

        Args:
            xyz_ego: (N, 3) 车辆坐标系下的点云
            semantic_tags: (N,) 每个点的 CARLA 语义标签

        Returns:
            occupancy: (X, Y, Z) 体素语义标签
            mask: (X, Y, Z) 有效观测掩码 (True=有激光雷达观测)
        """
        # 初始化空网格
        occupancy = np.zeros(self.grid_size, dtype=np.uint8)
        count = np.zeros(self.grid_size, dtype=np.int32)

        # ========================================
        # 核心: 点坐标 → 网格索引
        # ========================================

        # 计算网格索引
        grid_x = ((xyz_ego[:, 0] - self.x_range[0]) / self.resolution).astype(np.int32)
        grid_y = ((xyz_ego[:, 1] - self.y_range[0]) / self.resolution).astype(np.int32)
        grid_z = ((xyz_ego[:, 2] - self.z_range[0]) / self.resolution).astype(np.int32)

        # 过滤超出范围的点
        valid = (
            (grid_x >= 0) & (grid_x < self.grid_size[0]) &
            (grid_y >= 0) & (grid_y < self.grid_size[1]) &
            (grid_z >= 0) & (grid_z < self.grid_size[2])
        )

        grid_x = grid_x[valid]
        grid_y = grid_y[valid]
        grid_z = grid_z[valid]
        tags = semantic_tags[valid]

        print(f"[OccupancyGenerator] 点云过滤:")
        print(f"  原始点数: {len(xyz_ego)}")
        print(f"  有效点数: {len(grid_x)} ({len(grid_x)/len(xyz_ego)*100:.1f}%)")

        # 映射 CARLA 语义标签 → Occupancy 类别
        occ_labels = np.array([
            CARLA_TO_OCCUPANCY_LABEL_MAP.get(int(t), 0) for t in tags
        ], dtype=np.uint8)

        # ========================================
        # 填充体素网格
        # ========================================
        # 策略: 非空标签优先,多个点投票
        for i in range(len(grid_x)):
            x, y, z = grid_x[i], grid_y[i], grid_z[i]
            label = occ_labels[i]

            # 非空标签优先覆盖空标签
            if label != 0 or occupancy[x, y, z] == 0:
                occupancy[x, y, z] = label
                count[x, y, z] += 1

        # 生成有效掩码 (有点云覆盖的区域)
        mask = count > 0

        # 统计
        self._print_statistics(occupancy, mask)

        return occupancy, mask

    def _print_statistics(self, occupancy: np.ndarray, mask: np.ndarray):
        """打印体素统计信息"""
        print(f"\n[OccupancyGenerator] 体素统计:")
        print(f"  网格形状: {occupancy.shape}")
        print(f"  总体素数: {occupancy.size}")
        print(f"  有效观测: {np.sum(mask)} ({np.sum(mask)/occupancy.size*100:.2f}%)")
        print(f"  非空体素: {np.sum(occupancy > 0)} ({np.sum(occupancy > 0)/occupancy.size*100:.2f}%)")

        print(f"\n  类别分布:")
        for label in range(len(OCCUPANCY_CLASS_NAMES)):
            count = np.sum(occupancy == label)
            if count > 0:
                class_name = OCCUPANCY_CLASS_NAMES[label]
                percentage = count / occupancy.size * 100
                print(f"    [{label:2d}] {class_name:15s}: {count:7d} ({percentage:5.2f}%)")

    def save_occupancy(self,
                      occupancy: np.ndarray,
                      mask: np.ndarray,
                      filepath: str):
        """
        保存 Occupancy 数据到 .npz 文件

        Args:
            occupancy: (X, Y, Z) 体素标签
            mask: (X, Y, Z) 有效掩码
            filepath: 保存路径 (如 'data/000000.npz')
        """
        np.savez_compressed(
            filepath,
            occupancy=occupancy,
            mask=mask,
            # 保存配置信息
            x_range=self.x_range,
            y_range=self.y_range,
            z_range=self.z_range,
            resolution=self.resolution,
            grid_size=self.grid_size
        )
        print(f"[OccupancyGenerator] 已保存: {filepath}")

    @staticmethod
    def load_occupancy(filepath: str) -> Dict:
        """
        从 .npz 文件加载 Occupancy 数据

        Args:
            filepath: .npz 文件路径

        Returns:
            data: 包含 occupancy, mask 等的字典
        """
        data = np.load(filepath)
        return {
            'occupancy': data['occupancy'],
            'mask': data['mask'],
            'x_range': data['x_range'].tolist(),
            'y_range': data['y_range'].tolist(),
            'z_range': data['z_range'].tolist(),
            'resolution': float(data['resolution']),
            'grid_size': data['grid_size'].tolist()
        }


def test_occupancy_generator():
    """测试 Occupancy 生成器"""
    print("="*60)
    print("测试 Occupancy Generator")
    print("="*60)

    # 创建模拟点云
    np.random.seed(42)

    # 生成 10000 个随机点
    xyz_ego = np.random.uniform(
        low=[-50, -50, -4],
        high=[50, 50, 4],
        size=(10000, 3)
    ).astype(np.float32)

    # 随机语义标签 (模拟 CARLA 标签)
    semantic_tags = np.random.choice(
        [0, 1, 4, 7, 8, 9, 10, 12],  # 常见标签
        size=10000,
        p=[0.3, 0.1, 0.05, 0.2, 0.15, 0.1, 0.08, 0.02]  # 概率分布
    )

    # 创建生成器
    generator = OccupancyGenerator()

    # 生成 Occupancy
    occupancy, mask = generator.generate(xyz_ego, semantic_tags)

    # 保存测试
    generator.save_occupancy(occupancy, mask, 'test_occupancy.npz')

    # 加载测试
    loaded_data = OccupancyGenerator.load_occupancy('test_occupancy.npz')
    print(f"\n加载成功: {loaded_data['occupancy'].shape}")

    print("\n✓ Occupancy Generator 测试通过!")


if __name__ == '__main__':
    test_occupancy_generator()

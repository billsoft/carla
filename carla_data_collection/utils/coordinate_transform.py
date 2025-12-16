"""
坐标转换工具
世界坐标系 ↔ 车辆坐标系
"""

import numpy as np
import carla


def get_transform_matrix(transform: carla.Transform) -> np.ndarray:
    """
    将 CARLA Transform 转换为 4×4 变换矩阵

    Args:
        transform: CARLA Transform 对象

    Returns:
        T: (4, 4) 齐次变换矩阵
           [[R, t],
            [0, 1]]
    """
    loc = transform.location
    rot = transform.rotation

    # 欧拉角转弧度
    pitch = np.radians(rot.pitch)
    yaw = np.radians(rot.yaw)
    roll = np.radians(rot.roll)

    # 计算旋转矩阵 (ZYX 欧拉角顺序)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    # 旋转矩阵
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr                ]
    ], dtype=np.float64)

    # 组合为 4×4 矩阵
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [loc.x, loc.y, loc.z]

    return T


def world_to_ego(points_world: np.ndarray,
                 ego_transform: carla.Transform) -> np.ndarray:
    """
    将点从世界坐标系转换到车辆坐标系

    Args:
        points_world: (N, 3) 世界坐标系下的点
        ego_transform: 车辆的 Transform

    Returns:
        points_ego: (N, 3) 车辆坐标系下的点
    """
    # 获取车辆位姿矩阵
    ego_matrix = get_transform_matrix(ego_transform)
    ego_matrix_inv = np.linalg.inv(ego_matrix)

    # 转换为齐次坐标
    N = points_world.shape[0]
    points_homo = np.hstack([points_world, np.ones((N, 1))])  # (N, 4)

    # 应用逆变换
    points_ego_homo = (ego_matrix_inv @ points_homo.T).T  # (N, 4)
    points_ego = points_ego_homo[:, :3]  # (N, 3)

    return points_ego


def ego_to_world(points_ego: np.ndarray,
                 ego_transform: carla.Transform) -> np.ndarray:
    """
    将点从车辆坐标系转换到世界坐标系

    Args:
        points_ego: (N, 3) 车辆坐标系下的点
        ego_transform: 车辆的 Transform

    Returns:
        points_world: (N, 3) 世界坐标系下的点
    """
    # 获取车辆位姿矩阵
    ego_matrix = get_transform_matrix(ego_transform)

    # 转换为齐次坐标
    N = points_ego.shape[0]
    points_homo = np.hstack([points_ego, np.ones((N, 1))])  # (N, 4)

    # 应用变换
    points_world_homo = (ego_matrix @ points_homo.T).T  # (N, 4)
    points_world = points_world_homo[:, :3]  # (N, 3)

    return points_world


def test_coordinate_transform():
    """测试坐标转换的正确性"""
    # 创建测试点 (世界坐标系)
    points_world = np.array([
        [10, 20, 0],
        [15, 25, 1],
        [5, 18, -0.5]
    ], dtype=np.float64)

    # 创建测试 Transform
    ego_transform = carla.Transform(
        carla.Location(x=100, y=200, z=0.5),
        carla.Rotation(pitch=0, yaw=45, roll=0)
    )

    # 世界 → 车辆
    points_ego = world_to_ego(points_world, ego_transform)

    # 车辆 → 世界 (应该还原)
    points_world_recovered = ego_to_world(points_ego, ego_transform)

    # 检查误差
    error = np.abs(points_world - points_world_recovered).max()
    print(f"坐标转换测试:")
    print(f"  原始世界坐标:\n{points_world}")
    print(f"  转换到车辆坐标:\n{points_ego}")
    print(f"  还原世界坐标:\n{points_world_recovered}")
    print(f"  最大误差: {error:.10f} (应接近 0)")

    assert error < 1e-6, f"坐标转换误差过大: {error}"
    print("✓ 坐标转换测试通过!")


if __name__ == '__main__':
    test_coordinate_transform()

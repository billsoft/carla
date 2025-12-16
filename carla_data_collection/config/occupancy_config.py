"""
Occupancy 体素配置
"""

# 体素空间配置
OCCUPANCY_CONFIG = {
    # 空间范围 (米)
    'x_range': [-50, 50],    # 前后 100 米
    'y_range': [-50, 50],    # 左右 100 米
    'z_range': [-4, 4],      # 上下 8 米

    # 体素分辨率
    'resolution': 0.5,       # 每个体素 0.5 米

    # 网格尺寸 (自动计算)
    'grid_size': [200, 200, 16],  # (100/0.5, 100/0.5, 8/0.5)
}

# 语义激光雷达配置
SEMANTIC_LIDAR_CONFIG = {
    'channels': 64,              # 64 线
    'points_per_second': 1200000,  # 每秒 120 万点
    'rotation_frequency': 20,    # 20Hz 旋转
    'range': 100.0,              # 100 米探测范围
    'upper_fov': 15.0,           # 上视角 15°
    'lower_fov': -25.0,          # 下视角 -25°

    # 安装位置 (相对车辆)
    'position': {'x': 0.0, 'y': 0.0, 'z': 2.5},  # 车顶中央
    'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0},
}

# CARLA 语义标签 → Occupancy 类别映射
# 参考 CARLA 语义分割标签: https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
CARLA_TO_OCCUPANCY_LABEL_MAP = {
    0:  0,   # Unlabeled → empty
    1:  14,  # Building → building
    2:  15,  # Fence → barrier
    3:  15,  # Other → barrier
    4:  6,   # Pedestrian → pedestrian
    5:  13,  # Pole → pole
    6:  9,   # RoadLine → road_marking
    7:  9,   # Road → road
    8:  10,  # SideWalk → sidewalk
    9:  12,  # Vegetation → vegetation
    10: 1,   # Vehicles → car (包括 Car, Truck, Bus)
    11: 17,  # Wall → wall
    12: 16,  # TrafficSign → traffic_sign
    13: 11,  # Sky → sky (通常不在体素范围内)
    14: 8,   # Ground → terrain
    15: 2,   # Bridge → construction
    16: 15,  # RailTrack → barrier
    17: 15,  # GuardRail → barrier
    18: 16,  # TrafficLight → traffic_sign
    19: 0,   # Static → empty
    20: 0,   # Dynamic → empty
    21: 0,   # Water → empty
    22: 8,   # Terrain → terrain
}

# Occupancy 类别名称 (与训练时一致)
OCCUPANCY_CLASS_NAMES = [
    'empty',           # 0
    'car',             # 1
    'truck',           # 2 (CARLA 中归为 Vehicles)
    'construction',    # 3
    'bicycle',         # 4 (CARLA 中可能没有)
    'motorcycle',      # 5 (CARLA 中可能没有)
    'pedestrian',      # 6
    'traffic_cone',    # 7 (CARLA 中可能没有)
    'terrain',         # 8
    'road',            # 9
    'sidewalk',        # 10
    'sky',             # 11
    'vegetation',      # 12
    'pole',            # 13
    'building',        # 14
    'barrier',         # 15
    'traffic_sign',    # 16
    'wall',            # 17
]

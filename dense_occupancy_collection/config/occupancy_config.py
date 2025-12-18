"""
Occupancy 体素空间配置
定义 3D 体素网格的范围、分辨率和语义类别映射
"""

# 体素空间范围 (以自车为中心, 单位: 米)
X_RANGE = [-50.0, 50.0]   # 前后 100m
Y_RANGE = [-50.0, 50.0]   # 左右 100m
Z_RANGE = [-4.0, 4.0]     # 上下 8m

# 体素分辨率 (单位: 米)
# 注意: 0.1m会导致1000x1000x80=8000万体素，浏览器无法渲染
# 0.2m是性能和精度的平衡点: 500x500x40=1000万体素
RESOLUTION = 0.2  # 每个体素边长 0.2m，可检测细小物体且性能可控

# 计算网格尺寸
GRID_SIZE = [
    int((X_RANGE[1] - X_RANGE[0]) / RESOLUTION),  # 500
    int((Y_RANGE[1] - Y_RANGE[0]) / RESOLUTION),  # 500
    int((Z_RANGE[1] - Z_RANGE[0]) / RESOLUTION),  # 40
]

# 语义激光雷达配置 (128线 - 旧版，用于体素生成)
SEMANTIC_LIDAR_CONFIG = {
    'channels': 128,
    'range': 100.0,
    'points_per_second': 1000000,
    'rotation_frequency': 20,
    'upper_fov': 30.0,
    'lower_fov': -40.0,
    'position': {'x': 0.0, 'y': 0.0, 'z': 2.4},  # 安装在车顶
    'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
}

# 64线语义激光雷达配置 (用于可见性检测) - 参考carla_data_collection配置
VISIBILITY_LIDAR_CONFIG = {
    'channels': 64,                # 64线（足够精度，不会超时）
    'range': 100.0,                # 100米范围
    'points_per_second': 1200000,  # 120万点/秒 (每帧6万点@20Hz)
    'rotation_frequency': 20,      # 20Hz
    'upper_fov': 15.0,             # 上15° (覆盖主要区域)
    'lower_fov': -25.0,            # 下-25° (覆盖地面)
    'position': {'x': 0.0, 'y': 0.0, 'z': 2.4},  # 安装在车顶
    'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
}

# CARLA 语义标签 (23类) 到 Occupancy 标签 (18类) 的映射
# 适配 OpenOccupancy / SemanticKITTI 格式
# [1] car (Red)
# [6] pedestrian (Green)
CARLA_TO_OCCUPANCY_MAPPING = {
    0: 0,   # Unlabeled → free / empty
    1: 15,  # Building → manmade
    2: 1,   # Fence → barrier
    3: 17,  # Other → general object / other
    4: 7,   # Pedestrian → pedestrian
    5: 15,  # Pole → manmade
    6: 11,  # RoadLine → driveable surface
    7: 11,  # Road → driveable surface
    8: 13,  # SideWalk → sidewalk
    9: 16,  # Vegetation → vegetation
    10: 4,  # Vehicles → car
    11: 15, # Wall → manmade
    12: 15, # TrafficSign → manmade
    13: 0,  # Sky → free / empty
    14: 12, # Ground → other flat
    15: 15, # Bridge → manmade
    16: 17, # RailTrack → general object / other
    17: 1,  # GuardRail → barrier
    18: 15, # TrafficLight → manmade
    19: 17, # Static → general object / other
    20: 17, # Dynamic → general object / other
    21: 12, # Water → other flat
    22: 14, # Terrain → terrain
}

# Occupancy 标签名称
OCCUPANCY_LABELS = [
    'free',               # 0
    'barrier',            # 1
    'bicycle',            # 2
    'bus',                # 3
    'car',                # 4
    'construction_vehicle', # 5
    'motorcycle',         # 6
    'pedestrian',         # 7
    'traffic_cone',       # 8
    'trailer',            # 9
    'truck',              # 10
    'driveable_surface',  # 11
    'other_flat',         # 12
    'sidewalk',           # 13
    'terrain',            # 14
    'manmade',            # 15
    'vegetation',         # 16
    'general_object',     # 17
]

# 可视化颜色映射 (RGB)
# 尝试匹配常见标准
OCCUPANCY_COLORS = [
    (0, 0, 0),        # 0: free
    (200, 200, 200),  # 1: barrier
    (128, 128, 0),    # 2: bicycle
    (0, 0, 128),      # 3: bus
    (0, 128, 0),      # 4: car
    (128, 0, 128),    # 5: construction_vehicle
    (128, 0, 0),      # 6: motorcycle
    (255, 0, 0),      # 7: pedestrian
    (255, 165, 0),    # 8: traffic_cone
    (0, 128, 128),    # 9: trailer
    (0, 0, 255),      # 10: truck
    (100, 100, 100),  # 11: driveable_surface
    (150, 150, 150),  # 12: other_flat
    (255, 192, 203),  # 13: sidewalk
    (0, 255, 0),      # 14: terrain
    (255, 255, 0),    # 15: manmade
    (0, 255, 128),    # 16: vegetation
    (255, 0, 255),    # 17: general_object
]


def get_voxel_config():
    """
    获取完整的体素配置字典

    Returns:
        dict: 包含所有体素参数的配置字典
    """
    return {
        'x_range': X_RANGE,
        'y_range': Y_RANGE,
        'z_range': Z_RANGE,
        'resolution': RESOLUTION,
        'grid_size': GRID_SIZE,
        'mapping': CARLA_TO_OCCUPANCY_MAPPING
    }

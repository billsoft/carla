"""
Occupancy 体素空间配置
定义 3D 体素网格的范围、分辨率和语义类别映射
"""

# 体素空间范围 (以自车为中心, 单位: 米)
X_RANGE = [-50.0, 50.0]   # 前后 100m
Y_RANGE = [-50.0, 50.0]   # 左右 100m
Z_RANGE = [-4.0, 4.0]     # 上下 8m

# 体素分辨率 (单位: 米)
RESOLUTION = 0.5  # 每个体素边长 0.5m

# 计算网格尺寸
GRID_SIZE = [
    int((X_RANGE[1] - X_RANGE[0]) / RESOLUTION),  # 200
    int((Y_RANGE[1] - Y_RANGE[0]) / RESOLUTION),  # 200
    int((Z_RANGE[1] - Z_RANGE[0]) / RESOLUTION),  # 16
]

# CARLA 语义标签 (23类) 到 Occupancy 标签 (18类) 的映射
CARLA_TO_OCCUPANCY_MAPPING = {
    0: 0,   # Unlabeled → Unlabeled
    1: 1,   # Building → Building
    2: 2,   # Fence → Fence
    3: 3,   # Other → Other
    4: 4,   # Pedestrian → Pedestrian
    5: 5,   # Pole → Pole
    6: 6,   # RoadLine → RoadLine
    7: 7,   # Road → Road
    8: 8,   # SideWalk → Sidewalk
    9: 9,   # Vegetation → Vegetation
    10: 10, # Vehicles → Vehicles
    11: 11, # Wall → Wall
    12: 12, # TrafficSign → TrafficSign
    13: 13, # Sky → Sky
    14: 14, # Ground → Ground
    15: 15, # Bridge → Bridge
    16: 16, # RailTrack → RailTrack
    17: 17, # GuardRail → GuardRail
    18: 3,  # TrafficLight → Other (合并)
    19: 3,  # Static → Other (合并)
    20: 9,  # Dynamic → Vegetation (合并)
    21: 14, # Water → Ground (合并)
    22: 9,  # Terrain → Vegetation (合并)
}

# Occupancy 标签名称
OCCUPANCY_LABELS = [
    'Unlabeled',
    'Building',
    'Fence',
    'Other',
    'Pedestrian',
    'Pole',
    'RoadLine',
    'Road',
    'Sidewalk',
    'Vegetation',
    'Vehicles',
    'Wall',
    'TrafficSign',
    'Sky',
    'Ground',
    'Bridge',
    'RailTrack',
    'GuardRail'
]

# 可视化颜色映射 (RGB, 0-255)
OCCUPANCY_COLORS = [
    (0, 0, 0),        # 0: Unlabeled - 黑色
    (70, 70, 70),     # 1: Building - 深灰
    (190, 153, 153),  # 2: Fence - 浅灰
    (250, 170, 160),  # 3: Other - 橙色
    (220, 20, 60),    # 4: Pedestrian - 红色
    (153, 153, 153),  # 5: Pole - 灰色
    (157, 234, 50),   # 6: RoadLine - 黄绿
    (128, 64, 128),   # 7: Road - 紫色
    (244, 35, 232),   # 8: Sidewalk - 粉色
    (107, 142, 35),   # 9: Vegetation - 绿色
    (0, 0, 142),      # 10: Vehicles - 深蓝
    (102, 102, 156),  # 11: Wall - 灰蓝
    (220, 220, 0),    # 12: TrafficSign - 黄色
    (70, 130, 180),   # 13: Sky - 天蓝
    (81, 0, 81),      # 14: Ground - 深紫
    (150, 100, 100),  # 15: Bridge - 棕色
    (230, 150, 140),  # 16: RailTrack - 浅红棕
    (180, 165, 180),  # 17: GuardRail - 浅紫灰
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
        'label_mapping': CARLA_TO_OCCUPANCY_MAPPING,
        'label_names': OCCUPANCY_LABELS,
        'colors': OCCUPANCY_COLORS
    }

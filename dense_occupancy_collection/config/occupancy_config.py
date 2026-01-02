"""
Occupancy 体素空间配置
定义 3D 体素网格的范围、分辨率和语义类别映射
"""

# 体素空间范围 (以自车为中心, 单位: 米)
# ⭐ 对齐 nuScenes/OpenOccupancy 标准
X_RANGE = [-51.2, 51.2]   # 前后 102.4m (512 × 0.2m)
Y_RANGE = [-51.2, 51.2]   # 左右 102.4m (512 × 0.2m)
Z_RANGE = [-4.0, 4.0]     # 上下 8m (40 × 0.2m)

# 体素分辨率 (单位: 米)
# ⭐ 0.2m 是自动驾驶可用的最低标准,可检测小于0.5m的障碍物
RESOLUTION = 0.2  # 每个体素边长 0.2m，可检测细小物体且性能可控

# 计算网格尺寸
# ⭐ 严格验证: GRID_SIZE * RESOLUTION 必须等于 RANGE
GRID_SIZE = [
    int((X_RANGE[1] - X_RANGE[0]) / RESOLUTION),  # 512
    int((Y_RANGE[1] - Y_RANGE[0]) / RESOLUTION),  # 512
    int((Z_RANGE[1] - Z_RANGE[0]) / RESOLUTION),  # 40
]

# 断言验证 (防止配置错误)
assert GRID_SIZE[0] == 512, f"X 网格尺寸错误: {GRID_SIZE[0]} != 512"
assert GRID_SIZE[1] == 512, f"Y 网格尺寸错误: {GRID_SIZE[1]} != 512"
assert GRID_SIZE[2] == 40, f"Z 网格尺寸错误: {GRID_SIZE[2]} != 40"
assert abs((X_RANGE[1] - X_RANGE[0]) - GRID_SIZE[0] * RESOLUTION) < 1e-6, "X 范围与分辨率不匹配"
assert abs((Y_RANGE[1] - Y_RANGE[0]) - GRID_SIZE[1] * RESOLUTION) < 1e-6, "Y 范围与分辨率不匹配"
assert abs((Z_RANGE[1] - Z_RANGE[0]) - GRID_SIZE[2] * RESOLUTION) < 1e-6, "Z 范围与分辨率不匹配"

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

# 256线高密度语义激光雷达配置 (平衡性能与密度)
# ⭐ 调整说明：
#    1. 1000万点/秒会导致 UE4 物理引擎光线投射过载卡死，回退到 350万点/秒
#    2. 保持 256 线以维持较好的垂直分辨率 (检测杆塔高度)
#    3. 配合保守光栅化算法，即使点数较少也能更好捕捉细小物体
VISIBILITY_LIDAR_CONFIG = {
    'channels': 256,               # 256线
    'range': 100.0,                # 100米范围
    'points_per_second': 3500000,  # ⭐ 350万点/秒 (每帧17.5万点)，避免服务端卡死
    'rotation_frequency': 20,      # 20Hz
    'upper_fov': 30.0,             # 上30°
    'lower_fov': -30.0,            # 下-30°
    'position': {'x': 0.0, 'y': 0.0, 'z': 1.5},
    'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
    'horizontal_fov': 360.0
}

# ⭐ 6路 Cube Map 深度相机配置 (用于高精度可见性过滤)
# Front, Right, Back, Left, Up, Down
# ⭐ 高度统一为 Z=2.2m (透波且不被车身遮挡)
# ⭐ 垂直 FOV 调整为 60 度 (上30, 下30) 以匹配特斯拉方案
DEPTH_CAMERA_CONFIG = {
    'width': 512,
    'height': 512,
    'fov': 60.0,  # 垂直 60度
    'cameras': [
        {'id': 'depth_front',       'pos': {'x': 0, 'y': 0, 'z': 2.2}, 'rot': {'pitch': 0, 'yaw': 0,    'roll': 0}},
        {'id': 'depth_front_right', 'pos': {'x': 0, 'y': 0, 'z': 2.2}, 'rot': {'pitch': 0, 'yaw': 45,   'roll': 0}},
        {'id': 'depth_right',       'pos': {'x': 0, 'y': 0, 'z': 2.2}, 'rot': {'pitch': 0, 'yaw': 90,   'roll': 0}},
        {'id': 'depth_back_right',  'pos': {'x': 0, 'y': 0, 'z': 2.2}, 'rot': {'pitch': 0, 'yaw': 135,  'roll': 0}},
        {'id': 'depth_back',        'pos': {'x': 0, 'y': 0, 'z': 2.2}, 'rot': {'pitch': 0, 'yaw': 180,  'roll': 0}},
        {'id': 'depth_back_left',   'pos': {'x': 0, 'y': 0, 'z': 2.2}, 'rot': {'pitch': 0, 'yaw': -135, 'roll': 0}},
        {'id': 'depth_left',        'pos': {'x': 0, 'y': 0, 'z': 2.2}, 'rot': {'pitch': 0, 'yaw': -90,  'roll': 0}},
        {'id': 'depth_front_left',  'pos': {'x': 0, 'y': 0, 'z': 2.2}, 'rot': {'pitch': 0, 'yaw': -45,  'roll': 0}},
    ]
}

# 可见性过滤配置 - 已弃用
# VISIBILITY_CONFIG 已移除，相关逻辑已迁移至 VisibilityFilter (Depth Camera)


# CARLA 语义标签 (CityObjectLabel) 到 Occupancy 标签 (18类) 的映射
# 基于 CARLA 0.9.15+ / UE5.5 CityObjectLabel 枚举值修正
# 参考 debug_labels.py 的输出
CARLA_TO_OCCUPANCY_MAPPING = {
    0: 0,   # NONE -> free
    1: 11,  # Roads -> driveable_surface
    2: 13,  # Sidewalks -> sidewalk
    3: 15,  # Buildings -> manmade
    4: 15,  # Walls -> manmade
    5: 1,   # Fences -> barrier
    6: 15,  # Poles -> manmade
    7: 8,   # TrafficLight -> traffic_cone (交通标识)
    8: 8,   # TrafficSigns -> traffic_cone (交通标识)
    9: 16,  # Vegetation -> vegetation
    10: 14, # Terrain -> terrain
    11: 0,  # Sky -> free
    12: 7,  # Pedestrians -> pedestrian
    13: 7,  # Rider -> pedestrian (骑手)
    14: 4,  # Car -> car
    15: 10, # Truck -> truck
    16: 3,  # Bus -> bus
    17: 3,  # Train -> bus (暂定，nuScenes无Train类)
    18: 6,  # Motorcycle -> motorcycle
    19: 2,  # Bicycle -> bicycle
    20: 17, # Static -> unknown
    21: 17, # Dynamic -> unknown
    22: 17, # Other -> unknown
    23: 12, # Water -> other_flat
    24: 11, # RoadLines -> driveable_surface
    25: 12, # Ground -> other_flat
    26: 15, # Bridge -> manmade
    27: 11, # RailTrack -> driveable_surface
    28: 1,  # GuardRail -> barrier
    29: 17, # Rock -> unknown
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
    'general_object',     # 17 (业界标准，包含未知障碍物)
]

# 可视化颜色映射 (RGB)
# 基于自动驾驶语义优化的配色方案 (与 occupancy_viewer/viewer.js 保持一致)
OCCUPANCY_COLORS = [
    (0, 0, 0),        # 0: free - 黑色/空白
    (200, 200, 200),  # 1: barrier - 银灰色/物理阻隔
    (255, 215, 0),    # 2: bicycle - 金黄色/脆弱交通参与者
    (255, 99, 71),    # 3: bus - 番茄红/大型公交
    (255, 140, 0),    # 4: car - 深橙色/最常见车辆
    (255, 165, 0),    # 5: construction_vehicle - 橙色/工程车
    (255, 20, 147),   # 6: motorcycle - 深粉红/高风险
    (255, 0, 0),      # 7: pedestrian - 纯红色/最高优先级 ⭐
    (255, 255, 0),    # 8: traffic_cone - 纯黄色/交通标识 ⭐
    (65, 105, 225),   # 9: trailer - 皇家蓝
    (0, 0, 255),      # 10: truck - 纯蓝色
    (80, 80, 80),     # 11: driveable_surface - 深灰/可行驶路面
    (120, 120, 120),  # 12: other_flat - 中灰/其他平面
    (160, 160, 160),  # 13: sidewalk - 浅灰/人行道
    (139, 69, 19),    # 14: terrain - 马鞍棕/泥土地形
    (220, 220, 220),  # 15: manmade - 淡灰白/建筑物
    (34, 139, 34),    # 16: vegetation - 森林绿/植被
    (255, 0, 255),    # 17: general_object - 洋红色/未知障碍物 ⭐
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

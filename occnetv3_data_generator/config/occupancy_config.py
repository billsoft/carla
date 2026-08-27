"""
OccNetV3 体素配置 - 400x400x32 体素网格
用户自定义高分辨率配置 (0.2m分辨率, 80m×80m×6.4m覆盖范围)
"""

# ========== 体素空间定义 ==========
# 车辆坐标系 (右手系): X前 Y左 Z上 -> 修正: CARLA 原生坐标系 (左手系 X前 Y右 Z上)
# 注意: 虽然 nuScenes 是 Y左，但为了保持与 CARLA 传感器一致，我们暂保持 Y右。
#       如果在后续训练中需要 Y左，请在 Dataset Loader 中进行 flip(axis=1)。
# 原点: 车辆后轴中心地面 (Z=0)

# 对齐 OccNetV3 / nuScenes 范围标准 (40.0m)
X_RANGE = [-40.0, 40.0]  # 前后 80.0m
Y_RANGE = [-40.0, 40.0]  # 左右 80.0m
Z_RANGE = [-1.0, 5.4]    # 上下 6.4m

RESOLUTION = 0.2  # 每个体素边长 0.2m

# 计算网格尺寸: (X_max - X_min) / resolution
GRID_SIZE = (
    int((X_RANGE[1] - X_RANGE[0]) / RESOLUTION),  # 400
    int((Y_RANGE[1] - Y_RANGE[0]) / RESOLUTION),  # 400
    int((Z_RANGE[1] - Z_RANGE[0]) / RESOLUTION),  # 32
)

assert GRID_SIZE == (400, 400, 32), f"网格尺寸计算错误: {GRID_SIZE}"

PC_RANGE = [X_RANGE[0], Y_RANGE[0], Z_RANGE[0], X_RANGE[1], Y_RANGE[1], Z_RANGE[1]]

# ========== 语义类别定义 (18类) ==========
SEMANTIC_CLASSES = {
    0: 'empty',                 # 空气/无物体
    1: 'barrier',               # 护栏/路障
    2: 'bicycle',               # 自行车
    3: 'bus',                   # 公交车
    4: 'car',                   # 小汽车
    5: 'construction_vehicle',  # 工程车辆
    6: 'motorcycle',            # 摩托车
    7: 'pedestrian',            # 行人
    8: 'traffic_cone',          # 交通锥
    9: 'trailer',               # 拖车/挂车
    10: 'truck',                # 卡车
    11: 'driveable_surface',    # 可行驶路面
    12: 'other_flat',           # 其他平坦表面
    13: 'sidewalk',             # 人行道
    14: 'terrain',              # 地形(草地等)
    15: 'manmade',              # 人造建筑
    16: 'vegetation',           # 植被
    17: 'general_object',       # 通用障碍物/其他 (nuScenes标准)
}

NUM_CLASSES = len(SEMANTIC_CLASSES)
IGNORE_LABEL = 255

# 动态物体类别 (用于流场计算)
DYNAMIC_CLASSES = {2, 3, 4, 5, 6, 7, 9, 10}  # bicycle ~ pedestrian ~ truck

# ========== CARLA语义标签映射到18类 ==========
CARLA_TO_OCCUPANCY = {
    # 车辆类 (0 = Unlabeled, 10 = Vehicles)
    10: 4,   # vehicle.* → car (默认)

    # 行人类 (4 = Pedestrian)
    4: 7,    # walker.pedestrian.* → pedestrian

    # 道路类 (7 = Road, 6 = RoadLines)
    7: 11,   # 路面 → driveable_surface
    6: 11,   # 道路标线 → driveable_surface

    # 人行道/停车区 (8 = Sidewalk, 9 = Parking)
    8: 13,   # sidewalk → sidewalk
    9: 12,   # parking → other_flat

    # 建筑物 (1 = Buildings)
    1: 15,   # building → manmade

    # 围栏/墙 (2 = Fences, 3 = Other, 5 = Poles, 12 = Walls)
    2: 1,    # fence → barrier
    3: 1,    # other → barrier (保守)
    5: 1,    # poles → barrier
    12: 1,   # wall → barrier

    # 植被 (21 = Vegetation)
    21: 16,  # vegetation → vegetation

    # 地形 (22 = Terrain)
    22: 14,  # terrain → terrain

    # 天空 (23 = Sky)
    23: 0,   # sky → empty

    # 交通设施 (18 = TrafficSigns, 19 = TrafficLight)
    18: 8,   # traffic sign → traffic_cone (近似)
    19: 8,   # traffic light → traffic_cone (近似)

    # 其他 (默认)
    0: 0,    # unlabeled → empty
}

# 强制保留的地面/标线类别 (不参与可见性过滤，即使LiDAR没有击中也保留)
# 11:Driveable, 12:OtherFlat, 13:Sidewalk, 14:Terrain
# 2026-08-27 修复: 曾经多了一个 "6"。这是旧版CARLA语义标签(RoadLines=6，见本文件
# CARLA_TO_OCCUPANCY[6]=11)的输入编号，误当成了18类occupancy的输出编号写进这里——
# 18类里 6 是 motorcycle，不是 RoadLine（RoadLines早就映射到11了，不需要单独再列）。
# 这个杂质位让所有摩托车(label=6)被visibility_filter_simple.py的地面保护逻辑
# 无条件强制保留，完全绕过了"不可见就归free"的可见性过滤规则。
GROUND_LABELS = [11, 12, 13, 14]

# ========== CARLA 语义标签 → Occupancy 映射（与 dense_occupancy_collection 对齐）==========
# 用于 ground_truth_voxel_generator 的地面查询，消除对 dense_occupancy_collection 包的依赖
CARLA_TO_OCCUPANCY_MAPPING = {
    0: 0,   # NONE -> free
    1: 11,  # Roads -> driveable_surface
    2: 13,  # Sidewalks -> sidewalk
    3: 15,  # Buildings -> manmade
    4: 15,  # Walls -> manmade
    5: 1,   # Fences -> barrier
    6: 15,  # Poles -> manmade
    7: 8,   # TrafficLight -> traffic_cone
    8: 8,   # TrafficSigns -> traffic_cone
    9: 16,  # Vegetation -> vegetation
    10: 14, # Terrain -> terrain
    11: 0,  # Sky -> free
    12: 7,  # Pedestrians -> pedestrian
    13: 7,  # Rider -> pedestrian
    14: 4,  # Car -> car
    15: 10, # Truck -> truck
    16: 3,  # Bus -> bus
    17: 3,  # Train -> bus
    18: 6,  # Motorcycle -> motorcycle
    19: 2,  # Bicycle -> bicycle
    20: 17, # Static -> general_object
    21: 17, # Dynamic -> general_object
    22: 17, # Other -> general_object
    23: 12, # Water -> other_flat
    24: 11, # RoadLines -> driveable_surface
    25: 12, # Ground -> other_flat
    26: 15, # Bridge -> manmade
    27: 11, # RailTrack -> driveable_surface
    28: 1,  # GuardRail -> barrier
    29: 17, # Rock -> general_object
}

# Occupancy 标签名称（18类，与 dense_occupancy_collection 完全对齐）
OCCUPANCY_LABELS = [
    'free',                 # 0
    'barrier',              # 1
    'bicycle',              # 2
    'bus',                  # 3
    'car',                  # 4
    'construction_vehicle', # 5
    'motorcycle',           # 6
    'pedestrian',           # 7
    'traffic_cone',         # 8
    'trailer',              # 9
    'truck',                # 10
    'driveable_surface',    # 11
    'other_flat',           # 12
    'sidewalk',             # 13
    'terrain',              # 14
    'manmade',              # 15
    'vegetation',           # 16
    'general_object',       # 17
]

# 可视化颜色（与 occupancy_viewer/viewer.js 保持一致）
OCCUPANCY_COLORS = [
    (0, 0, 0),        # 0: free
    (200, 200, 200),  # 1: barrier
    (255, 215, 0),    # 2: bicycle
    (255, 99, 71),    # 3: bus
    (255, 140, 0),    # 4: car
    (255, 165, 0),    # 5: construction_vehicle
    (255, 20, 147),   # 6: motorcycle
    (255, 0, 0),      # 7: pedestrian
    (255, 255, 0),    # 8: traffic_cone
    (65, 105, 225),   # 9: trailer
    (0, 0, 255),      # 10: truck
    (80, 80, 80),     # 11: driveable_surface
    (120, 120, 120),  # 12: other_flat
    (160, 160, 160),  # 13: sidewalk
    (139, 69, 19),    # 14: terrain
    (220, 220, 220),  # 15: manmade
    (34, 139, 34),    # 16: vegetation
    (255, 0, 255),    # 17: general_object
]

# ========== Actor类型映射 (更细粒度) ==========
ACTOR_TYPE_MAPPING = {
    # 车辆细分
    'vehicle.car': 4,              # car
    'vehicle.truck': 10,           # truck
    'vehicle.bus': 3,              # bus
    'vehicle.trailer': 9,          # trailer
    'vehicle.motorcycle': 6,       # motorcycle
    'vehicle.bicycle': 2,          # bicycle
    'vehicle.construction': 5,     # construction_vehicle

    # 行人
    'walker.pedestrian': 7,        # pedestrian

    # 交通设施
    'static.prop.trafficcone': 8,  # traffic_cone
    'static.prop.trafficwarning': 8,

    # 默认
    'vehicle.': 4,                 # 未知车辆默认 car
    'walker.': 7,                  # 未知行人
    'static.': 1,                  # 静态物体默认 barrier
}

# ========== Depth相机配置 (用于精确体素生成) ==========
# 6路 CubeMap 深度相机配置 (来自 dense_occupancy_collection)
# Front, Right, Back, Left, Up, Down
# 统一高度 Z=2.2m (透波且不被车身遮挡)
# 垂直 FOV 60度 (上30, 下30)
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

# 旧配置保留 (兼容性)
DEPTH_CAMERA_CONFIGS = DEPTH_CAMERA_CONFIG['cameras']
DEPTH_CAMERA_POSITION = {'x': 0.0, 'y': 0.0, 'z': 2.2}
DEPTH_IMAGE_SIZE = 512
MAX_DEPTH = 100.0

# ========== 语义激光雷达配置 (新增) ==========
# 垂直 FOV 计算: 匹配相机最大垂直 FOV (约 110度)
# 相机 Max HFOV = 120 (Rear), Aspect Ratio = 0.75 -> VFOV ~ 105度
# 设置 range=120度 (+60/-60) 以确保覆盖
SEMANTIC_LIDAR_CONFIG = {
    'channels': 256,             # 256 线 (高密度)
    'points_per_second': 2000000,  # 每秒 200 万点
    'rotation_frequency': 20,    # 20Hz 旋转 (与仿真同步)
    'range': 100.0,              # 100 米探测范围
    'upper_fov': 45.0,           # 上视角 45°
    'lower_fov': -45.0,          # 下视角 -45°
    'horizontal_fov': 360.0,     # 水平 360°

    # 安装位置 (相对车辆)
    # 尽量接近相机中心高度 (约1.4-1.6m)，但必须高于车顶以免被遮挡
    # Model 3 车高约 1.44m，B柱相机 1.7m
    # 设置 Z=1.0m (用户指定)
    'position': {'x': 0.0, 'y': 0.0, 'z': 1.0}, 
    'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0},
}

print(f"[OccupancyConfig] 体素空间: {GRID_SIZE[0]}×{GRID_SIZE[1]}×{GRID_SIZE[2]} = {GRID_SIZE[0]*GRID_SIZE[1]*GRID_SIZE[2]} 体素")
print(f"[OccupancyConfig] 空间范围: X=[{X_RANGE[0]}, {X_RANGE[1]}] Y=[{Y_RANGE[0]}, {Y_RANGE[1]}] Z=[{Z_RANGE[0]}, {Z_RANGE[1]}]")
print(f"[OccupancyConfig] 分辨率: {RESOLUTION}m/体素")
print(f"[OccupancyConfig] 语义类别: {NUM_CLASSES}类")

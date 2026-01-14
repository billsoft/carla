"""
OccNetV3 体素配置 - 400x400x32 体素网格
用户自定义高分辨率配置 (0.2m分辨率, 80m×80m×6.4m覆盖范围)
"""

# ========== 体素空间定义 ==========
# 车辆坐标系 (右手系): X前 Y左 Z上
# 原点: 车辆后轴中心地面 (Z=0)

X_RANGE = [-40.0, 40.0]  # 前后各40米 (总80米)
Y_RANGE = [-40.0, 40.0]  # 左右各40米 (总80米)
Z_RANGE = [-1.0, 5.4]    # 地下1米到地上5.4米 (总6.4米)

RESOLUTION = 0.2  # 每个体素边长 0.2米 (20cm)

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
SEMANTIC_LIDAR_CONFIG = {
    'channels': 64,              # 64 线
    'points_per_second': 1200000,  # 每秒 120 万点
    'rotation_frequency': 10,    # 10Hz 旋转 (与仿真同步)
    'range': 100.0,              # 100 米探测范围
    'upper_fov': 15.0,           # 上视角 15°
    'lower_fov': -25.0,          # 下视角 -25°

    # 安装位置 (相对车辆)
    'position': {'x': 0.0, 'y': 0.0, 'z': 2.5},  # 车顶中央
    'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0},
}

print(f"[OccupancyConfig] 体素空间: {GRID_SIZE[0]}×{GRID_SIZE[1]}×{GRID_SIZE[2]} = {GRID_SIZE[0]*GRID_SIZE[1]*GRID_SIZE[2]} 体素")
print(f"[OccupancyConfig] 空间范围: X=[{X_RANGE[0]}, {X_RANGE[1]}] Y=[{Y_RANGE[0]}, {Y_RANGE[1]}] Z=[{Z_RANGE[0]}, {Z_RANGE[1]}]")
print(f"[OccupancyConfig] 分辨率: {RESOLUTION}m/体素")
print(f"[OccupancyConfig] 语义类别: {NUM_CLASSES}类")

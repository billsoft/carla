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

# 强制保留的地面/标线类别 (不参与可见性过滤，即使LiDAR没有击中也保留)
# 11:Driveable, 12:OtherFlat, 13:Sidewalk, 14:Terrain
# 2026-08-27 修复: 曾经多了一个 "6"。这是旧版CARLA语义标签(RoadLines=6，见本文件
# CARLA_TO_OCCUPANCY[6]=11)的输入编号，误当成了18类occupancy的输出编号写进这里——
# 18类里 6 是 motorcycle，不是 RoadLine（RoadLines早就映射到11了，不需要单独再列）。
# 这个杂质位让所有摩托车(label=6)被visibility_filter_simple.py的地面保护逻辑
# 无条件强制保留，完全绕过了"不可见就归free"的可见性过滤规则。
GROUND_LABELS = [11, 12, 13, 14]

# 真值语义版本号，写进 calibration/intrinsics.json 顶层 `label_version`。
# 1 (缺省): 交通灯/标志→traffic_cone(8)，地面填充里车道线→8、路缘→barrier(1)
# 2 (2026-10-14): 交通灯/标志/杆/墙→manmade(15)，车道线→driveable(11)，路缘→sidewalk(13)；
#    新增 mask_camera/<sid>.npy 相机可见性掩码；静态物体 AABB 里相机视线穿过的部分雕成 free
#    (camera_visibility.carve_static)；flow 改为物体自身速度 (不再减自车速度)。
#    两个版本的类别/流场含义不同，不要混在一个训练集里。
LABEL_VERSION = 2

# ========== CARLA CityObjectLabel 编号 → Occupancy 映射 ==========
# ground_truth_voxel_generator 给 world.get_environment_objects() 的静态环境物体分类用。
# (2026-10-14 删除了从未被引用的旧版语义 ID 映射 CARLA_TO_OCCUPANCY、ACTOR_TYPE_MAPPING 和
#  dense_occupancy_collection 时代的 6/8 路 cubemap 深度相机配置 DEPTH_CAMERA_CONFIG*)
CARLA_TO_OCCUPANCY_MAPPING = {
    0: 0,   # NONE -> free
    1: 11,  # Roads -> driveable_surface
    2: 13,  # Sidewalks -> sidewalk
    3: 15,  # Buildings -> manmade
    4: 15,  # Walls -> manmade
    5: 1,   # Fences -> barrier
    6: 15,  # Poles -> manmade
    7: 15,  # TrafficLight -> manmade (label_version 2 起；之前错标为 traffic_cone)
    8: 15,  # TrafficSigns -> manmade
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

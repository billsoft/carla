"""
Actor类型到17分类Occupancy的完整映射表
基于业界标准（nuScenes）的17类分类体系
"""

import carla

# ============================================================================
# 17类Occupancy标签定义（业界标准）
# ============================================================================
OCCUPANCY_LABELS = {
    0: 'free',                  # 自由空间/空气
    1: 'barrier',               # 隔离栏/护栏
    2: 'bicycle',               # 自行车
    3: 'bus',                   # 公交车
    4: 'car',                   # 小汽车
    5: 'construction_vehicle',  # 工程车
    6: 'motorcycle',            # 摩托车
    7: 'pedestrian',            # 行人
    8: 'traffic_cone',          # 交通锥桶
    9: 'trailer',               # 拖车
    10: 'truck',                # 卡车
    11: 'driveable_surface',    # 可行驶路面
    12: 'other_flat',           # 其他平坦表面
    13: 'sidewalk',             # 人行道
    14: 'terrain',              # 地形（草地、泥地等）
    15: 'manmade',              # 人造物体（建筑、标志等）
    16: 'vegetation',           # 植被
    17: 'general_object',       # 通用障碍物/其他
}

# 推荐可视化颜色 (RGB)
OCCUPANCY_COLORS = {
    0: (0, 0, 0),           # free - 黑色/透明
    1: (112, 128, 144),     # barrier - 灰蓝色
    2: (255, 61, 99),       # bicycle - 粉红色
    3: (220, 20, 60),       # bus - 深红色
    4: (255, 158, 0),       # car - 橙色
    5: (233, 150, 70),      # construction_vehicle - 土黄色
    6: (255, 0, 255),       # motorcycle - 品红色
    7: (30, 144, 255),      # pedestrian - 道奇蓝（鲜艳蓝色）
    8: (255, 127, 80),      # traffic_cone - 珊瑚橙
    9: (255, 140, 0),       # trailer - 暗橙色
    10: (180, 165, 180),    # truck - 紫灰色
    11: (128, 64, 128),     # driveable_surface - 深紫色
    12: (244, 35, 232),     # other_flat - 洋红色
    13: (107, 142, 35),     # sidewalk - 橄榄绿
    14: (152, 251, 152),    # terrain - 淡绿色
    15: (70, 70, 70),       # manmade - 深灰色
    16: (0, 255, 0),        # vegetation - 绿色
    17: (255, 255, 255),    # general_object - 白色
}

# ============================================================================
# CARLA CityObjectLabel 到 Occupancy 的映射
# 用于静态环境对象（建筑、植被、交通标志等）
# ============================================================================
CITY_OBJECT_TO_OCCUPANCY = {
    carla.CityObjectLabel.Buildings: 15,       # 建筑 -> manmade
    carla.CityObjectLabel.Fences: 1,           # 围栏 -> barrier
    carla.CityObjectLabel.Other: 17,           # 其他 -> general_object
    carla.CityObjectLabel.Pedestrians: 7,      # 行人 -> pedestrian (通常不会出现在get_level_bbs中)
    carla.CityObjectLabel.Poles: 15,           # 杆 -> manmade
    carla.CityObjectLabel.RoadLines: 11,       # 道路标线 -> driveable_surface
    carla.CityObjectLabel.Roads: 11,           # 道路 -> driveable_surface
    carla.CityObjectLabel.Sidewalks: 13,       # 人行道 -> sidewalk
    carla.CityObjectLabel.TrafficSigns: 15,    # 交通标志 -> manmade
    carla.CityObjectLabel.Vegetation: 16,      # 植被 -> vegetation
    # carla.CityObjectLabel.Vehicles: 4,         # 车辆 -> car (通常不会出现在get_level_bbs中)
    carla.CityObjectLabel.Walls: 15,           # 墙 -> manmade
    carla.CityObjectLabel.Sky: 0,              # 天空 -> free
    carla.CityObjectLabel.Ground: 12,          # 地面 -> other_flat
    carla.CityObjectLabel.Bridge: 15,          # 桥 -> manmade
    carla.CityObjectLabel.RailTrack: 17,       # 铁轨 -> general_object
    carla.CityObjectLabel.GuardRail: 1,        # 护栏 -> barrier
    carla.CityObjectLabel.TrafficLight: 15,    # 交通灯 -> manmade
    carla.CityObjectLabel.Static: 17,          # 静态物体 -> general_object
    carla.CityObjectLabel.Dynamic: 17,         # 动态物体 -> general_object
    carla.CityObjectLabel.Water: 12,           # 水 -> other_flat
    carla.CityObjectLabel.Terrain: 14,         # 地形 -> terrain
}

# ============================================================================
# CARLA Actor type_id 到 Occupancy 的映射
# 用于动态对象（车辆、行人等）
# ============================================================================

# 车辆类型映射（基于type_id字符串匹配）
VEHICLE_TYPE_PATTERNS = {
    # Car (4)
    'car': [
        'vehicle.audi',
        'vehicle.bmw',
        'vehicle.chevrolet',
        'vehicle.citroen',
        'vehicle.dodge.charger',
        'vehicle.ford.mustang',
        'vehicle.jeep',
        'vehicle.lincoln',
        'vehicle.mercedes',
        'vehicle.mini',
        'vehicle.nissan',
        'vehicle.seat',
        'vehicle.tesla',
        'vehicle.toyota',
    ],

    # Bus (3)
    'bus': [
        'vehicle.volkswagen.t2',  # VW面包车/小巴
    ],

    # Truck (10)
    'truck': [
        'vehicle.carlamotors.carlacola',  # 卡车
        'vehicle.carlamotors.firetruck',  # 消防车
        'vehicle.ford.ambulance',         # 救护车
        'vehicle.mercedes.sprinter',      # 货车
        'vehicle.tesla.cybertruck',       # 赛博卡车
    ],

    # Bicycle (2)
    'bicycle': [
        'vehicle.bh.crossbike',
        'vehicle.diamondback.century',
        'vehicle.gazelle.omafiets',
    ],

    # Motorcycle (6)
    'motorcycle': [
        'vehicle.harley-davidson',
        'vehicle.kawasaki',
        'vehicle.vespa',
        'vehicle.yamaha',
    ],
}

# 行人类型（walker）
WALKER_PATTERNS = {
    'pedestrian': [  # 7
        'walker.pedestrian',
    ],
}

# 交通设施（props）
PROP_PATTERNS = {
    'traffic_cone': [  # 8
        'static.prop.trafficcone',
        'static.prop.constructioncone',
    ],
    'barrier': [  # 1
        'static.prop.barrier',
        'static.prop.chainbarrier',
        'static.prop.streetbarrier',
    ],
    'general_object': [  # 17
        'static.prop.bin',           # 垃圾桶
        'static.prop.box',           # 箱子
        'static.prop.briefcase',     # 公文包
        'static.prop.brokentile',    # 碎瓦片
        'static.prop.busstop',       # 公交站牌
        'static.prop.container',     # 集装箱
        'static.prop.creasedbox',    # 压扁的箱子
        'static.prop.dirtdebris',    # 泥土碎屑
        'static.prop.garbage',       # 垃圾
        'static.prop.plasticbag',    # 塑料袋
        'static.prop.trashcan',      # 垃圾桶
    ],
    'manmade': [  # 15
        'static.prop.bench',         # 长椅
        'static.prop.fountain',      # 喷泉
        'static.prop.mailbox',       # 邮箱
        'static.prop.phonebox',      # 电话亭
        'static.prop.streetsign',    # 街道标志
        'static.prop.trafficwarning',# 交通警告标志
        'static.prop.table',         # 桌子
        'static.prop.chair',         # 椅子
    ],
    'vegetation': [  # 16
        'static.prop.plant',         # 植物
        'static.prop.bush',          # 灌木
    ],
}

# ============================================================================
# 映射函数
# ============================================================================

def get_occupancy_label_from_actor(actor):
    """
    根据Actor的type_id和semantic_tags确定其Occupancy标签

    Args:
        actor: carla.Actor对象

    Returns:
        int: Occupancy标签ID (0-17)
    """
    type_id = actor.type_id.lower()

    # 1. 优先使用semantic_tags（如果可用）
    if hasattr(actor, 'semantic_tags') and actor.semantic_tags:
        semantic_tag = actor.semantic_tags[0]

        # CARLA语义标签到Occupancy的直接映射
        semantic_mapping = {
            4: 7,   # Pedestrian -> pedestrian
            10: 4,  # Vehicles -> car (默认，后续可细分)
        }

        if semantic_tag in semantic_mapping:
            occupancy_label = semantic_mapping[semantic_tag]

            # 车辆类型细分（基于type_id）
            if occupancy_label == 4:  # 如果是车辆，进一步细分
                occupancy_label = _classify_vehicle(type_id)

            return occupancy_label

    # 2. 基于type_id匹配

    # 行人
    for pattern_list in WALKER_PATTERNS.values():
        for pattern in pattern_list:
            if pattern in type_id:
                return 7  # pedestrian

    # 车辆
    vehicle_label = _classify_vehicle(type_id)
    if vehicle_label != 4:  # 如果不是默认的car，说明匹配成功
        return vehicle_label

    # Props
    for label_name, pattern_list in PROP_PATTERNS.items():
        for pattern in pattern_list:
            if pattern in type_id:
                label_id = next((k for k, v in OCCUPANCY_LABELS.items() if v == label_name), 17)
                return label_id

    # 3. 兜底：如果是vehicle开头，返回car；如果是walker，返回pedestrian；其他返回general_object
    if 'vehicle' in type_id:
        return 4  # car
    elif 'walker' in type_id:
        return 7  # pedestrian
    else:
        return 17  # general_object


def _classify_vehicle(type_id):
    """
    细分车辆类型

    Args:
        type_id: str, actor的type_id (小写)

    Returns:
        int: Occupancy标签ID
    """
    # Bus
    for pattern in VEHICLE_TYPE_PATTERNS['bus']:
        if pattern in type_id:
            return 3  # bus

    # Truck
    for pattern in VEHICLE_TYPE_PATTERNS['truck']:
        if pattern in type_id:
            return 10  # truck

    # Bicycle
    for pattern in VEHICLE_TYPE_PATTERNS['bicycle']:
        if pattern in type_id:
            return 2  # bicycle

    # Motorcycle
    for pattern in VEHICLE_TYPE_PATTERNS['motorcycle']:
        if pattern in type_id:
            return 6  # motorcycle

    # 默认：Car
    return 4  # car


def get_occupancy_label_from_city_object(city_object_label):
    """
    根据CityObjectLabel获取Occupancy标签

    Args:
        city_object_label: carla.CityObjectLabel枚举值

    Returns:
        int: Occupancy标签ID (0-17)
    """
    return CITY_OBJECT_TO_OCCUPANCY.get(city_object_label, 17)  # 默认为general_object


def get_occupancy_color(label_id):
    """
    获取Occupancy标签对应的可视化颜色

    Args:
        label_id: int, Occupancy标签ID (0-17)

    Returns:
        tuple: (R, G, B) 颜色值
    """
    return OCCUPANCY_COLORS.get(label_id, (255, 255, 255))  # 默认白色


def get_occupancy_name(label_id):
    """
    获取Occupancy标签名称

    Args:
        label_id: int, Occupancy标签ID (0-17)

    Returns:
        str: 标签名称
    """
    return OCCUPANCY_LABELS.get(label_id, 'unknown')


# ============================================================================
# 导出变量（为了向后兼容）
# ============================================================================

# 与原有occupancy_config.py兼容的变量名
CARLA_TO_OCCUPANCY_MAPPING = CITY_OBJECT_TO_OCCUPANCY

# 导出标签列表（按ID排序）
OCCUPANCY_LABELS_LIST = [OCCUPANCY_LABELS[i] for i in range(18)]
OCCUPANCY_COLORS_LIST = [OCCUPANCY_COLORS[i] for i in range(18)]

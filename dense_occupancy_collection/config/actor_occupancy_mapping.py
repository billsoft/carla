"""
CARLA Actor类型到17分类Occupancy的完整映射配置
基于业界标准（nuScenes）的17类分类体系

映射规则：
1. 优先使用 type_id 字符串匹配（最准确）
2. 兜底使用 semantic_tags（CARLA语义标签）
3. 未知类型默认为 general_object (17)
"""

# ============================================================================
# 17类Occupancy标签定义（业界标准 - nuScenes）
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

# 可视化颜色（RGB格式）
OCCUPANCY_COLORS = {
    0: (0, 0, 0),           # free - 黑色
    1: (112, 128, 144),     # barrier - 灰蓝色
    2: (255, 61, 99),       # bicycle - 粉红色
    3: (220, 20, 60),       # bus - 深红色
    4: (255, 158, 0),       # car - 橙色
    5: (233, 150, 70),      # construction_vehicle - 土黄色
    6: (255, 0, 255),       # motorcycle - 品红色
    7: (30, 144, 255),      # pedestrian - 道奇蓝
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
# CARLA Actor type_id 到 Occupancy 的映射（1对多关系）
# 这是最准确的映射方式，基于实际的 type_id 字符串
# ============================================================================

# 车辆类型映射（基于实际Town10HD_Opt场景查询结果）
VEHICLE_MAPPING = {
    # Bus (3) - 公交车/面包车
    3: [
        'vehicle.volkswagen.t2',
        'vehicle.fuso.mitsubishi',  # 三菱Fuso (semantic_tag=16=Bus)
        'vehicle.mitsubishi.fusorosa', # 别名
    ],

    # Truck (10) - 卡车/货车/特种车辆
    10: [
        'vehicle.carlacola.actors',      # 可乐卡车 (semantic_tag=15=Truck)
        'vehicle.firetruck.actors',      # 消防车 (semantic_tag=15=Truck)
        'vehicle.ambulance.ford',        # 救护车 (semantic_tag=15=Truck)
        'vehicle.sprinter.mercedes',     # 奔驰货车 (semantic_tag=15=Truck)
        'vehicle.tesla.cybertruck',      # 赛博卡车
        'vehicle.carlamotors.carlacola', # 别名
        'vehicle.carlamotors.firetruck', # 别名
        'vehicle.ford.ambulance',        # 别名
        'vehicle.mercedes.sprinter',     # 别名
    ],

    # Bicycle (2) - 自行车
    2: [
        'vehicle.bh.crossbike',
        'vehicle.diamondback.century',
        'vehicle.gazelle.omafiets',
    ],

    # Motorcycle (6) - 摩托车
    6: [
        'vehicle.harley-davidson.low_rider',
        'vehicle.kawasaki.ninja',
        'vehicle.vespa.zx125',
        'vehicle.yamaha.yzf',
    ],

    # Car (4) - 小汽车（默认）
    # 所有其他vehicle.*都归为car，所以不需要显式列出
    # 实际场景中的car类型 (semantic_tag=14=Car):
    # - vehicle.dodge.charger
    # - vehicle.dodgecop.charger
    # - vehicle.mini.cooper
    # - vehicle.nissan.patrol
    # - vehicle.taxi.ford
    # - vehicle.ue4.audi.tt
    # - vehicle.ue4.bmw.grantourer
    # - vehicle.ue4.chevrolet.impala
    # - vehicle.ue4.ford.crown
    # - vehicle.ue4.ford.mustang
    # - vehicle.ue4.mercedes.ccc
}

# 行人类型映射
WALKER_MAPPING = {
    # Pedestrian (7) - 行人
    7: [
        'walker.pedestrian.0001',
        'walker.pedestrian.0002',
        'walker.pedestrian.0003',
        'walker.pedestrian.0004',
        'walker.pedestrian.0005',
        'walker.pedestrian.0006',
        'walker.pedestrian.0007',
        'walker.pedestrian.0008',
        'walker.pedestrian.0009',
        'walker.pedestrian.0010',
        'walker.pedestrian.0011',
        'walker.pedestrian.0012',
        'walker.pedestrian.0013',
        'walker.pedestrian.0014',
        # ... 所有 walker.pedestrian.* 都是行人
    ],
}

# Props（道具/静态物体）映射
PROP_MAPPING = {
    # Traffic Cone (8) - 交通锥桶
    8: [
        'static.prop.trafficcone01',
        'static.prop.trafficcone02',
        'static.prop.constructioncone',
    ],

    # Barrier (1) - 隔离栏/护栏
    1: [
        'static.prop.streetbarrier',
        'static.prop.chainbarrier',
        'static.prop.chainbarriergate',
        'static.prop.warningconstruction',
        'static.prop.warningaccident',
        'static.prop.chainbarrierend', # Add BarrierEnd
    ],

    # Manmade (15) - 人造物体
    15: [
        'static.prop.bench01',
        'static.prop.bench02',
        'static.prop.bench03',
        'static.prop.fountain',
        'static.prop.streetsign',
        'static.prop.streetsign01',
        'static.prop.streetsign04',
        'static.prop.trafficwarning',
        'static.prop.mailbox',
        'static.prop.phonebox',
        'static.prop.table',
        'static.prop.busstop',
        'static.prop.atm',           # Add ATM
        'static.prop.advertisement', # Add Ad
        'static.prop.vendingmachine',# Add Vending
        'static.prop.doghouse',      # Add Doghouse
        'static.prop.barbeque',      # Add BBQ
    ],

    # Vegetation (16) - 植被
    16: [
        'static.prop.bush01',
        'static.prop.bush02',
        'static.prop.bush03',
        'static.prop.plant01',
        'static.prop.plant02',
        'static.prop.plant03',
        'static.prop.plantpot04',    # Add Plantpot
    ],

    # General Object (17) - 通用障碍物/其他
    17: [
        'static.prop.bin',
        'static.prop.trashcan01',
        'static.prop.trashcan02',
        'static.prop.trashcan03',
        'static.prop.trashcan04',
        'static.prop.trashcan05',
        'static.prop.container',
        'static.prop.box01',
        'static.prop.box02',
        'static.prop.box03',
        'static.prop.creasedbox01',
        'static.prop.creasedbox02',
        'static.prop.creasedbox03',
        'static.prop.briefcase',
        'static.prop.garbage01',
        'static.prop.garbage02',
        'static.prop.garbage03',
        'static.prop.garbage04',
        'static.prop.garbage05',
        'static.prop.garbage06',
        'static.prop.plasticbag',
        'static.prop.dirtdebris01',
        'static.prop.dirtdebris02',
        'static.prop.dirtdebris03',
        'static.prop.brokentile01',
        'static.prop.brokentile02',
        'static.prop.brokentile03',
        'static.prop.brokentile04',
        'static.prop.colacan',       # Add Cola
        'static.prop.recycleorganic',# Add Recycle
        'static.prop.platformgarbage01', # Add Garbage
        'static.prop.dumpster',      # Add Dumpster
        'static.prop.bike helmet',   # Add Helmet
    ],
}

# ============================================================================
# CARLA CityObjectLabel 到 Occupancy 的映射
# 用于静态环境对象（建筑、植被、道路等）
# ============================================================================

import carla

CITY_OBJECT_MAPPING = {
    # 基于Town10HD_Opt实际场景的CityObjectLabel映射
    carla.CityObjectLabel.NONE: 0,             # NONE (110个) -> free
    carla.CityObjectLabel.Roads: 11,           # 道路 (4个) -> driveable_surface
    carla.CityObjectLabel.Sidewalks: 13,       # 人行道 (269个) -> sidewalk
    carla.CityObjectLabel.Buildings: 15,       # 建筑 (48369个) -> manmade
    carla.CityObjectLabel.Walls: 15,           # 墙 (48个) -> manmade
    carla.CityObjectLabel.Fences: 1,           # 围栏 (1767个) -> barrier
    carla.CityObjectLabel.Poles: 15,           # 杆 (120个) -> manmade
    carla.CityObjectLabel.TrafficLight: 15,    # 交通灯 (62个) -> manmade
    carla.CityObjectLabel.TrafficSigns: 15,    # 交通标志 (147个) -> manmade
    carla.CityObjectLabel.Vegetation: 16,      # 植被 (4270个) -> vegetation
    carla.CityObjectLabel.Terrain: 14,         # 地形 (23个) -> terrain
    carla.CityObjectLabel.Car: 4,              # 车辆 (80个) -> car
    carla.CityObjectLabel.Truck: 10,           # 卡车 (45个) -> truck
    carla.CityObjectLabel.Bus: 3,              # 公交车 (5个) -> bus
    carla.CityObjectLabel.Motorcycle: 6,       # 摩托车 (8个) -> motorcycle
    carla.CityObjectLabel.Bicycle: 2,          # 自行车 (56个) -> bicycle
    carla.CityObjectLabel.Static: 17,          # 静态物体 (1223个) -> general_object
    carla.CityObjectLabel.Dynamic: 17,         # 动态物体 (489个) -> general_object
    carla.CityObjectLabel.Other: 17,           # 其他 (370个) -> general_object
    carla.CityObjectLabel.Water: 12,           # 水 (2个) -> other_flat
    carla.CityObjectLabel.RoadLines: 11,       # 道路标线 (4个) -> driveable_surface
    carla.CityObjectLabel.Ground: 12,          # 地面 (15个) -> other_flat
    carla.CityObjectLabel.Bridge: 15,          # 桥 (9个) -> manmade
    # carla.CityObjectLabel.Any: 不映射 (57495个，包含所有)

    # 以下在Town10HD_Opt中未出现，但保留以防其他地图使用
    carla.CityObjectLabel.Pedestrians: 7,      # 行人 -> pedestrian
    carla.CityObjectLabel.RailTrack: 17,       # 铁轨 -> general_object
    carla.CityObjectLabel.GuardRail: 1,        # 护栏 -> barrier
    carla.CityObjectLabel.Sky: 0,              # 天空 -> free
}

# ============================================================================
# 映射函数
# ============================================================================

def get_occupancy_label_from_type_id(type_id):
    """
    根据 type_id 获取 Occupancy 标签（基于精确匹配）

    Args:
        type_id: str, actor的type_id

    Returns:
        int or None: Occupancy标签ID (0-17)，如果未找到返回None
    """
    type_id_lower = type_id.lower()

    # 1. 检查行人
    if type_id_lower.startswith('walker.pedestrian'):
        return 7

    # 2. 检查车辆（精确匹配） - 使用小写比较
    for label_id, type_ids in VEHICLE_MAPPING.items():
        for tid in type_ids:
            if type_id_lower == tid.lower():
                return label_id

    # 3. 检查Props - 使用小写比较
    for label_id, type_ids in PROP_MAPPING.items():
        for tid in type_ids:
            if type_id_lower == tid.lower():
                return label_id

    # 4. 兜底：如果是vehicle开头但未匹配，默认为car
    if type_id_lower.startswith('vehicle.'):
        return 4  # car

    # 5. 兜底：如果是static.prop开头但未匹配，默认为general_object
    if type_id_lower.startswith('static.prop'):
        return 17  # general_object

    # 6. 未找到
    return None


def get_occupancy_label_from_actor(actor):
    """
    从Actor对象获取Occupancy标签（完整逻辑）

    Args:
        actor: carla.Actor对象

    Returns:
        int: Occupancy标签ID (0-17)
    """
    # 1. 优先使用type_id精确匹配
    label = get_occupancy_label_from_type_id(actor.type_id)
    if label is not None:
        # 调试日志
        import logging
        logging.info(f"Actor {actor.type_id} -> Label {label} ({OCCUPANCY_LABELS.get(label, 'unknown')})")
        return label

    # 2. 兜底：使用semantic_tags
    if hasattr(actor, 'semantic_tags') and actor.semantic_tags:
        sem_tag = actor.semantic_tags[0]
        # CARLA语义标签到Occupancy的基础映射
        sem_mapping = {
            4: 7,   # Pedestrian -> pedestrian
            10: 4,  # Vehicles -> car
        }
        if sem_tag in sem_mapping:
            return sem_mapping[sem_tag]

    # 3. 最终兜底
    return 17  # general_object


def get_occupancy_color(label_id):
    """获取Occupancy标签对应的颜色"""
    return OCCUPANCY_COLORS.get(label_id, (255, 255, 255))


def get_occupancy_name(label_id):
    """获取Occupancy标签名称"""
    return OCCUPANCY_LABELS.get(label_id, 'unknown')


# ============================================================================
# 导出为列表格式（用于兼容旧代码）
# ============================================================================

OCCUPANCY_LABELS_LIST = [OCCUPANCY_LABELS[i] for i in range(18)]
OCCUPANCY_COLORS_LIST = [OCCUPANCY_COLORS[i] for i in range(18)]

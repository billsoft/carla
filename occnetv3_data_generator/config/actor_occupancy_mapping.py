"""
CARLA Actor类型到18分类Occupancy的完整映射配置
基于业界标准（nuScenes + free类）的18类分类体系

映射规则：
1. 优先使用 type_id 字符串匹配（最准确）
2. 兜底使用 semantic_tags（CARLA语义标签）
3. 未知类型默认为 general_object (17)
"""

# ============================================================================
# 18类Occupancy标签和颜色定义（统一配置源）
# ⚠️ 所有标签和颜色定义已统一到 occupancy_config.py
# 此文件不再重复定义，避免配置冲突和颜色不一致问题
# ============================================================================

from config.occupancy_config import (
    OCCUPANCY_LABELS as OCCUPANCY_LABELS_LIST,
    OCCUPANCY_COLORS as OCCUPANCY_COLORS_LIST
)

# 转换为字典格式（兼容本文件中使用 .get() 的旧代码）
OCCUPANCY_LABELS = {i: OCCUPANCY_LABELS_LIST[i] for i in range(18)}
OCCUPANCY_COLORS = {i: OCCUPANCY_COLORS_LIST[i] for i in range(18)}

# ============================================================================
# CARLA Actor type_id 到 Occupancy 的映射（1对多关系）
# 这是最准确的映射方式，基于实际的 type_id 字符串
# ============================================================================

# 车辆类型映射（基于实际Town10HD_Opt场景查询结果）
VEHICLE_MAPPING = {
    # Truck (10)
    10: [
        'vehicle.carlamotors.carlacola',
        'vehicle.carlamotors.firetruck',
        'vehicle.ford.ambulance',
        'vehicle.mercedes.sprinter',
        'vehicle.tesla.cybertruck',
        'vehicle.miningtruck.miningtruck', 
        'vehicle.volkswagen.t2',
        
        # Scanned Real IDs
        'vehicle.ambulance.ford',
        'vehicle.carlacola.actors',
        'vehicle.firetruck.actors',
        'vehicle.sprinter.mercedes',
    ],
    
    # Construction Vehicle (5)
    5: [
        # currently mapped to truck or car
    ],

    # Bus (3)
    3: [
        'vehicle.mitsubishi.fusorosa',
        # Scanned Real IDs
        'vehicle.fuso.mitsubishi', # Was misclassified as Car
    ],
    
    # Motorcycle (6)
    6: [
        'vehicle.harley-davidson.low_rider',
        'vehicle.kawasaki.ninja',
        'vehicle.yamaha.yzf',
        'vehicle.vespa.zx125',
    ],

    # Bicycle (2)
    2: [
        'vehicle.bh.crossbike',
        'vehicle.diamondback.century',
        'vehicle.gazelle.omafiets',
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
# 注意: get_occupancy_label_from_type_id() 用的是开头的 `startswith('walker.pedestrian')`
# 整体判断（所有行人子型号统一映射到7），不会真的遍历这个字典逐条比对——这份列表只是
# 文档性质，列出蓝图库里实际存在的部分行人型号，新增/减少某个 walker.pedestrian.NNNN
# 型号不需要同步改这里。
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
    # Traffic Cone (8) - 只放可移动的锥桶/临时警示牌 (nuScenes 定义)
    8: [
        'static.prop.trafficcone01',
        'static.prop.trafficcone02',
        'static.prop.constructioncone',
        'static.prop.trafficwarning',
    ],

    # Barrier (1) - 隔离栏/护栏
    1: [
        'static.prop.streetbarrier',
        'static.prop.warningtape',
        'static.prop.chainbarrier',
        'static.prop.chainbarriergate',
        'static.prop.warningconstruction',
        'static.prop.warningaccident',
        'static.prop.chainbarrierend', # Add BarrierEnd
    ],

    # Manmade (15) - 人造物体
    15: [
        'static.prop.fountain',
        'static.prop.streetsign',
        'static.prop.streetsign01',
        'static.prop.streetsign04',
        'static.prop.advertisingboard',
    ],

    # Vegetation (16) - 植被
    16: [
        'static.prop.bush01',
        'static.prop.bush02',
        'static.prop.bush03',
        'static.prop.plant01',
        'static.prop.plant02',
        'static.prop.plant03',
        # 2026-08-27 用 survey_actor_types.py 核对蓝图库发现: 只列了 plantpot04，
        # 其余 plantpot01/02/03/05/06/07 之前掉进了 general_object 兜底——同一类
        # 道具(花盆，视觉上以盆栽植物为主)被不一致地分到两个类别，统一成vegetation。
        'static.prop.plantpot01',
        'static.prop.plantpot02',
        'static.prop.plantpot03',
        'static.prop.plantpot04',
        'static.prop.plantpot05',
        'static.prop.plantpot06',
        'static.prop.plantpot07',
    ],

    # General Object (17) - 通用障碍物/其他
    17: [
        'static.prop.ironplank',
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
# CARLA CityObjectLabel 编号 → Occupancy
# ============================================================================
# 唯一来源是 config/occupancy_config.CARLA_TO_OCCUPANCY_MAPPING (环境物体光栅化也用它)。2026-10-15 之前这里
# 另有一份以 carla.CityObjectLabel 枚举为键的表：(1) 与那份不一致 (RailTrack 17 vs 11)；(2) Actor 语义标签兜底
# 用 int 去查枚举键的 dict，永远查不到 (17 in CITY_OBJECT_MAPPING 为 False)，语义标签兜底从来没生效过。
# 现在统一为 int 键；用枚举查时先 int(label)。
from config.occupancy_config import CARLA_TO_OCCUPANCY_MAPPING

CITY_OBJECT_MAPPING = dict(CARLA_TO_OCCUPANCY_MAPPING)

# Actor 语义标签里优先认的类别 (车辆的 semantic_tags 可能同时带 Rider 等部件标签)
_PREFERRED_TAGS = (15, 16, 17, 14, 18, 19, 12)   # Truck, Bus, Train, Car, Motorcycle, Bicycle, Pedestrians

# ============================================================================
# 映射函数
# ============================================================================

def get_occupancy_label_from_type_id(type_id):
    """
    根据 type_id 精确匹配 Occupancy 标签 (行人前缀 / VEHICLE_MAPPING / PROP_MAPPING)。

    没有精确匹配时返回 None——**不再**在这里按 "vehicle." 前缀兜底成 car：那样外层永远轮不到语义标签，
    新资产里的卡车/公交会被悄悄标成小汽车。前缀兜底在 get_occupancy_label_from_actor 的最后一步。
    """
    type_id_lower = type_id.lower()

    if type_id_lower.startswith('walker.pedestrian'):
        return 7
    for label_id, type_ids in VEHICLE_MAPPING.items():
        for tid in type_ids:
            if type_id_lower == tid.lower():
                return label_id
    for label_id, type_ids in PROP_MAPPING.items():
        for tid in type_ids:
            if type_id_lower == tid.lower():
                return label_id
    return None


def _prefix_fallback(type_id):
    t = type_id.lower()
    if t.startswith('vehicle.'):
        return 4      # car
    if t.startswith('walker.'):
        return 7
    return 17         # general_object (含未收录的 static.prop.*)


# 兜底统计：{type_id: (标签, 来源)}，来源为 'semantic_tag' 或 'prefix'。每个 type_id 只打印一次
FALLBACK_LOG = {}


def get_occupancy_label_from_actor(actor):
    """
    Actor → Occupancy 标签。顺序：精确车型/道具映射 → 有效语义标签 → 前缀兜底。

    Returns:
        int: Occupancy标签ID (0-17)
    """
    label = get_occupancy_label_from_type_id(actor.type_id)
    if label is not None:
        return label

    tags = [int(t) for t in (getattr(actor, 'semantic_tags', None) or [])]
    picked = next((t for t in _PREFERRED_TAGS if t in tags), None)
    if picked is None:
        picked = next((t for t in tags if CITY_OBJECT_MAPPING.get(t, 0) != 0), None)
    if picked is not None:
        label, source = CITY_OBJECT_MAPPING[picked], 'semantic_tag'
    else:
        label, source = _prefix_fallback(actor.type_id), 'prefix'
    if actor.type_id not in FALLBACK_LOG:
        FALLBACK_LOG[actor.type_id] = (label, source)
        print(f"[类别兜底] type_id={actor.type_id!r} 没有精确映射，semantic_tags={tags} -> "
              f"{source} 得到 {label} ({OCCUPANCY_LABELS.get(label, '?')})；"
              f"确认无误后建议补进 VEHICLE_MAPPING/PROP_MAPPING")
    return label


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

"""
Actor类型普查脚本 - 枚举CARLA中"可能出现"的全部actor类型，核对18类occupancy映射表的覆盖度

用途：
    actor_occupancy_mapping.py 里的 type_id 映射表是手工维护的静态清单，容易和 CARLA
    实际提供的蓝图集合脱节（新增地图带新车型、CARLA版本升级加新蓝图等）。这个脚本枚举
    三类"全部actor来源"，逐一核对是否已被 actor_occupancy_mapping.py 显式覆盖：

    1. 蓝图库 (world.get_blueprint_library()) —— vehicle.*/walker.pedestrian.*/
       static.prop.*/traffic.* 全量可生成类型，与地图无关，是权威的"有多少种"来源。
    2. 当前地图已生成的traffic设施 actor (world.get_actors().filter('traffic.*'))——
       红绿灯/限速牌/停车让行标志是地图预置的，不是从蓝图库 spawn 出来的，必须单独查。
    3. 当前地图的静态环境物体 (world.get_environment_objects(CityObjectLabel.Any))——
       核对 CITY_OBJECT_MAPPING 的 CityObjectLabel 覆盖度。

    对每个 type_id/label，用 get_occupancy_label_from_type_id()/CITY_OBJECT_MAPPING
    判断落在哪一类：
    - explicit: 在 VEHICLE_MAPPING/WALKER_MAPPING/PROP_MAPPING 里被显式列出
    - fallback: 没被显式列出，靠 get_occupancy_label_from_type_id() 末尾的
      "vehicle.*默认car / static.prop.*默认general_object" 兜底
    - unmapped: 完全没有对应关系（函数返回 None，或 CityObjectLabel 不在
      CITY_OBJECT_MAPPING 里）

用法：
    python occnetv3_data_generator/survey_actor_types.py --host 127.0.0.1 --port 2000
"""

import sys
import os
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import carla

from config.actor_occupancy_mapping import (
    VEHICLE_MAPPING, WALKER_MAPPING, PROP_MAPPING, CITY_OBJECT_MAPPING,
    OCCUPANCY_LABELS, get_occupancy_label_from_type_id,
)


def explicit_type_ids():
    """VEHICLE_MAPPING/WALKER_MAPPING/PROP_MAPPING 里显式列出的 type_id（小写）集合"""
    out = set()
    for mapping in (VEHICLE_MAPPING, WALKER_MAPPING, PROP_MAPPING):
        for _, type_ids in mapping.items():
            for tid in type_ids:
                out.add(tid.lower())
    return out


def classify_blueprint(bp_id, explicit_set):
    label = get_occupancy_label_from_type_id(bp_id)
    if label is None:
        return 'unmapped', None
    if bp_id.lower() in explicit_set:
        return 'explicit', label
    return 'fallback', label


def survey_blueprint_library(world, explicit_set):
    bp_lib = world.get_blueprint_library()
    categories = {
        'vehicle': bp_lib.filter('vehicle.*'),
        'walker': bp_lib.filter('walker.pedestrian.*'),
        'prop': bp_lib.filter('static.prop.*'),
        'traffic_bp': bp_lib.filter('traffic.*'),
    }

    print("=" * 78)
    print("1. 蓝图库全量枚举 (world.get_blueprint_library())")
    print("=" * 78)

    report = defaultdict(list)
    for cat, bps in categories.items():
        print(f"\n[{cat}] 共 {len(bps)} 种蓝图")
        for bp in bps:
            status, label = classify_blueprint(bp.id, explicit_set)
            report[status].append((cat, bp.id, label))

    print(f"\n--- 汇总 ---")
    print(f"explicit (显式列在映射表里): {len(report['explicit'])}")
    print(f"fallback (靠前缀兜底，不在显式表里): {len(report['fallback'])}")
    print(f"unmapped (完全没有对应关系): {len(report['unmapped'])}")

    if report['fallback']:
        print(f"\n[fallback 明细] 靠前缀兜底的 type_id (按类别分组):")
        by_cat = defaultdict(list)
        for cat, tid, label in report['fallback']:
            by_cat[cat].append((tid, label))
        for cat, items in by_cat.items():
            print(f"  [{cat}] {len(items)} 个:")
            for tid, label in sorted(items):
                print(f"    {tid:45s} -> fallback label={label} ({OCCUPANCY_LABELS.get(label, '?')})")

    if report['unmapped']:
        print(f"\n[unmapped 明细] 完全没有对应关系的 type_id:")
        for cat, tid, label in sorted(report['unmapped']):
            print(f"    [{cat}] {tid}")

    return report


def survey_traffic_actors(world, explicit_set):
    print("\n" + "=" * 78)
    print("2. 当前地图已生成的 traffic.* Actor (world.get_actors().filter('traffic.*'))")
    print("=" * 78)

    traffic_actors = world.get_actors().filter('traffic.*')
    by_type = defaultdict(list)
    for a in traffic_actors:
        by_type[a.type_id].append(a)

    print(f"\n共 {len(traffic_actors)} 个 traffic actor，{len(by_type)} 种 type_id")

    unresolved = []
    for tid, actors in sorted(by_type.items()):
        sample = actors[0]
        tags = list(sample.semantic_tags) if hasattr(sample, 'semantic_tags') else []
        resolved_labels = sorted({CITY_OBJECT_MAPPING.get(int(t)) for t in tags if int(t) in CITY_OBJECT_MAPPING})
        unresolved_tags = sorted({int(t) for t in tags if int(t) not in CITY_OBJECT_MAPPING})
        status = 'OK' if resolved_labels and not unresolved_tags else 'GAP'
        if status == 'GAP':
            unresolved.append(tid)
        print(f"  [{status}] {tid:35s} 数量={len(actors):3d} semantic_tags={tags} "
              f"-> occupancy_labels={resolved_labels} 未覆盖tags={unresolved_tags}")

    if unresolved:
        print(f"\n[GAP] 以下 traffic type_id 的 semantic_tags 没有被 CITY_OBJECT_MAPPING 完全覆盖: {unresolved}")
    else:
        print("\n所有 traffic actor 的 semantic_tags 均能通过 CITY_OBJECT_MAPPING 兜底解析。")

    return by_type


def survey_environment_objects(world):
    print("\n" + "=" * 78)
    print("3. 静态环境物体 CityObjectLabel 覆盖度 (world.get_environment_objects(Any))")
    print("=" * 78)

    env_objs = world.get_environment_objects(carla.CityObjectLabel.Any)
    by_label = defaultdict(int)
    for obj in env_objs:
        by_label[obj.type] += 1

    print(f"\n共 {len(env_objs)} 个环境物体，{len(by_label)} 种 CityObjectLabel")
    unmapped = []
    for label, count in sorted(by_label.items(), key=lambda x: -x[1]):
        if label == carla.CityObjectLabel.Any:
            continue
        mapped = CITY_OBJECT_MAPPING.get(label)
        status = 'OK' if mapped is not None else 'GAP'
        if status == 'GAP':
            unmapped.append(label)
        occ_name = OCCUPANCY_LABELS.get(mapped, '?') if mapped is not None else '???'
        print(f"  [{status}] {str(label):20s} 数量={count:6d} -> occupancy={mapped} ({occ_name})")

    if unmapped:
        print(f"\n[GAP] 以下 CityObjectLabel 没有被 CITY_OBJECT_MAPPING 覆盖: {[str(l) for l in unmapped]}")
    else:
        print("\n所有出现过的 CityObjectLabel 均已被 CITY_OBJECT_MAPPING 覆盖。")


def main():
    parser = argparse.ArgumentParser(description="Actor类型普查：核对18类occupancy映射表的覆盖度")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()

    print(f"当前地图: {world.get_map().name}\n")

    explicit_set = explicit_type_ids()
    survey_blueprint_library(world, explicit_set)
    survey_traffic_actors(world, explicit_set)
    survey_environment_objects(world)


if __name__ == '__main__':
    main()

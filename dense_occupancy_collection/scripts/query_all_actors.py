#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查询CARLA场景中所有不重复的Actor类型和CityObjectLabel类型
用于建立Actor到17分类Occupancy的映射表
"""

import sys
import os
from pathlib import Path

# 添加项目路径
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'PythonAPI/carla'))
except IndexError:
    pass

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import carla
import time
import json
from collections import defaultdict


def query_all_actor_types(world):
    """查询场景中所有Actor的type_id和semantic_tags"""
    actor_info = defaultdict(set)

    actors = world.get_actors()
    print(f"\n总Actor数量: {len(actors)}")

    for actor in actors:
        type_id = actor.type_id

        # 获取semantic_tags
        semantic_tags = []
        if hasattr(actor, 'semantic_tags') and actor.semantic_tags:
            semantic_tags = list(actor.semantic_tags)

        # 存储
        actor_info[type_id].add(tuple(semantic_tags) if semantic_tags else ())

    return actor_info


def query_all_city_object_labels(world):
    """查询场景中所有CityObjectLabel类型"""
    city_label_info = {}

    # 遍历所有CityObjectLabel枚举值
    for label_name in dir(carla.CityObjectLabel):
        if label_name.startswith('_'):
            continue

        label_value = getattr(carla.CityObjectLabel, label_name)
        if not isinstance(label_value, int):
            continue

        try:
            bbs = world.get_level_bbs(label_value)
            count = len(bbs)
            if count > 0:
                city_label_info[label_name] = {
                    'value': label_value,
                    'count': count
                }
        except Exception as e:
            pass

    return city_label_info


def main():
    print("="*60)
    print("CARLA场景Actor类型查询工具")
    print("="*60)

    client = None
    world = None
    vehicle = None
    traffic_actors = []

    try:
        # 连接CARLA
        print("\n⏳ 连接CARLA服务器...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"✓ 已连接到CARLA: {world.get_map().name}")

        # 设置同步模式
        settings = world.get_settings()
        original_sync = settings.synchronous_mode
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # 创建hero车辆
        print("\n⏳ 生成hero车辆...")
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.*')[0]
        if vehicle_bp.has_attribute('role_name'):
            vehicle_bp.set_attribute('role_name', 'hero')
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        print(f"✓ Hero车辆已生成: {vehicle.type_id}")

        # 生成NPC
        print("\n⏳ 生成NPC...")
        tm_port = 8005
        traffic_manager = client.get_trafficmanager(tm_port)
        traffic_manager.set_synchronous_mode(True)

        vehicle_bps = bp_lib.filter('vehicle.*')
        vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute('number_of_wheels')) == 4]

        for i in range(min(50, len(spawn_points)-1)):
            bp = vehicle_bps[i % len(vehicle_bps)]
            npc_vehicle = world.try_spawn_actor(bp, spawn_points[i+1])
            if npc_vehicle:
                npc_vehicle.set_autopilot(True, tm_port)
                traffic_actors.append(npc_vehicle)

        print(f"✓ 已生成 {len(traffic_actors)} 辆NPC车辆")

        # 等待稳定
        print("\n⏳ 等待场景稳定...")
        for _ in range(20):
            world.tick()

        # 查询所有Actor类型
        print("\n" + "="*60)
        print("查询Actor类型...")
        print("="*60)

        actor_info = query_all_actor_types(world)

        print(f"\n发现 {len(actor_info)} 种不同的Actor类型:\n")

        # 按类型排序输出
        sorted_actors = sorted(actor_info.items())

        result_data = {
            'map_name': world.get_map().name,
            'total_actor_types': len(actor_info),
            'actor_types': []
        }

        for type_id, semantic_tags_set in sorted_actors:
            semantic_tags_list = [list(tags) for tags in semantic_tags_set]
            semantic_tags_list.sort()

            print(f"  - {type_id}")
            if semantic_tags_list and semantic_tags_list[0]:
                print(f"      semantic_tags: {semantic_tags_list}")

            result_data['actor_types'].append({
                'type_id': type_id,
                'semantic_tags': semantic_tags_list
            })

        # 查询所有CityObjectLabel类型
        print("\n" + "="*60)
        print("查询CityObjectLabel类型...")
        print("="*60)

        city_label_info = query_all_city_object_labels(world)

        print(f"\n发现 {len(city_label_info)} 种CityObjectLabel:\n")

        sorted_labels = sorted(city_label_info.items(), key=lambda x: x[1]['value'])

        result_data['city_object_labels'] = []

        for label_name, info in sorted_labels:
            print(f"  - {label_name} (value={info['value']}, count={info['count']})")
            result_data['city_object_labels'].append({
                'name': label_name,
                'value': info['value'],
                'count': info['count']
            })

        # 保存结果到JSON
        output_path = Path(__file__).parent.parent / 'config' / 'actor_types_query_result.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 查询结果已保存到: {output_path}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n" + "="*60)
        print("清理资源")
        print("="*60)

        if vehicle:
            vehicle.destroy()
            print("✓ Hero车辆已销毁")

        if traffic_actors:
            for actor in traffic_actors:
                actor.destroy()
            print(f"✓ {len(traffic_actors)} 个NPC已销毁")

        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            print("✓ 已恢复异步模式")

        print("\n✅ 查询完成!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试Actor到Occupancy的映射是否正确
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dense_occupancy_collection.config.actor_occupancy_mapping import (
    get_occupancy_label_from_type_id,
    get_occupancy_name,
    OCCUPANCY_LABELS
)

def test_vehicle_mapping():
    """测试车辆类型映射"""
    print("=" * 60)
    print("测试车辆类型映射")
    print("=" * 60)

    test_cases = [
        # (type_id, expected_label, expected_name)
        ('vehicle.ambulance.ford', 10, 'truck'),
        ('vehicle.carlacola.actors', 10, 'truck'),
        ('vehicle.firetruck.actors', 10, 'truck'),
        ('vehicle.sprinter.mercedes', 10, 'truck'),
        ('vehicle.fuso.mitsubishi', 3, 'bus'),
        ('vehicle.dodge.charger', 4, 'car'),
        ('vehicle.mini.cooper', 4, 'car'),
        ('vehicle.ue4.audi.tt', 4, 'car'),
    ]

    all_pass = True
    for type_id, expected_label, expected_name in test_cases:
        label = get_occupancy_label_from_type_id(type_id)
        name = get_occupancy_name(label)

        status = "✓" if label == expected_label else "✗"
        print(f"{status} {type_id:40s} -> {label} ({name})")

        if label != expected_label:
            print(f"  预期: {expected_label} ({expected_name}), 实际: {label} ({name})")
            all_pass = False

    return all_pass

def test_all_query_actors():
    """测试查询到的所有actor类型"""
    import json

    print("\n" + "=" * 60)
    print("测试所有查询到的Actor类型")
    print("=" * 60)

    query_file = Path(__file__).parent.parent / 'config' / 'actor_types_query_result.json'
    if not query_file.exists():
        print("❌ 查询结果文件不存在，请先运行 query_all_actors.py")
        return False

    with open(query_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n地图: {data['map_name']}")
    print(f"Actor类型总数: {data['total_actor_types']}\n")

    vehicle_stats = {
        'car': 0,
        'truck': 0,
        'bus': 0,
        'bicycle': 0,
        'motorcycle': 0,
        'other': 0
    }

    for actor_type in data['actor_types']:
        type_id = actor_type['type_id']

        # 只测试vehicle类型
        if not type_id.startswith('vehicle.'):
            continue

        label = get_occupancy_label_from_type_id(type_id)
        name = get_occupancy_name(label)

        # 统计
        if name == 'car':
            vehicle_stats['car'] += 1
        elif name == 'truck':
            vehicle_stats['truck'] += 1
        elif name == 'bus':
            vehicle_stats['bus'] += 1
        elif name == 'bicycle':
            vehicle_stats['bicycle'] += 1
        elif name == 'motorcycle':
            vehicle_stats['motorcycle'] += 1
        else:
            vehicle_stats['other'] += 1

        print(f"  {type_id:40s} -> {label:2d} ({name})")

    print("\n" + "-" * 60)
    print("车辆类型统计:")
    for vtype, count in vehicle_stats.items():
        if count > 0:
            print(f"  {vtype:15s}: {count}")

    return True

def main():
    print("🔍 Actor到Occupancy映射测试工具\n")

    # 测试1：预定义测试用例
    test1_pass = test_vehicle_mapping()

    # 测试2：查询结果中的所有actor
    test2_pass = test_all_query_actors()

    print("\n" + "=" * 60)
    if test1_pass and test2_pass:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败，请检查映射配置")
    print("=" * 60)

if __name__ == '__main__':
    main()

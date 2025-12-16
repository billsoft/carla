#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速连接测试"""

import sys
import os

# 添加 CARLA 路径
sys.path.append(r'd:\code\carla\PythonAPI\carla')

print("=" * 60)
print("CARLA 快速连接测试")
print("=" * 60)

# 测试 CARLA 导入
try:
    import carla
    print("✓ carla 模块导入成功")
except Exception as e:
    print(f"✗ carla 导入失败: {e}")
    sys.exit(1)

# 测试连接
try:
    print("\n连接 CARLA 服务器...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    print(f"✓ 连接成功!")
    print(f"  地图: {world.get_map().name}")
    print(f"  Actor 数量: {len(world.get_actors())}")

    # 获取蓝图库
    bp_lib = world.get_blueprint_library()
    print(f"  蓝图数量: {len(bp_lib)}")

    # 检查相机是否支持镜头畸变
    cam_bp = bp_lib.find('sensor.camera.rgb')
    attrs = [attr.id for attr in cam_bp]

    print(f"\n相机属性检查:")
    lens_attrs = ['lens_k', 'lens_kcube', 'lens_circle_multiplier', 'lens_circle_falloff']
    for attr in lens_attrs:
        if attr in attrs:
            print(f"  ✓ {attr}")
        else:
            print(f"  ✗ {attr} (不支持)")

    print("\n=" * 60)
    print("✓ 测试完成 - 环境正常!")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ 连接失败: {e}")
    print("  请确认 CARLA 服务器已运行")
    sys.exit(1)

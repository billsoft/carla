"""
测试 NPC 生成速度
"""
import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
build_dist = project_root / 'Build' / 'PythonAPI' / 'dist'
if build_dist.exists():
    for whl in build_dist.glob('*.whl'):
        sys.path.append(str(whl))
        break

import carla

# 添加 dense_occupancy_collection 到路径
sys.path.insert(0, str(project_root / 'dense_occupancy_collection'))
from core.scenario_manager import ScenarioManager

def main():
    print("=" * 60)
    print("NPC 生成速度测试 (dense_occupancy_collection)")
    print("=" * 60)

    # 连接 CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Traffic Manager
    tm = client.get_trafficmanager(8001)
    tm.set_synchronous_mode(True)

    scenario = None
    try:
        # 创建场景管理器
        scenario = ScenarioManager(world, tm_port=8001)

        # 生成 Hero
        print("\n[1/2] 生成 Hero 车辆...")
        hero_start = time.time()
        hero = scenario.spawn_hero(filter_pattern='vehicle.nissan.patrol', enable_autopilot=True)
        hero_time = time.time() - hero_start
        print(f"  ✅ Hero 生成完成，耗时: {hero_time:.2f}s")

        # 生成 NPC (30 车 + 10 行人)
        print("\n[2/2] 生成 NPC...")
        npc_start = time.time()
        scenario.spawn_npcs(num_vehicles=30, num_walkers=10)
        npc_time = time.time() - npc_start
        print(f"\n✅ 测试完成")
        print(f"  Hero 生成: {hero_time:.2f}s")
        print(f"  NPC 生成: {npc_time:.2f}s")
        print(f"  总耗时: {hero_time + npc_time:.2f}s")

    finally:
        if scenario:
            scenario.destroy()

        # 恢复设置
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == '__main__':
    main()

"""
测试可见性过滤
采集1帧数据并输出详细调试信息
"""

import carla
import time
import numpy as np
from pathlib import Path

from dense_occupancy_collection.sensors.semantic_lidar_sensor import SemanticLidarSensor
from dense_occupancy_collection.processing.ground_truth_voxel_generator import GroundTruthVoxelGenerator
from dense_occupancy_collection.config.occupancy_config import (
    X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION,
    VISIBILITY_LIDAR_CONFIG
)

def main():
    print("\n" + "="*60)
    print("可见性过滤测试")
    print("="*60 + "\n")

    # 连接CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    print(f"✓ 已连接到CARLA服务器")
    print(f"  地图: {world.get_map().name}")

    # 查找Hero车辆
    actors = world.get_actors()
    hero_vehicle = None

    for actor in actors.filter('vehicle.*'):
        if actor.attributes.get('role_name') == 'hero':
            hero_vehicle = actor
            break

    if hero_vehicle is None:
        print("❌ 未找到Hero车辆!")
        print("   请先启动手动控制: python PythonAPI/examples/manual_control.py")
        return

    print(f"✓ 找到Hero车辆: ID={hero_vehicle.id}, 类型={hero_vehicle.type_id}")

    # 获取场景中所有车辆和行人
    vehicles = actors.filter('vehicle.*')
    walkers = actors.filter('walker.pedestrian.*')

    print(f"\n📊 场景状态:")
    print(f"  总车辆数: {len(vehicles)}")
    print(f"  总行人数: {len(walkers)}")

    # 列出附近的车辆
    nearby_vehicles = []
    hero_loc = hero_vehicle.get_location()

    for v in vehicles:
        if v.id == hero_vehicle.id:
            continue
        dist = v.get_location().distance(hero_loc)
        if dist < 60.0:
            nearby_vehicles.append((v.id, v.type_id.split('.')[-1], dist))

    nearby_vehicles.sort(key=lambda x: x[2])

    print(f"\n  60米内的车辆 ({len(nearby_vehicles)}辆):")
    for vid, vtype, dist in nearby_vehicles[:10]:
        print(f"    ID {vid:4d}: {vtype:20s} 距离 {dist:5.1f}m")

    # 创建64线激光雷达
    print(f"\n🔍 创建64线语义激光雷达...")
    lidar_sensor = SemanticLidarSensor(world, hero_vehicle, config=VISIBILITY_LIDAR_CONFIG)
    lidar_sensor.listen_to_queue()

    # 等待传感器初始化
    print(f"   等待传感器初始化...")
    for i in range(5):
        world.tick()
        time.sleep(0.1)

    # 创建体素生成器
    print(f"\n🔷 创建体素生成器...")
    voxel_generator = GroundTruthVoxelGenerator(
        X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION
    )

    # Tick一次获取最新数据
    print(f"\n⏱️ Tick world...")
    world.tick()
    time.sleep(0.2)

    # 获取激光雷达数据
    print(f"\n📡 获取激光雷达数据...")
    try:
        lidar_data_dict = lidar_sensor.data_queue.get(timeout=3.0)
        lidar_raw = lidar_data_dict['raw_data']
        print(f"  ✓ 激光雷达数据大小: {len(lidar_raw):,} bytes")

        # 解析数据看看扫到了什么
        dtype = np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('cos', np.float32),
            ('obj_idx', np.uint32),
            ('tag', np.uint32)
        ])
        points = np.frombuffer(lidar_raw, dtype=dtype)

        visible_actor_ids = np.unique(points['obj_idx'])
        print(f"\n  激光雷达扫到的Actor IDs: {sorted(list(visible_actor_ids))}")
        print(f"  扫到的Actor数量: {len(visible_actor_ids)}")

    except Exception as e:
        print(f"  ❌ 获取激光雷达数据失败: {e}")
        lidar_sensor.destroy()
        return

    # 生成体素 (mask 已移除，使用 Label 0 表示不可见区域)
    print(f"\n🔷 生成体素...")
    occupancy, actor_ids = voxel_generator.generate(
        world, hero_vehicle, visibility_data=lidar_raw
    )

    # 统计最终结果
    print(f"\n📊 最终结果:")
    print(f"  总体素: {occupancy.size:,}")
    print(f"  占用体素: {np.sum(occupancy > 0):,}")

    final_actor_ids = np.unique(actor_ids[actor_ids > 0])
    print(f"  最终保留的Actor IDs: {sorted(list(final_actor_ids))}")
    print(f"  保留的Actor数量: {len(final_actor_ids)}")

    # 保存测试数据
    output_dir = Path("output/visibility_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "test.npz"
    voxel_generator.save_to_npz(
        npz_path, occupancy, actor_ids,
        metadata={'test': True, 'hero_id': hero_vehicle.id}
    )

    print(f"\n💾 测试数据已保存: {npz_path}")
    print(f"\n可以用以下命令检查:")
    print(f"  python dense_occupancy_collection/scripts/diagnose_voxel.py {npz_path}")

    # 清理
    lidar_sensor.destroy()

    print(f"\n" + "="*60)
    print("测试完成")
    print("="*60 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

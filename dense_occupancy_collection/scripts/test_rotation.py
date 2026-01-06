"""
测试车辆旋转下的地面一致性
验证 90度旋转变形修复是否成功
"""
import carla
import time
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.collect_panorama import collect_frame

def test_rotation():
    """测试不同旋转角度下的地面一致性"""

    # 连接 CARLA
    print("连接 CARLA 服务器...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
        print(f"✅ 连接成功: {world.get_map().name}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("请确保 CARLA 服务器正在运行:")
        print("  cmake --build Build --target launch")
        return

    # 获取或生成车辆
    vehicles = world.get_actors().filter('vehicle.*')
    if len(vehicles) == 0:
        print("\n未找到车辆,尝试生成车辆...")

        # 生成车辆
        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.filter('vehicle.tesla.model3')[0]

        spawn_points = world.get_map().get_spawn_points()
        if len(spawn_points) == 0:
            print("❌ 未找到生成点")
            return

        transform = spawn_points[0]
        ego = world.spawn_actor(bp, transform)
        print(f"✅ 生成车辆: {ego.type_id}")
    else:
        ego = vehicles[0]
        print(f"✅ 使用现有车辆: {ego.type_id}")

    # 保存初始位置
    initial_transform = ego.get_transform()
    print(f"初始位置: x={initial_transform.location.x:.2f}, y={initial_transform.location.y:.2f}, yaw={initial_transform.rotation.yaw:.2f}")

    # 测试角度
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    print(f"\n将测试 {len(angles)} 个旋转角度: {angles}")
    print("=" * 60)

    for idx, angle in enumerate(angles):
        print(f"\n[{idx+1}/{len(angles)}] 测试 yaw={angle}°")
        print("-" * 60)

        # 设置车辆朝向 (保持位置不变,只改变朝向)
        transform = ego.get_transform()
        transform.rotation.yaw = angle
        ego.set_transform(transform)

        # 等待物理稳定
        time.sleep(0.5)
        world.tick()  # 确保世界更新

        # 验证朝向
        current_yaw = ego.get_transform().rotation.yaw
        print(f"当前朝向: {current_yaw:.2f}° (目标: {angle}°)")

        # 采集数据
        try:
            print(f"开始采集数据...")
            # 注意: collect_frame 函数需要根据实际情况调整
            # 这里假设它会自动保存到 dataset_output 目录
            # collect_frame(world, ego, frame_id=f"rotation_test_{angle:03d}")

            # 如果 collect_frame 不存在或不适用,可以使用简化版本:
            from processing.ground_truth_voxel_generator import GroundTruthVoxelGenerator
            from config.occupancy_config import OccupancyConfig

            # 创建体素生成器
            config = OccupancyConfig()
            voxel_gen = GroundTruthVoxelGenerator(
                x_range=config.x_range,
                y_range=config.y_range,
                z_range=config.z_range,
                resolution=config.resolution
            )

            # 生成体素
            occupancy, mask, actor_ids = voxel_gen.generate(world, ego.get_transform())

            # 简单统计
            num_voxels = occupancy.size
            num_filled = np.count_nonzero(occupancy != 0)

            # 统计地面类型 (Road=11, Sidewalk=13, Terrain=14)
            num_road = np.count_nonzero(occupancy == 11)
            num_sidewalk = np.count_nonzero(occupancy == 13)
            num_terrain = np.count_nonzero(occupancy == 14)

            print(f"✅ 采集完成:")
            print(f"   总体素数: {num_voxels:,}")
            print(f"   非空体素: {num_filled:,} ({num_filled/num_voxels*100:.2f}%)")
            print(f"   地面分布:")
            print(f"     - Road (11):     {num_road:,}")
            print(f"     - Sidewalk (13): {num_sidewalk:,}")
            print(f"     - Terrain (14):  {num_terrain:,}")

            # 保存结果 (可选)
            import numpy as np
            output_dir = "rotation_test_results"
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, f"yaw_{angle:03d}.npz")
            np.savez_compressed(
                output_file,
                occupancy=occupancy,
                mask=mask,
                actor_ids=actor_ids,
                yaw=angle,
                x_range=config.x_range,
                y_range=config.y_range,
                z_range=config.z_range,
                resolution=config.resolution,
                grid_size=occupancy.shape
            )
            print(f"   保存至: {output_file}")

        except Exception as e:
            print(f"❌ 采集失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("所有角度采集完成！")
    print("\n下一步:")
    print("1. 使用 occupancy_viewer 可视化验证:")
    print("   python occupancy_viewer/run_viewer.py")
    print("   (修改 DATA_DIR 为 'rotation_test_results')")
    print("\n2. 检查项:")
    print("   ✅ 所有角度下地面类型一致")
    print("   ✅ 无 Index 超限错误")
    print("   ✅ 无地面错位或变形")
    print("   ✅ 道路、人行道边界清晰")

if __name__ == "__main__":
    import numpy as np  # 确保导入
    test_rotation()

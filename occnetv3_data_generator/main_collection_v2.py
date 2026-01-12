"""
OccNetV3 数据采集主脚本 (v2 - 支持NPC和红绿灯控制)
生成符合 occ_network 训练要求的数据集
"""
import sys
import os
from pathlib import Path
import argparse
import time

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入CARLA
try:
    build_dist = project_root / 'Build' / 'PythonAPI' / 'dist'
    if build_dist.exists():
        for whl in build_dist.glob('*.whl'):
            sys.path.append(str(whl))
            break
except: pass

import carla
import numpy as np

# 导入我们的模块
sys.path.insert(0, str(Path(__file__).parent))
from config.camera_config import TESLA_CAMERAS
from config.occupancy_config import (
    X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION, DEPTH_CAMERA_CONFIG
)
from sensors.camera_manager import GrayCameraManager
from sensors.depth_suite import DepthSuite
from processing.ground_truth_voxel_generator import GroundTruthVoxelGenerator
from data_utils.data_saver import OccNetDataSaver


def setup_carla(host='localhost', port=2000, town='Town10HD'):
    """连接CARLA并设置同步模式"""
    print(f"[CARLA] 连接到 {host}:{port}")
    client = carla.Client(host, port)
    client.set_timeout(10.0)

    # 加载地图
    try:
        world = client.load_world(town)
        print(f"  ✓ 加载地图: {town}")
    except RuntimeError as e:
        print(f"  ⚠️  使用当前地图")
        world = client.get_world()

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 10.0  # 10Hz
    world.apply_settings(settings)
    print(f"  ✓ 同步模式: 10Hz")

    return client, world


def set_traffic_lights_green(world):
    """设置所有红绿灯为常绿状态"""
    print("\n[TrafficLights] 设置所有红绿灯为常绿...")
    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    if traffic_lights:
        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.set_green_time(999999.0)
            tl.freeze(True)

        print(f"  ✓ 已设置 {len(traffic_lights)} 个红绿灯为常绿")
    else:
        print(f"  ⚠️ 未找到红绿灯")


def spawn_vehicle(world, spawn_point=None, tm_port=8000):
    """生成ego车辆"""
    bp_lib = world.get_blueprint_library()

    target_vehicles = [
        'vehicle.tesla.model3',
        'vehicle.lincoln.mkz_2017',
        'vehicle.audi.etron',
        'vehicle.audi.a2',
        'vehicle.nissan.patrol',
        'vehicle.nissan.micra',
        'vehicle.toyota.prius'
    ]

    vehicle_bp = None
    for v_name in target_vehicles:
        bps = bp_lib.filter(v_name)
        if bps:
            vehicle_bp = bps[0]
            print(f"[Vehicle] 找到目标车辆: {v_name}")
            break

    if vehicle_bp is None:
        print("[Vehicle] 警告: 未找到指定车型, 尝试使用任意车辆...")
        bps = bp_lib.filter('vehicle.*')
        if not bps:
            raise RuntimeError("无法找到任何车辆蓝图!")
        vehicle_bp = bps[0]
        for bp in bps:
            if int(bp.get_attribute('number_of_wheels')) == 4:
                vehicle_bp = bp
                break

    if spawn_point is None:
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
             raise RuntimeError("地图没有定义生成点!")
        spawn_point = np.random.choice(spawn_points)

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"[Vehicle] 已生成: {vehicle.type_id}")

    # 启用自动驾驶 (带端口重试)
    tm_ports = [tm_port, 8001, 8002, 8003, 8004, 8005]
    autopilot_enabled = False
    for port in tm_ports:
        try:
            vehicle.set_autopilot(True, port)
            print(f"  ✓ 自动驾驶已启用 (TM Port: {port})")
            autopilot_enabled = True
            break
        except RuntimeError as e:
            if "bind" in str(e).lower():
                continue
            raise

    if not autopilot_enabled:
        print(f"  ⚠️ 自动驾驶启用失败")

    return vehicle


def spawn_npcs(world, num_vehicles=30, num_walkers=10, tm_port=8000):
    """生成 NPC 车辆和行人"""
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    npc_actors = []

    print(f"\n[NPC] 生成 NPC: {num_vehicles} 辆车, {num_walkers} 个行人...")

    # 1. NPC 车辆
    vehicle_categories = {
        'car': {
            'patterns': ['vehicle.audi.*', 'vehicle.bmw.*', 'vehicle.tesla.*',
                        'vehicle.toyota.*', 'vehicle.nissan.*', 'vehicle.dodge.*'],
            'ratio': 0.5
        },
        'truck': {
            'patterns': ['vehicle.carlamotors.carlacola', 'vehicle.tesla.cybertruck',
                        'vehicle.ford.ambulance'],
            'ratio': 0.1
        },
        'bus': {
            'patterns': ['vehicle.mitsubishi.fusorosa', 'vehicle.volkswagen.t2'],
            'ratio': 0.1
        },
        'bicycle': {
            'patterns': ['vehicle.bh.crossbike', 'vehicle.diamondback.century',
                        'vehicle.gazelle.omafiets'],
            'ratio': 0.15
        },
        'motorcycle': {
            'patterns': ['vehicle.harley*', 'vehicle.kawasaki.*',
                        'vehicle.yamaha.*', 'vehicle.vespa.*'],
            'ratio': 0.15
        }
    }

    spawn_idx = 0
    for cat_name, cat_info in vehicle_categories.items():
        target_num = int(num_vehicles * cat_info['ratio'])
        if target_num == 0 and cat_info['ratio'] > 0:
            target_num = 1

        bps = []
        for pattern in cat_info['patterns']:
            bps.extend(list(bp_lib.filter(pattern)))

        if not bps:
            continue

        spawned_count = 0
        for _ in range(target_num):
            if spawn_idx >= len(spawn_points):
                break

            bp = np.random.choice(bps)
            if bp.has_attribute('color'):
                color = np.random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)

            while spawn_idx < len(spawn_points):
                npc = world.try_spawn_actor(bp, spawn_points[spawn_idx])
                spawn_idx += 1
                if npc:
                    try:
                        npc.set_autopilot(True, tm_port)
                    except RuntimeError as e:
                        if "bind" not in str(e).lower():
                            raise

                    npc_actors.append(npc)
                    spawned_count += 1
                    break

        print(f"  - {cat_name}: {spawned_count} 辆")

    # 2. 行人
    walker_bps = list(bp_lib.filter('walker.pedestrian.*'))
    controller_bp = bp_lib.find('controller.ai.walker')

    if walker_bps and num_walkers > 0:
        spawned_walkers = 0
        controllers = []

        for _ in range(num_walkers * 3):
            if spawned_walkers >= num_walkers:
                break

            spawn_point = np.random.choice(spawn_points)
            loc = spawn_point.location
            loc.x += np.random.uniform(-20, 20)
            loc.y += np.random.uniform(-20, 20)
            loc.z += 1.0

            bp = np.random.choice(walker_bps)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')

            walker = world.try_spawn_actor(bp, carla.Transform(loc))
            if walker:
                npc_actors.append(walker)

                controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                if controller:
                    npc_actors.append(controller)
                    controllers.append(controller)
                    spawned_walkers += 1

        world.tick()
        for controller in controllers:
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(1.4)

        print(f"  - pedestrian: {spawned_walkers} 个")

    # 物理稳定
    for _ in range(10):
        world.tick()

    return npc_actors


def main():
    parser = argparse.ArgumentParser(description='OccNetV3 数据采集 (支持NPC)')
    parser.add_argument('--host', default='localhost', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA端口')
    parser.add_argument('--town', default='Town10HD', help='地图名称')
    parser.add_argument('--output', default='D:/code/carla/dataset_occnet_v3', help='输出目录')
    parser.add_argument('--frames', type=int, default=10, help='采集帧数')
    parser.add_argument('--scene', default='scene', help='场景名称')
    parser.add_argument('--num-vehicles', type=int, default=30, help='NPC车辆数量')
    parser.add_argument('--num-walkers', type=int, default=10, help='NPC行人数量')
    args = parser.parse_args()

    client, world = None, None
    vehicle = None
    npc_actors = []
    camera_manager = None
    depth_suite = None
    voxel_generator = None
    data_saver = None

    try:
        # 1. 设置CARLA
        client, world = setup_carla(args.host, args.port, args.town)

        # 2. 设置红绿灯常绿
        set_traffic_lights_green(world)

        # 3. 生成车辆
        vehicle = spawn_vehicle(world)
        time.sleep(1.0)

        # 4. 生成 NPC
        npc_actors = spawn_npcs(world, args.num_vehicles, args.num_walkers)

        # 5. 附加传感器
        print("\n[Sensors] 附加8个灰度相机...")
        camera_manager = GrayCameraManager(world, vehicle)
        camera_manager.start_listening()

        print("[Sensors] 附加深度相机套件 (8路CubeMap)...")
        depth_suite = DepthSuite(world, vehicle, DEPTH_CAMERA_CONFIG)
        time.sleep(0.5)

        # 6. 初始化生成器和保存器
        print("[VoxelGenerator] 初始化 Ground Truth 生成器...")
        voxel_generator = GroundTruthVoxelGenerator(
            x_range=X_RANGE,
            y_range=Y_RANGE,
            z_range=Z_RANGE,
            resolution=RESOLUTION
        )
        data_saver = OccNetDataSaver(args.output, args.scene)

        # 7. 保存相机标定
        print("\n[Calibration] 保存相机标定...")
        intrinsics = {cam['id']: camera_manager.get_intrinsics(cam['id']) for cam in TESLA_CAMERAS}
        extrinsics = {cam['id']: camera_manager.get_extrinsics(cam['id']) for cam in TESLA_CAMERAS}
        data_saver.save_calibration(intrinsics, extrinsics, TESLA_CAMERAS)

        # 8. 采集数据
        print(f"\n[Collection] 开始采集 {args.frames} 帧数据...")
        print("="*60)

        ego_pose_prev = None

        for frame_idx in range(args.frames):
            start_time = time.time()

            world.tick()

            camera_data = camera_manager.get_synced_frame(timeout=2.0)
            if camera_data is None:
                print(f"  帧 {frame_idx}: 相机数据同步失败,跳过")
                continue

            try:
                depth_data = depth_suite.get_data(timeout=2.0)
            except Exception as e:
                print(f"  帧 {frame_idx}: 深度相机数据获取失败 ({e}),跳过")
                continue

            ego_transform = vehicle.get_transform()
            ego_pose = np.array(ego_transform.get_matrix(), dtype=np.float32)

            occupancy, actor_ids = voxel_generator.generate(
                world,
                vehicle,
                visibility_data=None
            )

            non_empty_count = np.count_nonzero(occupancy)

            if ego_pose_prev is not None:
                ego_motion = np.linalg.inv(ego_pose) @ ego_pose_prev
            else:
                ego_motion = np.eye(4, dtype=np.float32)

            ego_pose_prev = ego_pose.copy()

            images = {cam['id']: camera_data[cam['id']]['image'] for cam in TESLA_CAMERAS}

            sample_id = data_saver.generate_sample_id()

            data_saver.save_sample(
                sample_id=sample_id,
                images=images,
                occupancy=occupancy,
                flow=None,
                flow_mask=None,
                ego_pose=ego_pose,
                ego_motion=ego_motion,
            )

            elapsed_time = time.time() - start_time
            print(f"  ✓ 帧 {frame_idx}/{args.frames}: {sample_id} "
                  f"非空={non_empty_count} "
                  f"耗时={elapsed_time:.2f}s")

        # 9. 完成
        data_saver.finalize()
        print("\n" + "="*60)
        print("✅ 数据采集完成!")

    except KeyboardInterrupt:
        print("\n用户中断")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理
        print("\n[Cleanup] 清理资源...")
        if camera_manager:
            camera_manager.destroy()
        if depth_suite:
            depth_suite.destroy()
        if vehicle and vehicle.is_alive:
            vehicle.destroy()
        for npc in npc_actors:
            if npc.is_alive:
                npc.destroy()
        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        print("  ✓ 清理完成")


if __name__ == '__main__':
    main()

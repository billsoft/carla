"""
OccNetV3 数据采集主脚本 (v2 - 修复卡死问题)
参考 dense_occupancy_collection 的正确初始化流程
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
    # 优先查找 Build/PythonAPI/dist (源码编译版)
    build_dist = project_root / 'Build' / 'PythonAPI' / 'dist'
    if build_dist.exists():
        for whl in build_dist.glob('*.whl'):
            sys.path.insert(0, str(whl)) # ⭐ 优先使用自定义编译的 wheel
            print(f"[Import] Added custom build wheel: {whl}")
            break
    else:
        # 回退到 PythonAPI/carla (源码目录)
        source_path = project_root / 'PythonAPI' / 'carla'
        if source_path.exists():
            sys.path.insert(0, str(source_path)) # ⭐ 优先使用源码目录
            print(f"[Import] Added source path: {source_path}")
except: pass

import carla
# print(f"[Info] CARLA Version: {carla.__version__}")
print(f"[Info] CARLA Path: {carla.__file__}")
import numpy as np

# 导入我们的模块
sys.path.insert(0, str(Path(__file__).parent))
from config.camera_config import TESLA_CAMERAS
from config.occupancy_config import (
    X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION
)
from sensors.camera_manager import CameraManager
from sensors.semantic_lidar_sensor import SemanticLidarSensor
from processing.ground_truth_voxel_generator import GroundTruthVoxelGenerator
from processing.visibility_filter_simple import VisibilityFilterSimple # ⭐ 新增
from data_utils.data_saver import OccNetDataSaver


def setup_carla(host='localhost', port=2000, town='Town10HD'):
    """连接CARLA并设置同步模式 (参考 dense_occupancy_collection)"""
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

    world.set_weather(carla.WeatherParameters.ClearNoon)

    # 设置同步模式 (与 dense_occupancy_collection 完全一致)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20Hz
    world.apply_settings(settings)
    print(f"  ✓ 同步模式: 20Hz (delta={settings.fixed_delta_seconds}s)")

    # 设置 Traffic Manager (尝试多个端口)
    tm = None
    tm_ports = [8000, 8001, 8002, 8010, 8015]
    for tm_port in tm_ports:
        try:
            tm = client.get_trafficmanager(tm_port)
            tm.set_synchronous_mode(True)
            print(f"  ✓ Traffic Manager: Port {tm_port}")
            break
        except RuntimeError as e:
            if "bind" in str(e).lower():
                continue
            raise

    if tm is None:
        print(f"  ⚠️ Traffic Manager 无法启动,所有端口被占用")
        # 如果没有 TM，使用默认端口 8000
        tm_port = 8000

    # 设置红绿灯常绿
    traffic_lights = world.get_actors().filter('traffic.traffic_light*')
    if traffic_lights:
        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
        print(f"  ✓ 已设置 {len(traffic_lights)} 个红绿灯为常绿")
    else:
        print(f"  ⚠️ 未找到红绿灯")

    return client, world, tm_port


def spawn_vehicle(world, tm_port, spawn_point=None):
    """生成ego车辆 (使用 autopilot)"""
    bp_lib = world.get_blueprint_library()

    target_vehicles = [
        'vehicle.lincoln.mkz_2017',
        'vehicle.tesla.model3',
        'vehicle.audi.etron',
        'vehicle.nissan.patrol'
    ]

    vehicle_bp = None
    for v_name in target_vehicles:
        bps = bp_lib.filter(v_name)
        if bps:
            vehicle_bp = bps[0]
            print(f"[Vehicle] 找到目标车辆: {v_name}")
            break

    if vehicle_bp is None:
        bps = bp_lib.filter('vehicle.*')
        if not bps:
            raise RuntimeError("无法找到任何车辆蓝图!")
        vehicle_bp = bps[0]

    vehicle_bp.set_attribute('role_name', 'hero')

    if spawn_point is None:
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
             raise RuntimeError("地图没有定义生成点!")

        # 尝试多个生成点（避免碰撞）
        # ⭐ 稀疏生成优化: 对 spawn_points 进行洗牌，并检查最小距离
        import random
        random.shuffle(spawn_points)
        
        vehicle = None
        for attempt in range(min(50, len(spawn_points))):
            try:
                spawn_point = spawn_points[attempt]
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                print(f"[Vehicle] 已生成: {vehicle.type_id} (尝试 {attempt+1} 次)")
                break
            except RuntimeError as e:
                if "collision" in str(e).lower():
                    continue  # 尝试下一个点
                raise

        if vehicle is None:
            raise RuntimeError("无法找到空闲的生成点!")
    else:
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"[Vehicle] 已生成: {vehicle.type_id}")

    # ⭐ 先启用 autopilot，再物理稳定（避免同步模式 tick 死锁）
    # 启用 autopilot (使用 setup_carla 确定的 TM 端口)
    try:
        vehicle.set_autopilot(True, tm_port)
        print(f"  ✓ Autopilot 已启用 (TM Port: {tm_port})")
    except RuntimeError as e:
        print(f"  ⚠️ Autopilot 启用失败: {e}")

    # 物理稳定（必须在 autopilot 启用后，否则同步模式会死锁）
    for _ in range(10):
        world.tick()

    return vehicle


def spawn_npcs(client, world, num_vehicles=30, num_walkers=10, tm_port=8001):
    """生成 NPC 车辆和行人 (参考 scenario_manager.py)"""
    import time
    total_start = time.time()

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    npc_actors = []

    print(f"\n[NPC] 生成 NPC: {num_vehicles} 辆车, {num_walkers} 个行人...")

    # 1. NPC 车辆
    vehicle_start = time.time()
    vehicle_categories = {
        'car': {
            'patterns': ['vehicle.audi.*', 'vehicle.bmw.*', 'vehicle.tesla.*',
                        'vehicle.toyota.*', 'vehicle.nissan.*', 'vehicle.dodge.*', 
                        'vehicle.lincoln.*', 'vehicle.mini.*', 'vehicle.ford.*', 'vehicle.chevrolet.*'],
            'ratio': 0.4
        },
        'truck': {
            'patterns': ['vehicle.carlacola.actors', 'vehicle.firetruck.actors',
                        'vehicle.ambulance.ford', 'vehicle.sprinter.mercedes',
                        # Fallbacks
                        'vehicle.carlamotors.*', 'vehicle.tesla.cybertruck'],
            'ratio': 0.15
        },
        'bus': {
            'patterns': ['vehicle.fuso.mitsubishi', 'vehicle.mitsubishi.fusorosa'],
            'ratio': 0.05
        },
        'bicycle': {
            'patterns': ['vehicle.bh.crossbike', 'vehicle.diamondback.century',
                        'vehicle.gazelle.omafiets'],
            'ratio': 0.20
        },
        'motorcycle': {
            'patterns': ['vehicle.harley-davidson.low_rider', 'vehicle.kawasaki.ninja',
                        'vehicle.yamaha.yzf', 'vehicle.vespa.zx125'],
            'ratio': 0.20
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
                    # ⭐ 性能优化: 必须指定 TM 端口
                    try:
                        npc.set_autopilot(True, tm_port)
                        
                        # ⭐ TM 高级配置 (减少碰撞与拥堵)
                        tm = client.get_trafficmanager(tm_port)
                        
                        # 1. 混合物理模式: 远处车辆只进行简单运动计算 (半径50米)
                        # 这不仅提升性能，还能减少物理鬼畜
                        tm.set_hybrid_physics_mode(True)
                        tm.set_hybrid_physics_radius(50.0)
                        
                        # 2. 保持车距: 默认是 0，增加到 2.5 米
                        tm.distance_to_leading_vehicle(npc, 2.5)
                        
                        # 3. 忽略红绿灯概率 (0% = 完全遵守)
                        tm.ignore_lights_percentage(npc, 0.0)
                        
                        # 4. 自动换道 (50% 概率)
                        tm.auto_lane_change(npc, True)
                        
                        # 5. 自动重生 (如果被卡住不动超过 10 秒，自动重生)
                        # 注意: 这需要 CARLA 0.9.13+，如果是旧版本可能会报错，加个 try-catch
                        try:
                            tm.set_respawn_dormant_vehicles(True)
                        except: pass
                        
                    except RuntimeError as e:
                        if "bind error" not in str(e).lower():
                            pass  # 忽略其他错误

                    npc_actors.append(npc)
                    spawned_count += 1
                    
                    # ⭐ 稀疏生成: 如果成功生成，从 spawn_points 中移除附近的所有点
                    # 防止下一辆车生成得太近
                    current_loc = spawn_points[spawn_idx-1].location # idx已经+1了
                    to_remove = []
                    for i, sp in enumerate(spawn_points):
                        if i >= spawn_idx: # 只检查还没用过的点
                            dist = sp.location.distance(current_loc)
                            if dist < 20.0: # 最小间距 20 米
                                to_remove.append(sp)
                    
                    for sp in to_remove:
                        spawn_points.remove(sp)
                        
                    break

        print(f"  - {cat_name}: {spawned_count} 辆")

    vehicle_time = time.time() - vehicle_start
    print(f"  ⏱️  车辆生成耗时: {vehicle_time:.2f}s")

    # 2. 行人
    walker_start = time.time()
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

        # 启动行人 AI
        world.tick()
        for controller in controllers:
            controller.start()
            # ⭐ 性能优化: 移除 go_to_location() 调用
            # go_to_location(world.get_random_location_from_navigation()) 查询导航网格极慢
            # (~15秒/个 × 10个 = 150秒), 直接使用 start() 让行人随机游荡
            controller.set_max_speed(1.0 + np.random.random())

        print(f"  - pedestrian: {spawned_walkers} 个")

    walker_time = time.time() - walker_start
    print(f"  ⏱️  行人生成耗时: {walker_time:.2f}s")

    # 物理稳定 (重要!)
    print("  等待 NPC 物理稳定...")
    stab_start = time.time()
    for _ in range(10):
        world.tick()

    stab_time = time.time() - stab_start
    total_time = time.time() - total_start
    print(f"  ⏱️  物理稳定耗时: {stab_time:.2f}s")
    print(f"  ⏱️  总耗时: {total_time:.2f}s (车辆{vehicle_time:.1f}s + 行人{walker_time:.1f}s + 稳定{stab_time:.1f}s)")

    return npc_actors


def main():
    parser = argparse.ArgumentParser(description='OccNetV3 数据采集 (修复版)')
    parser.add_argument('--host', default='localhost', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA端口')
    parser.add_argument('--town', default='Town10HD', help='地图名称')
    parser.add_argument('--output', default='d:/code/carla/dataset_10k_bak', help='输出目录 (默认: dataset_10k_bak)')
    parser.add_argument('--frames', type=int, default=10, help='采集帧数 (默认: 10)')
    parser.add_argument('--scene', default='scene', help='场景名称前缀')
    parser.add_argument('--num-vehicles', type=int, default=30, help='NPC车辆数量')
    parser.add_argument('--num-walkers', type=int, default=10, help='NPC行人数量')
    parser.add_argument('--clear-output', action='store_true', default=True, help='生成前清空输出目录 (默认: True)')
    args = parser.parse_args()

    # 清空输出目录
    import shutil
    output_dir = Path(args.output)
    if output_dir.exists() and args.clear_output:
        print(f"[Cleanup] 清理输出目录: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client, world = None, None
    vehicle = None
    camera_manager = None
    semantic_lidar = None
    visibility_filter = None # ⭐ 初始化变量
    npc_actors = []

    try:
        # 1. 设置CARLA
        step_start = time.time()
        client, world, tm_port = setup_carla(args.host, args.port, args.town)
        print(f"  耗时: {(time.time()-step_start):.2f}s")

        # 2. 生成车辆
        print("\n[1/5] 生成 ego 车辆...")
        step_start = time.time()
        vehicle = spawn_vehicle(world, tm_port)
        print(f"  耗时: {(time.time()-step_start):.2f}s")

        # 3. 生成 NPC
        print("\n[2/5] 生成 NPC...")
        step_start = time.time()
        npc_actors = spawn_npcs(client, world, args.num_vehicles, args.num_walkers, tm_port=8001)
        print(f"  ✅ NPC 生成完成，耗时: {(time.time()-step_start):.2f}s")

        # 4. 附加传感器
        print("\n[3/5] 附加传感器...")
        step_start = time.time()
        camera_manager = CameraManager(world, vehicle)
        camera_manager.start_listening()
        print(f"  ✓ 相机: {len(TESLA_CAMERAS)} 个")

        # 强制 256 线配置 (使用默认 SEMANTIC_LIDAR_CONFIG)
        semantic_lidar = SemanticLidarSensor(world, vehicle, config=None)
        semantic_lidar.listen_to_queue() # ⭐ 必须启动监听
        print(f"  ✓ 语义激光雷达: 256线 (Using Default Config)")
        print(f"  耗时: {(time.time()-step_start):.2f}s")

        # 5. 初始化生成器和保存器
        step_start = time.time()
        voxel_generator = GroundTruthVoxelGenerator(
            x_range=X_RANGE,
            y_range=Y_RANGE,
            z_range=Z_RANGE,
            resolution=RESOLUTION
        )
        visibility_filter = VisibilityFilterSimple() # ⭐ 新增
        data_saver = OccNetDataSaver(args.output, args.scene)
        print(f"  ✓ 体素生成器和数据保存器")

        # 6. 保存相机标定
        intrinsics = {cam['id']: camera_manager.get_intrinsics(cam['id']) for cam in TESLA_CAMERAS}
        extrinsics = {cam['id']: camera_manager.get_extrinsics(cam['id']) for cam in TESLA_CAMERAS}
        data_saver.save_calibration(intrinsics, extrinsics, TESLA_CAMERAS)
        print(f"  ✓ 相机标定已保存")
        print(f"  耗时: {(time.time()-step_start):.2f}s")

        # 7. 等待传感器初始化 (重要!)
        print("\n[4/5] 等待传感器初始化...")
        step_start = time.time()
        for i in range(10):
            world.tick()
            if i % 2 == 0:
                print(f"  {(i+1)*10}%")
        print(f"  ✓ 传感器就绪，耗时: {(time.time()-step_start):.2f}s")

        # 7.5 清空队列 (关键: 修复前3帧数据不同步问题)
        # 传感器初始化期间积累了旧数据，必须在正式采集前清空
        print("\n[4.5/5] 清空传感器队列...")
        camera_manager.clear_queues()
        semantic_lidar.clear_queues()

        # 8. 开始采集
        print(f"\n{'='*60}")
        print(f"[5/5] 开始采集 {args.frames} 帧".center(60))
        print(f"{'='*60}\n")

        ego_pose_prev = None

        for frame_idx in range(args.frames):
            frame_start = time.time()
            print(f"\n[帧 {frame_idx+1}/{args.frames}]")

            # Step 1: World Tick
            step_start = time.time()
            world.tick()
            print(f"  [1/6] World Tick: {(time.time()-step_start)*1000:.1f}ms")

            # Step 2: 采集相机数据
            step_start = time.time()
            camera_data = camera_manager.get_synced_frame(timeout=2.0)
            if camera_data is None:
                print(f"  ❌ 相机数据同步失败,跳过")
                continue
            print(f"  [2/6] 相机采集 (8个): {(time.time()-step_start)*1000:.1f}ms")

            # Step 3: 采集LiDAR数据
            step_start = time.time()
            try:
                lidar_data = semantic_lidar.get_data(timeout=2.0)
                print(f"  [3/6] LiDAR采集: {(time.time()-step_start)*1000:.1f}ms")
            except Exception as e:
                print(f"  ❌ LiDAR失败 ({e}),跳过")
                continue

            # Step 4: 获取 Ego Pose
            step_start = time.time()
            ego_transform = vehicle.get_transform()
            ego_pose = np.array(ego_transform.get_matrix(), dtype=np.float32)
            print(f"  [4/6] Ego Pose: {(time.time()-step_start)*1000:.1f}ms")

            # Step 5: 生成体素 (最耗时)
            step_start = time.time()
            # 5.1 生成原始全透视体素
            occupancy, actor_ids = voxel_generator.generate(
                world,
                vehicle,
                visibility_data=None
            )
            # 5.2 生成 Flow (基于真实速度)
            flow, flow_mask = voxel_generator.generate_flow(
                world,
                vehicle,
                dt=0.05
            )
            # 5.3 可见性过滤 (基于 LiDAR)
            # 使用 Semantic LiDAR 过滤被遮挡的物体，但保留地面和 Ego
            occupancy, actor_ids = visibility_filter.run(
                occupancy,
                actor_ids,
                {'x_range': X_RANGE, 'y_range': Y_RANGE, 'z_range': Z_RANGE, 'resolution': RESOLUTION},
                lidar_data,
                ego_id=vehicle.id
            )
            # print(f"  [Visibility] Filter Disabled (Full Ground Truth)")

            voxel_time = (time.time() - step_start) * 1000
            print(f"  [5/6] 体素生成 + Flow + 过滤: {voxel_time:.1f}ms ⭐")

            non_empty_count = np.count_nonzero(occupancy)

            if ego_pose_prev is not None:
                ego_motion = np.linalg.inv(ego_pose) @ ego_pose_prev
            else:
                ego_motion = np.eye(4, dtype=np.float32)

            ego_pose_prev = ego_pose.copy()

            images = {cam['id']: camera_data[cam['id']] for cam in TESLA_CAMERAS}

            sample_id = data_saver.generate_sample_id()

            # Step 6: 保存数据
            step_start = time.time()
            data_saver.save_sample(
                sample_id=sample_id,
                images=images,
                occupancy=occupancy,
                flow=flow,         # ⭐ 保存 Flow
                flow_mask=flow_mask, # ⭐ 保存 Flow Mask
                ego_pose=ego_pose,
                ego_motion=ego_motion,
            )
            save_time = (time.time() - step_start) * 1000
            print(f"  [6/6] 数据保存 (8 DNG + 1 NPY): {save_time:.1f}ms")

            frame_time = (time.time() - frame_start) * 1000
            print(f"  ✅ 帧完成: 总耗时={frame_time:.0f}ms, 非空体素={non_empty_count:,}")

        # 9. 完成
        data_saver.finalize()
        print("\n" + "="*60)
        print("✅ 数据采集完成!")
        print(f"输出目录: {args.output}")

    except KeyboardInterrupt:
        print("\n用户中断")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理资源
        print("\n[Cleanup] 清理资源...")

        if camera_manager:
            try:
                camera_manager.destroy()
            except:
                pass

        if semantic_lidar:
            try:
                semantic_lidar.destroy()
            except:
                pass

        if vehicle and vehicle.is_alive:
            try:
                vehicle.destroy()
            except:
                pass

        # 清理 NPC (带详细错误处理)
        for npc in npc_actors:
            try:
                if npc.is_alive:
                    npc.destroy()
            except RuntimeError as e:
                # Actor 已经被自动销毁（如行人 AI Controller），忽略
                if "unable to destroy actor" in str(e) and "not found" in str(e):
                    pass
                else:
                    print(f"  ⚠️ 清理 Actor {npc.id if hasattr(npc, 'id') else 'unknown'} 失败: {e}")
            except Exception as e:
                print(f"  ⚠️ 清理 NPC 时发生未知错误: {e}")

        # 关闭同步模式
        if world:
            try:
                settings = world.get_settings()
                settings.synchronous_mode = False
                world.apply_settings(settings)
            except:
                pass

        print("  ✓ 清理完成")


if __name__ == '__main__':
    main()

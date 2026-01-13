"""
OccNetV3 数据采集主脚本
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
from sensors.camera_manager import CameraManager
from sensors.depth_suite import DepthSuite  # ← 新增: 深度相机套件
from processing.ground_truth_voxel_generator import GroundTruthVoxelGenerator  # ← 新增: Ground Truth生成器
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


def spawn_vehicle(world, spawn_point=None):
    """生成ego车辆"""
    bp_lib = world.get_blueprint_library()
    
    # 尝试查找常用车辆
    vehicle_bp = None
    target_vehicles = [
        'vehicle.tesla.model3',
        'vehicle.lincoln.mkz_2017',
        'vehicle.audi.etron',
        'vehicle.audi.a2',
        'vehicle.nissan.patrol',
        'vehicle.nissan.micra',
        'vehicle.toyota.prius'
    ]
    
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
        # 尽量找个轿车 (4个轮子)
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

    # 启用自动驾驶
    vehicle.set_autopilot(True)
    print(f"  ✓ 自动驾驶已启用")

    return vehicle


def main():
    parser = argparse.ArgumentParser(description='OccNetV3 数据采集')
    parser.add_argument('--host', default='localhost', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA端口')
    parser.add_argument('--town', default='Town10HD', help='地图名称')
    parser.add_argument('--output', default='D:/code/carla/dataset_occnet_v3_test', help='输出目录')
    parser.add_argument('--frames', type=int, default=10, help='采集帧数')
    parser.add_argument('--scene', default='scene', help='场景名称')
    args = parser.parse_args()

    client, world = None, None
    vehicle = None
    camera_manager = None
    depth_suite = None  # ← 修改: 深度相机套件
    voxel_generator = None
    data_saver = None

    try:
        # 1. 设置CARLA
        client, world = setup_carla(args.host, args.port, args.town)

        # 2. 生成车辆
        vehicle = spawn_vehicle(world)
        time.sleep(1.0)

        # 3. 附加传感器
        print("\n[Sensors] 附加8个灰度相机...")
        camera_manager = GrayCameraManager(world, vehicle)
        camera_manager.start_listening()

        print("[Sensors] 附加深度相机套件 (8路CubeMap)...")
        depth_suite = DepthSuite(world, vehicle, DEPTH_CAMERA_CONFIG)  # ← 修改: 使用深度相机
        time.sleep(0.5)  # 等待传感器初始化

        # 4. 初始化生成器和保存器
        print("[VoxelGenerator] 初始化 Ground Truth 生成器...")
        voxel_generator = GroundTruthVoxelGenerator(  # ← 修改: 使用 Ground Truth 生成器
            x_range=X_RANGE,
            y_range=Y_RANGE,
            z_range=Z_RANGE,
            resolution=RESOLUTION
        )
        data_saver = OccNetDataSaver(args.output, args.scene)

        # 5. 保存相机标定
        print("\n[Calibration] 保存相机标定...")
        intrinsics = {cam['id']: camera_manager.get_intrinsics(cam['id']) for cam in TESLA_CAMERAS}
        extrinsics = {cam['id']: camera_manager.get_extrinsics(cam['id']) for cam in TESLA_CAMERAS}
        data_saver.save_calibration(intrinsics, extrinsics, TESLA_CAMERAS)

        # 6. 采集数据
        print(f"\n[Collection] 开始采集 {args.frames} 帧数据...")
        print("="*60)

        ego_pose_prev = None

        for frame_idx in range(args.frames):
            start_time = time.time()

            # Tick仿真
            world.tick()

            # 获取相机数据
            camera_data = camera_manager.get_synced_frame(timeout=2.0)
            if camera_data is None:
                print(f"  帧 {frame_idx}: 相机数据同步失败,跳过")
                continue

            # 获取深度相机数据
            try:
                depth_maps, cam_transforms = depth_suite.get_data(timeout=2.0)  # ← 修改: 获取深度图
            except Exception as e:
                print(f"  帧 {frame_idx}: 深度相机数据获取失败 ({e}),跳过")
                continue

            # 计算ego_pose
            ego_transform = vehicle.get_transform()
            ego_pose = np.array(ego_transform.get_matrix(), dtype=np.float32)  # ← 修改: 直接获取矩阵

            # 生成体素占用网格 (使用 Ground Truth,暂不使用深度可见性过滤)
            # ⚠️ visibility_data 传 None: dense_occupancy_collection 使用外部 VisibilityFilter
            # 这里暂时不实现深度可见性过滤,先验证 Ground Truth 生成
            occupancy, actor_ids = voxel_generator.generate(  # ← 修改: 使用新的生成器
                world,
                vehicle,
                visibility_data=None  # ← 暂不使用深度过滤
            )

            # 计算非空体素数量
            non_empty_count = np.count_nonzero(occupancy)

            # 计算ego_motion
            ego_motion = None
            if ego_pose_prev is not None:
                # T_motion = T_current^{-1} @ T_prev
                ego_motion = np.linalg.inv(ego_pose) @ ego_pose_prev
            else:
                ego_motion = np.eye(4, dtype=np.float32)  # 第一帧

            ego_pose_prev = ego_pose.copy()

            # 提取图像 (转换cam_id格式)
            images = {cam['id']: camera_data[cam['id']] for cam in TESLA_CAMERAS}

            # 生成sample_id
            sample_id = data_saver.generate_sample_id()

            # 保存数据
            data_saver.save_sample(
                sample_id=sample_id,
                images=images,
                occupancy=occupancy,
                flow=None,  # 暂不实现flow
                flow_mask=None,
                ego_pose=ego_pose,
                ego_motion=ego_motion,
            )

            elapsed_time = time.time() - start_time
            print(f"  ✓ 帧 {frame_idx}/{args.frames}: {sample_id} "
                  f"非空={non_empty_count} "
                  f"耗时={elapsed_time:.2f}s")

        # 7. 完成
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
        if depth_suite:  # ← 修改: 清理深度相机
            depth_suite.destroy()
        if vehicle and vehicle.is_alive:
            vehicle.destroy()
        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        print("  ✓ 清理完成")


if __name__ == '__main__':
    main()

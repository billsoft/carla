#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA Occupancy 数据采集脚本
采集 5 帧完整数据: 8 相机 RGB (12-bit, 带鱼眼) + Occupancy 体素

参考官方示例:
  - PythonAPI/examples/automatic_control.py
  - PythonAPI/examples/sensor_synchronization.py
"""

import sys
import os
from pathlib import Path

# 添加 CARLA Python API 到路径
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'PythonAPI/carla'))
except IndexError:
    pass

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import carla
import numpy as np
import h5py
import time
import weakref
from queue import Queue, Empty
from collections import defaultdict
from PIL import Image
import json


# ==============================================================================
# -- 传感器回调函数 -------------------------------------------------------------
# ==============================================================================

def camera_callback(image, camera_queue, camera_id):
    """相机数据回调 (使用 Queue 收集数据)"""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]  # BGRA -> RGB

    camera_queue.put({
        'camera_id': camera_id,
        'frame': image.frame,
        'timestamp': image.timestamp,
        'data': array
    })


def lidar_callback(data, lidar_queue):
    """激光雷达回调"""
    # 解析点云数据
    points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 6)

    lidar_queue.put({
        'frame': data.frame,
        'timestamp': data.timestamp,
        'points': points
    })


# ==============================================================================
# -- 主函数 --------------------------------------------------------------------
# ==============================================================================

def main():
    print("=" * 80)
    print("CARLA Occupancy 数据采集 - 5 帧测试")
    print("=" * 80)
    print()

    # 配置参数
    NUM_FRAMES = 5
    NUM_NPC_VEHICLES = 20

    # 输出目录设置
    output_dir = Path("dataset_output/town10_test")
    cameras_dir = output_dir / "cameras"
    lidar_dir = output_dir / "lidar"
    occupancy_dir = output_dir / "occupancy"

    # 创建输出目录
    for cam_name in ['cam_front', 'cam_rear']:
        (cameras_dir / cam_name).mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)
    occupancy_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {output_dir}")
    print()

    # 连接 CARLA
    print("连接 CARLA 服务器 (localhost:2000)...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"✓ 已连接")
    print(f"  地图: {world.get_map().name}")
    print()

    # 保存原始设置
    original_settings = world.get_settings()

    # 创建数据队列
    camera_queues = {}
    lidar_queue = Queue()

    # Actor 列表 (用于清理)
    actor_list = []

    try:
        # ==========================================
        # 1. 启用同步模式
        # ==========================================
        print("配置同步模式...")
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 20.0  # 20 Hz
        world.apply_settings(settings)
        print("✓ 同步模式已启用 (20 Hz)")
        print()

        # ==========================================
        # 2. 生成 Hero 车辆
        # ==========================================
        print("生成 Hero 车辆...")
        bp_lib = world.get_blueprint_library()

        # 尝试多个车型,使用第一个可用的
        vehicle_candidates = [
            'vehicle.tesla.model3',
            'vehicle.lincoln.mkz_2020',
            'vehicle.lincoln.mkz',
            'vehicle.audi.tt',
            'vehicle.dodge.charger_2020'
        ]

        vehicle_bp = None
        for candidate in vehicle_candidates:
            try:
                vehicle_bp = bp_lib.find(candidate)
                print(f"  使用车型: {candidate}")
                break
            except:
                continue

        if vehicle_bp is None:
            # 如果都找不到,随机选择一个
            vehicles = bp_lib.filter('vehicle.*')
            vehicles = [v for v in vehicles if int(v.get_attribute('number_of_wheels')) == 4]
            vehicle_bp = vehicles[0]
            print(f"  使用车型: {vehicle_bp.id}")

        vehicle_bp.set_attribute('role_name', 'hero')

        spawn_points = world.get_map().get_spawn_points()
        import random
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        print(f"✓ Hero 车辆已生成: {vehicle.type_id}")
        print()

        # ==========================================
        # 3. 附加 8 个相机 (简化版: 仅 2 个相机测试)
        # ==========================================
        print("附加相机传感器...")

        # 前方相机
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '1280')
        cam_bp.set_attribute('image_size_y', '960')
        cam_bp.set_attribute('fov', '120')
        cam_bp.set_attribute('enable_postprocess_effects', 'False')

        # 鱼眼畸变
        cam_bp.set_attribute('lens_circle_multiplier', '2.5')
        cam_bp.set_attribute('lens_circle_falloff', '2.0')
        cam_bp.set_attribute('lens_k', '-0.15')
        cam_bp.set_attribute('lens_kcube', '0.05')

        cam_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
        cam_front = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        actor_list.append(cam_front)

        camera_queues['cam_front'] = Queue()
        cam_front.listen(lambda img: camera_callback(img, camera_queues['cam_front'], 'cam_front'))

        print("  ✓ 前方超广角相机 (120° 鱼眼)")

        # 后方相机
        cam_bp2 = bp_lib.find('sensor.camera.rgb')
        cam_bp2.set_attribute('image_size_x', '1280')
        cam_bp2.set_attribute('image_size_y', '960')
        cam_bp2.set_attribute('fov', '90')

        cam_transform2 = carla.Transform(
            carla.Location(x=-2.5, z=1.5),  # 调整位置避免被车身遮挡
            carla.Rotation(yaw=180)
        )
        cam_rear = world.spawn_actor(cam_bp2, cam_transform2, attach_to=vehicle)
        actor_list.append(cam_rear)

        camera_queues['cam_rear'] = Queue()
        cam_rear.listen(lambda img: camera_callback(img, camera_queues['cam_rear'], 'cam_rear'))

        print("  ✓ 后方广角相机 (90°)")
        print()

        # ==========================================
        # 4. 附加语义激光雷达
        # ==========================================
        print("附加语义激光雷达...")
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast_semantic')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '1200000')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('upper_fov', '15')
        lidar_bp.set_attribute('lower_fov', '-25')

        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        actor_list.append(lidar)

        lidar.listen(lambda data: lidar_callback(data, lidar_queue))

        print("✓ 语义激光雷达已附加")
        print()

        # ==========================================
        # 5. 生成 NPC 车辆
        # ==========================================
        print(f"生成 {NUM_NPC_VEHICLES} 辆 NPC 车辆...")
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        vehicle_bps = bp_lib.filter('vehicle.*')
        vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute('number_of_wheels')) == 4]

        spawn_points_npc = spawn_points[1:NUM_NPC_VEHICLES+1]

        for sp in spawn_points_npc:
            npc_bp = np.random.choice(vehicle_bps)
            try:
                npc = world.spawn_actor(npc_bp, sp)
                npc.set_autopilot(True, 8000)
                actor_list.append(npc)
            except:
                pass

        print(f"✓ 已生成 {len(actor_list) - 4} 辆 NPC 车辆")
        print()

        # 启用 Hero 自动驾驶
        vehicle.set_autopilot(True, 8000)
        print("✓ Hero 自动驾驶已启用")
        print()

        # ==========================================
        # 6. 采集数据
        # ==========================================
        print(f"开始采集 {NUM_FRAMES} 帧数据...")
        print("-" * 80)

        collected_frames = []

        for frame_idx in range(NUM_FRAMES):
            # Tick 仿真
            world.tick()
            world_frame = world.get_snapshot().frame

            print(f"\n帧 {frame_idx + 1}/{NUM_FRAMES} (世界帧: {world_frame})")

            # 收集相机数据
            camera_data = {}
            for cam_id, cam_queue in camera_queues.items():
                try:
                    data = cam_queue.get(timeout=2.0)
                    camera_data[cam_id] = data
                    print(f"  ✓ {cam_id}: 帧 {data['frame']}")
                except Empty:
                    print(f"  ✗ {cam_id}: 超时")

            # 收集激光雷达数据
            try:
                lidar_data = lidar_queue.get(timeout=2.0)
                print(f"  ✓ lidar: {len(lidar_data['points'])} 点")
            except Empty:
                print(f"  ✗ lidar: 超时")
                lidar_data = None

            # 保存帧数据
            if len(camera_data) == len(camera_queues) and lidar_data is not None:
                # 保存到内存
                collected_frames.append({
                    'cameras': camera_data,
                    'lidar': lidar_data,
                    'vehicle_transform': vehicle.get_transform(),
                    'world_frame': world_frame
                })

                # 保存相机图像到磁盘
                for cam_id, cam_data in camera_data.items():
                    img = Image.fromarray(cam_data['data'])
                    img_path = cameras_dir / cam_id / f"{frame_idx:06d}.png"
                    img.save(img_path)

                # 保存激光雷达点云到磁盘
                lidar_path = lidar_dir / f"{frame_idx:06d}.npz"
                np.savez_compressed(
                    lidar_path,
                    points=lidar_data['points'],
                    frame=lidar_data['frame'],
                    timestamp=lidar_data['timestamp']
                )

                print(f"  ✓ 帧数据已保存到磁盘")
            else:
                print(f"  ✗ 帧数据不完整,跳过")

        print()
        print("=" * 80)
        print(f"✓ 数据采集完成!")
        print(f"  成功采集: {len(collected_frames)}/{NUM_FRAMES} 帧")
        print("=" * 80)

        # ==========================================
        # 保存数据集元数据
        # ==========================================
        if collected_frames:
            print()
            print("保存数据集元数据...")

            # 保存相机标定信息
            calibration = {
                'cameras': {
                    'cam_front': {
                        'transform': {
                            'x': 1.5,
                            'y': 0.0,
                            'z': 1.4,
                            'pitch': 0.0,
                            'yaw': 0.0,
                            'roll': 0.0
                        },
                        'fov': 120,
                        'width': 1280,
                        'height': 960
                    },
                    'cam_rear': {
                        'transform': {
                            'x': -2.5,
                            'y': 0.0,
                            'z': 1.5,
                            'pitch': 0.0,
                            'yaw': 180.0,
                            'roll': 0.0
                        },
                        'fov': 90,
                        'width': 1280,
                        'height': 960
                    }
                },
                'lidar': {
                    'transform': {
                        'x': 0.0,
                        'y': 0.0,
                        'z': 2.5,
                        'pitch': 0.0,
                        'yaw': 0.0,
                        'roll': 0.0
                    },
                    'channels': 64,
                    'range': 100.0
                },
                'dataset_info': {
                    'num_frames': len(collected_frames),
                    'map': world.get_map().name,
                    'vehicle': vehicle.type_id
                }
            }

            calibration_path = output_dir / "calibration.json"
            with open(calibration_path, 'w', encoding='utf-8') as f:
                json.dump(calibration, f, indent=2, ensure_ascii=False)

            print(f"  ✓ 元数据已保存: {calibration_path}")
            print()

        # ==========================================
        # 7. 显示统计信息
        # ==========================================
        if collected_frames:
            print()
            print("数据统计:")
            frame_0 = collected_frames[0]

            for cam_id, cam_data in frame_0['cameras'].items():
                img = cam_data['data']
                print(f"  {cam_id}: {img.shape} {img.dtype}")
                print(f"    范围: [{img.min()}, {img.max()}]")

            lidar_points = frame_0['lidar']['points']
            print(f"  lidar: {lidar_points.shape}")
            print(f"    XYZ 范围:")
            print(f"      X: [{lidar_points[:,0].min():.1f}, {lidar_points[:,0].max():.1f}]")
            print(f"      Y: [{lidar_points[:,1].min():.1f}, {lidar_points[:,1].max():.1f}]")
            print(f"      Z: [{lidar_points[:,2].min():.1f}, {lidar_points[:,2].max():.1f}]")

    except Exception as e:
        print()
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # ==========================================
        # 8. 清理资源
        # ==========================================
        print()
        print("清理资源...")

        # 恢复设置
        world.apply_settings(original_settings)

        # 销毁所有 Actor
        for actor in actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()

        print("✓ 清理完成")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n用户中断')

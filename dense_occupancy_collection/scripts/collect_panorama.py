#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA 360° 全景体素数据采集脚本
基于 CubeMap 方案生成稠密体素

使用方法:
    conda activate carla
    cd d:\\code\\carla
    python dense_occupancy_collection\\scripts\\collect_panorama_fixed.py --frames 5
"""

import sys
import os
from pathlib import Path

# 添加项目路径 (与 carla_data_collection 一致的方式)
try:
    # 优先添加 PythonAPI/carla
    # 这是 UE5.5 CARLA 0.10.0 的源码路径
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'PythonAPI/carla'))
    
    # 移除 dist 下的 egg 文件添加，避免加载到错误的 0.9.16 版本
    # carla_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'PythonAPI/carla/dist')
    # if os.path.exists(carla_path):
    #     for file in os.listdir(carla_path):
    #         if file.endswith('.egg'):
    #             sys.path.append(os.path.join(carla_path, file))
except IndexError:
    pass

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import carla
import numpy as np
import time
import argparse
from PIL import Image
import json
import cv2

from dense_occupancy_collection.processing.ground_truth_voxel_generator import GroundTruthVoxelGenerator
# from dense_occupancy_collection.processing.lidar_voxel_generator import LidarVoxelGenerator
from dense_occupancy_collection.config.occupancy_config import (
    X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION, CARLA_TO_OCCUPANCY_MAPPING
)

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 960

# 镜头畸变参数 (基于 CARLA 文档)
# 鱼眼镜头参数 (适用于 120度+ 超广角)
FISHEYE_DISTORTION = {
    'lens_circle_multiplier': 3.0,  # 使用典型值 3.0
    'lens_circle_falloff': 3.0,     # 使用典型值 3.0
    'lens_k': -1.0,                 # 桶形畸变系数
    'lens_kcube': 0.0,
    'lens_x_size': 0.0,
    'lens_y_size': 0.0
}

# 广角镜头参数 (适用于 90度 广角) - 轻微畸变以减少拉伸
WIDE_ANGLE_DISTORTION = {
    'lens_circle_multiplier': 0.0,
    'lens_circle_falloff': 5.0,
    'lens_k': -0.2,                 # 非常轻微的畸变
    'lens_kcube': 0.0,
    'lens_x_size': 0.0,
    'lens_y_size': 0.0
}

# 特斯拉 8 相机配置 - 与 carla_data_collection 保持一致
# 1280x960 (960p) 4:3 宽高比
TESLA_CAMERA_CONFIGS = [
    # --- 前视相机组 (Windshield Triple Cam) ---
    {
        'id': 'cam_front_main',  # Main
        'fov': 50,  # 主摄标准 50度
        'position': {'x': 1.0, 'y': 0.0, 'z': 1.6},
        'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0},
        'description': '前视主摄 (Main)',
        'lens_distortion': None
    },
    {
        'id': 'cam_front_wide',  # Wide
        'fov': 120, # 广角 120度
        'position': {'x': 1.0, 'y': 0.0, 'z': 1.6},
        'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0},
        'description': '前视广角 (Wide/Fisheye)',
        'lens_distortion': FISHEYE_DISTORTION
    },
    {
        'id': 'cam_front_narrow', # Narrow
        'fov': 35,  # 长焦 35度
        'position': {'x': 1.0, 'y': 0.0, 'z': 1.6},
        'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0},
        'description': '前视长焦 (Narrow)',
        'lens_distortion': None
    },
    
    # --- 侧向前视 (B-Pillar) ---
    # B柱位置，向前看，用于路口检测
    {
        'id': 'cam_left_pillar',
        'fov': 80,
        'position': {'x': 0.0, 'y': -0.9, 'z': 1.7}, # B柱高位
        'rotation': {'pitch': 0, 'yaw': -60, 'roll': 0}, # 指向左前
        'description': '左侧 B 柱 (Left Pillar)',
        'lens_distortion': None
    },
    {
        'id': 'cam_right_pillar',
        'fov': 80,
        'position': {'x': 0.0, 'y': 0.9, 'z': 1.7}, # B柱高位
        'rotation': {'pitch': 0, 'yaw': 60, 'roll': 0}, # 指向右前
        'description': '右侧 B 柱 (Right Pillar)',
        'lens_distortion': None
    },

    # --- 侧向后视 (Repeater/Fender) ---
    # 翼子板位置，向后看，用于盲区/变道
    {
        'id': 'cam_left_repeater',
        'fov': 100,
        'position': {'x': 1.2, 'y': -0.9, 'z': 1.0}, # 翼子板低位
        'rotation': {'pitch': 0, 'yaw': -160, 'roll': 0}, # 指向左后
        'description': '左侧翼子板 (Left Repeater)',
        'lens_distortion': None
    },
    {
        'id': 'cam_right_repeater',
        'fov': 100,
        'position': {'x': 1.2, 'y': 0.9, 'z': 1.0}, # 翼子板低位
        'rotation': {'pitch': 0, 'yaw': 160, 'roll': 0}, # 指向右后
        'description': '右侧翼子板 (Right Repeater)',
        'lens_distortion': None
    },

    # --- 后视 (Backup) ---
    {
        'id': 'cam_rear',
        'fov': 120,
        'position': {'x': -2.5, 'y': 0.0, 'z': 1.2}, # 车尾
        'rotation': {'pitch': -5, 'yaw': 180, 'roll': 0}, # 略微向下
        'description': '后视 (Rear)',
        'lens_distortion': FISHEYE_DISTORTION
    }
]
from dense_occupancy_collection.sensors.rgb_camera_manager import RGBCameraManager
# from dense_occupancy_collection.sensors.panorama_manager import PanoramaSensorManager
from dense_occupancy_collection.sensors.semantic_lidar_sensor import SemanticLidarSensor
from dense_occupancy_collection.processing.ground_truth_voxel_generator import GroundTruthVoxelGenerator
from dense_occupancy_collection.processing.lidar_voxel_generator import LidarVoxelGenerator

import queue


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CARLA 360° 全景体素数据采集')
    parser.add_argument('--host', default='localhost', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口')
    parser.add_argument('--frames', type=int, default=5, help='采集帧数')
    parser.add_argument('--output', default='dataset_output', help='输出目录')
    parser.add_argument('--town', default='Town10HD_Opt', help='地图名称')
    return parser.parse_args()


def setup_hero_vehicle(world):
    """创建hero车辆"""
    bp_lib = world.get_blueprint_library()

    # 尝试多种车辆
    vehicle_candidates = [
        'vehicle.tesla.model3',
        'vehicle.lincoln.mkz_2020',
        'vehicle.lincoln.mkz',
        'vehicle.audi.tt'
    ]

    vehicle_bp = None
    for candidate in vehicle_candidates:
        try:
            vehicle_bp = bp_lib.find(candidate)
            break
        except:
            continue

    if vehicle_bp is None:
        # 备选: 任意4轮车
        vehicles = bp_lib.filter('vehicle.*')
        for v in vehicles:
            if int(v.get_attribute('number_of_wheels')) == 4:
                vehicle_bp = v
                break

    if vehicle_bp.has_attribute('role_name'):
        vehicle_bp.set_attribute('role_name', 'hero')

    spawn_points = world.get_map().get_spawn_points()
    
    # 尝试找到一个空闲的生成点
    vehicle = None
    for point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, point)
        if vehicle is not None:
            break
            
    if vehicle is None:
        raise RuntimeError("无法找到空闲的生成点生成Hero车辆")
        
    print(f"✓ Hero车辆已生成: {vehicle.type_id}")

    return vehicle


def spawn_traffic(world, tm_port, num_vehicles=20):
    """生成交通NPC"""
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    actors = []

    vehicle_bps = bp_lib.filter('vehicle.*')
    vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute('number_of_wheels')) == 4]

    for i in range(min(num_vehicles, len(spawn_points))):
        bp = np.random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = np.random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        vehicle = world.try_spawn_actor(bp, spawn_points[i])
        if vehicle is not None:
            vehicle.set_autopilot(True, tm_port)
            actors.append(vehicle)

    print(f"✓ 已生成 {len(actors)} 辆NPC车辆")
    return actors


def prepare_rgb_camera_configs():
    """准备RGB相机配置"""
    configs = []
    for cam_cfg in TESLA_CAMERA_CONFIGS:
        config = {
            'id': cam_cfg['id'],
            'fov': cam_cfg['fov'],
            'x': cam_cfg['position']['x'],
            'y': cam_cfg['position']['y'],
            'z': cam_cfg['position']['z'],
            'pitch': cam_cfg['rotation']['pitch'],
            'yaw': cam_cfg['rotation']['yaw'],
            'roll': cam_cfg['rotation']['roll'],
            'image_size_x': IMAGE_WIDTH,
            'image_size_y': IMAGE_HEIGHT,
            'sensor_tick': 0.1,
            'lens_distortion': cam_cfg['lens_distortion']
        }
        configs.append(config)
    return configs


def main():
    args = parse_args()

    # 准备输出目录
    output_dir = Path(args.output)
    if output_dir.exists():
        import shutil
        print(f"⚠ 清理旧输出目录: {output_dir}")
        shutil.rmtree(output_dir)
        time.sleep(1.0)

    cameras_dir = output_dir / 'cameras'
    depth_dir = output_dir / 'depth'
    semantic_color_dir = output_dir / 'semantic_color'
    occupancy_dir = output_dir / 'occupancy'
    lidar_dir = output_dir / 'lidar_semantic'

    for cam_cfg in TESLA_CAMERA_CONFIGS:
        (cameras_dir / cam_cfg['id']).mkdir(parents=True, exist_ok=True)

    depth_dir.mkdir(parents=True, exist_ok=True)
    semantic_color_dir.mkdir(parents=True, exist_ok=True)
    occupancy_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CARLA 360° 全景体素数据采集")
    print(f"{'='*60}")
    print(f"服务器: {args.host}:{args.port}")
    print(f"地图: {args.town}")
    print(f"采集帧数: {args.frames}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")

    client = None
    world = None
    vehicle = None
    rgb_manager = None
    # pano_manager = None
    lidar_sensor = None
    traffic_actors = []

    try:
        # 连接CARLA
        print("⏳ 连接CARLA服务器...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        world = client.get_world()
        print(f"✓ 已连接到CARLA: {world.get_map().name}")

        # 加载地图
        if args.town and args.town not in world.get_map().name:
            print(f"⏳ 加载地图: {args.town}...")
            try:
                world = client.load_world(args.town)
                time.sleep(2.0)
            except RuntimeError:
                print(f"⚠ 地图 {args.town} 加载失败，使用当前地图")

        # 设置天气
        world.set_weather(carla.WeatherParameters.ClearNoon)
        print("✓ 天气已设置为 ClearNoon")

        # Traffic Manager
        tm_port = 8010
        traffic_manager = client.get_trafficmanager(tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_synchronous_mode(True)
        print(f"✓ Traffic Manager 已启动 (端口 {tm_port})")

        # 同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        print("✓ 同步模式已启用 (20 FPS)")

        # 创建hero车辆
        print("\n⏳ 生成hero车辆...")
        vehicle = setup_hero_vehicle(world)

        # 生成NPC (减少数量以防卡顿)
        print("\n⏳ 生成交通NPC...", flush=True)
        traffic_actors = spawn_traffic(world, tm_port, num_vehicles=15)

        # 等待稳定
        print("\n⏳ 等待场景稳定...")
        for _ in range(20):
           world.tick()
        time.sleep(1.0)
        print("✓ 场景已稳定", flush=True)

        # 创建传感器
        print("\n⏳ 创建传感器...", flush=True)
        camera_configs = prepare_rgb_camera_configs()

        # 1. RGB相机 (8个)
        rgb_manager = RGBCameraManager(world, vehicle, camera_configs)

        # 2. 全景相机 (CubeMap: 6个深度 + 6个语义)
        # pano_manager = PanoramaSensorManager(world, vehicle)

        # 3. 语义激光雷达
        lidar_sensor = SemanticLidarSensor(world, vehicle)
        lidar_sensor.listen_to_queue()

        # 等待传感器初始化
        print("\n⏳ 等待传感器初始化...", flush=True)
        for i in range(10):
            print(f"   Tick {i+1}/10", flush=True)
            world.tick()
            time.sleep(0.1) # 增加延时防止过载
        time.sleep(0.5)

        print("正在初始化体素生成器...", flush=True)
        # 创建体素生成器 (使用 Ground Truth 生成器)
        # voxel_generator = LidarVoxelGenerator(X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION)
        voxel_generator = GroundTruthVoxelGenerator(X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION)
        print("体素生成器初始化完成 (Ground Truth Mode)", flush=True)

        # 开始采集
        print(f"\n{'='*60}")
        print("开始数据采集")
        print(f"{'='*60}\n")

        # 确保输出目录存在
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
            (cameras_dir / cam_cfg['id']).mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            semantic_color_dir.mkdir(parents=True, exist_ok=True)
            occupancy_dir.mkdir(parents=True, exist_ok=True)
            lidar_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in range(args.frames):
            print(f"📷 采集帧 {frame_idx + 1}/{args.frames}...", flush=True)
            
            # Tick
            # print(f"   [DEBUG] Ticking world...", flush=True)
            world.tick()

            # 获取RGB数据
            if rgb_manager:
                print(f"   [DEBUG] Getting RGB data...", flush=True)
                # 增加timeout，因为可能需要等待新数据
                rgb_data = rgb_manager.get_data(timeout=2.0)
                if rgb_data is None:
                    print(f"⚠ RGB数据超时，跳过帧 {frame_idx}")
                    continue
            
            # 获取全景数据 (已禁用)
            # pano_data = pano_manager.get_panorama_frame(timeout=2.0)
            # if pano_data is None:
            #     print(f"⚠ 全景数据超时，跳过帧 {frame_idx}")
            #     continue

            # 获取LiDAR数据
            try:
                lidar_data_dict = lidar_sensor.data_queue.get(timeout=2.0)
                lidar_raw = lidar_data_dict['raw_data']
                lidar_timestamp = lidar_data_dict['timestamp']
                lidar_points, lidar_labels = lidar_sensor.parse_lidar_data(lidar_raw)
                
                # 保存LiDAR数据
                lidar_save_path = lidar_dir / f"{frame_idx:06d}.npz"
                np.savez_compressed(
                    lidar_save_path,
                    points=lidar_points,
                    labels=lidar_labels
                )
                # print(f"   ✓ LiDAR数据已保存")
                
            except queue.Empty:
                print(f"⚠ LiDAR数据超时，跳过帧 {frame_idx}")
                continue

            # 保存RGB图像 (8-bit)
            print(f"   [DEBUG] Saving RGB images...", flush=True)
            for cam_cfg in camera_configs:
                cam_id = cam_cfg['id']
                rgb_array = rgb_data[cam_id]['data']  # (H, W, 3) RGB

                # RGB -> BGR for OpenCV
                bgr_array = rgb_array[:, :, ::-1]

                img_path = cameras_dir / cam_id / f"{frame_idx:06d}.png"
                cv2.imwrite(str(img_path), bgr_array)

            print(f"   ✓ RGB图像已保存 (8个相机)")

            # 保存全景深度图 (黑白) - 禁用
            # depth_pano = pano_data['depth_pano']
            # viz_depth = np.clip(depth_pano / 100.0 * 255.0, 0, 255).astype(np.uint8)
            # depth_img = Image.fromarray(viz_depth, mode='L')
            # depth_path = depth_dir / f"{frame_idx:06d}.png"
            # depth_img.save(depth_path)

            # print(f"   ✓ 全景深度图已保存")

            # 保存全景语义图 (彩色) - 禁用
            # semantic_pano = pano_data['semantic_pano']
            # h, w = semantic_pano.shape
            # sem_color = np.zeros((h, w, 3), dtype=np.uint8)

            # # 简单彩色映射 (使用配置文件的映射)
            # from dense_occupancy_collection.config.occupancy_config import OCCUPANCY_COLORS
            # for class_id in range(len(OCCUPANCY_COLORS)):
            #     mask = (semantic_pano == class_id)
            #     sem_color[mask] = OCCUPANCY_COLORS[class_id]

            # sem_color_img = Image.fromarray(sem_color)
            # sem_color_path = semantic_color_dir / f"{frame_idx:06d}.png"
            # sem_color_img.save(sem_color_path)

            # print(f"   ✓ 全景语义图已保存")

            # 生成体素 (Ground Truth 方案)
            # 传入: world, vehicle (不再依赖 LiDAR 点云生成体素，但仍保存 LiDAR 点云)
            occupancy, mask = voxel_generator.generate(world, vehicle)
            
            voxel_stats = voxel_generator.get_statistics(occupancy, mask)

            print(f"   ✓ 体素网格: {voxel_stats['observed_voxels']} / {voxel_stats['total_voxels']} "
                  f"({voxel_stats['observation_rate']*100:.1f}% 观测率)")

            # 保存体素
            occupancy_path = occupancy_dir / f"{frame_idx:06d}.npz"
            voxel_generator.save_to_npz(
                occupancy_path, occupancy, mask,
                metadata={
                    'frame': frame_idx,
                    'timestamp': lidar_timestamp,
                    'map': args.town
                }
            )

            print(f"   ✓ 体素文件已保存: {occupancy_path.name}\n")

    except KeyboardInterrupt:
        print("\n⚠ 用户中断")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print(f"\n{'='*60}")
        print("清理资源")
        print(f"{'='*60}\n")

        if rgb_manager:
            rgb_manager.destroy()
        # if pano_manager:
        #     pano_manager.destroy()
        if lidar_sensor:
            lidar_sensor.destroy()
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
            # pass

    print("\n✅ 数据采集完成!")


if __name__ == '__main__':
    main()

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
    # sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'PythonAPI/carla'))
    
    # ⭐ 源码构建版: 直接添加编译好的 .whl 文件 (Python 3.10)
    # 路径: d:\code\carla\Build\PythonAPI\dist\carla-0.10.0-cp310-cp310-win_amd64.whl
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_dist_path = os.path.join(project_root, 'Build', 'PythonAPI', 'dist')
    
    found_whl = False
    if os.path.exists(build_dist_path):
        for file in os.listdir(build_dist_path):
            if file.endswith('.whl'):
                whl_path = os.path.join(build_dist_path, file)
                print(f"[Import] Found CARLA wheel: {whl_path}")
                sys.path.append(whl_path)
                found_whl = True
                break
    
    if not found_whl:
        # Fallback to source path if no wheel found (might fail if .pyd not there)
        src_path = os.path.join(project_root, 'PythonAPI', 'carla')
        print(f"[Import] No wheel found, adding source path: {src_path}")
        sys.path.append(src_path)

    # 添加 PythonAPI 目录以支持 agents
    agents_path = os.path.join(project_root, 'PythonAPI', 'carla')
    sys.path.append(agents_path)
    
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
    X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION, CARLA_TO_OCCUPANCY_MAPPING,
    VISIBILITY_LIDAR_CONFIG, VISIBILITY_CONFIG, DEPTH_CAMERA_CONFIG  # ⭐ 新增配置
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


class DepthCameraManager:
    """管理6路 Cube Map 深度相机"""
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.cameras = []
        self.queues = {}
        self.blueprint = self._setup_blueprint()
        self._spawn_cameras()
        
    def _setup_blueprint(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        bp.set_attribute('image_size_x', str(DEPTH_CAMERA_CONFIG['width']))
        bp.set_attribute('image_size_y', str(DEPTH_CAMERA_CONFIG['height']))
        bp.set_attribute('fov', str(DEPTH_CAMERA_CONFIG['fov']))
        return bp
        
    def _spawn_cameras(self):
        print(f"\n[DepthCamera] 正在创建 6 路深度相机 (Cube Map)...")
        for cam_conf in DEPTH_CAMERA_CONFIG['cameras']:
            transform = carla.Transform(
                carla.Location(**cam_conf['pos']),
                carla.Rotation(**cam_conf['rot'])
            )
            sensor = self.world.spawn_actor(self.blueprint, transform, attach_to=self.vehicle)
            q = queue.Queue()
            sensor.listen(q.put)
            
            self.cameras.append(sensor)
            self.queues[cam_conf['id']] = q
            print(f"  - {cam_conf['id']} created")
            
    def get_data(self, timeout=2.0):
        """获取所有深度图和相机变换"""
        depth_maps = []
        cam_transforms = []
        
        # 必须按顺序获取: Front, Right, Back, Left, Up, Down
        cam_ids = [c['id'] for c in DEPTH_CAMERA_CONFIG['cameras']]
        
        for cid in cam_ids:
            try:
                image = self.queues[cid].get(timeout=timeout)
                
                # Decode Depth
                # format: BGRA, float32
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                
                # (R + G*256 + B*256*256) / (256**3 - 1) * 1000
                normalized = (array[:,:,2] + array[:,:,1]*256.0 + array[:,:,0]*256.0*256.0) / (256.0**3 - 1)
                depth_meters = normalized * 1000.0
                
                depth_maps.append(depth_meters)
                cam_transforms.append(image.transform.get_matrix())
                
            except queue.Empty:
                print(f"[Error] Depth camera {cid} timeout")
                return None
                
        return {
            'depth_maps': np.stack(depth_maps), # (6, H, W)
            'cam_transforms': np.stack(cam_transforms) # (6, 4, 4)
        }
        
    def destroy(self):
        for cam in self.cameras:
            if cam.is_alive:
                cam.destroy()


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
    print(f"  [调试] 地图总spawn点数: {len(spawn_points)}")

    # ⭐ 过滤掉原点附近的spawn点，避免(0,0,0)位置
    valid_spawn_points = [
        point for point in spawn_points
        if abs(point.location.x) > 5 or abs(point.location.y) > 5
    ]
    print(f"  [调试] 过滤后有效spawn点数: {len(valid_spawn_points)}")

    if not valid_spawn_points:
        print("⚠ 警告: 没有找到有效的spawn点，使用原始spawn点列表")
        valid_spawn_points = spawn_points

    # 显示前5个spawn点位置
    print("  [调试] 前5个有效spawn点位置:")
    for i, point in enumerate(valid_spawn_points[:5]):
        loc = point.location
        print(f"    #{i+1}: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

    # 尝试找到一个空闲的生成点
    vehicle = None
    for idx, point in enumerate(valid_spawn_points):
        vehicle = world.try_spawn_actor(vehicle_bp, point)
        if vehicle is not None:
            # ⭐ CRITICAL: spawn后必须tick几次让物理系统稳定，否则车辆会被传送到(0,0,0)
            for _ in range(5):
                world.tick()

            loc = vehicle.get_location()
            print(f"✓ Hero车辆已生成: {vehicle.type_id}")
            print(f"  尝试spawn点索引: #{idx+1}/{len(valid_spawn_points)}")
            print(f"  最终位置: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

            # ⭐ 验证位置不是原点
            if abs(loc.x) < 1.0 and abs(loc.y) < 1.0:
                print(f"  ⚠ 警告: Hero位置接近原点！这可能导致行人生成失败！")
                # 如果还是在原点，销毁重试下一个spawn点
                vehicle.destroy()
                vehicle = None
                continue
            break

    if vehicle is None:
        raise RuntimeError("无法找到空闲的生成点生成Hero车辆")

    return vehicle


def spawn_traffic(world, tm_port, num_vehicles=50, num_walkers=20, hero_location=None):
    """
    生成丰富的交通NPC，覆盖17类Occupancy分类

    包括：
    - Car (4): 普通轿车、SUV
    - Bus (3): 公交车
    - Truck (10): 卡车、货车
    - Bicycle (2): 自行车
    - Motorcycle (6): 摩托车
    - Pedestrian (7): 行人

    Args:
        world: CARLA world对象
        tm_port: Traffic Manager端口
        num_vehicles: 车辆数量
        num_walkers: 行人数量
        hero_location: hero车辆位置(carla.Location)，用于在hero附近spawn行人
    """
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    all_actors = []

    # === 1. 车辆NPC（按类别分配） ===
    print(f"  正在生成车辆...")

    # 定义车辆类型及其占比（⭐ 增加自行车和摩托车比例）
    vehicle_categories = {
        'car': {          # 轿车/SUV - 占比50%
            'filters': ['vehicle.audi.*', 'vehicle.bmw.*', 'vehicle.mercedes.*',
                       'vehicle.tesla.*', 'vehicle.toyota.*', 'vehicle.nissan.*',
                       'vehicle.dodge.charger*', 'vehicle.lincoln.*', 'vehicle.jeep.*'],
            'ratio': 0.50
        },
        'truck': {        # 卡车/货车 - 占比10%
            'filters': ['vehicle.carlamotors.firetruck', 'vehicle.ford.ambulance',
                       'vehicle.carlamotors.carlacola', 'vehicle.carlamotors.european_hgv',
                       'vehicle.tesla.cybertruck'],
            'ratio': 0.10
        },
        'bus': {          # 公交车 - 占比10%
            'filters': ['vehicle.mitsubishi.fusorosa'],
            'ratio': 0.10
        },
        'bicycle': {      # 自行车 - 占比15% (⭐ 增加)
            'filters': ['vehicle.bh.crossbike', 'vehicle.diamondback.century',
                       'vehicle.gazelle.omafiets'],
            'ratio': 0.15
        },
        'motorcycle': {   # 摩托车 - 占比15% (⭐ 增加)
            'filters': ['vehicle.harley*', 'vehicle.kawasaki.*', 'vehicle.yamaha.*',
                       'vehicle.vespa.*'],
            'ratio': 0.15
        }
    }

    # 按类别生成车辆
    spawned_vehicles = []
    spawn_idx = 0

    for category, info in vehicle_categories.items():
        num_this_category = int(num_vehicles * info['ratio'])
        if num_this_category == 0 and info['ratio'] > 0:
            num_this_category = 1  # 确保每个类别至少有1个

        # 收集该类别的所有可用蓝图
        category_bps = []
        for filter_pattern in info['filters']:
            category_bps.extend(list(bp_lib.filter(filter_pattern)))

        if not category_bps:
            print(f"    ⚠ 警告: 找不到{category}类型的车辆蓝图")
            continue

        # 生成该类别的车辆
        count = 0
        for _ in range(num_this_category):
            if spawn_idx >= len(spawn_points):
                break

            bp = np.random.choice(category_bps)

            # 设置颜色
            if bp.has_attribute('color'):
                color = np.random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)

            # 生成车辆
            vehicle = world.try_spawn_actor(bp, spawn_points[spawn_idx])
            spawn_idx += 1

            if vehicle is not None:
                vehicle.set_autopilot(True, tm_port)
                spawned_vehicles.append(vehicle)
                count += 1

        if count > 0:
            print(f"    ✓ {category}: {count} 辆")

    all_actors.extend(spawned_vehicles)

    # === 2. 行人NPC ===
    print(f"  正在生成行人...")

    # 获取行人蓝图
    walker_bps = list(bp_lib.filter('walker.pedestrian.*'))
    walker_controller_bp = bp_lib.find('controller.ai.walker')

    spawned_walkers = []
    walker_controllers = []

    # ⭐ 优先在hero车辆附近生成行人，确保在60m范围内
    if hero_location is not None:
        print(f"    [行人Spawn] Hero位置: ({hero_location.x:.1f}, {hero_location.y:.1f}, {hero_location.z:.1f})")
        # 在hero周围15-45m半径范围内均匀分布生成行人（避免太近和太远）
        spawn_attempts = 0
        max_attempts = num_walkers * 5  # 最多尝试5倍数量

        while len(spawned_walkers) < num_walkers and spawn_attempts < max_attempts:
            # 极坐标：距离15-45m，角度0-360度
            distance = np.random.uniform(15, 45)
            angle = np.random.uniform(0, 2 * np.pi)
            offset_x = distance * np.cos(angle)
            offset_y = distance * np.sin(angle)

            spawn_point = carla.Transform(
                carla.Location(
                    x=hero_location.x + offset_x,
                    y=hero_location.y + offset_y,
                    z=hero_location.z + 1.5  # ⭐ 提高z坐标，避免地面碰撞
                )
            )

            walker_bp = np.random.choice(walker_bps)

            # 随机设置行人属性
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')

            # 生成行人
            walker = world.try_spawn_actor(walker_bp, spawn_point)
            spawn_attempts += 1

            if walker is not None:
                spawned_walkers.append(walker)
                print(f"    [行人Spawn] ✓ 成功 #{len(spawned_walkers)}: 距离={distance:.1f}m, 位置=({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")

                # 为行人添加AI控制器
                controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
                walker_controllers.append(controller)

        if spawn_attempts >= max_attempts:
            print(f"    ⚠ 警告: 行人spawn达到最大尝试次数 ({max_attempts}次)，只成功spawn {len(spawned_walkers)} 个")
    else:
        # 兜底：在车辆生成点附近随机偏移生成行人
        print(f"    [行人Spawn] ⚠ Hero位置为None，使用兜底逻辑")
        for i in range(num_walkers):
            if i < len(spawn_points):
                loc = spawn_points[i].location
                # 随机偏移5-10米到人行道
                offset_x = np.random.uniform(-10, 10)
                offset_y = np.random.uniform(-10, 10)
                spawn_point = carla.Transform(
                    carla.Location(x=loc.x + offset_x, y=loc.y + offset_y, z=loc.z + 1.0)
                )

                walker_bp = np.random.choice(walker_bps)
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')

                walker = world.try_spawn_actor(walker_bp, spawn_point)
                if walker is not None:
                    spawned_walkers.append(walker)
                    controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
                    walker_controllers.append(controller)

    all_actors.extend(spawned_walkers)
    all_actors.extend(walker_controllers)

    # 启动行人AI - ⚠ CRITICAL: 不使用go_to_location，它在同步模式下会导致CARLA服务端崩溃
    print(f"    [行人AI] 正在启动 {len(walker_controllers)} 个行人控制器...")

    # ⭐ 一次性启动所有控制器（不需要分批，go_to_location才是问题所在）
    for i, controller in enumerate(walker_controllers):
        try:
            controller.start()
            # ⭐ 只设置速度，让行人自然行走（不调用go_to_location）
            controller.set_max_speed(1.0 + np.random.random())  # 1.0-2.0 m/s
        except Exception as e:
            print(f"    ⚠ 警告: 行人控制器 #{i+1} 启动失败: {e}")

    # tick几次让AI系统稳定
    for _ in range(5):
        world.tick()

    print(f"    [行人AI] ✓ 已成功启动 {len(walker_controllers)} 个控制器")

    print(f"    ✓ 行人: {len(spawned_walkers)} 人")

    print(f"✓ 已生成 {len(spawned_vehicles)} 辆车辆 + {len(spawned_walkers)} 个行人")
    print(f"  总计: {len(all_actors)} 个NPC Actor")

    return all_actors


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
    depth_manager = None
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

        # 生成丰富的NPC（覆盖17类occupancy分类）
        # ⭐ 传入hero位置，确保行人在hero附近生成
        # ⭐ 增加行人和自行车等NPC数量
        print("\n⏳ 生成交通NPC...", flush=True)
        # ⭐ 减少行人数量到10，避免AI控制器启动崩溃
        traffic_actors = spawn_traffic(world, tm_port, num_vehicles=30, num_walkers=10,
                                      hero_location=vehicle.get_location())

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

        # 3. 可见性传感器 (LiDAR 或 Depth Camera)
        filter_mode = VISIBILITY_CONFIG.get('filter_mode', 'lidar')
        print(f"\n[Visibility] 使用过滤模式: {filter_mode}")

        if filter_mode == 'lidar':
            lidar_sensor = SemanticLidarSensor(world, vehicle, config=VISIBILITY_LIDAR_CONFIG)
            lidar_sensor.listen_to_queue()
        elif filter_mode == 'depth_camera':
            depth_manager = DepthCameraManager(world, vehicle)

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

            # 获取可见性数据
            visibility_data = None
            lidar_timestamp = 0.0

            if filter_mode == 'lidar':
                try:
                    lidar_data_dict = lidar_sensor.data_queue.get(timeout=2.0)
                    lidar_raw = lidar_data_dict['raw_data']
                    lidar_timestamp = lidar_data_dict['timestamp']
                    lidar_points, lidar_labels = lidar_sensor.parse_lidar_data(lidar_raw)

                    # 保存LiDAR点云数据
                    lidar_save_path = lidar_dir / f"{frame_idx:06d}.npz"
                    np.savez_compressed(
                        lidar_save_path,
                        points=lidar_points,
                        labels=lidar_labels
                    )
                    
                    visibility_data = lidar_raw
                except queue.Empty:
                    print(f"⚠ LiDAR数据超时，跳过帧 {frame_idx}")
                    continue
            
            elif filter_mode == 'depth_camera':
                visibility_data = depth_manager.get_data()
                if visibility_data is None:
                    print(f"⚠ Depth Camera数据超时，跳过帧 {frame_idx}")
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

            # 生成体素 (Ground Truth 方案 + 可见性过滤)
            # 传入 visibility_data (可能是 LiDAR RawData 或 Depth Map Dict)
            occupancy, actor_ids, mask = voxel_generator.generate(
                world, vehicle, visibility_data=visibility_data
            )

            voxel_stats = voxel_generator.get_statistics(occupancy, mask)

            print(f"   ✓ 体素网格: {voxel_stats['observed_voxels']} / {voxel_stats['total_voxels']} "
                  f"({voxel_stats['observation_rate']*100:.1f}% 观测率)")

            # 保存体素（包含actor_ids）
            occupancy_path = occupancy_dir / f"{frame_idx:06d}.npz"
            voxel_generator.save_to_npz(
                occupancy_path, occupancy, actor_ids, mask,  # ⭐ 增加actor_ids参数
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
        if depth_manager:
            depth_manager.destroy()
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

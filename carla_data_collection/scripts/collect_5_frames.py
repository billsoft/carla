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
import threading
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
from data.occupancy_generator import OccupancyGenerator
from config.camera_config import TESLA_CAMERA_CONFIGS, CAMERA_SENSOR_CONFIG

# ==============================================================================
# -- 异步保存 Worker -------------------------------------------------------------
# ==============================================================================

class DataSaver(threading.Thread):
    """异步数据保存线程"""
    def __init__(self, output_dir, occ_gen):
        super().__init__()
        self.queue = Queue()
        self.output_dir = Path(output_dir)
        self.occ_gen = occ_gen
        self.cameras_dir = self.output_dir / "cameras"
        self.lidar_dir = self.output_dir / "lidar"
        self.occupancy_dir = self.output_dir / "occupancy"
        self.running = True
        self.daemon = True  # 设置为守护线程

    def run(self):
        while self.running or not self.queue.empty():
            try:
                # 获取数据 (带超时，以便检查 running 标志)
                item = self.queue.get(timeout=1.0)
            except Empty:
                continue

            try:
                self._save_frame(item)
                self.queue.task_done()
            except Exception as e:
                print(f"\n[Saver] 保存失败: {e}")

    def _save_frame(self, item):
        frame_idx = item['frame_idx']
        camera_data = item['cameras']
        lidar_data = item['lidar']
        
        # 1. 保存相机图像
        for cam_id, cam_data in camera_data.items():
            img = Image.fromarray(cam_data['data'])
            img_path = self.cameras_dir / cam_id / f"{frame_idx:06d}.png"
            img.save(img_path)

        # 2. 保存激光雷达
        lidar_path = self.lidar_dir / f"{frame_idx:06d}.npz"
        np.savez_compressed(
            lidar_path,
            points=lidar_data['points'],
            frame=lidar_data['frame'],
            timestamp=lidar_data['timestamp']
        )

        # 3. 生成并保存 Occupancy (在后台线程计算，不阻塞主循环)
        try:
            points_raw = lidar_data['points'] # (N, 6)
            xyz_sensor = points_raw[:, :3]
            semantic_tags = points_raw[:, 5].astype(int)
            
            # Lidar 安装在 (0, 0, 2.5)
            xyz_ego = xyz_sensor + np.array([0, 0, 2.5])
            
            occupancy_grid, occ_mask = self.occ_gen.generate(xyz_ego, semantic_tags)
            
            occ_path = self.occupancy_dir / f"{frame_idx:06d}.npz"
            self.occ_gen.save_occupancy(occupancy_grid, occ_mask, str(occ_path))
            print(f"  [Saver] 帧 {frame_idx} 数据保存完成")
        except Exception as e:
            print(f"  [Saver] 帧 {frame_idx} Occupancy 生成失败: {e}")

    def stop(self):
        self.running = False
        # 等待队列处理完毕
        if not self.queue.empty():
            print(f"[Saver] 等待剩余 {self.queue.qsize()} 帧保存...")
            self.queue.join()

# ==============================================================================
# -- 传感器回调函数 -------------------------------------------------------------
# ==============================================================================

def camera_callback(image, camera_queue, camera_id):
    """相机数据回调 (使用 Queue 收集数据)"""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    # CARLA 原始数据是 BGRA 格式
    array = array.reshape((image.height, image.width, 4))
    # BGRA -> RGB: 取前3个通道并反转顺序 (B,G,R) -> (R,G,B)
    array = array[:, :, :3][:, :, ::-1]

    camera_queue.put({
        'camera_id': camera_id,
        'frame': image.frame,
        'timestamp': image.timestamp,
        'data': array
    })


def lidar_callback(data, lidar_queue):
    """激光雷达回调"""
    # 解析点云数据 (SemanticLidar 使用混合类型)
    data_struct = np.frombuffer(data.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('cos_angle', np.float32), ('object_idx', np.uint32), ('tag', np.uint32)
    ]))
    
    # 转换为 float32 数组 (N, 6) 以保持兼容性
    # 注意: object_idx 如果很大可能会有精度损失,但 tag 通常很小没问题
    points = np.zeros((data_struct.shape[0], 6), dtype=np.float32)
    points[:, 0] = data_struct['x']
    points[:, 1] = data_struct['y']
    points[:, 2] = data_struct['z']
    points[:, 3] = data_struct['cos_angle']
    points[:, 4] = data_struct['object_idx'].astype(np.float32)
    points[:, 5] = data_struct['tag'].astype(np.float32)

    lidar_queue.put({
        'frame': data.frame,
        'timestamp': data.timestamp,
        'points': points
    })


def get_sensor_data(sensor_queue, target_frame, timeout=20.0):
    """从队列中获取指定帧的数据，丢弃旧数据"""
    while True:
        try:
            data = sensor_queue.get(timeout=timeout)
            if data['frame'] == target_frame:
                return data
            elif data['frame'] < target_frame:
                continue  # 丢弃旧数据
            else:
                # 收到未来帧的数据? 这在同步模式下不应发生，除非tick逻辑有问题
                # 暂时返回它，让上层处理或报错
                return data
        except Empty:
            return None

# ==============================================================================
# -- 主函数 --------------------------------------------------------------------
# ==============================================================================

def main():
    print("=" * 80)
    print("CARLA Occupancy 数据采集 - 5 帧测试 (异步保存优化版)")
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
    cam_names = [cfg['id'] for cfg in TESLA_CAMERA_CONFIGS]
    for cam_name in cam_names:
        (cameras_dir / cam_name).mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)
    occupancy_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {output_dir}")
    print()

    # 连接 CARLA
    print("连接 CARLA 服务器 (localhost:2000)...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    print(f"✓ 已连接")
    print(f"  地图: {world.get_map().name}")
    print()

    # 保存原始设置
    original_settings = world.get_settings()

    # 创建数据队列
    camera_queues = {}
    lidar_queue = Queue()

    # 初始化 Occupancy 生成器
    occ_gen = OccupancyGenerator()
    
    # 初始化异步保存器
    saver = DataSaver(output_dir, occ_gen)
    saver.start()

    # Actor 列表 (用于清理)
    actor_list = []

    try:
        # ==========================================
        # 1. 启用同步模式
        # ==========================================
        print("配置同步模式...")
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 10.0  # 10 Hz (降低频率以减轻服务器负载)
        world.apply_settings(settings)
        print("✓ 同步模式已启用 (10 Hz)")
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
        # 尝试多次生成，避免碰撞
        max_retries = 10
        vehicle = None
        for _ in range(max_retries):
            try:
                spawn_point = random.choice(spawn_points)
                # 稍微抬高一点，避免与地面碰撞
                spawn_point.location.z += 0.5
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                break
            except RuntimeError:
                continue
        
        if vehicle is None:
            raise RuntimeError(f"Failed to spawn hero vehicle after {max_retries} attempts")
            
        actor_list.append(vehicle)

        print(f"✓ Hero 车辆已生成: {vehicle.type_id}")
        print()

        # ==========================================
        # 3. 附加 8 个相机
        # ==========================================
        print("附加 8 个相机传感器...")
        
        for cam_config in TESLA_CAMERA_CONFIGS:
            cam_bp = bp_lib.find('sensor.camera.rgb')
            
            # 使用配置中的通用参数
            for key, val in CAMERA_SENSOR_CONFIG.items():
                if cam_bp.has_attribute(key):
                    if isinstance(val, bool):
                         cam_bp.set_attribute(key, 'True' if val else 'False')
                    else:
                        cam_bp.set_attribute(key, str(val))
            
            # 覆盖特定参数
            cam_bp.set_attribute('image_size_x', str(cam_config['width']))
            cam_bp.set_attribute('image_size_y', str(cam_config['height']))
            cam_bp.set_attribute('fov', str(cam_config['fov']))
            
            # 后处理 Profile
            map_name = world.get_map().name
            if cam_bp.has_attribute('post_process_profile'):
                if "Town10HD_Opt" in map_name:
                    cam_bp.set_attribute('post_process_profile', 'Town10HD_Opt')
                else:
                    cam_bp.set_attribute('post_process_profile', 'Default')
            
            # 应用畸变参数
            if 'lens_distortion' in cam_config and cam_config['lens_distortion']:
                dist = cam_config['lens_distortion']
                for key, val in dist.items():
                    if cam_bp.has_attribute(key):
                        cam_bp.set_attribute(key, str(val))

            transform = cam_config['transform']
            camera = world.spawn_actor(cam_bp, transform, attach_to=vehicle)
            actor_list.append(camera)
            
            cam_id = cam_config['id']
            camera_queues[cam_id] = Queue()
            # 注意: 使用默认参数捕获 cam_id
            camera.listen(lambda img, cid=cam_id: camera_callback(img, camera_queues[cid], cid))
            
            print(f"  ✓ {cam_config['description']} ({cam_config['fov']}°)")

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

        collected_frames_count = 0
        
        # 预热: 让仿真运行几帧，使车辆稳定
        print("预热仿真 (10 帧)...")
        for _ in range(10):
            world.tick()
        
        while collected_frames_count < NUM_FRAMES:
            # Tick 仿真
            world.tick()
            world_frame = world.get_snapshot().frame

            print(f"\n尝试采集帧 {collected_frames_count + 1}/{NUM_FRAMES} (世界帧: {world_frame})")

            # 收集相机数据
            camera_data = {}
            all_cameras_ok = True
            for cam_id, cam_queue in camera_queues.items():
                data = get_sensor_data(cam_queue, world_frame, timeout=5.0) # 缩短超时，快速失败
                if data:
                    camera_data[cam_id] = data
                    # print(f"  ✓ {cam_id}")
                else:
                    print(f"  ✗ {cam_id}: 超时或未找到帧 {world_frame}")
                    all_cameras_ok = False
                    # 只要有一个相机失败，就不必等待其他相机了，直接跳过本帧
                    break

            # 收集激光雷达数据
            lidar_data = None
            if all_cameras_ok:
                lidar_data = get_sensor_data(lidar_queue, world_frame, timeout=5.0)
                if lidar_data:
                    pass # print(f"  ✓ lidar: {len(lidar_data['points'])} 点")
                else:
                    print(f"  ✗ lidar: 超时或未找到帧 {world_frame}")

            # 提交到保存队列
            if all_cameras_ok and lidar_data is not None:
                # 提交给 Worker
                saver.queue.put({
                    'frame_idx': collected_frames_count,
                    'cameras': camera_data,
                    'lidar': lidar_data,
                    'vehicle_transform': vehicle.get_transform(),
                    'world_frame': world_frame
                })
                
                print(f"  ✓ 帧 {collected_frames_count} 数据已提交后台保存")
                collected_frames_count += 1
            else:
                print(f"  ✗ 帧数据不完整,跳过, 尝试下一帧...")
                # 这里我们不减 collected_frames_count，只是进入下一次循环
                # "不用绑定死" - 丢帧就丢帧，继续跑

        print()
        print("=" * 80)
        print(f"✓ 数据采集循环结束，等待保存完成...")
        print("=" * 80)
        
        # 停止 Saver 并等待
        saver.stop()

        # ==========================================
        # 保存数据集元数据
        # ==========================================
        print()
        print("保存数据集元数据...")
        
        # 构建相机配置字典 (使用真实配置)
        camera_configs = {}
        for cfg in TESLA_CAMERA_CONFIGS:
            transform_dict = {
                'x': cfg['position']['x'],
                'y': cfg['position']['y'],
                'z': cfg['position']['z'],
                'pitch': cfg['rotation']['pitch'],
                'yaw': cfg['rotation']['yaw'],
                'roll': cfg['rotation']['roll']
            }

            camera_configs[cfg['id']] = {
                'transform': transform_dict,
                'fov': cfg['fov'],
                'width': cfg['width'],
                'height': cfg['height'],
                'lens_distortion': cfg.get('lens_distortion', None)
            }

        calibration = {
            'cameras': camera_configs,
            'lidar': {
                'transform': {'x': 0.0, 'y': 0.0, 'z': 2.5, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
                'channels': 64,
                'range': 100.0
            },
            'dataset_info': {
                'num_frames': collected_frames_count,
                'map': world.get_map().name,
                'vehicle': vehicle.type_id
            }
        }

        calibration_path = output_dir / "calibration.json"
        with open(calibration_path, 'w', encoding='utf-8') as f:
            json.dump(calibration, f, indent=2, ensure_ascii=False)

        print(f"  ✓ 元数据已保存: {calibration_path}")
        print()

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
        
        if 'saver' in locals() and saver.is_alive():
            saver.stop()

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

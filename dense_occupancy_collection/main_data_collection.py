"""
CARLA 360° 全景体素数据采集 (重构版)
入口脚本
"""
import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 处理 PythonAPI 导入
try:
    build_dist = project_root / 'Build' / 'PythonAPI' / 'dist'
    if build_dist.exists():
        for whl in build_dist.glob('*.whl'):
            sys.path.append(str(whl))
            print(f"[Import] Added wheel: {whl}")
            break
    else:
        sys.path.append(str(project_root / 'PythonAPI' / 'carla'))
except: pass

# ⭐ Add project root to sys.path to allow absolute imports
sys.path.append(str(project_root))

import carla
from dense_occupancy_collection.config.occupancy_config import (
    X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION, GRID_SIZE, 
    CARLA_TO_OCCUPANCY_MAPPING, DEPTH_CAMERA_CONFIG
)
# Tesla Style Config
TESLA_CONFIGS = [
    {'id': 'cam_front_main', 'fov': 50, 'x': 1.0, 'y': 0.0, 'z': 1.6, 'pitch': 0, 'yaw': 0, 'roll': 0},
    {'id': 'cam_front_wide', 'fov': 120, 'x': 1.0, 'y': 0.0, 'z': 1.6, 'pitch': 0, 'yaw': 0, 'roll': 0}, # Fisheye
    {'id': 'cam_front_narrow', 'fov': 35, 'x': 1.0, 'y': 0.0, 'z': 1.6, 'pitch': 0, 'yaw': 0, 'roll': 0},
    {'id': 'cam_left_pillar', 'fov': 80, 'x': 0.0, 'y': -0.9, 'z': 1.7, 'pitch': 0, 'yaw': -60, 'roll': 0},
    {'id': 'cam_right_pillar', 'fov': 80, 'x': 0.0, 'y': 0.9, 'z': 1.7, 'pitch': 0, 'yaw': 60, 'roll': 0},
    {'id': 'cam_left_repeater', 'fov': 100, 'x': 1.2, 'y': -0.9, 'z': 1.0, 'pitch': 0, 'yaw': -160, 'roll': 0},
    {'id': 'cam_right_repeater', 'fov': 100, 'x': 1.2, 'y': 0.9, 'z': 1.0, 'pitch': 0, 'yaw': 160, 'roll': 0},
    {'id': 'cam_rear', 'fov': 120, 'x': -2.5, 'y': 0.0, 'z': 1.2, 'pitch': -5, 'yaw': 180, 'roll': 0}
]

# Core Modules
from dense_occupancy_collection.core.scenario_manager import ScenarioManager
from dense_occupancy_collection.core.rgb_suite import RGBSuite
from dense_occupancy_collection.core.depth_suite import DepthSuite
from dense_occupancy_collection.core.voxel_generator import VoxelGenerator
from dense_occupancy_collection.core.visibility_filter import VisibilityFilter
from dense_occupancy_collection.utils.data_saver import DataSaver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--town', default='Town10HD_Opt')
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--output', default='dataset_output')
    args = parser.parse_args()

    # Init
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.load_world(args.town)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    scenario = ScenarioManager(world)
    hero = None
    rgb_suite = None
    depth_suite = None
    
    try:
        # 1. Setup Scenario
        hero = scenario.spawn_hero()
        scenario.spawn_npcs(num_vehicles=30, num_walkers=10)
        
        # 2. Setup Sensors
        rgb_suite = RGBSuite(world, hero, TESLA_CONFIGS)
        depth_suite = DepthSuite(world, hero, DEPTH_CAMERA_CONFIG)
        
        # 3. Setup Processors
        voxel_gen = VoxelGenerator({
            'x_range': X_RANGE, 'y_range': Y_RANGE, 'z_range': Z_RANGE,
            'resolution': RESOLUTION, 'grid_size': GRID_SIZE,
            'mapping': CARLA_TO_OCCUPANCY_MAPPING
        })
        vis_filter = VisibilityFilter(
            width=DEPTH_CAMERA_CONFIG['width'],
            height=DEPTH_CAMERA_CONFIG['height'],
            fov=DEPTH_CAMERA_CONFIG['fov']
        )
        saver = DataSaver(args.output)
        
        # 4. Wait for Sensors
        print("等待传感器初始化...")
        for _ in range(10): world.tick()
        
        # 5. Loop
        print(f"开始采集 {args.frames} 帧...")
        for frame in range(args.frames):
            world.tick()
            print(f"Frame {frame+1}/{args.frames}")
            
            # Get Data
            rgb_data = rgb_suite.get_data()
            depth_data = depth_suite.get_data()
            if not rgb_data or not depth_data:
                print("Data Timeout, skipping")
                continue
                
            # Generate Voxel
            occ, aids = voxel_gen.generate(world, hero)
            
            # Filter
            ego_trans = hero.get_transform()
            ego_matrix = np.array(ego_trans.get_matrix())
            
            occ_filtered, aids_filtered, mask = vis_filter.run(
                occ, aids, 
                {'x_range': X_RANGE, 'y_range': Y_RANGE, 'z_range': Z_RANGE, 'resolution': RESOLUTION},
                depth_data, ego_matrix
            )
            
            # Stats
            total = np.sum(occ > 0)
            kept = np.sum(occ_filtered > 0)
            print(f"  Voxel: {kept}/{total} ({kept/total*100:.1f}%)")
            
            # Save
            saver.save_rgb(frame, rgb_data)
            saver.save_depth(frame, depth_data) # Add depth saving
            
            # 显式保存网格配置，防止前端解析错误导致乱码
            meta = {
                'town': args.town,
                'x_range': np.array(X_RANGE),
                'y_range': np.array(Y_RANGE),
                'z_range': np.array(Z_RANGE),
                'resolution': np.array([RESOLUTION]),
                'grid_size': np.array(GRID_SIZE)
            }
            saver.save_voxel(frame, occ_filtered, aids_filtered, mask, metadata=meta)
            
    finally:
        print("清理资源...")
        if rgb_suite: rgb_suite.destroy()
        if depth_suite: depth_suite.destroy()
        scenario.destroy()
        
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Done.")

if __name__ == '__main__':
    main()

"""
CARLA 360° 全景体素数据采集
支持 Unreal Editor 和打包 exe 两种模式
"""
import sys
import os
import argparse
import shutil
from pathlib import Path
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent  # d:\code\carla
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

sys.path.append(str(project_root))

import carla
from dense_occupancy_collection.config.occupancy_config import (
    X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION, GRID_SIZE,
    CARLA_TO_OCCUPANCY_MAPPING, DEPTH_CAMERA_CONFIG
)

# Tesla Style Config
TESLA_CONFIGS = [
    {'id': 'cam_front_main', 'fov': 50, 'x': 1.0, 'y': 0.0, 'z': 1.6, 'pitch': 0, 'yaw': 0, 'roll': 0},
    {'id': 'cam_front_wide', 'fov': 120, 'x': 1.0, 'y': 0.0, 'z': 1.6, 'pitch': 0, 'yaw': 0, 'roll': 0},
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
from dense_occupancy_collection.utils.camera_utils import compute_camera_params


def setup_world(client, town_name, no_load=False):
    """
    设置世界和地图

    Args:
        client: CARLA client
        town_name: 地图名称
        no_load: 是否跳过加载地图（用于打包 exe）

    Returns:
        world: CARLA world 对象
    """
    if no_load:
        print(f"[模式] 使用已加载的地图")
        world = client.get_world()
        current_map = world.get_map().name
        print(f"  当前地图: {current_map}")

        if town_name and town_name not in current_map:
            print(f"⚠️  警告: 请求的地图 '{town_name}' 与当前地图 '{current_map}' 不匹配")
            response = input("是否继续使用当前地图？(y/n): ").lower().strip()
            if response != 'y':
                print("已取消操作")
                sys.exit(0)
    else:
        print(f"[模式] 动态加载地图: {town_name}")
        try:
            world = client.load_world(town_name)
            print(f"✅ 成功加载地图: {town_name}")
        except RuntimeError as e:
            print(f"❌ 无法加载地图 '{town_name}'")
            print(f"   错误: {e}")
            print("\n这通常发生在使用打包的 CarlaUnreal.exe 时")
            print("解决方案:")
            print(f"  1. 启动时指定地图: CarlaUnreal.exe {town_name}")
            print(f"  2. 使用 --no-load 参数运行此脚本")
            print(f"  3. 使用 Unreal Editor 模式 (start_carla_server.bat)")
            sys.exit(1)

    return world


def main():
    parser = argparse.ArgumentParser(
        description='CARLA 体素数据采集（支持 Editor 和打包 exe）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. Unreal Editor 模式（推荐）:
   启动: start_carla_server.bat
   运行: python main_data_collection.py --frames 100

2. 打包 exe 模式:
   启动: server\\Windows\\启动_Town10HD.bat
   运行: python main_data_collection.py --frames 100 --no-load
        """
    )
    parser.add_argument('--host', default='localhost', help='CARLA 服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA 服务器端口')
    parser.add_argument('--town', default='Town10HD_Opt', help='地图名称')
    parser.add_argument('--frames', type=int, default=5, help='采集帧数')
    parser.add_argument('--output', default='dataset_output', help='输出目录')
    parser.add_argument('--no-load', action='store_true',
                        help='不加载地图，使用当前已加载的地图（用于打包 exe）')
    parser.add_argument('--resume', action='store_true',
                        help='增量模式：继续在已有数据集上追加数据，不清空目录')
    parser.add_argument('--num-vehicles', type=int, default=30, help='NPC车辆数量（默认30）')
    parser.add_argument('--num-walkers', type=int, default=10, help='NPC行人数量（默认10）')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("CARLA 体素数据采集".center(70))
    print("="*70 + "\n")

    # 连接服务器
    print(f"连接服务器: {args.host}:{args.port}")
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)  # 增加超时时间到 30秒

    try:
        print(f"服务器版本: {client.get_server_version()}")
    except Exception as e:
        print(f"⚠️  无法获取服务器版本: {e}")

    # 检查并处理输出目录
    start_frame_idx = 0
    if os.path.exists(args.output):
        if args.resume:
            # 增量模式：查找已有的最大帧索引
            existing_frames = []
            occupancy_dir = Path(args.output) / 'occupancy'
            if occupancy_dir.exists():
                existing_frames = [int(f.stem) for f in occupancy_dir.glob('*.npz')]

            if existing_frames:
                start_frame_idx = max(existing_frames) + 1
                print(f"📂 增量模式: 检测到已有 {len(existing_frames)} 帧数据")
                print(f"   从帧 {start_frame_idx} 开始继续采集")
            else:
                print(f"📂 增量模式: 输出目录存在但为空，从帧 0 开始")
        else:
            # 覆盖模式：清空旧数据
            print(f"🗑️  清理旧数据目录: {args.output}")
            shutil.rmtree(args.output)
    else:
        print(f"📁 创建新输出目录: {args.output}")

    # 设置世界
    world = setup_world(client, args.town, no_load=args.no_load)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # 同步模式
    settings = world.get_settings()
    original_sync_mode = settings.synchronous_mode
    original_fixed_delta = settings.fixed_delta_seconds

    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    print(f"✅ 同步模式: {settings.synchronous_mode}, 时间步: {settings.fixed_delta_seconds}s")
    
    scenario = ScenarioManager(world)
    hero = None
    rgb_suite = None
    depth_suite = None

    try:
        # 1. Setup Scenario
        print("\n[1/4] 设置场景...")
        hero = scenario.spawn_hero()
        print(f"  生成主车: {hero.type_id}")

        scenario.spawn_npcs(num_vehicles=args.num_vehicles, num_walkers=args.num_walkers)
        print(f"  生成 NPC: {args.num_vehicles} 辆车, {args.num_walkers} 个行人")

        # 2. Setup Sensors
        print("\n[2/4] 设置传感器...")
        rgb_suite = RGBSuite(world, hero, TESLA_CONFIGS)
        print(f"  RGB 相机: {len(TESLA_CONFIGS)} 个")

        depth_suite = DepthSuite(world, hero, DEPTH_CAMERA_CONFIG)
        print(f"  深度相机: 1 个")

        # 3. Setup Processors
        print("\n[3/4] 设置处理器...")
        voxel_gen = VoxelGenerator({
            'x_range': X_RANGE, 'y_range': Y_RANGE, 'z_range': Z_RANGE,
            'resolution': RESOLUTION, 'grid_size': GRID_SIZE,
            'mapping': CARLA_TO_OCCUPANCY_MAPPING
        })
        print(f"  体素生成器: {GRID_SIZE}")

        vis_filter = VisibilityFilter(
            width=DEPTH_CAMERA_CONFIG['width'],
            height=DEPTH_CAMERA_CONFIG['height'],
            fov=DEPTH_CAMERA_CONFIG['fov']
        )
        print(f"  可见性过滤器: {DEPTH_CAMERA_CONFIG['width']}x{DEPTH_CAMERA_CONFIG['height']}")

        saver = DataSaver(args.output)
        print(f"  数据保存器: {args.output}")

        # 4. Wait for Sensors
        print("\n[4/4] 等待传感器初始化...")
        for i in range(10):
            world.tick()
            if i % 2 == 0:
                print(f"  初始化中... {(i+1)*10}%")
        print("  ✅ 传感器就绪")

        # 5. Loop
        print(f"\n{'='*70}")
        if start_frame_idx > 0:
            print(f"继续采集: 帧 {start_frame_idx} 到 {start_frame_idx + args.frames - 1} (共 {args.frames} 帧)".center(70))
        else:
            print(f"开始采集 {args.frames} 帧".center(70))
        print(f"{'='*70}\n")

        for i in range(args.frames):
            frame_idx = start_frame_idx + i
            world.tick()
            print(f"\n[Frame {frame_idx} ({i+1}/{args.frames})]")

            # Get Data
            rgb_data = rgb_suite.get_data()
            depth_data = depth_suite.get_data()

            if not rgb_data or not depth_data:
                print("  ⚠️  数据超时，跳过此帧")
                continue

            print(f"  ✅ RGB 数据: {len(rgb_data)} 张图像")
            print(f"  ✅ 深度数据: 1 张图像")

            # Generate Voxel
            occ, aids = voxel_gen.generate(world, hero)
            total_voxels = np.sum(occ > 0)
            print(f"  ✅ 原始体素: {total_voxels} 个")

            # Filter
            ego_trans = hero.get_transform()
            ego_matrix = np.array(ego_trans.get_matrix())

            occ_filtered, aids_filtered, mask = vis_filter.run(
                occ, aids,
                {'x_range': X_RANGE, 'y_range': Y_RANGE, 'z_range': Z_RANGE, 'resolution': RESOLUTION},
                depth_data, ego_matrix
            )

            # Stats
            kept_voxels = np.sum(occ_filtered > 0)
            filter_rate = kept_voxels / total_voxels * 100 if total_voxels > 0 else 0
            print(f"  ✅ 过滤后体素: {kept_voxels}/{total_voxels} ({filter_rate:.1f}%)")

            # 计算相机参数
            intrinsics, extrinsics = compute_camera_params(
                camera_configs=TESLA_CONFIGS,
                ego_transform=ego_trans,
                image_width=640,
                image_height=384
            )
            print(f"  ✅ 相机参数: 8 个相机")

            # Save
            saver.save_rgb(frame_idx, rgb_data)
            saver.save_depth(frame_idx, depth_data)
            saver.save_camera_params(frame_idx, TESLA_CONFIGS, intrinsics, extrinsics)

            # 保存网格配置
            meta = {
                'town': world.get_map().name,
                'x_range': np.array(X_RANGE),
                'y_range': np.array(Y_RANGE),
                'z_range': np.array(Z_RANGE),
                'resolution': np.array([RESOLUTION]),
                'grid_size': np.array(GRID_SIZE)
            }
            saver.save_voxel(frame_idx, occ_filtered, aids_filtered, mask, metadata=meta)
            print(f"  ✅ 数据已保存")

    finally:
        print("\n" + "="*70)
        print("清理资源...".center(70))
        print("="*70 + "\n")

        if rgb_suite:
            rgb_suite.destroy()
            print("  ✅ 清理 RGB 传感器")

        if depth_suite:
            depth_suite.destroy()
            print("  ✅ 清理深度传感器")

        scenario.destroy()
        print("  ✅ 清理场景")

        # 恢复原始设置
        settings.synchronous_mode = original_sync_mode
        settings.fixed_delta_seconds = original_fixed_delta
        world.apply_settings(settings)
        print("  ✅ 恢复世界设置")

        print("\n" + "="*70)
        print("采集完成！".center(70))
        print("="*70 + "\n")
        print(f"数据保存位置: {args.output}")
        print()

if __name__ == '__main__':
    main()

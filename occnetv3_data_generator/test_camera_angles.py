"""
测试修复后的相机角度配置
生成1帧数据,使用 dataset_viewer_v2 检查相机画面
"""
import sys
from pathlib import Path

# 添加 CARLA 到路径
project_root = Path(__file__).parent.parent
build_dist = project_root / 'Build' / 'PythonAPI' / 'dist'
if build_dist.exists():
    for whl in build_dist.glob('*.whl'):
        sys.path.append(str(whl))
        break

import carla
import time
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
from sensors.camera_manager import CameraManager
from config.camera_config import TESLA_CAMERAS, CAMERA_SENSOR_CONFIG

def main():
    print("=" * 80)
    print("测试相机角度配置 - 生成1帧数据")
    print("=" * 80)

    # 连接 CARLA
    print("\n[1/5] 连接 CARLA...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 同步模式
    print("[2/5] 设置同步模式...")
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Traffic Manager
    tm = client.get_trafficmanager(8001)
    tm.set_synchronous_mode(True)

    ego_vehicle = None
    camera_manager = None

    try:
        # 生成ego车辆
        print("[3/5] 生成 ego 车辆...")
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.nissan.patrol')[0]
        spawn_points = world.get_map().get_spawn_points()
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        ego_vehicle.set_autopilot(True, 8001)

        # 物理稳定
        for _ in range(10):
            world.tick()

        print(f"  ✓ ego 车辆已生成: {ego_vehicle.type_id}")

        # 创建相机管理器
        print("[4/5] 初始化相机管理器...")
        camera_manager = CameraManager(world, ego_vehicle, TESLA_CAMERAS, CAMERA_SENSOR_CONFIG)

        # 等待传感器稳定
        for _ in range(5):
            world.tick()

        print(f"  ✓ 8个相机已初始化")
        print("\n相机配置检查:")
        for cam in TESLA_CAMERAS:
            pos = cam['position']
            rot = cam['rotation']
            print(f"  {cam['id']:15s}: pos=({pos[0]:+.1f}, {pos[1]:+.1f}, {pos[2]:+.1f}), "
                  f"rot=(pitch={rot[0]:+.0f}°, yaw={rot[1]:+.0f}°, roll={rot[2]:+.0f}°)")

        # 采集1帧数据
        print("\n[5/5] 采集1帧数据...")
        world.tick()
        camera_data = camera_manager.get_synced_frame(timeout=2.0)

        if camera_data:
            print(f"  ✓ 采集成功,共 {len(camera_data)} 个相机")

            # 保存到临时目录供 viewer 检查
            output_dir = Path("d:/code/carla/dataset_camera_test")
            cameras_dir = output_dir / "cameras"

            for cam_id, img_array in camera_data.items():
                cam_dir = cameras_dir / f"cam_{cam_id}"
                cam_dir.mkdir(parents=True, exist_ok=True)

                # 保存为 PNG (灰度)
                png_path = cam_dir / "000000.png"
                import cv2
                # img_array shape: (1, 960, 1280) float16 [0, 1]
                gray = (img_array[0] * 255).astype(np.uint8)
                cv2.imwrite(str(png_path), gray)
                print(f"    已保存: {cam_id} -> {png_path}")

            print(f"\n✅ 测试完成!")
            print(f"  数据保存到: {output_dir}")
            print(f"\n使用 dataset_viewer_v2 检查:")
            print(f"  1. 修改 server.py 中 DATASET_PATH = r'{output_dir}'")
            print(f"  2. 运行: python dataset_viewer_v2/server.py")
            print(f"  3. 浏览器访问: http://localhost:8085/")
            print(f"\n⭐ 检查要点:")
            print(f"  - 左B柱/右B柱: 应拍摄车外侧前方,不应看到车内")
            print(f"  - 左Repeater/右Repeater: 应拍摄侧后方,不应看到车身")
            print(f"  - 后视相机: 应完全伸出车尾,不应有车身遮挡")
        else:
            print("  ❌ 采集失败!")

    finally:
        # 清理
        print("\n清理...")
        if camera_manager:
            camera_manager.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()

        # 恢复异步模式
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("✓ 清理完成")

if __name__ == '__main__':
    main()

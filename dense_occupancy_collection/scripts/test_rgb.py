"""
RGB 相机测试脚本
验证 UE5.5 Lumen 光照修复效果
"""
import sys
import os
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 处理 PythonAPI 导入
try:
    build_dist = project_root / 'Build' / 'PythonAPI' / 'dist'
    if build_dist.exists():
        for whl in build_dist.glob('*.whl'):
            sys.path.append(str(whl))
            break
    else:
        sys.path.append(str(project_root / 'PythonAPI' / 'carla'))
except: pass

import carla
from dense_occupancy_collection.core.rgb_suite import RGBSuite
from dense_occupancy_collection.utils.data_saver import DataSaver

def main():
    print("连接 CARLA...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # 确保在 Town10HD
    if 'Town10HD' not in world.get_map().name:
        print(f"当前地图是 {world.get_map().name}，建议切换到 Town10HD_Opt 以测试后处理修复")
    
    # 获取 Hero
    vehicles = world.get_actors().filter('vehicle.*')
    hero = None
    for v in vehicles:
        if v.attributes.get('role_name') == 'hero':
            hero = v
            break
    
    if not hero:
        print("未找到 Hero 车辆，将使用第一个车辆")
        if len(vehicles) > 0:
            hero = vehicles[0]
        else:
            print("场景中没有车辆，无法挂载相机")
            return

    # 测试配置
    configs = [
        {'id': 'test_front', 'fov': 90, 'x': 1.0, 'y': 0.0, 'z': 1.6, 'pitch': 0, 'yaw': 0, 'roll': 0}
    ]
    
    print("初始化 RGB Suite...")
    suite = RGBSuite(world, hero, configs)
    saver = DataSaver('dataset_output_test')
    
    # 预热
    for _ in range(10): world.tick()
    
    print("采集 1 帧...")
    data = suite.get_data()
    if data:
        saver.save_rgb(0, data)
        print(f"已保存到 dataset_output_test/cameras/test_front/000000.png")
    else:
        print("采集失败")
        
    suite.destroy()
    print("测试完成")

if __name__ == '__main__':
    main()

"""
清理 CARLA 世界中的所有 Actor
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

def cleanup_world(host='localhost', port=2000):
    """清理世界中的所有 Actor"""
    print(f"连接到 CARLA: {host}:{port}")
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()

    print("\n清理所有 Actor...")

    # 获取所有 Actor
    actors = world.get_actors()

    # 清理车辆
    vehicles = actors.filter('vehicle.*')
    print(f"  清理 {len(vehicles)} 辆车辆...")
    for vehicle in vehicles:
        try:
            vehicle.destroy()
        except:
            pass

    # 清理行人
    walkers = actors.filter('walker.*')
    print(f"  清理 {len(walkers)} 个行人...")
    for walker in walkers:
        try:
            walker.destroy()
        except:
            pass

    # 清理传感器
    sensors = actors.filter('sensor.*')
    print(f"  清理 {len(sensors)} 个传感器...")
    for sensor in sensors:
        try:
            sensor.destroy()
        except:
            pass

    print("\n✅ 清理完成!")

if __name__ == '__main__':
    cleanup_world()

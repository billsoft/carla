"""
把常驻的 CARLA 服务端恢复到干净状态（服务端常开时，被中断的脚本会留下同步模式、天气、车辆和传感器）：
  - 关闭同步模式、清除 fixed_delta_seconds，Traffic Manager 也退出同步
  - 销毁所有传感器、车辆、行人（及行人控制器）
  - 天气恢复为地图默认（carla.WeatherParameters.Default）

用法：
    C:\\Users\\bills\\.conda\\envs\\carla58\\python.exe remote\\reset_carla_world.py --host 192.168.100.81
"""

import argparse

import carla


def reset_world(client, tm_port=8000, keep_weather=False):
    """关同步模式、销毁全部传感器/车辆/行人(及控制器)，可选恢复默认天气。返回 (销毁数, 剩余 type_id 列表)。
    occnetv3_data_generator/main_collection.py 启动时和控制台被关闭时也调用它 (服务端不会在客户端断开时
    自动销毁该客户端创建的 actor：直接关掉终端会把 24 路相机留在服务端一直渲染、占满显存)。"""
    world = client.get_world()
    s = world.get_settings()
    s.synchronous_mode = False
    s.fixed_delta_seconds = None
    world.apply_settings(s)
    try:
        client.get_trafficmanager(tm_port).set_synchronous_mode(False)
    except RuntimeError:
        pass

    actors = world.get_actors()
    groups = [a for a in actors if a.type_id.startswith('sensor.')]         + [a for a in actors if a.type_id.startswith('controller.')]         + [a for a in actors if a.type_id.startswith(('vehicle.', 'walker.'))]
    for a in groups:
        if a.type_id.startswith('sensor.') and a.is_listening:
            a.stop()
    client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in groups])

    if not keep_weather:
        world.set_weather(carla.WeatherParameters.Default)
    left = [a.type_id for a in world.get_actors() if a.type_id.startswith(('sensor.', 'vehicle.', 'walker.'))]
    return len(groups), left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='192.168.100.81')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--tm-port', type=int, default=8000)
    ap.add_argument('--keep-weather', action='store_true')
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30)
    destroyed, left = reset_world(client, args.tm_port, args.keep_weather)
    world = client.get_world()
    w = world.get_weather()
    print(f'map={world.get_map().name} sync={world.get_settings().synchronous_mode} destroyed={destroyed} '
          f'remaining={left} sun_alt={w.sun_altitude_angle:.1f} sun_az={w.sun_azimuth_angle:.1f}')


if __name__ == '__main__':
    main()

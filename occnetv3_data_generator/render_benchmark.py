"""
渲染性能基准：按 main_collection.py 的相机配置（8 路等距 Bayer RAW + 8 路深度）挂到一辆车上，
同步模式下连续 tick，统计"world.tick() + 收齐 16 路图像"的耗时。用于在调整渲染参数（Lumen 硬件光追、
光追阴影、曝光等）前后比较是否还能流畅运行。

用法（5.8 服务器已在 Play）：
    C:\\Users\\bills\\.conda\\envs\\carla58\\python.exe render_benchmark.py --ticks 40
"""

import argparse
import os
import sys
import time

import numpy as np
import carla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sensors.camera_manager import CameraManager  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--ticks', type=int, default=40)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--no-depth', action='store_true')
    ap.add_argument('--autopilot', action='store_true', help='车辆行驶（相机运动，Lumen/曝光更接近采集时的负载）')
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(120)
    world = client.get_world()
    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 0.05
    world.apply_settings(s)

    bl = world.get_blueprint_library()
    vbp = (bl.filter('vehicle.tesla.model3') or bl.filter('vehicle.*'))[0]
    vehicle = None
    for p in world.get_map().get_spawn_points():
        vehicle = world.try_spawn_actor(vbp, p)
        if vehicle:
            break
    cm = None
    try:
        tm = None
        if args.autopilot:
            tm = client.get_trafficmanager(8000)
            tm.set_synchronous_mode(True)
            vehicle.set_autopilot(True, tm.get_port())
        cm = CameraManager(world, vehicle, enable_depth=not args.no_depth)
        cm.start_listening()
        queues = list(cm.data_queues.values()) + (list(cm.depth_queues.values()) if not args.no_depth else [])
        times = []
        for i in range(args.warmup + args.ticks):
            t0 = time.perf_counter()
            world.tick()
            for q in queues:
                q.get(timeout=120)
            dt = time.perf_counter() - t0
            if i >= args.warmup:
                times.append(dt)
        t = np.array(times) * 1000
        n = len(queues)
        print(f'{n} 路相机, {len(t)} ticks: 平均 {t.mean():.0f} ms/帧 ({1000 / t.mean():.1f} FPS), '
              f'中位数 {np.median(t):.0f} ms, P95 {np.percentile(t, 95):.0f} ms, 最大 {t.max():.0f} ms')
    finally:
        if cm is not None:
            cm.destroy()
        if vehicle is not None:
            vehicle.destroy()
        world.apply_settings(original)
        if args.autopilot:
            client.get_trafficmanager(8000).set_synchronous_mode(False)


if __name__ == '__main__':
    main()

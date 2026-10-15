"""
体素真值 vs UE 渲染结果的逐像素核对 (几何 + 语义)。

在 8 个相机位姿各挂一个针孔深度相机和一个语义分割相机 (水平 FOV = depth_fov，与采集一致)，
生成体素真值 (与 main_collection.py 同一条流程：生成 → 相机侧后处理 finalize_ground_truth
 = 地面语义重绘 + 自由空间雕刻)，
然后把每个语义像素按深度反投影成 3D 点，查它落在哪个体素：
  - 落在 free → 几何漏标 (照片里有东西，真值说是空)
  - 落在别的类 → 语义不一致
UE 语义分割的标签值就是 CityObjectLabel 编号，按 config/occupancy_config.CARLA_TO_OCCUPANCY_MAPPING
换成 18 类后对比。输出按 UE 类别统计的一致率和主要混淆，以及按体素类别统计的精度。

    python audit_labels.py --host 192.168.100.81 --frames 5 --frame-skip 10
"""
import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import carla  # noqa: E402
from config.camera_config import TESLA_CAMERAS, DEPTH_SENSOR_CONFIG  # noqa: E402
from config.occupancy_config import (X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION, OCCUPANCY_LABELS,  # noqa: E402
                                     CARLA_TO_OCCUPANCY_MAPPING)
from main_collection import setup_carla, spawn_vehicle, spawn_npcs, choose_weather  # noqa: E402
from processing.camera_visibility import CameraVisibility, finalize_ground_truth  # noqa: E402
from processing.ground_truth_voxel_generator import GroundTruthVoxelGenerator  # noqa: E402
from processing.transform_utils import rotation_matrix  # noqa: E402
from processing.visibility_filter_simple import VisibilityFilterSimple  # noqa: E402
from sensors.camera_manager import CameraManager  # noqa: E402
from sensors.semantic_lidar_sensor import SemanticLidarSensor  # noqa: E402

UE_NAMES = {0: 'Unlabeled', 1: 'Roads', 2: 'Sidewalks', 3: 'Buildings', 4: 'Walls', 5: 'Fences', 6: 'Poles',
            7: 'TrafficLight', 8: 'TrafficSigns', 9: 'Vegetation', 10: 'Terrain', 11: 'Sky', 12: 'Pedestrians',
            13: 'Rider', 14: 'Car', 15: 'Truck', 16: 'Bus', 17: 'Train', 18: 'Motorcycle', 19: 'Bicycle',
            20: 'Static', 21: 'Dynamic', 22: 'Other', 23: 'Water', 24: 'RoadLines', 25: 'Ground', 26: 'Bridge',
            27: 'RailTrack', 28: 'GuardRail', 29: 'Rock'}
SKIP_TAGS = {0, 11}                      # 未标注、天空


class PinholeRig:
    """8 个相机位姿上的针孔深度 + 语义相机，按帧号取数据。"""

    def __init__(self, world, vehicle):
        lib = world.get_blueprint_library()
        self.sensors, self.queues = [], []
        for cam in sorted(TESLA_CAMERAS, key=lambda c: c['index']):
            pos, rot = cam['position'], cam['rotation']
            tf = carla.Transform(carla.Location(*pos), carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]))
            pair = {}
            for kind, bp_name in (('depth', 'sensor.camera.depth'), ('sem', 'sensor.camera.semantic_segmentation')):
                bp = lib.find(bp_name)
                bp.set_attribute('image_size_x', str(DEPTH_SENSOR_CONFIG['image_size_x']))
                bp.set_attribute('image_size_y', str(DEPTH_SENSOR_CONFIG['image_size_y']))
                bp.set_attribute('fov', str(cam['depth_fov']))
                s = world.spawn_actor(bp, tf, attach_to=vehicle)
                q = queue.Queue()
                s.listen(q.put)
                self.sensors.append(s)
                pair[kind] = q
            self.queues.append(pair)

    def get(self, frame, timeout=10.0):
        out = []
        for pair in self.queues:
            item = {}
            for kind, q in pair.items():
                deadline = time.time() + timeout
                while True:
                    img = q.get(timeout=max(0.0, deadline - time.time()))
                    if img.frame == frame:
                        break
                    if img.frame > frame:
                        raise RuntimeError(f'{kind} 丢帧 {frame}')
                if kind == 'depth':
                    item[kind] = CameraManager.convert_depth(img)
                else:
                    a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
                    item[kind] = a[:, :, 2].copy()      # BGRA，标签在 R 通道
            out.append(item)
        return out

    def destroy(self):
        for s in self.sensors:
            try:
                s.stop()
                s.destroy()
            except RuntimeError:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--town', default='Town10HD_Opt')
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--frame-skip', type=int, default=10)
    ap.add_argument('--warmup-ticks', type=int, default=60)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--num-vehicles', type=int, default=30)
    ap.add_argument('--num-walkers', type=int, default=30)
    ap.add_argument('--lidar-filter', action='store_true', help='加上旧的 LiDAR 实例过滤 (A/B 对比用)')
    ap.add_argument('--raw', action='store_true', help='不做相机侧后处理 (地面重绘/雕刻)，看生成器原始输出')
    args = ap.parse_args()

    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    weather, _ = choose_weather('ClearNoon', None)
    client, world, tm_port = setup_carla(args.host, args.port, args.town, weather)
    vehicle, rig, lidar, npcs = None, None, None, []
    lut = np.zeros(256, np.uint8)
    for k, v in CARLA_TO_OCCUPANCY_MAPPING.items():
        lut[k] = v
    conf = np.zeros((256, 18), np.int64)            # 行 = UE 语义标签，列 = 体素标签
    try:
        vehicle = spawn_vehicle(world, tm_port)
        npcs = spawn_npcs(client, world, args.num_vehicles, args.num_walkers, tm_port=8001)
        rig = PinholeRig(world, vehicle)
        if args.lidar_filter:
            lidar = SemanticLidarSensor(world, vehicle, config=None)
            lidar.listen_to_queue()
        cams = [{'R': rotation_matrix(*c['rotation']).astype(np.float64), 't': np.asarray(c['position'], np.float64),
                 'hfov': c['depth_fov'], 'width': DEPTH_SENSOR_CONFIG['image_size_x'],
                 'height': DEPTH_SENSOR_CONFIG['image_size_y']}
                for c in sorted(TESLA_CAMERAS, key=lambda c: c['index'])]
        vis = CameraVisibility(cams, X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION)
        gen = GroundTruthVoxelGenerator(X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION)
        vfilter = VisibilityFilterSimple()
        client.set_timeout(60.0)
        for _ in range(args.warmup_ticks):
            world.tick()
        ego_bb = vehicle.bounding_box
        half = RESOLUTION / 2
        for fi in range(args.frames):
            for _ in range(args.frame_skip):
                frame = world.tick()
            data = rig.get(frame)
            occ, aids = gen.generate(world, vehicle)
            if args.lidar_filter:
                lidar_data = lidar.get_data(timeout=10.0, frame=frame)
                occ, aids = vfilter.run(occ, aids, {'x_range': X_RANGE, 'y_range': Y_RANGE, 'z_range': Z_RANGE,
                                                    'resolution': RESOLUTION}, lidar_data, ego_id=vehicle.id)
            if not args.raw:
                _, st = finalize_ground_truth(vis, occ, aids, [d['depth'] for d in data], [d['sem'] for d in data])
                print(f'  后处理: {st}')
            for cam, (ray_v, unit), d in zip(cams, vis._rays, data):
                depth = d['depth'].reshape(-1)
                tag = d['sem'].reshape(-1)
                ok = (depth < 200) & ~np.isin(tag, list(SKIP_TAGS))
                pts = ray_v[ok] * depth[ok, None] + cam['t'].astype(np.float32)
                # 去掉拍到的自车车身
                ego = ((np.abs(pts[:, 0] - ego_bb.location.x) < ego_bb.extent.x + 0.1)
                       & (np.abs(pts[:, 1] - ego_bb.location.y) < ego_bb.extent.y + 0.1))
                pts, t = pts[~ego], tag[ok][~ego]
                lab = np.zeros(len(pts), np.uint8)
                inside = np.ones(len(pts), bool)
                # 表面点常落在体素边界：依次看 点本身 / 沿视线后半格 / 往下半格，取第一个非空
                for p in (pts, pts + unit[ok][~ego] * half, pts - np.array([0, 0, half], np.float32)):
                    ix = np.floor((p[:, 0] - X_RANGE[0]) / RESOLUTION).astype(int)
                    iy = np.floor((p[:, 1] - Y_RANGE[0]) / RESOLUTION).astype(int)
                    iz = np.floor((p[:, 2] - Z_RANGE[0]) / RESOLUTION).astype(int)
                    inb = (ix >= 0) & (ix < 400) & (iy >= 0) & (iy < 400) & (iz >= 0) & (iz < 32)
                    inside &= inb
                    v = np.zeros(len(p), np.uint8)
                    v[inb] = occ[ix[inb], iy[inb], iz[inb]]
                    lab = np.where(lab == 0, v, lab)
                np.add.at(conf, (t[inside], lab[inside]), 1)
            print(f'帧 {fi + 1}/{args.frames} (sim frame {frame}) 完成', flush=True)
    finally:
        for a in [rig, lidar]:
            if a is not None:
                a.destroy()
        for a in npcs + ([vehicle] if vehicle else []):
            try:
                a.destroy()
            except RuntimeError:
                pass
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)

    print('\n=== 按 UE 语义类别 (像素→3D 点→体素) ===')
    print(f'{"UE 类别":14s} {"期望 18 类":18s} {"点数":>9s} {"一致%":>7s} {"落在free%":>9s}  主要混淆')
    for tag in np.flatnonzero(conf.sum(1)):
        row = conf[tag]
        n = row.sum()
        exp = lut[tag]
        top = [(c, row[c]) for c in np.argsort(-row) if c not in (exp, 0) and row[c] > 0][:3]
        print(f'{UE_NAMES.get(tag, tag):14s} {OCCUPANCY_LABELS[exp]:18s} {n:9d} {100 * row[exp] / n:7.1f} '
              f'{100 * row[0] / n:9.1f}  ' + ', '.join(f'{OCCUPANCY_LABELS[c]} {100 * k / n:.1f}%' for c, k in top))
    print('\n=== 按体素类别 (被照片表面点命中时，UE 说它是什么) ===')
    exp_of_tag = lut[:256]
    for c in range(1, 18):
        col = conf[:, c]
        n = col.sum()
        if n == 0:
            continue
        correct = col[exp_of_tag == c].sum()
        top = [(t, col[t]) for t in np.argsort(-col) if exp_of_tag[t] != c and col[t] > 0][:3]
        print(f'{c:2d} {OCCUPANCY_LABELS[c]:18s} 命中 {n:9d}  正确 {100 * correct / n:5.1f}%  其实是: '
              + ', '.join(f'{UE_NAMES.get(t, t)} {100 * k / n:.1f}%' for t, k in top))
    total = conf[:, 1:].sum() + conf[:, 0].sum()
    agree = sum(conf[t, lut[t]] for t in range(256))
    print(f'\n总计 {total} 个表面点：语义一致 {100 * agree / total:.1f}%，落在 free {100 * conf[:, 0].sum() / total:.1f}%')


if __name__ == '__main__':
    main()

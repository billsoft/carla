"""
体素真值 vs UE 渲染结果的逐像素核对 (几何 + 语义)，含独立的留出相机检查。

两组针孔 深度 + UE 语义分割 相机：
  - 生成相机：8 个鱼眼相机位姿 (水平 FOV = depth_fov)，和采集时一样参与真值生成 (地面语义重绘、自由空间雕刻、
    观测掩码)。用它们检查的是**内部一致性**——生成用了它们的深度/语义，再用它们检查，只能发现"照片里有、
    体素里没有"之类的矛盾，发现不了多填的内部体素和朝向错误。
  - 留出相机 (--holdout，默认开)：车顶上方 3.5m、四个对角方向向下 25° 的 90° 针孔相机，**不参与**真值生成，
    只用来评估：(1) 它们的语义像素反投影到体素，类别是否一致；(2) 它们的视线明确穿过的占用体素
    (自由空间被误占用，按类别统计)；(3) 上面两项单独统计"带旋转包围盒"的静态物体。
真值生成流程与 main_collection.py 相同：生成 → (可选 LiDAR 过滤) → finalize_ground_truth。
--legacy 用 2026-10-15 之前的几何 (静态包围盒按世界轴对齐判定、地面高度 = h - ego.z)，同一场景 A/B 对比。

    python audit_labels.py --host 192.168.100.81 --frames 5 [--town Town15] [--legacy]
"""
import argparse
import math
import queue
import random
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
W, H = DEPTH_SENSOR_CONFIG['image_size_x'], DEPTH_SENSOR_CONFIG['image_size_y']
# 留出相机：车顶上方，四个对角方向俯视 (位置/朝向与 8 个生成相机都不同)
HOLDOUT = [{'position': (0.0, 0.0, 3.5), 'rotation': (-25.0, yaw, 0.0), 'fov': 90.0} for yaw in (45, 135, 225, 315)]


def fisheye_params(cam):
    """camera_config 条目 → 鱼眼内参 (与 camera_manager.get_intrinsics 相同的公式)，供观测掩码裁剪鱼眼视野。"""
    fov_v = math.radians(cam['fov_vertical'])
    fov_h = math.radians(cam.get('fov_horizontal', cam['fov_vertical'] * W / H))
    cx, cy = cam.get('principal_point', (W / 2.0, H / 2.0))
    k = tuple(cam.get('distortion_coeffs', (0.0, 0.0, 0.0, 0.0))) if cam.get('lens_model') == 'kannala-brandt' \
        else (0.0, 0.0, 0.0, 0.0)
    return {'fx': (W / 2.0) / (fov_h / 2.0), 'fy': (H / 2.0) / (fov_v / 2.0), 'cx': cx, 'cy': cy,
            'width': W, 'height': H, 'k': k}


class PinholeRig:
    """一组位姿上的针孔深度 + 语义相机，按帧号取数据。poses: [{'position','rotation'(p,y,r),'fov'}]"""

    def __init__(self, world, vehicle, poses):
        lib = world.get_blueprint_library()
        self.sensors, self.queues = [], []
        for pose in poses:
            pos, rot = pose['position'], pose['rotation']
            tf = carla.Transform(carla.Location(*pos), carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]))
            pair = {}
            for kind, bp_name in (('depth', 'sensor.camera.depth'), ('sem', 'sensor.camera.semantic_segmentation')):
                bp = lib.find(bp_name)
                bp.set_attribute('image_size_x', str(W))
                bp.set_attribute('image_size_y', str(H))
                bp.set_attribute('fov', str(pose['fov']))
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


def surface_labels(vis, data, occ, ego_bb, extra_mask=None):
    """每个语义像素按深度反投影到体素 → (UE 标签, 体素标签, 是否落在 extra_mask 体素上)。"""
    half = RESOLUTION / 2
    tags, labs, flags = [], [], []
    for cam, (ray_v, unit, _), d in zip(vis.cameras, vis._rays, data):
        depth = d['depth'].reshape(-1)
        tag = d['sem'].reshape(-1)
        ok = (depth < 200) & ~np.isin(tag, list(SKIP_TAGS))
        pts = ray_v[ok] * depth[ok, None] + cam['t'].astype(np.float32)
        ego = ((np.abs(pts[:, 0] - ego_bb.location.x) < ego_bb.extent.x + 0.1)
               & (np.abs(pts[:, 1] - ego_bb.location.y) < ego_bb.extent.y + 0.1))
        pts, t, un = pts[~ego], tag[ok][~ego], unit[ok][~ego]
        lab = np.zeros(len(pts), np.uint8)
        flag = np.zeros(len(pts), bool)
        inside = np.ones(len(pts), bool)
        # 表面点常落在体素边界：依次看 点本身 / 沿视线后半格 / 往下半格，取第一个非空
        for p in (pts, pts + un * half, pts - np.array([0, 0, half], np.float32)):
            ix = np.floor((p[:, 0] - X_RANGE[0]) / RESOLUTION).astype(int)
            iy = np.floor((p[:, 1] - Y_RANGE[0]) / RESOLUTION).astype(int)
            iz = np.floor((p[:, 2] - Z_RANGE[0]) / RESOLUTION).astype(int)
            inb = (ix >= 0) & (ix < occ.shape[0]) & (iy >= 0) & (iy < occ.shape[1]) & (iz >= 0) & (iz < occ.shape[2])
            inside &= inb
            v = np.zeros(len(p), np.uint8)
            v[inb] = occ[ix[inb], iy[inb], iz[inb]]
            if extra_mask is not None:
                f = np.zeros(len(p), bool)
                f[inb] = extra_mask[ix[inb], iy[inb], iz[inb]]
                flag |= f & (lab == 0) & (v != 0)
            lab = np.where(lab == 0, v, lab)
        tags.append(t[inside])
        labs.append(lab[inside])
        flags.append(flag[inside])
    return np.concatenate(tags), np.concatenate(labs), np.concatenate(flags)


def print_confusion(title, conf, lut):
    print(f'\n=== {title}：按 UE 语义类别 ===')
    print(f'{"UE 类别":14s} {"期望 18 类":18s} {"点数":>9s} {"一致%":>7s} {"落在free%":>9s}  主要混淆')
    for tag in np.flatnonzero(conf.sum(1)):
        row = conf[tag]
        n = row.sum()
        exp = lut[tag]
        top = [(c, row[c]) for c in np.argsort(-row) if c not in (exp, 0) and row[c] > 0][:3]
        print(f'{UE_NAMES.get(tag, tag):14s} {OCCUPANCY_LABELS[exp]:18s} {n:9d} {100 * row[exp] / n:7.1f} '
              f'{100 * row[0] / n:9.1f}  ' + ', '.join(f'{OCCUPANCY_LABELS[c]} {100 * k / n:.1f}%' for c, k in top))
    total = conf.sum()
    agree = sum(conf[t, lut[t]] for t in range(256))
    print(f'总计 {total} 个表面点：语义一致 {100 * agree / max(total, 1):.1f}%，'
          f'落在 free {100 * conf[:, 0].sum() / max(total, 1):.1f}%')


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
    ap.add_argument('--legacy', action='store_true', help='用 2026-10-15 之前的包围盒/地面几何 (A/B 对比用)')
    ap.add_argument('--no-holdout', action='store_true', help='不挂留出相机 (只做内部一致性检查)')
    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    weather, _ = choose_weather('ClearNoon', None)
    client, world, tm_port = setup_carla(args.host, args.port, args.town, weather)
    client.get_trafficmanager(tm_port).set_random_device_seed(args.seed)
    world.set_pedestrians_seed(args.seed)
    vehicle, rig, hold_rig, lidar, npcs = None, None, None, None, []
    lut = np.zeros(256, np.uint8)
    for k, v in CARLA_TO_OCCUPANCY_MAPPING.items():
        lut[k] = v
    conf_gen = np.zeros((256, 18), np.int64)        # 行 = UE 语义标签，列 = 体素标签
    conf_hold = np.zeros((256, 18), np.int64)
    rot_hits = np.zeros(2, np.int64)                # 留出相机打在带旋转静态物体体素上的点：[总数, 类别一致]
    seen_occ = np.zeros(18, np.int64)               # 留出相机观测到的占用体素 (按类别)
    phantom = np.zeros(18, np.int64)                # 其中视线明确穿过的 (自由空间被误占用)
    rot_seen_phantom = np.zeros(2, np.int64)        # 带旋转静态物体：[观测到的体素, 被穿过的体素]
    ego_heights = []
    try:
        vehicle = spawn_vehicle(world, tm_port)
        npcs = spawn_npcs(client, world, args.num_vehicles, args.num_walkers, tm_port=tm_port)
        gen_cams = sorted(TESLA_CAMERAS, key=lambda c: c['index'])
        rig = PinholeRig(world, vehicle, [{'position': c['position'], 'rotation': c['rotation'],
                                           'fov': c['depth_fov']} for c in gen_cams])
        if not args.no_holdout:
            hold_rig = PinholeRig(world, vehicle, HOLDOUT)
        if args.lidar_filter:
            lidar = SemanticLidarSensor(world, vehicle, config=None)
            lidar.listen_to_queue()
        cams = [{'R': rotation_matrix(*c['rotation']).astype(np.float64), 't': np.asarray(c['position'], np.float64),
                 'hfov': c['depth_fov'], 'width': W, 'height': H, 'fisheye': fisheye_params(c)} for c in gen_cams]
        vis = CameraVisibility(cams, X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION)
        hold_vis = None
        if hold_rig is not None:
            hold_cams = [{'R': rotation_matrix(*h['rotation']), 't': np.asarray(h['position'], np.float64),
                          'hfov': h['fov'], 'width': W, 'height': H} for h in HOLDOUT]
            hold_vis = CameraVisibility(hold_cams, X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION)
        gen = GroundTruthVoxelGenerator(X_RANGE, Y_RANGE, Z_RANGE, RESOLUTION, legacy=args.legacy)
        vfilter = VisibilityFilterSimple()
        client.set_timeout(60.0)
        for _ in range(args.warmup_ticks):
            world.tick()
        ego_bb = vehicle.bounding_box
        for fi in range(args.frames):
            for _ in range(args.frame_skip):
                frame = world.tick()
            data = rig.get(frame)
            hold = hold_rig.get(frame) if hold_rig is not None else None
            tf = vehicle.get_transform()
            ego_heights.append((tf.location.z, tf.rotation.pitch, tf.rotation.roll))
            occ, aids = gen.generate(world, vehicle)
            if args.lidar_filter:
                lidar_data = lidar.get_data(timeout=10.0, frame=frame)
                occ, aids = vfilter.run(occ, aids, {'x_range': X_RANGE, 'y_range': Y_RANGE, 'z_range': Z_RANGE,
                                                    'resolution': RESOLUTION}, lidar_data, ego_id=vehicle.id)
            if not args.raw:
                _, st = finalize_ground_truth(vis, occ, aids, [d['depth'] for d in data], [d['sem'] for d in data],
                                              object_volume=gen.static_object_volume)
                print(f'  后处理: {st}')
            t, lab, _ = surface_labels(vis, data, occ, ego_bb)
            np.add.at(conf_gen, (t, lab), 1)
            if hold is not None:
                rotated_ids = [vid for vid, r in gen.static_object_rotated.items() if r]
                rot_mask = np.isin(aids, rotated_ids)
                t, lab, on_rot = surface_labels(hold_vis, hold, occ, ego_bb, rot_mask)
                np.add.at(conf_hold, (t, lab), 1)
                rot_hits += [on_rot.sum(), (lut[t[on_rot]] == lab[on_rot]).sum()]
                seen, through = hold_vis.analyze([h['depth'] for h in hold])
                occupied = occ > 0
                seen_occ += np.bincount(occ[seen & occupied], minlength=18)[:18]
                phantom += np.bincount(occ[through & occupied], minlength=18)[:18]
                rot_seen_phantom += [(seen & rot_mask).sum(), (through & rot_mask).sum()]
            print(f'帧 {fi + 1}/{args.frames} (sim frame {frame}) 完成', flush=True)
    finally:
        for a in [rig, hold_rig, lidar]:
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

    mode = 'legacy 几何' if args.legacy else '当前几何'
    print(f'\n自车 (z, pitch, roll): ' + ', '.join(f'({z:.1f},{p:.1f},{r:.1f})' for z, p, r in ego_heights))
    print_confusion(f'[{mode}] 生成相机 (内部一致性)', conf_gen, lut)
    if conf_hold.sum():
        print_confusion(f'[{mode}] 留出相机 (独立检查)', conf_hold, lut)
        print(f'\n=== [{mode}] 留出相机：自由空间被误占用 (占用体素被视线明确穿过的比例) ===')
        for c in range(1, 18):
            if seen_occ[c] > 200:
                print(f'{c:2d} {OCCUPANCY_LABELS[c]:18s} 观测到 {seen_occ[c]:8d}  被穿过 {100 * phantom[c] / seen_occ[c]:5.1f}%')
        print(f'全部占用体素: 被穿过 {100 * phantom[1:].sum() / max(seen_occ[1:].sum(), 1):.2f}%')
        print(f'\n=== [{mode}] 带旋转包围盒的静态物体 ===')
        print(f'留出相机表面点落在其体素上 {rot_hits[0]} 个，类别一致 {100 * rot_hits[1] / max(rot_hits[0], 1):.1f}%；'
              f'其体素被观测 {rot_seen_phantom[0]} 个，被视线穿过 {100 * rot_seen_phantom[1] / max(rot_seen_phantom[0], 1):.1f}%')


if __name__ == '__main__':
    main()

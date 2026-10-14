"""
光影效果对照预览：同一辆车、同一位置、同一天气下，分别用
  1) 我们自研的等距 Bayer RAW 相机（与 main_collection.py 完全相同的属性） → 去马赛克转图片
  2) 引擎标准针孔相机 sensor.camera.rgb（光栅化 + Lumen 软件 GI + 虚拟阴影贴图，即“UE 默认光影”）
  3) 上游 5.8 路径追踪相机 sensor.camera.rt_lens（equidistant 镜头，真正的光线追踪）
各拍一张，并拼成一张带标注的对照图。

三类相机分阶段拍摄、不同时存在：rt_lens（路径追踪视图）与光栅化相机在同一帧一起渲染会触发引擎断言
IsCompatible(Renderer->ViewFamily.EngineShowFlags) 导致编辑器崩溃（2026-10-14 实测）。

用法（5.8 服务器已在 Play）：
    C:\\Users\\bills\\.conda\\envs\\carla58\\python.exe lighting_preview.py --weather ClearNoon
    ... --weather ClearSunset --camera rear --skip-rt
    ... --sun-altitude 25          # 在预设基础上把太阳压低，阴影更长
"""

import argparse
import os
import queue
import sys

import numpy as np
import carla
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.camera_config import TESLA_CAMERAS, CAMERA_SENSOR_CONFIG  # noqa: E402

W = CAMERA_SENSOR_CONFIG['image_size_x']
H = CAMERA_SENSOR_CONFIG['image_size_y']


def cam_transform(cfg):
    p, r = cfg['position'], cfg['rotation']
    return carla.Transform(carla.Location(x=p[0], y=p[1], z=p[2]),
                           carla.Rotation(pitch=r[0], yaw=r[1], roll=r[2]))


def set_attrs(bp, attrs):
    for k, v in attrs.items():
        if bp.has_attribute(k):
            bp.set_attribute(k, str(v))
        else:
            print(f'    [跳过] {bp.id} 没有属性 {k}')


def ours_blueprint(bl, cfg, raw_type):
    """与 sensors/camera_manager.py::_setup_cameras 相同的属性。"""
    bp = bl.find('sensor.camera.rgb_fisheye')
    set_attrs(bp, {
        'image_size_x': W, 'image_size_y': H, 'sensor_tick': CAMERA_SENSOR_CONFIG['sensor_tick'],
        'perspective': 'False', 'equirectangular': 'False', 'fov_mask': 'False',
        'enable_postprocess_effects': CAMERA_SENSOR_CONFIG['enable_postprocess_effects'],
        'gamma': CAMERA_SENSOR_CONFIG['gamma'],
        'motion_blur_intensity': CAMERA_SENSOR_CONFIG['motion_blur_intensity'],
        'exposure_mode': CAMERA_SENSOR_CONFIG['exposure_mode'],
        **{k: CAMERA_SENSOR_CONFIG[k] for k in ('exposure_compensation', 'exposure_min_bright', 'exposure_max_bright',
                                               'exposure_speed_up', 'exposure_speed_down') if k in CAMERA_SENSOR_CONFIG},
        'shutter_speed': CAMERA_SENSOR_CONFIG['shutter_speed'],
        'iso': CAMERA_SENSOR_CONFIG['iso'],
        'fov': cfg['fov_vertical'],
        'camera_model': cfg.get('lens_model', 'equidistant'),
        'raw_type': raw_type,
    })
    return bp


def bayer_to_rgb(image):
    """RGGB uint16 马赛克 → 2x2 合并成半分辨率 RGB（不依赖 OpenCV 的 Bayer 命名约定），再放大回原尺寸。"""
    n = image.width * image.height
    raw = np.frombuffer(image.raw_data, dtype=np.uint16)[-n:].reshape(image.height, image.width).astype(np.float32)
    full_scale = 65535.0 if raw.max() > 4095 else 4095.0
    r = raw[0::2, 0::2]
    g = 0.5 * (raw[0::2, 1::2] + raw[1::2, 0::2])
    b = raw[1::2, 1::2]
    rgb = np.clip(np.stack([r, g, b], -1) / full_scale, 0, 1)
    rgb = np.repeat(np.repeat(rgb, 2, 0), 2, 1)
    stats = f'raw max={raw.max():.0f} full_scale={full_scale:.0f} mean={raw.mean():.0f}'
    return (rgb * 255 + 0.5).astype(np.uint8), stats


def bgra_to_rgb(image):
    n = image.width * image.height * 4
    a = np.frombuffer(image.raw_data, dtype=np.uint8)[-n:].reshape(image.height, image.width, 4)
    return a[:, :, 2::-1].copy()


def capture_phase(world, vehicle, specs, ticks):
    """specs: [(name, blueprint, transform, convert_fn)]，同一阶段一起 tick，最后一帧保存。"""
    actors, qs = [], {}
    try:
        for name, bp, tf, _ in specs:
            a = world.spawn_actor(bp, tf, attach_to=vehicle)
            q = queue.Queue()
            a.listen(q.put)
            actors.append(a)
            qs[name] = q
        last = {}
        for i in range(ticks):
            world.tick()
            for name, q in qs.items():
                try:
                    last[name] = q.get(timeout=120 if i == ticks - 1 else 30)
                except queue.Empty:
                    pass
                while not q.empty():
                    last[name] = q.get_nowait()
        out = {}
        for name, _, _, conv in specs:
            if name not in last:
                print(f'  {name}: 没有收到图像')
                continue
            res = conv(last[name])
            img, info = res if isinstance(res, tuple) else (res, '')
            out[name] = img
            print(f'  {name:28s} frame={last[name].frame} mean={img.mean():.1f} {info}')
        return out
    finally:
        for a in actors:
            a.stop()
        for a in actors:
            a.destroy()
        for _ in range(3):
            world.tick()


def contact_sheet(items, path):
    tw = 640
    tiles = []
    for label, img in items:
        im = Image.fromarray(img).resize((tw, int(img.shape[0] * tw / img.shape[1])))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, tw, 26], fill=(0, 0, 0))
        d.text((8, 6), label, fill=(255, 255, 0))
        tiles.append(im)
    sheet = Image.new('RGB', (tw * len(tiles), max(t.height for t in tiles)))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * tw, 0))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--weather', default='ClearNoon', help='carla.WeatherParameters 预设名（采集脚本用 ClearNoon）')
    ap.add_argument('--sun-altitude', type=float, default=None, help='覆盖太阳高度角（度），越低阴影越长')
    ap.add_argument('--camera', default='front_wide', help='camera_config.py 里的相机 id')
    ap.add_argument('--spawn-index', type=int, default=0)
    ap.add_argument('--ticks', type=int, default=30, help='每阶段 tick 数（让自动曝光/Lumen/路径追踪收敛）')
    ap.add_argument('--rt-spp', type=int, default=16, help='rt_lens samples_per_pixel')
    ap.add_argument('--skip-rt', action='store_true', help='不拍 rt_lens（路径追踪）')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    cfg = next(c for c in TESLA_CAMERAS if c['id'] == args.camera)
    tag = args.weather + (f'_sun{args.sun_altitude:g}' if args.sun_altitude is not None else '')
    out_dir = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs',
                                          'lighting_preview', f'{tag}_{args.camera}')
    os.makedirs(out_dir, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(120)
    world = client.get_world()
    original = world.get_settings()
    original_weather = world.get_weather()

    weather = getattr(carla.WeatherParameters, args.weather)
    if args.sun_altitude is not None:
        weather.sun_altitude_angle = args.sun_altitude
    world.set_weather(weather)
    print(f'天气 {args.weather}: sun_altitude={weather.sun_altitude_angle:.1f} '
          f'sun_azimuth={weather.sun_azimuth_angle:.1f} cloudiness={weather.cloudiness:.1f}')

    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 0.05
    world.apply_settings(s)

    bl = world.get_blueprint_library()
    vbp = (bl.filter('vehicle.tesla.model3') or bl.filter('vehicle.*'))[0]
    points = world.get_map().get_spawn_points()
    vehicle = None
    for idx in range(args.spawn_index, len(points)):
        vehicle = world.try_spawn_actor(vbp, points[idx])
        if vehicle is not None:
            print(f'车辆生成在 spawn point {idx}')
            break
    if vehicle is None:
        world.apply_settings(original)
        raise RuntimeError('所有 spawn point 都被占用')
    vehicle.apply_control(carla.VehicleControl(hand_brake=True))
    tf = cam_transform(cfg)
    hfov = cfg['fov']   # 历史上的水平 FOV（= fov_vertical * W/H）

    labels = {
        'ours_equidistant_bayer': '1 ours equidistant BAYER RAW (demosaic)',
        'ours_equidistant_uint8': '1b ours equidistant uint8',
        'ue_pinhole_raster': f'2 UE pinhole raster+Lumen (fov {min(hfov, 120):g})',
        'rt_lens_pathtraced': f'3 rt_lens path traced equidistant spp{args.rt_spp}',
    }
    results = []

    def save(phase_out):
        # 每阶段拍完立即落盘：rt_lens 阶段若把服务器弄崩，前面的结果不丢
        for name, img in phase_out.items():
            Image.fromarray(img).save(os.path.join(out_dir, name + '.png'))
            results.append((labels[name], img))
        contact_sheet(results, os.path.join(out_dir, 'compare.png'))

    try:
        print('阶段 1：我们的等距相机（Bayer RAW 与 uint8 两种输出）')
        save(capture_phase(world, vehicle, [
            ('ours_equidistant_bayer', ours_blueprint(bl, cfg, 'bayer_rggb'), tf, bayer_to_rgb),
            ('ours_equidistant_uint8', ours_blueprint(bl, cfg, 'uint8'), tf, bgra_to_rgb),
        ], args.ticks))

        print('阶段 2：引擎标准针孔相机（光栅化 + Lumen + VSM）')
        pin = bl.find('sensor.camera.rgb')
        set_attrs(pin, {'image_size_x': W, 'image_size_y': H, 'fov': min(hfov, 120)})
        save(capture_phase(world, vehicle, [('ue_pinhole_raster', pin, tf, bgra_to_rgb)], args.ticks))

        if not args.skip_rt:
            print('阶段 3：上游路径追踪相机 rt_lens（equidistant）')
            rt = bl.find('sensor.camera.rt_lens')
            set_attrs(rt, {'image_size_x': W, 'image_size_y': H, 'fov': hfov, 'camera_model': 'equidistant',
                           'samples_per_pixel': args.rt_spp, 'enable_denoiser': 'true'})
            save(capture_phase(world, vehicle, [('rt_lens_pathtraced', rt, tf, bgra_to_rgb)], args.ticks))

        print('对照图:', os.path.abspath(os.path.join(out_dir, 'compare.png')))
    finally:
        vehicle.destroy()
        world.set_weather(original_weather)   # 不要把预览用的天气留在服务器上
        world.apply_settings(original)


if __name__ == '__main__':
    main()

"""
CARLA 服务端实时预览窗口（pygame）：在服务端挂一个观察相机，把画面传回本窗口；可自由漫游或跟随车辆。
- 本机（182）运行：连远端 81，观察你的脚本在服务端做了什么。
- 81 上运行（由 carla_remote.ps1 的计划任务 CARLA-Viewer 在登录桌面里自动启动）：连 localhost，作为服务端的可视化界面。
两个窗口互不影响：各自是独立的只读客户端，各自挂自己的观察相机。

操作：
  自由漫游（默认）  W/S 前进后退  A/D 左右平移  Q 上升  E 下降  Shift 加速（x4）
                    按住鼠标左键或右键拖动 / 方向键 转视角  滚轮 调速度
  F     切换 自由漫游 / 跟随车辆（优先 role_name=hero/ego 的车，否则第一辆车）
  Tab   跟随时切换 车后追尾 / 车头 视角
  R     自由漫游时把相机移到服务端 spectator 位置
  Esc   退出（销毁观察相机）

特性：
- 只读：不调用 world.tick()、不改 settings/天气，不影响同步模式的采集脚本（同步模式下画面随采集脚本的 tick 刷新，
  漫游移动也在下一次 tick 才生效）。
- 服务端没启动 / 重启 / 崩溃时自动重连（窗口保持打开，显示等待状态）。
- ⚠️ 观察相机是光栅化相机，不要和 sensor.camera.rt_lens 同时使用（同帧渲染会触发引擎断言崩溃，见
  UE58_RENDERING_ISSUES.md 问题 4）。

用法：
    python remote\\carla_viewer.py                       # 默认连 81
    python remote\\carla_viewer.py --host localhost      # 在 81 本机
    python remote\\carla_viewer.py --follow hero --width 1600 --height 900
"""

import argparse
import queue
import socket
import time

import carla
import numpy as np
import pygame

SCANCODE_TO_KEY = {
    pygame.KSCAN_W: pygame.K_w, pygame.KSCAN_A: pygame.K_a, pygame.KSCAN_S: pygame.K_s, pygame.KSCAN_D: pygame.K_d,
    pygame.KSCAN_Q: pygame.K_q, pygame.KSCAN_E: pygame.K_e, pygame.KSCAN_LSHIFT: pygame.K_LSHIFT,
    pygame.KSCAN_RSHIFT: pygame.K_RSHIFT, pygame.KSCAN_UP: pygame.K_UP, pygame.KSCAN_DOWN: pygame.K_DOWN,
    pygame.KSCAN_LEFT: pygame.K_LEFT, pygame.KSCAN_RIGHT: pygame.K_RIGHT,
}

FOLLOW_VIEWS = [
    ('chase', carla.Transform(carla.Location(x=-7.0, z=3.2), carla.Rotation(pitch=-12.0))),
    ('hood', carla.Transform(carla.Location(x=0.8, z=1.7))),
]


def find_vehicle(world, follow):
    if follow not in ('auto', 'hero'):
        try:
            return world.get_actor(int(follow))
        except (ValueError, RuntimeError):
            return None
    vehicles = list(world.get_actors().filter('vehicle.*'))
    for v in vehicles:
        if v.attributes.get('role_name') in ('hero', 'ego', 'ego_vehicle'):
            return v
    return vehicles[0] if (vehicles and follow == 'auto') else None


class Viewer:
    def __init__(self, args):
        self.args = args
        pygame.init()
        pygame.display.set_caption(f'CARLA viewer - {args.host}:{args.port}')
        self.screen = pygame.display.set_mode((args.width, args.height))
        # SDL2 starts with text input enabled; a Chinese IME then swallows letter keys (WASD/QE never arrive,
        # while the mouse still works). Movement is not text, so turn text input / IME off for this window.
        pygame.key.stop_text_input()
        self.held = set()            # pygame key codes currently held (from KEYDOWN/KEYUP events)
        self.font = pygame.font.SysFont('consolas', 16)
        self.frames = queue.Queue(maxsize=2)
        self.world = None
        self.cam = None
        self.target = None
        self.mode = 'free' if args.follow == 'free' else 'follow'
        self.view_idx = 0
        self.free_tf = None          # carla.Transform of the free camera
        self.speed = 15.0            # m/s
        self.last_target_check = 0.0
        self.last_image = None
        self.frame_times = []        # receive timestamps of the last second (fps display)
        self.client = None
        self.tag = 'viewer@' + socket.gethostname()
        self.last_frame_time = time.time()

    # -- connection -------------------------------------------------------------------------------------------
    def connect(self):
        client = carla.Client(self.args.host, self.args.port)
        client.set_timeout(10.0)
        self.client = client
        self.world = client.get_world()
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Render/stream at a lower resolution and scale up locally: a raw BGRA frame is w*h*4 bytes over the
        # network (1280x720 = 3.7 MB, ~30 fps max on gigabit and shared with the user's scripts).
        bp.set_attribute('image_size_x', str(self.args.render_width))
        bp.set_attribute('image_size_y', str(self.args.render_height))
        bp.set_attribute('fov', str(self.args.fov))
        # Tag the camera with this machine: a viewer killed without cleanup (task restart, force-closed window)
        # leaves its camera rendering on the server forever. On connect, remove this machine's leftovers only;
        # viewers on other machines keep theirs.
        bp.set_attribute('role_name', self.tag)
        stale = [a.id for a in self.world.get_actors().filter('sensor.camera.rgb')
                 if a.attributes.get('role_name') == self.tag]
        if stale:
            client.apply_batch_sync([carla.command.DestroyActor(i) for i in stale])
        self.bp = bp
        if self.free_tf is None:
            self.free_tf = self.world.get_spectator().get_transform()
        self.respawn()

    def destroy_cam(self):
        if self.cam is not None:
            try:
                self.cam.stop()
                self.cam.destroy()
            except RuntimeError:
                pass
            self.cam = None

    def respawn(self):
        self.destroy_cam()
        self.target = find_vehicle(self.world, self.args.follow if self.args.follow != 'free' else 'auto') \
            if self.mode == 'follow' else None
        if self.target is not None:
            name, tf = FOLLOW_VIEWS[self.view_idx]
            kind = carla.AttachmentType.SpringArmGhost if name == 'chase' else carla.AttachmentType.Rigid
            self.cam = self.world.spawn_actor(self.bp, tf, attach_to=self.target, attachment_type=kind)
        else:
            self.cam = self.world.spawn_actor(self.bp, self.free_tf)
        self.cam.listen(lambda img: self.frames.full() or self.frames.put(img))

    # -- input --------------------------------------------------------------------------------------------------
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return False
            # Track held keys from events as well (by key code and by physical scancode -> key code), so movement
            # still works where pygame.key.get_pressed() is unreliable (IME, remote desktop key injection).
            if e.type in (pygame.KEYDOWN, pygame.KEYUP):
                codes = {e.key}
                mapped = SCANCODE_TO_KEY.get(getattr(e, 'scancode', None))
                if mapped is not None:
                    codes.add(mapped)   # physical key position survives IME / keyboard-layout remapping
                if e.type == pygame.KEYDOWN:
                    self.held |= codes
                else:
                    self.held -= codes
            elif e.type in (pygame.WINDOWFOCUSLOST, pygame.WINDOWLEAVE):
                self.held.clear()
            elif e.type == pygame.WINDOWFOCUSGAINED:
                pygame.key.stop_text_input()
            if e.type == pygame.KEYDOWN and self.world is not None:
                if e.key == pygame.K_f:
                    if self.mode == 'follow' and self.cam is not None:
                        self.free_tf = self.cam.get_transform()   # continue roaming from the current view
                    self.mode = 'free' if self.mode == 'follow' else 'follow'
                    self.respawn()
                elif e.key == pygame.K_TAB and self.mode == 'follow':
                    self.view_idx = (self.view_idx + 1) % len(FOLLOW_VIEWS)
                    self.respawn()
                elif e.key == pygame.K_r and self.mode == 'free':
                    self.free_tf = self.world.get_spectator().get_transform()
            if e.type == pygame.MOUSEWHEEL:
                self.speed = float(np.clip(self.speed * (1.25 ** e.y), 1.0, 200.0))
            if e.type == pygame.MOUSEMOTION and (e.buttons[0] or e.buttons[2]) and self.mode == 'free' \
                    and self.free_tf is not None:
                r = self.free_tf.rotation
                r.yaw += e.rel[0] * 0.15
                r.pitch = float(np.clip(r.pitch - e.rel[1] * 0.15, -89.0, 89.0))
                self.view_dirty = True
        return True

    def fly(self, dt):
        if self.mode != 'free' or self.free_tf is None or self.cam is None:
            return
        state = pygame.key.get_pressed()

        def down(k):
            return 1 if (k in self.held or state[k]) else 0

        r = self.free_tf.rotation
        turn = 60.0 * dt
        r.yaw += (down(pygame.K_RIGHT) - down(pygame.K_LEFT)) * turn
        r.pitch = float(np.clip(r.pitch + (down(pygame.K_UP) - down(pygame.K_DOWN)) * turn, -89.0, 89.0))
        fwd = self.free_tf.get_forward_vector()
        right = self.free_tf.get_right_vector()
        f = down(pygame.K_w) - down(pygame.K_s)
        s = down(pygame.K_d) - down(pygame.K_a)
        u = down(pygame.K_q) - down(pygame.K_e)
        turned = down(pygame.K_LEFT) or down(pygame.K_RIGHT) or down(pygame.K_UP) or down(pygame.K_DOWN)
        if f or s or u or turned or getattr(self, 'view_dirty', False):
            step = self.speed * dt * (4.0 if (down(pygame.K_LSHIFT) or down(pygame.K_RSHIFT)) else 1.0)
            loc = self.free_tf.location
            loc.x += (fwd.x * f + right.x * s) * step
            loc.y += (fwd.y * f + right.y * s) * step
            loc.z += (fwd.z * f) * step + u * step
            # Fire-and-forget: actor.set_transform() is a blocking RPC that waits for the server's reply, which
            # stalled every movement step for one network round trip. apply_batch() returns immediately.
            self.client.apply_batch([carla.command.ApplyTransform(self.cam.id, self.free_tf)])
            self.view_dirty = False

    # -- drawing ------------------------------------------------------------------------------------------------
    def draw(self, status):
        if self.last_image is not None:
            self.screen.blit(self.last_image, (0, 0))
        else:
            self.screen.fill((20, 20, 20))
        lines = [status]
        if self.mode == 'free':
            lines.append(f'free roam  speed {self.speed:.0f} m/s   WASD move  Q/E up/down  Shift x4  '
                         f'mouse-drag/arrows look  wheel speed  R spectator  F follow  Esc quit')
        else:
            who = f'{self.target.type_id} id={self.target.id}' if self.target is not None else 'no vehicle (free cam)'
            lines.append(f'follow {who}  view {FOLLOW_VIEWS[self.view_idx][0]}   Tab view  F free roam  Esc quit')
        for i, text in enumerate(lines):
            self.screen.blit(self.font.render(text, True, (255, 255, 0), (0, 0, 0)), (6, 6 + 20 * i))
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        status = 'connecting...'
        running = True
        while running:
            dt = clock.tick(60) / 1000.0
            running = self.handle_events()
            if not running:
                break
            try:
                if self.world is None:
                    self.connect()
                now = time.time()
                if self.mode == 'follow' and now - self.last_target_check > 2.0:
                    self.last_target_check = now
                    new_target = find_vehicle(self.world, self.args.follow if self.args.follow != 'free' else 'auto')
                    alive = self.target is not None and self.target.is_alive
                    if (self.target is None) != (new_target is None) or (self.target is not None and not alive) or \
                            (new_target is not None and self.target is not None and new_target.id != self.target.id):
                        self.respawn()
                if now - getattr(self, 'last_sync_check', 0.0) > 3.0:   # blocking RPC: keep it rare
                    self.last_sync_check = now
                    self.sync = self.world.get_settings().synchronous_mode
                self.fly(dt)
                img = None
                while True:   # keep only the newest frame
                    try:
                        img = self.frames.get_nowait()
                    except queue.Empty:
                        break
                if img is None and now - self.last_frame_time > 5.0 and getattr(self, 'sync', False) is False:
                    # No frames for 5 s in asynchronous mode: our camera was destroyed (e.g. by a reset script).
                    self.last_frame_time = now
                    self.respawn()
                if img is not None:
                    self.last_frame_time = now
                    a = np.frombuffer(img.raw_data, dtype=np.uint8)[-img.width * img.height * 4:]
                    a = a.reshape(img.height, img.width, 4)[:, :, 2::-1]
                    surf = pygame.surfarray.make_surface(a.swapaxes(0, 1))
                    if surf.get_size() != self.screen.get_size():
                        surf = pygame.transform.smoothscale(surf, self.screen.get_size())
                    self.last_image = surf
                    self.frame_times.append(now)
                    while self.frame_times and now - self.frame_times[0] > 1.0:
                        self.frame_times.pop(0)
                    status = (f'{self.args.host}  frame {img.frame}  recv {len(self.frame_times)} fps  '
                              f'stream {img.width}x{img.height}  sync={getattr(self, "sync", "?")}')
            except RuntimeError as err:
                # Server not up yet, restarting or crashed: drop everything and retry.
                self.destroy_cam()
                self.world = None
                self.last_image = None
                status = f'waiting for CARLA server {self.args.host}:{self.args.port} ... ({str(err)[:60]})'
                self.draw(status)
                time.sleep(3.0)
                continue
            self.draw(status)
        self.destroy_cam()
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='192.168.100.81')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--width', type=int, default=1280, help='窗口宽')
    ap.add_argument('--height', type=int, default=720, help='窗口高')
    ap.add_argument('--render-width', type=int, default=960,
                    help='服务端渲染/传输宽度（画面在本机放大到窗口大小；越小越流畅，在 81 本机可设成与窗口相同）')
    ap.add_argument('--render-height', type=int, default=540)
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--follow', default='free',
                    help="free（默认自由漫游）| auto（跟随 hero 或第一辆车）| hero | <actor id>")
    Viewer(ap.parse_args()).run()


if __name__ == '__main__':
    main()

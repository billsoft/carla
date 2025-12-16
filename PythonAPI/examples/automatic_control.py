#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
客户端自动车辆控制示例，使用Python定义的代理（Agent）。
本脚本展示了如何使用CARLA Python API创建一个自动驾驶车辆，
并使用预定义的Agent（如BehaviorAgent）来控制车辆在城镇中行驶。
"""

# ==============================================================================
# -- 导入模块说明 ---------------------------------------------------------------
# ==============================================================================
import argparse     # 用于解析命令行参数，例如设置服务器IP、端口、窗口分辨率等
import collections  # 用于使用高级数据结构，如 defaultdict（在CollisionSensor中记录碰撞历史）
import datetime     # 用于处理时间，例如显示仿真运行了多长时间
import logging      # 用于记录日志信息，替代print，方便控制输出级别（INFO, DEBUG等）
import math         # 数学函数库，用于计算向量长度（速度）、角度等
import os           # 操作系统接口，用于文件路径操作、判断操作系统类型等
import numpy.random as random # NumPy的随机数生成器，用于随机选择生成点、车辆蓝图等
import re           # 正则表达式模块，用于解析天气预设名称、过滤Actor蓝图等
import sys          # 系统相关参数和函数，主要用于修改 sys.path 以导入 carla 模块
import weakref      # 弱引用模块，关键！用于在传感器回调中引用自身对象，防止循环引用导致内存泄漏

try:
    import pygame   # Pygame库，用于创建图形窗口、渲染画面、处理键盘输入
    from pygame.locals import KMOD_CTRL # 导入键盘修饰键常量（如Ctrl键）
    from pygame.locals import K_ESCAPE  # 导入ESC键常量
    from pygame.locals import K_q       # 导入Q键常量
except ImportError:
    # 如果没有安装pygame，抛出运行时错误提示用户安装
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np # NumPy库，用于高效处理图像数据（如将相机原始数据转换为数组）
except ImportError:
    # 如果没有安装numpy，抛出运行时错误提示用户安装
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- 添加 PythonAPI 到系统路径 --------------------------------------------------
# ==============================================================================
# 这一步是为了确保 Python 解释器能找到 carla 模块。
# CARLA 的 PythonAPI 通常位于安装目录的 PythonAPI/carla/dist 目录下（以 .egg 文件形式存在）
# 或者在源码编译版本的 PythonAPI/carla 目录下。
# 下面的代码尝试将父目录的父目录（即假设脚本在 PythonAPI/examples 下）下的 carla 目录添加到搜索路径。
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla # 导入 CARLA 核心库
from carla import ColorConverter as cc # 导入颜色转换器，用于处理深度图、语义分割图等的颜色映射

# 导入 CARLA 官方提供的代理（Agent）类
# 这些 Agent 实现了基于路点（Waypoint）的导航和基本的交通规则遵守
# BehaviorAgent: 更高级的代理，可以模拟不同的驾驶风格（如谨慎、正常、激进）
# BasicAgent: 基础代理，能够规划路径并避障，但行为较简单
# ConstantVelocityAgent: 仅仅保持恒定速度行驶，不考虑交通规则
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error


# ==============================================================================
# -- 全局函数 ------------------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """
    查找并返回所有可用的天气预设。
    CARLA 在 carla.WeatherParameters 类中定义了一系列静态常量作为预设天气（如 ClearNoon, HardRainNight 等）。
    此函数通过反射（dir()）遍历该类属性，提取出大写字母开头的属性名作为预设。
    """
    # 正则表达式：用于将驼峰命名（如ClearNoon）拆分为单词（Clear Noon）
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    # 过滤出所有大写字母开头的属性，即天气预设常量
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    # 返回一个列表，包含 (天气参数对象, 可读名称) 的元组
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """
    获取 Actor 的易读显示名称。
    Actor 的 type_id 通常形如 'vehicle.tesla.model3'。
    此函数将其转换为 'Tesla Model3' 并在过长时截断。
    """
    # 将下划线替换为点，使用 title() 首字母大写，然后分割并取后面部分
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    # 如果名字太长，截断并添加省略号
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    """
    根据过滤器和生成代数（Generation）获取 Actor 的蓝图列表。
    Blueprints 是创建 Actor 的模板（相当于类），Actor 是实例（相当于对象）。
    """
    # 使用 world.get_blueprint_library() 获取所有可用蓝图，并应用 filter（如 'vehicle.*'）
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # 如果过滤器只返回一个蓝图，我们假设这就是用户明确指定的，忽略 generation 参数
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # 检查 generation 是否有效 (CARLA 车辆资源分为第1、2、3代)
        if int_generation in [1, 2, 3]:
            # 筛选出特定 generation 属性的蓝图
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# ==============================================================================
# -- World 类 ------------------------------------------------------------------
# ==============================================================================

class World(object):
    """
    World 类负责管理仿真环境的核心对象。
    它封装了 carla.World 对象，并管理玩家车辆、传感器、天气以及渲染回调。
    """

    def __init__(self, carla_world, hud, args):
        """
        初始化 World 对象。
        :param carla_world: carla.Client.get_world() 返回的世界对象
        :param hud: HUD 对象，用于显示信息
        :param args: 命令行参数
        """
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map() # 获取当前地图信息（OpenDRIVE格式），包含道路拓扑、路点等
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud # Heads-Up Display，用于在 Pygame 窗口绘制文本信息
        self.player = None # 玩家控制的车辆 Actor 对象
        self.collision_sensor = None # 碰撞传感器对象
        self.lane_invasion_sensor = None # 车道入侵传感器对象
        self.gnss_sensor = None # GNSS (GPS) 传感器对象
        self.camera_manager = None # 相机管理器，负责 RGB/深度/语义分割相机的切换和渲染
        self._weather_presets = find_weather_presets() # 加载天气预设
        self._weather_index = 0
        self._actor_filter = args.filter # 命令行参数指定的车辆过滤器（如 'vehicle.*'）
        self._actor_generation = args.generation # 命令行参数指定的车辆代数
        self.restart(args) # 调用 restart 方法初始化/重置环境
        # 注册回调函数：每当服务器更新一帧（tick），就会调用 hud.on_world_tick
        # 这用于同步客户端显示的时间和帧数
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        """
        重置世界状态：
        1. 选择车辆蓝图。
        2. 生成（Spawn）玩家车辆。
        3. 初始化并绑定传感器。
        """
        # 如果相机管理器已存在，保持其配置（如当前选中的相机类型和位置）
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # 获取符合条件的车辆蓝图列表
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("Couldn't find any blueprints with the specified filters")
        # 随机选择一个车辆蓝图
        blueprint = random.choice(blueprint_list)
        # 设置角色名为 'hero'。这对某些 Traffic Manager 或 Agent 很重要，
        # 它们可能会根据 role_name 区分主车和背景交通流车辆。
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            # 随机选择一个推荐颜色
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # 生成玩家车辆
        if self.player is not None:
            # 如果已有车辆（例如按了重置键），获取当前车辆的位置
            spawn_point = self.player.get_transform()
            # 稍微抬高一点，防止生成时陷入地下
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy() # 销毁旧车辆和传感器
            # 尝试在原位置生成新车
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            # 如果是第一次启动或原位置生成失败，从地图预定义的生成点（Spawn Points）中随机选择一个
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            # try_spawn_actor 如果位置被占用会返回 None，不会抛出异常
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

        # 确保车辆生成后再进行下一步
        if self._args.sync:
            self.world.tick() # 同步模式：强制服务器推进一步
        else:
            self.world.wait_for_tick() # 异步模式：等待服务器的下一个 tick

        # 初始化并绑定所有传感器到新生成的车辆上
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        # 恢复相机之前的状态
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        
        # 在 HUD 上显示当前车辆型号
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """切换下一个天气预设"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        # 应用新的天气参数到世界
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        """
        修改车辆物理属性。
        这里主要启用了 use_sweep_wheel_collision，这是一种更精确但开销稍大的车轮碰撞检测模式。
        """
        # 如果 actor 不是车辆，get_physics_control 会失败
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """每帧调用的更新逻辑，主要用于更新 HUD"""
        self.hud.tick(self, clock)

    def render(self, display):
        """
        每帧调用的渲染逻辑。
        1. 让 CameraManager 渲染相机图像。
        2. 让 HUD 渲染文字信息。
        """
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """仅销毁传感器"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """
        销毁所有由 World 管理的 Actor（包括传感器和车辆）。
        清理环境非常重要，否则下次运行时可能会有残留的幽灵车辆。
        """
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl 类 --------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """
    键盘控制类。
    虽然是自动驾驶，但仍需要处理系统级按键（如ESC退出）。
    注意：在 automatic_control.py 中，方向键不会控制车辆，车辆由 Agent 控制。
    """
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        """
        解析 pygame 事件队列。
        返回 True 表示需要退出程序。
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # 点击窗口关闭按钮
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key): # 按下 ESC 或 Ctrl+Q
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """判断是否为退出快捷键"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD 类 (Heads-Up Display) -------------------------------------------------
# ==============================================================================


class HUD(object):
    """
    HUD 类用于在屏幕上显示车辆状态、传感器信息和通知。
    """

    def __init__(self, width, height):
        """构造函数"""
        self.dim = (width, height)
        # 初始化字体
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        # 初始化辅助显示模块
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """
        World.on_tick 回调函数。
        当服务器完成一次 tick 时调用，timestamp 包含服务器端的仿真时间信息。
        """
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps() # 计算服务器端的 FPS
        self.frame = timestamp.frame_count # 当前帧数
        self.simulation_time = timestamp.elapsed_seconds # 仿真经过的秒数

    def tick(self, world, clock):
        """
        HUD 每帧更新逻辑。
        收集车辆和环境数据，准备 info_text 用于渲染。
        """
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        
        # 获取车辆的各种状态信息
        transform = world.player.get_transform() # 位置和旋转
        vel = world.player.get_velocity() # 速度向量
        control = world.player.get_control() # 当前的控制输入（油门、刹车、转向）
        
        # 计算朝向字符串 (N, S, E, W)
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        
        # 获取碰撞历史
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        
        # 获取周围车辆数量
        vehicles = world.world.get_actors().filter('vehicle.*')

        # 组装显示文本列表
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)), # 速度转换 m/s -> km/h
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        
        # 根据控制类型（车辆/行人）显示不同的控制信息
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        # 计算并显示最近车辆的距离
        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """切换是否显示信息"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """显示通知文本"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """显示错误文本（红色）"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """
        渲染 HUD 到 display Surface 上。
        绘制半透明背景板和文本信息。
        """
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100) # 设置透明度
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    # 绘制折线图（如碰撞历史）
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    # 绘制进度条（如油门、刹车力度）
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # 如果是普通字符串
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText 类 -------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """
    用于显示会随时间淡出的文本的辅助类。
    """

    def __init__(self, font, dim, pos):
        """构造函数"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """设置显示的文本和持续时间"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """每帧更新淡出效果"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left) # 根据剩余时间调整透明度

    def render(self, display):
        """渲染文本"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText 类 ---------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """
    用于渲染帮助文本的辅助类（显示脚本的 docstring）。
    """

    def __init__(self, font, width, height):
        """构造函数"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """切换显示/隐藏帮助"""
        self._render = not self._render

    def render(self, display):
        """渲染帮助文本"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor 类 --------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """
    碰撞传感器类。
    当车辆与其他物体碰撞时触发。
    """

    def __init__(self, parent_actor, hud):
        """构造函数"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        # 查找碰撞传感器的蓝图
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        # 生成传感器并附着到父对象（车辆）上
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        
        # 关键点：使用 weakref 避免循环引用。
        # 如果 lambda 直接引用 self，那么 self -> sensor -> callback -> self 形成引用环，导致对象无法被垃圾回收。
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """获取碰撞历史"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """碰撞回调函数"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        # 获取碰撞冲量，计算强度
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor 类 -----------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """
    车道入侵传感器类。
    当车辆跨越车道线（实线、虚线等）时触发。
    """

    def __init__(self, parent_actor, hud):
        """构造函数"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # 避免循环引用
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """车道入侵回调函数"""
        self = weak_self()
        if not self:
            return
        # event.crossed_lane_markings 包含跨过的所有车道线类型
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor 类 -------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """
    GNSS (Global Navigation Satellite System) 传感器类。
    报告车辆的地理坐标（经纬度）。
    """

    def __init__(self, parent_actor):
        """构造函数"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        # 将 GNSS 传感器安装在车辆中心偏上位置
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # 避免循环引用
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS 数据回调函数"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager 类 ----------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """
    相机管理器类。
    负责管理和渲染不同的相机和传感器（RGB, Depth, LiDAR等）。
    支持在多个安装位置和多个传感器类型之间切换。
    """

    def __init__(self, parent_actor, hud, gamma_correction=2.2):
        """构造函数"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        # 计算相机的安装位置，基于车辆的边界框（Bounding Box）进行相对定位
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        
        # AttachmentType 定义了传感器如何跟随车辆移动
        # Rigid: 刚性连接，完全跟随车辆的旋转和平移（类似固定在车架上）
        # SpringArmGhost: 弹簧臂连接，类似第三人称游戏视角，会有平滑的跟随效果
        attachment = carla.AttachmentType
        
        # 定义预设的相机变换列表 (位置/旋转, 连接方式)
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), attachment.Rigid),
            (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), attachment.Rigid)]

        world = self._parent.get_world()
        map_name = world.get_map().name
        post_process_profile = self.get_post_process_profile(map_name)

        self.transform_index = 1
        # 定义支持的传感器列表
        # 格式: [蓝图名称, 颜色转换器, 显示名称, 蓝图属性字典]
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {'post_process_profile': post_process_profile}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}]]

        # 初始化蓝图库，配置传感器属性
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                # 设置相机的分辨率为 HUD 窗口大小
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
                if blp.has_attribute('gamma'):
                    blp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    blp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp) # 将配置好的蓝图对象追加到列表末尾
        self.index = None

    def toggle_camera(self):
        """切换相机安装位置/视角"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """
        设置当前激活的传感器。
        :param index: 传感器在 self.sensors 中的索引
        :param notify: 是否在 HUD 显示通知
        :param force_respawn: 是否强制重新生成（例如切换视角时需要）
        """
        index = index % len(self.sensors)
        # 判断是否需要重新生成传感器：如果是新索引，或者强制刷新
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            # 生成新的传感器 Actor
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1], # 蓝图
                self._camera_transforms[self.transform_index][0], # 变换
                attach_to=self._parent, # 附着到车辆
                attachment_type=self._camera_transforms[self.transform_index][1]) # 附着类型

            # 注册数据回调
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """切换到下一个类型的传感器（如从 RGB 切换到 Depth）"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """切换是否将传感器数据保存到磁盘"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """将传感器画面渲染到屏幕"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    def get_post_process_profile(self, map_name: str) -> str:
        """根据地图获取后处理配置"""
        if "Town10HD_Opt" in map_name:
            return "Town10HD_Opt"
        return "Default"

    @staticmethod
    def _parse_image(weak_self, image):
        """
        传感器数据回调函数。
        将 CARLA 的 raw_data 转换为 Pygame 可以显示的 Surface。
        """
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            # 处理 LiDAR 数据
            # LiDAR 数据是 float32 点云 (x, y, z, intensity)
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            # 简单的顶视图投影：取 x, y 坐标
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            # 在黑色背景上绘制白色点
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            # 处理相机图像数据 (RGB, Depth, Segmentation)
            # 应用选定的颜色转换器 (cc)
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4)) # RGBA
            array = array[:, :, :3] # 去掉 Alpha 通道
            array = array[:, :, ::-1] # RGB -> BGR (Pygame 需要) 或者反之，视具体情况
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            # 保存图片到磁盘
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- Game Loop -----------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """
    仿真的主循环。
    负责初始化 Pygame，连接服务器，创建 World，并进行 tick 循环。
    """

    pygame.init()
    pygame.font.init()
    world = None

    try:
        # 设置随机种子 (Random Seed)
        # 如果用户在命令行提供了 --seed 参数，则初始化随机数生成器。
        # 作用：确保每次运行仿真时的随机行为（如车辆生成位置、颜色、Agent的行为决策等）是一致的。
        # 这对于复现 Bug、调试或进行对比实验非常关键。
        if args.seed:
            random.seed(args.seed)

        # 1. 连接到 CARLA 服务器
        client = carla.Client(args.host, args.port)
        # 设置客户端超时时间，防止网络问题导致无限等待
        client.set_timeout(60.0)

        traffic_manager = client.get_trafficmanager()
        # sim_world 是 CARLA API 原生提供的 carla.World 对象。
        # 它代表了仿真器中的整个世界环境（服务端），用于获取地图、设置天气、配置同步模式等全局操作。
        sim_world = client.get_world()

        # 2. 设置同步模式 (Synchronous Mode)
        # 在同步模式下，服务器会等待客户端发送 tick 信号后才进行下一次更新。
        # 这保证了传感器数据的严格同步和仿真的一致性。
        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            # 设置固定的时间步长，例如 0.05秒 (20 FPS)
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        # 3. 初始化 Pygame 显示窗口
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        # 4. 创建 World 和 HUD 对象
        hud = HUD(args.width, args.height)
        # world 是本脚本自定义的 World 类（见第134行定义）的实例。
        # 它是一个客户端的包装类 (Wrapper)，用于封装和管理当前客户端感兴趣的状态，
        # 例如：玩家车辆 (Hero Vehicle)、传感器 (Sensors)、相机管理器、天气预设状态等。
        # 它内部持有一个 carla.World 对象 (sim_world) 的引用来与服务器交互。
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        # 5. 初始化自动驾驶代理 (Agent)
        if args.agent == "Basic":
            agent = BasicAgent(world.player, 30) # 目标速度 30 km/h
            agent.follow_speed_limits(True)
        elif args.agent == "Constant":
            agent = ConstantVelocityAgent(world.player, 30)
            # 简单的防止车辆生成在空中的修正
            ground_loc = world.world.ground_projection(world.player.get_location(), 5)
            if ground_loc:
                world.player.set_location(ground_loc.location + carla.Location(z=0.01))
            agent.follow_speed_limits(True)
        elif args.agent == "Behavior":
            agent = BehaviorAgent(world.player, behavior=args.behavior)

        # 6. 设置 Agent 的目的地
        spawn_points = world.map.get_spawn_points()
        # 随机选择一个目的地
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)

        clock = pygame.time.Clock()

        # 7. 进入主循环
        while True:
            clock.tick() # 限制客户端 FPS（如果是异步模式）
            
            # 核心同步逻辑
            if args.sync:
                world.world.tick() # 发送信号让服务器计算下一帧
            else:
                world.world.wait_for_tick() # 等待服务器更新
                
            # 处理键盘事件（退出）
            if controller.parse_events():
                return

            # 更新客户端逻辑（HUD, 传感器处理等）
            world.tick(clock)
            # 渲染画面
            world.render(display)
            pygame.display.flip() # 刷新屏幕

            # 检查 Agent 状态
            if agent.done():
                if args.loop:
                    # 如果开启循环模式，到达目的地后选择新目的地
                    agent.set_destination(random.choice(spawn_points).location)
                    world.hud.notification("Target reached", seconds=4.0)
                    print("The target has been reached, searching for another target")
                else:
                    print("The target has been reached, stopping the simulation")
                    break

            # 8. Agent 计算控制指令
            control = agent.run_step()
            control.manual_gear_shift = False
            # 9. 将控制指令应用到车辆
            world.player.apply_control(control)

    finally:
        # 清理工作：恢复服务器设置，销毁 Actor
        # 无论程序如何退出（正常结束或异常中断），都必须执行清理
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """入口函数，处理命令行参数"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose', action='store_true', dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host', metavar='H', default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port', metavar='P', default=2000, type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res', metavar='WIDTHxHEIGHT', default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync', action='store_true',
        help='Synchronous mode execution (推荐开启，保证仿真稳定)')
    argparser.add_argument(
        '--filter', metavar='PATTERN', default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation', metavar='G', default='All',
        help='restrict to certain actor generation (values: "2","3","All" - default: "All")')
    argparser.add_argument(
        '-l', '--loop', action='store_true', dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str, choices=["Behavior", "Basic", "Constant"], default="Behavior",
        help="select which agent to run (选择代理类型)")
    argparser.add_argument(
        '-b', '--behavior', type=str, choices=["cautious", "normal", "aggressive"], default='normal',
        help='Choose one of the possible agent behaviors (default: normal) (仅BehaviorAgent有效)')
    argparser.add_argument(
        '-s', '--seed', default=None, type=int,
        help='Set seed for repeating executions (default: None)')

    args = argparser.parse_args()

    # 解析分辨率字符串 "1280x720" -> width=1280, height=720
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()

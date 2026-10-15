"""NPC 围绕自车循环回收 (2026-10-15)。

体素真值只覆盖自车周围 X/Y ±40m；NPC 按地图生成点铺满整张地图时，大部分车/人离自车几百米、
从来进不了体素范围，却照样占渲染、物理和显存。这里让固定数量的 NPC 始终集中在自车周围：

- 初始：生成点按离自车的距离排序 (spawn_npcs(center=...))，就近生成。
- 循环：离自车超过 recycle_radius (默认 80m = 体素半径 40m 的 2 倍) 的车辆/行人瞬移到离自车
  [ring_min, ring_max] (默认 45~75m，体素范围外) 的环形区域 (不销毁重生，见 recycle() 里的说明)。
  优先放在自车侧后方，减少在前视相机里"凭空出现"；新车在体素范围外出现，不会突兀地进入体素真值，
  时序样本里也不会有凭空冒出来的体素。

Traffic Manager 自带的 set_respawn_dormant_vehicles 只对 Large Map (World Partition) 生效，
Town10HD / Town15 用不上，所以自己做。
"""
import math

import carla
import numpy as np

VEHICLE_SPACING_M = 10.0     # 重生点与任何车辆的最小距离
WALKER_SPACING_M = 2.0


def _forward_score(ego_tf, loc):
    """越靠自车正前方越大 (cos)，用来优先选侧后方的重生点。"""
    f = ego_tf.get_forward_vector()
    dx, dy = loc.x - ego_tf.location.x, loc.y - ego_tf.location.y
    d = math.hypot(dx, dy)
    return (f.x * dx + f.y * dy) / d if d > 1e-3 else 0.0


def configure_npc_vehicle(tm, npc, tm_port):
    npc.set_autopilot(True, tm_port)
    tm.distance_to_leading_vehicle(npc, 2.5)
    tm.ignore_lights_percentage(npc, 0.0)
    tm.auto_lane_change(npc, True)


def sidewalk_point(cmap, center, ring, rng, accept=None, tries=30):
    """离 center 在 [ring[0], ring[1]] 米内的一个人行道点 (路点就近投影，比导航网格随机查询快几个数量级)。"""
    lo, hi = ring
    for _ in range(tries):
        r = rng.uniform(lo, hi)
        a = rng.uniform(-math.pi, math.pi)
        p = carla.Location(center.x + r * math.cos(a), center.y + r * math.sin(a), center.z)
        wp = cmap.get_waypoint(p, project_to_road=True, lane_type=carla.LaneType.Sidewalk)
        if wp is None:
            continue
        loc = wp.transform.location
        if lo <= loc.distance(center) <= hi and (accept is None or accept(loc)):
            return loc
    return None


def start_walker_controllers(cmap, controllers, rng):
    """启动行人 AI，并给每人一个 10~40m 外的人行道目标 (只 start() 不给目标的行人多数原地不动)。
    控制器要在生成后至少一次 world.tick() 才能 start。"""
    for ctrl in controllers:
        try:
            if not ctrl.is_alive:
                continue
            ctrl.start()
            target = sidewalk_point(cmap, ctrl.get_location(), (10.0, 40.0), rng)
            if target is not None:
                ctrl.go_to_location(target)
            ctrl.set_max_speed(1.0 + float(rng.random()))
        except RuntimeError:
            pass


def walker_controller_map(actors):
    """{行人 id: 其 AI 控制器}。优先用控制器的 parent；取不到时按 spawn_npcs 的生成顺序 (行人后紧跟其控制器) 配对。"""
    out = {}
    prev_walker = None
    for a in actors:
        try:
            alive = a.is_alive
        except RuntimeError:
            alive = False
        if a.type_id.startswith('walker.'):
            prev_walker = a if alive else None
        elif a.type_id.startswith('controller.') and alive:
            parent = a.parent
            if parent is not None:
                out[parent.id] = a
            elif prev_walker is not None and prev_walker.id not in out:
                out[prev_walker.id] = a
            prev_walker = None
    return out


class NpcRecycler:
    def __init__(self, client, world, ego, npc_actors, tm_port, recycle_radius=80.0, ring=(45.0, 75.0),
                 max_per_call=8, rng=None):
        self.client = client
        self.world = world
        self.ego = ego
        self.npc_actors = npc_actors          # main_collection 清理时用的同一个列表，原地增删
        self.tm_port = tm_port
        self.tm = client.get_trafficmanager(tm_port)
        self.recycle_radius = recycle_radius
        self.ring = ring
        self.max_per_call = max_per_call
        self.rng = rng if rng is not None else np.random.default_rng()
        self.bp_lib = world.get_blueprint_library()
        self.map = world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.controller_bp = self.bp_lib.find('controller.ai.walker')
        self.pending_controllers = []         # 新行人的 AI 控制器要在一次 tick 之后才能 start
        self.stats = {'vehicles': 0, 'walkers': 0}

    # ---------------------------------------------------------------- 工具
    def _alive(self, a):
        try:
            return a.is_alive
        except RuntimeError:
            return False

    def _too_close(self, loc, kind, spacing):
        for a in self.world.get_actors().filter(kind):
            if a.get_location().distance(loc) < spacing:
                return True
        return False

    def _vehicle_slots(self, ego_tf):
        lo, hi = self.ring
        c = ego_tf.location
        cand = [sp for sp in self.spawn_points if lo <= sp.location.distance(c) <= hi]
        # 侧后方优先，同档内随机
        self.rng.shuffle(cand)
        cand.sort(key=lambda sp: _forward_score(ego_tf, sp.location) > 0.5)
        return cand

    def _walker_location(self, ego_tf):
        loc = sidewalk_point(self.map, ego_tf.location, self.ring, self.rng,
                             accept=lambda l: _forward_score(ego_tf, l) < 0.7)
        return None if loc is None else carla.Location(loc.x, loc.y, loc.z + 1.0)

    # ---------------------------------------------------------------- 主入口
    def start_pending(self):
        """在 recycle() 之后的第一次 world.tick() 之后调用：启动新行人的 AI 控制器并给一个人行道目标点。"""
        start_walker_controllers(self.map, self.pending_controllers, self.rng)
        self.pending_controllers = []

    def recycle(self):
        """把离自车太远的 NPC 挪回自车周围。返回 (回收车辆数, 回收行人数)。"""
        if self.recycle_radius <= 0:
            return 0, 0
        ego_tf = self.ego.get_transform()
        c = ego_tf.location
        controllers = walker_controller_map(self.npc_actors)

        far_vehicles, far_walkers = [], []
        for a in self.npc_actors:
            if not self._alive(a):
                continue
            if a.type_id.startswith('vehicle.') and a.get_location().distance(c) > self.recycle_radius:
                far_vehicles.append(a)
            elif a.type_id.startswith('walker.') and a.get_location().distance(c) > self.recycle_radius:
                far_walkers.append(a)

        # 瞬移而不是销毁+重生 (2026-10-15)：采集中途销毁/生成 actor 时 81 触发了光追 shader 访问显存页错误
        # (MMU fault, ray_tracing_*) 整机 GPU 崩溃——每帧约 30 个 cube 面视图在做光追，中途增删几何体风险高。
        # 瞬移保持场景里的几何体集合不变；车型/颜色在一个场景内不变，对数据无影响
        nv = nw = 0
        slots = self._vehicle_slots(ego_tf) if far_vehicles else []
        for veh in far_vehicles[:self.max_per_call]:
            target = None
            while slots and target is None:
                sp = slots.pop(0)
                if not self._too_close(sp.location, 'vehicle.*', VEHICLE_SPACING_M):
                    target = sp
            if target is None:
                break                                  # 环形区域里没有空位了，下次再试
            tf = carla.Transform(carla.Location(target.location.x, target.location.y, target.location.z + 0.3),
                                 target.rotation)
            veh.set_target_velocity(carla.Vector3D(0, 0, 0))
            veh.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            veh.set_transform(tf)
            nv += 1

        for walker in far_walkers[:self.max_per_call]:
            loc = self._walker_location(ego_tf)
            if loc is None or self._too_close(loc, 'walker.*', WALKER_SPACING_M):
                continue
            ctrl = controllers.get(walker.id)
            if ctrl is not None:
                ctrl.stop()
            walker.set_target_velocity(carla.Vector3D(0, 0, 0))
            walker.set_location(loc)
            if ctrl is not None:
                self.pending_controllers.append(ctrl)   # 下一次 tick 之后重新 start + 给新目标点
            nw += 1

        self.stats['vehicles'] += nv
        self.stats['walkers'] += nw
        return nv, nw

    def nearby_counts(self, radius=40.0):
        """自车 radius 内的 NPC 车辆/行人数 (日志用)。"""
        c = self.ego.get_location()
        nv = sum(1 for a in self.npc_actors if a.type_id.startswith('vehicle.') and self._alive(a)
                 and a.get_location().distance(c) <= radius)
        nw = sum(1 for a in self.npc_actors if a.type_id.startswith('walker.') and self._alive(a)
                 and a.get_location().distance(c) <= radius)
        return nv, nw

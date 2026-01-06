"""
场景管理器 (Scenario Manager)
负责管理 CARLA 世界中的 Actor 生成与销毁，包括：
1. Hero 车辆 (Ego Vehicle)
2. NPC 车辆 (Car, Truck, Bus, Bike, Moto)
3. 行人 (Pedestrians + AI Controllers)
"""

import carla
import numpy as np
import time

class ScenarioManager:
    def __init__(self, world, tm_port=8010):
        self.world = world
        self.tm_port = tm_port
        self.hero_vehicle = None
        self.npc_actors = [] # [vehicles..., walkers..., controllers...]
        self.bp_lib = world.get_blueprint_library()
        self.spawn_points = world.get_map().get_spawn_points()

    def spawn_hero(self, filter_pattern='vehicle.lincoln.mkz*', enable_autopilot=True):
        """生成 Hero 车辆并启用自动驾驶

        Args:
            filter_pattern: 车辆蓝图过滤模式
            enable_autopilot: 是否启用自动驾驶（默认 True）
        """
        print(f"\n[Scenario] 正在生成 Hero 车辆 ({filter_pattern})...")

        # 1. 选择蓝图
        bp = self.bp_lib.filter(filter_pattern)[0]
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name', 'hero')

        # 2. 选择生成点 (避开原点)
        valid_points = [p for p in self.spawn_points if abs(p.location.x) > 5 or abs(p.location.y) > 5]
        if not valid_points:
            valid_points = self.spawn_points

        # 3. 尝试生成
        for idx, point in enumerate(valid_points):
            self.hero_vehicle = self.world.try_spawn_actor(bp, point)
            if self.hero_vehicle:
                # 物理稳定
                for _ in range(10):
                    self.world.tick()

                loc = self.hero_vehicle.get_location()
                print(f"  ✓ Hero 生成成功: ID={self.hero_vehicle.id}, Loc=({loc.x:.1f}, {loc.y:.1f})")

                # 再次检查是否被弹回原点
                if abs(loc.x) < 1.0 and abs(loc.y) < 1.0:
                    print("  ⚠ 警告: Hero 位于原点，可能发生碰撞，重试...")
                    self.hero_vehicle.destroy()
                    self.hero_vehicle = None
                    continue

                # 启用自动驾驶
                if enable_autopilot:
                    # ⭐ 修复: 添加端口重试机制,避免 TM 端口绑定失败
                    tm_ports_to_try = [self.tm_port, 8011, 8012, 8013, 8014, 8015]
                    autopilot_success = False

                    for port in tm_ports_to_try:
                        try:
                            self.hero_vehicle.set_autopilot(True, port)
                            self.tm_port = port  # 更新成功的端口
                            print(f"  ✓ 自动驾驶已启用 (TM Port: {port})")
                            autopilot_success = True
                            break
                        except RuntimeError as e:
                            if "bind error" in str(e):
                                print(f"  ⚠ TM 端口 {port} 被占用,尝试下一个...")
                                continue
                            else:
                                raise

                    if not autopilot_success:
                        print(f"  ⚠ 警告: 无法启用自动驾驶 (所有端口都被占用)")
                        print(f"  → 车辆将保持静止状态")

                return self.hero_vehicle

        raise RuntimeError("无法生成 Hero 车辆")

    def spawn_npcs(self, num_vehicles=30, num_walkers=10):
        """生成 NPC (车辆 + 行人)"""
        print(f"\n[Scenario] 正在生成 NPC (Vehicles={num_vehicles}, Walkers={num_walkers})...")
        self._spawn_vehicles(num_vehicles)
        self._spawn_walkers(num_walkers)
        
        # 稳定物理
        print("  [Scenario] 等待 NPC 物理稳定...")
        for _ in range(10):
            self.world.tick()

    def _spawn_vehicles(self, count):
        """生成 NPC 车辆 (按比例)"""
        categories = {
            'car':      {'pattern': 'vehicle.audi.*', 'ratio': 0.5},
            'truck':    {'pattern': 'vehicle.carlamotors.firetruck', 'ratio': 0.1}, # 简化pattern，可扩展
            'bus':      {'pattern': 'vehicle.mitsubishi.fusorosa', 'ratio': 0.1},
            'bicycle':  {'pattern': 'vehicle.bh.crossbike', 'ratio': 0.15},
            'moto':     {'pattern': 'vehicle.yamaha.*', 'ratio': 0.15}
        }
        
        # 扩展 Pattern 列表 (这里简化展示，实际可引用之前的完整列表)
        full_patterns = {
            'car': ['vehicle.audi.*', 'vehicle.bmw.*', 'vehicle.tesla.*', 'vehicle.toyota.*', 'vehicle.lincoln.*'],
            'truck': ['vehicle.carlamotors.carlacola', 'vehicle.tesla.cybertruck', 'vehicle.ford.ambulance'],
            'bus': ['vehicle.mitsubishi.fusorosa'],
            'bicycle': ['vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets'],
            'moto': ['vehicle.harley*', 'vehicle.kawasaki.*', 'vehicle.yamaha.*', 'vehicle.vespa.*']
        }

        spawn_idx = 0
        total_spawned = 0
        
        for cat, info in categories.items():
            target_num = int(count * info['ratio'])
            if target_num == 0 and info['ratio'] > 0: target_num = 1
            
            # 获取蓝图
            bps = []
            for pat in full_patterns.get(cat, [info['pattern']]):
                bps.extend(list(self.bp_lib.filter(pat)))
            
            if not bps: continue
            
            spawned_count = 0
            for _ in range(target_num):
                if spawn_idx >= len(self.spawn_points): break
                
                bp = np.random.choice(bps)
                if bp.has_attribute('color'):
                    color = np.random.choice(bp.get_attribute('color').recommended_values)
                    bp.set_attribute('color', color)
                
                # 尝试生成
                # 注意：spawn_points 列表可能已经被 hero 占用了一个，这里简单顺序往下
                # 更好的做法是打乱或检查占用，但 try_spawn_actor 会自动处理碰撞返回 None
                while spawn_idx < len(self.spawn_points):
                    npc = self.world.try_spawn_actor(bp, self.spawn_points[spawn_idx])
                    spawn_idx += 1
                    if npc:
                        # ⭐ 修复: 添加错误处理,避免 TM 端口问题导致崩溃
                        try:
                            npc.set_autopilot(True, self.tm_port)
                        except RuntimeError as e:
                            if "bind error" in str(e):
                                print(f"  ⚠ NPC 自动驾驶失败 (TM 端口问题),车辆将保持静止")
                            else:
                                raise

                        self.npc_actors.append(npc)
                        spawned_count += 1
                        total_spawned += 1
                        break
            
            print(f"  - {cat}: {spawned_count} 辆")

    def _spawn_walkers(self, count):
        """生成行人 (在 Hero 附近)"""
        if not self.hero_vehicle:
            print("  ⚠ 警告: 没有 Hero，行人将随机生成")
            hero_loc = carla.Location(0,0,0)
        else:
            hero_loc = self.hero_vehicle.get_location()

        walker_bps = list(self.bp_lib.filter('walker.pedestrian.*'))
        controller_bp = self.bp_lib.find('controller.ai.walker')
        
        spawned_count = 0
        controllers = []
        
        for _ in range(count):
            # 在 Hero 周围 15-50m 生成
            dist = np.random.uniform(15, 50)
            angle = np.random.uniform(0, 6.28)
            loc = carla.Location(
                x = hero_loc.x + dist * np.cos(angle),
                y = hero_loc.y + dist * np.sin(angle),
                z = hero_loc.z + 1.0
            )
            trans = carla.Transform(loc)
            
            bp = np.random.choice(walker_bps)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
                
            walker = self.world.try_spawn_actor(bp, trans)
            if walker:
                self.npc_actors.append(walker)
                
                # Controller
                con = self.world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                if con:
                    self.npc_actors.append(con)
                    controllers.append(con)
                    spawned_count += 1
        
        # 启动 AI
        for con in controllers:
            con.start()
            con.set_max_speed(1.0 + np.random.random())
            
        print(f"  - Walkers: {spawned_count} 人")

    def destroy(self):
        """清理所有 Actor"""
        print("\n[Scenario] 清理场景 Actor...")
        
        # 1. 停止 AI
        for actor in self.npc_actors:
            if 'controller' in actor.type_id:
                actor.stop()
        
        # 2. 销毁 NPC
        # client.apply_batch is faster, but simple loop is safer for logic
        for actor in self.npc_actors:
            if actor.is_alive:
                actor.destroy()
        self.npc_actors.clear()
        print("  ✓ NPC 已销毁")

        # 3. 销毁 Hero
        if self.hero_vehicle and self.hero_vehicle.is_alive:
            self.hero_vehicle.destroy()
            self.hero_vehicle = None
            print("  ✓ Hero 已销毁")

"""NPC环境管理模块

负责:
1. 生成NPC车辆
2. 生成行人
3. 配置Traffic Manager
4. 销毁所有NPC
"""

import carla
import random
import time


class NPCManager:
    """NPC车辆和行人管理器"""

    def __init__(self, world, traffic_manager_port=8000):
        """
        初始化NPC管理器

        参数:
            world: CARLA World对象
            traffic_manager_port: Traffic Manager端口
        """
        self.world = world
        self.client = world.get_client()
        self.map = world.get_map()
        self.bp_lib = world.get_blueprint_library()

        # Traffic Manager
        self.tm = self.client.get_trafficmanager(traffic_manager_port)
        self.tm.set_global_distance_to_leading_vehicle(2.5)
        self.tm.set_synchronous_mode(True)

        # 已生成的Actor列表
        self.vehicle_list = []
        self.walker_list = []
        self.walker_controller_list = []

        print(f"NPC管理器初始化完成 (TM端口: {traffic_manager_port})")

    def spawn_vehicles(self, num_vehicles=50, autopilot=True):
        """
        生成NPC车辆

        参数:
            num_vehicles: 要生成的车辆数量
            autopilot: 是否启用自动驾驶

        返回:
            成功生成的车辆列表
        """
        print(f"\n开始生成 {num_vehicles} 辆NPC车辆...")

        # 获取所有车辆蓝图
        vehicle_bps = self.bp_lib.filter('vehicle.*')

        # 过滤掉不常见的车型
        vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute('number_of_wheels')) == 4]

        # 获取所有可用的生成点
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)

        # 批量生成
        batch = []
        for i, spawn_point in enumerate(spawn_points[:num_vehicles]):
            bp = random.choice(vehicle_bps)

            # 随机设置颜色
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)

            # 设置驾驶员
            if bp.has_attribute('driver_id'):
                driver_id = random.choice(bp.get_attribute('driver_id').recommended_values)
                bp.set_attribute('driver_id', driver_id)

            # 创建生成命令
            batch.append(carla.command.SpawnActor(bp, spawn_point))

        # 执行批量生成
        results = self.client.apply_batch_sync(batch, True)

        # 收集成功生成的车辆
        for i, response in enumerate(results):
            if response.error:
                print(f"  车辆 {i} 生成失败: {response.error}")
            else:
                vehicle = self.world.get_actor(response.actor_id)
                self.vehicle_list.append(vehicle)

                # 启用自动驾驶
                if autopilot:
                    vehicle.set_autopilot(True, self.tm.get_port())

        print(f"成功生成 {len(self.vehicle_list)} 辆NPC车辆")
        return self.vehicle_list

    def spawn_pedestrians(self, num_pedestrians=50, crossing_factor=0.3):
        """
        生成行人

        参数:
            num_pedestrians: 要生成的行人数量
            crossing_factor: 穿越马路的概率 (0-1)

        返回:
            成功生成的行人列表
        """
        print(f"\n开始生成 {num_pedestrians} 个行人...")

        # 获取所有行人蓝图
        walker_bps = self.bp_lib.filter('walker.pedestrian.*')

        # 随机选择生成位置
        spawn_points = []
        for i in range(num_pedestrians):
            spawn_point = carla.Transform()

            # 随机选择地图上的一个位置
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        # 批量生成行人
        batch = []
        for spawn_point in spawn_points:
            bp = random.choice(walker_bps)

            # 随机设置性别、年龄等
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')

            batch.append(carla.command.SpawnActor(bp, spawn_point))

        # 执行批量生成
        results = self.client.apply_batch_sync(batch, True)

        # 收集成功生成的行人
        walkers = []
        for response in results:
            if not response.error:
                walker = self.world.get_actor(response.actor_id)
                walkers.append(walker)
                self.walker_list.append(walker)

        # 生成行人控制器
        batch = []
        walker_controller_bp = self.bp_lib.find('controller.ai.walker')
        for walker in walkers:
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker))

        results = self.client.apply_batch_sync(batch, True)

        for response in results:
            if not response.error:
                controller = self.world.get_actor(response.actor_id)
                self.walker_controller_list.append(controller)

        # 等待控制器初始化
        self.world.tick()

        # 启动行人AI
        for controller in self.walker_controller_list:
            controller.start()

            # 随机设置目标位置
            target_loc = self.world.get_random_location_from_navigation()
            if target_loc is not None:
                controller.go_to_location(target_loc)

            # 设置移动速度
            controller.set_max_speed(1.4 + random.random())  # 1.4-2.4 m/s

        # 设置一些行人穿越马路
        num_crossing = int(len(self.walker_controller_list) * crossing_factor)
        for controller in random.sample(self.walker_controller_list, num_crossing):
            # 让行人忽略交通灯(敢闯红灯)
            controller.set_max_speed(1.8)

        print(f"成功生成 {len(self.walker_list)} 个行人 "
              f"(其中 {num_crossing} 个会穿越马路)")

        return self.walker_list

    def configure_traffic_manager(self, global_speed_perc=0, vehicle_speed_perc=None,
                                   ignore_lights_perc=0, ignore_signs_perc=0):
        """
        配置Traffic Manager行为

        参数:
            global_speed_perc: 全局速度偏移百分比 (负数=减速, 正数=加速)
            vehicle_speed_perc: 各车辆速度偏移字典 {vehicle: perc}
            ignore_lights_perc: 忽略交通灯的车辆百分比 (0-100)
            ignore_signs_perc: 忽略交通标志的车辆百分比 (0-100)
        """
        print("\n配置Traffic Manager...")

        # 全局速度限制
        if global_speed_perc != 0:
            self.tm.global_percentage_speed_difference(global_speed_perc)
            print(f"  全局速度偏移: {global_speed_perc:+.0f}%")

        # 单独车辆速度
        if vehicle_speed_perc:
            for vehicle, perc in vehicle_speed_perc.items():
                self.tm.vehicle_percentage_speed_difference(vehicle, perc)

        # 随机设置一些车辆忽略交通规则
        if ignore_lights_perc > 0 and self.vehicle_list:
            num_ignore_lights = int(len(self.vehicle_list) * ignore_lights_perc / 100.0)
            for vehicle in random.sample(self.vehicle_list, num_ignore_lights):
                self.tm.ignore_lights_percentage(vehicle, 100.0)
            print(f"  {num_ignore_lights} 辆车忽略交通灯")

        if ignore_signs_perc > 0 and self.vehicle_list:
            num_ignore_signs = int(len(self.vehicle_list) * ignore_signs_perc / 100.0)
            for vehicle in random.sample(self.vehicle_list, num_ignore_signs):
                self.tm.ignore_signs_percentage(vehicle, 100.0)
            print(f"  {num_ignore_signs} 辆车忽略交通标志")

    def get_statistics(self):
        """获取NPC统计信息"""
        return {
            'num_vehicles': len(self.vehicle_list),
            'num_pedestrians': len(self.walker_list),
            'total_actors': len(self.vehicle_list) + len(self.walker_list)
        }

    def destroy_all(self):
        """销毁所有NPC"""
        print("\n开始清理NPC...")

        # 停止所有行人控制器
        for controller in self.walker_controller_list:
            if controller.is_alive:
                controller.stop()

        # 批量销毁
        all_actors = self.vehicle_list + self.walker_list + self.walker_controller_list

        print(f"销毁 {len(all_actors)} 个Actor...")

        self.client.apply_batch([carla.command.DestroyActor(x) for x in all_actors])

        # 清空列表
        self.vehicle_list.clear()
        self.walker_list.clear()
        self.walker_controller_list.clear()

        print("NPC清理完成")

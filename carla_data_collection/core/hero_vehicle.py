"""
Hero 车辆管理器
管理主车辆、8 个相机和语义激光雷达
"""

import carla
import numpy as np
from typing import Optional, Dict, List

from config.camera_config import TESLA_CAMERA_CONFIGS


class HeroVehicleManager:
    """Hero 车辆管理器"""

    def __init__(self,
                 world: carla.World,
                 vehicle_model: str = 'vehicle.tesla.model3',
                 spawn_point: Optional[carla.Transform] = None):
        """
        Args:
            world: CARLA 世界对象
            vehicle_model: 车辆模型蓝图 ID
            spawn_point: 生成位置 (None 则随机选择)
        """
        self.world = world
        self.vehicle_model = vehicle_model
        self.vehicle: Optional[carla.Vehicle] = None
        self.cameras: Dict = {}
        self.lidar: Optional = None  # 语义激光雷达

        # 生成车辆
        self._spawn_vehicle(spawn_point)

    def _spawn_vehicle(self, spawn_point: Optional[carla.Transform]):
        """生成 Hero 车辆"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(self.vehicle_model)

        # 设置为 Hero 角色
        vehicle_bp.set_attribute('role_name', 'hero')

        # 设置颜色 (可选: Tesla 银色)
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '200,200,200')

        # 选择生成点
        if spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = np.random.choice(spawn_points)

        # 生成车辆
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        print(f"[HeroVehicle] 车辆已生成: {self.vehicle.type_id}")
        print(f"  位置: ({spawn_point.location.x:.1f}, "
              f"{spawn_point.location.y:.1f}, "
              f"{spawn_point.location.z:.1f})")

    def attach_cameras(self) -> Dict:
        """
        附加 8 个相机传感器

        Returns:
            cameras: {camera_id: CameraSensor} 字典
        """
        from sensors.camera_sensor import CameraSensor

        for cam_config in TESLA_CAMERA_CONFIGS:
            camera_sensor = CameraSensor(
                world=self.world,
                vehicle=self.vehicle,
                config=cam_config
            )

            self.cameras[cam_config['id']] = camera_sensor

        print(f"[HeroVehicle] 已附加 {len(self.cameras)} 个相机")

        return self.cameras

    def attach_semantic_lidar(self) -> 'SemanticLidarSensor':
        """
        附加语义激光雷达 (用于生成 Occupancy GT)

        Returns:
            lidar: SemanticLidarSensor 对象
        """
        from sensors.semantic_lidar_sensor import SemanticLidarSensor

        self.lidar = SemanticLidarSensor(
            world=self.world,
            vehicle=self.vehicle
        )

        print(f"[HeroVehicle] 已附加语义激光雷达")

        return self.lidar

    def enable_autopilot(self, traffic_manager_port: int = 8000):
        """启用自动驾驶"""
        if self.vehicle is None:
            raise RuntimeError("车辆未生成!")

        self.vehicle.set_autopilot(True, traffic_manager_port)
        print("[HeroVehicle] 自动驾驶已启用")

    def get_vehicle_state(self) -> Dict:
        """
        获取车辆状态

        Returns:
            state: 包含位置、速度、加速度等信息的字典
        """
        if self.vehicle is None:
            return {}

        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()

        return {
            'timestamp': self.world.get_snapshot().timestamp.elapsed_seconds,
            'frame': self.world.get_snapshot().frame,
            'transform': transform,  # CARLA Transform 对象
            'location': {
                'x': transform.location.x,
                'y': transform.location.y,
                'z': transform.location.z
            },
            'rotation': {
                'pitch': transform.rotation.pitch,
                'yaw': transform.rotation.yaw,
                'roll': transform.rotation.roll
            },
            'velocity': {
                'x': velocity.x,
                'y': velocity.y,
                'z': velocity.z,
                'magnitude': np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            },
            'acceleration': {
                'x': acceleration.x,
                'y': acceleration.y,
                'z': acceleration.z
            }
        }

    def destroy(self):
        """销毁车辆和所有传感器"""
        # 销毁相机
        for camera in self.cameras.values():
            camera.destroy()

        # 销毁激光雷达
        if self.lidar is not None:
            self.lidar.destroy()

        # 销毁车辆
        if self.vehicle is not None:
            self.vehicle.destroy()

        print("[HeroVehicle] 已销毁所有资源")

"""
语义激光雷达传感器
用于生成 3D Occupancy Ground Truth
"""

import carla
import numpy as np
import queue
from typing import Optional, Dict, Tuple

from config.occupancy_config import SEMANTIC_LIDAR_CONFIG


class SemanticLidarSensor:
    """语义激光雷达传感器"""

    def __init__(self,
                 world: carla.World,
                 vehicle: carla.Vehicle,
                 config: Optional[Dict] = None):
        """
        Args:
            world: CARLA 世界对象
            vehicle: 附加的车辆
            config: 激光雷达配置 (默认使用 SEMANTIC_LIDAR_CONFIG)
        """
        self.world = world
        self.vehicle = vehicle
        self.config = config or SEMANTIC_LIDAR_CONFIG
        self.sensor: Optional[carla.Sensor] = None
        self.data_queue = queue.Queue(maxsize=2)

        # 创建传感器
        self._create_sensor()

    def _create_sensor(self):
        """创建语义激光雷达传感器"""
        blueprint_library = self.world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')

        # 设置参数
        lidar_bp.set_attribute('channels', str(self.config['channels']))
        lidar_bp.set_attribute('points_per_second', str(self.config['points_per_second']))
        lidar_bp.set_attribute('rotation_frequency', str(self.config['rotation_frequency']))
        lidar_bp.set_attribute('range', str(self.config['range']))
        lidar_bp.set_attribute('upper_fov', str(self.config['upper_fov']))
        lidar_bp.set_attribute('lower_fov', str(self.config['lower_fov']))
        
        # ⭐ 支持水平 FOV 设置 (默认 360)
        if 'horizontal_fov' in self.config:
             lidar_bp.set_attribute('horizontal_fov', str(self.config['horizontal_fov']))

        # 创建 Transform
        position = self.config['position']
        rotation = self.config['rotation']

        transform = carla.Transform(
            carla.Location(x=position['x'], y=position['y'], z=position['z']),
            carla.Rotation(pitch=rotation['pitch'], yaw=rotation['yaw'], roll=rotation['roll'])
        )

        # 生成传感器
        self.sensor = self.world.spawn_actor(
            lidar_bp, transform, attach_to=self.vehicle
        )

        print(f"[SemanticLidar] 已创建: {self.config['channels']}线, "
              f"{self.config['range']}m 范围, "
              f"{self.config['points_per_second']/1e6:.1f}M点/秒")

    def listen_to_queue(self):
        """将数据推送到队列"""
        def queue_callback(lidar_data):
            try:
                # copy raw_data to bytes to avoid memory corruption
                # when the C++ object is destroyed
                data_copy = bytes(lidar_data.raw_data)
                
                self.data_queue.put_nowait({
                    'timestamp': lidar_data.timestamp,
                    'frame': lidar_data.frame,
                    'raw_data': data_copy
                })
            except queue.Full:
                # 队列满,丢弃旧数据
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    pass
                
                data_copy = bytes(lidar_data.raw_data)
                self.data_queue.put_nowait({
                    'timestamp': lidar_data.timestamp,
                    'frame': lidar_data.frame,
                    'raw_data': data_copy
                })

        self.sensor.listen(queue_callback)

    def get_data(self, timeout: float = 2.0):
        """
        获取一帧 LiDAR 数据
        Returns:
            {
                'points': (N, 3),
                'obj_idx': (N,),
                'tags': (N,),
                'timestamp': float,
                'frame': int
            }
        """
        try:
            data = self.data_queue.get(timeout=timeout)
            parsed = self._parse_lidar_data(data['raw_data']) # Pass bytes directly
            parsed['timestamp'] = data['timestamp']
            parsed['frame'] = data['frame']
            return parsed
        except queue.Empty:
            raise TimeoutError("Semantic LiDAR data timeout")

    def clear_queues(self):
        """清空数据队列"""
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
        print(f"[SemanticLidar] 已清空数据队列")

    def _parse_lidar_data(self, raw_bytes):
        """
        解析 Semantic LiDAR 数据 (结构化 numpy 数组)
        Format: x, y, z, cos_angle, object_idx, semantic_tag
        """
        # 定义混合结构
        dtype = np.dtype([
            ('x', np.float32), 
            ('y', np.float32), 
            ('z', np.float32), 
            ('cos', np.float32), 
            ('obj_idx', np.uint32), 
            ('tag', np.uint32)
        ])
        
        # 从 raw_data 直接读取
        data = np.frombuffer(raw_bytes, dtype=dtype)
        
        # 提取坐标 (N, 3)
        points = np.stack((data['x'], data['y'], data['z']), axis=-1)
        
        # 坐标系转换: CARLA (X-Forward, Y-Right, Z-Up) -> Custom if needed
        # 这里保持 CARLA 坐标系，后续在 Filter 中处理转换
        # points[:, 1] = -points[:, 1] # Flip Y if converting to Left-Handed
        
        return {
            'points': points,             # (N, 3) float32
            'obj_idx': data['obj_idx'],   # (N,) uint32
            'tags': data['tag']           # (N,) uint32
        }

    def get_point_cloud_stats(self, xyz_world: np.ndarray, semantic_tags: np.ndarray):
        """
        打印点云统计信息

        Args:
            xyz_world: (N, 3) 点云坐标
            semantic_tags: (N,) 语义标签
        """
        print(f"\n[SemanticLidar] 点云统计:")
        print(f"  总点数: {len(xyz_world)}")
        print(f"  坐标范围:")
        print(f"    X: {xyz_world[:, 0].min():.1f} ~ {xyz_world[:, 0].max():.1f} 米")
        print(f"    Y: {xyz_world[:, 1].min():.1f} ~ {xyz_world[:, 1].max():.1f} 米")
        print(f"    Z: {xyz_world[:, 2].min():.1f} ~ {xyz_world[:, 2].max():.1f} 米")

        print(f"  语义标签分布:")
        unique_tags, counts = np.unique(semantic_tags, return_counts=True)
        for tag, count in sorted(zip(unique_tags, counts), key=lambda x: -x[1])[:10]:
            percentage = count / len(semantic_tags) * 100
            print(f"    标签 {tag:2d}: {count:6d} 点 ({percentage:5.2f}%)")

    def destroy(self):
        """销毁传感器"""
        if self.sensor is not None:
            self.sensor.destroy()
            print("[SemanticLidar] 已销毁")

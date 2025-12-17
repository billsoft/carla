"""
语义激光雷达传感器
用于生成 3D Occupancy Ground Truth
"""

import carla
import numpy as np
import queue
from typing import Optional, Dict, Tuple

from dense_occupancy_collection.config.occupancy_config import SEMANTIC_LIDAR_CONFIG


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
                self.data_queue.put_nowait({
                    'timestamp': lidar_data.timestamp,
                    'frame': lidar_data.frame,
                    'raw_data': lidar_data.raw_data
                })
            except queue.Full:
                # 队列满,丢弃旧数据
                self.data_queue.get()
                self.data_queue.put_nowait({
                    'timestamp': lidar_data.timestamp,
                    'frame': lidar_data.frame,
                    'raw_data': lidar_data.raw_data
                })

        self.sensor.listen(queue_callback)

    def parse_lidar_data(self, raw_data: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """
        解析激光雷达原始数据
        
        Args:
            raw_data: CARLA 激光雷达原始二进制数据
        
        Returns:
            xyz_world: (N, 3) 局部坐标系下的点云
            semantic_tags: (N,) 每个点的语义标签 (CARLA 标签)
        """
        # 使用结构化数组正确读取混合类型数据
        # Layout: x(f32), y(f32), z(f32), cos(f32), idx(u32), tag(u32)
        dtype = np.dtype([
            ('x', np.float32), 
            ('y', np.float32), 
            ('z', np.float32), 
            ('cos', np.float32), 
            ('obj_idx', np.uint32), 
            ('tag', np.uint32)
        ])
        
        data = np.frombuffer(raw_data, dtype=dtype)
        
        # 提取坐标 (N, 3)
        xyz_local = np.stack([data['x'], data['y'], data['z']], axis=1)
        
        # 提取标签 (N,)
        semantic_tags = data['tag'].astype(np.int32)
        
        return xyz_local, semantic_tags

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

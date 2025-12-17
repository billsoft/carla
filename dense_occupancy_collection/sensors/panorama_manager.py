
import carla
import numpy as np
import weakref
import cv2
from queue import Queue, Empty
from dense_occupancy_collection.config.panorama_config import (
    PANO_WIDTH, PANO_HEIGHT, CUBE_SIZE, CUBE_FOV, 
    CUBE_FACE_CONFIGS, PANO_LOCATION
)
from dense_occupancy_collection.processing.panorama_tools import PanoramaTools

class PanoramaSensorManager:
    """
    全景传感器管理器
    管理 6个方向 x 2种类型(深度+语义) = 12个传感器
    负责采集并拼接生成全景图
    """
    
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.bp_library = world.get_blueprint_library()
        
        # 传感器列表
        self.sensors = []
        self.queues = {} # { 'front_depth': q, 'front_sem': q, ... }
        
        # 工具
        self.pano_tools = PanoramaTools(PANO_WIDTH, PANO_HEIGHT, CUBE_SIZE)
        
        # 初始化
        self._setup_sensors()
        
    def _setup_sensors(self):
        """创建 12 个传感器"""
        
        # 基础位置 (车顶中心)
        base_loc = carla.Location(x=PANO_LOCATION['x'], y=PANO_LOCATION['y'], z=PANO_LOCATION['z'])
        
        for face in CUBE_FACE_CONFIGS:
            name = face['name']
            rot = face['rot']
            
            transform = carla.Transform(
                base_loc,
                carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
            )
            
            # 1. 深度相机
            self._spawn_camera(name, 'depth', transform)
            
            # 2. 语义分割相机
            self._spawn_camera(name, 'semantic', transform)
            
    def _spawn_camera(self, face_name, sensor_type, transform):
        """生成单个相机"""
        if sensor_type == 'depth':
            bp_name = 'sensor.camera.depth'
        else:
            bp_name = 'sensor.camera.semantic_segmentation'
            
        bp = self.bp_library.find(bp_name)
        bp.set_attribute('image_size_x', str(CUBE_SIZE))
        bp.set_attribute('image_size_y', str(CUBE_SIZE))
        bp.set_attribute('fov', str(CUBE_FOV))
        bp.set_attribute('sensor_tick', '0.0') # 同步模式下跟随 tick
        
        sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        self.sensors.append(sensor)
        
        # 队列键名: e.g. "front_depth"
        queue_key = f"{face_name}_{sensor_type}"
        queue = Queue()
        self.queues[queue_key] = queue
        
        weak_queue = weakref.ref(queue)
        sensor.listen(lambda data, q=weak_queue: self._sensor_callback(data, q))
        
    def _sensor_callback(self, data, weak_queue):
        queue = weak_queue()
        if queue:
            queue.put(data)
            
    def get_panorama_frame(self, timeout=2.0):
        """
        获取一帧全景数据
        
        Returns:
            dict: {
                'depth_pano': (H, W) float32 (meters),
                'semantic_pano': (H, W) uint8 (class_id),
                'timestamp': float
            }
        """
        # 1. 获取所有面数据
        raw_data = {}
        timestamp = 0
        
        try:
            for key, queue in self.queues.items():
                data = queue.get(timeout=timeout)
                raw_data[key] = data
                timestamp = data.timestamp
        except Empty:
            print("⚠ 全景相机数据超时")
            return None
            
        # 2. 整理数据用于拼接
        # 顺序: Front, Right, Back, Left, Up, Down
        faces_order = ['front', 'right', 'back', 'left', 'up', 'down']
        
        depth_faces = []
        semantic_faces = []
        
        for face_name in faces_order:
            # 处理深度
            d_img = raw_data[f"{face_name}_depth"]
            d_array = np.frombuffer(d_img.raw_data, dtype=np.uint8)
            d_array = d_array.reshape((CUBE_SIZE, CUBE_SIZE, 4))
            
            # 解码深度
            R = d_array[:, :, 2].astype(np.float32)
            G = d_array[:, :, 1].astype(np.float32)
            B = d_array[:, :, 0].astype(np.float32)
            normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
            depth_meters = normalized * 1000.0
            depth_faces.append(depth_meters)
            
            # 处理语义
            s_img = raw_data[f"{face_name}_semantic"]
            s_array = np.frombuffer(s_img.raw_data, dtype=np.uint8)
            s_array = s_array.reshape((CUBE_SIZE, CUBE_SIZE, 4))
            semantic_faces.append(s_array[:, :, 2]) # Red channel
            
        # 3. 拼接
        depth_pano = self.pano_tools.stitch(depth_faces)
        semantic_pano = self.pano_tools.stitch(semantic_faces)
        
        return {
            'depth_pano': depth_pano,
            'semantic_pano': semantic_pano,
            'timestamp': timestamp
        }
        
    def destroy(self):
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors = []
        self.queues = {}

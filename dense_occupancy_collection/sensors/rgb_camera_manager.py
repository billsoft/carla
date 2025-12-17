"""
RGB相机管理器
管理8个RGB相机的创建、数据采集和同步
用于可视化和模型输入
"""

import carla
import numpy as np
import weakref
from queue import Queue, Empty


class RGBCameraManager:
    """
    RGB相机管理器

    为每个相机配置创建RGB相机，带物理鱼眼畸变效果
    """

    def __init__(self, world, vehicle, camera_configs):
        """
        初始化RGB相机管理器

        Args:
            world: carla.World 对象
            vehicle: carla.Vehicle 对象 (hero车辆)
            camera_configs: list of camera config dicts
        """
        self.world = world
        self.vehicle = vehicle
        self.camera_configs = camera_configs

        self.sensors = {}  # {cam_id: sensor}
        self.queues = {}   # {cam_id: Queue}
        self.transforms = {}  # {cam_id: carla.Transform}

        self._setup_sensors()

    def _setup_sensors(self):
        """
        创建所有RGB相机传感器
        """
        bp_library = self.world.get_blueprint_library()
        rgb_bp = bp_library.find('sensor.camera.rgb')

        for cam_cfg in self.camera_configs:
            cam_id = cam_cfg['id']

            # 配置传感器属性
            rgb_bp.set_attribute('image_size_x', str(cam_cfg.get('image_size_x', 1280)))
            rgb_bp.set_attribute('image_size_y', str(cam_cfg.get('image_size_y', 960)))
            rgb_bp.set_attribute('fov', str(cam_cfg['fov']))
            rgb_bp.set_attribute('sensor_tick', str(cam_cfg.get('sensor_tick', 0.1)))

            # 启用后处理效果 (解决光照缺失问题)
            # 参考 PythonAPI/examples/manual_control.py:
            # 必须开启 postprocess 才能应用 post_process_profile (Town10HD_Opt)
            if rgb_bp.has_attribute('enable_postprocess_effects'):
                rgb_bp.set_attribute('enable_postprocess_effects', 'True')

            # 设置 Post Process Profile (针对 Town10HD_Opt)
            map_name = self.world.get_map().name
            if 'Town10HD_Opt' in map_name and rgb_bp.has_attribute('post_process_profile'):
                rgb_bp.set_attribute('post_process_profile', 'Town10HD_Opt')
            
            # 设置 Gamma (参考 manual_control.py 默认值 2.2)
            if rgb_bp.has_attribute('gamma'):
                rgb_bp.set_attribute('gamma', '2.2')
                
            # 设置快门速度和 ISO 以增加亮度 (如果场景太暗)
            if rgb_bp.has_attribute('shutter_speed'):
                rgb_bp.set_attribute('shutter_speed', '200.0') # 默认通常是 60-200
            if rgb_bp.has_attribute('iso'):
                rgb_bp.set_attribute('iso', '1200.0') # 提高 ISO 增加亮度
            
            # 鱼眼畸变参数
            # 情况1: 直接在 cam_cfg 中
            if 'lens_circle_multiplier' in cam_cfg:
                rgb_bp.set_attribute('lens_circle_multiplier', str(cam_cfg['lens_circle_multiplier']))
                rgb_bp.set_attribute('lens_circle_falloff', str(cam_cfg['lens_circle_falloff']))
                rgb_bp.set_attribute('lens_k', str(cam_cfg['lens_k']))
                rgb_bp.set_attribute('lens_kcube', str(cam_cfg['lens_kcube']))
                if 'lens_x_size' in cam_cfg:
                    rgb_bp.set_attribute('lens_x_size', str(cam_cfg['lens_x_size']))
                if 'lens_y_size' in cam_cfg:
                    rgb_bp.set_attribute('lens_y_size', str(cam_cfg['lens_y_size']))
            
            # 情况2: 在 lens_distortion 嵌套字典中 (carla_data_collection 风格)
            elif 'lens_distortion' in cam_cfg and cam_cfg['lens_distortion'] is not None:
                dist = cam_cfg['lens_distortion']
                for key, val in dist.items():
                    if rgb_bp.has_attribute(key):
                        rgb_bp.set_attribute(key, str(val))

            # 创建Transform
            transform = carla.Transform(
                carla.Location(
                    x=cam_cfg['x'],
                    y=cam_cfg['y'],
                    z=cam_cfg['z']
                ),
                carla.Rotation(
                    pitch=cam_cfg['pitch'],
                    yaw=cam_cfg['yaw'],
                    roll=cam_cfg['roll']
                )
            )
            self.transforms[cam_id] = transform

            # 生成传感器
            sensor = self.world.spawn_actor(rgb_bp, transform, attach_to=self.vehicle)
            self.sensors[cam_id] = sensor

            # 创建队列
            queue = Queue()
            self.queues[cam_id] = queue

            # 注册回调
            weak_queue = weakref.ref(queue)
            sensor.listen(lambda image, q=weak_queue: self._sensor_callback(image, q))

            print(f"✓ RGB相机创建成功: {cam_id} (FOV={cam_cfg['fov']}°)")

    def _sensor_callback(self, image, weak_queue):
        """
        传感器数据回调函数

        Args:
            image: carla.Image 对象 (RGB图)
            weak_queue: 弱引用队列
        """
        queue = weak_queue()
        if queue is not None:
            queue.put(image)

    def get_data(self, timeout=2.0):
        """
        获取所有相机的最新数据

        Args:
            timeout: 超时时间 (秒)

        Returns:
            dict: {cam_id: rgb_image_array} or None if timeout
        """
        data = {}

        for cam_id, queue in self.queues.items():
            try:
                image = queue.get(timeout=timeout)

                # 转换为numpy数组 (BGRA格式)
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))

                # 转换为RGB格式
                rgb_array = array[:, :, [2, 1, 0]]  # BGR → RGB

                data[cam_id] = {
                    'data': rgb_array,
                    'frame': image.frame,
                    'timestamp': image.timestamp,
                    'transform': self.transforms[cam_id]
                }

            except Empty:
                print(f"⚠ RGB相机数据超时: {cam_id}")
                return None

        return data

    def destroy(self):
        """
        销毁所有传感器
        """
        for cam_id, sensor in self.sensors.items():
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
                print(f"✓ RGB相机已销毁: {cam_id}")

        self.sensors.clear()
        self.queues.clear()

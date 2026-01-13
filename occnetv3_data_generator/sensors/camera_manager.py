"""
Tesla 8相机管理器
支持 Bayer RGGB Raw 数据采集
"""
import carla
import numpy as np
import queue
import weakref
from typing import Dict, Optional

from config.camera_config import TESLA_CAMERAS, CAMERA_SENSOR_CONFIG


class CameraManager:
    """
    管理8个RGB相机, 支持 Bayer RGGB 转换
    """

    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        """
        Args:
            world: CARLA世界对象
            vehicle: 车辆actor
        """
        self.world = world
        self.vehicle = vehicle
        self.cameras = {}  # {cam_id: carla.Sensor}
        self.camera_configs = TESLA_CAMERAS
        self.data_queues = {}  # {cam_id: queue.Queue}

        self._setup_cameras()

    def _setup_cameras(self):
        """创建并附加8个相机"""
        bp_library = self.world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')

        # 设置基础属性
        camera_bp.set_attribute('image_size_x', str(CAMERA_SENSOR_CONFIG['image_size_x']))
        camera_bp.set_attribute('image_size_y', str(CAMERA_SENSOR_CONFIG['image_size_y']))
        camera_bp.set_attribute('sensor_tick', str(CAMERA_SENSOR_CONFIG['sensor_tick']))

        # 后处理设置
        if camera_bp.has_attribute('enable_postprocess_effects'):
            camera_bp.set_attribute('enable_postprocess_effects',
                                   str(CAMERA_SENSOR_CONFIG['enable_postprocess_effects']))
        
        # 自动应用 Town10HD_Opt 的后处理配置文件 (解决光照问题)
        map_name = self.world.get_map().name
        if 'Town10HD_Opt' in map_name and camera_bp.has_attribute('post_process_profile'):
            camera_bp.set_attribute('post_process_profile', 'Town10HD_Opt')

        if camera_bp.has_attribute('gamma'):
            camera_bp.set_attribute('gamma', str(CAMERA_SENSOR_CONFIG['gamma']))
            
        if camera_bp.has_attribute('motion_blur_intensity'):
            camera_bp.set_attribute('motion_blur_intensity',
                                   str(CAMERA_SENSOR_CONFIG['motion_blur_intensity']))
                                   
        if camera_bp.has_attribute('exposure_mode'):
            camera_bp.set_attribute('exposure_mode', CAMERA_SENSOR_CONFIG['exposure_mode'])
            
        if camera_bp.has_attribute('shutter_speed'):
            camera_bp.set_attribute('shutter_speed', str(CAMERA_SENSOR_CONFIG['shutter_speed']))
            
        if camera_bp.has_attribute('iso'):
            camera_bp.set_attribute('iso', str(CAMERA_SENSOR_CONFIG['iso']))

        for cam_config in self.camera_configs:
            cam_id = cam_config['id']

            # 设置FOV
            camera_bp.set_attribute('fov', str(cam_config['fov']))

            # 创建Transform
            pos = cam_config['position']
            rot = cam_config['rotation']
            transform = carla.Transform(
                carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                carla.Rotation(pitch=rot[0], yaw=rot[2], roll=rot[1])  # CARLA: pitch, yaw, roll
            )

            # 生成相机
            camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
            self.cameras[cam_id] = camera

            # 创建数据队列
            self.data_queues[cam_id] = queue.Queue(maxsize=2)

            print(f"  ✓ {cam_id}: FOV={cam_config['fov']}° "
                  f"pos={pos} rot={rot}")

        print(f"[CameraManager] 已创建 {len(self.cameras)} 个相机")

    def start_listening(self):
        """开始监听所有相机"""
        for cam_id, camera in self.cameras.items():
            # 使用弱引用避免循环引用
            weak_self = weakref.ref(self)
            # 获取该相机的 raw_type 配置
            raw_type = 'uint8'
            for cfg in self.camera_configs:
                if cfg['id'] == cam_id:
                    raw_type = cfg.get('raw_type', 'uint8')
                    break
            
            camera.listen(lambda image, cid=cam_id, rt=raw_type: CameraManager._camera_callback(weak_self, cid, image, rt))

        print(f"[CameraManager] 已启动所有相机监听")

    @staticmethod
    def _camera_callback(weak_self, cam_id: str, image: carla.Image, raw_type: str):
        """
        相机回调函数
        Args:
            weak_self: 弱引用
            cam_id: 相机ID
            image: CARLA Image对象
            raw_type: 数据类型 ('bayer_rggb' or 'uint8')
        """
        self = weak_self()
        if self is None:
            return

        try:
            processed_data = None
            
            if raw_type == 'bayer_rggb':
                # 转换为 Bayer RGGB (uint16)
                processed_data = CameraManager.convert_to_bayer(image)
            else:
                # 默认: 转灰度 float16 (兼容旧逻辑)
                processed_data = CameraManager.convert_to_grayscale(image)

            # 放入队列
            if self.data_queues[cam_id].full():
                try:
                    self.data_queues[cam_id].get_nowait()  # 丢弃旧数据
                except queue.Empty:
                    pass

            self.data_queues[cam_id].put({
                'data': processed_data,
                'timestamp': image.timestamp,
                'frame': image.frame,
                'raw_type': raw_type
            })

        except Exception as e:
            print(f"[CameraManager] {cam_id} 回调错误: {e}")

    @staticmethod
    def convert_to_bayer(image: carla.Image) -> np.ndarray:
        """
        将CARLA RGB图像转换为单通道 Bayer RGGB (uint16)
        """
        # 解析BGRA数据
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = bgra.reshape((image.height, image.width, 4))
        
        # 创建 Bayer 容器
        bayer = np.zeros((image.height, image.width), dtype=np.uint8)
        
        # RGGB 采样:
        # R: (0,0), (0,2)... -> bgra[..., 2]
        # G: (0,1), (1,0)... -> bgra[..., 1]
        # B: (1,1), (1,3)... -> bgra[..., 0]
        
        # Row 0, 2, ... (Even rows)
        bayer[0::2, 0::2] = bgra[0::2, 0::2, 2] # R
        bayer[0::2, 1::2] = bgra[0::2, 1::2, 1] # G
        
        # Row 1, 3, ... (Odd rows)
        bayer[1::2, 0::2] = bgra[1::2, 0::2, 1] # G
        bayer[1::2, 1::2] = bgra[1::2, 1::2, 0] # B
        
        # 转为 uint16 (左移 8 位, 模拟 16-bit 传感器)
        return bayer.astype(np.uint16) << 8

    @staticmethod
    def convert_to_grayscale(image: carla.Image) -> np.ndarray:
        """
        将CARLA RGB图像转换为单通道灰度图像
        Returns:
            gray_image: (1, H, W) float16, 归一化到 [0, 1]
        """
        # 解析BGRA数据
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # (H, W, 4) BGRA

        # 提取RGB通道 (忽略Alpha)
        bgr = array[:, :, :3]  # (H, W, 3)

        # 转灰度: Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601标准)
        gray = (
            0.114 * bgr[:, :, 0].astype(np.float32) +  # B
            0.587 * bgr[:, :, 1].astype(np.float32) +  # G
            0.299 * bgr[:, :, 2].astype(np.float32)    # R
        )

        # 归一化到 [0, 1]
        gray = gray / 255.0

        # 添加通道维度: (H, W) → (1, H, W)
        gray = gray[np.newaxis, :, :]

        # 转换为float16
        return gray.astype(np.float16)

    def get_synced_frame(self, timeout: float = 2.0) -> Optional[Dict]:
        """
        获取同步的一帧数据 (8个相机)
        Returns:
            {
                'cam_front_main': {'data': array, 'timestamp': float, 'frame': int, 'raw_type': str},
                ...
            }
            如果超时返回None
        """
        import time
        start_time = time.time()
        synced_data = {}

        for cam_id in self.data_queues.keys():
            try:
                remaining = timeout - (time.time() - start_time)
                if remaining <= 0:
                    return None

                data = self.data_queues[cam_id].get(timeout=remaining)
                synced_data[cam_id] = data

            except queue.Empty:
                return None

        return synced_data

    def get_intrinsics(self, cam_id: str) -> np.ndarray:
        """
        获取相机内参矩阵
        Returns:
            K: (3, 3) 内参矩阵
        """
        # 查找配置
        cam_config = None
        for cfg in self.camera_configs:
            if cfg['id'] == cam_id:
                cam_config = cfg
                break
        
        if cam_config is None:
             raise ValueError(f"Unknown camera id: {cam_id}")

        width = CAMERA_SENSOR_CONFIG['image_size_x']
        height = CAMERA_SENSOR_CONFIG['image_size_y']
        fov = cam_config['fov']
        
        focal = width / (2.0 * np.tan(np.radians(fov) / 2.0))
        cx = width / 2.0
        cy = height / 2.0

        K = np.array([
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K

    def get_extrinsics(self, cam_id: str) -> np.ndarray:
        """
        获取相机外参矩阵 (相机相对车辆的变换)
        Returns:
            T: (4, 4) 外参矩阵 (车辆→相机)
        """
        camera = self.cameras[cam_id]
        transform = camera.get_transform()

        # 转换为4x4矩阵
        T = np.eye(4, dtype=np.float32)

        # 旋转矩阵
        pitch = np.radians(transform.rotation.pitch)
        yaw = np.radians(transform.rotation.yaw)
        roll = np.radians(transform.rotation.roll)

        # 旋转矩阵 (ZYX顺序)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])

        T[:3, :3] = R
        T[:3, 3] = [transform.location.x, transform.location.y, transform.location.z]

        return T

    def destroy(self):
        """销毁所有相机"""
        for camera in self.cameras.values():
            if camera.is_alive:
                camera.destroy()

        self.cameras.clear()
        self.data_queues.clear()
        print(f"[CameraManager] 已销毁所有相机")

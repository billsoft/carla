"""
Tesla 8相机管理器
支持 Bayer RGGB Raw 数据采集
"""
import carla
import numpy as np
import queue
import weakref
from typing import Dict, Optional

from config.camera_config import TESLA_CAMERAS, CAMERA_SENSOR_CONFIG, DEPTH_SENSOR_CONFIG


class CameraManager:
    """
    管理8个RGB相机 + 8个深度相机, 支持 Bayer RGGB 转换
    """

    def __init__(self, world: carla.World, vehicle: carla.Vehicle, enable_depth: bool = True):
        """
        Args:
            world: CARLA世界对象
            vehicle: 车辆actor
            enable_depth: 是否启用深度相机 (默认: True)
        """
        self.world = world
        self.vehicle = vehicle
        self.cameras = {}  # {cam_id: carla.Sensor}
        self.depth_cameras = {}  # {cam_id: carla.Sensor} 深度相机
        self.camera_configs = TESLA_CAMERAS
        self.data_queues = {}  # {cam_id: queue.Queue}
        self.depth_queues = {}  # {cam_id: queue.Queue} 深度数据队列
        self.enable_depth = enable_depth

        self._setup_cameras()
        if self.enable_depth:
            self._setup_depth_cameras()

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
                carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])  # CARLA: pitch, yaw, roll (修复: rotation数组顺序)
            )

            # 生成相机
            camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
            self.cameras[cam_id] = camera

            # 创建数据队列
            self.data_queues[cam_id] = queue.Queue(maxsize=2)

            print(f"  ✓ {cam_id}: FOV={cam_config['fov']}° "
                  f"pos={pos} rot={rot}")

        print(f"[CameraManager] 已创建 {len(self.cameras)} 个 RGB 相机")

    def _setup_depth_cameras(self):
        """创建并附加8个深度相机 (与 RGB 相机完全重合)"""
        bp_library = self.world.get_blueprint_library()
        depth_bp = bp_library.find('sensor.camera.depth')

        # 设置属性
        depth_bp.set_attribute('image_size_x', str(DEPTH_SENSOR_CONFIG['image_size_x']))
        depth_bp.set_attribute('image_size_y', str(DEPTH_SENSOR_CONFIG['image_size_y']))
        depth_bp.set_attribute('sensor_tick', str(DEPTH_SENSOR_CONFIG['sensor_tick']))

        for cam_config in self.camera_configs:
            cam_id = cam_config['id']

            # 设置 FOV (与 RGB 相机相同)
            depth_bp.set_attribute('fov', str(cam_config['fov']))

            # 创建 Transform (与 RGB 相机完全相同的位置和朝向)
            pos = cam_config['position']
            rot = cam_config['rotation']
            transform = carla.Transform(
                carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
            )

            # 生成深度相机
            depth_camera = self.world.spawn_actor(depth_bp, transform, attach_to=self.vehicle)
            self.depth_cameras[cam_id] = depth_camera

            # 创建深度数据队列
            self.depth_queues[cam_id] = queue.Queue(maxsize=2)

        print(f"[CameraManager] 已创建 {len(self.depth_cameras)} 个深度相机 (与 RGB 重合)")

    def start_listening(self):
        """开始监听所有相机 (RGB + 深度)"""
        # RGB 相机
        for cam_id, camera in self.cameras.items():
            weak_self = weakref.ref(self)
            raw_type = 'uint8'
            for cfg in self.camera_configs:
                if cfg['id'] == cam_id:
                    raw_type = cfg.get('raw_type', 'uint8')
                    break

            camera.listen(lambda image, cid=cam_id, rt=raw_type: CameraManager._camera_callback(weak_self, cid, image, rt))

        # 深度相机
        if self.enable_depth:
            for cam_id, depth_camera in self.depth_cameras.items():
                weak_self = weakref.ref(self)
                depth_camera.listen(lambda image, cid=cam_id: CameraManager._depth_callback(weak_self, cid, image))
            print(f"[CameraManager] 已启动所有相机监听 (RGB: {len(self.cameras)}, Depth: {len(self.depth_cameras)})")
        else:
            print(f"[CameraManager] 已启动 RGB 相机监听 ({len(self.cameras)} 个)")

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
    def _depth_callback(weak_self, cam_id: str, image: carla.Image):
        """
        深度相机回调函数
        """
        self = weak_self()
        if self is None:
            return

        try:
            # 转换深度数据
            depth_data = CameraManager.convert_depth(image)

            # 放入队列
            if self.depth_queues[cam_id].full():
                try:
                    self.depth_queues[cam_id].get_nowait()
                except queue.Empty:
                    pass

            self.depth_queues[cam_id].put({
                'data': depth_data,
                'timestamp': image.timestamp,
                'frame': image.frame,
            })

        except Exception as e:
            print(f"[CameraManager] Depth {cam_id} 回调错误: {e}")

    @staticmethod
    def convert_depth(image: carla.Image) -> np.ndarray:
        """
        将 CARLA 深度图像转换为实际深度值 (米)
        CARLA 深度格式: BGRA, 每像素 4 字节
        深度 = (R + G*256 + B*256*256) / (256^3 - 1) * 1000.0
        Returns:
            depth: (H, W) float32, 单位: 米
        """
        # 解析 BGRA 数据
        # CARLA UE5 raw_data 可能含4字节header，截取精确像素字节数
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        expected = image.height * image.width * 4
        array = raw[-expected:].reshape((image.height, image.width, 4))

        # CARLA 深度编码: 24-bit normalized depth in RGB channels
        # depth_normalized = (R + G*256 + B*256*256) / (256^3 - 1)
        # depth_meters = depth_normalized * 1000.0
        R = array[:, :, 2].astype(np.float32)
        G = array[:, :, 1].astype(np.float32)
        B = array[:, :, 0].astype(np.float32)

        depth_normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 ** 3 - 1)
        depth_meters = depth_normalized * 1000.0  # 转换为米

        return depth_meters.astype(np.float32)

    @staticmethod
    def convert_to_bayer(image: carla.Image) -> np.ndarray:
        """
        将CARLA RGB图像转换为单通道 Bayer RGGB (uint16)
        """
        # 解析BGRA数据
        # CARLA UE5 raw_data 可能含4字节header，截取精确像素字节数
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        expected = image.height * image.width * 4
        bgra = raw[-expected:].reshape((image.height, image.width, 4))
        
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
        # CARLA UE5 raw_data 可能含4字节header，截取精确像素字节数
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        expected = image.height * image.width * 4
        array = raw[-expected:].reshape((image.height, image.width, 4))  # (H, W, 4) BGRA

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
        获取同步的一帧数据 (8个相机) - 并行等待所有相机，避免顺序等待超时级联失败
        Returns:
            {cam_id: {'data': array, 'timestamp': float, 'frame': int, 'raw_type': str}}
            如果超时返回None
        """
        import time
        import concurrent.futures as cf

        cam_ids = list(self.data_queues.keys())

        def _fetch(cam_id):
            return cam_id, self.data_queues[cam_id].get(timeout=timeout)

        synced_data = {}
        with cf.ThreadPoolExecutor(max_workers=len(cam_ids)) as ex:
            futures = {ex.submit(_fetch, cid): cid for cid in cam_ids}
            deadline = time.time() + timeout
            for fut in cf.as_completed(futures, timeout=timeout):
                try:
                    cam_id, data = fut.result()
                    synced_data[cam_id] = data
                except Exception:
                    return None  # 任一相机失败立即返回

        if len(synced_data) != len(cam_ids):
            return None
        return synced_data

    def get_synced_depth_frame(self, timeout: float = 2.0) -> Optional[Dict]:
        """
        获取同步的一帧深度数据 (8个相机) - 并行等待
        如果超时或未启用深度返回 None
        """
        if not self.enable_depth:
            return None

        import concurrent.futures as cf

        cam_ids = list(self.depth_queues.keys())

        def _fetch(cam_id):
            return cam_id, self.depth_queues[cam_id].get(timeout=timeout)

        synced_data = {}
        with cf.ThreadPoolExecutor(max_workers=len(cam_ids)) as ex:
            futures = {ex.submit(_fetch, cid): cid for cid in cam_ids}
            for fut in cf.as_completed(futures, timeout=timeout):
                try:
                    cam_id, data = fut.result()
                    synced_data[cam_id] = data
                except Exception:
                    return None

        if len(synced_data) != len(cam_ids):
            return None
        return synced_data

    def clear_queues(self):
        """清空所有相机的队列 (RGB + 深度)"""
        for cam_id in self.data_queues:
            while not self.data_queues[cam_id].empty():
                try:
                    self.data_queues[cam_id].get_nowait()
                except queue.Empty:
                    break

        if self.enable_depth:
            for cam_id in self.depth_queues:
                while not self.depth_queues[cam_id].empty():
                    try:
                        self.depth_queues[cam_id].get_nowait()
                    except queue.Empty:
                        break

        print(f"[CameraManager] 已清空所有数据队列 (RGB + Depth)")

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
        获取相机安装外参矩阵（Camera→Vehicle，相机相对于车辆的固定安装位姿）。

        注意：此处从 camera_config.py 中的安装参数计算，而非运行时 get_transform()。
        原因：carla.Sensor.get_transform() 返回世界坐标系中的绝对变换，
        在车辆 spawn 初期可能不准确，且会随车辆移动而变化，不适合作为恒定标定参数。

        Returns:
            T: (4, 4) float32，Camera→Vehicle 变换矩阵（安装位姿，帧间恒定）
        """
        # 找到对应相机的安装配置
        cam_cfg = None
        for cfg in self.camera_configs:
            if cfg['id'] == cam_id:
                cam_cfg = cfg
                break
        if cam_cfg is None:
            raise ValueError(f"Unknown camera id: {cam_id}")

        # 安装位置（相机光心在车辆坐标系中的位置，单位：米）
        px, py, pz = cam_cfg['position']
        # 安装旋转（pitch, yaw, roll，单位：度）
        pitch_deg, yaw_deg, roll_deg = cam_cfg['rotation']

        pitch = np.radians(pitch_deg)
        yaw   = np.radians(yaw_deg)
        roll  = np.radians(roll_deg)

        cy, sy = np.cos(yaw),   np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll),  np.sin(roll)

        # ZYX 旋转矩阵（Vehicle→Camera 方向，对应 Camera→Vehicle 的转置）
        # 这里构建的是 Camera→Vehicle（即相机轴在车辆坐标系中的方向）
        R = np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [  -sp,            cp*sr,             cp*cr  ]
        ], dtype=np.float32)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3]  = [px, py, pz]

        return T

    def destroy(self):
        """销毁所有相机 (RGB + 深度)"""
        # 销毁 RGB 相机
        for camera in self.cameras.values():
            if camera.is_alive:
                camera.destroy()

        # 销毁深度相机
        for depth_camera in self.depth_cameras.values():
            if depth_camera.is_alive:
                depth_camera.destroy()

        self.cameras.clear()
        self.depth_cameras.clear()
        self.data_queues.clear()
        self.depth_queues.clear()
        print(f"[CameraManager] 已销毁所有相机 (RGB + Depth)")

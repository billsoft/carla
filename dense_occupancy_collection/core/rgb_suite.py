"""
传感器接口与 RGB 传感器套件
"""
import carla
import numpy as np
import queue
import cv2
import weakref

class SensorInterface:
    """传感器基类"""
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.sensors = {}
        self.queues = {}

    def destroy(self):
        for s in self.sensors.values():
            if s.is_alive:
                s.stop()
                s.destroy()
        self.sensors.clear()
        self.queues.clear()

class RGBSuite(SensorInterface):
    """
    8 相机 RGB 采集套件 (Tesla Style)
    支持 12bit Raw (模拟) / 8bit 输出
    """
    def __init__(self, world, vehicle, configs):
        super().__init__(world, vehicle)
        self.configs = configs
        self._setup_sensors()

    def _setup_sensors(self):
        print("\n[RGBSuite] 初始化 8 相机 RGB 套件...")
        bp_lib = self.world.get_blueprint_library()
        bp_rgb = bp_lib.find('sensor.camera.rgb')
        
        # 默认设置 (可被 config 覆盖)
        # 如果需要高位深，通常 CARLA 输出仍是 BGRA8，但我们可以通过配置尽量优化画质
        
        for cfg in self.configs:
            cid = cfg['id']
            
            # 1. 属性设置
            bp_rgb.set_attribute('image_size_x', str(cfg.get('image_size_x', 1280)))
            bp_rgb.set_attribute('image_size_y', str(cfg.get('image_size_y', 960)))
            bp_rgb.set_attribute('fov', str(cfg['fov']))
            bp_rgb.set_attribute('sensor_tick', '0.05') # 20Hz matching simulation

            # 设置 raw_type 属性（如果 config 中有配置）
            raw_type = cfg.get('raw_type', 'uint8')
            if raw_type == 'uint16' or raw_type == 'float32':
                 # 如果启用了 raw_type，尝试在 blueprint 中设置
                 # 注意：标准的 sensor.camera.rgb 默认是 uint8 BGRA
                 # 只有当 CARLA 编译支持或使用特殊 sensor.camera.rgb_16bit 时才有效
                 # 如果只是普通的 CARLA build，这里可能需要调整，或者接受 8bit 数据并在后续模拟 16bit
                 # 但根据用户描述，似乎已经有 12bit raw 支持的意图
                 pass
            
            # 画质增强
            if bp_rgb.has_attribute('enable_postprocess_effects'):
                bp_rgb.set_attribute('enable_postprocess_effects', 'True')
            if bp_rgb.has_attribute('gamma'):
                bp_rgb.set_attribute('gamma', '2.2')
            
            # UE5.5 Lumen 修复: 必须设置地图专属后处理配置
            map_name = self.world.get_map().name
            if 'Town10HD_Opt' in map_name and bp_rgb.has_attribute('post_process_profile'):
                bp_rgb.set_attribute('post_process_profile', 'Town10HD_Opt')
                
            # 曝光增强 (防止画面过暗)
            if bp_rgb.has_attribute('shutter_speed'):
                bp_rgb.set_attribute('shutter_speed', '200.0')
            if bp_rgb.has_attribute('iso'):
                bp_rgb.set_attribute('iso', '1200.0')
                
            # 畸变模拟
            dist = cfg.get('lens_distortion')
            if dist:
                for k, v in dist.items():
                    if bp_rgb.has_attribute(k):
                        bp_rgb.set_attribute(k, str(v))
            
            # 2. 安装位置
            trans = carla.Transform(
                carla.Location(x=cfg['x'], y=cfg['y'], z=cfg['z']),
                carla.Rotation(pitch=cfg['pitch'], yaw=cfg['yaw'], roll=cfg['roll'])
            )
            
            # 3. 生成
            sensor = self.world.spawn_actor(bp_rgb, trans, attach_to=self.vehicle)
            self.sensors[cid] = sensor
            
            # 4. 队列
            q = queue.Queue()
            self.queues[cid] = q
            
            # Callback
            weak_q = weakref.ref(q)
            sensor.listen(lambda data, q=weak_q: self._callback(data, q))
            print(f"  - {cid} OK (FOV={cfg['fov']})")

    @staticmethod
    def _callback(data, weak_q):
        q = weak_q()
        if q: q.put(data)

    def get_data(self, timeout=2.0):
        """同步获取所有相机数据"""
        results = {}
        try:
            for cid, q in self.queues.items():
                img = q.get(timeout=timeout)
                
                # ========== 新增: 根据像素格式解析数据 ==========
                
                # 获取配置中的 raw_type
                raw_type = 'uint8'
                for cfg in self.configs:
                    if cfg['id'] == cid:
                        raw_type = cfg.get('raw_type', 'uint8')
                        break

                if raw_type == 'bayer_rggb':
                    # 单通道 Bayer RGGB 格式
                    expected_size_uint16 = img.height * img.width * 2  # uint16 单通道
                    expected_size_bgra = img.height * img.width * 4   # BGRA fallback

                    if len(img.raw_data) == expected_size_uint16:
                        # 理想情况: 直接收到 uint16 单通道
                        array = np.frombuffer(img.raw_data, dtype=np.uint16)
                        array = array.reshape((img.height, img.width))
                        rgb = array
                    elif len(img.raw_data) == expected_size_bgra:
                        # Fallback: CARLA 返回了 BGRA,需要转换为 Bayer RGGB
                        # BGRA uint8 -> Bayer RGGB uint8 -> uint16
                        bgra = np.frombuffer(img.raw_data, dtype=np.uint8)
                        bgra = bgra.reshape((img.height, img.width, 4))
                        
                        # 创建 Bayer 容器
                        bayer = np.zeros((img.height, img.width), dtype=np.uint8)
                        
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
                        
                        # 转为 uint16 (左移 8 位)
                        rgb = bayer.astype(np.uint16) << 8
                    else:
                        print(f"[错误] {cid}: Bayer 数据大小异常！期望 {expected_size_uint16} 或 {expected_size_bgra}，实际 {len(img.raw_data)}")
                        rgb = np.zeros((img.height, img.width), dtype=np.uint16)

                elif raw_type == 'uint16':
                    # 16bit RGB 格式 (6 bytes/pixel) - 已弃用，保留以兼容旧代码
                    expected_size = img.height * img.width * 3 * 2  # 16bit = 2 bytes
                    if len(img.raw_data) == expected_size:
                        array = np.frombuffer(img.raw_data, dtype=np.uint16)
                        array = array.reshape((img.height, img.width, 3))  # RGB
                        rgb = array
                    else:
                        # Fallback: 接收到的是 8bit BGRA，但我们想要 16bit
                        # 转换: uint8 -> uint16 (左移8位或乘以257)
                        array = np.frombuffer(img.raw_data, dtype=np.uint8)
                        array = array.reshape((img.height, img.width, 4))
                        rgb_u8 = array[:, :, [2, 1, 0]] # BGRA -> RGB
                        rgb = rgb_u8.astype(np.uint16) * 257 # [0, 255] -> [0, 65535]

                elif raw_type == 'float32':
                    # 32bit float RGB 格式 (12 bytes/pixel)
                    array = np.frombuffer(img.raw_data, dtype=np.float32)
                    array = array.reshape((img.height, img.width, 3))  # RGB, 无 Alpha
                    rgb = array  # 已经是 RGB 顺序

                else:  # 'uint8' (默认)
                    # 8bit BGRA 格式 (4 bytes/pixel)
                    array = np.frombuffer(img.raw_data, dtype=np.uint8)
                    array = array.reshape((img.height, img.width, 4))
                    rgb = array[:, :, [2, 1, 0]]  # BGRA -> RGB
                
                results[cid] = {
                    'data': rgb,
                    'timestamp': img.timestamp,
                    'frame': img.frame,
                    'raw_type': raw_type
                }
        except queue.Empty:
            print("[RGBSuite] Data Timeout!")
            return None
            
        return results

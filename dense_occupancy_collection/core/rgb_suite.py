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
                
                # Convert to numpy
                array = np.frombuffer(img.raw_data, dtype=np.uint8)
                array = array.reshape((img.height, img.width, 4))
                # BGRA -> RGB
                rgb = array[:, :, [2, 1, 0]]
                
                results[cid] = {
                    'data': rgb,
                    'timestamp': img.timestamp,
                    'frame': img.frame
                }
        except queue.Empty:
            print("[RGBSuite] Data Timeout!")
            return None
            
        return results

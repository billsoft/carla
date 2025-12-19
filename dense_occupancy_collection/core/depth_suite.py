"""
深度相机全景套件 (Depth Panorama Suite)
6路 Cube Map 配置，用于生成全景深度和可见性过滤
"""
import carla
import numpy as np
import queue
import weakref

class DepthSuite:
    """
    6 相机深度全景套件
    配置: 水平360° (4x90), 垂直±30° (60° total)
    注意: Cube Map 实际上是 6x90°，覆盖全球面。
    若只需要垂直±30°，可以通过裁剪或调整 Up/Down 相机的 FOV/Pitch。
    但为了完整遮挡剔除，建议保持全球面覆盖 (90° FOV x 6)。
    
    用户需求: "垂直正负30共60fov"
    实现: 我们设置 FOV=60 对于 Surround Cameras (Front/Back/Left/Right)。
    Up/Down 相机用于补全天顶地底，确保遮挡剔除的完整性。
    """
    
    def __init__(self, world, vehicle, config):
        self.world = world
        self.vehicle = vehicle
        self.config = config
        self.sensors = []
        self.queues = {}
        self._setup()

    def _setup(self):
        print("\n[DepthSuite] 初始化 6 路深度全景套件...")
        bp_lib = self.world.get_blueprint_library()
        bp_depth = bp_lib.find('sensor.camera.depth')
        
        width = self.config['width']
        height = self.config['height']
        fov = self.config['fov'] # 60 or 90
        
        bp_depth.set_attribute('image_size_x', str(width))
        bp_depth.set_attribute('image_size_y', str(height))
        bp_depth.set_attribute('fov', str(fov))
        bp_depth.set_attribute('sensor_tick', '0.05')
        
        for cam_cfg in self.config['cameras']:
            cid = cam_cfg['id']
            pos = cam_cfg['pos']
            rot = cam_cfg['rot']
            
            trans = carla.Transform(
                carla.Location(x=pos['x'], y=pos['y'], z=pos['z']),
                carla.Rotation(pitch=rot['pitch'], yaw=rot['yaw'], roll=rot['roll'])
            )
            
            sensor = self.world.spawn_actor(bp_depth, trans, attach_to=self.vehicle)
            self.sensors.append(sensor)
            
            q = queue.Queue()
            self.queues[cid] = q
            
            weak_q = weakref.ref(q)
            sensor.listen(lambda data, q=weak_q: self._callback(data, q))
            print(f"  - {cid} OK")

    @staticmethod
    def _callback(data, weak_q):
        q = weak_q()
        if q: q.put(data)

    def get_data(self, timeout=2.0):
        """
        获取深度数据和变换矩阵
        返回:
            depth_maps: (6, H, W) float32 (Meters)
            cam_transforms: (6, 4, 4) Camera->World Matrices
        """
        depth_maps = []
        cam_transforms = []
        
        # 顺序必须固定，与 Config 列表一致
        cam_ids = [c['id'] for c in self.config['cameras']]
        
        try:
            for cid in cam_ids:
                img = self.queues[cid].get(timeout=timeout)
                
                # 1. 解码深度 (CARLA Logarithmic Depth -> Meters)
                # buffer is BGRA, uint8
                raw = np.frombuffer(img.raw_data, dtype=np.uint8)
                raw = raw.reshape((img.height, img.width, 4))
                
                # Formula: (R + G*256 + B*256*256) / (256^3 - 1) * 1000
                normalized = (raw[:,:,2].astype(np.float32) + 
                              raw[:,:,1].astype(np.float32) * 256.0 + 
                              raw[:,:,0].astype(np.float32) * 256.0 * 256.0) / (16777215.0)
                meters = normalized * 1000.0
                
                depth_maps.append(meters)
                cam_transforms.append(img.transform.get_matrix())
                
        except queue.Empty:
            print("[DepthSuite] Data Timeout!")
            return None
            
        return {
            'depth_maps': np.stack(depth_maps),     # (6, H, W)
            'cam_transforms': np.stack(cam_transforms) # (6, 4, 4)
        }

    def destroy(self):
        for s in self.sensors:
            if s.is_alive:
                s.stop()
                s.destroy()
        self.sensors.clear()
        self.queues.clear()

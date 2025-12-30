"""
相机参数计算工具
"""
import numpy as np
import carla

def build_projection_matrix(w, h, fov):
    """
    计算投影矩阵 (Intrinsics)
    
    Args:
        w: 图像宽度
        h: 图像高度
        fov: 视场角 (度)
        
    Returns:
        K: 3x3 内参矩阵
    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    """
    将 3D 点投影到 2D 图像平面
    
    Args:
        loc: 3D 坐标 (carla.Location)
        K: 内参矩阵
        w2c: 世界到相机变换矩阵 (4x4)
        
    Returns:
        point: [u, v, 1]
    """
    # 转换坐标系 (CARLA → 标准相机坐标系)
    # CARLA: X-前, Y-右, Z-上
    # Camera: X-右, Y-下, Z-前
    
    # 构建点向量 [x, y, z, 1]
    point = np.array([loc.x, loc.y, loc.z, 1])
    
    # 变换到相机坐标系
    point_camera = np.dot(w2c, point)
    
    # 标准相机坐标转换 [y, -z, x] -> [x, -z, y] ? 
    # CARLA 的 get_matrix() 返回的是变换矩阵，但坐标轴需要调整
    # 标准变换: [y, -z, x]
    
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    
    # 投影
    point_img = np.dot(K, point_camera)
    
    # 归一化
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    
    return point_img

def compute_camera_params(camera_configs, ego_transform, image_width=640, image_height=384):
    """
    计算所有相机的内参和外参

    Args:
        camera_configs: 相机配置列表
        ego_transform: 主车变换 (carla.Transform)
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        intrinsics: [N, 3, 3] 内参矩阵（暂时填充单位矩阵，保留字段）
        extrinsics: [N, 4, 4] 外参矩阵 (World -> Camera)
    """
    num_cams = len(camera_configs)
    intrinsics = np.zeros((num_cams, 3, 3), dtype=np.float32)
    extrinsics = np.zeros((num_cams, 4, 4), dtype=np.float32)

    # Ego 变换矩阵
    ego_matrix = np.array(ego_transform.get_matrix())

    for i, config in enumerate(camera_configs):
        # 1. 内参：暂时填充单位矩阵（CARLA 仿真数据无畸变，暂不使用）
        # 保留字段，以后可根据实际镜头畸变调整
        intrinsics[i] = np.eye(3, dtype=np.float32)
        
        # 2. 计算外参 (World -> Camera)
        # 相对变换: Sensor -> Ego
        # 注意: config 中的 x, y, z, pitch, yaw, roll 是相对于 Ego 的
        sensor_loc = carla.Location(x=config['x'], y=config['y'], z=config['z'])
        sensor_rot = carla.Rotation(pitch=config['pitch'], yaw=config['yaw'], roll=config['roll'])
        sensor_trans = carla.Transform(sensor_loc, sensor_rot)
        
        # Sensor -> World = Ego -> World * Sensor -> Ego
        sensor_matrix = np.dot(ego_matrix, np.array(sensor_trans.get_matrix()))
        
        # World -> Sensor (Extrinsics) = inv(Sensor -> World)
        world_to_sensor = np.linalg.inv(sensor_matrix)
        
        # 坐标系修正矩阵 (UE4 -> OpenCV/Standard Camera)
        # UE4: X-前, Y-右, Z-上
        # Cam: X-右, Y-下, Z-前
        # 变换: Xc=Yue, Yc=-Zue, Zc=Xue
        # R = [[0, 1, 0], [0, 0, -1], [1, 0, 0]]
        correction = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        
        # 最终外参: World -> Standard Camera
        extrinsics[i] = np.dot(correction, world_to_sensor)
        
    return intrinsics, extrinsics

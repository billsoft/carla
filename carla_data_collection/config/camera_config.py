"""
相机配置 - 特斯拉 FSD 硬件 3.0/4.0 布局
8 个相机: 3 前向 + 2 前侧 + 2 后侧 + 1 后向

使用 CARLA UE5.5 物理镜头模型:
- 超广角 (120°): 鱼眼畸变 (lens_k, lens_circle_*)
- 广角 (90°): 轻度桶形畸变
- 长焦 (50°): 无畸变
"""

import carla

# 特斯拉 8 相机配置
TESLA_CAMERA_CONFIGS = [
    {
        'id': 'cam_front_ultra_wide',
        'index': 0,
        'fov': 120,  # 超广角 - 鱼眼镜头
        'position': {'x': 1.5, 'y': 0.0, 'z': 1.4},   # 前挡风玻璃上沿
        'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0},
        'description': '前方超广角 - 近距离障碍物检测',
        # 鱼眼畸变参数 (模拟真实超广角镜头)
        'lens_distortion': {
            'lens_circle_multiplier': 2.5,  # 鱼眼效果强度
            'lens_circle_falloff': 2.0,     # 边缘衰减
            'lens_k': -0.15,                # 径向畸变 k1
            'lens_kcube': 0.05,             # 径向畸变 k3
            'lens_x_size': 0.95,            # X 方向有效范围
            'lens_y_size': 0.95,            # Y 方向有效范围
        }
    },
    {
        'id': 'cam_front_wide',
        'index': 1,
        'fov': 90,   # 广角
        'position': {'x': 1.5, 'y': 0.0, 'z': 1.4},
        'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0},
        'description': '前方广角 - 主视野/车道线',
        # 轻度桶形畸变 (模拟真实广角镜头)
        'lens_distortion': {
            'lens_circle_multiplier': 1.2,
            'lens_circle_falloff': 1.0,
            'lens_k': -0.05,
            'lens_kcube': 0.01,
            'lens_x_size': 1.0,
            'lens_y_size': 1.0,
        }
    },
    {
        'id': 'cam_front_narrow',
        'index': 2,
        'fov': 50,   # 长焦 - 无畸变
        'position': {'x': 1.5, 'y': 0.0, 'z': 1.4},
        'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0},
        'description': '前方长焦 - 远距离目标/交通标志',
        # 长焦镜头 - 无明显畸变
        'lens_distortion': None
    },
    {
        'id': 'cam_front_left',
        'index': 3,
        'fov': 90,
        'position': {'x': 1.2, 'y': -0.6, 'z': 1.2},  # 左前 A 柱附近
        'rotation': {'pitch': 0, 'yaw': -55, 'roll': 0},
        'description': '前左广角 - 左前盲区',
        # 广角畸变
        'lens_distortion': {
            'lens_circle_multiplier': 1.2,
            'lens_circle_falloff': 1.0,
            'lens_k': -0.05,
            'lens_kcube': 0.01,
            'lens_x_size': 1.0,
            'lens_y_size': 1.0,
        }
    },
    {
        'id': 'cam_front_right',
        'index': 4,
        'fov': 90,
        'position': {'x': 1.2, 'y': 0.6, 'z': 1.2},   # 右前 A 柱附近
        'rotation': {'pitch': 0, 'yaw': 55, 'roll': 0},
        'description': '前右广角 - 右前盲区',
        # 广角畸变
        'lens_distortion': {
            'lens_circle_multiplier': 1.2,
            'lens_circle_falloff': 1.0,
            'lens_k': -0.05,
            'lens_kcube': 0.01,
            'lens_x_size': 1.0,
            'lens_y_size': 1.0,
        }
    },
    {
        'id': 'cam_rear_left',
        'index': 5,
        'fov': 90,
        'position': {'x': -0.5, 'y': -0.8, 'z': 1.2},  # 左后视镜位置
        'rotation': {'pitch': 0, 'yaw': -110, 'roll': 0},
        'description': '左后广角 - 左后方/变道监控',
        # 广角畸变
        'lens_distortion': {
            'lens_circle_multiplier': 1.2,
            'lens_circle_falloff': 1.0,
            'lens_k': -0.05,
            'lens_kcube': 0.01,
            'lens_x_size': 1.0,
            'lens_y_size': 1.0,
        }
    },
    {
        'id': 'cam_rear_right',
        'index': 6,
        'fov': 90,
        'position': {'x': -0.5, 'y': 0.8, 'z': 1.2},   # 右后视镜位置
        'rotation': {'pitch': 0, 'yaw': 110, 'roll': 0},
        'description': '右后广角 - 右后方/变道监控',
        # 广角畸变
        'lens_distortion': {
            'lens_circle_multiplier': 1.2,
            'lens_circle_falloff': 1.0,
            'lens_k': -0.05,
            'lens_kcube': 0.01,
            'lens_x_size': 1.0,
            'lens_y_size': 1.0,
        }
    },
    {
        'id': 'cam_rear',
        'index': 7,
        'fov': 120,  # 超广角 - 鱼眼镜头
        'position': {'x': -1.8, 'y': 0.0, 'z': 1.0},   # 后备箱上沿
        'rotation': {'pitch': 0, 'yaw': 180, 'roll': 0},
        'description': '后方超广角 - 倒车/后方车辆',
        # 鱼眼畸变参数
        'lens_distortion': {
            'lens_circle_multiplier': 2.5,
            'lens_circle_falloff': 2.0,
            'lens_k': -0.15,
            'lens_kcube': 0.05,
            'lens_x_size': 0.95,
            'lens_y_size': 0.95,
        }
    }
]

# 为每个相机添加 Transform 对象
for cam_config in TESLA_CAMERA_CONFIGS:
    pos = cam_config['position']
    rot = cam_config['rotation']
    cam_config['transform'] = carla.Transform(
        carla.Location(x=pos['x'], y=pos['y'], z=pos['z']),
        carla.Rotation(pitch=rot['pitch'], yaw=rot['yaw'], roll=rot['roll'])
    )
    # 添加图像尺寸
    cam_config['width'] = 1280
    cam_config['height'] = 960
]

# 相机传感器参数
CAMERA_SENSOR_CONFIG = {
    'image_size_x': 1280,
    'image_size_y': 960,
    'sensor_tick': 1.0 / 36.0,  # 36 fps
    'gamma': 2.2,
    'motion_blur_intensity': 0.0,
    'enable_postprocess_effects': True,

    # 曝光设置 (模拟硬件相机)
    'exposure_mode': 'manual',
    'exposure_compensation': 0.0,
    'shutter_speed': 60.0,  # 1/60s
    'iso': 100.0
}

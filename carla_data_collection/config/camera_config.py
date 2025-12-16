"""
相机配置 - 特斯拉 FSD 硬件 3.0/4.0 布局
8 个相机: 3 前向 + 2 前侧 + 2 后侧 + 1 后向

使用 CARLA UE5.5 物理镜头模型:
- 超广角 (120°): 鱼眼畸变 (lens_k, lens_circle_*)
- 广角 (90°): 轻度桶形畸变
- 长焦 (50°): 无畸变
"""

# 镜头畸变参数 (基于 CARLA 文档)
# 鱼眼镜头参数 (适用于 120度+ 超广角)
FISHEYE_DISTORTION = {
    'lens_circle_multiplier': 3.0,  # 使用典型值 3.0
    'lens_circle_falloff': 3.0,     # 使用典型值 3.0
    'lens_k': -1.0,                 # 桶形畸变系数
    'lens_kcube': 0.0,
    'lens_x_size': 0.0,
    'lens_y_size': 0.0
}

# 广角镜头参数 (适用于 90度 广角) - 轻微畸变以减少拉伸
WIDE_ANGLE_DISTORTION = {
    'lens_circle_multiplier': 0.0,
    'lens_circle_falloff': 5.0,
    'lens_k': -0.2,                 # 非常轻微的畸变
    'lens_kcube': 0.0,
    'lens_x_size': 0.0,
    'lens_y_size': 0.0
}

# 特斯拉 8 相机配置 - 基于真实 Tesla Autopilot 传感器布局
# 参考: Tesla AI Day & 社区拆解数据
# 1280x960 (960p) 4:3 宽高比
TESLA_CAMERA_CONFIGS = [
    # --- 前视相机组 (Windshield Triple Cam) ---
    {
        'id': 'cam_front_main',  # Main
        'index': 0,
        'fov': 50,  # 主摄标准 50度
        'position': {'x': 1.0, 'y': 0.0, 'z': 1.6},
        'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0},
        'description': '前视主摄 (Main)',
        'lens_distortion': None
    },
    {
        'id': 'cam_front_wide',  # Wide
        'index': 1,
        'fov': 120, # 广角 120度
        'position': {'x': 1.0, 'y': 0.0, 'z': 1.6},
        'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0},
        'description': '前视广角 (Wide/Fisheye)',
        'lens_distortion': FISHEYE_DISTORTION
    },
    {
        'id': 'cam_front_narrow', # Narrow
        'index': 2,
        'fov': 35,  # 长焦 35度
        'position': {'x': 1.0, 'y': 0.0, 'z': 1.6},
        'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0},
        'description': '前视长焦 (Narrow)',
        'lens_distortion': None
    },
    
    # --- 侧向前视 (B-Pillar) ---
    # B柱位置，向前看，用于路口检测
    {
        'id': 'cam_left_pillar',
        'index': 3,
        'fov': 80,
        'position': {'x': 0.0, 'y': -0.9, 'z': 1.7}, # B柱高位
        'rotation': {'pitch': 0, 'yaw': -60, 'roll': 0}, # 指向左前
        'description': '左侧 B 柱 (Left Pillar)',
        'lens_distortion': None
    },
    {
        'id': 'cam_right_pillar',
        'index': 4,
        'fov': 80,
        'position': {'x': 0.0, 'y': 0.9, 'z': 1.7}, # B柱高位
        'rotation': {'pitch': 0, 'yaw': 60, 'roll': 0}, # 指向右前
        'description': '右侧 B 柱 (Right Pillar)',
        'lens_distortion': None
    },

    # --- 侧向后视 (Repeater/Fender) ---
    # 翼子板位置，向后看，用于盲区/变道
    {
        'id': 'cam_left_repeater',
        'index': 5,
        'fov': 100,
        'position': {'x': 1.2, 'y': -0.9, 'z': 1.0}, # 翼子板低位
        'rotation': {'pitch': 0, 'yaw': -160, 'roll': 0}, # 指向左后
        'description': '左侧翼子板 (Left Repeater)',
        'lens_distortion': None
    },
    {
        'id': 'cam_right_repeater',
        'index': 6,
        'fov': 100,
        'position': {'x': 1.2, 'y': 0.9, 'z': 1.0}, # 翼子板低位
        'rotation': {'pitch': 0, 'yaw': 160, 'roll': 0}, # 指向右后
        'description': '右侧翼子板 (Right Repeater)',
        'lens_distortion': None
    },

    # --- 后视 (Backup) ---
    {
        'id': 'cam_rear',
        'index': 7,
        'fov': 120,
        'position': {'x': -2.5, 'y': 0.0, 'z': 1.2}, # 车尾
        'rotation': {'pitch': -5, 'yaw': 180, 'roll': 0}, # 略微向下
        'description': '后视 (Rear)',
        'lens_distortion': FISHEYE_DISTORTION
    }
]

# 相机传感器参数
# 启用后处理以获得 Lumen/RayTracing 光照效果
CAMERA_SENSOR_CONFIG = {
    'image_size_x': 1280,
    'image_size_y': 960,
    'sensor_tick': 0.0, # 与仿真步长同步 (10Hz)
    
    # 启用后处理 (Lumen/RT 需要)
    'enable_postprocess_effects': True,
    'gamma': 2.2, # 标准 Gamma 2.2，确保光照亮度正常
    
    # 运动模糊 (0.0 = 关闭)
    'motion_blur_intensity': 0.0,
    
    # 曝光设置
    'exposure_mode': 'manual',
    'exposure_compensation': 0.0,
    'shutter_speed': 200.0, # 1/200s (运动捕捉更清晰)
    'iso': 100.0
}

"""
Tesla 8相机配置 - 用于OccNetV3
输出单通道灰度图像 (1, 960, 1280) float16
"""

# ========== Tesla 8相机布局 ==========
# 与occ_network/configs/default.py完全对齐

TESLA_CAMERAS = [
    # 前方3相机 (三焦段) - 对齐 dense_occupancy_collection
    {
        'id': 'front_main',
        'index': 0,
        'fov': 50.0,
        'position': [1.0, 0.0, 1.6],    # (x, y, z) 米 (对齐 dense_occupancy)
        'rotation': [0.0, 0.0, 0.0],    # (pitch, yaw, roll) 度
        'description': '前主相机 (Main, 50° 标准视野)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },
    {
        'id': 'front_wide',
        'index': 1,
        'fov': 120.0,
        'position': [1.0, 0.0, 1.6],    # 对齐 dense_occupancy
        'rotation': [0.0, 0.0, 0.0],
        'description': '前广角 (Wide, 120° 广角)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },
    {
        'id': 'front_narrow',
        'index': 2,
        'fov': 35.0,
        'position': [1.0, 0.0, 1.6],    # 对齐 dense_occupancy
        'rotation': [0.0, 0.0, 0.0],
        'description': '前窄角/长焦 (Narrow, 35° 长焦)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },

    # 侧向前视 (B柱) - 参考 dense_occupancy_collection
    {
        'id': 'left_pillar',
        'index': 3,
        'fov': 80.0,
        'position': [0.0, -0.9, 1.7],   # B柱左侧 (x=0, z=1.7 对齐 dense_occupancy)
        'rotation': [0.0, -60.0, 0.0],  # yaw=-60° 左前方 (对齐 dense_occupancy)
        'description': '左B柱 (Left Pillar, 左前方 -60°)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },
    {
        'id': 'right_pillar',
        'index': 4,
        'fov': 80.0,
        'position': [0.0, 0.9, 1.7],    # B柱右侧 (x=0, z=1.7 对齐 dense_occupancy)
        'rotation': [0.0, 60.0, 0.0],   # yaw=60° 右前方 (对齐 dense_occupancy)
        'description': '右B柱 (Right Pillar, 右前方 60°)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },

    # 侧向后视 (翼子板/Repeater) - 参考 dense_occupancy_collection
    {
        'id': 'left_repeater',
        'index': 5,
        'fov': 100.0,                   # FOV=100 (对齐 dense_occupancy)
        'position': [1.2, -0.9, 1.0],   # x=1.2, z=1.0 (对齐 dense_occupancy)
        'rotation': [0.0, -160.0, 0.0], # yaw=-160° 左后方 (对齐 dense_occupancy)
        'description': '左后视镜 (Left Repeater, 左后方 -160°)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },
    {
        'id': 'right_repeater',
        'index': 6,
        'fov': 100.0,                   # FOV=100 (对齐 dense_occupancy)
        'position': [1.2, 0.9, 1.0],    # x=1.2, z=1.0 (对齐 dense_occupancy)
        'rotation': [0.0, 160.0, 0.0],  # yaw=160° 右后方 (对齐 dense_occupancy)
        'description': '右后视镜 (Right Repeater, 右后方 160°)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },

    # 后视 - 参考 dense_occupancy_collection
    {
        'id': 'rear',
        'index': 7,
        'fov': 120.0,                   # FOV=120 (对齐 dense_occupancy, 广角)
        'position': [-2.5, 0.0, 1.2],   # x=-2.5 (对齐 dense_occupancy)
        'rotation': [-5.0, 180.0, 0.0], # pitch=-5°, yaw=180° (对齐 dense_occupancy)
        'description': '后视 (Rear, 正后方 180°, 俯仰 -5°)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },
]

# 确认相机数量
assert len(TESLA_CAMERAS) == 8, f"相机数量错误: {len(TESLA_CAMERAS)}"

# ========== 相机传感器参数 ==========
CAMERA_SENSOR_CONFIG = {
    'image_size_x': 1280,
    'image_size_y': 960,
    'sensor_tick': 0.0,  # 与仿真同步

    # 使用RGB相机 (后续转灰度)
    'enable_postprocess_effects': True,
    'gamma': 2.2,
    'motion_blur_intensity': 0.0,

    # 曝光设置 (使用直方图自动曝光以匹配 dense_occupancy_collection)
    'exposure_mode': 'histogram',
    'exposure_compensation': 0.0,
    'shutter_speed': 1/200.0,  # 仅在 manual 模式下有效
    'iso': 100.0,              # 仅在 manual 模式下有效
}

# 输出规范 (对齐网络输入)
OUTPUT_SPEC = {
    'channels': 1,          # 单通道灰度
    'height': 960,
    'width': 1280,
    'dtype': 'float16',     # 16位浮点
    'range': [0.0, 1.0],    # 归一化范围
    'format': 'CHW',        # (C, H, W)
}

print(f"[CameraConfig] 相机数量: {len(TESLA_CAMERAS)}")
print(f"[CameraConfig] 输出规格: ({OUTPUT_SPEC['channels']}, {OUTPUT_SPEC['height']}, {OUTPUT_SPEC['width']}) {OUTPUT_SPEC['dtype']}")

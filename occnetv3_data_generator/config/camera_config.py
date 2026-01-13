"""
Tesla 8相机配置 - 用于OccNetV3
输出单通道灰度图像 (1, 960, 1280) float16
"""

# ========== Tesla 8相机布局 ==========
# 与occ_network/configs/default.py完全对齐

TESLA_CAMERAS = [
    # 前方3相机 (三焦段)
    {
        'id': 'front_main',
        'index': 0,
        'fov': 50.0,
        'position': [1.5, 0.0, 1.5],    # (x, y, z) 米
        'rotation': [0.0, 0.0, 0.0],    # (pitch, yaw, roll) 度
        'description': '前主相机 (Main, 50° 标准视野)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },
    {
        'id': 'front_wide',
        'index': 1,
        'fov': 120.0,
        'position': [1.5, 0.0, 1.5],
        'rotation': [0.0, 0.0, 0.0],
        'description': '前广角 (Wide, 120° 广角)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },
    {
        'id': 'front_narrow',
        'index': 2,
        'fov': 35.0,
        'position': [1.5, 0.0, 1.5],
        'rotation': [0.0, 0.0, 0.0],
        'description': '前窄角/长焦 (Narrow, 35° 长焦)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },

    # 侧向前视 (B柱) - ⚠️ 修复: 交换左右侧位置 (Y坐标取反)
    {
        'id': 'left_pillar',
        'index': 3,
        'fov': 80.0,
        'position': [0.5, -0.9, 1.3],   # B柱左侧 (驾驶员左手侧)
        'rotation': [0.0, 0.0, 55.0],   # 指向左前方
        'description': '左B柱 (Left Pillar, 55° 朝向)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },
    {
        'id': 'right_pillar',
        'index': 4,
        'fov': 80.0,
        'position': [0.5, 0.9, 1.3],    # B柱右侧 (驾驶员右手侧)
        'rotation': [0.0, 0.0, -55.0],  # 指向右前方
        'description': '右B柱 (Right Pillar, -55° 朝向)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },

    # 侧向后视 (翼子板/Repeater) - ⚠️ 修复: 交换左右侧位置 + 向外延伸避免玻璃遮挡
    {
        'id': 'left_repeater',
        'index': 5,
        'fov': 80.0,
        'position': [1.0, -1.2, 0.8],   # 左翼子板 (向外延伸到 -1.2)
        'rotation': [0.0, 0.0, 135.0],  # 指向左后方
        'description': '左后视镜 (Left Repeater, 135° 朝向)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },
    {
        'id': 'right_repeater',
        'index': 6,
        'fov': 80.0,
        'position': [1.0, 1.2, 0.8],    # 右翼子板 (向外延伸到 1.2)
        'rotation': [0.0, 0.0, -135.0], # 指向右后方
        'description': '右后视镜 (Right Repeater, -135° 朝向)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },

    # 后视 - ⚠️ 修复: 向后移动避免车内遮挡
    {
        'id': 'rear',
        'index': 7,
        'fov': 80.0,
        'position': [-2.0, 0.0, 1.2],   # 车尾 (向后移动到 -2.0)
        'rotation': [0.0, 0.0, 180.0],  # 朝向后方
        'description': '后视 (Rear, 180° 朝向)',
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

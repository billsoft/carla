"""
Tesla 8相机配置 - 用于OccNetV3
输出单通道灰度图像 (1, 960, 1280) float16

相机模型: 等距投影(fisheye)，对应 CARLA sensor.camera.rgb_fisheye + camera_model=equidistant
(见 sensors/camera_manager.py)。等距投影的 fov 属性是垂直 FOV (YFOVAngle)，而这里每个相机
的 'fov' 字段沿用的是历史上的水平 FOV 语义，'fov_vertical' 才是实际传给传感器/用于计算
intrinsics 的值。两者换算关系(各向同性等距投影下精确成立): fov_vertical = fov * (image_size_y / image_size_x)
= fov * 0.75 (960/1280)。
"""

# ========== Tesla 8相机布局 ==========
# 与occ_network/configs/default.py完全对齐

TESLA_CAMERAS = [
    # 前方3相机 (三焦段) - 对齐 dense_occupancy_collection
    {
        'id': 'front_main',
        'index': 0,
        'fov': 50.0,
        'fov_vertical': 37.5,
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
        'fov_vertical': 90.0,
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
        'fov_vertical': 26.25,
        'position': [1.0, 0.0, 1.6],    # 对齐 dense_occupancy
        'rotation': [0.0, 0.0, 0.0],
        'description': '前窄角/长焦 (Narrow, 35° 长焦)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },

    # 侧向前视 (B柱) - ⭐ 修复: 增大Y轴偏移量,避免拍摄车内
    # 特斯拉实际安装: B柱靠近车窗外侧,朝向侧前方约60°
    {
        'id': 'left_pillar',
        'index': 3,
        'fov': 80.0,
        'fov_vertical': 60.0,
        'position': [0.0, -1.1, 1.7],   # ⭐ Y=-1.1 (增加向左偏移,避免车架遮挡)
        'rotation': [0.0, -55.0, 0.0],  # ⭐ yaw=-55° (轻微调整朝向,更贴近侧前方)
        'description': '左B柱 (Left Pillar, 左前方 -55°)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },
    {
        'id': 'right_pillar',
        'index': 4,
        'fov': 80.0,
        'fov_vertical': 60.0,
        'position': [0.0, 1.1, 1.7],    # ⭐ Y=1.1 (增加向右偏移,避免车架遮挡)
        'rotation': [0.0, 55.0, 0.0],   # ⭐ yaw=55° (轻微调整朝向,更贴近侧前方)
        'description': '右B柱 (Right Pillar, 右前方 55°)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },

    # 侧向后视 (翼子板/Repeater) - ⭐ 修复: 调整朝向角度避免拍摄车内
    # 特斯拉实际安装: 翼子板前轮上方,朝向侧后方约45°
    {
        'id': 'left_repeater',
        'index': 5,
        'fov': 100.0,
        'fov_vertical': 75.0,
        'position': [1.0, -1.0, 1.0],   # ⭐ X=1.0 (靠近前轮), Y=-1.0 (翼子板外侧)
        'rotation': [0.0, -130.0, 0.0], # ⭐ yaw=-130° (侧后方45°, 避免拍摄车身)
        'description': '左Repeater (Left Repeater, 左后方 -130°)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },
    {
        'id': 'right_repeater',
        'index': 6,
        'fov': 100.0,
        'fov_vertical': 75.0,
        'position': [1.0, 1.0, 1.0],    # ⭐ X=1.0 (靠近前轮), Y=1.0 (翼子板外侧)
        'rotation': [0.0, 130.0, 0.0],  # ⭐ yaw=130° (侧后方45°, 避免拍摄车身)
        'description': '右Repeater (Right Repeater, 右后方 130°)',
        'raw_type': 'bayer_rggb',
        'bit_depth': 12,
    },

    # 后视 - ⭐ 修复: 向车尾外延15-20cm,避免穿模遮挡
    # 特斯拉实际安装: 车牌上方,轻微俯仰向下,视距50米
    {
        'id': 'rear',
        'index': 7,
        'fov': 120.0,
        'fov_vertical': 90.0,
        'position': [-2.7, 0.0, 1.2],   # ⭐ X=-2.7 (向车尾外延20cm, 原-2.5)
        'rotation': [-8.0, 180.0, 0.0], # ⭐ pitch=-8° (增加俯仰角,看清地面)
        'description': '后视 (Rear, 正后方 180°, 俯仰 -8°)',
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

# ========== 深度相机配置 ==========
# 与每个 RGB 相机完全重合（相同位置、相同 FOV）
# 用于监督训练

DEPTH_SENSOR_CONFIG = {
    'image_size_x': 1280,
    'image_size_y': 960,
    'sensor_tick': 0.0,  # 与仿真同步
}

# 深度输出规范
DEPTH_OUTPUT_SPEC = {
    'channels': 1,          # 单通道深度
    'height': 960,
    'width': 1280,
    'dtype': 'float32',     # 32位浮点 (米)
    'range': [0.0, 1000.0], # 深度范围 (米)
    'format': 'HW',         # (H, W)
}

print(f"[CameraConfig] 相机数量: {len(TESLA_CAMERAS)}")
print(f"[CameraConfig] 输出规格: ({OUTPUT_SPEC['channels']}, {OUTPUT_SPEC['height']}, {OUTPUT_SPEC['width']}) {OUTPUT_SPEC['dtype']}")
print(f"[CameraConfig] 深度相机: 8 个 (与 RGB 重合)")

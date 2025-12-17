
"""
全景相机配置
用于生成 360° 全景深度图和语义分割图
"""

import numpy as np

# 全景图分辨率 (2:1)
PANO_WIDTH = 1024
PANO_HEIGHT = 512

# 立方体贴图分辨率 (每个面)
CUBE_SIZE = 512

# 立方体相机的 FOV
CUBE_FOV = 90.0

# 6个面的配置 (相对于车辆中心的旋转)
# CARLA Rotation: (pitch, yaw, roll)
# 顺序对应 Cubemap 的: Front, Right, Back, Left, Up, Down
CUBE_FACE_CONFIGS = [
    {'name': 'front', 'rot': (0, 0, 0),     'face_idx': 0},
    {'name': 'right', 'rot': (0, 90, 0),    'face_idx': 1},
    {'name': 'back',  'rot': (0, 180, 0),   'face_idx': 2},
    {'name': 'left',  'rot': (0, -90, 0),   'face_idx': 3},
    {'name': 'up',    'rot': (90, 0, 0),    'face_idx': 4},
    {'name': 'down',  'rot': (-90, 0, 0),   'face_idx': 5},
]

# 相机安装位置 (相对于车辆中心)
# 为了"透视"车体，建议安装在车顶上方，并结合语义过滤
PANO_LOCATION = {'x': 0.0, 'y': 0.0, 'z': 2.0}

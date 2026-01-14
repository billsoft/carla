# OccNetV3 数据集格式规范

> 📦 完整的训练数据准备指南，对齐网络输入输出

---

## 目录

1. [数据集总览](#一数据集总览)
2. [目录结构](#二目录结构)
3. [输入数据：8相机图像](#三输入数据8相机图像)
4. [输出标签：3D占用网格](#四输出标签3d占用网格)
5. [流场标签](#五流场标签)
6. [时序数据：Ego Motion](#六时序数据ego-motion)
7. [相机标定文件](#七相机标定文件)
8. [数据划分文件](#八数据划分文件)
9. [完整示例](#九完整示例)
10. [数据验证脚本](#十数据验证脚本)
11. [从CARLA生成数据](#十一从carla生成数据)

---

## 一、数据集总览

### 1.1 数据流对应关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OccNetV3 数据流                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  数据集文件                          网络张量                            │
│  ═══════════                        ════════                            │
│                                                                         │
│  images/                                                                │
│    ├── cam_0.npy  ─┐                                                    │
│    ├── cam_1.npy   │                                                    │
│    ├── cam_2.npy   │                                                    │
│    ├── cam_3.npy   ├──→  images: [B, 8, 1, 960, 1280]  (float16)       │
│    ├── cam_4.npy   │                                                    │
│    ├── cam_5.npy   │                                                    │
│    ├── cam_6.npy   │                                                    │
│    └── cam_7.npy  ─┘                                                    │
│                                                                         │
│  occupancy.npy  ────────→  semantic: [B, 400, 400, 32]  (int64)        │
│                                                                         │
│  flow.npy  ─────────────→  flow: [B, 3, 400, 400, 32]  (float16)       │
│                                                                         │
│  flow_mask.npy  ────────→  flow_mask: [B, 400, 400, 32]  (bool)        │
│                                                                         │
│  ego_motion.npy  ───────→  ego_motion: [B, 4, 4]  (float32)            │
│                                                                         │
│  ego_pose.npy  ─────────→  ego_pose: [B, 4, 4]  (float32)              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 坐标系定义

```
                        Z (高度, 向上)
                        │
                        │
                        │
                        │
                        └───────────── Y (左)
                       ╱
                      ╱
                     ╱
                    X (前)

车辆坐标系 (右手系):
- X轴: 车头方向 (正前方)
- Y轴: 车左侧 (正左方)  
- Z轴: 向上

原点: 车辆后轴中心，地面高度
```

### 1.3 体素空间定义

| 参数 | 值 | 说明 |
|------|-----|------|
| X范围 | [-40.0m, 40.0m] | 前后各40米 |
| Y范围 | [-40.0m, 40.0m] | 左右各40米 |
| Z范围 | [-1.0m, 5.4m] | 地下1米到地上5.4米 |
| 体素分辨率 | 0.2m | 每个体素20cm边长 |
| 体素网格 | 400 × 400 × 32 | 约512万个体素 |

**体素索引到世界坐标的转换**：

```python
def voxel_to_world(voxel_idx, pc_range, resolution=0.2):
    """
    voxel_idx: [i, j, k] 体素索引
    pc_range: [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    """
    x = voxel_idx[0] * resolution + pc_range[0]  # X世界坐标
    y = voxel_idx[1] * resolution + pc_range[1]  # Y世界坐标
    z = voxel_idx[2] * resolution + pc_range[2]  # Z世界坐标
    return [x, y, z]

# 示例：体素 [200, 200, 5] 对应世界坐标
# x = 200 * 0.2 + (-40.0) = 0.0  (车辆正中央)
# y = 200 * 0.2 + (-40.0) = 0.0  (车辆正中央)
# z = 5 * 0.2 + (-1.0) = 0.0     (地面高度)
```

### 1.4 图像格式支持

OccNetV3 数据加载器支持两种图像格式:

**1. DNG (推荐) - 12-bit Bayer RGGB 原始格式**
- 直接从 CARLA 相机读取,无损质量
- 格式: 单通道 12-bit (0-4095)
- 尺寸: 1280×960 (W×H)
- 加载时自动归一化到 [0, 1]
- **依赖**: `pip install rawpy` (推荐) 或 OpenCV
- **路径**: `dataset/cameras/cam_<camera_id>/<frame_id>.dng`

**2. NPY - float16 预处理格式**
- 已归一化的单通道图像
- 形状: (1, 960, 1280) - CHW格式
- 数据类型: float16 或 float32
- 值范围: [0, 1]
- **路径**: `dataset/images/<frame_id>/cam_<camera_id>.npy`

**数据加载器会自动识别格式**，优先尝试 NPY，如果不存在则尝试 DNG。无需手动转换。

**示例: 安装 DNG 加载依赖**
```bash
# 在 deepsys 或 carla 环境中
pip install rawpy
```

---

## 二、目录结构

```
dataset/
├── train.txt                    # 训练集样本列表
├── val.txt                      # 验证集样本列表
├── test.txt                     # 测试集样本列表
│
├── calibration/                 # 相机标定文件
│   ├── intrinsics.json          # 内参
│   └── extrinsics.json          # 外参
│
├── images/                      # 输入图像
│   ├── scene_0001_frame_0000/
│   │   ├── cam_0.npy            # 前主相机
│   │   ├── cam_1.npy            # 前广角
│   │   ├── cam_2.npy            # 前窄角
│   │   ├── cam_3.npy            # 左B柱
│   │   ├── cam_4.npy            # 右B柱
│   │   ├── cam_5.npy            # 左后
│   │   ├── cam_6.npy            # 右后
│   │   └── cam_7.npy            # 后
│   ├── scene_0001_frame_0001/
│   │   └── ...
│   └── ...
│
├── occupancy/                   # 3D占用语义标签
│   ├── scene_0001_frame_0000.npy
│   ├── scene_0001_frame_0001.npy
│   └── ...
│
├── flow/                        # 3D流场标签
│   ├── scene_0001_frame_0000.npy
│   ├── scene_0001_frame_0001.npy
│   └── ...
│
├── flow_mask/                   # 流场有效掩码
│   ├── scene_0001_frame_0000.npy
│   └── ...
│
├── ego_motion/                  # 帧间运动 (t-1 → t)
│   ├── scene_0001_frame_0000.npy
│   └── ...
│
└── ego_pose/                    # 全局位姿
    ├── scene_0001_frame_0000.npy
    └── ...
```

---

## 三、输入数据：8相机图像

### 3.1 相机配置

| ID | 名称 | FOV | 位置 (x,y,z) | 朝向 (yaw) | 说明 |
|----|------|-----|--------------|-----------|------|
| 0 | front_main | 50° | (1.5, 0, 1.5) | 0° | 前主相机，标准视野 |
| 1 | front_wide | 120° | (1.5, 0, 1.5) | 0° | 前广角，近距离广覆盖 |
| 2 | front_narrow | 35° | (1.5, 0, 1.5) | 0° | 前窄角/长焦，远距离 |
| 3 | left_pillar | 80° | (0.5, 0.9, 1.3) | 55° | 左B柱，覆盖左前方 |
| 4 | right_pillar | 80° | (0.5, -0.9, 1.3) | -55° | 右B柱，覆盖右前方 |
| 5 | left_repeater | 80° | (1.0, 1.0, 0.8) | 135° | 左后视镜，覆盖左后方 |
| 6 | right_repeater | 80° | (1.0, -1.0, 0.8) | -135° | 右后视镜，覆盖右后方 |
| 7 | rear | 80° | (-1.5, 0, 1.2) | 180° | 后相机 |

### 3.2 图像格式规范

```python
# 每张图像的规格
image_spec = {
    'shape': (1, 960, 1280),      # (通道, 高度, 宽度)
    'dtype': np.float16,          # 或 np.float32
    'range': [0.0, 1.0],          # 归一化范围
    'format': 'CHW',              # 通道优先
}

# 8张图像堆叠后
images_spec = {
    'shape': (8, 1, 960, 1280),   # (相机数, 通道, 高度, 宽度)
    'dtype': np.float16,
}
```

### 3.3 图像保存格式

**方式A：单独保存每个相机（推荐）**

```python
import numpy as np

def save_camera_images(sample_id, images_dict, output_dir):
    """
    images_dict: {
        'cam_0': np.array of shape (1, 960, 1280),
        'cam_1': np.array of shape (1, 960, 1280),
        ...
    }
    """
    sample_dir = os.path.join(output_dir, 'images', sample_id)
    os.makedirs(sample_dir, exist_ok=True)
    
    for cam_id in range(8):
        img = images_dict[f'cam_{cam_id}']
        
        # 确保格式正确
        assert img.shape == (1, 960, 1280), f"Expected (1, 960, 1280), got {img.shape}"
        assert img.dtype in [np.float16, np.float32]
        
        # 保存为压缩的npy
        np.save(os.path.join(sample_dir, f'cam_{cam_id}.npy'), img.astype(np.float16))
```

**方式B：合并保存（节省IO）**

```python
def save_all_cameras(sample_id, images, output_dir):
    """
    images: np.array of shape (8, 1, 960, 1280)
    """
    np.save(os.path.join(output_dir, 'images', f'{sample_id}.npy'), images.astype(np.float16))
```

### 3.4 RAW图像预处理

如果你的相机输出RAW Bayer格式：

```python
def preprocess_raw_bayer(raw_image, pattern='RGGB'):
    """
    raw_image: (H, W) uint16 RAW Bayer
    返回: (1, H, W) float16 归一化灰度
    """
    # 方式1: 简单平均 (快速)
    gray = raw_image.astype(np.float32)
    gray = gray / 65535.0  # 假设16位RAW
    gray = gray[np.newaxis, :, :]  # 添加通道维度
    return gray.astype(np.float16)
    
    # 方式2: Bayer解码后转灰度 (更准确)
    import cv2
    rgb = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_RG2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0
    return gray[np.newaxis, :, :].astype(np.float16)
```

### 3.5 从常见格式转换

```python
def convert_image_to_network_format(image_path, target_size=(960, 1280)):
    """
    支持 PNG, JPG, EXR 等常见格式
    """
    import cv2
    
    # 读取图像
    if image_path.endswith('.exr'):
        img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    else:
        img = cv2.imread(image_path)
    
    # 转灰度 (如果是RGB)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 调整尺寸
    if img.shape != target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    
    # 归一化
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.max() > 1.0:
        img = img / img.max()
    
    # 添加通道维度: (H, W) → (1, H, W)
    img = img[np.newaxis, :, :]
    
    return img.astype(np.float16)
```

---

## 四、输出标签：3D占用网格

### 4.1 语义类别定义

```python
CLASS_NAMES = {
    0: 'empty',                 # 空气/无物体
    1: 'barrier',               # 护栏/路障
    2: 'bicycle',               # 自行车
    3: 'bus',                   # 公交车
    4: 'car',                   # 小汽车
    5: 'construction_vehicle',  # 工程车辆
    6: 'motorcycle',            # 摩托车
    7: 'pedestrian',            # 行人
    8: 'traffic_cone',          # 交通锥
    9: 'trailer',               # 拖车/挂车
    10: 'truck',                # 卡车
    11: 'driveable_surface',    # 可行驶路面
    12: 'other_flat',           # 其他平坦表面
    13: 'sidewalk',             # 人行道
    14: 'terrain',              # 地形(草地等)
    15: 'manmade',              # 人造建筑
    16: 'vegetation',           # 植被
    17: 'free',                 # 自由空间(可通行但非路面)
}

# 特殊值
IGNORE_LABEL = 255  # 忽略标签 (不计入loss)
```

### 4.2 占用网格格式

```python
occupancy_spec = {
    'shape': (400, 400, 32),     # (X, Y, Z)
    'dtype': np.int64,           # 或 np.uint8 (如果类别<256)
    'value_range': [0, 17],      # 18个类别
    'special_values': {
        255: 'ignore'            # 忽略标签
    }
}
```

### 4.3 保存占用标签

```python
def save_occupancy(sample_id, occupancy, output_dir):
    """
    occupancy: np.array of shape (400, 400, 32), dtype=int64
    """
    assert occupancy.shape == (400, 400, 32)
    assert occupancy.dtype in [np.int64, np.uint8, np.int32]
    
    # 检查值范围
    unique_values = np.unique(occupancy)
    valid_values = set(range(18)) | {255}  # 0-17 + ignore
    assert all(v in valid_values for v in unique_values), f"Invalid labels: {unique_values}"
    
    # 保存
    save_path = os.path.join(output_dir, 'occupancy', f'{sample_id}.npy')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, occupancy.astype(np.uint8))  # uint8节省空间
```

### 4.4 从3D点云生成占用网格

```python
def pointcloud_to_occupancy(points, labels, pc_range, voxel_size=0.2):
    """
    points: (N, 3) 点云坐标 [x, y, z]
    labels: (N,) 每个点的语义标签
    pc_range: [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    
    返回: (400, 400, 32) 占用网格
    """
    # 计算网格尺寸
    grid_size = [
        int((pc_range[3] - pc_range[0]) / voxel_size),  # 400
        int((pc_range[4] - pc_range[1]) / voxel_size),  # 400
        int((pc_range[5] - pc_range[2]) / voxel_size),  # 32
    ]
    
    # 初始化为空
    occupancy = np.zeros(grid_size, dtype=np.uint8)
    
    # 过滤范围外的点
    mask = (
        (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
        (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) &
        (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
    )
    points = points[mask]
    labels = labels[mask]
    
    # 计算体素索引
    voxel_idx = ((points - np.array(pc_range[:3])) / voxel_size).astype(np.int32)
    
    # 裁剪到有效范围
    voxel_idx = np.clip(voxel_idx, 0, np.array(grid_size) - 1)
    
    # 填充占用网格
    occupancy[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]] = labels
    
    return occupancy
```

### 4.5 从3D Bounding Box生成占用网格

```python
def bbox3d_to_occupancy(bboxes, labels, pc_range, voxel_size=0.2):
    """
    bboxes: list of dict, each dict contains:
        - center: [x, y, z]
        - size: [length, width, height]
        - rotation: yaw angle (radians)
    labels: list of int, semantic label for each bbox
    
    返回: (400, 400, 32) 占用网格
    """
    grid_size = [400, 400, 32]
    occupancy = np.zeros(grid_size, dtype=np.uint8)
    
    for bbox, label in zip(bboxes, labels):
        cx, cy, cz = bbox['center']
        l, w, h = bbox['size']
        yaw = bbox['rotation']
        
        # 生成长方体内的点
        # 简化版: 轴对齐，忽略旋转
        x_min = max(0, int((cx - l/2 - pc_range[0]) / voxel_size))
        x_max = min(511, int((cx + l/2 - pc_range[0]) / voxel_size))
        y_min = max(0, int((cy - w/2 - pc_range[1]) / voxel_size))
        y_max = min(511, int((cy + w/2 - pc_range[1]) / voxel_size))
        z_min = max(0, int((cz - h/2 - pc_range[2]) / voxel_size))
        z_max = min(39, int((cz + h/2 - pc_range[2]) / voxel_size))
        
        occupancy[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = label
    
    # 填充地面
    occupancy[:, :, :2] = 11  # driveable_surface (假设地面在z=0~0.4m)
    
    return occupancy
```

---

## 五、流场标签

### 5.1 流场定义

流场描述每个体素在下一帧的运动方向和速度（单位：米/帧）。

```python
flow_spec = {
    'shape': (3, 400, 400, 32),   # (方向, X, Y, Z)
    'dtype': np.float16,
    'channels': {
        0: 'flow_x',              # X方向速度 (m/frame)
        1: 'flow_y',              # Y方向速度 (m/frame)
        2: 'flow_z',              # Z方向速度 (m/frame)
    },
    'typical_range': [-5.0, 5.0], # 典型速度范围 (m/frame)
}
```

### 5.2 流场掩码

流场只对**动态物体**有意义，静态物体的流场应该被忽略。

```python
flow_mask_spec = {
    'shape': (400, 400, 32),
    'dtype': np.bool_,            # 或 np.uint8
    'meaning': {
        True: '有效流场 (动态物体)',
        False: '无效流场 (静态物体/空气)',
    }
}

# 动态物体类别
DYNAMIC_CLASSES = {2, 3, 4, 5, 6, 7, 9, 10}  # bicycle, bus, car, 等
```

### 5.3 从物体轨迹生成流场

```python
def generate_flow_from_tracking(
    occupancy_t0,      # 当前帧占用 (400, 400, 32)
    occupancy_t1,      # 下一帧占用 (400, 400, 32)
    tracking_info,     # 物体追踪信息
    pc_range,
    voxel_size=0.2
):
    """
    tracking_info: list of dict
        - object_id: int
        - class_id: int
        - center_t0: [x, y, z]
        - center_t1: [x, y, z]
    """
    flow = np.zeros((3, 400, 400, 32), dtype=np.float32)
    flow_mask = np.zeros((400, 400, 32), dtype=np.bool_)
    
    for obj in tracking_info:
        if obj['class_id'] not in DYNAMIC_CLASSES:
            continue
        
        # 计算速度向量
        velocity = np.array(obj['center_t1']) - np.array(obj['center_t0'])
        
        # 找到该物体占据的体素
        obj_mask = (occupancy_t0 == obj['class_id'])  # 简化：按类别
        
        # 填充流场
        flow[0][obj_mask] = velocity[0]
        flow[1][obj_mask] = velocity[1]
        flow[2][obj_mask] = velocity[2]
        flow_mask[obj_mask] = True
    
    return flow.astype(np.float16), flow_mask
```

### 5.4 保存流场数据

```python
def save_flow(sample_id, flow, flow_mask, output_dir):
    """
    flow: (3, 400, 400, 32) float16
    flow_mask: (400, 400, 32) bool
    """
    # 保存流场
    flow_path = os.path.join(output_dir, 'flow', f'{sample_id}.npy')
    os.makedirs(os.path.dirname(flow_path), exist_ok=True)
    np.save(flow_path, flow.astype(np.float16))
    
    # 保存掩码
    mask_path = os.path.join(output_dir, 'flow_mask', f'{sample_id}.npy')
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    np.save(mask_path, flow_mask.astype(np.uint8))
```

---

## 六、时序数据：Ego Motion

### 6.1 Ego Motion vs Ego Pose

| 概念 | 定义 | 形状 | 用途 |
|------|------|------|------|
| **Ego Pose** | 车辆在世界坐标系的绝对位姿 | (4, 4) | 多帧对齐 |
| **Ego Motion** | 从上一帧到当前帧的相对变换 | (4, 4) | 时序融合 |

```
世界坐标系
    │
    ├── t=0帧: Ego Pose P0
    │
    ├── t=1帧: Ego Pose P1
    │
    └── Ego Motion M = P1 @ inv(P0)  # 从t0到t1的变换
```

### 6.2 4x4变换矩阵格式

```python
# 变换矩阵结构
T = np.array([
    [r11, r12, r13, tx],    # 第1行: 旋转 + X平移
    [r21, r22, r23, ty],    # 第2行: 旋转 + Y平移
    [r31, r32, r33, tz],    # 第3行: 旋转 + Z平移
    [  0,   0,   0,  1],    # 第4行: 齐次坐标
], dtype=np.float32)

# 其中:
# R = [[r11, r12, r13],    3x3旋转矩阵 (正交矩阵)
#      [r21, r22, r23],
#      [r31, r32, r33]]
#
# t = [tx, ty, tz]         平移向量 (单位: 米)
```

### 6.3 从位姿计算Ego Motion

```python
def compute_ego_motion(pose_t0, pose_t1):
    """
    计算从t0到t1的相对变换
    
    pose_t0: (4, 4) 上一帧的世界位姿
    pose_t1: (4, 4) 当前帧的世界位姿
    
    返回: (4, 4) ego_motion，满足: P_world = ego_motion @ P_t0
    """
    ego_motion = pose_t1 @ np.linalg.inv(pose_t0)
    return ego_motion.astype(np.float32)
```

### 6.4 从欧拉角/四元数构建位姿

```python
import numpy as np
from scipy.spatial.transform import Rotation

def euler_to_pose(x, y, z, roll, pitch, yaw):
    """
    从位置和欧拉角构建4x4位姿矩阵
    
    x, y, z: 位置 (米)
    roll, pitch, yaw: 欧拉角 (弧度)
    """
    R = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = [x, y, z]
    
    return pose

def quaternion_to_pose(x, y, z, qx, qy, qz, qw):
    """
    从位置和四元数构建4x4位姿矩阵
    """
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = [x, y, z]
    
    return pose
```

### 6.5 保存时序数据

```python
def save_ego_data(sample_id, ego_pose, ego_motion, output_dir):
    """
    ego_pose: (4, 4) 当前帧世界位姿
    ego_motion: (4, 4) 从上一帧到当前帧的变换
    """
    # 第一帧的ego_motion设为单位阵
    if ego_motion is None:
        ego_motion = np.eye(4, dtype=np.float32)
    
    pose_path = os.path.join(output_dir, 'ego_pose', f'{sample_id}.npy')
    motion_path = os.path.join(output_dir, 'ego_motion', f'{sample_id}.npy')
    
    os.makedirs(os.path.dirname(pose_path), exist_ok=True)
    os.makedirs(os.path.dirname(motion_path), exist_ok=True)
    
    np.save(pose_path, ego_pose.astype(np.float32))
    np.save(motion_path, ego_motion.astype(np.float32))
```

---

## 七、相机标定文件

### 7.1 内参文件 (intrinsics.json)

```json
{
    "cam_0": {
        "fx": 1142.5,
        "fy": 1142.5,
        "cx": 640.0,
        "cy": 480.0,
        "width": 1280,
        "height": 960,
        "fov": 50.0,
        "distortion": {
            "model": "pinhole",
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0
        }
    },
    "cam_1": {
        "fx": 426.7,
        "fy": 426.7,
        "cx": 640.0,
        "cy": 480.0,
        "width": 1280,
        "height": 960,
        "fov": 120.0,
        "distortion": {
            "model": "fisheye",
            "k1": -0.1,
            "k2": 0.05,
            "k3": 0.0,
            "k4": 0.0
        }
    },
    "...": "其他相机"
}
```

**内参计算公式**：

```python
def fov_to_focal(fov_degrees, image_width):
    """
    从FOV计算焦距
    """
    fov_rad = np.radians(fov_degrees)
    fx = image_width / (2 * np.tan(fov_rad / 2))
    return fx
```

### 7.2 外参文件 (extrinsics.json)

```json
{
    "cam_0": {
        "translation": [1.5, 0.0, 1.5],
        "rotation": {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0
        },
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    },
    "cam_3": {
        "translation": [0.5, 0.9, 1.3],
        "rotation": {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 55.0
        },
        "rotation_matrix": [
            [0.574, -0.819, 0.0],
            [0.819, 0.574, 0.0],
            [0.0, 0.0, 1.0]
        ]
    },
    "...": "其他相机"
}
```

---

## 八、数据划分文件

### 8.1 格式

```
# train.txt
scene_0001_frame_0000
scene_0001_frame_0001
scene_0001_frame_0002
...
scene_0100_frame_0099

# val.txt
scene_0101_frame_0000
scene_0101_frame_0001
...

# test.txt
scene_0201_frame_0000
...
```

### 8.2 划分建议

```python
def split_dataset(all_samples, train_ratio=0.8, val_ratio=0.1):
    """
    按场景划分，避免同一场景的帧出现在不同集合
    """
    # 按场景分组
    scenes = {}
    for sample in all_samples:
        scene_id = sample.split('_frame_')[0]
        if scene_id not in scenes:
            scenes[scene_id] = []
        scenes[scene_id].append(sample)
    
    # 随机打乱场景
    scene_list = list(scenes.keys())
    np.random.shuffle(scene_list)
    
    # 划分
    n_train = int(len(scene_list) * train_ratio)
    n_val = int(len(scene_list) * val_ratio)
    
    train_scenes = scene_list[:n_train]
    val_scenes = scene_list[n_train:n_train+n_val]
    test_scenes = scene_list[n_train+n_val:]
    
    # 展开为样本列表
    train_samples = [s for sc in train_scenes for s in scenes[sc]]
    val_samples = [s for sc in val_scenes for s in scenes[sc]]
    test_samples = [s for sc in test_scenes for s in scenes[sc]]
    
    return train_samples, val_samples, test_samples
```

---

## 九、完整示例

### 9.1 生成单个样本

```python
import numpy as np
import os
import json

def create_sample(
    sample_id,
    output_dir,
    camera_images,      # dict: {cam_0: (1,960,1280), ...}
    occupancy,          # (400, 400, 32) int
    flow,               # (3, 400, 400, 32) float
    flow_mask,          # (400, 400, 32) bool
    ego_pose,           # (4, 4) float
    ego_motion,         # (4, 4) float
):
    """
    创建完整的训练样本
    """
    # 1. 保存图像
    img_dir = os.path.join(output_dir, 'images', sample_id)
    os.makedirs(img_dir, exist_ok=True)
    for cam_id in range(8):
        img = camera_images[f'cam_{cam_id}']
        assert img.shape == (1, 960, 1280), f"cam_{cam_id} shape error: {img.shape}"
        np.save(os.path.join(img_dir, f'cam_{cam_id}.npy'), img.astype(np.float16))
    
    # 2. 保存占用标签
    occ_dir = os.path.join(output_dir, 'occupancy')
    os.makedirs(occ_dir, exist_ok=True)
    assert occupancy.shape == (400, 400, 32)
    np.save(os.path.join(occ_dir, f'{sample_id}.npy'), occupancy.astype(np.uint8))
    
    # 3. 保存流场
    flow_dir = os.path.join(output_dir, 'flow')
    os.makedirs(flow_dir, exist_ok=True)
    assert flow.shape == (3, 400, 400, 32)
    np.save(os.path.join(flow_dir, f'{sample_id}.npy'), flow.astype(np.float16))
    
    # 4. 保存流场掩码
    mask_dir = os.path.join(output_dir, 'flow_mask')
    os.makedirs(mask_dir, exist_ok=True)
    assert flow_mask.shape == (400, 400, 32)
    np.save(os.path.join(mask_dir, f'{sample_id}.npy'), flow_mask.astype(np.uint8))
    
    # 5. 保存时序数据
    pose_dir = os.path.join(output_dir, 'ego_pose')
    motion_dir = os.path.join(output_dir, 'ego_motion')
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(motion_dir, exist_ok=True)
    np.save(os.path.join(pose_dir, f'{sample_id}.npy'), ego_pose.astype(np.float32))
    np.save(os.path.join(motion_dir, f'{sample_id}.npy'), ego_motion.astype(np.float32))
    
    print(f"✓ Sample {sample_id} saved")
```

### 9.2 合成数据示例

```python
def generate_synthetic_sample(sample_id, output_dir):
    """
    生成一个合成样本用于测试
    """
    np.random.seed(hash(sample_id) % (2**32))
    
    # 1. 生成随机图像
    camera_images = {}
    for cam_id in range(8):
        img = np.random.randn(1, 960, 1280).astype(np.float16) * 0.1 + 0.5
        img = np.clip(img, 0, 1)
        camera_images[f'cam_{cam_id}'] = img
    
    # 2. 生成占用标签
    occupancy = np.zeros((400, 400, 32), dtype=np.uint8)
    
    # 地面 (z = 0~2，对应体素索引 10~12)
    occupancy[:, :, 10:12] = 11  # driveable_surface
    
    # 添加一些车辆
    for _ in range(5):
        cx = np.random.randint(100, 400)
        cy = np.random.randint(100, 400)
        occupancy[cx-10:cx+10, cy-5:cy+5, 12:18] = 4  # car
    
    # 添加行人
    for _ in range(3):
        px = np.random.randint(50, 450)
        py = np.random.randint(50, 450)
        occupancy[px-1:px+1, py-1:py+1, 12:20] = 7  # pedestrian
    
    # 3. 生成流场
    flow = np.zeros((3, 400, 400, 32), dtype=np.float16)
    flow_mask = np.zeros((400, 400, 32), dtype=np.uint8)
    
    # 给车辆添加流场
    car_mask = (occupancy == 4)
    flow[0][car_mask] = np.random.randn(car_mask.sum()).astype(np.float16) * 0.5
    flow[1][car_mask] = np.random.randn(car_mask.sum()).astype(np.float16) * 0.3
    flow_mask[car_mask] = 1
    
    # 4. 生成时序数据
    ego_pose = np.eye(4, dtype=np.float32)
    ego_pose[0, 3] = np.random.randn() * 0.5  # 随机x位移
    ego_pose[1, 3] = np.random.randn() * 0.1  # 随机y位移
    
    ego_motion = np.eye(4, dtype=np.float32)
    ego_motion[0, 3] = 0.5  # 假设每帧前进0.5米
    
    # 保存
    create_sample(
        sample_id=sample_id,
        output_dir=output_dir,
        camera_images=camera_images,
        occupancy=occupancy,
        flow=flow,
        flow_mask=flow_mask,
        ego_pose=ego_pose,
        ego_motion=ego_motion,
    )

# 生成100个样本
output_dir = './dataset'
for i in range(100):
    scene_id = i // 10
    frame_id = i % 10
    sample_id = f'scene_{scene_id:04d}_frame_{frame_id:04d}'
    generate_synthetic_sample(sample_id, output_dir)
```

---

## 十、数据验证脚本

```python
import numpy as np
import os
import json

def validate_dataset(dataset_dir):
    """
    验证数据集格式是否正确
    """
    errors = []
    warnings = []
    
    # 1. 检查目录结构
    required_dirs = ['images', 'occupancy', 'calibration']
    optional_dirs = ['flow', 'flow_mask', 'ego_pose', 'ego_motion']
    
    for d in required_dirs:
        if not os.path.exists(os.path.join(dataset_dir, d)):
            errors.append(f"缺少必需目录: {d}")
    
    for d in optional_dirs:
        if not os.path.exists(os.path.join(dataset_dir, d)):
            warnings.append(f"缺少可选目录: {d}")
    
    # 2. 检查标定文件
    calib_dir = os.path.join(dataset_dir, 'calibration')
    if os.path.exists(calib_dir):
        if not os.path.exists(os.path.join(calib_dir, 'intrinsics.json')):
            warnings.append("缺少 intrinsics.json")
        if not os.path.exists(os.path.join(calib_dir, 'extrinsics.json')):
            warnings.append("缺少 extrinsics.json")
    
    # 3. 检查样本文件
    train_file = os.path.join(dataset_dir, 'train.txt')
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            samples = [line.strip() for line in f if line.strip()]
        
        # 抽样检查
        check_samples = samples[:10] if len(samples) > 10 else samples
        
        for sample_id in check_samples:
            # 检查图像
            img_dir = os.path.join(dataset_dir, 'images', sample_id)
            if os.path.exists(img_dir):
                for cam_id in range(8):
                    img_path = os.path.join(img_dir, f'cam_{cam_id}.npy')
                    if os.path.exists(img_path):
                        img = np.load(img_path)
                        if img.shape != (1, 960, 1280):
                            errors.append(f"{sample_id}/cam_{cam_id}: 形状错误 {img.shape}")
                        if img.dtype not in [np.float16, np.float32]:
                            warnings.append(f"{sample_id}/cam_{cam_id}: dtype={img.dtype}")
                    else:
                        errors.append(f"缺少图像: {img_path}")
            else:
                errors.append(f"缺少图像目录: {img_dir}")
            
            # 检查占用标签
            occ_path = os.path.join(dataset_dir, 'occupancy', f'{sample_id}.npy')
            if os.path.exists(occ_path):
                occ = np.load(occ_path)
                if occ.shape != (400, 400, 32):
                    errors.append(f"{sample_id} occupancy: 形状错误 {occ.shape}")
                unique = np.unique(occ)
                invalid = [v for v in unique if v not in range(18) and v != 255]
                if invalid:
                    errors.append(f"{sample_id} occupancy: 无效标签 {invalid}")
            else:
                errors.append(f"缺少占用标签: {occ_path}")
            
            # 检查流场
            flow_path = os.path.join(dataset_dir, 'flow', f'{sample_id}.npy')
            if os.path.exists(flow_path):
                flow = np.load(flow_path)
                if flow.shape != (3, 400, 400, 32):
                    errors.append(f"{sample_id} flow: 形状错误 {flow.shape}")
    else:
        warnings.append("缺少 train.txt")
    
    # 打印结果
    print("=" * 50)
    print("数据集验证报告")
    print("=" * 50)
    
    if errors:
        print(f"\n❌ 错误 ({len(errors)}):")
        for e in errors[:20]:  # 最多显示20条
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... 还有 {len(errors)-20} 条错误")
    
    if warnings:
        print(f"\n⚠️ 警告 ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"  - {w}")
    
    if not errors and not warnings:
        print("\n✅ 数据集格式正确!")
    
    return len(errors) == 0

# 使用
validate_dataset('./dataset')
```

---

## 十一、从CARLA生成数据

### 11.1 CARLA相机设置

```python
import carla
import numpy as np

def setup_cameras(world, vehicle):
    """
    在CARLA中设置8个相机
    """
    camera_configs = [
        # (name, fov, x, y, z, pitch, yaw, roll)
        ('front_main', 50, 1.5, 0, 1.5, 0, 0, 0),
        ('front_wide', 120, 1.5, 0, 1.5, 0, 0, 0),
        ('front_narrow', 35, 1.5, 0, 1.5, 0, 0, 0),
        ('left_pillar', 80, 0.5, 0.9, 1.3, 0, 55, 0),
        ('right_pillar', 80, 0.5, -0.9, 1.3, 0, -55, 0),
        ('left_repeater', 80, 1.0, 1.0, 0.8, 0, 135, 0),
        ('right_repeater', 80, 1.0, -1.0, 0.8, 0, -135, 0),
        ('rear', 80, -1.5, 0, 1.2, 0, 180, 0),
    ]
    
    cameras = {}
    blueprint_lib = world.get_blueprint_library()
    camera_bp = blueprint_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '960')
    
    for name, fov, x, y, z, pitch, yaw, roll in camera_configs:
        camera_bp.set_attribute('fov', str(fov))
        
        transform = carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
        )
        
        camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        cameras[name] = camera
    
    return cameras
```

### 11.2 获取占用标签

```python
def get_occupancy_from_carla(world, ego_location, pc_range, voxel_size=0.2):
    """
    从CARLA获取3D占用标签
    """
    grid_size = (400, 400, 32)
    occupancy = np.zeros(grid_size, dtype=np.uint8)
    
    # CARLA类别到我们类别的映射
    carla_to_ours = {
        'vehicle.car': 4,
        'vehicle.bus': 3,
        'vehicle.truck': 10,
        'vehicle.motorcycle': 6,
        'vehicle.bicycle': 2,
        'walker.pedestrian': 7,
        'static.prop.trafficcone': 8,
    }
    
    # 获取所有Actor
    for actor in world.get_actors():
        type_id = actor.type_id
        
        # 匹配类别
        label = 0
        for carla_type, our_label in carla_to_ours.items():
            if carla_type in type_id:
                label = our_label
                break
        
        if label == 0:
            continue
        
        # 获取Bounding Box
        bbox = actor.bounding_box
        transform = actor.get_transform()
        
        # 转换到世界坐标
        # ... (需要实现bbox填充逻辑)
    
    # 填充地面
    occupancy[:, :, 10:12] = 11  # driveable_surface
    
    return occupancy
```

---

## 总结

| 数据类型 | 文件格式 | 形状 | dtype | 必需 |
|---------|---------|------|-------|-----|
| 相机图像 | npy | (1, 960, 1280) × 8 | float16 | ✅ |
| 占用标签 | npy | (400, 400, 32) | uint8 | ✅ |
| 流场 | npy | (3, 400, 400, 32) | float16 | ⚪ |
| 流场掩码 | npy | (400, 400, 32) | uint8 | ⚪ |
| Ego Pose | npy | (4, 4) | float32 | ⚪ |
| Ego Motion | npy | (4, 4) | float32 | ⚪ |
| 内参 | json | - | - | ✅ |
| 外参 | json | - | - | ✅ |

**关键要点**：
1. 图像必须是 **(1, 960, 1280)** 单通道，归一化到 **[0, 1]**
2. 占用标签必须是 **(400, 400, 32)**，值范围 **[0, 17]** + **255**(忽略)
3. 体素坐标系遵循 **右手系**，原点在车辆后轴中心
4. 时序数据需要按场景连续，支持帧间对齐

有问题欢迎反馈！

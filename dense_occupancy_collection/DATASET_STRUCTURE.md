# 数据集结构说明

本数据集同时支持两种网络训练：
- **occ_network_nano**: 使用单通道 Bayer RAW（12-bit DNG）
- **occ_network_lite**: 使用彩色 RGB PNG（8-bit）

---

## 目录结构

```
dataset_output/
├── cameras/                          # Bayer RAW 数据（用于 occ_network_nano）
│   ├── cam_front_main/
│   │   ├── 000000.dng               # 单通道 12-bit Bayer RGGB (960×1280)
│   │   ├── 000001.dng
│   │   └── ...
│   ├── cam_front_wide/
│   ├── cam_front_narrow/
│   ├── cam_left_pillar/
│   ├── cam_right_pillar/
│   ├── cam_left_repeater/
│   ├── cam_right_repeater/
│   └── cam_rear/
│
├── cameras_rgb/                      # RGB 预览图（用于 occ_network_lite）
│   ├── cam_front_main/
│   │   ├── 000000.png               # 8-bit RGB 彩色图 (960×1280×3)
│   │   ├── 000001.png
│   │   └── ...
│   ├── cam_front_wide/
│   ├── cam_front_narrow/
│   ├── cam_left_pillar/
│   ├── cam_right_pillar/
│   ├── cam_left_repeater/
│   ├── cam_right_repeater/
│   └── cam_rear/
│
├── depth/                            # 深度图像（两种网络都需要）
│   ├── depth_front/
│   │   ├── 000000.png               # 16-bit PNG (单位: mm)
│   │   └── ...
│   ├── depth_right/
│   ├── depth_back/
│   ├── depth_left/
│   ├── depth_up/
│   └── depth_down/
│
├── occupancy/                        # 体素占据（两种网络都需要）
│   ├── 000000.npz                   # [200, 200, 16] uint8
│   │   ├── occupancy                # 占据类别
│   │   ├── actor_ids                # Actor ID
│   │   ├── mask                     # 可见性掩码
│   │   └── metadata                 # 元数据
│   └── ...
│
└── camera_params/                    # 相机参数（两种网络都需要）
    ├── 000000.npz                   # 相机内外参
    │   ├── intrinsics               # [8, 3, 3] 内参矩阵
    │   ├── extrinsics               # [8, 4, 4] 外参矩阵
    │   └── configs                  # JSON 配置
    └── ...
```

---

## 数据格式详细说明

### 1. Bayer RAW 数据（cameras/）

**用途**: occ_network_nano 训练

**格式**: 单通道 12-bit DNG/TIFF
- **文件格式**: `.dng` (本质是 16-bit TIFF，但数据范围 [0, 4095])
- **形状**: (960, 1280) 单通道
- **数据类型**: uint16
- **数值范围**: [0, 4095] (12-bit)
- **文件大小**: 约 2.46 MB/帧
- **Bayer 模式**: RGGB
  ```
  R G R G R G ...
  G B G B G B ...
  R G R G R G ...
  G B G B G B ...
  ```

**读取示例**:
```python
import cv2
import numpy as np

# 读取 12-bit DNG
img = cv2.imread('000000.dng', cv2.IMREAD_UNCHANGED)
# 形状: (960, 1280), 类型: uint16, 范围: [0, 4095]

# 扩展到 16-bit 完整范围（用于训练）
img_16bit = (img << 4).astype(np.uint16)  # [0, 65535]

# 归一化到 [0, 1]
img_norm = img_16bit.astype(np.float32) / 65535.0
```

**DNG 元数据**:
- Make: "CARLA Simulator"
- Model: "Bayer RGGB Camera"
- PhotometricInterpretation: 32803 (CFA - Color Filter Array)
- BitsPerSample: 12

---

### 2. RGB 预览图（cameras_rgb/）

**用途**: occ_network_lite 训练

**格式**: 8-bit PNG
- **文件格式**: `.png`
- **形状**: (960, 1280, 3) BGR 顺序
- **数据类型**: uint8
- **数值范围**: [0, 255]
- **文件大小**: 约 2-3 MB/帧（PNG 压缩）
- **生成方式**: 从 Bayer RAW 去马赛克得到

**读取示例**:
```python
import cv2

# 读取 RGB 图像
img_bgr = cv2.imread('000000.png')  # BGR 顺序
img_rgb = img_bgr[:, :, ::-1]       # 转换为 RGB

# 归一化到 [0, 1]
img_norm = img_rgb.astype(np.float32) / 255.0
```

---

### 3. 深度图（depth/）

**用途**: 两种网络都需要（用于可见性过滤）

**格式**: 16-bit PNG
- **文件格式**: `.png`
- **形状**: (H, W) 单通道
- **数据类型**: uint16
- **单位**: 毫米 (mm)
- **数值范围**: [0, 65535] (最大约 65.5 米)
- **方向**: 6 个方向（front, right, back, left, up, down）

**读取示例**:
```python
import cv2

# 读取深度图（单位: mm）
depth_mm = cv2.imread('depth_front/000000.png', cv2.IMREAD_UNCHANGED)

# 转换为米
depth_m = depth_mm.astype(np.float32) / 1000.0
```

---

### 4. 体素占据（occupancy/）

**用途**: 两种网络都需要（训练标签）

**格式**: NPZ 压缩格式
- **文件格式**: `.npz`
- **occupancy**: [200, 200, 16] uint8 - 占据类别
  - 0: 空白
  - 1-18: 语义类别
- **actor_ids**: [200, 200, 16] int32 - Actor ID（调试用）
- **mask**: [200, 200, 16] bool - 可见性掩码
- **metadata**: 元数据字典
  - town: 地图名称
  - x_range, y_range, z_range: 空间范围
  - resolution: 分辨率
  - grid_size: 网格尺寸

**读取示例**:
```python
import numpy as np

# 加载体素数据
data = np.load('000000.npz')
occupancy = data['occupancy']  # [200, 200, 16]
mask = data['mask']            # [200, 200, 16]
```

---

### 5. 相机参数（camera_params/）

**用途**: 两种网络都需要

**格式**: NPZ 压缩格式
- **intrinsics**: [8, 3, 3] float32 - 相机内参矩阵
- **extrinsics**: [8, 4, 4] float32 - 相机外参矩阵（世界 → 相机）
- **configs**: JSON 字符串 - 相机配置

**读取示例**:
```python
import numpy as np
import json

# 加载相机参数
data = np.load('000000.npz')
intrinsics = data['intrinsics']  # [8, 3, 3]
extrinsics = data['extrinsics']  # [8, 4, 4]
configs = json.loads(data['configs'])  # List[Dict]
```

---

## 相机配置（Tesla 风格）

8 个环视相机：

| 相机 ID | FOV | 位置 (x, y, z) | 朝向 (yaw) | 分辨率 |
|---------|-----|----------------|-----------|--------|
| cam_front_main | 50° | (1.0, 0.0, 1.6) | 0° | 1280×960 |
| cam_front_wide | 120° | (1.0, 0.0, 1.6) | 0° | 1280×960 |
| cam_front_narrow | 35° | (1.0, 0.0, 1.6) | 0° | 1280×960 |
| cam_left_pillar | 80° | (0.0, -0.9, 1.7) | -60° | 1280×960 |
| cam_right_pillar | 80° | (0.0, 0.9, 1.7) | 60° | 1280×960 |
| cam_left_repeater | 100° | (1.2, -0.9, 1.0) | -160° | 1280×960 |
| cam_right_repeater | 100° | (1.2, 0.9, 1.0) | 160° | 1280×960 |
| cam_rear | 120° | (-2.5, 0.0, 1.2) | 180° | 1280×960 |

---

## 数据量对比

### 单帧数据（8 个相机）

| 数据类型 | 大小 | 说明 |
|---------|------|------|
| Bayer RAW (cameras/) | 19.7 MB | 8 × 2.46 MB |
| RGB PNG (cameras_rgb/) | 16-24 MB | 8 × (2-3 MB)，PNG 压缩 |
| 深度图 (depth/) | 12 MB | 6 × 2 MB |
| 体素 (occupancy/) | < 1 MB | 压缩后很小 |
| 相机参数 (camera_params/) | < 0.1 MB | 很小 |
| **总计** | **约 48-56 MB/帧** | |

### 10K 数据集

| 数据类型 | 大小 |
|---------|------|
| Bayer RAW | ~197 GB |
| RGB PNG | ~160-240 GB |
| 深度图 | ~120 GB |
| 体素 + 参数 | ~10 GB |
| **总计** | **约 487-567 GB** |

**注意**: 如果只用于单一网络训练，可以删除不需要的数据：
- 只训练 occ_network_nano: 删除 `cameras_rgb/`，节省 160-240 GB
- 只训练 occ_network_lite: 删除 `cameras/`，节省 197 GB

---

## 使用专业 DNG 查看工具

支持的专业工具：
1. **Adobe Camera RAW** (Photoshop/Lightroom)
2. **RawTherapee** (免费开源)
3. **darktable** (免费开源，Linux/Mac/Windows)
4. **FastStone Image Viewer** (Windows，支持基本 DNG 查看)

**注意**:
- 由于是仿真数据的 Bayer 转换（非真实传感器 RAW），部分高级 DNG 功能可能不可用
- 基本的单通道灰度显示和 Bayer 模式查看应该都支持
- 如果需要更好的兼容性，可以安装 `Pillow` 和 `piexif` 包以生成完整 DNG 元数据

---

## 常见问题

### Q1: Bayer RAW 和 RGB PNG 有什么区别？

**A**:
- **Bayer RAW**: 单通道原始传感器数据，数据量小（-66%），保留更多动态范围信息，网络需要学习去马赛克
- **RGB PNG**: 标准彩色图像，已去马赛克，适合标准 CNN 网络，兼容性好

### Q2: 为什么需要两种格式？

**A**:
- **occ_network_nano**: 专为移动端优化，使用 Bayer RAW 降低计算量和内存带宽
- **occ_network_lite**: 标准网络架构，使用 RGB 数据更简单

### Q3: 可以只生成一种格式吗？

**A**: 可以！修改 `main_data_collection.py` 第 291-295 行：
```python
# 只生成 Bayer RAW（用于 occ_network_nano）
saver.save_bayer_as_dng(frame_idx, rgb_data, camera_configs=TESLA_CONFIGS)
# saver.save_rgb_preview(frame_idx, rgb_data)  # 注释掉这行

# 或者只生成 RGB PNG（用于 occ_network_lite）
# saver.save_bayer_as_dng(frame_idx, rgb_data, camera_configs=TESLA_CONFIGS)  # 注释掉这行
saver.save_rgb_preview(frame_idx, rgb_data)
```

### Q4: DNG 文件无法用普通图片查看器打开？

**A**:
- DNG 是 RAW 格式，需要专业工具（见上面"使用专业 DNG 查看工具"）
- 如果只想快速预览，使用 `cameras_rgb/` 中的 PNG 图像
- 或者安装 Python 包：`pip install Pillow piexif rawpy`

---

**祝数据采集和训练顺利！** 🚀

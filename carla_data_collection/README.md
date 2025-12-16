# CARLA Occupancy 数据采集系统

为纯视觉 Occupancy Network 训练准备数据对：**8 相机 RGB (12-bit, 带物理鱼眼畸变) → 3D 体素标签**

---

## 🚀 快速开始

### 前提条件
1. ✅ CARLA UE5.5 服务器已运行
2. ✅ Anaconda `carla` 环境已激活

### 运行命令

```cmd
conda activate carla
cd d:\code\carla\carla_data_collection
python scripts\collect_5_frames.py
```

**采集内容**: 5 帧数据 (8 相机 + 语义激光雷达 + 20 NPC + Occupancy Grid)

---

## 📁 输出结构

脚本运行后会在 `dataset_output/town10_test/` 目录下生成:

```
dataset_output/town10_test/
├── cameras/               # 相机图像 (PNG 格式)
│   ├── cam_front_main/    # 前置主摄 (50°)
│   ├── cam_front_wide/    # 前置广角 (120° 鱼眼)
│   ├── cam_front_narrow/  # 前置长焦 (35°)
│   ├── cam_left_pillar/   # 左侧 B 柱
│   ├── cam_right_pillar/  # 右侧 B 柱
│   ├── cam_left_repeater/ # 左侧翼子板
│   ├── cam_right_repeater/# 右侧翼子板
│   └── cam_rear/          # 后置相机 (120° 鱼眼)
├── lidar/                 # 激光雷达点云 (NPZ 压缩格式)
│   ├── 000000.npz         # 包含: points (N,6), frame, timestamp
│   └── ...
├── occupancy/             # 3D 体素标签 (NPZ 格式)
│   ├── 000000.npz         # 包含: occupancy (200,200,16), mask
│   └── ...
└── calibration.json       # 相机/激光雷达标定参数
```

---

## 🔧 已修复问题 (2025-12-16)

### Bug 修复
1. **车型硬编码**: 修复了 `vehicle.tesla.model3` 导致的崩溃，增加了智能车型选择。
2. **后置摄像头遮挡**: 调整了后置摄像头位置 (`x=-2.5, z=1.5`)，解决了图像全黑问题。
3. **数据保存**: 实现了完整的磁盘保存功能 (PNG/NPZ/JSON)。
4. **NPY 解析**: 修复了 Occupancy Viewer 中 NPY 解析器对标量数据的支持问题。

### 功能增强
1. **Occupancy 生成**: 实现了从语义激光雷达直接生成 Occupancy Grid 的算法。
2. **可视化工具**: 开发了基于 Three.js 的 `occupancy_viewer`，支持自动加载和 3D 交互。
3. **数据验证**: 提供了 `verify_occupancy.py` 脚本，可全面验证数据集完整性。

---

## 📁 关键文件说明

| 文件 | 说明 |
|------|------|
| `scripts/collect_5_frames.py` | **主采集脚本**，执行 5 帧完整采集流程 |
| `scripts/verify_occupancy.py` | **验证脚本**，检查数据集结构、完整性和有效性 |
| `data/occupancy_generator.py` | **核心算法**，点云转体素网格 |
| `config/camera_config.py` | **配置**，特斯拉 8 相机布局参数 |
| `config/occupancy_config.py` | **配置**，体素网格尺寸和语义映射 |

---

## 📊 数据验证

运行以下命令验证采集的数据集：

```cmd
python scripts\verify_occupancy.py
```

预期输出：
```
✓ 相机数据: 通过 (8 相机 × 5 帧)
✓ 标定文件: 通过
✓ 体素数据: 通过 (格式正确, Mask 一致)
✓ 所有验证通过! 数据集完整。
```

---

## 📖 参考文档

- [CARLA 3D Occupancy 体素数据获取原理](../.trae/documents/hero车辆与npc初始化/CARLA 3D Occupancy 体素数据获取原理.md)
- [CARLA UE5.5 Fisheye Camera 文档](https://carla-ue5.readthedocs.io/en/latest/cameras_and_sensors/)

---

## 🚧 待实现功能

- [ ] **大规模采集**: 支持多地图自动切换、长时间采集
- [ ] **多天气支持**: 随机天气变化 (雨/雾/夜)
- [ ] **动态场景**: 增加行人、骑行者等更多交通参与者

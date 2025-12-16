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

**采集内容**: 5 帧数据 (2 相机 + 语义激光雷达 + 20 NPC)

---

## 📁 输出结构

脚本运行后会在 `dataset_output/town10_test/` 目录下生成:

```
dataset_output/town10_test/
├── cameras/               # 相机图像 (PNG 格式)
│   ├── cam_front/        # 前置广角相机 (120° 鱼眼)
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   └── cam_rear/         # 后置广角相机 (90°)
│       ├── 000000.png
│       ├── 000001.png
│       └── ...
├── lidar/                # 激光雷达点云 (NPZ 压缩格式)
│   ├── 000000.npz        # 包含: points, frame, timestamp
│   ├── 000001.npz
│   └── ...
├── occupancy/            # 3D 体素标签 (TODO: 待实现)
│   └── ...
└── calibration.json      # 相机/激光雷达标定参数
```

---

## 📁 关键文件

- [collect_5_frames.py](scripts/collect_5_frames.py) - 主测试脚本
- [quick_test.py](scripts/quick_test.py) - 快速连接测试
- [validate_dataset.py](scripts/validate_dataset.py) - 数据集验证
- [运行测试.txt](运行测试.txt) - 详细运行说明

---

## 🔧 最新修复 (2025-12-16)

✅ **问题 1**: 硬编码车型 `vehicle.tesla.model3` 导致崩溃
**修复**: 智能车型选择,自动尝试多个候选车型

✅ **问题 2**: 后置摄像头全黑 (范围 [0, 0])
**修复**: 调整位置从 `x=-1.8, z=1.0` → `x=-2.5, z=1.5` (避免车身遮挡)

✅ **问题 3**: 数据未保存到磁盘
**修复**: 添加图像保存 (PNG) + 激光雷达保存 (NPZ) + 元数据保存 (JSON)

---

## 📖 参考文档

- [CARLA 3D Occupancy 体素数据获取原理](../.trae/documents/hero车辆与npc初始化/CARLA 3D Occupancy 体素数据获取原理.md)
- [CARLA UE5.5 Fisheye Camera 文档](https://carla-ue5.readthedocs.io/en/latest/cameras_and_sensors/)

---

## 🚧 待实现功能

- [ ] 体素化处理 (LiDAR → Occupancy Grid)
- [ ] 完整 8 相机布局 (当前仅 2 相机测试)
- [ ] 语义标签映射 (23 CARLA 类 → 18 Occupancy 类)
- [ ] 大规模数据采集脚本 (多场景、多地图)

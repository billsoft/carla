# CARLA 360° 全景稠密体素数据采集

基于 CubeMap 全景深度图方案的高质量 Occupancy 数据集生成工具。

> **最新更新 (2025-12-16)**: 修复全景反投影算法，消除体素间隙问题。详见 [REFACTOR_NOTES.md](REFACTOR_NOTES.md)

## 📁 项目结构

```
dense_occupancy_collection/
├── config/
│   ├── panorama_config.py       # 全景相机配置 (6个CubeMap面)
│   └── occupancy_config.py      # 体素空间配置 (200×200×16)
├── sensors/
│   ├── panorama_manager.py      # 全景传感器管理 (6深度+6语义)
│   └── rgb_camera_manager.py    # RGB相机管理 (8个训练相机)
├── processing/
│   ├── panorama_tools.py        # CubeMap→全景转换 + 反投影
│   └── dense_voxel_generator.py # 稠密体素生成
└── scripts/
    └── collect_panorama.py      # 主采集脚本
```

## 🚀 快速开始

```bash
# 1. 激活环境
conda activate carla

# 2. 确保CARLA服务器运行中 (localhost:2000)

# 3. 运行采集
cd d:\code\carla
python dense_occupancy_collection\scripts\collect_panorama.py --frames 10
```

## 📊 输出数据

```
dataset_output/
├── cameras/          # 8个RGB相机图像 (12-bit PNG, 1280×960)
│   ├── cam_front_main/
│   ├── cam_front_wide/
│   └── ...
├── depth/            # 全景深度图 (灰度PNG, 2048×1024)
├── semantic_color/   # 全景语义图 (彩色PNG, 2048×1024)
└── occupancy/        # 稠密体素 (NPZ, 200×200×16)
```

## 🎯 核心优势

与激光雷达和8独立相机方案对比：

| 特性 | 激光雷达 | 8独立相机 | **360°全景** |
|------|---------|-----------|-------------|
| 密度 | 稀疏 (~6万点) | 稠密但边界有问题 | **稠密且无缝** (~800万点) |
| 边界 | 扫描线间隙 | 拼接错位 | **连续无缝** |
| 适用 | 快速验证 | 原型测试 | **生产数据集** |

## 📖 参考文档

- [基于全景深度图方案的体素生成指南](../.trae/documents/hero车辆与npc初始化/基于全景深度图方案的CARLA_UE5_3D体素数据集生成指南.md)
- [方案对比与修正建议](../.trae/documents/hero车辆与npc初始化/方案对比与修正建议.md)

---

**更新时间**: 2025-12-16
**基于**: CARLA 0.10.0 (UE5.5)

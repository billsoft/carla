# OccNetV3 数据生成器

为 `occ_network` 训练生成符合格式的数据集

## 📋 生成的数据格式

### 输入: 8相机灰度图像
- **形状**: `(1, 960, 1280)` × 8
- **类型**: `float16`
- **值域**: `[0, 1]`

### 输出: 400×400×32 体素占用网格
- **形状**: `(400, 400, 32)`
- **类型**: `uint8`
- **类别**: 0-17 (18类语义)
- **空间**: X=[-40.0, 40.0], Y=[-40.0, 40.0], Z=[-1.0, 5.4]
- **分辨率**: 0.2m/体素

### 可选: 流场和时序
- **flow**: `(3, 400, 400, 32)` float16
- **ego_motion**: `(4, 4)` float32
- **ego_pose**: `(4, 4)` float32

## 🚀 快速开始

### 1. 启动CARLA服务器
```bash
cmake --build Build --target launch
```

### 2. 运行数据采集 (生成10条数据)
```bash
cd D:\code\carla
python occnetv3_data_generator\main_collection.py --frames 10 --output D:\code\carla\dataset_10k_bak
```

### 3. 检查生成的数据
```
dataset_10k_bak/
├── calibration/
│   ├── intrinsics.json
│   └── extrinsics.json
├── images/
│   └── scene_0000_frame_0000/
│       ├── cam_0.npy (1, 960, 1280) float16
│       ├── cam_1.npy
│       └── ...
├── occupancy/
│   └── scene_0000_frame_0000.npy (400, 400, 32) uint8
├── ego_pose/
│   └── scene_0000_frame_0000.npy (4, 4) float32
├── ego_motion/
│   └── scene_0000_frame_0000.npy (4, 4) float32
├── train.txt
├── val.txt
└── test.txt
```

## ⚙️ 参数说明

```bash
python occnetv3_data_generator\main_collection.py \
    --host localhost \        # CARLA服务器地址
    --port 2000 \             # CARLA端口
    --town Town10HD \         # 地图名称
    --output <path> \         # 输出目录
    --frames 10 \             # 采集帧数
    --scene scene             # 场景名称前缀
```

## 🔧 技术细节

### 相机配置
- **8相机布局**: 前3 + 侧4 + 后1 (Tesla FSD风格)
- **FOV**: 35°-120° (长焦到广角)
- **RGB → 灰度**: ITU-R BT.601 标准

### 体素生成
- **方法**: 复用 `dense_occupancy_collection` 的深度相机方法
- **6方向CubeMap**: 前/后/左/右/上/下 90° FOV
- **精度**: 0.2m 分辨率, 保守光栅化

### 语义映射
- **18类标准**: empty, barrier, bicycle, bus, car, construction_vehicle, motorcycle, pedestrian, traffic_cone, trailer, truck, driveable_surface, other_flat, sidewalk, terrain, manmade, vegetation, free

## 📊 数据验证

生成数据后,可使用 occ_network 的数据集验证:

```bash
python occ_network/data/validate_dataset.py --data_dir D:\code\carla\dataset_10k_bak
```

## 🐛 已知问题

1. **flow未实现**: 当前版本不生成流场,可后续添加
2. **性能**: 每帧约2-4秒 (取决于场景复杂度)
3. **内存**: 峰值约4-6 GB

## 📝 与 dense_occupancy_collection 的区别

| 项目 | dense_occupancy_collection | occnetv3_data_generator |
|------|----------------------------|-------------------------|
| 体素尺寸 | 500×500×40 | **512×512×40** |
| 图像格式 | RGB DNG | **单通道灰度 float16** |
| 数据格式 | NPZ (合并) | **独立NPY** |
| 类别数 | 17类 | **18类** |
| 用途 | 通用数据集 | **OccNetV3训练** |

## 🎯 后续改进

- [ ] 实现流场计算 (需要actor tracking)
- [ ] 支持多场景自动切换
- [ ] 添加数据增强 (亮度/对比度)
- [ ] 优化采集速度 (并行处理)

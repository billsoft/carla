# CARLA 360° 全景稠密体素数据采集 (Dense Occupancy Collection)

基于 **CARLA UE5.5** 和 **CubeMap 全景深度图** 的高质量 3D Semantic Occupancy 数据集生成工具。支持 17 类 nuScenes 语义标准，具备实例级补全 (Instance Completion) 和保守光栅化 (Conservative Rasterization) 功能。

## 🌟 核心特性

*   **全景覆盖**: 使用 6 路深度相机合成 360° 无死角视野。
*   **稠密体素**: 基于保守光栅化算法生成 100% 几何密度的体素，无扫描线间隙。
*   **实例补全**: 自动补全被部分遮挡的物体（如车辆），防止产生“空心”或“切片”数据。
*   **标准语义**: 对齐 nuScenes 17 类语义标签 (Car, Truck, Bus, Pedestrian, Drivable Surface 等)。
*   **UE5 光照**: 适配 UE5.5 Lumen 全局光照系统，生成高质量 RGB 图像。
*   **自动清洗**: 每次运行自动清理旧数据，防止数据集污染。

## 📁 项目结构

```
dense_occupancy_collection/
├── config/
│   ├── occupancy_config.py      # 体素空间参数 (范围、分辨率、颜色)
│   └── actor_occupancy_mapping.py # Actor 到 17 类语义的详细映射表
├── core/
│   ├── rgb_suite.py             # 8 路 RGB 相机套件 (带 UE5 光照修复)
│   ├── depth_suite.py           # 6 路深度相机套件
│   ├── voxel_generator.py       # 稠密体素生成器 (保守光栅化)
│   ├── visibility_filter.py     # 可见性过滤器 (深度图投影 + 实例补全)
│   └── scenario_manager.py      # 场景管理 (生成 Hero 和 NPC)
├── scripts/
│   ├── list_actor_types.py      # 列出所有可用 Actor 蓝图
│   ├── query_all_actors.py      # 查询当前场景中的 Actor 实例
│   ├── test_actor_mapping.py    # 验证语义映射逻辑
│   ├── test_rgb.py              # 验证 RGB 相机光照
│   └── diagnose_npz.py          # 诊断生成的 .npz 数据文件
├── main_data_collection.py      # 主采集脚本
└── README_映射配置.md            # 详细的语义映射文档
```

## 🚀 快速开始

### 1. 环境准备
确保已安装 CARLA 0.10.0 (UE5.5) 及其 Python API。

```bash
conda activate carla
```

### 2. 运行数据采集
启动 CARLA 服务器后，运行主脚本：

```bash
# 采集 10 帧数据 (默认地图 Town10HD_Opt)
python dense_occupancy_collection/main_data_collection.py --frames 10
```

### 3. 查看结果
启动可视化查看器：

```bash
python occupancy_viewer/run_viewer.py
# 浏览器打开 http://localhost:8085
```

## 📊 输出数据格式

数据保存在 `dataset_output/` 目录下，每次运行前会自动清理。

| 目录/文件 | 格式 | 描述 |
| :--- | :--- | :--- |
| `cameras/` | PNG | 8 路环视 RGB 图像 (Front, Wide, Pillars, Repeaters, Rear) |
| `depth/` | PNG | 6 路深度图 (16-bit, 单位: 毫米) |
| `occupancy/*.npz` | NPZ | 压缩的体素数据 (包含 `occupancy`, `actor_ids`, `mask`) |

### Occupancy 数据结构 (.npz)
*   **occupancy**: `(X, Y, Z)` uint8 数组，存储语义标签 (0-17)。
*   **actor_ids**: `(X, Y, Z)` int32 数组，存储每个体素所属的 Actor ID (用于实例分割)。
*   **mask**: `(X, Y, Z)` bool 数组，表示体素的可见性 (True=Visible)。

## 🎨 语义标签 (17类 nuScenes 标准)

| ID | 类别 (Name) | 颜色 (RGB) | 包含对象示例 |
| :--- | :--- | :--- | :--- |
| 0 | **free** | (0,0,0) | 空气 |
| 1 | **barrier** | (200,200,200) | 护栏, 施工围挡 |
| 2 | **bicycle** | (128,128,0) | 自行车 (Crossbike, Gazelle) |
| 3 | **bus** | (0,0,128) | 公交车, 面包车 (VW T2) |
| 4 | **car** | (0,128,0) | 轿车, SUV, Taxi |
| 5 | **construction_vehicle** | (128,0,128) | 工程车 |
| 6 | **motorcycle** | (128,0,0) | 摩托车 (Harley, Yamaha) |
| 7 | **pedestrian** | (255,0,0) | 行人 |
| 8 | **traffic_cone** | (255,165,0) | 交通锥桶 |
| 9 | **trailer** | (0,128,128) | 拖车 |
| 10 | **truck** | (0,0,255) | 卡车, 消防车, 救护车, Cybertruck |
| 11 | **driveable_surface** | (100,100,100) | 道路路面 |
| 12 | **other_flat** | (150,150,150) | 路肩, 停车位 |
| 13 | **sidewalk** | (255,192,203) | 人行道 |
| 14 | **terrain** | (0,255,0) | 草地, 地形 |
| 15 | **manmade** | (255,255,0) | 建筑, 路牌, ATM, 垃圾桶 |
| 16 | **vegetation** | (0,255,128) | 树木, 灌木, 花盆 |
| 17 | **general_object** | (255,0,255) | 未知障碍物, 垃圾, 杂物 |

## 🛠️ 技术细节

### 1. 坐标系与网格
*   **坐标系**: 左手系 (Left-Handed), Z-up (与 UE5 一致)。
*   **范围**: X: ±50m, Y: ±50m, Z: ±4m (相对于自车)。
*   **分辨率**: 0.2m (Grid Size: 500 × 500 × 40)。

### 2. 可见性过滤算法
采用 **"Depth Projection + Instance Completion"** 混合策略：
1.  **深度投影**: 将每个体素中心投影到 6 个深度相机平面，对比深度值判断遮挡。
2.  **实例补全**: 只要物体 (Actor ID) 有任何一部分被判定为可见，则强制该物体的**所有**体素可见。这解决了“空心车”问题。
3.  **地面保护**: 强制保留 `driveable_surface` 和 `sidewalk`，防止远处地面被错误过滤。

### 3. 光照修复 (UE5.5)
针对 UE5.5 Lumen 系统导致的黑屏问题，我们在 `RGBSuite` 中强制应用了：
*   `post_process_profile='Town10HD_Opt'` (加载地图专属 LUT)
*   `gamma=2.2`
*   `shutter_speed=200.0`, `iso=1200.0` (手动曝光控制)

## 📝 维护指南

### 添加新车型
如果 CARLA 更新了新车型，请修改 `config/actor_occupancy_mapping.py`：
```python
VEHICLE_MAPPING = {
    10: [ # Truck
        'vehicle.tesla.cybertruck',
        'vehicle.new.truck_model', # 新增
    ]
}
```

### 验证映射
运行测试脚本验证所有蓝图是否正确归类：
```bash
python dense_occupancy_collection/scripts/test_actor_mapping.py
```

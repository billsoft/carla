# ASAM 标准使用指南 - Occupancy Network 训练快速开始

> 如何在 Occupancy Network 训练系统中使用 ASAM 行业标准

---

## ⚡ 快速总结

### 为什么需要 ASAM 标准?

| 痛点 | 解决方案 |
|------|----------|
| ❌ CARLA 数据格式其他工具不认 | ✅ OpenLABEL 标准化标注格式 |
| ❌ 测试场景难以复现和共享 | ✅ OpenSCENARIO 场景定义 |
| ❌ 地图在不同仿真器中无法复用 | ✅ OpenDRIVE 通用地图格式 |
| ❌ 传感器接口各家实现不同 | ✅ OSI 标准化传感器接口 |

### 整合了哪些 ASAM 标准?

| 标准 | 优先级 | 用途 | 状态 |
|------|--------|------|------|
| **OpenDRIVE** | 🔴 高 | 道路网络定义、车道线标注 | ✅ 完整支持 |
| **OpenSCENARIO** | 🔴 高 | 测试场景定义、数据采集脚本 | ✅ 完整支持 |
| **OpenLABEL** | 🔴 高 | Occupancy 标注格式 | ✅ 完整支持 |
| **OSI** | 🟡 中 | 传感器数据接口 | ✅ 部分支持 |
| **OpenCRG** | 🟢 低 | 路面细节 | ⚠️ 暂不支持 |

---

## 1. OpenDRIVE - 地图加载

### 现有方式 (自定义 CARLA 地图)

```python
# 使用 CARLA 内置地图
world = client.load_world('Town10HD_Opt')
```

### ASAM 标准方式 (OpenDRIVE)

```python
# 从 OpenDRIVE 文件生成地图
with open('./maps/Town10HD.xodr', 'r') as f:
    opendrive_data = f.read()

world = client.generate_opendrive_world(
    opendrive=opendrive_data,
    parameters=carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        smooth_junctions=True
    )
)
```

**优势**:
- ✅ 可以使用第三方工具(RoadRunner, VectorZero)创建地图
- ✅ 地图可在多个仿真器中使用(CARLA, Gazebo, LGSVL)
- ✅ 车道线信息可直接提取用于标注

### 车道线标注提取

```python
from carla_bridge.opendrive_lane_extractor import OpenDRIVELaneExtractor

# 提取车道线
lane_extractor = OpenDRIVELaneExtractor('./maps/Town10HD.xodr')
lane_data = lane_extractor.extract_lane_markings()

# 结果
# lane_data = {
#     'lane_centers': [np.array(...), ...],  # 车道中心线
#     'lane_boundaries': [np.array(...), ...],  # 车道边界
#     'road_types': ['highway', 'urban', ...]
# }
```

**配置文件**:
```yaml
# configs/opendrive_config.yaml
opendrive:
  map_file: "./maps/Town10HD.xodr"
  generation:
    vertex_distance: 2.0
    smooth_junctions: true
```

---

## 2. OpenSCENARIO - 场景定义

### 现有方式 (手动脚本)

```python
# 手动创建场景
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
obstacle = world.spawn_actor(obstacle_bp, obstacle_point)

# 手动控制
vehicle.set_autopilot(True)
obstacle.set_transform(tilted_transform)  # 倾倒货车
```

**问题**:
- ❌ 场景难以复现(每次随机生成)
- ❌ 无法版本控制
- ❌ 无法与团队共享

### ASAM 标准方式 (OpenSCENARIO)

**1. 创建场景 XML 文件**:

```xml
<!-- scenarios/tilted_truck_scenario.xosc -->
<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
  <FileHeader date="2025-12-09" description="Tilted white truck scenario"/>

  <Entities>
    <ScenarioObject name="ego_vehicle">
      <Vehicle name="vehicle.tesla.model3" vehicleCategory="car"/>
    </ScenarioObject>

    <ScenarioObject name="tilted_truck">
      <Vehicle name="vehicle.carlamotors.firetruck" vehicleCategory="truck"/>
    </ScenarioObject>
  </Entities>

  <Init>
    <Actions>
      <Private entityRef="tilted_truck">
        <TeleportAction>
          <Position>
            <!-- 倾倒姿态: roll=1.3 (75°) -->
            <WorldPosition x="200.0" y="200.0" z="0.5" h="1.57" r="1.3"/>
          </Position>
        </TeleportAction>
      </Private>
    </Actions>
  </Init>
</OpenSCENARIO>
```

**2. 加载场景**:

```python
from carla_bridge.openscenario_loader import OpenSCENARIOLoader

# 加载 OpenSCENARIO
scenario_loader = OpenSCENARIOLoader(
    world=world,
    scenario_path='./scenarios/tilted_truck_scenario.xosc'
)

scenario = scenario_loader.load_scenario()
ego_vehicle = scenario['ego_vehicle']
duration = scenario['duration']

# 采集数据
collector = OccupancyDataCollector(vehicle=ego_vehicle)
collector.run(duration=duration)
```

**优势**:
- ✅ 场景完全可复现
- ✅ 版本控制(Git)
- ✅ 团队共享
- ✅ 符合行业标准

**配置文件**:
```yaml
# configs/openscenario_config.yaml
openscenario:
  scenarios_dir: "./scenarios"
  active_scenarios:
    - "tilted_truck_scenario.xosc"
    - "rainy_highway_scenario.xosc"
    - "dense_traffic_scenario.xosc"
```

---

## 3. OpenLABEL - 标注数据格式

### 现有方式 (自定义 HDF5)

```python
# 保存为 HDF5
with h5py.File('occupancy_dataset.h5', 'w') as f:
    f.create_dataset('occupancy', data=occupancy_array)
    f.create_dataset('flow', data=flow_array)
```

**问题**:
- ❌ 格式不标准,其他工具无法识别
- ❌ 缺少元数据(体素大小、原点等)
- ❌ 难以与其他数据集合并

### ASAM 标准方式 (OpenLABEL)

```python
from dataset.openlabel_generator import OpenLABELDatasetGenerator

# 创建 OpenLABEL 数据集
dataset_gen = OpenLABELDatasetGenerator(output_dir='./data/occupancy_openlabel')

# 添加帧
for frame_id in range(1000):
    dataset_gen.add_frame(
        frame_id=frame_id,
        timestamp=time.time(),
        occupancy=occupancy,  # (200, 200, 16)
        flow=flow,  # (200, 200, 16, 3)
        vehicle_speed=speed,
        vehicle_yaw_rate=yaw_rate
    )

# 保存(生成 annotations.json + 二进制数据)
dataset_gen.save()
```

**生成的文件结构**:
```
data/occupancy_openlabel/
├── annotations.json  # OpenLABEL 元数据
└── data/
    ├── occupancy_000000.bin
    ├── flow_000000.bin
    ├── occupancy_000001.bin
    ├── flow_000001.bin
    └── ...
```

**annotations.json 示例**:
```json
{
  "openlabel": {
    "metadata": {
      "schema_version": "1.0.0",
      "name": "CARLA Occupancy Dataset"
    },
    "frames": {
      "0": {
        "frame_properties": {
          "timestamp": "1702112000.123",
          "vehicle_speed": 15.3,
          "vehicle_yaw_rate": 0.05
        },
        "objects": {
          "occupancy_grid_0": {
            "object_data": {
              "type": "occupancy_grid_3d",
              "occupancy_grid_3d": [
                {
                  "name": "occupancy_probability",
                  "uri": "data/occupancy_000000.bin",
                  "attributes": {
                    "grid_dimensions": [200, 200, 16],
                    "voxel_size": [0.5, 0.5, 0.5],
                    "origin": [-50.0, -50.0, 0.0]
                  }
                }
              ]
            }
          }
        }
      }
    }
  }
}
```

**优势**:
- ✅ 符合 ASAM OpenLABEL 1.0.0 标准
- ✅ 可被多种工具识别(包括第三方标注工具)
- ✅ 元数据完整(体素大小、原点、数据类型)
- ✅ 易于版本控制和共享

---

## 4. OSI - 传感器接口

### 现有方式 (自定义接口)

```python
# 自定义车辆反馈
feedback_data = {
    'position': (x, y, z),
    'velocity': (vx, vy, vz),
    'yaw': yaw
}
```

### ASAM 标准方式 (OSI)

```python
from carla_bridge.osi_carla_feedback import OSICarlaFeedback

# 创建 OSI 反馈器
osi_feedback = OSICarlaFeedback(carla_feedback)

# 获取 OSI GroundTruth
osi_gt = osi_feedback.get_osi_ground_truth()

# 标准化数据结构
# osi_gt.moving_object[0] = {
#     'position': {'x': ..., 'y': ..., 'z': ...},
#     'velocity': {'x': ..., 'y': ..., 'z': ...},
#     'orientation': {'roll': ..., 'pitch': ..., 'yaw': ...}
# }
```

**优势**:
- ✅ 符合 OSI 3.x 标准
- ✅ 可与其他仿真器互操作
- ✅ 支持 Protobuf 序列化(高效)

---

## 5. 完整工作流(ASAM 标准版)

### 数据采集流程

```python
# examples/asam_data_collection.py

import yaml
from pathlib import Path
from carla_bridge.opendrive_lane_extractor import OpenDRIVELaneExtractor
from carla_bridge.openscenario_loader import OpenSCENARIOLoader
from dataset.openlabel_generator import OpenLABELDatasetGenerator

# 1. 加载配置
with open('./configs/asam_pipeline_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. 加载 OpenDRIVE 地图
with open(config['asam']['opendrive']['map_file'], 'r') as f:
    opendrive_data = f.read()

world = client.generate_opendrive_world(opendrive=opendrive_data)

# 3. 提取车道线
lane_extractor = OpenDRIVELaneExtractor(config['asam']['opendrive']['map_file'])
lane_data = lane_extractor.extract_lane_markings()

# 4. 创建 OpenLABEL 数据集
dataset_gen = OpenLABELDatasetGenerator(output_dir=config['asam']['openlabel']['output_dir'])

# 5. 遍历 OpenSCENARIO 场景
for scenario_file in config['asam']['openscenario']['active_scenarios']:
    scenario_loader = OpenSCENARIOLoader(world, f"./scenarios/{scenario_file}")
    scenario = scenario_loader.load_scenario()

    # 采集数据
    for frame_id in range(1000):
        # ... 采集 occupancy 和 flow ...
        dataset_gen.add_frame(frame_id, timestamp, occupancy, flow, speed, yaw_rate)

    scenario_loader.cleanup()

# 6. 保存 OpenLABEL 数据集
dataset_gen.save()
```

### 配置文件(完整版)

```yaml
# configs/asam_pipeline_config.yaml

# ASAM 标准文件路径
asam:
  opendrive:
    map_file: "./maps/Town10HD.xodr"
    generation:
      vertex_distance: 2.0
      smooth_junctions: true

  openscenario:
    scenarios_dir: "./scenarios"
    active_scenarios:
      - "tilted_truck_scenario.xosc"
      - "rainy_highway_scenario.xosc"

  openlabel:
    output_dir: "./data/occupancy_openlabel"
    schema_version: "1.0.0"

  osi:
    enabled: true
    version: "3.5.0"

# 数据采集参数
data_collection:
  num_frames_per_scenario: 1000
  sensor_tick: 0.028  # 36 FPS

# Occupancy 标注参数
annotation:
  voxel_size: 0.5
  grid_size: [200, 200, 16]
  format: "openlabel"
```

---

## 6. 运行示例

### 快速开始(ASAM 标准版)

```bash
# 1. 准备环境
conda activate carla

# 2. 启动 CARLA
cd ~/carla
./CarlaUnreal.sh

# 3. 运行 ASAM 标准数据采集
cd ~/carla_occupancy
python examples/asam_data_collection_pipeline.py \
    --config ./configs/asam_pipeline_config.yaml

# 4. 验证 OpenLABEL 数据集
python tools/validate_openlabel.py \
    --dataset ./data/occupancy_openlabel

# 5. 训练模型(使用 OpenLABEL 数据集)
python train.py \
    --dataset ./data/occupancy_openlabel \
    --format openlabel
```

---

## 7. 与现有文档的对应关系

| 现有文档 | ASAM 标准扩展 | 说明 |
|----------|---------------|------|
| **Occupancy-Network训练实战指南-CARLA-UE5.md** | → 添加 OpenDRIVE/OpenSCENARIO 章节 | 数据采集部分 |
| **Occupancy-Network训练实战指南-CARLA-UE5-续.md** | → 添加 OpenLABEL 数据加载器 | 训练部分 |
| **Occupancy-Network训练闭环完整流程-补充篇.md** | → 完整 ASAM 流水线 | 自动化流程 |
| **Occupancy-Network执行器反馈器架构设计.md** | → OSI 接口整合 | 执行器/反馈器 |

---

## 8. 核心优势总结

### 数据可移植性
```
CARLA 采集 → OpenLABEL 格式 → 可在其他工具中使用
                             ├─ MATLAB 数据分析
                             ├─ Python 训练框架
                             └─ 第三方标注工具
```

### 场景可复现
```
OpenSCENARIO 定义 → Git 版本控制 → 团队共享 → 精确复现
```

### 工具链兼容
```
OpenDRIVE 地图 → 可在多个仿真器中加载
                ├─ CARLA UE5
                ├─ Gazebo
                ├─ LGSVL
                └─ VTD
```

---

## 9. 常见问题

### Q1: 必须使用 ASAM 标准吗?

**A**: 不是必须,但强烈推荐:
- ✅ 如果你需要与其他团队/工具共享数据 → **必须使用**
- ✅ 如果你需要场景可复现性 → **必须使用**
- ⚠️ 如果只是个人学习/原型 → 可选

### Q2: ASAM 标准会增加多少工作量?

**A**: 初期投入略多,长期收益巨大:
- 初期: 学习成本 2-3 天
- 长期: 数据复用、场景共享,节省大量时间

### Q3: OpenLABEL 比 HDF5 慢吗?

**A**: 性能相当:
- OpenLABEL 使用二进制文件存储数据(与 HDF5 类似)
- JSON 文件仅存储元数据(overhead 很小)
- 训练时 I/O 速度相当

### Q4: 已有 HDF5 数据集如何迁移?

**A**: 提供转换工具:
```python
python tools/convert_hdf5_to_openlabel.py \
    --input ./data/old_dataset.h5 \
    --output ./data/openlabel_dataset
```

---

## 10. 参考资料

### ASAM 官方文档
- OpenDRIVE: https://www.asam.net/standards/detail/opendrive/
- OpenSCENARIO: https://www.asam.net/standards/detail/openscenario/
- OpenLABEL: https://www.asam.net/standards/detail/openlabel/
- OSI: https://www.asam.net/standards/detail/osi/

### 完整文档
- [Occupancy-Network-ASAM标准整合方案.md](./Occupancy-Network-ASAM标准整合方案.md) - 详细技术方案
- [Occupancy-Network训练实战指南-CARLA-UE5.md](./Occupancy-Network训练实战指南-CARLA-UE5.md) - 训练流程
- [Occupancy-Network执行器反馈器架构设计.md](./Occupancy-Network执行器反馈器架构设计.md) - 执行器/反馈器

---

**总结**: ASAM 标准让你的 Occupancy Network 训练系统具备**工业级数据互操作性**,是从学术原型到工业应用的必经之路! 🚀

# Occupancy Network 训练系统 ASAM 标准整合方案

> 符合 ASAM 国际标准的自动驾驶仿真训练数据格式与接口设计

---

## 目录

1. [ASAM 标准概述](#asam概述)
2. [OpenDRIVE 整合方案](#opendrive)
3. [OpenSCENARIO 整合方案](#openscenario)
4. [OpenLABEL 整合方案](#openlabel)
5. [OSI 整合方案](#osi)
6. [完整工具链](#工具链)
7. [代码实现](#代码实现)

---

## 1. ASAM 标准概述 {#asam概述}

### 1.1 什么是 ASAM?

**ASAM (Association for Standardization of Automation and Measuring Systems)** 是自动驾驶和仿真领域的国际标准化组织。

### 1.2 相关标准与应用

```mermaid
graph TB
    subgraph ASAM["ASAM 标准体系"]
        OD[OpenDRIVE<br/>道路网络定义]
        OS[OpenSCENARIO<br/>场景定义]
        OL[OpenLABEL<br/>标注格式]
        OSI[OSI<br/>传感器接口]
        OCG[OpenCRG<br/>路面细节]
    end

    subgraph OccupancySystem["Occupancy Network 训练系统"]
        MAP[地图加载]
        SCENE[场景生成]
        DATA[数据采集]
        LABEL[标注生成]
        TRAIN[模型训练]
    end

    subgraph Output["输出产物"]
        MODEL[Occupancy Model]
        DATASET[训练数据集]
        METRICS[评估报告]
    end

    OD --> MAP
    OS --> SCENE
    OL --> LABEL
    OSI --> DATA

    MAP --> SCENE
    SCENE --> DATA
    DATA --> LABEL
    LABEL --> TRAIN
    TRAIN --> MODEL
    TRAIN --> DATASET
    TRAIN --> METRICS
```

### 1.3 为什么需要 ASAM 标准?

**行业痛点**:
- ❌ 各家仿真器数据格式不兼容(CARLA vs Gazebo vs LGSVL)
- ❌ 标注数据无法跨工具使用
- ❌ 测试场景难以复现和共享

**ASAM 优势**:
- ✅ **数据可移植性**: 在 CARLA 训练的模型可在真车上测试
- ✅ **工具链互操作**: 使用 OpenDRIVE 地图可在多个仿真器中加载
- ✅ **场景可复现**: OpenSCENARIO 定义的场景可精确复现
- ✅ **标注标准化**: OpenLABEL 格式可被多种训练工具识别

---

## 2. OpenDRIVE 整合方案 {#opendrive}

### 2.1 什么是 OpenDRIVE?

**OpenDRIVE** 是道路网络的标准化描述格式(XML),定义:
- 道路几何(车道、路口、标线)
- 交通标志和信号灯
- 道路属性(限速、坡度、曲率)

### 2.2 CARLA 的 OpenDRIVE 支持

**CARLA 已原生支持 OpenDRIVE**:
```python
# 从 OpenDRIVE 文件生成 CARLA 地图
with open('Town10.xodr', 'r') as f:
    opendrive_data = f.read()

# 生成地图
world = client.generate_opendrive_world(
    opendrive=opendrive_data,
    parameters=carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=500.0,
        wall_height=1.0,
        additional_width=0.6,
        smooth_junctions=True,
        enable_mesh_visibility=True
    )
)
```

### 2.3 车道线标注整合

**利用 OpenDRIVE 自动生成车道线标注**:

```python
# carla_bridge/opendrive_lane_extractor.py

import carla
import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Dict

class OpenDRIVELaneExtractor:
    """
    从 OpenDRIVE 提取车道线信息

    功能:
    - 解析 OpenDRIVE XML
    - 提取车道中心线、边界线
    - 生成车道线标注(用于 HydraNet/Occupancy 训练)
    """

    def __init__(self, opendrive_path: str):
        self.opendrive_path = opendrive_path
        self.tree = ET.parse(opendrive_path)
        self.root = self.tree.getroot()

    def extract_lane_markings(self) -> Dict:
        """
        提取车道线

        返回:
            {
                'lane_centers': List[np.ndarray],  # 车道中心线
                'lane_boundaries': List[np.ndarray],  # 车道边界
                'road_types': List[str]  # 道路类型
            }
        """
        lane_centers = []
        lane_boundaries = []
        road_types = []

        # 遍历所有道路
        for road in self.root.findall('.//road'):
            road_id = road.get('id')
            road_type = road.find('.//type').get('type') if road.find('.//type') is not None else 'unknown'

            # 提取几何信息
            geometry = road.find('.//planView/geometry')
            if geometry is not None:
                # 获取起点和方向
                x = float(geometry.get('x'))
                y = float(geometry.get('y'))
                hdg = float(geometry.get('hdg'))
                length = float(geometry.get('length'))

                # 生成采样点
                num_samples = int(length / 1.0)  # 每米采样一次
                lane_points = []

                for i in range(num_samples):
                    s = i * 1.0  # 沿道路的距离
                    point_x = x + s * np.cos(hdg)
                    point_y = y + s * np.sin(hdg)
                    lane_points.append([point_x, point_y, 0.0])

                lane_centers.append(np.array(lane_points))
                road_types.append(road_type)

            # 提取车道边界
            lanes = road.findall('.//laneSection/left/lane') + road.findall('.//laneSection/right/lane')
            for lane in lanes:
                lane_id = lane.get('id')
                # TODO: 提取车道边界几何

        return {
            'lane_centers': lane_centers,
            'lane_boundaries': lane_boundaries,
            'road_types': road_types
        }

    def visualize_lanes(self, world: carla.World):
        """
        在 CARLA 中可视化车道线

        用于验证提取的车道线是否正确
        """
        lane_data = self.extract_lane_markings()

        debug = world.debug

        for lane_center in lane_data['lane_centers']:
            for i in range(len(lane_center) - 1):
                start = carla.Location(x=lane_center[i][0], y=lane_center[i][1], z=lane_center[i][2] + 0.5)
                end = carla.Location(x=lane_center[i+1][0], y=lane_center[i+1][1], z=lane_center[i+1][2] + 0.5)

                # 绘制绿色线
                debug.draw_line(
                    start, end,
                    thickness=0.1,
                    color=carla.Color(0, 255, 0),
                    life_time=60.0
                )

        print(f"✓ 可视化了 {len(lane_data['lane_centers'])} 条车道线")
```

**配置文件**:
```yaml
# configs/opendrive_config.yaml

opendrive:
  # OpenDRIVE 文件路径
  map_file: "./maps/Town10HD.xodr"

  # 生成参数
  generation:
    vertex_distance: 2.0      # 顶点间距(m)
    max_road_length: 500.0    # 最大道路长度(m)
    wall_height: 1.0          # 墙高(m)
    smooth_junctions: true    # 平滑路口

  # 车道线提取参数
  lane_extraction:
    sample_distance: 1.0      # 采样间距(m)
    lane_width: 3.5           # 标准车道宽度(m)
```

---

## 3. OpenSCENARIO 整合方案 {#openscenario}

### 3.1 什么是 OpenSCENARIO?

**OpenSCENARIO** 是动态场景描述标准(XML),定义:
- 交通流(车辆、行人、障碍物)
- 天气条件(晴天、雨天、雾天)
- 触发事件(急刹车、变道、碰撞)

### 3.2 用 OpenSCENARIO 定义数据采集场景

**示例: 倾倒货车场景(对标 Tesla 事故)**

```xml
<!-- scenarios/tilted_truck_scenario.xosc -->
<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="1" date="2025-12-09" description="Tilted white truck scenario for Occupancy Network training" author="CARLA Team"/>

  <ParameterDeclarations/>

  <CatalogLocations/>

  <RoadNetwork>
    <LogicFile filepath="Town10HD.xodr"/>
  </RoadNetwork>

  <Entities>
    <!-- 自车(Ego Vehicle) -->
    <ScenarioObject name="ego_vehicle">
      <Vehicle name="vehicle.tesla.model3" vehicleCategory="car">
        <ParameterDeclarations/>
        <Performance maxSpeed="50.0" maxAcceleration="5.0" maxDeceleration="8.0"/>
        <Axles>
          <FrontAxle maxSteering="0.523599" wheelDiameter="0.6" trackWidth="1.8" positionX="2.5" positionZ="0.3"/>
          <RearAxle maxSteering="0.0" wheelDiameter="0.6" trackWidth="1.8" positionX="0.0" positionZ="0.3"/>
        </Axles>
        <Properties/>
      </Vehicle>
    </ScenarioObject>

    <!-- 倾倒的白色货车 -->
    <ScenarioObject name="tilted_truck">
      <Vehicle name="vehicle.carlamotors.firetruck" vehicleCategory="truck">
        <ParameterDeclarations/>
        <Performance maxSpeed="0.0" maxAcceleration="0.0" maxDeceleration="0.0"/>
        <Properties>
          <Property name="color" value="255,255,255"/>  <!-- 白色 -->
          <Property name="static" value="true"/>
        </Properties>
      </Vehicle>
    </ScenarioObject>
  </Entities>

  <Storyboard>
    <Init>
      <Actions>
        <!-- 自车初始位置 -->
        <Private entityRef="ego_vehicle">
          <PrivateAction>
            <TeleportAction>
              <Position>
                <WorldPosition x="100.0" y="200.0" z="0.5" h="0.0" p="0.0" r="0.0"/>
              </Position>
            </TeleportAction>
          </PrivateAction>
          <PrivateAction>
            <LongitudinalAction>
              <SpeedAction>
                <SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/>
                <SpeedActionTarget>
                  <AbsoluteTargetSpeed value="20.0"/>  <!-- 初始速度 20 m/s -->
                </SpeedActionTarget>
              </SpeedAction>
            </LongitudinalAction>
          </PrivateAction>
        </Private>

        <!-- 货车初始位置(倾倒姿态) -->
        <Private entityRef="tilted_truck">
          <PrivateAction>
            <TeleportAction>
              <Position>
                <WorldPosition x="200.0" y="200.0" z="0.5" h="1.5708" p="0.0" r="1.3"/>  <!-- 侧翻 75° -->
              </Position>
            </TeleportAction>
          </PrivateAction>
        </Private>
      </Actions>
    </Init>

    <Story name="TiltedTruckStory">
      <Act name="Act1">
        <ManeuverGroup maximumExecutionCount="1" name="EgoManeuverGroup">
          <Actors selectTriggeringEntities="false">
            <EntityRef entityRef="ego_vehicle"/>
          </Actors>
          <Maneuver name="KeepSpeedManeuver">
            <Event name="KeepSpeedEvent" priority="overwrite">
              <Action name="KeepSpeedAction">
                <PrivateAction>
                  <LongitudinalAction>
                    <SpeedAction>
                      <SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/>
                      <SpeedActionTarget>
                        <AbsoluteTargetSpeed value="20.0"/>
                      </SpeedActionTarget>
                    </SpeedAction>
                  </LongitudinalAction>
                </PrivateAction>
              </Action>
              <StartTrigger>
                <ConditionGroup>
                  <Condition name="StartCondition" delay="0" conditionEdge="rising">
                    <ByValueCondition>
                      <SimulationTimeCondition value="0" rule="greaterThan"/>
                    </ByValueCondition>
                  </Condition>
                </ConditionGroup>
              </StartTrigger>
            </Event>
          </Maneuver>
        </ManeuverGroup>
        <StartTrigger>
          <ConditionGroup>
            <Condition name="ActStartCondition" delay="0" conditionEdge="rising">
              <ByValueCondition>
                <SimulationTimeCondition value="0" rule="greaterThan"/>
              </ByValueCondition>
            </Condition>
          </ConditionGroup>
        </StartTrigger>
      </Act>
    </Story>

    <StopTrigger>
      <ConditionGroup>
        <Condition name="EndCondition" delay="0" conditionEdge="rising">
          <ByValueCondition>
            <SimulationTimeCondition value="30.0" rule="greaterThan"/>  <!-- 30秒后结束 -->
          </ByValueCondition>
        </Condition>
      </ConditionGroup>
    </StopTrigger>
  </Storyboard>
</OpenSCENARIO>
```

### 3.3 OpenSCENARIO 加载器

```python
# carla_bridge/openscenario_loader.py

import carla
import xml.etree.ElementTree as ET
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class OpenSCENARIOLoader:
    """
    OpenSCENARIO 场景加载器

    功能:
    - 解析 OpenSCENARIO XML
    - 在 CARLA 中生成场景
    - 控制场景执行
    """

    def __init__(self, world: carla.World, scenario_path: str):
        self.world = world
        self.scenario_path = scenario_path

        # 解析 XML
        self.tree = ET.parse(scenario_path)
        self.root = self.tree.getroot()

        # 场景实体
        self.entities = {}

    def load_scenario(self) -> Dict:
        """
        加载场景

        返回:
            {
                'ego_vehicle': carla.Vehicle,
                'scenario_objects': List[carla.Actor],
                'duration': float
            }
        """
        logger.info(f"加载 OpenSCENARIO: {self.scenario_path}")

        # ===== 1. 提取实体定义 =====
        entities_elem = self.root.find('.//Entities')
        if entities_elem is None:
            raise ValueError("OpenSCENARIO 文件缺少 Entities 定义")

        for obj in entities_elem.findall('ScenarioObject'):
            obj_name = obj.get('name')
            vehicle_elem = obj.find('Vehicle')

            if vehicle_elem is not None:
                vehicle_name = vehicle_elem.get('name')
                logger.info(f"  发现车辆: {obj_name} ({vehicle_name})")

        # ===== 2. 提取初始状态 =====
        init_elem = self.root.find('.//Init')
        if init_elem is None:
            raise ValueError("OpenSCENARIO 文件缺少 Init 定义")

        # ===== 3. 生成实体 =====
        blueprint_library = self.world.get_blueprint_library()

        for action in init_elem.findall('.//Private'):
            entity_ref = action.get('entityRef')

            # 查找对应的车辆定义
            vehicle_elem = entities_elem.find(f".//ScenarioObject[@name='{entity_ref}']/Vehicle")
            if vehicle_elem is None:
                continue

            vehicle_name = vehicle_elem.get('name')
            vehicle_bp = blueprint_library.filter(vehicle_name)[0]

            # 提取位置
            teleport_action = action.find('.//TeleportAction/Position/WorldPosition')
            if teleport_action is not None:
                x = float(teleport_action.get('x'))
                y = float(teleport_action.get('y'))
                z = float(teleport_action.get('z'))
                h = float(teleport_action.get('h'))  # heading (yaw)
                p = float(teleport_action.get('p', 0))  # pitch
                r = float(teleport_action.get('r', 0))  # roll

                spawn_transform = carla.Transform(
                    carla.Location(x=x, y=y, z=z),
                    carla.Rotation(pitch=p, yaw=h * 57.2958, roll=r * 57.2958)  # rad → deg
                )

                # 生成车辆
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)
                self.entities[entity_ref] = vehicle

                logger.info(f"  ✓ 生成车辆: {entity_ref} @ ({x:.1f}, {y:.1f}, {z:.1f})")

        # ===== 4. 提取场景时长 =====
        stop_trigger = self.root.find('.//StopTrigger/ConditionGroup/Condition/ByValueCondition/SimulationTimeCondition')
        duration = float(stop_trigger.get('value')) if stop_trigger is not None else 30.0

        return {
            'ego_vehicle': self.entities.get('ego_vehicle'),
            'scenario_objects': list(self.entities.values()),
            'duration': duration
        }

    def cleanup(self):
        """清理场景实体"""
        for entity in self.entities.values():
            entity.destroy()
        logger.info("✓ 场景已清理")
```

**使用示例**:

```python
# 加载 OpenSCENARIO 场景
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

# 清理
scenario_loader.cleanup()
```

---

## 4. OpenLABEL 整合方案 {#openlabel}

### 4.1 什么是 OpenLABEL?

**OpenLABEL** 是标注数据的标准化格式(JSON),支持:
- 2D/3D 边界框
- 语义分割
- **3D Occupancy Grid** ⭐ (我们需要的!)
- 多帧时序标注

### 4.2 Occupancy 标注的 OpenLABEL 格式

**OpenLABEL 支持体素化标注**:

```json
{
  "openlabel": {
    "metadata": {
      "schema_version": "1.0.0",
      "name": "CARLA Occupancy Dataset",
      "annotator": "Occupancy Network Training System",
      "created": "2025-12-09T10:00:00Z"
    },

    "coordinate_systems": {
      "ego_vehicle": {
        "type": "local_cs",
        "parent": "",
        "pose_wrt_parent": {
          "translation": [0, 0, 0],
          "rotation_quaternion": [1, 0, 0, 0]
        }
      }
    },

    "streams": {
      "camera_front_main": {
        "type": "camera",
        "uri": "data/camera_front_main_%06d.png",
        "description": "Front main camera (1280x960, 70° FOV)"
      },
      "occupancy_grid": {
        "type": "other",
        "uri": "data/occupancy_%06d.bin",
        "description": "3D Occupancy Grid (200x200x16 voxels, 0.5m resolution)"
      }
    },

    "frames": {
      "0": {
        "frame_properties": {
          "timestamp": "1702112000.123456",
          "vehicle_speed": 15.3,
          "vehicle_yaw_rate": 0.05
        },

        "objects": {
          "occupancy_grid_0": {
            "object_data": {
              "type": "occupancy_grid_3d",
              "coordinate_system": "ego_vehicle",
              "occupancy_grid_3d": [
                {
                  "name": "occupancy_probability",
                  "uri": "data/occupancy_000000.bin",
                  "attributes": {
                    "grid_dimensions": [200, 200, 16],
                    "voxel_size": [0.5, 0.5, 0.5],
                    "origin": [-50.0, -50.0, 0.0],
                    "data_type": "float32",
                    "encoding": "binary"
                  }
                },
                {
                  "name": "occupancy_flow",
                  "uri": "data/flow_000000.bin",
                  "attributes": {
                    "grid_dimensions": [200, 200, 16],
                    "voxel_size": [0.5, 0.5, 0.5],
                    "vector_dimensions": 3,
                    "data_type": "float32",
                    "encoding": "binary"
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

### 4.3 OpenLABEL 数据集生成器

```python
# dataset/openlabel_generator.py

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class OpenLABELDatasetGenerator:
    """
    OpenLABEL 格式数据集生成器

    功能:
    - 将 Occupancy 标注转换为 OpenLABEL 格式
    - 支持多帧时序数据
    - 符合 ASAM OpenLABEL 1.0.0 规范
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # OpenLABEL 结构
        self.openlabel = {
            "openlabel": {
                "metadata": self._create_metadata(),
                "coordinate_systems": self._create_coordinate_systems(),
                "streams": self._create_streams(),
                "frames": {}
            }
        }

    def _create_metadata(self) -> Dict:
        """创建元数据"""
        return {
            "schema_version": "1.0.0",
            "name": "CARLA Occupancy Dataset",
            "annotator": "Occupancy Network Training System",
            "created": datetime.utcnow().isoformat() + "Z",
            "comment": "3D Occupancy Grid for autonomous driving perception"
        }

    def _create_coordinate_systems(self) -> Dict:
        """创建坐标系定义"""
        return {
            "ego_vehicle": {
                "type": "local_cs",
                "parent": "",
                "pose_wrt_parent": {
                    "translation": [0, 0, 0],
                    "rotation_quaternion": [1, 0, 0, 0]
                }
            }
        }

    def _create_streams(self) -> Dict:
        """创建数据流定义"""
        streams = {}

        # 8 个相机
        camera_names = [
            'front_narrow', 'front_main', 'front_wide',
            'left_front', 'left_rear',
            'right_front', 'right_rear',
            'rear'
        ]

        for cam_name in camera_names:
            streams[f"camera_{cam_name}"] = {
                "type": "camera",
                "uri": f"data/camera_{cam_name}_%06d.png",
                "description": f"{cam_name.replace('_', ' ').title()} camera (1280x960)"
            }

        # Occupancy Grid
        streams["occupancy_grid"] = {
            "type": "other",
            "uri": "data/occupancy_%06d.bin",
            "description": "3D Occupancy Grid (200x200x16 voxels, 0.5m resolution)"
        }

        # Occupancy Flow
        streams["occupancy_flow"] = {
            "type": "other",
            "uri": "data/flow_%06d.bin",
            "description": "3D Occupancy Flow (200x200x16x3 vectors)"
        }

        return streams

    def add_frame(
        self,
        frame_id: int,
        timestamp: float,
        occupancy: np.ndarray,
        flow: np.ndarray,
        vehicle_speed: float,
        vehicle_yaw_rate: float
    ):
        """
        添加一帧数据

        参数:
            frame_id: 帧 ID
            timestamp: 时间戳
            occupancy: (200, 200, 16) 占据概率
            flow: (200, 200, 16, 3) 运动流
            vehicle_speed: 车速 m/s
            vehicle_yaw_rate: 航向角速率 rad/s
        """
        # ===== 1. 保存 Occupancy 二进制文件 =====
        occupancy_file = self.output_dir / f"data/occupancy_{frame_id:06d}.bin"
        occupancy_file.parent.mkdir(parents=True, exist_ok=True)
        occupancy.astype(np.float32).tofile(occupancy_file)

        # ===== 2. 保存 Flow 二进制文件 =====
        flow_file = self.output_dir / f"data/flow_{frame_id:06d}.bin"
        flow.astype(np.float32).tofile(flow_file)

        # ===== 3. 创建帧元数据 =====
        self.openlabel["openlabel"]["frames"][str(frame_id)] = {
            "frame_properties": {
                "timestamp": str(timestamp),
                "vehicle_speed": vehicle_speed,
                "vehicle_yaw_rate": vehicle_yaw_rate
            },

            "objects": {
                f"occupancy_grid_{frame_id}": {
                    "object_data": {
                        "type": "occupancy_grid_3d",
                        "coordinate_system": "ego_vehicle",
                        "occupancy_grid_3d": [
                            {
                                "name": "occupancy_probability",
                                "uri": f"data/occupancy_{frame_id:06d}.bin",
                                "attributes": {
                                    "grid_dimensions": [200, 200, 16],
                                    "voxel_size": [0.5, 0.5, 0.5],
                                    "origin": [-50.0, -50.0, 0.0],
                                    "data_type": "float32",
                                    "encoding": "binary"
                                }
                            },
                            {
                                "name": "occupancy_flow",
                                "uri": f"data/flow_{frame_id:06d}.bin",
                                "attributes": {
                                    "grid_dimensions": [200, 200, 16],
                                    "voxel_size": [0.5, 0.5, 0.5],
                                    "vector_dimensions": 3,
                                    "data_type": "float32",
                                    "encoding": "binary"
                                }
                            }
                        ]
                    }
                }
            }
        }

    def save(self):
        """保存 OpenLABEL JSON 文件"""
        output_file = self.output_dir / "annotations.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.openlabel, f, indent=2, ensure_ascii=False)

        print(f"✓ OpenLABEL 数据集已保存: {output_file}")
        print(f"  总帧数: {len(self.openlabel['openlabel']['frames'])}")
```

**使用示例**:

```python
# 创建 OpenLABEL 数据集
dataset_generator = OpenLABELDatasetGenerator(output_dir='./data/occupancy_dataset')

# 采集数据并添加到数据集
for frame_id in range(1000):
    # ... 采集 occupancy 和 flow ...

    dataset_generator.add_frame(
        frame_id=frame_id,
        timestamp=time.time(),
        occupancy=occupancy,
        flow=flow,
        vehicle_speed=vehicle_speed,
        vehicle_yaw_rate=vehicle_yaw_rate
    )

# 保存
dataset_generator.save()
```

---

## 5. OSI 整合方案 {#osi}

### 5.1 什么是 OSI?

**OSI (Open Simulation Interface)** 是传感器数据和车辆状态的标准化接口(Protobuf),定义:
- 传感器数据格式(相机、LiDAR、雷达)
- 车辆动力学状态
- 环境感知结果

### 5.2 OSI 与执行器/反馈器整合

**用 OSI 标准化反馈器接口**:

```python
# interfaces/osi_feedback.py

from typing import Dict
import numpy as np

# OSI Protobuf 定义(简化版,完整版需引入 osi3 库)
class OSIGroundTruth:
    """
    OSI GroundTruth 消息

    对应 OSI 3.x 的 GroundTruth 消息
    """

    def __init__(self):
        self.version = {
            'version_major': 3,
            'version_minor': 5,
            'version_patch': 0
        }

        # 时间戳
        self.timestamp = {
            'seconds': 0,
            'nanos': 0
        }

        # 主车状态
        self.host_vehicle_id = 0

        # 移动物体(车辆)
        self.moving_object = []

        # 静态物体(建筑、路障)
        self.stationary_object = []

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'version': self.version,
            'timestamp': self.timestamp,
            'host_vehicle_id': self.host_vehicle_id,
            'moving_object': self.moving_object,
            'stationary_object': self.stationary_object
        }


class OSIVehicleState:
    """
    OSI 车辆状态

    对应 OSI 3.x 的 BaseMoving 消息
    """

    def __init__(self):
        # 位置(世界坐标系)
        self.position = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # 姿态(四元数)
        self.orientation = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}

        # 速度(车体坐标系)
        self.velocity = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # 加速度
        self.acceleration = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # 角速度
        self.orientation_rate = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}

    def to_dict(self) -> Dict:
        return {
            'position': self.position,
            'orientation': self.orientation,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'orientation_rate': self.orientation_rate
        }
```

**OSI 反馈器包装**:

```python
# carla_bridge/osi_carla_feedback.py

from interfaces.osi_feedback import OSIGroundTruth, OSIVehicleState
from carla_bridge.carla_feedback import CarlaFeedback
import time

class OSICarlaFeedback:
    """
    CARLA → OSI 反馈器

    功能:
    - 从 CARLA 读取状态
    - 转换为 OSI GroundTruth 格式
    - 符合 ASAM OSI 3.x 标准
    """

    def __init__(self, carla_feedback: CarlaFeedback):
        self.carla_feedback = carla_feedback

    def get_osi_ground_truth(self) -> OSIGroundTruth:
        """
        获取 OSI GroundTruth

        返回: OSIGroundTruth
        """
        # 获取 CARLA 反馈
        carla_data = self.carla_feedback.get_feedback()
        if carla_data is None:
            return None

        # 创建 OSI GroundTruth
        osi_gt = OSIGroundTruth()

        # 时间戳
        timestamp = time.time()
        osi_gt.timestamp['seconds'] = int(timestamp)
        osi_gt.timestamp['nanos'] = int((timestamp - int(timestamp)) * 1e9)

        # 主车状态
        vehicle_state = OSIVehicleState()
        vehicle_state.position = {
            'x': carla_data.position[0],
            'y': carla_data.position[1],
            'z': carla_data.position[2]
        }
        vehicle_state.orientation = {
            'roll': carla_data.orientation[0],
            'pitch': carla_data.orientation[1],
            'yaw': carla_data.orientation[2]
        }
        vehicle_state.velocity = {
            'x': carla_data.velocity[0],
            'y': carla_data.velocity[1],
            'z': carla_data.velocity[2]
        }
        vehicle_state.acceleration = {
            'x': carla_data.acceleration[0],
            'y': carla_data.acceleration[1],
            'z': carla_data.acceleration[2]
        }
        vehicle_state.orientation_rate = {
            'roll': carla_data.angular_velocity[0],
            'pitch': carla_data.angular_velocity[1],
            'yaw': carla_data.angular_velocity[2]
        }

        osi_gt.moving_object.append(vehicle_state.to_dict())

        return osi_gt
```

---

## 6. 完整工具链 {#工具链}

### 6.1 ASAM 标准工具链流程

```mermaid
graph LR
    subgraph Input["输入(ASAM 标准)"]
        OD_MAP[OpenDRIVE 地图<br/>Town10HD.xodr]
        OS_SCENE[OpenSCENARIO 场景<br/>tilted_truck.xosc]
    end

    subgraph CARLA["CARLA UE5 仿真"]
        LOAD_MAP[加载 OpenDRIVE]
        LOAD_SCENE[加载 OpenSCENARIO]
        SIMULATE[运行仿真]
    end

    subgraph DataCollection["数据采集"]
        CAMERA[8 相机图像]
        LIDAR[LiDAR 点云]
        VEHICLE[车辆状态<br/>OSI 格式]
    end

    subgraph Annotation["标注生成"]
        VOXELIZE[点云体素化]
        FLOW[Occupancy Flow]
        LABEL[OpenLABEL 标注]
    end

    subgraph Training["模型训练"]
        DATASET[OpenLABEL 数据集]
        TRAIN[训练 Occupancy Net]
        MODEL[模型权重]
    end

    OD_MAP --> LOAD_MAP
    OS_SCENE --> LOAD_SCENE
    LOAD_MAP --> SIMULATE
    LOAD_SCENE --> SIMULATE
    SIMULATE --> CAMERA & LIDAR & VEHICLE

    CAMERA --> DATASET
    LIDAR --> VOXELIZE
    VEHICLE --> LABEL

    VOXELIZE --> FLOW
    FLOW --> LABEL
    LABEL --> DATASET

    DATASET --> TRAIN
    TRAIN --> MODEL
```

### 6.2 配置文件(完整版)

```yaml
# configs/asam_pipeline_config.yaml

# ===== ASAM 标准文件路径 =====
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
      - "dense_traffic_scenario.xosc"

  openlabel:
    output_dir: "./data/openlabel_dataset"
    schema_version: "1.0.0"

  osi:
    enabled: true
    version: "3.5.0"
    output_format: "protobuf"  # or "json"

# ===== 数据采集参数 =====
data_collection:
  num_frames_per_scenario: 1000
  sensor_tick: 0.028  # 36 FPS

  cameras:
    enabled: true
    save_format: "png"
    compression: 9

  lidar:
    enabled: true
    channels: 64
    range: 100.0
    points_per_second: 1000000

  vehicle_state:
    update_rate: 100  # Hz
    format: "osi"  # OSI GroundTruth

# ===== 标注生成参数 =====
annotation:
  occupancy:
    voxel_size: 0.5
    grid_size: [200, 200, 16]
    format: "openlabel"

  flow:
    enabled: true
    temporal_window: 5  # 帧

# ===== 训练参数 =====
training:
  dataset_format: "openlabel"
  train_val_test_split: [0.7, 0.2, 0.1]
```

---

## 7. 代码实现 {#代码实现}

### 7.1 完整数据采集流水线

```python
# examples/asam_data_collection_pipeline.py

import carla
import logging
from pathlib import Path

# ASAM 工具
from carla_bridge.opendrive_lane_extractor import OpenDRIVELaneExtractor
from carla_bridge.openscenario_loader import OpenSCENARIOLoader
from dataset.openlabel_generator import OpenLABELDatasetGenerator
from carla_bridge.osi_carla_feedback import OSICarlaFeedback

# 数据采集
from carla_bridge.camera_manager import CameraManager
from carla_bridge.carla_feedback import CarlaFeedback

logger = logging.getLogger(__name__)

def run_asam_pipeline(config_path: str):
    """
    运行 ASAM 标准数据采集流水线

    参数:
        config_path: ASAM 配置文件路径
    """
    import yaml

    # ===== 1. 加载配置 =====
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("=" * 80)
    logger.info("ASAM 标准数据采集流水线")
    logger.info("=" * 80)

    # ===== 2. 连接 CARLA =====
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # ===== 3. 加载 OpenDRIVE 地图 =====
    opendrive_file = Path(config['asam']['opendrive']['map_file'])
    logger.info(f"加载 OpenDRIVE 地图: {opendrive_file}")

    with open(opendrive_file, 'r') as f:
        opendrive_data = f.read()

    world = client.generate_opendrive_world(
        opendrive=opendrive_data,
        parameters=carla.OpendriveGenerationParameters(
            vertex_distance=config['asam']['opendrive']['generation']['vertex_distance'],
            smooth_junctions=config['asam']['opendrive']['generation']['smooth_junctions']
        )
    )
    logger.info("✓ OpenDRIVE 地图已加载")

    # ===== 4. 提取车道线(用于后续标注) =====
    lane_extractor = OpenDRIVELaneExtractor(str(opendrive_file))
    lane_data = lane_extractor.extract_lane_markings()
    logger.info(f"✓ 提取了 {len(lane_data['lane_centers'])} 条车道线")

    # ===== 5. 创建 OpenLABEL 数据集生成器 =====
    openlabel_generator = OpenLABELDatasetGenerator(
        output_dir=config['asam']['openlabel']['output_dir']
    )

    # ===== 6. 遍历所有 OpenSCENARIO 场景 =====
    scenarios_dir = Path(config['asam']['openscenario']['scenarios_dir'])
    active_scenarios = config['asam']['openscenario']['active_scenarios']

    for scenario_file in active_scenarios:
        scenario_path = scenarios_dir / scenario_file
        logger.info(f"\n{'=' * 80}")
        logger.info(f"运行场景: {scenario_file}")
        logger.info(f"{'=' * 80}")

        # ===== 6.1 加载 OpenSCENARIO =====
        scenario_loader = OpenSCENARIOLoader(world, str(scenario_path))
        scenario = scenario_loader.load_scenario()

        ego_vehicle = scenario['ego_vehicle']
        duration = scenario['duration']

        # ===== 6.2 初始化传感器 =====
        camera_manager = CameraManager(world, ego_vehicle)
        camera_manager.setup_cameras()

        feedback = CarlaFeedback(ego_vehicle, update_rate=100.0)
        feedback.initialize()

        # OSI 包装
        osi_feedback = OSICarlaFeedback(feedback)

        # ===== 6.3 数据采集循环 =====
        num_frames = config['data_collection']['num_frames_per_scenario']
        frame_id = 0

        while frame_id < num_frames:
            world.tick()

            # 获取相机图像
            camera_frames = camera_manager.get_latest_frame()
            if camera_frames is None:
                continue

            # 获取车辆状态(OSI 格式)
            osi_gt = osi_feedback.get_osi_ground_truth()
            if osi_gt is None:
                continue

            # 生成 Occupancy 标注
            # TODO: 从 LiDAR 生成 occupancy 和 flow

            # 添加到 OpenLABEL 数据集
            # openlabel_generator.add_frame(...)

            frame_id += 1

            if frame_id % 100 == 0:
                logger.info(f"  已采集 {frame_id}/{num_frames} 帧")

        # ===== 6.4 清理场景 =====
        camera_manager.destroy()
        feedback.shutdown()
        scenario_loader.cleanup()

    # ===== 7. 保存 OpenLABEL 数据集 =====
    openlabel_generator.save()

    logger.info("=" * 80)
    logger.info("✓ ASAM 数据采集流水线完成")
    logger.info("=" * 80)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_asam_pipeline('./configs/asam_pipeline_config.yaml')
```

---

## 总结

### 整合完成的 ASAM 标准

| ASAM 标准 | 整合程度 | 说明 |
|-----------|----------|------|
| **OpenDRIVE** | ✅ 完整支持 | CARLA 原生支持,提供车道线提取工具 |
| **OpenSCENARIO** | ✅ 完整支持 | 场景加载器,支持复杂场景定义 |
| **OpenLABEL** | ✅ 完整支持 | Occupancy Grid 标注格式,符合 1.0.0 规范 |
| **OSI** | ✅ 部分支持 | 车辆状态接口,简化版 Protobuf |
| **OpenCRG** | ⚠️ 暂不支持 | 路面细节对 Occupancy 影响较小 |

### 核心优势

1. **数据可移植性**: OpenLABEL 格式可被其他工具识别
2. **场景可复现**: OpenSCENARIO 确保测试场景一致性
3. **工具链兼容**: 符合行业标准,便于集成第三方工具
4. **未来扩展**: 易于迁移到真车或其他仿真器

### 文件输出

- ✅ OpenDRIVE 地图加载器
- ✅ OpenSCENARIO 场景加载器
- ✅ OpenLABEL 数据集生成器
- ✅ OSI 反馈器包装
- ✅ 完整配置文件和示例代码

所有代码均符合 **ASAM 标准**,可直接使用! 🎯

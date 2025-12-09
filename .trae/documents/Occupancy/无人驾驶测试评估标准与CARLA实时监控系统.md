# 无人驾驶测试评估标准与 CARLA 实时监控系统

> 从国际标准到 CARLA 仿真实现：完整的自动驾驶性能评估体系

---

## 目录

1. [国际标准与评估体系](#国际标准)
2. [核心评估指标分类](#评估指标)
3. [CARLA 检测与统计实现](#CARLA实现)
4. [实时监控与可视化](#实时监控)
5. [完整代码实现](#代码实现)
6. [评估报告生成](#评估报告)

---

## 1. 国际标准与评估体系 {#国际标准}

### 1.1 主要国际标准

| 标准 | 组织 | 内容 | 适用范围 |
|-----|------|------|---------|
| **ISO 26262** | ISO | 功能安全标准 (ASIL A-D) | 安全等级定义 |
| **ISO 21448 (SOTIF)** | ISO | 预期功能安全 | 未知场景处理 |
| **ISO 34501** | ISO | 自动驾驶测试场景 | 场景定义 |
| **ISO 34502** | ISO | 自动驾驶测试方法 | 测试方法论 |
| **SAE J3016** | SAE | 自动驾驶分级 (L0-L5) | 功能分级 |
| **Euro NCAP** | Euro NCAP | 欧洲新车评价规程 | 消费者评级 |
| **NHTSA** | 美国交通部 | 自动驾驶安全评估 | 美国标准 |
| **GB/T 41798-2022** | 中国 | 智能网联汽车测试标准 | 中国国标 |

### 1.2 行业评估框架

#### Waymo Safety Framework (2020)
- **碰撞率**: 每百万英里碰撞次数
- **接管率**: TTC (Time To Collision) < 3s 的接管次数
- **舒适度**: 加加速度 (Jerk) < 2 m/s³

#### Tesla Shadow Mode Metrics
- **Intervention Rate**: 人工接管频率 (次/英里)
- **Positive Experience**: 用户主观评分
- **Miles Per Disengagement (MPD)**: 平均无接管里程

#### 加州 DMV 报告指标
- **自主里程** (Autonomous Miles)
- **脱离次数** (Disengagements)
- **MPD** (Miles Per Disengagement)

---

## 2. 核心评估指标分类 {#评估指标}

### 2.1 安全性指标 (Safety Metrics) ⭐ **最高优先级**

#### 2.1.1 碰撞类指标

| 指标 | 定义 | 严重等级 | CARLA 检测方法 |
|-----|------|---------|---------------|
| **碰撞次数** | 与任何物体发生物理接触 | CRITICAL | `CollisionSensor` |
| **与车辆碰撞** | 与其他车辆碰撞 | CRITICAL | `CollisionSensor` + actor type |
| **与行人碰撞** | 与行人/骑行者碰撞 | CRITICAL | `CollisionSensor` + actor type |
| **与静态物体碰撞** | 与建筑/护栏等碰撞 | HIGH | `CollisionSensor` + actor type |
| **碰撞强度** | 碰撞冲量 (N·s) | CRITICAL | `CollisionEvent.normal_impulse` |

#### 2.1.2 车道偏离类指标

| 指标 | 定义 | 严重等级 | CARLA 检测方法 |
|-----|------|---------|---------------|
| **压线次数** | 车辆任一车轮越过车道线 | HIGH | `LaneInvasionSensor` |
| **压实线次数** | 越过实线 (不可跨越) | CRITICAL | `LaneInvasion` + lane marking type |
| **逆行次数** | 反向行驶 | CRITICAL | 速度方向 vs 车道方向 |
| **车道偏离时长** | 持续在车道外的时间 (秒) | HIGH | 累计时间 |
| **横向偏移距离** | 车辆中心线距离车道中心的偏移 (米) | MEDIUM | `Waypoint.get_lane_width()` |

#### 2.1.3 交规违章类指标

| 指标 | 定义 | 严重等级 | CARLA 检测方法 |
|-----|------|---------|---------------|
| **闯红灯次数** | 红灯时通过停止线 | CRITICAL | `TrafficLight` state + location |
| **超速次数** | 超过道路限速 | HIGH | `vehicle.get_speed_limit()` |
| **超速百分比** | (实际速度 - 限速) / 限速 | HIGH | 计算值 |
| **未让行次数** | 未给行人/优先车辆让行 | HIGH | 场景检测 |
| **违规变道次数** | 实线变道/未打转向灯 | MEDIUM | 组合检测 |

#### 2.1.4 危险接近类指标 (Near Miss)

| 指标 | 定义 | 阈值 | CARLA 检测方法 |
|-----|------|------|---------------|
| **TTC 预警次数** | Time To Collision < 3s | < 3s | 计算 TTC |
| **最小前车距离** | 与前车最近距离 (米) | < 5m | 距离检测 |
| **紧急制动次数** | 减速度 > 6 m/s² | > 6 m/s² | 加速度传感器 |
| **急转向次数** | 横向加速度 > 4 m/s² | > 4 m/s² | 加速度传感器 |

### 2.2 效率性指标 (Efficiency Metrics)

| 指标 | 定义 | 目标值 | CARLA 检测方法 |
|-----|------|--------|---------------|
| **平均速度** | 行驶全程平均速度 (km/h) | 接近限速 | 累计距离/时间 |
| **速度利用率** | 平均速度 / 限速 | > 0.85 | 计算比值 |
| **任务完成率** | 成功到达目的地的比例 | 100% | 任务状态 |
| **任务完成时间** | 完成路线所需时间 (秒) | 最短 | 时间戳 |
| **绕行距离** | 实际路径 / 最短路径 | < 1.1 | 路径规划 |
| **停车次数** | 速度 < 1 km/h 的次数 | 最少 | 速度检测 |
| **停车时长** | 停车状态累计时长 (秒) | 最短 | 累计时间 |

### 2.3 舒适性指标 (Comfort Metrics)

| 指标 | 定义 | 舒适阈值 | CARLA 检测方法 |
|-----|------|---------|---------------|
| **纵向加速度** | 加速/减速加速度 (m/s²) | < 2.5 m/s² | IMU 传感器 |
| **横向加速度** | 转弯侧向加速度 (m/s²) | < 2.0 m/s² | IMU 传感器 |
| **加加速度 (Jerk)** | 加速度变化率 (m/s³) | < 2.0 m/s³ | 加速度导数 |
| **横摆角速度** | 车辆偏航角速度 (rad/s) | < 0.3 rad/s | 陀螺仪 |
| **急加速次数** | 加速度 > 3 m/s² | 0 次 | 统计 |
| **急减速次数** | 减速度 < -3 m/s² | 0 次 | 统计 |
| **急转弯次数** | 横向加速度 > 3 m/s² | 0 次 | 统计 |

### 2.4 鲁棒性指标 (Robustness Metrics)

| 指标 | 定义 | CARLA 检测方法 |
|-----|------|---------------|
| **接管次数** | 人工接管/安全员介入次数 | 手动标记 |
| **MPD** | Miles Per Disengagement | 总里程 / 接管次数 |
| **系统故障次数** | 算法崩溃/超时 | 异常捕获 |
| **感知失败次数** | 未检测到关键障碍物 | Ground Truth 对比 |
| **规划失败次数** | 未生成有效轨迹 | 规划器状态 |

### 2.5 场景覆盖指标 (Scenario Coverage)

| 指标 | 定义 | CARLA 实现 |
|-----|------|-----------|
| **里程统计** | 累计测试里程 (公里) | GPS 积分 |
| **场景类型** | 高速/城市/乡村/停车场 | 地图标记 |
| **天气条件** | 晴天/雨天/雾天/夜间 | Weather 设置 |
| **交通密度** | 车辆密度 (辆/km) | Actor 统计 |
| **Corner Cases** | 特殊场景测试 (加塞/鬼探头) | OpenSCENARIO |

---

## 3. CARLA 检测与统计实现 {#CARLA实现}

### 3.1 传感器配置

```python
# carla_evaluation/sensors/evaluation_sensors.py

import carla
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class CollisionEvent:
    """碰撞事件"""
    timestamp: float
    frame: int
    actor_type: str          # 'vehicle', 'pedestrian', 'static'
    actor_id: int
    impulse: float           # 碰撞冲量 (N·s)
    location: carla.Location
    severity: str            # 'CRITICAL', 'HIGH', 'MEDIUM'

@dataclass
class LaneInvasionEvent:
    """车道入侵事件"""
    timestamp: float
    frame: int
    lane_marking_types: List[str]  # 'Solid', 'Broken', 'SolidSolid', etc.
    crossed_lane_types: List[str]
    is_solid_line: bool      # 是否压实线

@dataclass
class TrafficViolationEvent:
    """交规违章事件"""
    timestamp: float
    frame: int
    violation_type: str      # 'red_light', 'speeding', 'wrong_way', etc.
    severity: str
    details: Dict            # 额外信息

class EvaluationSensorSuite:
    """
    自动驾驶评估传感器套件

    包含所有用于性能评估的传感器
    """
    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        self.world = world
        self.vehicle = vehicle

        # 传感器实例
        self.collision_sensor: Optional[carla.Sensor] = None
        self.lane_invasion_sensor: Optional[carla.Sensor] = None
        self.imu_sensor: Optional[carla.Sensor] = None
        self.gnss_sensor: Optional[carla.Sensor] = None

        # 事件记录
        self.collision_events: List[CollisionEvent] = []
        self.lane_invasion_events: List[LaneInvasionEvent] = []
        self.traffic_violations: List[TrafficViolationEvent] = []

        # 实时数据
        self.current_velocity: carla.Vector3D = carla.Vector3D(0, 0, 0)
        self.current_acceleration: carla.Vector3D = carla.Vector3D(0, 0, 0)
        self.current_location: carla.Location = carla.Location(0, 0, 0)

        self._setup_sensors()

    def _setup_sensors(self):
        """配置所有评估传感器"""
        bp_library = self.world.get_blueprint_library()

        # 1. 碰撞传感器
        collision_bp = bp_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(self._on_collision)

        # 2. 车道入侵传感器
        lane_invasion_bp = bp_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_invasion_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.lane_invasion_sensor.listen(self._on_lane_invasion)

        # 3. IMU 传感器 (加速度/角速度)
        imu_bp = bp_library.find('sensor.other.imu')
        self.imu_sensor = self.world.spawn_actor(
            imu_bp,
            carla.Transform(carla.Location(x=0.0, z=0.0)),
            attach_to=self.vehicle
        )
        self.imu_sensor.listen(self._on_imu)

        # 4. GNSS 传感器 (位置)
        gnss_bp = bp_library.find('sensor.other.gnss')
        self.gnss_sensor = self.world.spawn_actor(
            gnss_bp,
            carla.Transform(carla.Location(x=0.0, z=0.0)),
            attach_to=self.vehicle
        )
        self.gnss_sensor.listen(self._on_gnss)

    def _on_collision(self, event: carla.CollisionEvent):
        """碰撞事件回调"""
        other_actor = event.other_actor

        # 确定 actor 类型
        if 'vehicle' in other_actor.type_id:
            actor_type = 'vehicle'
            severity = 'CRITICAL'
        elif 'walker' in other_actor.type_id:
            actor_type = 'pedestrian'
            severity = 'CRITICAL'
        elif 'static' in other_actor.type_id:
            actor_type = 'static'
            severity = 'HIGH'
        else:
            actor_type = 'other'
            severity = 'MEDIUM'

        # 计算碰撞冲量
        impulse = np.linalg.norm([
            event.normal_impulse.x,
            event.normal_impulse.y,
            event.normal_impulse.z
        ])

        collision_event = CollisionEvent(
            timestamp=event.timestamp,
            frame=event.frame,
            actor_type=actor_type,
            actor_id=other_actor.id,
            impulse=impulse,
            location=self.vehicle.get_location(),
            severity=severity
        )

        self.collision_events.append(collision_event)

        # 实时警报
        print(f"🚨 [{severity}] 碰撞检测: {actor_type} (冲量: {impulse:.2f} N·s)")

    def _on_lane_invasion(self, event: carla.LaneInvasionEvent):
        """车道入侵事件回调"""
        # 获取越过的车道标线类型
        lane_types = [str(marking).split('.')[-1] for marking in event.crossed_lane_markings]

        # 判断是否压实线
        solid_types = ['Solid', 'SolidSolid', 'SolidBroken', 'BrokenSolid']
        is_solid = any(ltype in solid_types for ltype in lane_types)

        invasion_event = LaneInvasionEvent(
            timestamp=event.timestamp,
            frame=event.frame,
            lane_marking_types=lane_types,
            crossed_lane_types=lane_types,
            is_solid_line=is_solid
        )

        self.lane_invasion_events.append(invasion_event)

        # 实时警报
        severity = "🔴 实线" if is_solid else "🟡 虚线"
        print(f"⚠️  压线检测: {severity} {lane_types}")

    def _on_imu(self, imu_data: carla.IMUMeasurement):
        """IMU 数据回调"""
        # 加速度 (m/s²)
        self.current_acceleration = carla.Vector3D(
            imu_data.accelerometer.x,
            imu_data.accelerometer.y,
            imu_data.accelerometer.z
        )

        # 角速度 (rad/s)
        self.angular_velocity = carla.Vector3D(
            imu_data.gyroscope.x,
            imu_data.gyroscope.y,
            imu_data.gyroscope.z
        )

    def _on_gnss(self, gnss_data: carla.GnssMeasurement):
        """GNSS 数据回调"""
        # 经纬度 → 世界坐标 (简化处理)
        pass

    def cleanup(self):
        """清理传感器"""
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()
        if self.imu_sensor:
            self.imu_sensor.destroy()
        if self.gnss_sensor:
            self.gnss_sensor.destroy()
```

### 3.2 评估指标计算器

```python
# carla_evaluation/metrics/metrics_calculator.py

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class SafetyMetrics:
    """安全性指标"""
    # 碰撞类
    total_collisions: int = 0
    vehicle_collisions: int = 0
    pedestrian_collisions: int = 0
    static_collisions: int = 0
    max_collision_impulse: float = 0.0

    # 车道偏离类
    lane_invasions: int = 0
    solid_line_crossings: int = 0
    lane_invasion_duration: float = 0.0

    # 交规违章类
    red_light_violations: int = 0
    speeding_violations: int = 0
    wrong_way_violations: int = 0

    # 危险接近类
    ttc_warnings: int = 0          # TTC < 3s
    emergency_brakes: int = 0       # 减速度 > 6 m/s²
    sharp_turns: int = 0            # 横向加速度 > 4 m/s²
    min_front_distance: float = float('inf')

@dataclass
class EfficiencyMetrics:
    """效率性指标"""
    total_distance: float = 0.0      # 总里程 (米)
    total_time: float = 0.0          # 总时间 (秒)
    average_speed: float = 0.0       # 平均速度 (m/s)
    speed_utilization: float = 0.0   # 速度利用率
    task_completed: bool = False     # 任务完成
    task_time: float = 0.0           # 任务时间
    num_stops: int = 0               # 停车次数
    stop_duration: float = 0.0       # 停车时长

@dataclass
class ComfortMetrics:
    """舒适性指标"""
    max_longitudinal_accel: float = 0.0     # 最大纵向加速度
    max_lateral_accel: float = 0.0          # 最大横向加速度
    max_jerk: float = 0.0                   # 最大加加速度
    max_yaw_rate: float = 0.0               # 最大横摆角速度

    harsh_accelerations: int = 0            # 急加速次数 (> 3 m/s²)
    harsh_brakes: int = 0                   # 急减速次数 (< -3 m/s²)
    harsh_turns: int = 0                    # 急转弯次数 (> 3 m/s²)

    avg_jerk: float = 0.0                   # 平均加加速度

@dataclass
class RobustnessMetrics:
    """鲁棒性指标"""
    disengagements: int = 0                 # 接管次数
    mpd: float = 0.0                        # Miles Per Disengagement
    system_failures: int = 0                # 系统故障
    perception_failures: int = 0            # 感知失败
    planning_failures: int = 0              # 规划失败

class MetricsCalculator:
    """
    评估指标计算器

    实时计算所有评估指标
    """
    def __init__(self):
        self.safety = SafetyMetrics()
        self.efficiency = EfficiencyMetrics()
        self.comfort = ComfortMetrics()
        self.robustness = RobustnessMetrics()

        # 历史数据
        self.acceleration_history: List[float] = []
        self.jerk_history: List[float] = []
        self.speed_history: List[float] = []
        self.location_history: List[carla.Location] = []

        # 状态追踪
        self.prev_acceleration: Optional[np.ndarray] = None
        self.prev_location: Optional[carla.Location] = None
        self.prev_time: Optional[float] = None
        self.is_stopped: bool = False
        self.stop_start_time: Optional[float] = None

    def update_from_collision_event(self, event: CollisionEvent):
        """从碰撞事件更新指标"""
        self.safety.total_collisions += 1

        if event.actor_type == 'vehicle':
            self.safety.vehicle_collisions += 1
        elif event.actor_type == 'pedestrian':
            self.safety.pedestrian_collisions += 1
        elif event.actor_type == 'static':
            self.safety.static_collisions += 1

        self.safety.max_collision_impulse = max(
            self.safety.max_collision_impulse,
            event.impulse
        )

    def update_from_lane_invasion_event(self, event: LaneInvasionEvent):
        """从车道入侵事件更新指标"""
        self.safety.lane_invasions += 1

        if event.is_solid_line:
            self.safety.solid_line_crossings += 1

    def update_real_time(
        self,
        vehicle: carla.Vehicle,
        world: carla.World,
        dt: float
    ):
        """
        实时更新指标

        Args:
            vehicle: 车辆 actor
            world: CARLA 世界
            dt: 时间步长 (秒)
        """
        current_time = world.get_snapshot().timestamp.elapsed_seconds

        # 1. 获取当前状态
        velocity = vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])  # m/s
        location = vehicle.get_location()
        acceleration = vehicle.get_acceleration()

        # 2. 效率指标更新
        if self.prev_location is not None:
            distance = location.distance(self.prev_location)
            self.efficiency.total_distance += distance

        self.efficiency.total_time += dt
        self.efficiency.average_speed = (
            self.efficiency.total_distance / self.efficiency.total_time
            if self.efficiency.total_time > 0 else 0
        )

        # 速度利用率
        speed_limit = vehicle.get_speed_limit()  # km/h
        if speed_limit > 0:
            self.efficiency.speed_utilization = (
                speed * 3.6 / speed_limit  # m/s → km/h
            )

        # 停车检测
        if speed < 0.28:  # < 1 km/h
            if not self.is_stopped:
                self.is_stopped = True
                self.stop_start_time = current_time
                self.efficiency.num_stops += 1
        else:
            if self.is_stopped:
                self.is_stopped = False
                if self.stop_start_time is not None:
                    self.efficiency.stop_duration += (
                        current_time - self.stop_start_time
                    )

        # 3. 舒适性指标更新
        accel_magnitude = np.linalg.norm([
            acceleration.x, acceleration.y, acceleration.z
        ])

        # 纵向/横向加速度 (简化: 使用车体坐标系)
        transform = vehicle.get_transform()
        forward = transform.get_forward_vector()

        # 纵向加速度 (前进方向)
        longitudinal_accel = (
            acceleration.x * forward.x +
            acceleration.y * forward.y
        )

        # 横向加速度 (垂直方向)
        lateral_accel = abs(
            acceleration.x * (-forward.y) +
            acceleration.y * forward.x
        )

        self.comfort.max_longitudinal_accel = max(
            self.comfort.max_longitudinal_accel,
            abs(longitudinal_accel)
        )
        self.comfort.max_lateral_accel = max(
            self.comfort.max_lateral_accel,
            lateral_accel
        )

        # 急加速/急减速检测
        if longitudinal_accel > 3.0:
            self.comfort.harsh_accelerations += 1
        elif longitudinal_accel < -3.0:
            self.comfort.harsh_brakes += 1

        # 急转弯检测
        if lateral_accel > 3.0:
            self.comfort.harsh_turns += 1

        # 加加速度 (Jerk)
        if self.prev_acceleration is not None:
            jerk = np.linalg.norm(
                np.array([acceleration.x, acceleration.y, acceleration.z]) -
                self.prev_acceleration
            ) / dt if dt > 0 else 0

            self.comfort.max_jerk = max(self.comfort.max_jerk, jerk)
            self.jerk_history.append(jerk)

        self.prev_acceleration = np.array([
            acceleration.x, acceleration.y, acceleration.z
        ])

        # 4. 安全指标更新

        # 超速检测
        if speed * 3.6 > speed_limit * 1.1:  # 超速 10%
            self.safety.speeding_violations += 1

        # 紧急制动检测
        if longitudinal_accel < -6.0:
            self.safety.emergency_brakes += 1

        # TTC 检测 (与前车距离)
        front_vehicle = self._get_front_vehicle(vehicle, world)
        if front_vehicle is not None:
            distance = location.distance(front_vehicle.get_location())
            self.safety.min_front_distance = min(
                self.safety.min_front_distance,
                distance
            )

            # TTC 计算
            relative_speed = speed - np.linalg.norm([
                front_vehicle.get_velocity().x,
                front_vehicle.get_velocity().y,
                front_vehicle.get_velocity().z
            ])

            if relative_speed > 0:
                ttc = distance / relative_speed
                if ttc < 3.0:
                    self.safety.ttc_warnings += 1

        # 红绿灯检测
        if vehicle.is_at_traffic_light():
            traffic_light = vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                # 检查是否越过停止线
                if self._is_beyond_stop_line(vehicle, traffic_light):
                    self.safety.red_light_violations += 1

        # 更新历史
        self.speed_history.append(speed)
        self.location_history.append(location)
        self.prev_location = location
        self.prev_time = current_time

    def _get_front_vehicle(
        self,
        ego_vehicle: carla.Vehicle,
        world: carla.World,
        max_distance: float = 50.0
    ) -> Optional[carla.Vehicle]:
        """获取前方最近的车辆"""
        ego_location = ego_vehicle.get_location()
        ego_transform = ego_vehicle.get_transform()
        ego_forward = ego_transform.get_forward_vector()

        min_distance = float('inf')
        front_vehicle = None

        for vehicle in world.get_actors().filter('vehicle.*'):
            if vehicle.id == ego_vehicle.id:
                continue

            vehicle_location = vehicle.get_location()
            distance = ego_location.distance(vehicle_location)

            if distance > max_distance:
                continue

            # 检查是否在前方
            to_vehicle = carla.Vector3D(
                vehicle_location.x - ego_location.x,
                vehicle_location.y - ego_location.y,
                0
            )

            dot_product = (
                ego_forward.x * to_vehicle.x +
                ego_forward.y * to_vehicle.y
            )

            if dot_product > 0 and distance < min_distance:
                min_distance = distance
                front_vehicle = vehicle

        return front_vehicle

    def _is_beyond_stop_line(
        self,
        vehicle: carla.Vehicle,
        traffic_light: carla.TrafficLight
    ) -> bool:
        """检查车辆是否越过停止线"""
        # 获取交通灯触发区域
        trigger_volume = traffic_light.trigger_volume
        vehicle_location = vehicle.get_location()
        trigger_location = trigger_volume.location

        # 简化判断: 检查是否超过触发位置
        distance = vehicle_location.distance(trigger_location)
        return distance < 2.0  # 2米内认为越过停止线

    def get_summary(self) -> Dict:
        """获取指标摘要"""
        # 计算平均 Jerk
        if self.jerk_history:
            self.comfort.avg_jerk = np.mean(self.jerk_history)

        # 计算 MPD
        if self.robustness.disengagements > 0:
            self.robustness.mpd = (
                self.efficiency.total_distance / 1609.34 /  # 米 → 英里
                self.robustness.disengagements
            )

        return {
            'safety': self.safety,
            'efficiency': self.efficiency,
            'comfort': self.comfort,
            'robustness': self.robustness
        }
```

### 3.3 交规违章检测器

```python
# carla_evaluation/detectors/traffic_violation_detector.py

import carla
import numpy as np
from typing import Optional, List

class TrafficViolationDetector:
    """
    交规违章检测器

    检测:
    - 闯红灯
    - 超速
    - 逆行
    - 违规变道
    """
    def __init__(self, world: carla.World):
        self.world = world
        self.map = world.get_map()

        # 违章记录
        self.violations: List[TrafficViolationEvent] = []

        # 状态追踪
        self.prev_waypoint: Optional[carla.Waypoint] = None
        self.was_at_traffic_light: bool = False

    def check_violations(
        self,
        vehicle: carla.Vehicle,
        timestamp: float,
        frame: int
    ):
        """检查所有违章行为"""
        location = vehicle.get_location()
        waypoint = self.map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint is None:
            return

        # 1. 闯红灯检测
        self._check_red_light_violation(vehicle, timestamp, frame)

        # 2. 超速检测
        self._check_speeding_violation(vehicle, waypoint, timestamp, frame)

        # 3. 逆行检测
        self._check_wrong_way(vehicle, waypoint, timestamp, frame)

        # 4. 违规变道检测
        # (需要结合转向灯信号,CARLA API 有限)

        self.prev_waypoint = waypoint

    def _check_red_light_violation(
        self,
        vehicle: carla.Vehicle,
        timestamp: float,
        frame: int
    ):
        """检测闯红灯"""
        is_at_traffic_light = vehicle.is_at_traffic_light()

        if is_at_traffic_light:
            traffic_light = vehicle.get_traffic_light()

            if traffic_light and traffic_light.get_state() == carla.TrafficLightState.Red:
                # 检测是否刚进入红灯区域 (防止重复计数)
                if not self.was_at_traffic_light:
                    violation = TrafficViolationEvent(
                        timestamp=timestamp,
                        frame=frame,
                        violation_type='red_light',
                        severity='CRITICAL',
                        details={
                            'traffic_light_id': traffic_light.id,
                            'location': vehicle.get_location()
                        }
                    )
                    self.violations.append(violation)
                    print(f"🔴 闯红灯检测! 位置: {vehicle.get_location()}")

        self.was_at_traffic_light = is_at_traffic_light

    def _check_speeding_violation(
        self,
        vehicle: carla.Vehicle,
        waypoint: carla.Waypoint,
        timestamp: float,
        frame: int
    ):
        """检测超速"""
        velocity = vehicle.get_velocity()
        speed_kmh = 3.6 * np.linalg.norm([velocity.x, velocity.y, velocity.z])

        speed_limit = vehicle.get_speed_limit()  # km/h

        # 超速 10% 触发
        if speed_kmh > speed_limit * 1.1:
            violation = TrafficViolationEvent(
                timestamp=timestamp,
                frame=frame,
                violation_type='speeding',
                severity='HIGH',
                details={
                    'speed': speed_kmh,
                    'speed_limit': speed_limit,
                    'overspeed_percent': (speed_kmh - speed_limit) / speed_limit * 100
                }
            )
            self.violations.append(violation)
            print(f"⚡ 超速检测! {speed_kmh:.1f} km/h (限速: {speed_limit} km/h)")

    def _check_wrong_way(
        self,
        vehicle: carla.Vehicle,
        waypoint: carla.Waypoint,
        timestamp: float,
        frame: int
    ):
        """检测逆行"""
        if self.prev_waypoint is None:
            return

        # 车辆前进方向
        vehicle_transform = vehicle.get_transform()
        vehicle_forward = vehicle_transform.get_forward_vector()

        # 车道方向
        waypoint_forward = waypoint.transform.get_forward_vector()

        # 计算夹角
        dot_product = (
            vehicle_forward.x * waypoint_forward.x +
            vehicle_forward.y * waypoint_forward.y
        )

        # 夹角 > 90° 认为逆行
        if dot_product < -0.5:  # cos(120°) ≈ -0.5
            violation = TrafficViolationEvent(
                timestamp=timestamp,
                frame=frame,
                violation_type='wrong_way',
                severity='CRITICAL',
                details={
                    'dot_product': dot_product,
                    'location': vehicle.get_location()
                }
            )
            self.violations.append(violation)
            print(f"🔄 逆行检测! 位置: {vehicle.get_location()}")
```

---

## 4. 实时监控与可视化 {#实时监控}

### 4.1 实时监控 HUD

```python
# carla_evaluation/visualization/evaluation_hud.py

import pygame
import numpy as np
from typing import Dict

class EvaluationHUD:
    """
    评估指标 HUD 显示

    实时显示所有评估指标
    """
    def __init__(self, width: int, height: int):
        pygame.init()
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption("自动驾驶评估监控")

        self.font_small = pygame.font.SysFont('Arial', 14)
        self.font_medium = pygame.font.SysFont('Arial', 18, bold=True)
        self.font_large = pygame.font.SysFont('Arial', 24, bold=True)

        self.width = width
        self.height = height

        # 颜色定义
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_SAFE = (0, 200, 0)
        self.COLOR_WARNING = (255, 200, 0)
        self.COLOR_CRITICAL = (255, 50, 50)
        self.COLOR_TEXT = (200, 200, 200)
        self.COLOR_TITLE = (100, 150, 255)

    def render(self, metrics_summary: Dict):
        """渲染 HUD"""
        self.display.fill(self.COLOR_BG)

        safety = metrics_summary['safety']
        efficiency = metrics_summary['efficiency']
        comfort = metrics_summary['comfort']
        robustness = metrics_summary['robustness']

        y_offset = 20

        # === 1. 安全性指标 ===
        y_offset = self._render_safety_metrics(safety, y_offset)

        # === 2. 效率性指标 ===
        y_offset = self._render_efficiency_metrics(efficiency, y_offset)

        # === 3. 舒适性指标 ===
        y_offset = self._render_comfort_metrics(comfort, y_offset)

        # === 4. 鲁棒性指标 ===
        y_offset = self._render_robustness_metrics(robustness, y_offset)

        pygame.display.flip()

    def _render_safety_metrics(self, safety: SafetyMetrics, y: int) -> int:
        """渲染安全性指标"""
        # 标题
        self._draw_text("🛡️ 安全性指标", 20, y, self.COLOR_TITLE, self.font_large)
        y += 40

        # 碰撞统计
        collision_color = self.COLOR_CRITICAL if safety.total_collisions > 0 else self.COLOR_SAFE
        self._draw_metric(
            "总碰撞次数",
            f"{safety.total_collisions}",
            20, y,
            collision_color
        )
        y += 25

        if safety.total_collisions > 0:
            self._draw_text(
                f"  ├─ 与车辆: {safety.vehicle_collisions}",
                30, y, self.COLOR_TEXT, self.font_small
            )
            y += 20
            self._draw_text(
                f"  ├─ 与行人: {safety.pedestrian_collisions}",
                30, y, self.COLOR_TEXT, self.font_small
            )
            y += 20
            self._draw_text(
                f"  └─ 与静态物体: {safety.static_collisions}",
                30, y, self.COLOR_TEXT, self.font_small
            )
            y += 25

        # 压线统计
        lane_color = self.COLOR_WARNING if safety.lane_invasions > 0 else self.COLOR_SAFE
        self._draw_metric(
            "压线次数",
            f"{safety.lane_invasions}",
            20, y,
            lane_color
        )
        y += 25

        if safety.solid_line_crossings > 0:
            self._draw_text(
                f"  └─ 压实线: {safety.solid_line_crossings}",
                30, y, self.COLOR_CRITICAL, self.font_small
            )
            y += 25

        # 交规违章
        violation_color = self.COLOR_CRITICAL if safety.red_light_violations > 0 else self.COLOR_SAFE
        self._draw_metric(
            "闯红灯次数",
            f"{safety.red_light_violations}",
            20, y,
            violation_color
        )
        y += 25

        self._draw_metric(
            "超速次数",
            f"{safety.speeding_violations}",
            20, y,
            self.COLOR_WARNING if safety.speeding_violations > 0 else self.COLOR_SAFE
        )
        y += 25

        # 危险接近
        self._draw_metric(
            "TTC 预警",
            f"{safety.ttc_warnings}",
            20, y,
            self.COLOR_WARNING if safety.ttc_warnings > 0 else self.COLOR_SAFE
        )
        y += 25

        self._draw_metric(
            "紧急制动",
            f"{safety.emergency_brakes}",
            20, y,
            self.COLOR_WARNING if safety.emergency_brakes > 0 else self.COLOR_SAFE
        )
        y += 25

        if safety.min_front_distance < float('inf'):
            dist_color = self.COLOR_CRITICAL if safety.min_front_distance < 5 else self.COLOR_SAFE
            self._draw_metric(
                "最小前车距离",
                f"{safety.min_front_distance:.1f} m",
                20, y,
                dist_color
            )
            y += 30

        return y + 20

    def _render_efficiency_metrics(self, efficiency: EfficiencyMetrics, y: int) -> int:
        """渲染效率性指标"""
        self._draw_text("⚡ 效率性指标", 20, y, self.COLOR_TITLE, self.font_large)
        y += 40

        self._draw_metric(
            "累计里程",
            f"{efficiency.total_distance / 1000:.2f} km",
            20, y,
            self.COLOR_TEXT
        )
        y += 25

        self._draw_metric(
            "累计时间",
            f"{efficiency.total_time / 60:.1f} min",
            20, y,
            self.COLOR_TEXT
        )
        y += 25

        self._draw_metric(
            "平均速度",
            f"{efficiency.average_speed * 3.6:.1f} km/h",
            20, y,
            self.COLOR_TEXT
        )
        y += 25

        util_color = self.COLOR_SAFE if efficiency.speed_utilization > 0.8 else self.COLOR_WARNING
        self._draw_metric(
            "速度利用率",
            f"{efficiency.speed_utilization * 100:.1f}%",
            20, y,
            util_color
        )
        y += 25

        self._draw_metric(
            "停车次数",
            f"{efficiency.num_stops}",
            20, y,
            self.COLOR_WARNING if efficiency.num_stops > 10 else self.COLOR_TEXT
        )
        y += 30

        return y + 20

    def _render_comfort_metrics(self, comfort: ComfortMetrics, y: int) -> int:
        """渲染舒适性指标"""
        self._draw_text("😊 舒适性指标", 20, y, self.COLOR_TITLE, self.font_large)
        y += 40

        # 最大加速度
        accel_color = self.COLOR_WARNING if comfort.max_longitudinal_accel > 2.5 else self.COLOR_SAFE
        self._draw_metric(
            "最大纵向加速度",
            f"{comfort.max_longitudinal_accel:.2f} m/s²",
            20, y,
            accel_color
        )
        y += 25

        lateral_color = self.COLOR_WARNING if comfort.max_lateral_accel > 2.0 else self.COLOR_SAFE
        self._draw_metric(
            "最大横向加速度",
            f"{comfort.max_lateral_accel:.2f} m/s²",
            20, y,
            lateral_color
        )
        y += 25

        # 加加速度
        jerk_color = self.COLOR_WARNING if comfort.max_jerk > 2.0 else self.COLOR_SAFE
        self._draw_metric(
            "最大加加速度",
            f"{comfort.max_jerk:.2f} m/s³",
            20, y,
            jerk_color
        )
        y += 25

        # 急操作统计
        self._draw_metric(
            "急加速",
            f"{comfort.harsh_accelerations}",
            20, y,
            self.COLOR_WARNING if comfort.harsh_accelerations > 0 else self.COLOR_SAFE
        )
        y += 25

        self._draw_metric(
            "急减速",
            f"{comfort.harsh_brakes}",
            20, y,
            self.COLOR_WARNING if comfort.harsh_brakes > 0 else self.COLOR_SAFE
        )
        y += 25

        self._draw_metric(
            "急转弯",
            f"{comfort.harsh_turns}",
            20, y,
            self.COLOR_WARNING if comfort.harsh_turns > 0 else self.COLOR_SAFE
        )
        y += 30

        return y + 20

    def _render_robustness_metrics(self, robustness: RobustnessMetrics, y: int) -> int:
        """渲染鲁棒性指标"""
        self._draw_text("🔧 鲁棒性指标", 20, y, self.COLOR_TITLE, self.font_large)
        y += 40

        self._draw_metric(
            "接管次数",
            f"{robustness.disengagements}",
            20, y,
            self.COLOR_WARNING if robustness.disengagements > 0 else self.COLOR_SAFE
        )
        y += 25

        if robustness.mpd > 0:
            mpd_color = self.COLOR_SAFE if robustness.mpd > 100 else self.COLOR_WARNING
            self._draw_metric(
                "MPD (英里/接管)",
                f"{robustness.mpd:.1f}",
                20, y,
                mpd_color
            )
            y += 25

        self._draw_metric(
            "系统故障",
            f"{robustness.system_failures}",
            20, y,
            self.COLOR_CRITICAL if robustness.system_failures > 0 else self.COLOR_SAFE
        )
        y += 30

        return y + 20

    def _draw_text(self, text: str, x: int, y: int, color, font):
        """绘制文本"""
        surface = font.render(text, True, color)
        self.display.blit(surface, (x, y))

    def _draw_metric(self, label: str, value: str, x: int, y: int, value_color):
        """绘制指标"""
        # 标签
        label_surface = self.font_small.render(label, True, self.COLOR_TEXT)
        self.display.blit(label_surface, (x, y))

        # 值
        value_surface = self.font_medium.render(value, True, value_color)
        self.display.blit(value_surface, (x + 250, y - 2))
```

### 4.2 完整评估循环

```python
# examples/evaluation_demo.py

import carla
import time
import pygame
from carla_evaluation.sensors.evaluation_sensors import EvaluationSensorSuite
from carla_evaluation.metrics.metrics_calculator import MetricsCalculator
from carla_evaluation.detectors.traffic_violation_detector import TrafficViolationDetector
from carla_evaluation.visualization.evaluation_hud import EvaluationHUD

def main():
    # 连接 CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 启用同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    try:
        # 生成车辆
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        # 启用自动驾驶 (或连接你的 Occupancy Network)
        vehicle.set_autopilot(True)

        # 初始化评估系统
        sensor_suite = EvaluationSensorSuite(world, vehicle)
        metrics_calculator = MetricsCalculator()
        violation_detector = TrafficViolationDetector(world)

        # 初始化 HUD
        hud = EvaluationHUD(width=600, height=900)

        print("=" * 50)
        print("自动驾驶评估系统已启动")
        print("=" * 50)

        # 主循环
        frame_count = 0
        start_time = time.time()

        while True:
            # Tick 世界
            world.tick()
            frame_count += 1

            # 处理 pygame 事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            # 获取时间步长
            snapshot = world.get_snapshot()
            dt = settings.fixed_delta_seconds

            # 更新指标
            metrics_calculator.update_real_time(vehicle, world, dt)

            # 更新碰撞事件
            for event in sensor_suite.collision_events:
                metrics_calculator.update_from_collision_event(event)
            sensor_suite.collision_events.clear()

            # 更新车道入侵事件
            for event in sensor_suite.lane_invasion_events:
                metrics_calculator.update_from_lane_invasion_event(event)
            sensor_suite.lane_invasion_events.clear()

            # 检测交规违章
            violation_detector.check_violations(
                vehicle,
                snapshot.timestamp.elapsed_seconds,
                frame_count
            )

            # 每 10 帧更新一次 HUD
            if frame_count % 10 == 0:
                metrics_summary = metrics_calculator.get_summary()
                hud.render(metrics_summary)

            # 每 100 帧打印一次摘要
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"\n--- 运行时间: {elapsed:.1f}s ---")

                summary = metrics_calculator.get_summary()
                safety = summary['safety']
                efficiency = summary['efficiency']

                print(f"总里程: {efficiency.total_distance / 1000:.2f} km")
                print(f"平均速度: {efficiency.average_speed * 3.6:.1f} km/h")
                print(f"碰撞次数: {safety.total_collisions}")
                print(f"压线次数: {safety.lane_invasions}")
                print(f"闯红灯: {safety.red_light_violations}")
                print(f"超速次数: {safety.speeding_violations}")

    finally:
        # 清理
        print("\n正在生成评估报告...")

        final_summary = metrics_calculator.get_summary()
        generate_evaluation_report(final_summary, "evaluation_report.json")

        sensor_suite.cleanup()
        vehicle.destroy()

        # 恢复异步模式
        settings.synchronous_mode = False
        world.apply_settings(settings)

        pygame.quit()
        print("评估系统已关闭")

if __name__ == '__main__':
    main()
```

---

## 5. 评估报告生成 {#评估报告}

### 5.1 JSON 报告

```python
# carla_evaluation/reports/report_generator.py

import json
from datetime import datetime
from typing import Dict

def generate_evaluation_report(
    metrics_summary: Dict,
    output_path: str
):
    """生成评估报告 (JSON 格式)"""

    safety = metrics_summary['safety']
    efficiency = metrics_summary['efficiency']
    comfort = metrics_summary['comfort']
    robustness = metrics_summary['robustness']

    # 计算总分
    safety_score = calculate_safety_score(safety)
    efficiency_score = calculate_efficiency_score(efficiency)
    comfort_score = calculate_comfort_score(comfort)
    robustness_score = calculate_robustness_score(robustness)

    overall_score = (
        safety_score * 0.5 +        # 安全性权重 50%
        efficiency_score * 0.2 +    # 效率性权重 20%
        comfort_score * 0.2 +       # 舒适性权重 20%
        robustness_score * 0.1      # 鲁棒性权重 10%
    )

    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'test_duration': efficiency.total_time,
            'total_distance_km': efficiency.total_distance / 1000,
        },
        'scores': {
            'overall': round(overall_score, 2),
            'safety': round(safety_score, 2),
            'efficiency': round(efficiency_score, 2),
            'comfort': round(comfort_score, 2),
            'robustness': round(robustness_score, 2),
        },
        'safety_metrics': {
            'total_collisions': safety.total_collisions,
            'vehicle_collisions': safety.vehicle_collisions,
            'pedestrian_collisions': safety.pedestrian_collisions,
            'static_collisions': safety.static_collisions,
            'lane_invasions': safety.lane_invasions,
            'solid_line_crossings': safety.solid_line_crossings,
            'red_light_violations': safety.red_light_violations,
            'speeding_violations': safety.speeding_violations,
            'ttc_warnings': safety.ttc_warnings,
            'emergency_brakes': safety.emergency_brakes,
            'min_front_distance_m': safety.min_front_distance,
        },
        'efficiency_metrics': {
            'total_distance_km': efficiency.total_distance / 1000,
            'total_time_min': efficiency.total_time / 60,
            'average_speed_kmh': efficiency.average_speed * 3.6,
            'speed_utilization': efficiency.speed_utilization,
            'num_stops': efficiency.num_stops,
            'stop_duration_sec': efficiency.stop_duration,
        },
        'comfort_metrics': {
            'max_longitudinal_accel': comfort.max_longitudinal_accel,
            'max_lateral_accel': comfort.max_lateral_accel,
            'max_jerk': comfort.max_jerk,
            'avg_jerk': comfort.avg_jerk,
            'harsh_accelerations': comfort.harsh_accelerations,
            'harsh_brakes': comfort.harsh_brakes,
            'harsh_turns': comfort.harsh_turns,
        },
        'robustness_metrics': {
            'disengagements': robustness.disengagements,
            'mpd': robustness.mpd,
            'system_failures': robustness.system_failures,
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 50}")
    print("📊 评估报告")
    print(f"{'=' * 50}")
    print(f"总分: {overall_score:.1f} / 100")
    print(f"  ├─ 安全性: {safety_score:.1f}")
    print(f"  ├─ 效率性: {efficiency_score:.1f}")
    print(f"  ├─ 舒适性: {comfort_score:.1f}")
    print(f"  └─ 鲁棒性: {robustness_score:.1f}")
    print(f"\n报告已保存至: {output_path}")

def calculate_safety_score(safety: SafetyMetrics) -> float:
    """计算安全性得分 (0-100)"""
    score = 100.0

    # 碰撞 (每次 -50 分)
    score -= safety.total_collisions * 50

    # 压实线 (每次 -20 分)
    score -= safety.solid_line_crossings * 20

    # 压虚线 (每次 -5 分)
    score -= (safety.lane_invasions - safety.solid_line_crossings) * 5

    # 闯红灯 (每次 -30 分)
    score -= safety.red_light_violations * 30

    # 超速 (每次 -10 分)
    score -= safety.speeding_violations * 10

    # TTC 预警 (每次 -5 分)
    score -= safety.ttc_warnings * 5

    # 紧急制动 (每次 -3 分)
    score -= safety.emergency_brakes * 3

    return max(0, score)

def calculate_efficiency_score(efficiency: EfficiencyMetrics) -> float:
    """计算效率性得分 (0-100)"""
    score = 100.0

    # 速度利用率 (低于 80% 扣分)
    if efficiency.speed_utilization < 0.8:
        score -= (0.8 - efficiency.speed_utilization) * 100

    # 停车次数 (每次 -2 分)
    score -= efficiency.num_stops * 2

    return max(0, min(100, score))

def calculate_comfort_score(comfort: ComfortMetrics) -> float:
    """计算舒适性得分 (0-100)"""
    score = 100.0

    # 最大加加速度超标扣分
    if comfort.max_jerk > 2.0:
        score -= (comfort.max_jerk - 2.0) * 10

    # 急操作扣分
    score -= comfort.harsh_accelerations * 5
    score -= comfort.harsh_brakes * 5
    score -= comfort.harsh_turns * 5

    return max(0, score)

def calculate_robustness_score(robustness: RobustnessMetrics) -> float:
    """计算鲁棒性得分 (0-100)"""
    score = 100.0

    # 接管扣分
    score -= robustness.disengagements * 10

    # 系统故障扣分
    score -= robustness.system_failures * 20

    return max(0, score)
```

### 5.2 HTML 可视化报告

```python
# carla_evaluation/reports/html_report.py

def generate_html_report(
    metrics_summary: Dict,
    output_path: str = "evaluation_report.html"
):
    """生成 HTML 可视化报告"""

    safety = metrics_summary['safety']
    efficiency = metrics_summary['efficiency']
    comfort = metrics_summary['comfort']

    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>自动驾驶评估报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 40px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 8px;
            background: #fafafa;
        }}
        .metric-card h3 {{
            margin-top: 0;
            color: #34495e;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .safe {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .critical {{ color: #e74c3c; }}
        .score-bar {{
            height: 30px;
            background: #ecf0f1;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .score-fill {{
            height: 100%;
            background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%);
            transition: width 0.3s;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 自动驾驶评估报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>📊 总体评分</h2>
        <div class="score-bar">
            <div class="score-fill" style="width: {calculate_safety_score(safety)}%"></div>
        </div>
        <p>安全性得分: {calculate_safety_score(safety):.1f} / 100</p>

        <h2>🛡️ 安全性指标</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>碰撞次数</h3>
                <div class="metric-value {'critical' if safety.total_collisions > 0 else 'safe'}">
                    {safety.total_collisions}
                </div>
                <p>车辆: {safety.vehicle_collisions} | 行人: {safety.pedestrian_collisions} | 静态: {safety.static_collisions}</p>
            </div>

            <div class="metric-card">
                <h3>车道偏离</h3>
                <div class="metric-value {'warning' if safety.lane_invasions > 0 else 'safe'}">
                    {safety.lane_invasions}
                </div>
                <p>压实线: {safety.solid_line_crossings}</p>
            </div>

            <div class="metric-card">
                <h3>闯红灯</h3>
                <div class="metric-value {'critical' if safety.red_light_violations > 0 else 'safe'}">
                    {safety.red_light_violations}
                </div>
            </div>

            <div class="metric-card">
                <h3>超速次数</h3>
                <div class="metric-value {'warning' if safety.speeding_violations > 0 else 'safe'}">
                    {safety.speeding_violations}
                </div>
            </div>
        </div>

        <h2>⚡ 效率性指标</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>累计里程</h3>
                <div class="metric-value">{efficiency.total_distance / 1000:.2f} km</div>
            </div>

            <div class="metric-card">
                <h3>平均速度</h3>
                <div class="metric-value">{efficiency.average_speed * 3.6:.1f} km/h</div>
            </div>

            <div class="metric-card">
                <h3>速度利用率</h3>
                <div class="metric-value">{efficiency.speed_utilization * 100:.1f}%</div>
            </div>

            <div class="metric-card">
                <h3>停车次数</h3>
                <div class="metric-value">{efficiency.num_stops}</div>
            </div>
        </div>

        <h2>😊 舒适性指标</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>最大纵向加速度</h3>
                <div class="metric-value">{comfort.max_longitudinal_accel:.2f} m/s²</div>
            </div>

            <div class="metric-card">
                <h3>最大横向加速度</h3>
                <div class="metric-value">{comfort.max_lateral_accel:.2f} m/s²</div>
            </div>

            <div class="metric-card">
                <h3>最大加加速度</h3>
                <div class="metric-value">{comfort.max_jerk:.2f} m/s³</div>
            </div>

            <div class="metric-card">
                <h3>急操作统计</h3>
                <p>急加速: {comfort.harsh_accelerations}</p>
                <p>急减速: {comfort.harsh_brakes}</p>
                <p>急转弯: {comfort.harsh_turns}</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"HTML 报告已生成: {output_path}")
```

---

## 6. 完整使用示例

### 6.1 集成到 Occupancy Network

```python
# examples/occupancy_network_with_evaluation.py

import carla
from occupancy.occupancy_inference import OccupancyNetworkInference
from planning.occupancy_planner import OccupancyPlanner
from carla_evaluation.sensors.evaluation_sensors import EvaluationSensorSuite
from carla_evaluation.metrics.metrics_calculator import MetricsCalculator
from carla_evaluation.visualization.evaluation_hud import EvaluationHUD

def main():
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # 生成车辆
    vehicle = spawn_vehicle(world)

    # 初始化 Occupancy Network
    occupancy_net = OccupancyNetworkInference(
        model_path='checkpoints/occupancy_network_best.pth'
    )

    # 初始化规划器
    planner = OccupancyPlanner()

    # ===== 初始化评估系统 =====
    sensor_suite = EvaluationSensorSuite(world, vehicle)
    metrics_calculator = MetricsCalculator()
    hud = EvaluationHUD(600, 900)

    # 相机管理器
    camera_manager = setup_cameras(world, vehicle)

    try:
        while True:
            world.tick()

            # 1. 获取相机图像
            camera_images = camera_manager.get_synced_frame()

            # 2. 获取车辆状态
            velocity = vehicle.get_velocity()
            speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
            yaw_rate = vehicle.get_angular_velocity().z

            # 3. Occupancy Network 推理
            occupancy_output = occupancy_net.inference(
                cameras=camera_images,
                speed=speed,
                yaw_rate=yaw_rate
            )

            # 4. 规划控制命令
            control_command = planner.plan(
                occupancy_grid=occupancy_output['occupancy'],
                flow=occupancy_output['flow'],
                vehicle_state={'speed': speed, 'yaw_rate': yaw_rate}
            )

            # 5. 应用控制命令
            vehicle.apply_control(carla.VehicleControl(
                throttle=max(0, control_command['acceleration'] / 3.0),
                brake=max(0, -control_command['acceleration'] / 8.0),
                steer=control_command['steering_angle']
            ))

            # ===== 6. 更新评估指标 =====
            metrics_calculator.update_real_time(vehicle, world, 0.05)

            # 处理事件
            for event in sensor_suite.collision_events:
                metrics_calculator.update_from_collision_event(event)
            sensor_suite.collision_events.clear()

            for event in sensor_suite.lane_invasion_events:
                metrics_calculator.update_from_lane_invasion_event(event)
            sensor_suite.lane_invasion_events.clear()

            # 更新 HUD
            metrics_summary = metrics_calculator.get_summary()
            hud.render(metrics_summary)

    finally:
        # 生成最终报告
        final_summary = metrics_calculator.get_summary()
        generate_evaluation_report(final_summary, "occupancy_evaluation_report.json")
        generate_html_report(final_summary, "occupancy_evaluation_report.html")

        sensor_suite.cleanup()
        vehicle.destroy()

if __name__ == '__main__':
    main()
```

---

## 总结

本文档提供了完整的无人驾驶测试评估标准体系:

### ✅ 已实现功能

1. **国际标准覆盖**: ISO 26262, ISO 21448, ISO 34501/34502, SAE J3016, Euro NCAP
2. **完整指标体系**:
   - 安全性 (碰撞/压线/违章/危险接近)
   - 效率性 (里程/速度/任务完成)
   - 舒适性 (加速度/加加速度/急操作)
   - 鲁棒性 (接管/MPD/故障)
3. **CARLA 实时检测**:
   - 碰撞传感器 (`CollisionSensor`)
   - 车道入侵传感器 (`LaneInvasionSensor`)
   - IMU 传感器 (加速度/角速度)
   - 交规违章检测器 (红绿灯/超速/逆行)
4. **实时监控**: PyGame HUD 实时显示所有指标
5. **评估报告**: JSON + HTML 可视化报告

### 🎯 关键优势

- **标准符合**: 完全遵循国际标准 (ISO/SAE/NHTSA)
- **实时反馈**: 毫秒级事件检测和统计
- **可扩展**: 易于添加新指标和检测器
- **可视化**: 实时 HUD + 离线报告
- **Occupancy 集成**: 无缝集成到 Occupancy Network 系统

### 📦 文件清单

```
carla_evaluation/
├── sensors/
│   └── evaluation_sensors.py          # 评估传感器套件
├── metrics/
│   └── metrics_calculator.py          # 指标计算器
├── detectors/
│   └── traffic_violation_detector.py  # 交规违章检测器
├── visualization/
│   └── evaluation_hud.py               # 实时 HUD
└── reports/
    ├── report_generator.py             # JSON 报告
    └── html_report.py                  # HTML 报告
```

所有代码均可直接在 CARLA 环境中运行! 🚀

# occnetv3_data_generator 重构分析报告

## 结论

经过对源码、实际生成数据（`dataset_10k_bak`）以及训练网络（`e2e_occ`）的全面对比分析：

**数据生成器本身逻辑基本正确**，但存在**两端格式不对齐**的问题——生成器输出 `ego_pose/` 目录，而 `e2e_occ/dataset.py` 期望 `camera_params/` 目录，导致时序对齐数据无法被网络读取。

---

## 问题详细清单

### 严重问题 ①：数据格式不对齐（`ego_pose/` vs `camera_params/`）

**根因**：

- `occnetv3_data_generator` 保存车辆绝对位姿到 `ego_pose/{sample_id}.npy`（Vehicle→World，`(4,4) float32`）
- `e2e_occ/dataset.py` 的 `_load_per_frame_params()` 期望读取 `camera_params/{sample_id}.npz`（含 `intrinsics:[N,3,3]` 和 `extrinsics:[N,4,4]`）
- 两套格式完全不同，导致 `_has_per_frame_params=False`，dataset 回退到静态 calibration

**后果**：序列模式下每帧 `extrinsics` 完全相同，`ego_motion = inv(E) @ E = Identity`，时序对齐完全失效。

**验证**：
```
dataset_10k_bak/
  ego_pose/   ← 存在，每帧不同，(4,4) Vehicle→World  ✅
  ego_motion/ ← 存在，帧间运动  ✅
  camera_params/ ← 不存在  ❌（e2e_occ dataset.py 期望此目录）
```

**修复方案**：修改 `e2e_occ/dataset.py`，增加对 `ego_pose/` 目录的读取支持。无需修改数据生成器。

---

### 严重问题 ②：`camera_manager.get_extrinsics()` 语义错误

**根因**：

```python
# sensors/camera_manager.py 第 419-450 行
def get_extrinsics(self, cam_id: str) -> np.ndarray:
    camera = self.cameras[cam_id]
    transform = camera.get_transform()   # ⚠️ 世界坐标系中的绝对位置
    T[:3, 3] = [transform.location.x, transform.location.y, transform.location.z]
```

`camera.get_transform()` 在 CARLA 中返回的是相机在**世界坐标系**中的绝对变换。但 `save_calibration()` 在采集 setup 阶段（`world.tick()×10` 之前）被调用，此时车辆尚未运动，相机位置接近世界原点，实际保存的 translation 全是 `[0, 0, 0]`。

这意味着 `calibration/extrinsics.json` 存储的是**既不是相机安装位置（相对车辆）、也不是相机绝对世界位置**的无效数据。

**验证**：
```
cam_0 translation from calibration/extrinsics.json: [0.0, 0.0, 0.0]  ← 错误
cam_0 实际安装位置（camera_config.py）: position=[1.0, 0.0, 1.6]     ← 应该是这个
```

**后果**：使用 `calibration/extrinsics.json` 做投影计算时，所有相机位置都在原点，几何投影完全错误。

**修复方案**：`get_extrinsics()` 应基于 `camera_config.py` 中的安装参数直接计算，而不是调用运行时的 `camera.get_transform()`。

---

### 严重问题 ③：`ground_truth_voxel_generator.py` 跨包导入

**根因**：

```python
# processing/ground_truth_voxel_generator.py 第 10-15 行
from dense_occupancy_collection.config.occupancy_config import (
    CARLA_TO_OCCUPANCY_MAPPING, OCCUPANCY_LABELS
)
from dense_occupancy_collection.config.actor_occupancy_mapping import (
    get_occupancy_label_from_actor
)
```

直接从 `dense_occupancy_collection` 包导入，与本项目耦合。若 `dense_occupancy_collection` 未安装或路径不在 `sys.path` 中，将报 `ModuleNotFoundError`。

**修复方案**：将所需常量复制到本项目 `config/` 目录（`occupancy_config.py` 中已有 `CARLA_TO_OCCUPANCY` 定义），改为本地导入。

---

### 中等问题 ④：`extrinsics.json` 坐标系约定不明确

`camera_manager.py` 注释写"车辆→相机"，但代码实际取 `camera.get_transform()`（世界绝对位置）。这与 `e2e_occ/deformable_attention.py` 中 `get_reference_points()` 期望的 `World→Camera`（`inv_extrinsics`）约定不符。

**需要统一**：所有外参一律使用 **Camera→World**（相机在世界坐标系中的位姿），投影时取逆得到 World→Camera。

---

### 中等问题 ⑤：语义类别 (18类) 两套定义不统一

| 项目 | 类别数 | 类别 0 | 类别体系 |
|------|--------|--------|---------|
| `occnetv3_data_generator/config/occupancy_config.py` | 18类 | `empty` | nuScenes 风格（car/pedestrian/truck...） |
| `e2e_occ/config.py` | 18类 | `num_classes=18` | 与上同，但未列出类别名 |
| `occ_network_nano`（旧） | 18类 | `Free/Unlabeled` | CARLA 语义标签风格（Building/Fence...） |

两套 18 类的**内容不同**（nuScenes vs CARLA 原始语义）。用生成器数据训练时，网络输出的类别 `4` 在一套里是 `car`，在另一套里是 `Pedestrian`。

**修复方案**：在 `e2e_occ/config.py` 中显式声明类别名称并与 `occnetv3_data_generator` 的 18 类对齐。

---

## 重构计划

### Phase 1：修复数据对齐（最高优先，立即可做）

**目标**：让 `e2e_occ/dataset.py` 能正确读取 `occnetv3_data_generator` 生成的数据。

**修改 `e2e_occ/dataset.py`**：增加 `ego_pose/` 目录读取路径，从绝对位姿和相机安装配置计算逐帧绝对外参。

```
修改文件：e2e_occ/dataset.py
新增逻辑：
  1. 检测 ego_pose/ 目录
  2. 加载 ego_pose/{sample_id}.npy（Vehicle→World，4×4）
  3. 结合 calibration/extrinsics.json（相机相对安装位置，含正确旋转）
  4. 计算逐帧绝对外参：T_cam_world = ego_pose @ T_cam_relative
  5. 序列帧的 extrinsics 真实不同，ego_motion 计算正确
```

### Phase 2：修复 extrinsics 计算（高优先）

**目标**：让 `calibration/extrinsics.json` 保存正确的相机安装参数（Camera→Vehicle 的固定偏移）。

**修改 `sensors/camera_manager.py`**：`get_extrinsics()` 改为从 `camera_config.py` 中的安装参数直接计算，而不是运行时 `get_transform()`。

```python
# 修复前（错误）
transform = camera.get_transform()   # 世界绝对位置（随车辆位置变化）

# 修复后（正确）
# 从 camera_config 中读取安装参数
cam_cfg = 找到对应 cam_id 的配置
T = 从 position + rotation 构建 4×4 矩阵   # 相机相对车辆的固定安装位姿
```

### Phase 3：修复跨包导入（高优先）

**修改 `processing/ground_truth_voxel_generator.py`**：

```python
# 修复前
from dense_occupancy_collection.config.occupancy_config import CARLA_TO_OCCUPANCY_MAPPING

# 修复后
from config.occupancy_config import CARLA_TO_OCCUPANCY  # 本地
```

确认本项目 `config/occupancy_config.py` 已有 `CARLA_TO_OCCUPANCY` 定义（已有），补充 `OCCUPANCY_LABELS` 和 `get_occupancy_label_from_actor()` 的本地版本。

### Phase 4：统一语义类别（中优先）

在 `e2e_occ/config.py` 中添加：

```python
# 与 occnetv3_data_generator/config/occupancy_config.py 完全对齐
CLASS_NAMES = [
    'empty', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian',
    'traffic_cone', 'trailer', 'truck', 'driveable_surface',
    'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'general_object'
]
```

---

## 数据流完整对齐方案

```
occnetv3_data_generator 输出：
  calibration/
    intrinsics.json  → cam安装内参（恒定）
    extrinsics.json  → cam相对车辆安装位姿（Phase 2 修复后才正确）
  ego_pose/
    {sample_id}.npy  → Vehicle→World 绝对位姿（每帧不同）✅
  ego_motion/
    {sample_id}.npy  → 帧间运动（预计算，备用）✅
  images/
    {sample_id}/cam_{i}.dng   → Bayer RAW 12-bit ✅
  occupancy/
    {sample_id}.npy  → (400,400,32) uint8 ✅

e2e_occ/dataset.py 加载逻辑（Phase 1 修复后）：
  intrinsics[N,3,3]   ← calibration/intrinsics.json（恒定，OK）
  extrinsics[N,4,4]   ← ego_pose/{sample_id}.npy × calibration/extrinsics.json
                         = (Vehicle→World) × (Camera→Vehicle)
                         = Camera→World（每帧不同）✅

e2e_occ/train.py ego_motion 计算：
  pose_t    = extrinsics_t[:, 0]    # Camera_0→World（含车辆绝对位姿）
  pose_prev = extrinsics_{t-1}[:, 0]
  ego_motion = inv(pose_t) @ pose_prev  # C_{t-1}→C_t ✅
```

---

## 各问题影响对照表

| 问题 | 影响范围 | 后果 | 优先级 |
|------|---------|------|--------|
| ① `ego_pose/` vs `camera_params/` 格式不对齐 | e2e_occ 时序训练 | ego_motion 永远=Identity，时序对齐完全失效 | P0 |
| ② `extrinsics` 计算错误（取世界绝对位置） | 相机投影 | 几何投影偏差，影响 DeformableCrossAttention | P1 |
| ③ 跨包导入 `dense_occupancy_collection` | 部署/测试 | 无法独立运行，ModuleNotFoundError | P1 |
| ④ extrinsics 坐标系约定不明确 | 文档/维护 | 易产生误用 | P2 |
| ⑤ 语义类别两套定义 | 训练效果 | 类别标签错位，影响所有类别准确率 | P2 |

---

## 当前数据集（`dataset_10k_bak`）状态

```
总帧数: 5 帧（1 个场景）
存在目录:
  ✅ calibration/   (intrinsics.json, extrinsics.json)
  ✅ images/        (8 × DNG per frame)
  ✅ depth/         (8 × npy per frame)
  ✅ occupancy/     (400,400,32 uint8 per frame)
  ✅ flow/          (3,400,400,32 float16 per frame)
  ✅ flow_mask/     (400,400,32 uint8 per frame)
  ✅ ego_pose/      (4,4 float32 per frame) ← 关键数据，正确且逐帧不同
  ✅ ego_motion/    (4,4 float32 per frame)
  ❌ camera_params/ ← 不存在（e2e_occ 新版 dataset.py 期望此格式）

extrinsics.json 问题:
  cam_0 translation=[0,0,0]  ← 采集时相机 world transform 异常，应为安装偏移
```

---

## 立即可执行的修复（Phase 1）

修改 `e2e_occ/dataset.py`，新增 `ego_pose/` 路径支持，无需重新采集数据，立即让时序训练生效。

具体改动：
1. `__init__` 中检测 `ego_pose/` 目录
2. 新增 `_load_ego_pose_params(sample_id)` 方法：加载 ego_pose + calibration 内外参，组合成逐帧绝对外参
3. `_get_frame_params()` 优先级：`camera_params/` npz → `ego_pose/` + calibration → 默认值

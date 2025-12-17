# 全景反投影算法重构说明

**更新时间**: 2025-12-16
**问题**: 体素生成存在间隙断层，尽管全景图连续
**根因**: `panorama_tools.py` 中的坐标变换算法错误

---

## 问题诊断

### 症状
- 全景深度图和语义图是连续的（无间隙）
- 但生成的体素存在间歇性断层
- 物体边界不清晰，出现"撕裂"现象

### 根本原因

#### 错误实现（修正前）
```python
# panorama_tools.py::unproject_panorama() (旧版)
lon = (u / W - 0.5) * 2 * np.pi  # ❌ 范围: [-π, π]
lat = -(v / H - 0.5) * np.pi     # ❌ 范围: [π/2, -π/2] (符号错)

x = d * np.cos(lat) * np.cos(lon)  # ❌ 角度范围错误
y = d * np.cos(lat) * np.sin(lon)  # ❌ Y轴方向错误（应该是"左"不是"右"）
z = d * np.sin(lat)
```

**问题点**：
1. **经度偏移错误**: `(u / W - 0.5) * 2π` 导致 θ ∈ [-π, π]，应该是 [0, 2π]
2. **纬度符号错误**: `-(v / H - 0.5) * π` 的符号和计算顺序都不对
3. **Y轴语义错误**: CARLA坐标系中 Y 是"左"，不是"右"

#### 正确实现（修正后）
```python
# panorama_tools.py::unproject_panorama() (新版)
theta = (u / W) * 2 * np.pi        # ✓ 范围: [0, 2π]
phi = (0.5 - v / H) * np.pi        # ✓ 范围: [π/2, -π/2]

dir_x = np.cos(phi) * np.cos(theta)   # ✓ 前
dir_y = np.cos(phi) * np.sin(theta)   # ✓ 左
dir_z = np.sin(phi)                   # ✓ 上

x = dir_x * depth
y = dir_y * depth
z = dir_z * depth
```

---

## 修改内容

### 1. `processing/panorama_tools.py`

#### `unproject_panorama()` 方法（lines 145-204）
- **修正**: 采用参考文档第5.2节的标准算法
- **变更**:
  - 角度计算: `theta = (u / W) * 2π`, `phi = (0.5 - v / H) * π`
  - 坐标系: 明确 [X前, Y左, Z上] 的 CARLA 惯例
  - 注释: 增加详细的步骤说明

#### `_init_remap_tables()` 方法（lines 14-42）
- **修正**: 统一使用与反投影一致的坐标系统
- **变更**:
  - 角度计算公式与 `unproject_panorama()` 保持一致
  - 注释优化，明确坐标系定义

---

## 理论依据

### Equirectangular 投影标准公式

**像素 → 球面角度**:
```
θ = (u / W) × 2π         # 水平角 (经度) [0, 2π]
φ = (0.5 - v / H) × π    # 垂直角 (纬度) [π/2, -π/2]
```

**球面角度 → 3D 方向**:
```
dir_x = cos(φ) × cos(θ)  # 前 (X轴)
dir_y = cos(φ) × sin(θ)  # 左 (Y轴)
dir_z = sin(φ)           # 上 (Z轴)
```

**3D 点坐标**:
```
point = direction × depth
```

### 为什么这样计算？

1. **θ = (u / W) × 2π**:
   - u ∈ [0, W] 映射到完整圆周 [0, 2π]
   - u=0 对应前方 (θ=0)
   - u=W/4 对应左侧 (θ=π/2)
   - u=W/2 对应后方 (θ=π)
   - u=3W/4 对应右侧 (θ=3π/2)

2. **φ = (0.5 - v / H) × π**:
   - v ∈ [0, H] 映射到 [π/2, -π/2]
   - v=0 对应天顶 (φ=π/2)
   - v=H/2 对应水平 (φ=0)
   - v=H 对应地面 (φ=-π/2)

3. **dir = (cos(φ)cos(θ), cos(φ)sin(θ), sin(φ))**:
   - 标准球面坐标转笛卡尔坐标
   - 确保 |dir| = 1 (单位方向向量)

---

## 验证方法

### 测试脚本
运行验证脚本检查修复效果：

```bash
conda activate carla
cd d:\code\carla
python dense_occupancy_collection\scripts\verify_unprojection.py
```

### 预期结果
✓ 点云距离一致性良好（偏差 < 0.01米）
✓ 体素连续无间隙
✓ 水平切片无断层

### 实际数据测试
```bash
python dense_occupancy_collection\scripts\collect_panorama.py --frames 1
```

检查输出的 `occupancy/*.npz` 文件，验证体素是否连续。

---

## 影响范围

### 修改文件
- `processing/panorama_tools.py` (核心算法修正)

### 不受影响
- `sensors/panorama_manager.py` (传感器管理)
- `sensors/rgb_camera_manager.py` (RGB相机)
- `config/panorama_config.py` (配置)
- `config/occupancy_config.py` (体素配置)
- `processing/dense_voxel_generator.py` (体素化逻辑)

### 兼容性
- ✓ 向后兼容：配置文件无需修改
- ✓ 数据格式不变：输出 NPZ 格式保持一致
- ✓ API 不变：函数签名保持不变

---

## 参考文档

**主要依据**:
`.trae/documents/hero车辆与npc初始化/基于全景深度图方案的CARLA_UE5_3D体素数据集生成指南.md`

**关键章节**:
- 第5.2节：全景深度图反投影算法
- 第2.2节：坐标映射公式
- 第8章：核心技术总结

---

## 附加说明

### CARLA 坐标系约定
```
X轴: 前方 (Forward)
Y轴: 左侧 (Left)
Z轴: 上方 (Up)
```

### 全景图布局
```
┌────────────────────────────────────┐
│           v=0 (天顶, φ=π/2)        │
├────────────────────────────────────┤
│ u=0    u=W/4    u=W/2    u=3W/4    │
│ 前方    左侧     后方      右侧     │
│ θ=0    θ=π/2    θ=π      θ=3π/2   │
├────────────────────────────────────┤
│          v=H (地面, φ=-π/2)        │
└────────────────────────────────────┘
       宽度 = 2:1 比例（标准ERP）
```

---

**状态**: ✅ 算法已修正
**测试**: 待运行验证脚本
**下一步**: 使用真实 CARLA 数据测试

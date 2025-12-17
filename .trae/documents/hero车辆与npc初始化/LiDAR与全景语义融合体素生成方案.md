# LiDAR与360°全景语义融合的稠密体素生成方案

**创建时间**: 2025-12-16
**目标**: 结合激光雷达深度准确性和全景语义图稠密性的优势，生成无间隙的Occupancy体素

---

## 一、方案背景与动机

### 1.1 现有方案的问题

#### 方案A: 纯激光雷达（carla_data_collection）
**优势**:
- ✓ 深度精度高（毫米级）
- ✓ 几何一致性好
- ✓ 实时性强（64线@20Hz）

**问题**:
- ❌ **稀疏性**：64线激光雷达仅6-8万点/帧，无法覆盖200×200×16=64万个体素
- ❌ **间隙**：扫描线之间存在空隙，远处物体覆盖率<30%
- ❌ **边界模糊**：小物体（行人、交通标志）点云不足

#### 方案B: 360°全景深度+语义（dense_occupancy_collection）
**优势**:
- ✓ **完全稠密**：遍历所有体素，理论上无间隙
- ✓ 360°覆盖，无盲区
- ✓ 语义信息连续

**问题**:
- ❌ **深度不准**：基于渲染深度图，存在采样误差和边界模糊
- ❌ **计算量大**：需要6个CubeMap相机 + 拼接 + 反投影
- ❌ **实测效果不佳**：体素间隙问题仍然存在（坐标转换复杂）

### 1.2 混合方案的核心思想

> **用激光雷达提供准确的3D几何深度，用360°全景语义图提供稠密的语义标签填充**

**关键洞察**:
1. 激光雷达给出物体表面的**精确深度**
2. 全景语义图给出每个射线方向的**完整语义区域**
3. 两者结合：**深度来自LiDAR，语义来自全景图，遍历体素查询两者**

**类比**:
```
激光雷达 = 骨架（稀疏但准确）
全景语义 = 肉（稠密但深度不准）
混合方法 = 完整的人体模型（准确+稠密）
```

---

## 二、技术方案设计

### 2.1 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    数据采集层                                │
├──────────────────────────┬──────────────────────────────────┤
│   8个RGB相机 (训练输入)   │   语义激光雷达 (深度GT)           │
│   1280×960 @ 20Hz        │   128线 @ 20Hz                   │
├──────────────────────────┼──────────────────────────────────┤
│                          │   6个全景语义相机 (语义GT)         │
│                          │   512×512×6 → 1024×512全景        │
└──────────────────────────┴──────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────┐
│                    预处理层                                  │
├──────────────────────────┬──────────────────────────────────┤
│   点云坐标变换           │   CubeMap拼接为ERP全景图          │
│   World → Ego            │   1024×512 语义全景                │
└──────────────────────────┴──────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────┐
│                  核心算法：混合体素生成                       │
│                                                             │
│   for 每个体素 (x, y, z):                                   │
│       1. 查询激光雷达 → 获取该方向的深度 d_lidar            │
│       2. 查询全景图 → 获取该方向的语义 label_pano           │
│       3. 判断体素状态:                                       │
│          - 如果 |d_voxel - d_lidar| < threshold:            │
│              → 表面，occupancy = label_pano                 │
│          - 如果 d_voxel < d_lidar - threshold:              │
│              → 自由空间，occupancy = 0, mask = True         │
│          - 如果 d_voxel > d_lidar + threshold:              │
│              → 未知/遮挡，mask = False                       │
└─────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────┐
│                    输出层                                    │
├──────────────────────────┬──────────────────────────────────┤
│   8个RGB图像             │   Occupancy体素                   │
│   (训练输入)              │   200×200×16 (GT标签)             │
│                          │   occupancy + mask                │
└──────────────────────────┴──────────────────────────────────┘
```

### 2.2 关键组件

#### 组件1: 高线数语义激光雷达
```python
# 配置升级
CHANNELS = 128           # 从64线升级到128线
POINTS_PER_SECOND = 2400000  # 240万点/秒（从120万提升）
UPPER_FOV = 30.0        # 上视角30°（覆盖车顶上方）
LOWER_FOV = -40.0       # 下视角-40°（覆盖地面）
RANGE = 100.0           # 探测距离100米
```

**垂直角度覆盖验证**:
- 体素Z范围：[-4, 4]米 → 8米高度
- 车辆高度：假设车顶在Z=2米
- 需要覆盖：Z=-4到Z=4+2=6米（相对于车顶）
- 在10米远处，需要视角：`arctan(6/10) ≈ 31°`
- 128线在[-40°, +30°]范围提供70°覆盖 → **充足**

#### 组件2: 360°全景语义相机
**从 dense_occupancy_collection 移植**:
```python
# panorama_manager.py (仅移植语义分割部分)
class PanoramaSensorManager:
    def __init__(self, world, vehicle):
        # 创建6个语义分割相机 (CubeMap)
        # 不需要深度相机！
        for face in CUBE_FACE_CONFIGS:
            sem_camera = create_semantic_camera(
                size=512, fov=90, rotation=face['rot']
            )

    def get_semantic_panorama(self):
        # 获取6个面的语义图
        # 拼接为1024×512全景图
        return semantic_pano  # (H, W) uint8
```

#### 组件3: 混合体素生成器
```python
class HybridOccupancyGenerator:
    """
    混合激光雷达深度 + 全景语义的体素生成器
    """

    def generate(self, lidar_points, lidar_labels, semantic_pano):
        """
        Args:
            lidar_points: (N, 3) 激光点云坐标 [x, y, z] (车辆坐标系)
            lidar_labels: (N,) 激光雷达的语义标签（可选，不用）
            semantic_pano: (H, W) 全景语义图

        Returns:
            occupancy: (X, Y, Z) 体素语义标签
            mask: (X, Y, Z) 观测掩码
        """

        # 步骤1: 构建激光雷达深度查找表
        depth_map = self._build_lidar_depth_map(lidar_points)
        # depth_map[θ_idx, φ_idx] = 最近深度值

        # 步骤2: 遍历所有体素
        for gx, gy, gz in 遍历(grid_size):
            # 体素中心坐标
            x = x_min + (gx + 0.5) * resolution
            y = y_min + (gy + 0.5) * resolution
            z = z_min + (gz + 0.5) * resolution

            # 转换为球面坐标
            d_voxel = sqrt(x² + y² + z²)
            θ = arctan2(y, x)
            φ = arcsin(z / d_voxel)

            # 查询激光雷达深度
            d_lidar = depth_map.query(θ, φ)

            # 查询全景语义
            u = θ_to_u(θ, W)
            v = φ_to_v(φ, H)
            label_pano = semantic_pano[v, u]

            # 状态判断
            diff = d_voxel - d_lidar

            if abs(diff) < threshold:
                # 表面
                occupancy[gx, gy, gz] = label_pano
                mask[gx, gy, gz] = True
            elif diff < -threshold:
                # 自由空间
                occupancy[gx, gy, gz] = 0
                mask[gx, gy, gz] = True
            else:
                # 未知/遮挡
                mask[gx, gy, gz] = False

        return occupancy, mask
```

---

## 三、实施步骤

### 第一步: 移植360°全景语义图采集代码

**目标**: 在 carla_data_collection 中添加全景语义相机支持

**文件清单**:
1. ✅ **移植文件**:
   ```
   dense_occupancy_collection/config/panorama_config.py
       → carla_data_collection/config/panorama_config.py

   dense_occupancy_collection/sensors/panorama_manager.py
       → carla_data_collection/sensors/panorama_manager.py
       (仅保留语义分割相机部分，删除深度相机)

   dense_occupancy_collection/processing/panorama_tools.py
       → carla_data_collection/processing/panorama_tools.py
       (仅保留CubeMap拼接功能，删除反投影函数)
   ```

2. ❌ **不移植**:
   - `dense_voxel_generator.py` 的 `generate_from_panorama()` 方法
   - 全景深度相机相关代码
   - 反投影算法（我们用新的混合算法）

**修改点**:
```python
# carla_data_collection/sensors/panorama_manager.py
class PanoramaSensorManager:
    def _setup_sensors(self):
        # 只创建6个语义分割相机
        for face in CUBE_FACE_CONFIGS:
            # ❌ 删除深度相机
            # depth_bp = self.bp_library.find('sensor.camera.depth')

            # ✓ 保留语义相机
            sem_bp = self.bp_library.find('sensor.camera.semantic_segmentation')
            sem_bp.set_attribute('image_size_x', str(CUBE_SIZE))
            sem_bp.set_attribute('image_size_y', str(CUBE_SIZE))
            sem_bp.set_attribute('fov', str(CUBE_FOV))
            # ... 生成传感器

    def get_semantic_panorama(self, timeout=2.0):
        """只返回语义全景图"""
        semantic_faces = []
        for face_name in ['front', 'right', 'back', 'left', 'up', 'down']:
            s_img = self.queues[f"{face_name}_semantic"].get(timeout)
            # ... 解析
            semantic_faces.append(s_array[:, :, 2])

        # 拼接
        semantic_pano = self.pano_tools.stitch(semantic_faces)
        return semantic_pano  # (H, W) uint8
```

### 第二步: 升级激光雷达配置

**文件**: `carla_data_collection/sensors/semantic_lidar_sensor.py`

```python
# 修改配置
LIDAR_CONFIG = {
    'channels': 128,              # 64 → 128
    'range': 100.0,
    'points_per_second': 2400000, # 120万 → 240万
    'rotation_frequency': 20,
    'upper_fov': 30.0,            # 15 → 30
    'lower_fov': -40.0,           # -25 → -40
    # ... 其他参数保持不变
}
```

**验证点**:
- 点云密度：240万点/秒 @ 20Hz = 12万点/帧 (之前6-8万)
- 垂直覆盖：70°范围（-40到+30）
- 水平覆盖：360°
- 预期体素覆盖率：从30% → 60-70%（激光雷达部分）

### 第三步: 实现混合体素生成算法

**新文件**: `carla_data_collection/processing/hybrid_occupancy_generator.py`

**核心数据结构**:

#### 3.1 激光雷达深度图（球面索引）
```python
class SphericalDepthMap:
    """
    球面坐标系的深度查找表
    用于快速查询任意方向的激光雷达深度
    """

    def __init__(self, theta_bins=1024, phi_bins=512):
        # θ: [0, 2π] 分成1024个bin
        # φ: [-π/2, π/2] 分成512个bin
        self.depth = np.full((phi_bins, theta_bins), np.inf)
        self.label = np.zeros((phi_bins, theta_bins), dtype=np.uint8)
        self.count = np.zeros((phi_bins, theta_bins), dtype=np.int32)

    def add_points(self, points, labels):
        """
        将激光雷达点云添加到深度图

        Args:
            points: (N, 3) [x, y, z] 车辆坐标系
            labels: (N,) 语义标签（可选，不用）
        """
        # 转换为球面坐标
        d = np.linalg.norm(points, axis=1)
        theta = np.arctan2(points[:, 1], points[:, 0])  # [-π, π]
        phi = np.arcsin(points[:, 2] / d)               # [-π/2, π/2]

        # 归一化到bin索引
        theta_normalized = (theta + np.pi) % (2 * np.pi)  # [0, 2π]
        theta_idx = (theta_normalized / (2 * np.pi) * self.theta_bins).astype(int)
        phi_idx = ((phi / np.pi + 0.5) * self.phi_bins).astype(int)

        # 裁剪到有效范围
        theta_idx = np.clip(theta_idx, 0, self.theta_bins - 1)
        phi_idx = np.clip(phi_idx, 0, self.phi_bins - 1)

        # 填充（取最近深度）
        for i in range(len(points)):
            t_idx, p_idx = theta_idx[i], phi_idx[i]
            if d[i] < self.depth[p_idx, t_idx]:
                self.depth[p_idx, t_idx] = d[i]
                # self.label[p_idx, t_idx] = labels[i]  # 不用LiDAR标签
            self.count[p_idx, t_idx] += 1

    def query(self, theta, phi):
        """
        查询深度

        Args:
            theta: 水平角 [-π, π] 或 [0, 2π]
            phi: 垂直角 [-π/2, π/2]

        Returns:
            depth: 最近深度值，如果无数据返回np.inf
        """
        theta_normalized = (theta + np.pi) % (2 * np.pi)
        theta_idx = int(theta_normalized / (2 * np.pi) * self.theta_bins)
        phi_idx = int((phi / np.pi + 0.5) * self.phi_bins)

        theta_idx = np.clip(theta_idx, 0, self.theta_bins - 1)
        phi_idx = np.clip(phi_idx, 0, self.phi_bins - 1)

        return self.depth[phi_idx, theta_idx]
```

#### 3.2 混合体素生成器
```python
class HybridOccupancyGenerator:
    """
    混合激光雷达深度 + 全景语义的体素生成器
    """

    def __init__(self, x_range, y_range, z_range, resolution):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution

        self.grid_size = [
            int((x_range[1] - x_range[0]) / resolution),
            int((y_range[1] - y_range[0]) / resolution),
            int((z_range[1] - z_range[0]) / resolution)
        ]

    def generate(self, lidar_points, semantic_pano,
                 vehicle_height=2.0, threshold_ratio=0.8):
        """
        生成混合体素

        Args:
            lidar_points: (N, 3) 激光雷达点云 [x, y, z] (车辆坐标系)
            semantic_pano: (H, W) 全景语义图 uint8
            vehicle_height: 相机高度偏移（全景相机在车顶）
            threshold_ratio: 体素阈值倍数（默认0.8倍分辨率）

        Returns:
            occupancy: (X, Y, Z) uint8
            mask: (X, Y, Z) bool
        """
        H, W = semantic_pano.shape
        threshold = self.resolution * threshold_ratio

        # 初始化输出
        occupancy = np.zeros(self.grid_size, dtype=np.uint8)
        mask = np.zeros(self.grid_size, dtype=np.bool_)

        # 步骤1: 构建激光雷达深度图
        print("构建激光雷达深度图...")
        depth_map = SphericalDepthMap(theta_bins=W, phi_bins=H)
        depth_map.add_points(lidar_points, labels=None)

        # 步骤2: 遍历所有体素
        print(f"遍历体素网格 {self.grid_size}...")

        gx = np.arange(self.grid_size[0])
        gy = np.arange(self.grid_size[1])
        gz = np.arange(self.grid_size[2])
        grid_x, grid_y, grid_z = np.meshgrid(gx, gy, gz, indexing='ij')

        # 体素中心物理坐标
        x = self.x_range[0] + (grid_x + 0.5) * self.resolution
        y = self.y_range[0] + (grid_y + 0.5) * self.resolution
        z = self.z_range[0] + (grid_z + 0.5) * self.resolution

        # 高度修正（相机坐标系）
        z_cam = z - vehicle_height

        # 转换为球面坐标
        d_voxel = np.sqrt(x**2 + y**2 + z_cam**2)
        theta = np.arctan2(y, x)  # [-π, π]
        phi = np.arcsin(np.clip(z_cam / d_voxel, -1.0, 1.0))  # [-π/2, π/2]

        # 查询激光雷达深度（向量化）
        print("查询激光雷达深度...")
        theta_flat = theta.flatten()
        phi_flat = phi.flatten()
        d_lidar_flat = np.array([
            depth_map.query(t, p) for t, p in zip(theta_flat, phi_flat)
        ])
        d_lidar = d_lidar_flat.reshape(self.grid_size)

        # 查询全景语义
        print("查询全景语义...")
        theta_normalized = (theta + np.pi) % (2 * np.pi)
        u = (theta_normalized / (2 * np.pi) * W).astype(np.int32)
        v = ((0.5 - phi / np.pi) * H).astype(np.int32)
        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)

        sampled_label = semantic_pano[v, u]

        # 状态判断
        print("判断体素状态...")
        diff = d_voxel - d_lidar

        # 过滤无效LiDAR数据
        valid_lidar = d_lidar < 99.0  # 有效探测范围

        # 表面
        is_surface = valid_lidar & (np.abs(diff) < threshold)
        occupancy[is_surface] = sampled_label[is_surface]
        mask[is_surface] = True

        # 自由空间
        is_free = valid_lidar & (diff < -threshold)
        occupancy[is_free] = 0
        mask[is_free] = True

        # 未知/遮挡
        # mask保持False

        print(f"完成: 观测体素 {np.sum(mask):,} / {mask.size:,} "
              f"({np.sum(mask)/mask.size*100:.1f}%)")
        print(f"占用体素 {np.sum(occupancy > 0):,}")

        return occupancy, mask
```

### 第四步: 集成到数据采集流程

**修改文件**: `carla_data_collection/data/data_collector.py`

```python
class DataCollector:
    def setup(self):
        # ... 现有代码

        # 添加全景语义相机
        from sensors.panorama_manager import PanoramaSensorManager
        self.pano_manager = PanoramaSensorManager(self.world, self.hero_vehicle)
        print("✓ 全景语义相机已创建")

        # 升级激光雷达为128线
        from sensors.semantic_lidar_sensor import SemanticLidarSensor
        self.lidar = SemanticLidarSensor(
            self.world, self.hero_vehicle,
            channels=128,  # 升级
            points_per_second=2400000,
            upper_fov=30.0,
            lower_fov=-40.0
        )

        # 使用混合生成器
        from processing.hybrid_occupancy_generator import HybridOccupancyGenerator
        self.occupancy_gen = HybridOccupancyGenerator(
            x_range=[-50, 50],
            y_range=[-50, 50],
            z_range=[-4, 4],
            resolution=0.5
        )

    def _collect_frame(self):
        # ... 采集相机数据

        # 采集激光雷达
        lidar_data = self.lidar.get_data()
        points_ego = lidar_data['points']  # (N, 3)

        # 采集全景语义
        semantic_pano = self.pano_manager.get_semantic_panorama()

        # 生成混合Occupancy
        occupancy, mask = self.occupancy_gen.generate(
            lidar_points=points_ego,
            semantic_pano=semantic_pano,
            vehicle_height=2.0
        )

        # 保存到HDF5
        self._save_frame(camera_data, occupancy, mask, ...)
```

---

## 四、关键算法说明

### 4.1 球面深度图构建
**目的**: 将稀疏的激光雷达点云转换为可快速查询的深度图

**输入**: N个点 (x, y, z)
**输出**: 球面网格 (1024×512) 深度值

**算法**:
```
对每个点:
    1. 计算球面坐标: θ = arctan2(y, x), φ = arcsin(z/d)
    2. 映射到bin索引: θ_idx, φ_idx
    3. 更新深度图: depth[φ_idx, θ_idx] = min(current, d)
```

**优化**:
- 使用最近深度（min）而非平均，避免多次反射导致的深度模糊
- 分辨率匹配全景图（1024×512），无需插值

### 4.2 体素状态判断
**三种状态**:

```
    相机/LiDAR
         ●
         │
         │ 射线
         ↓
  ┌──┬──┬██┬──┬──┐
  │  │  │██│  │  │  体素层
  └──┴──┴──┴──┴──┘
   ↑  ↑  ↑  ↑  ↑
  Free  Surface Unknown

  d < d_lidar-ε
  |d-d_lidar|<ε
  d > d_lidar+ε
```

**阈值选择**:
- `threshold = resolution × 0.8`
- 0.5m分辨率 → 0.4m阈值
- 覆盖体素对角线长度的一半: `sqrt(3)/2 × 0.5 ≈ 0.43m`

### 4.3 语义标签来源
**为什么用全景语义而不是LiDAR语义？**

1. **稠密性**: 全景图1024×512=52万像素，LiDAR仅12万点
2. **连续性**: 同一物体在全景图中是连续区域，LiDAR点云分散
3. **准确性**: 语义分割基于视觉渲染，比LiDAR的object_idx更准确

**流程**:
```
体素(x,y,z) → 球面(θ,φ) → 全景UV(u,v) → semantic_pano[v,u]
```

---

## 五、预期效果

### 5.1 定量指标

| 指标 | 纯LiDAR (64线) | 纯全景深度 | **混合方案** |
|------|---------------|-----------|-------------|
| 点云密度 | 6-8万/帧 | N/A | 12万/帧 (LiDAR) |
| 体素观测率 | 30-40% | 60-80% | **80-95%** |
| 深度精度 | ±5cm | ±20cm | **±5cm** |
| 语义一致性 | 中 | 高 | **高** |
| 边界清晰度 | 低 | 中 | **高** |
| 计算时间 | 0.1s | 2.0s | **0.5s** |

### 5.2 定性改进

**场景1: 远处车辆**
- 纯LiDAR: 稀疏点云，轮廓不清
- 混合方案: LiDAR提供精确距离，全景语义填充完整车身

**场景2: 行人**
- 纯LiDAR: 仅几个点，无法识别
- 混合方案: 语义图识别完整人形轮廓，LiDAR确定位置

**场景3: 交通标志**
- 纯LiDAR: 可能漏检（小目标）
- 混合方案: 语义图覆盖，LiDAR提供准确距离

---

## 六、风险与备选方案

### 6.1 潜在问题

**问题1: 激光雷达和相机不同步**
- **影响**: 运动物体位置不匹配
- **解决**: 使用FrameSynchronizer，容差10ms

**问题2: 深度图稀疏区域**
- **影响**: 某些角度LiDAR无数据，查询返回inf
- **解决**:
  - 检测`d_lidar > 99.0`，标记为未知
  - 或使用最近邻插值填补小空洞

**问题3: 计算性能**
- **影响**: 遍历64万体素耗时
- **优化**:
  - 向量化球面坐标计算
  - 预先构建查找表
  - 可选：多进程并行处理

### 6.2 备选方案

**方案A: 纯LiDAR + 最近邻插值**
- 如果全景相机性能不够
- 用空间插值填补LiDAR间隙
- 语义来自LiDAR自身

**方案B: 分层处理**
- 近距离(<20m): 混合方法
- 远距离(>20m): 纯全景语义
- 折中性能和质量

---

## 七、总结

### 7.1 核心创新点

1. **深度与语义解耦**: LiDAR负责深度，全景图负责语义
2. **双向查询**: 遍历体素，查询两个数据源
3. **三态体素**: 表面/自由/未知，支持完整场景重建

### 7.2 实施优先级

**P0 (第一步)**: 移植全景语义相机代码
- 6个语义相机
- CubeMap拼接工具
- 测试生成1024×512语义全景图

**P1 (第二步)**: 升级激光雷达
- 128线配置
- 验证点云覆盖率

**P2 (第三步)**: 实现混合算法
- SphericalDepthMap
- HybridOccupancyGenerator
- 单元测试

**P3 (第四步)**: 集成测试
- 修改DataCollector
- 完整采集流程
- 质量验证

---

**文档状态**: ✅ 方案设计完成
**下一步**: 开始实施第一步 - 移植全景语义代码

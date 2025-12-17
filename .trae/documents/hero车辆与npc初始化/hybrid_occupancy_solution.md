# LiDAR + 360°全景融合：稠密体素生成方案

> 激光雷达管深度精度，全景图管覆盖密度，射线投射填充体素

---

## 一、核心思想（一句话）

```
遍历全景图52万像素 → 每个像素发射一条射线 → 沿射线填充体素（自由空间+表面）
```

---

## 二、数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                         传感器层                                 │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  语义激光雷达    │  全景深度相机    │  全景语义相机                │
│  128线 12万点   │  1024×512       │  1024×512                   │
│  精确深度       │  补充深度        │  稠密语义                    │
└────────┬────────┴────────┬────────┴──────────┬──────────────────┘
         │                 │                   │
         ▼                 ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                       深度融合                                   │
│                                                                 │
│   depth[u,v] = LiDAR有效 ? LiDAR深度 : 全景深度                  │
│                                                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      射线投射填充                                │
│                                                                 │
│   for 每个像素(u,v):                                            │
│       direction = pixel_to_ray(u, v)                            │
│       depth = fused_depth[v, u]                                 │
│       label = semantic_pano[v, u]                               │
│                                                                 │
│       沿射线填充:                                                │
│         [0, depth) → 自由空间 (label=0)                         │
│         [depth]    → 表面 (label=语义)                          │
│         (depth, ∞) → 不处理（遮挡）                              │
│                                                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         输出                                     │
│                                                                 │
│   occupancy: (200, 200, 16) uint8  → 语义标签                    │
│   mask:      (200, 200, 16) bool   → 观测有效性                  │
│                                                                 │
│   覆盖率: 95%+                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、核心算法

### 3.1 深度融合

```python
def fuse_depth(lidar_depth_map, pano_depth):
    """
    融合激光雷达深度和全景深度
    
    策略：激光雷达优先，全景补充
    
    Args:
        lidar_depth_map: (H, W) 激光雷达投影的深度图，无数据处为inf
        pano_depth: (H, W) 全景深度相机的深度图
    
    Returns:
        fused: (H, W) 融合后的深度图
        source: (H, W) 深度来源标记 (0=无效, 1=LiDAR, 2=全景)
    """
    H, W = lidar_depth_map.shape
    fused = np.zeros((H, W), dtype=np.float32)
    source = np.zeros((H, W), dtype=np.uint8)
    
    # 激光雷达有效区域
    lidar_valid = lidar_depth_map < 99.0
    
    # 全景深度有效区域
    pano_valid = (pano_depth > 0.1) & (pano_depth < 100.0)
    
    # 优先使用激光雷达
    fused[lidar_valid] = lidar_depth_map[lidar_valid]
    source[lidar_valid] = 1
    
    # 激光雷达无效处，用全景补充
    pano_fill = ~lidar_valid & pano_valid
    fused[pano_fill] = pano_depth[pano_fill]
    source[pano_fill] = 2
    
    return fused, source
```

### 3.2 激光雷达点云→深度图

```python
def lidar_to_depth_map(points, pano_size=(1024, 512)):
    """
    将激光雷达点云投影到全景深度图
    
    Args:
        points: (N, 3) 点云坐标 [x_front, y_left, z_up]
        pano_size: (W, H) 全景图尺寸
    
    Returns:
        depth_map: (H, W) 深度图，无数据处为inf
    """
    W, H = pano_size
    depth_map = np.full((H, W), np.inf, dtype=np.float32)
    
    # 计算深度和方向
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    depth = np.sqrt(x**2 + y**2 + z**2)
    
    # 转球面坐标
    # theta: 水平角 [0, 2π], 0=前, π/2=左, π=后, 3π/2=右
    # phi: 垂直角 [-π/2, π/2], 0=水平, +π/2=上
    theta = np.arctan2(y, x)  # [-π, π]
    theta = (theta + 2 * np.pi) % (2 * np.pi)  # [0, 2π]
    phi = np.arctan2(z, np.sqrt(x**2 + y**2))  # [-π/2, π/2]
    
    # 转像素坐标
    u = (theta / (2 * np.pi) * W).astype(np.int32)
    v = ((0.5 - phi / np.pi) * H).astype(np.int32)
    
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    
    # 填充深度图（取最近值）
    for i in range(len(points)):
        if depth[i] < depth_map[v[i], u[i]]:
            depth_map[v[i], u[i]] = depth[i]
    
    return depth_map
```

### 3.3 射线投射填充（核心！）

```python
def ray_cast_voxelize(fused_depth, semantic_pano, voxel_config):
    """
    射线投射法生成稠密体素
    
    核心思想：
    - 遍历全景图每个像素
    - 每个像素对应一条从原点出发的射线
    - 沿射线填充体素：深度前=自由，深度处=语义
    
    Args:
        fused_depth: (H, W) 融合深度图
        semantic_pano: (H, W) 全景语义图
        voxel_config: 体素配置
    
    Returns:
        occupancy: (X, Y, Z) 体素语义
        mask: (X, Y, Z) 观测掩码
    """
    H, W = fused_depth.shape
    
    x_range = voxel_config['x_range']  # [-50, 50]
    y_range = voxel_config['y_range']  # [-50, 50]
    z_range = voxel_config['z_range']  # [-4, 4]
    res = voxel_config['resolution']    # 0.5
    
    # 网格尺寸
    X = int((x_range[1] - x_range[0]) / res)  # 200
    Y = int((y_range[1] - y_range[0]) / res)  # 200
    Z = int((z_range[1] - z_range[0]) / res)  # 16
    
    # 初始化
    occupancy = np.zeros((X, Y, Z), dtype=np.uint8)
    mask = np.zeros((X, Y, Z), dtype=np.bool_)
    
    # 预计算每个像素的射线方向
    u_coords = np.arange(W)
    v_coords = np.arange(H)
    
    # 像素 → 球面角度
    theta = u_coords / W * 2 * np.pi          # [0, 2π]
    phi = (0.5 - v_coords / H) * np.pi        # [π/2, -π/2]
    
    # 球面角度 → 方向向量
    # dir: [x_front, y_left, z_up]
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # 遍历每个像素
    for vi, p in enumerate(phi):
        for ui, t in enumerate(theta):
            depth = fused_depth[vi, ui]
            
            # 跳过无效深度
            if depth <= 0.1 or depth >= 100.0:
                continue
            
            label = semantic_pano[vi, ui]
            
            # 射线方向
            dir_x = cos_phi[vi] * cos_theta[ui]  # 前
            dir_y = cos_phi[vi] * sin_theta[ui]  # 左
            dir_z = sin_phi[vi]                   # 上
            
            # 沿射线采样
            # 采样步长 = 体素分辨率的一半，确保不漏
            step = res * 0.5
            num_steps = int(depth / step) + 1
            
            for i in range(num_steps):
                t_dist = i * step
                
                # 当前点坐标
                px = dir_x * t_dist
                py = dir_y * t_dist
                pz = dir_z * t_dist
                
                # 转体素索引
                gx = int((px - x_range[0]) / res)
                gy = int((py - y_range[0]) / res)
                gz = int((pz - z_range[0]) / res)
                
                # 边界检查
                if not (0 <= gx < X and 0 <= gy < Y and 0 <= gz < Z):
                    continue
                
                # 判断是自由空间还是表面
                if t_dist < depth - res * 0.5:
                    # 自由空间
                    if not mask[gx, gy, gz]:  # 未被标记过
                        occupancy[gx, gy, gz] = 0
                        mask[gx, gy, gz] = True
                else:
                    # 表面（深度附近）
                    occupancy[gx, gy, gz] = label
                    mask[gx, gy, gz] = True
                    break  # 表面之后不再处理
    
    return occupancy, mask
```

### 3.4 向量化加速版本（生产用）

```python
import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def ray_cast_voxelize_fast(fused_depth, semantic_pano,
                            x_min, x_max, y_min, y_max, z_min, z_max,
                            resolution):
    """
    Numba加速的射线投射体素化
    
    速度提升: 约50-100倍
    """
    H, W = fused_depth.shape
    
    X = int((x_max - x_min) / resolution)
    Y = int((y_max - y_min) / resolution)
    Z = int((z_max - z_min) / resolution)
    
    occupancy = np.zeros((X, Y, Z), dtype=np.uint8)
    mask = np.zeros((X, Y, Z), dtype=np.uint8)  # numba不支持bool
    
    step = resolution * 0.5
    
    # 并行遍历像素
    for vi in prange(H):
        phi = (0.5 - vi / H) * np.pi
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        for ui in range(W):
            depth = fused_depth[vi, ui]
            
            if depth <= 0.1 or depth >= 100.0:
                continue
            
            label = semantic_pano[vi, ui]
            
            theta = ui / W * 2 * np.pi
            dir_x = cos_phi * np.cos(theta)
            dir_y = cos_phi * np.sin(theta)
            dir_z = sin_phi
            
            num_steps = int(depth / step) + 1
            
            for i in range(num_steps):
                t_dist = i * step
                
                px = dir_x * t_dist
                py = dir_y * t_dist
                pz = dir_z * t_dist
                
                gx = int((px - x_min) / resolution)
                gy = int((py - y_min) / resolution)
                gz = int((pz - z_min) / resolution)
                
                if gx < 0 or gx >= X or gy < 0 or gy >= Y or gz < 0 or gz >= Z:
                    continue
                
                if t_dist < depth - resolution * 0.5:
                    if mask[gx, gy, gz] == 0:
                        occupancy[gx, gy, gz] = 0
                        mask[gx, gy, gz] = 1
                else:
                    occupancy[gx, gy, gz] = label
                    mask[gx, gy, gz] = 1
                    break
    
    return occupancy, mask.astype(np.bool_)
```

---

## 四、完整生成器类

```python
import numpy as np

class HybridOccupancyGenerator:
    """
    LiDAR + 全景融合的稠密体素生成器
    """
    
    def __init__(self, 
                 x_range=(-50, 50),
                 y_range=(-50, 50),
                 z_range=(-4, 4),
                 resolution=0.5,
                 pano_size=(1024, 512)):
        
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution
        self.pano_size = pano_size
        
        self.grid_size = (
            int((x_range[1] - x_range[0]) / resolution),
            int((y_range[1] - y_range[0]) / resolution),
            int((z_range[1] - z_range[0]) / resolution),
        )
        
        # CARLA语义标签映射
        self.label_map = {
            0: 0,    # Unlabeled → empty
            1: 14,   # Building
            4: 6,    # Pedestrian
            7: 9,    # Road
            8: 10,   # Sidewalk
            9: 12,   # Vegetation
            10: 1,   # Vehicles → car
            12: 16,  # TrafficSign
            # ... 其他
        }
    
    def generate(self, lidar_points, pano_depth, pano_semantic):
        """
        生成稠密体素
        
        Args:
            lidar_points: (N, 3) 激光雷达点云 [x, y, z]
            pano_depth: (H, W) 全景深度图，米
            pano_semantic: (H, W) 全景语义图，CARLA标签
        
        Returns:
            occupancy: (200, 200, 16) 体素语义
            mask: (200, 200, 16) 观测掩码
        """
        W, H = self.pano_size
        
        # 1. 激光雷达点云 → 深度图
        print("构建激光雷达深度图...")
        lidar_depth = self._lidar_to_depth_map(lidar_points)
        
        # 2. 深度融合
        print("融合深度...")
        fused_depth, source = self._fuse_depth(lidar_depth, pano_depth)
        
        lidar_ratio = np.sum(source == 1) / np.sum(source > 0) * 100
        print(f"  激光雷达覆盖: {lidar_ratio:.1f}%")
        print(f"  全景补充: {100 - lidar_ratio:.1f}%")
        
        # 3. 语义标签映射
        mapped_semantic = self._map_labels(pano_semantic)
        
        # 4. 射线投射体素化
        print("射线投射填充...")
        occupancy, mask = self._ray_cast(fused_depth, mapped_semantic)
        
        coverage = np.sum(mask) / mask.size * 100
        occupied = np.sum(occupancy > 0)
        print(f"  体素覆盖率: {coverage:.1f}%")
        print(f"  非空体素: {occupied:,}")
        
        return occupancy, mask
    
    def _lidar_to_depth_map(self, points):
        """点云→深度图"""
        W, H = self.pano_size
        depth_map = np.full((H, W), np.inf, dtype=np.float32)
        
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        depth = np.sqrt(x**2 + y**2 + z**2)
        
        # 过滤无效点
        valid = depth > 0.1
        x, y, z, depth = x[valid], y[valid], z[valid], depth[valid]
        
        theta = np.arctan2(y, x)
        theta = (theta + 2 * np.pi) % (2 * np.pi)
        phi = np.arctan2(z, np.sqrt(x**2 + y**2))
        
        u = (theta / (2 * np.pi) * W).astype(np.int32)
        v = ((0.5 - phi / np.pi) * H).astype(np.int32)
        
        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)
        
        # 取最近深度
        for i in range(len(depth)):
            if depth[i] < depth_map[v[i], u[i]]:
                depth_map[v[i], u[i]] = depth[i]
        
        return depth_map
    
    def _fuse_depth(self, lidar_depth, pano_depth):
        """深度融合"""
        fused = np.zeros_like(lidar_depth)
        source = np.zeros(lidar_depth.shape, dtype=np.uint8)
        
        lidar_valid = lidar_depth < 99.0
        pano_valid = (pano_depth > 0.1) & (pano_depth < 100.0)
        
        # 激光雷达优先
        fused[lidar_valid] = lidar_depth[lidar_valid]
        source[lidar_valid] = 1
        
        # 全景补充
        fill_mask = ~lidar_valid & pano_valid
        fused[fill_mask] = pano_depth[fill_mask]
        source[fill_mask] = 2
        
        return fused, source
    
    def _map_labels(self, semantic):
        """CARLA标签 → Occupancy标签"""
        mapped = np.zeros_like(semantic)
        for carla_id, occ_id in self.label_map.items():
            mapped[semantic == carla_id] = occ_id
        return mapped
    
    def _ray_cast(self, depth, semantic):
        """射线投射体素化"""
        # 使用Numba加速版本
        try:
            return ray_cast_voxelize_fast(
                depth, semantic,
                self.x_range[0], self.x_range[1],
                self.y_range[0], self.y_range[1],
                self.z_range[0], self.z_range[1],
                self.resolution
            )
        except:
            # 回退到纯Python版本
            return ray_cast_voxelize(
                depth, semantic,
                {'x_range': self.x_range, 'y_range': self.y_range,
                 'z_range': self.z_range, 'resolution': self.resolution}
            )
```

---

## 五、CARLA传感器配置

```python
# 1. 语义激光雷达（128线）
LIDAR_CONFIG = {
    'channels': 128,
    'range': 100.0,
    'points_per_second': 2400000,
    'rotation_frequency': 20,
    'upper_fov': 30.0,
    'lower_fov': -40.0,
}

# 2. 全景CubeMap相机（6个面）
CUBEMAP_CONFIG = {
    'size': 512,        # 每个面 512×512
    'fov': 90,          # 必须90°无缝拼接
    'faces': {
        'front':  (0, 0, 0),      # pitch, yaw, roll
        'back':   (0, 180, 0),
        'left':   (0, -90, 0),
        'right':  (0, 90, 0),
        'up':     (-90, 0, 0),
        'down':   (90, 0, 0),
    }
}

# 3. 全景图输出尺寸
PANORAMA_SIZE = (1024, 512)  # 宽×高
```

---

## 六、使用示例

```python
# 采集数据
lidar_points = lidar_sensor.get_points()      # (N, 3)
pano_depth = pano_manager.get_depth()          # (512, 1024)
pano_semantic = pano_manager.get_semantic()    # (512, 1024)

# 生成体素
generator = HybridOccupancyGenerator()
occupancy, mask = generator.generate(lidar_points, pano_depth, pano_semantic)

# 保存
np.savez_compressed('frame_000000.npz', 
                    occupancy=occupancy,  # (200, 200, 16)
                    mask=mask)            # (200, 200, 16)
```

---

## 七、效果预期

| 指标 | 纯LiDAR | 纯全景 | **融合方案** |
|------|--------|--------|-------------|
| 深度精度 | ±5cm | ±20cm | **±5cm (LiDAR区) / ±20cm (补充区)** |
| 体素覆盖率 | 30-40% | 70-80% | **95%+** |
| 边界清晰度 | 高 | 中 | **高** |
| 计算时间 | 0.1s | 0.5s | **0.3s** (Numba加速) |

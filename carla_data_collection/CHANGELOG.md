# 数据采集系统更新日志

## [修复版本 v2] - 2025-12-16

### 🐛 Bug 修复

#### 1. 修复车型硬编码导致崩溃
**问题**:
- 脚本硬编码 `vehicle.tesla.model3`,但该车型在某些 CARLA 配置中不存在
- 导致运行时崩溃: `RuntimeError: blueprint 'vehicle.tesla.model3' not found`

**修复**:
```python
# 智能车型选择,按优先级尝试多个候选
vehicle_candidates = [
    'vehicle.tesla.model3',
    'vehicle.lincoln.mkz_2020',
    'vehicle.lincoln.mkz',
    'vehicle.audi.tt',
    'vehicle.dodge.charger_2020'
]

# 如果都不可用,则从所有 4 轮车辆中随机选择
```

**文件**: [collect_5_frames.py:133-156](scripts/collect_5_frames.py#L133-L156)

---

#### 2. 修复后置摄像头全黑问题
**问题**:
- 后置摄像头位置 `x=-1.8, z=1.0` 可能被车身遮挡
- 导致图像全黑,像素范围 `[0, 0]`

**修复**:
```python
# 调整后置摄像头位置
cam_transform2 = carla.Transform(
    carla.Location(x=-2.5, z=1.5),  # 向后和向上移动,避免遮挡
    carla.Rotation(yaw=180)
)
```

**文件**: [collect_5_frames.py:199-202](scripts/collect_5_frames.py#L199-L202)

---

#### 3. 添加数据保存到磁盘功能
**问题**:
- 脚本仅将数据存储在内存 `collected_frames` 列表中
- 程序结束后没有保存任何文件到磁盘

**修复**:
- 添加相机图像保存 (PNG 格式)
- 添加激光雷达点云保存 (NPZ 压缩格式)
- 添加数据集元数据保存 (calibration.json)

```python
# 保存相机图像
for cam_id, cam_data in camera_data.items():
    img = Image.fromarray(cam_data['data'])
    img_path = cameras_dir / cam_id / f"{frame_idx:06d}.png"
    img.save(img_path)

# 保存激光雷达点云
lidar_path = lidar_dir / f"{frame_idx:06d}.npz"
np.savez_compressed(
    lidar_path,
    points=lidar_data['points'],
    frame=lidar_data['frame'],
    timestamp=lidar_data['timestamp']
)

# 保存标定信息
calibration_path = output_dir / "calibration.json"
with open(calibration_path, 'w', encoding='utf-8') as f:
    json.dump(calibration, f, indent=2, ensure_ascii=False)
```

**文件**: [collect_5_frames.py:304-390](scripts/collect_5_frames.py#L304-L390)

---

### 📁 输出目录结构

```
dataset_output/town10_test/
├── cameras/
│   ├── cam_front/
│   │   ├── 000000.png  (1280×960 RGB)
│   │   ├── 000001.png
│   │   └── ...
│   └── cam_rear/
│       ├── 000000.png
│       └── ...
├── lidar/
│   ├── 000000.npz  (points: N×6, frame, timestamp)
│   ├── 000001.npz
│   └── ...
└── calibration.json  (相机/激光雷达参数)
```

---

### 📋 验证清单

运行脚本后,请检查:

- [ ] 脚本成功连接 CARLA 服务器
- [ ] Hero 车辆成功生成 (显示使用的车型)
- [ ] 2 个相机传感器成功附加
- [ ] 语义激光雷达成功附加
- [ ] NPC 车辆成功生成
- [ ] 成功采集 5 帧数据
- [ ] `cam_front` 像素范围不是 [0, 0]
- [ ] `cam_rear` 像素范围不是 [0, 0] ← **修复验证点**
- [ ] `dataset_output/town10_test/` 目录存在
- [ ] `cameras/cam_front/` 包含 5 张 PNG 图像
- [ ] `cameras/cam_rear/` 包含 5 张 PNG 图像 ← **修复验证点**
- [ ] `lidar/` 包含 5 个 NPZ 文件
- [ ] `calibration.json` 文件存在并包含完整参数

---

### 🚧 已知问题 (待实现)

1. **体素化未实现**: 激光雷达点云已保存,但未转换为 Occupancy 体素网格
2. **仅 2 相机**: 完整系统需要 8 相机,当前仅实现前/后 2 个相机测试
3. **语义标签未映射**: 激光雷达返回 CARLA 23 类,需映射到 Occupancy 18 类
4. **Windows 进程清理**: 程序退出时可能崩溃 (CARLA Windows 已知问题,不影响数据)

---

### 📖 相关文档更新

- [README.md](README.md) - 添加输出结构、修复说明、待实现功能
- [运行测试.txt](运行测试.txt) - 更新预期输出和验证清单

---

## [初始版本 v1] - 2025-12-15

### ✨ 初始功能

- CARLA 客户端连接
- Hero 车辆生成
- 2 相机传感器 (前置 120° 鱼眼 + 后置 90° 广角)
- 语义激光雷达传感器
- NPC 车辆生成 (20 辆 + Traffic Manager)
- 同步模式数据采集 (20 Hz)
- 帧同步 (Queue 模式)
- 5 帧限制采集
- 自动清理资源

### 🐛 已知问题 (v1)

- ❌ 硬编码车型导致崩溃
- ❌ 后置摄像头全黑
- ❌ 数据未保存到磁盘

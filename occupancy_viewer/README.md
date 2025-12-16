# 🚗 CARLA Occupancy 3D 体素可视化查看器

基于 Three.js 的交互式 3D 体素查看器,用于可视化 CARLA 数据采集系统生成的 Occupancy 数据。

---

## ✨ 功能特性

- ✅ **3D 体素渲染**: 使用 Three.js 高性能渲染引擎
- ✅ **交互式控制**: 鼠标旋转/缩放/平移
- ✅ **语义类别可视化**: 18 种语义类别,不同颜色区分
- ✅ **多视角切换**: 俯视图/前视图/侧视图/自由视角
- ✅ **帧浏览**: 快速切换不同帧的体素数据
- ✅ **实时统计**: 显示体素数量、类别分布等信息
- ✅ **自动加载**: 配合 Python 后端自动加载数据集

---

## 🚀 启动指南

为了解决浏览器安全限制（CORS）和路径访问问题，本项目提供了一个专用的 Python 服务器脚本。

### 1. 启动服务器

在命令行中运行以下命令：

```bash
# 激活环境 (如果需要)
conda activate carla

# 运行启动脚本
python d:\code\carla\occupancy_viewer\run_viewer.py
```

服务器启动后会显示：
```
============================================================
Occupancy Viewer Server
============================================================
Viewer Directory: d:\code\carla\occupancy_viewer
Data Directory:   d:\code\carla\dataset_output\town10_test\occupancy
URL:              http://localhost:8000/
============================================================
Serving at port 8000...
```

### 2. 访问查看器

打开浏览器（推荐 Chrome 或 Edge），访问：

👉 **[http://localhost:8000/](http://localhost:8000/)**

查看器会自动加载 `dataset_output` 目录下的数据，无需手动选择文件。

---

## 🎮 操作指南

### 鼠标控制

| 操作 | 功能 |
|------|------|
| **左键拖拽** | 旋转视角 (绕中心点) |
| **右键拖拽** | 平移场景 |
| **滚轮滚动** | 缩放视图 (拉近/拉远) |
| **双击** | 重置视角到默认位置 |

### 界面功能

- **帧列表**: 点击左侧列表切换不同时刻的数据
- **视图按钮**: 快速切换 俯视图 / 前视图 / 侧视图 / 自由视角
- **图例**: 查看不同颜色代表的物体类别

---

## 📁 数据格式

查看器支持标准的 CARLA Occupancy NPZ 格式:

```python
# .npz 文件内容
{
    'occupancy': np.uint8[X, Y, Z],  # 体素标签 (0-17)
    'mask': np.bool[X, Y, Z],        # 有效观测掩码
    'x_range': [float, float],       # X 范围 (米)
    'y_range': [float, float],       # Y 范围 (米)
    'z_range': [float, float],       # Z 范围 (米)
    'resolution': float,             # 分辨率 (米)
    'grid_size': [int, int, int]     # 网格尺寸
}
```

---

## 🔧 技术实现

- **前端**: Three.js (r160), OrbitControls, fflate (NPZ 解压)
- **后端**: Python `http.server` (提供静态文件和数据 API)
- **渲染**: InstancedMesh (实例化渲染) 优化性能

---

## ❓ 常见问题

**Q: 为什么显示 "Failed to fetch"?**
A: 请确保您是通过 `run_viewer.py` 启动的，并且保持命令行窗口开启。不要直接双击 `index.html` 打开。

**Q: 数据目录不对怎么办？**
A: 修改 `run_viewer.py` 中的 `DATA_DIR` 变量指向您的数据目录。

**Q: 渲染卡顿?**
A: 正常现象，体素数量较多时可能会卡顿。尝试缩小浏览器窗口或使用性能更好的显卡。

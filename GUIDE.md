# CARLA UE5.5 完整指南 (VS 2026)

本指南适用于使用 Visual Studio 2026 编译和运行 CARLA UE5.5 版本。

---

## 目录

- [快速开始](#快速开始)
- [构建指南](#构建指南)
- [问题排查](#问题排查)
- [性能优化](#性能优化)
- [技术修复说明](#技术修复说明)

---

## 快速开始

### 步骤 1: 启动 CARLA 服务器

```cmd
start_carla_server.bat
```

等待：
- Unreal Editor 打开
- 着色器编译完成（首次运行：5-15 分钟）
- 项目加载完成

然后：**点击绿色 Play 按钮** (▶️)

验证：左上角能看到 FPS 计数器

### 步骤 2: 运行示例脚本

```cmd
run_carla_examples.bat
```

**推荐首次运行**: Manual Control (选项 1)
- 使用键盘驾驶汽车 (WASD)
- 按 P 键启用自动驾驶
- 按 ESC 退出

### 步骤 3: 测试连接（如有问题）

```cmd
conda activate carla
python test_connection.py
```

成功：显示 "Connection Successful!"
失败：检查 Editor 中是否点击了 Play 按钮

---

## 文件说明

### 核心脚本

| 文件 | 说明 |
|------|------|
| `BUILD_FINAL.bat` | 编译整个项目（仅需运行一次） |
| `start_carla_server.bat` | 启动 Unreal Editor 服务器 |
| `run_carla_examples.bat` | 运行 Python 示例脚本 |
| `test_connection.py` | 测试服务器连接 |

### 文档

| 文件 | 说明 |
|------|------|
| `GUIDE.md` | 本文档 - 完整指南 |
| `CLAUDE.md` | Claude Code 项目配置说明 |

---

## 可用示例

### 1. Manual Control
- **用途**: 手动驾驶汽车
- **命令**: `python manual_control.py`
- **控制**:
  - `WASD` - 驾驶
  - `Space` - 刹车
  - `Q` - 切换倒档
  - `P` - 自动驾驶
  - `TAB` - 切换相机视角
  - `C` - 改变天气
  - `ESC` - 退出

### 2. Automatic Control
- **用途**: 观看 AI 自动驾驶
- **命令**: `python automatic_control.py`
- **说明**: AI 会自动导航驾驶

### 3. Generate Traffic
- **用途**: 生成 AI 车辆和行人
- **命令**: `python generate_traffic.py -n 30 -w 10`
- **参数**:
  - `-n 30` - 生成 30 辆车
  - `-w 10` - 生成 10 个行人

### 4. Sensor Visualization
- **用途**: 可视化传感器数据（相机、激光雷达）
- **命令**: `python sensor_synchronization.py`

### 5. Vehicle Gallery
- **用途**: 浏览所有可用车辆
- **命令**: `python vehicle_gallery.py`

---

## 构建指南

### 前置条件

1. **Visual Studio 2026 (v18)** 已安装
2. **Anaconda** 环境 `carla` 已创建
3. **Unreal Engine 5.5** 源代码已编译

### 构建步骤

#### 步骤 1: 打开 VS 命令提示符

- 按 Windows 键
- 搜索 "x64 Native Tools Command Prompt for VS 18"
- 打开它

#### 步骤 2: 运行构建脚本

```cmd
cd D:\code\carla
BUILD_FINAL.bat
```

此脚本会：
1. 加载 VS 2026 环境
2. 激活 `carla` conda 环境
3. 配置 CMake（包含 /wd4723 修复）
4. 编译 Editor 插件

#### 步骤 3: 启动服务器

```cmd
start_carla_server.bat
```

等待着色器编译完成，然后点击 Play 按钮

#### 步骤 4: 测试连接

```cmd
conda activate carla
python test_connection.py
```

看到 "Connection Successful!" 表示成功！

---

## 问题排查

### 问题 1: RuntimeError: time-out while waiting for the simulator

**错误信息**:
```
RuntimeError: time-out of 2000ms while waiting for the simulator
```

**原因**:
- Unreal Editor 运行中但**未点击 Play 按钮**
- 或者点击了 Play 但按了 Esc**暂停**了场景

**解决方案**:
1. 检查 Unreal Editor 窗口
2. 查看顶部工具栏的绿色 Play 按钮
3. 如果显示 ▶️ (Play)，点击它
4. 如果显示 ⏸️ (Pause)，再次点击取消暂停
5. 等待场景加载（3-10 秒）
6. 确认：左上角应该能看到 FPS 计数器
7. 重新运行 Python 脚本

---

### 问题 2: Pygame 窗口显示黑屏

**原因**:
- 正常现象 - pygame 正在等待服务器数据
- 场景仍在加载中

**解决方案**:
- 等待 10-30 秒
- 检查 Unreal Editor 是否点击了 Play 按钮
- 如果 1 分钟后仍然黑屏，重启服务器和脚本

---

### 问题 3: 有些示例运行正常，有些不行

**原因**:
- 不同示例的超时设置不同
- 有些使用 2.0s（对 UE5 太短）
- 有些使用 10.0s（更好）

**解决方案**:

**方法 1**（快速）:
- 确保服务器运行并点击了 Play
- 先运行 `test_connection.py` 验证连接

**方法 2**（永久）:
- 编辑示例脚本
- 修改: `client.set_timeout(2.0)`
- 改为: `client.set_timeout(10.0)`

---

### 问题 4: "Connection refused" 错误

**原因**:
- Unreal Editor 根本没有运行
- 或者防火墙阻止了端口 2000

**解决方案**:
1. 运行 `start_carla_server.bat`
2. 等待 Unreal Editor 完全加载
3. 点击 Play 按钮
4. 检查防火墙：允许 UnrealEditor.exe 使用端口 2000

---

### 问题 5: Traffic Manager "bind error"

**错误信息**:
```
RuntimeError: trying to create rpc server for traffic manager;
but the system failed to create because of bind error.
```

**原因**:
- Traffic Manager 端口 8000 已被占用
- 通常发生在先运行 manual_control.py 后运行 automatic_control.py 时
- 或者另一个 CARLA Python 客户端仍在运行

**解决方案**:

**方法 1**（推荐）:
1. 关闭所有 CARLA Python 脚本
2. 检查任务管理器中的 `python.exe` 进程
3. 终止所有运行 CARLA 示例的 Python 进程
4. 重启 Unreal Editor（清理服务端 TM）
5. 点击 Play 按钮
6. 再次运行 automatic_control.py

**方法 2**（快速）:
1. 关闭 Unreal Editor
2. 终止所有 python.exe 进程
3. 运行 `start_carla_server.bat`
4. 点击 Play 按钮
5. 运行 `run_carla_examples.bat` → 选择 4 (Automatic Control)

**重要提示**:
- 与 manual_control.py 不同，automatic_control.py **必须**使用 Traffic Manager
- 同一时间**只能有一个** Traffic Manager 在端口 8000 上运行
- 运行 automatic_control 前始终关闭之前的示例

---

### 问题 6: Editor 点击 Play 时崩溃

**原因**:
- 插件编译不正确
- 或者缺少 DLL 文件

**解决方案**:
1. 检查: `Unreal\CarlaUnreal\Plugins\Carla\Binaries\Win64`
2. 应该包含 DLL 文件
3. 如果缺失：重新运行 `BUILD_FINAL.bat`
4. 检查构建日志是否有错误

---

### 问题 7: 画面又黑又暗又卡顿

**症状**:
- 画面非常暗，只能看到最亮的交通灯
- 运行非常卡顿
- 光照不自然

**原因**:
1. 启用了硬件光线追踪（对 GPU 要求极高）
2. Lumen 配置不当导致光照过暗
3. Auto Exposure 配置不完整

**解决方案**:

已自动优化配置文件 `DefaultEngine.ini`，**需要重启 Unreal Editor 才能生效！**

1. 完全关闭 Unreal Editor
2. 关闭所有 Python 脚本窗口
3. 运行 `start_carla_server.bat`
4. 等待加载，点击 Play 按钮
5. 运行 `run_carla_examples.bat`

详见 [性能优化](#性能优化) 章节。

---

### 问题 8: automatic_control.py 光照很暗

**症状**:
- manual_control.py 光照正常 ✅
- automatic_control.py 画面很暗 ❌
- 重启 Editor 无效

**原因**:
`automatic_control.py` 的相机配置缺少：
- Gamma 校正参数
- Post-Process Profile 配置
- 相机属性设置

**解决方案**:

已自动修复 `automatic_control.py`，添加了：
- Gamma 校正 (默认 2.2)
- Post-Process Profile (Town10HD_Opt 专用)
- 相机属性设置

**无需重启 Editor**，直接运行即可生效。

详见 [技术修复说明](#技术修复说明) 章节。

---

## 诊断检查清单

在运行任何 Python 脚本之前，验证：

- [ ] Unreal Editor 已打开
- [ ] 项目已加载: CarlaUnreal.uproject
- [ ] 绿色 Play 按钮已点击 (▶️)
- [ ] 场景正在运行（未暂停）
- [ ] 左上角显示 FPS 计数器
- [ ] 端口 2000 未被防火墙阻止

如果以上全部确认但仍然失败：
- 运行 `python test_connection.py`
- 如果测试通过，问题在于脚本
- 如果测试失败，重启 Unreal Editor

---

## 性能优化

### 已优化配置 (DefaultEngine.ini)

#### 禁用硬件光线追踪（性能优化）

```ini
r.RayTracing=False                          # 原: True
r.Lumen.HardwareRayTracing=False            # 原: True
r.Lumen.HardwareRayTracing.LightingMode=0   # 原: 3
r.RayTracing.Shadows=False                  # 原: True
r.Lumen.TraceMeshSDFs=0                     # 原: 1
```

#### 增强自动曝光（光照修复）

```ini
r.DefaultFeature.AutoExposure=True          # 原: False
r.DefaultFeature.AutoExposure.Method=1      # 新增
r.DefaultFeature.AutoExposure.Bias=1.0      # 新增
r.EyeAdaptation.EditorOnly=False            # 新增
r.EyeAdaptationQuality=2                    # 新增
```

#### 视觉增强

```ini
r.DefaultFeature.Bloom=True                               # 原: False
r.DefaultFeature.AmbientOcclusion=True                    # 原: False
r.DefaultFeature.AmbientOcclusionStaticFraction=True      # 原: False
```

#### 内存优化

```ini
r.ReflectionCaptureResolution=128           # 原: 256
r.SkinCache.SceneMemoryLimitInMB=512        # 原: 1024
r.Lumen.TraceDistanceScale=0.5              # 原: 1.0
```

### GPU 要求对比

| 配置 | 最低 | 推荐 | 最佳 |
|------|------|------|------|
| **优化前**（硬件光追） | RTX 2060 | RTX 3070+ | RTX 4070+ |
| **优化后**（软件光照） | GTX 1650 | RTX 3060 | RTX 3070+ |

### 预期效果

优化后你应该看到：

- ✅ 光照自然明亮
- ✅ 交通灯清晰可见
- ✅ 帧率大幅提升（2-5 倍）
- ✅ GPU 负载降低
- ✅ 自动曝光调整

### 进一步优化

如果仍然卡顿，可以在 Unreal Editor 中：

1. **降低图形质量**:
   - 设置 → Engine Scalability Settings → Low 或 Medium

2. **减少 NPC 数量**:
   - 使用: `generate_traffic.py -n 10`（而不是 30）

3. **降低分辨率**:
   - 设置 → Resolution → 1280x720

4. **关闭其他应用程序**

---

## 技术修复说明

### automatic_control.py 光照修复

#### 问题分析

通过对比 `manual_control.py` 和 `automatic_control.py`，发现后者缺少关键相机配置：

| 配置项 | manual_control.py | automatic_control.py |
|--------|-------------------|----------------------|
| Gamma 校正 | ✅ 有 (参数化) | ❌ 无 |
| Post-Process Profile | ✅ 有 | ❌ 无 |
| 相机属性设置 | ✅ 完整 | ❌ 不完整 |

#### 已修复内容

**文件**: `d:\code\carla\PythonAPI\examples\automatic_control.py`

**修改 1**: 添加 gamma_correction 参数
```python
# 第 576 行
# 原: def __init__(self, parent_actor, hud):
# 改: def __init__(self, parent_actor, hud, gamma_correction=2.2):
```

**修改 2**: 添加 post_process_profile 获取
```python
# 第 594-596 行 (新增)
world = self._parent.get_world()
map_name = world.get_map().name
post_process_profile = self.get_post_process_profile(map_name)
```

**修改 3**: 更新相机传感器配置
```python
# 第 600 行
# 原: ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
# 改: ['sensor.camera.rgb', cc.Raw, 'Camera RGB',
#      {'post_process_profile': post_process_profile}],
```

**修改 4**: 设置 gamma 和自定义属性
```python
# 第 615-618 行 (新增)
if blp.has_attribute('gamma'):
    blp.set_attribute('gamma', str(gamma_correction))
for attr_name, attr_value in item[3].items():
    blp.set_attribute(attr_name, attr_value)
```

**修改 5**: 添加 get_post_process_profile 方法
```python
# 第 666-670 行 (新增)
def get_post_process_profile(self, map_name: str) -> str:
    """Get the post-process profile for the current map"""
    if "Town10HD_Opt" in map_name:
        return "Town10HD_Opt"
    return "Default"
```

#### 技术解释

**1. Gamma 校正**
- 控制亮度的非线性映射
- 默认 gamma=1.0 对 UE5 Lumen 太暗
- gamma=2.2 是标准 sRGB 值，适合 UE5

**2. Post-Process Profile**
- Town10HD_Opt 是 UE5 高清地图
- 需要专门的后处理配置
- 包含曝光、色调映射、Bloom 等
- 不设置会使用 UE4 时代的默认配置

**3. 为什么 DefaultEngine.ini 修改无效?**
- DefaultEngine.ini 只影响 Editor 中的渲染
- Python 客户端的相机是**独立创建**的
- 需要在 Python 脚本中**显式设置**相机属性

#### Gamma 值调整

如需调整亮度，可修改 gamma 值：

| Gamma 值 | 效果 | 用途 |
|----------|------|------|
| 1.8 | 更亮 | 暗场景 |
| 2.2 | 标准 | 推荐（默认） |
| 2.4 | 稍暗 | 电影风格 |

推荐范围: **1.8 - 2.4**

---

## 性能提示

### 一般建议

1. 关闭不必要的应用程序
2. 降低 UE Editor 图形质量:
   - 设置 → Engine Scalability → Low
3. 使用较少车辆:
   - `generate_traffic.py -n 10`（而不是 30）
4. 降低编辑器分辨率

### 日志位置

- **Unreal**: `Unreal\CarlaUnreal\Saved\Logs\CarlaUnreal.log`
- **Python**: 脚本的错误输出

---

## 获取帮助

如果以上方法都不奏效：

1. **检查日志**:
   - Unreal 日志: `Unreal\CarlaUnreal\Saved\Logs\CarlaUnreal.log`
   - Python 错误输出

2. **搜索 GitHub Issues**:
   - https://github.com/carla-simulator/carla/issues

3. **查看官方文档**:
   - https://carla-ue5.readthedocs.io

---

## 构建问题

### cl.exe not found

**错误**: `ERROR: cl.exe not found! VS environment setup failed.`

**解决方案**:
- 确保 VS 2026 安装在: `C:\Program Files\Microsoft Visual Studio\18\Professional`

### divide by zero (C4723)

**错误**: 编译时出现 C4723 错误

**解决方案**:
- 确保 `BUILD_FINAL.bat` 的 `CXX_FLAGS` 包含 `/wd4723`
- 检查 `D:\code\UnrealEngine5_carla\Engine\Source\Runtime\Core\Public\Windows\WindowsPlatformCompilerSetup.h` 是否已修复

---

## 版本信息

- **CARLA**: UE5 dev branch (0.10.0)
- **Unreal Engine**: 5.5 (自定义 fork)
- **Visual Studio**: 2026 (v18)
- **Python**: 3.10.19
- **CMake**: 4.2.0
- **Conda 环境**: carla

---

## 许可证

本项目遵循 MIT 许可证。详见 CARLA 官方仓库。

---

**最后更新**: 2024

# Traffic Manager 端口绑定错误修复

## ❌ 错误信息

```
RuntimeError: trying to create rpc server for traffic manager;
but the system failed to create because of bind error.
```

## 🐛 根本原因

**问题**: CARLA Traffic Manager 尝试绑定到一个已被占用的端口 (默认 8010)

**可能原因**:
1. **之前的进程未清理**: 上一次运行崩溃或未正常退出,端口仍被占用
2. **多个 CARLA 实例**: 同时运行多个数据采集脚本
3. **端口冲突**: 其他程序占用了 8010 端口
4. **CARLA 内部问题**: Traffic Manager 初始化失败

---

## ✅ 修复方案

### 修复 1: 端口重试机制 (已实现)

**文件**: `dense_occupancy_collection/core/scenario_manager.py`

**修改位置**: Line 59-81 (Hero 车辆) 和 Line 147-154 (NPC 车辆)

**修复逻辑**:
```python
# Hero 车辆: 尝试多个端口
tm_ports_to_try = [self.tm_port, 8011, 8012, 8013, 8014, 8015]
autopilot_success = False

for port in tm_ports_to_try:
    try:
        self.hero_vehicle.set_autopilot(True, port)
        self.tm_port = port  # 更新成功的端口
        print(f"  ✓ 自动驾驶已启用 (TM Port: {port})")
        autopilot_success = True
        break
    except RuntimeError as e:
        if "bind error" in str(e):
            print(f"  ⚠ TM 端口 {port} 被占用,尝试下一个...")
            continue
        else:
            raise

if not autopilot_success:
    print(f"  ⚠ 警告: 无法启用自动驾驶 (所有端口都被占用)")
    print(f"  → 车辆将保持静止状态")
```

**NPC 车辆: 添加错误处理**:
```python
try:
    npc.set_autopilot(True, self.tm_port)
except RuntimeError as e:
    if "bind error" in str(e):
        print(f"  ⚠ NPC 自动驾驶失败 (TM 端口问题),车辆将保持静止")
    else:
        raise
```

**效果**:
- ✅ 自动尝试多个端口 (8010, 8011, 8012, 8013, 8014, 8015)
- ✅ 即使所有端口被占用,也不会崩溃 (车辆保持静止)
- ✅ 打印清晰的日志,方便调试

---

### 修复 2: 手动清理端口 (临时方案)

如果修复 1 仍然失败,可以手动清理端口:

```powershell
# 1. 查找占用端口的进程
netstat -ano | findstr :8010
netstat -ano | findstr :8011
netstat -ano | findstr :8012

# 2. 杀死进程 (替换 <PID> 为实际 PID)
taskkill /F /PID <PID>

# 3. 重启 CARLA 服务器
# 关闭 CARLA (Ctrl+C 或任务管理器)
# 重新启动:
cmake --build Build --target launch
```

---

### 修复 3: 重启 CARLA 服务器

**最简单但最有效的方法**:

```powershell
# 1. 关闭当前 CARLA 进程
# 按 Ctrl+C 或在任务管理器中结束 CarlaUE5.exe

# 2. 清理所有 CARLA 相关进程
taskkill /F /IM CarlaUE5.exe /T
taskkill /F /IM CarlaUE5-Win64-Shipping.exe /T

# 3. 等待 5 秒,确保端口释放
timeout /t 5

# 4. 重新启动 CARLA
cmake --build Build --target launch

# 5. 等待服务器完全启动 (看到 "Listening on port 2000")

# 6. 重新运行采集
python dense_occupancy_collection/main_data_collection.py --output dataset_10k --frames 10000
```

---

## 📋 验证修复

### 测试步骤

```powershell
# 1. 确保 CARLA 服务器正在运行
cmake --build Build --target launch

# 2. 运行数据采集
python dense_occupancy_collection/main_data_collection.py --output dataset_10k --frames 10

# 3. 查看日志
```

**预期输出**:
```
[Scenario] 正在生成 Hero 车辆 (vehicle.lincoln.mkz*)...
  ✓ Hero 生成成功: ID=49, Loc=(106.4, -12.7)
  ✓ 自动驾驶已启用 (TM Port: 8010)  # 或 8011, 8012, ...
```

**如果端口被占用**:
```
  ⚠ TM 端口 8010 被占用,尝试下一个...
  ✓ 自动驾驶已启用 (TM Port: 8011)  # 成功使用备用端口
```

**如果所有端口都被占用** (极端情况):
```
  ⚠ TM 端口 8010 被占用,尝试下一个...
  ⚠ TM 端口 8011 被占用,尝试下一个...
  ...
  ⚠ 警告: 无法启用自动驾驶 (所有端口都被占用)
  → 车辆将保持静止状态
```

---

## 🔍 故障排查

### 问题 1: 修复后仍然失败

**可能原因**: 端口范围不够

**解决**:
编辑 `scenario_manager.py`, 扩大端口范围:
```python
tm_ports_to_try = [self.tm_port, 8011, 8012, 8013, 8014, 8015, 8016, 8017, 8018, 8019]
```

---

### 问题 2: CARLA 服务器本身无法启动

**可能原因**: CARLA RPC 端口 (2000) 被占用

**解决**:
```powershell
# 查找占用 2000 端口的进程
netstat -ano | findstr :2000

# 杀死进程
taskkill /F /PID <PID>

# 重启 CARLA
cmake --build Build --target launch
```

---

### 问题 3: 车辆保持静止,无法移动

**原因**: 自动驾驶未启用 (所有端口被占用)

**解决方案 A - 手动控制** (临时):
修改 `spawn_hero()` 参数:
```python
hero = scenario.spawn_hero(enable_autopilot=False)
```

**解决方案 B - 清理端口** (推荐):
```powershell
# 清理所有 CARLA 相关进程
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *carla*"
taskkill /F /IM CarlaUE5.exe /T

# 重启 CARLA
cmake --build Build --target launch
```

---

## 📊 端口使用表

| 端口 | 用途 | 优先级 |
|------|------|--------|
| 2000 | CARLA RPC Server | 固定 |
| 2001 | CARLA Streaming Server | 固定 |
| 8000 | Traffic Manager (常见默认) | - |
| 8010 | Traffic Manager (我们的默认) | 1 |
| 8011 | Traffic Manager (备用 1) | 2 |
| 8012 | Traffic Manager (备用 2) | 3 |
| 8013 | Traffic Manager (备用 3) | 4 |
| 8014 | Traffic Manager (备用 4) | 5 |
| 8015 | Traffic Manager (备用 5) | 6 |

---

## ✅ 修复状态

- [x] 代码修改完成 (scenario_manager.py)
- [x] 端口重试机制实现
- [x] 错误处理添加
- [x] 文档编写完成
- [ ] 测试验证 (待用户执行)

**修复日期**: 2025-01-06
**问题严重性**: P1 (High)
**修复状态**: ✅ 代码修改完成,待测试验证

---

## 🎯 快速恢复命令

如果再次遇到此问题,执行以下命令快速恢复:

```powershell
# 一键清理脚本
taskkill /F /IM CarlaUE5.exe /T 2>nul
taskkill /F /IM CarlaUE5-Win64-Shipping.exe /T 2>nul
timeout /t 3
cmake --build Build --target launch
```

---

**最后更新**: 2025-01-06
**适用版本**: CARLA UE5 (ue5-dev branch)

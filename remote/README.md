# CARLA 远程服务端使用手册（192.168.100.81）

本机做开发 + 客户端，服务器（主机名 XTWY20240424，i9-13900K / 64GB / RTX 4090）跑 CARLA 服务端。所有远程操作通过 SSH 公钥免密登录，不存密码。

> **UE5.8（本目录 `D:\code\carla_ue58`，2026-10 起）与 5.5 的差异**
> - 打包：本目录的 `build_package_development.bat`（5.8 引擎 + MSVC 14.44）；本机 D 盘不够，`Build\Package`、
>   `Unreal\CarlaUnreal\Saved\Cooked` 用目录链接指到 `C:\carla_build58\`，`carla_remote.ps1` 的默认 zip 路径已指向这里。
> - 客户端 Python：`C:\Users\bills\.conda\envs\carla58\python.exe`（必须与服务端同一次编译），`-Action test` 已改用它。
> - 计划任务改为 `ONSTART`：服务器重启后服务端自动拉起（常驻）。
> - 新增 `-Action firewall`：放行入站 TCP 2000-2002、8000。
> - 新增 `reset_carla_world.py`：服务端常驻时，被中断的脚本会把同步模式、天气、车辆和传感器留在服务端；
>   连接异常或画面不对时先跑它复位。
> - 公钥失效时（服务器重装/换人）：先 `scp %USERPROFILE%\.ssh\id_ed25519.pub administrator@192.168.100.81:C:/ProgramData/ssh/new_key.pub`，
>   再 `ssh administrator@192.168.100.81 "type C:\ProgramData\ssh\new_key.pub >> C:\ProgramData\ssh\administrators_authorized_keys && icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant Administrators:F /grant SYSTEM:F && del C:\ProgramData\ssh\new_key.pub"`
>   （各输一次密码；不要写成一行嵌套引号的 PowerShell，会被本地和远端各解析一次而失败）。

## 一、架构总览

```
本机 (开发机)                                服务器 192.168.100.81 (XTWY20260917)
─────────────────────────                   ─────────────────────────────────────
d:\code\carla            C++/Python 源码     D:\carla-server\           当前运行包
  ├─ BUILD_FINAL.bat     C++ 全量编译        D:\carla-server.bak\       上一版本(回滚用)
  ├─ build_package_development.bat           D:\carla-update\           上传的 zip 暂存
  │     ↓ 打包 zip                            计划任务 CARLA-Server
  │     ↓ (scp 上传)                        (SYSTEM, Session 0) 启动
  └─ remote\carla_remote.ps1  ←── SSH ──→   CarlaUnreal.exe -RenderOffScreen
  ├─ remote_*.bat (5个快捷方式)                 端口: 2000 RPC / 2001 流 / 8000 TM
  └─ 远程根目录参数: D:\carla-server
```

关键点：

- 服务端跑的是 **Development 打包**（不是源码），源码改动必须先本地编译 → 打包 → 部署才能生效。
- 服务端启动参数：`-carla-server -quality-level=Epic -RenderOffScreen -nosound`。
- **客户端必须用本仓库自编译的 carla Python 模块**（含等距鱼眼 Bayer RAW 协议增强），标准 CARLA 发行版的 Python API 与自定义传感器格式不兼容。

## 二、文件清单

| 文件 | 作用 |
|---|---|
| `remote/carla_remote.ps1` | 核心脚本，7 个 action：`status / start / stop / restart / logs / deploy / test` |
| `remote_status_carla.bat` | 查看远端进程、端口 2000/2001、GPU 占用（双击即可） |
| `remote_start_carla.bat` | 注册并启动计划任务 CARLA-Server |
| `remote_stop_carla.bat` | 停止服务 |
| `remote_logs_carla.bat` | 看远端最新 CarlaUnreal.log 尾部 |
| `remote_deploy_carla.bat` | 上传本地 zip → 远端换版本 → 自动重启 |
| `../build_package_development.bat` | 本地打 Development 包（输出到 `Build\Package\`） |

自定义参数（如服务器 IP、启动参数、zip 路径）都在 `carla_remote.ps1` 顶部的 `param()` 里改。

## 三、首次部署（从零到服务端跑起来）

```bat
:: 1. 本地编译打包含最新鱼眼 RAW 的 Development 包（1-2 小时，产物约 15GB）
build_package_development.bat

:: 2. 上传并部署（scp 千兆约 20 分钟 + 远端解压 10-20 分钟），完成后自动启动
remote_deploy_carla.bat

:: 3. 验证
remote_status_carla.bat          :: nvidia-smi 里必须看到 CarlaUnreal.exe 占用显存
remote_logs_carla.bat            :: 看日志确认加载完成
remote\carla_remote.ps1 -Action test   :: 本机 RPC 连通性测试
```

验证通过的三个标准：

1. `remote_status_carla.bat` 显示 `CarlaUnreal RUNNING` + `TCP 2000 LISTENING`；
2. nvidia-smi 进程列表里**有 CarlaUnreal.exe 且占用显存**（证明 Session 0 下真 GPU 渲染，不是软件渲染）；
3. `test` action 打印出 server version。

## 四、日常开发流程

### 场景 A：只改 Python（e2e_occ、occnetv3_data_generator、dataset_viewer_v2）

不用动服务器，代码全在本机跑，直接 `--host 192.168.100.81` 连远程服务端即可（见第六节）。

### 场景 B：改了 C++ 引擎代码（LibCarla / Unreal 插件 / PythonAPI 绑定）

```bat
:: 1. 全量编译（很慢）
BUILD_FINAL.bat
:: 验证：对比 Build\...\UnrealEditor-Carla.dll 的 LastWriteTime 是否真的更新了

:: 2. 本地自测（可选但推荐）：编辑器里 Play 起来验证功能没退化
cmake --build Build --target launch-only

:: 3. 重新打 Development 包
build_package_development.bat

:: 4. 部署到 81 并自动重启服务端
remote_deploy_carla.bat

:: 5. 验证（同上三条标准）
remote_status_carla.bat
```

### 场景 C：只想改服务端启动参数（地图、quality-level 等）

改 `remote\carla_remote.ps1` 顶部的 `$CarlaArgs`，然后 `remote_stop_carla.bat` + `remote_start_carla.bat`，**不需要重新打包**。

## 五、远程服务端控制命令速查

双击对应 bat，或在 PowerShell 里直接调（可带参数）：

```powershell
# 指定其它服务器 / 启动参数
powershell -NoProfile -ExecutionPolicy Bypass -File remote\carla_remote.ps1 -Action start `
    -RemoteHost 192.168.100.81 `
    -CarlaArgs "-carla-server -quality-level=High -RenderOffScreen -nosound"

# 只启停不重新部署
powershell -NoProfile -File remote\carla_remote.ps1 -Action restart
```

| 命令 | 说明 |
|---|---|
| `status` | 进程 / 计划任务 / 端口 / GPU 一屏看清 |
| `start` | 注册 SYSTEM 计划任务并 Run（Session 0，断开 SSH 也不退出） |
| `stop` | 停任务 + 强杀进程 |
| `restart` | stop + start |
| `logs` | 远端最新 CarlaUnreal.log 尾部 50 行（`-Tail N` 可调） |
| `deploy` | 见第四节场景 B 第 4 步，自动 4 段流水线 |
| `test` | 本机用 carla Python 包连 2000 端口打印 server version |

注意：

- 启动后 15 秒内没看到 `TCP 2000 LISTENING` 不一定是失败——首次启动要加载着色器，可能要几分钟，隔一会再 `status`。
- `stop` 用的是计划任务 End + 强杀，Blender 之类其它 GPU 进程不受影响。

## 六、本地客户端连接

服务端起来后，本机任何脚本把连接地址从 `localhost` 换成 `192.168.100.81` 即可：

```bash
# 数据采集（脚本已支持 --host）
/c/ProgramData/anaconda3/envs/deepsys/python.exe occnetv3_data_generator/main_collection.py `
    --host 192.168.100.81 --port 2000 --frames 100 --output <输出目录> `
    --town Town10HD --num-vehicles 30 --num-walkers 10

# 训练不受影响（吃的是已落盘的数据集，不需要连服务器）

# 快速连通性自检
/c/ProgramData/anaconda3/envs/carla/python.exe -c "import carla; c=carla.Client('192.168.100.81',2000); c.set_timeout(30.0); print(c.get_server_version())"
```

要求：

- **必须用本机自编译安装的 carla 模块**（`conda activate carla` 环境里那份，含 RAW/bayer 协议），别用 pip 装的官方 carla。
- 连接超时常见原因：服务端还在首次加载（看 `remote_logs_carla.bat`）、Windows 防火墙拦了 2000/2001（首次启动若被拦，在服务器上放行或临时关防火墙）、Blender 等进程把 GPU 显存占满导致服务端启动失败。
- Traffic Manager 要显式指定：`trafficmanager = client.get_trafficmanager(8000)`，端口 8000 也要能在防火墙里通。

## 七、故障排查 / 回滚

| 现象 | 处理 |
|---|---|
| SSH 连不上 | 手动 `ssh Administrator@192.168.100.81` 看报错；公钥失效则重装 `~/.ssh/id_ed25519.pub` 到服务器 `C:\ProgramData\ssh\administrators_authorized_keys` |
| GPU 显示占用但无 CarlaUnreal | 服务器上可能有 Blender 等渲染任务占卡，`status` 的 nvidia-smi 里确认 |
| Session 0 下渲染异常/黑屏（软件渲染嫌疑） | 看日志里 GPU adapter 初始化行；若确认不行，切方案 B：自动登录 + Interactive 会话计划任务 |
| 新版本有问题要回滚 | `remote_stop_carla.bat`，然后在服务器上手动把 `D:\carla-server` 和 `D:\carla-server.bak` 互换目录名，再 `remote_start_carla.bat` |
| 部署中断电/中断 | zip 还在 `D:\carla-update\`，重跑 `remote_deploy_carla.bat` 即可（幂等，会重新解压覆盖） |

---

## 附：本套方案的关键设计决策

- **Session 0 启动（SYSTEM 计划任务）**：SSH 会话退出不影响服务端；代价是 Session 0 无真实桌面，GPU 渲染需要用 nvidia-smi 进程列表验证（待首次部署后确认）。
- **EncodedCommand 传脚本**：所有远程 PowerShell 用 Base64(UTF-16LE) 编码传输，彻底规避引号/换行/中文编码问题。
- **bak 目录回滚**：deploy 每次换版本前把旧版改名保留，一键目录互换即回滚。
- **zip 全量替换**：不玩增量 rsync——CARLA 包内 PAK/二进制互相依赖，全量替换最不容易出"半新半旧"玄学问题。

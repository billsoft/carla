# CARLA 分布式部署指南 - 服务端与客户端分离

> 将自编译的 CARLA UE5 服务端部署到云GPU/局域网服务器，客户端分布在多台机器上

---

## 目录

1. [部署架构说明](#部署架构)
2. [服务端部署完整流程](#服务端部署)
3. [客户端部署完整流程](#客户端部署)
4. [网络配置与防火墙](#网络配置)
5. [验证与测试](#验证测试)
6. [常见问题排查](#常见问题)
7. [性能优化建议](#性能优化)

---

## 1. 部署架构说明 {#部署架构}

### 1.1 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        云GPU服务器 / 局域网服务器                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  CARLA Server (Unreal Engine 5.5)                         │  │
│  │  ├─ CarlaUE5.exe (服务端可执行文件)                         │  │
│  │  ├─ 地图资源 (Town01-13, HD Maps)                          │  │
│  │  ├─ 车辆/行人模型                                           │  │
│  │  └─ 物理引擎 (Chaos)                                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  监听端口:                                                         │
│    - 2000: RPC 指令端口 (TCP)                                    │
│    - 2001: 传感器数据流端口 (TCP)                                │
│    - 2002: 多GPU路由端口 (可选)                                  │
│                                                                   │
│  GPU: NVIDIA RTX 3070+ (至少 8GB 显存)                           │
│  RAM: 32GB+                                                      │
│  操作系统: Windows Server 2022 / Ubuntu 22.04                    │
└─────────────────────────────────────────────────────────────────┘
                              ↕ 网络连接 (TCP/IP)
                              ↕ 公网IP / 内网IP
┌─────────────────────────────────────────────────────────────────┐
│                        客户端机器 (多台)                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Python 客户端                                             │  │
│  │  ├─ carla Python 包 (PythonAPI/carla/)                    │  │
│  │  ├─ 训练脚本 / 数据采集脚本                                 │  │
│  │  ├─ 模型推理代码                                            │  │
│  │  └─ 可视化工具 (可选)                                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  GPU: 不需要 (或仅用于模型训练/推理)                              │
│  RAM: 8GB+                                                       │
│  操作系统: Windows 10/11, Ubuntu 20.04+, macOS                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心概念

**服务端 (Server)**:
- 运行 Unreal Engine 5.5 + CARLA 插件
- 负责 3D 渲染、物理模拟、传感器数据生成
- 需要强大的 GPU
- 可以无头运行 (Headless Mode, 无显示器)

**客户端 (Client)**:
- 仅运行 Python 代码
- 通过网络连接到服务端
- 发送控制指令 (生成车辆、设置天气等)
- 接收传感器数据 (图像、点云等)
- 不需要 GPU (除非本地做深度学习推理)

---

## 2. 服务端部署完整流程 {#服务端部署}

### 2.1 服务端硬件与系统要求

#### 最低配置:
- **GPU**: NVIDIA RTX 3070 (8GB 显存)
- **CPU**: Intel i7-10700 / AMD Ryzen 7 3700X
- **RAM**: 32GB
- **存储**: 300GB SSD
- **操作系统**: Windows Server 2022 或 Ubuntu 22.04

#### 推荐配置 (云GPU):
- **GPU**: NVIDIA RTX 4090 / A100 (24GB+ 显存)
- **CPU**: 16 核心+
- **RAM**: 64GB+
- **存储**: 500GB NVMe SSD
- **网络**: 100Mbps+ 上传带宽 (传输传感器数据)

### 2.2 准备服务端文件

在你的**编译机器** (d:\code\carla) 上执行以下步骤:

#### 步骤 2.2.1: 确认编译完成

```bash
# 确保你已经完成完整编译
cd d:\code\carla
cmake --build Build --target package
```

**编译完成后的目录结构**:
```
d:\code\carla\
├── Build\
│   └── UE5\
│       └── WindowsNoEditor\        # ← 打包后的可执行文件
│           ├── CarlaUE5.exe        # ← 服务端主程序
│           ├── CarlaUE5\
│           │   ├── Binaries\
│           │   ├── Content\        # ← 地图、模型资源
│           │   └── Config\
│           └── Engine\
├── PythonAPI\
│   └── carla\                      # ← 客户端需要的 Python 包
└── Unreal\
```

#### 步骤 2.2.2: 打包服务端文件

**创建服务端部署包**:

```powershell
# 在 PowerShell 中执行

# 创建临时打包目录
New-Item -ItemType Directory -Path "D:\CARLA_Server_Package" -Force

# 1. 复制主程序和资源 (约 15-20 GB)
Write-Host "正在复制服务端文件 (需要 5-10 分钟)..." -ForegroundColor Yellow
Copy-Item -Path "d:\code\carla\Build\UE5\WindowsNoEditor\*" `
          -Destination "D:\CARLA_Server_Package\" `
          -Recurse -Force

# 2. 复制依赖的 DLL (如果有额外依赖)
# 注意: 大部分 DLL 已经在 WindowsNoEditor 中了

# 3. 创建启动脚本
@"
@echo off
REM CARLA UE5 服务端启动脚本

echo ========================================
echo CARLA UE5 服务端启动中...
echo ========================================

REM 设置渲染质量 (可选: Low, Medium, High, Epic)
set QUALITY=Low

REM 设置监听 IP (0.0.0.0 表示所有网卡)
set CARLA_SERVER_IP=0.0.0.0

REM 设置 RPC 端口
set CARLA_SERVER_PORT=2000

REM 启动服务端
CarlaUE5.exe ^
    -quality-level=%QUALITY% ^
    -RenderOffScreen ^
    -carla-rpc-port=%CARLA_SERVER_PORT% ^
    -carla-streaming-port=2001 ^
    -log

echo.
echo 服务端已关闭
pause
"@ | Out-File -FilePath "D:\CARLA_Server_Package\start_server.bat" -Encoding ASCII

# 4. 创建配置文件
@"
# CARLA 服务端配置文件

# 网络配置
RPC_PORT=2000
STREAMING_PORT=2001
TIMEOUT=10000

# 渲染配置
QUALITY_LEVEL=Low
RENDER_OFF_SCREEN=True

# 物理配置
FIXED_DELTA_SECONDS=0.05
SYNCHRONOUS_MODE=False

# 日志配置
LOG_LEVEL=INFO
"@ | Out-File -FilePath "D:\CARLA_Server_Package\server_config.ini" -Encoding UTF8

Write-Host "`n服务端打包完成!" -ForegroundColor Green
Write-Host "打包位置: D:\CARLA_Server_Package" -ForegroundColor Cyan
Write-Host "大小: " -NoNewline
$size = (Get-ChildItem -Path "D:\CARLA_Server_Package" -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host ("{0:N2} GB" -f $size) -ForegroundColor Cyan
```

**打包后的服务端目录结构**:
```
D:\CARLA_Server_Package\
├── CarlaUE5.exe                    # 主程序 (约 500 MB)
├── CarlaUE5\
│   ├── Binaries\                   # 二进制文件和插件
│   ├── Content\                    # 地图、模型资源 (约 10-15 GB)
│   │   ├── Carla\
│   │   │   ├── Maps\               # 地图文件
│   │   │   │   ├── Town01.umap
│   │   │   │   ├── Town02.umap
│   │   │   │   └── ...
│   │   │   ├── Static\             # 静态模型
│   │   │   └── Blueprints\         # 蓝图
│   │   └── ...
│   ├── Config\                     # 配置文件
│   └── Plugins\
│       └── Carla\                  # CARLA 插件
├── Engine\                         # UE5 引擎文件
├── start_server.bat                # 启动脚本
└── server_config.ini               # 配置文件
```

#### 步骤 2.2.3: 压缩并传输到服务器

**方案1: 使用 7-Zip 压缩** (推荐)

```powershell
# 安装 7-Zip (如果没有)
# 下载地址: https://www.7-zip.org/download.html

# 压缩 (需要 10-20 分钟)
& "C:\Program Files\7-Zip\7z.exe" a -t7z `
    "D:\CARLA_Server.7z" `
    "D:\CARLA_Server_Package\*" `
    -mx=5  # 压缩级别 (5=正常, 9=最大)

Write-Host "压缩完成!"
Write-Host "压缩包: D:\CARLA_Server.7z"
$compressedSize = (Get-Item "D:\CARLA_Server.7z").Length / 1GB
Write-Host ("压缩后大小: {0:N2} GB" -f $compressedSize) -ForegroundColor Cyan
```

**预期压缩大小**: 约 5-8 GB (取决于资源数量)

**方案2: 直接传输** (局域网)

```bash
# 如果是局域网服务器,可以直接复制
# 使用 Windows 文件共享或 SCP

# 示例: 使用 SCP (需要安装 OpenSSH)
scp -r "D:\CARLA_Server_Package" user@192.168.1.100:/opt/carla/
```

**方案3: 云服务器上传**

```bash
# 使用云服务商提供的工具
# 例如: 阿里云 OSS, 腾讯云 COS, AWS S3

# 或使用 FTP/SFTP 客户端 (FileZilla, WinSCP)
```

### 2.3 服务器端安装与配置

#### 步骤 2.3.1: 服务器前置条件 (Windows Server)

**必须安装的软件**:

1. **NVIDIA GPU 驱动**
```powershell
# 下载最新的 NVIDIA 驱动
# 地址: https://www.nvidia.com/Download/index.aspx

# 验证安装
nvidia-smi
# 应该能看到 GPU 信息
```

2. **Visual C++ 运行库** (UE5 依赖)
```powershell
# 下载并安装 Visual C++ Redistributable 2015-2022
# 地址: https://aka.ms/vs/17/release/vc_redist.x64.exe

# 下载后直接安装
Start-Process -FilePath "vc_redist.x64.exe" -ArgumentList "/install /quiet /norestart" -Wait
```

3. **DirectX 运行库**
```powershell
# 下载 DirectX End-User Runtime
# 地址: https://www.microsoft.com/en-us/download/details.aspx?id=35

# 解压并安装
Start-Process -FilePath "DXSETUP.exe" -ArgumentList "/silent" -Wait
```

#### 步骤 2.3.2: 解压服务端文件

```powershell
# 创建安装目录
New-Item -ItemType Directory -Path "C:\CARLA_Server" -Force

# 解压 (使用 7-Zip)
& "C:\Program Files\7-Zip\7z.exe" x `
    "D:\CARLA_Server.7z" `
    -o"C:\CARLA_Server\" `
    -y

Write-Host "解压完成: C:\CARLA_Server"
```

**最终服务器目录结构**:
```
C:\CARLA_Server\
├── CarlaUE5.exe
├── CarlaUE5\
├── Engine\
├── start_server.bat
└── server_config.ini
```

#### 步骤 2.3.3: 配置防火墙规则

**Windows 防火墙** (在服务器上执行):

```powershell
# 添加入站规则 - 允许 TCP 2000 (RPC)
New-NetFirewallRule -DisplayName "CARLA RPC Port" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 2000 `
    -Action Allow

# 添加入站规则 - 允许 TCP 2001 (Streaming)
New-NetFirewallRule -DisplayName "CARLA Streaming Port" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 2001 `
    -Action Allow

# 添加程序规则 - 允许 CarlaUE5.exe
New-NetFirewallRule -DisplayName "CARLA UE5 Server" `
    -Direction Inbound `
    -Program "C:\CARLA_Server\CarlaUE5.exe" `
    -Action Allow

Write-Host "防火墙规则已添加" -ForegroundColor Green
```

**云服务器安全组** (在云控制台配置):

| 规则类型 | 协议 | 端口 | 源地址 | 说明 |
|---------|------|------|--------|------|
| 入站 | TCP | 2000 | 0.0.0.0/0 或客户端IP | CARLA RPC |
| 入站 | TCP | 2001 | 0.0.0.0/0 或客户端IP | CARLA Streaming |
| 出站 | ALL | ALL | 0.0.0.0/0 | 允许所有出站 |

**重要**: 如果只允许特定客户端连接,将 `0.0.0.0/0` 改为客户端的公网 IP

#### 步骤 2.3.4: 启动服务端

**首次启动** (测试模式):

```powershell
cd C:\CARLA_Server

# 手动启动 (查看日志)
.\CarlaUE5.exe -quality-level=Low -RenderOffScreen -carla-rpc-port=2000

# 如果看到以下输出,说明启动成功:
# [CARLA] Server listening on 0.0.0.0:2000
# [CARLA] Streaming port: 2001
# [CARLA] World: /Game/Carla/Maps/Town01
```

**后台运行** (生产模式):

```powershell
# 方法1: 使用启动脚本
.\start_server.bat

# 方法2: 使用 Windows 服务 (推荐)
# 下载 NSSM (Non-Sucking Service Manager)
# 地址: https://nssm.cc/download

# 安装为服务
nssm install CARLA_Server "C:\CARLA_Server\CarlaUE5.exe"
nssm set CARLA_Server AppParameters "-quality-level=Low -RenderOffScreen -carla-rpc-port=2000"
nssm set CARLA_Server AppDirectory "C:\CARLA_Server"
nssm set CARLA_Server DisplayName "CARLA UE5 Server"
nssm set CARLA_Server Start SERVICE_AUTO_START

# 启动服务
nssm start CARLA_Server

# 查看状态
nssm status CARLA_Server
```

**验证服务端运行**:

```powershell
# 检查端口是否监听
netstat -ano | findstr :2000
# 应该看到类似:
# TCP    0.0.0.0:2000           0.0.0.0:0              LISTENING       1234

# 检查进程
Get-Process | Where-Object {$_.ProcessName -like "*Carla*"}
```

#### 步骤 2.3.5: 获取服务器 IP 地址

**内网 IP** (局域网部署):
```powershell
# 查看本机 IP
ipconfig

# 记录 IPv4 地址,例如: 192.168.1.100
```

**公网 IP** (云服务器):
```powershell
# 方法1: 查询公网 IP
(Invoke-WebRequest -Uri "http://ifconfig.me/ip").Content.Trim()

# 方法2: 在云控制台查看
# 阿里云: ECS 实例详情 → 公网 IP
# 腾讯云: CVM 实例 → 公网 IP
# AWS: EC2 实例 → 公有 IPv4 地址
```

**记录这个 IP 地址,客户端需要用到!**

---

## 3. 客户端部署完整流程 {#客户端部署}

### 3.1 客户端硬件与系统要求

#### 最低配置:
- **CPU**: Intel i5 / AMD Ryzen 5
- **RAM**: 8GB
- **存储**: 2GB (仅 Python 包)
- **网络**: 10Mbps+ 下载带宽
- **操作系统**: Windows 10/11, Ubuntu 20.04+, macOS 11+

#### 推荐配置 (如需本地训练):
- **GPU**: NVIDIA RTX 3060+ (训练神经网络)
- **RAM**: 16GB+
- **存储**: 100GB+ (存储训练数据)

### 3.2 准备客户端文件

在你的**编译机器** (d:\code\carla) 上执行:

#### 步骤 3.2.1: 构建 Python API

```bash
cd d:\code\carla

# 编译并安装 Python API
cmake --build Build --target carla-python-api-install
```

**生成的文件位置**:
```
d:\code\carla\PythonAPI\carla\
├── dist\
│   └── carla-0.9.15-cp310-cp310-win_amd64.whl  # ← Python 轮子包
├── agents\                                      # ← 高级 Agent 代码
└── setup.py
```

#### 步骤 3.2.2: 打包客户端文件

```powershell
# 创建客户端打包目录
New-Item -ItemType Directory -Path "D:\CARLA_Client_Package" -Force

# 1. 复制 Python 包
Copy-Item -Path "d:\code\carla\PythonAPI\carla\dist\*.whl" `
          -Destination "D:\CARLA_Client_Package\" -Force

# 2. 复制示例脚本 (可选)
Copy-Item -Path "d:\code\carla\PythonAPI\examples\*" `
          -Destination "D:\CARLA_Client_Package\examples\" -Recurse -Force

# 3. 复制 agents 模块
Copy-Item -Path "d:\code\carla\PythonAPI\carla\agents" `
          -Destination "D:\CARLA_Client_Package\carla\" -Recurse -Force

# 4. 创建快速开始脚本
@"
@echo off
REM CARLA 客户端环境设置

echo ========================================
echo CARLA 客户端环境配置
echo ========================================

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python!
    echo 请先安装 Python 3.7+
    pause
    exit /b 1
)

REM 安装 CARLA Python 包
echo.
echo 正在安装 CARLA Python 包...
pip install carla-0.9.15-cp310-cp310-win_amd64.whl --force-reinstall

REM 安装其他依赖
echo.
echo 正在安装依赖包...
pip install numpy pygame

echo.
echo ========================================
echo 安装完成!
echo ========================================
echo.
echo 下一步: 编辑 config.py 设置服务器 IP
pause
"@ | Out-File -FilePath "D:\CARLA_Client_Package\install.bat" -Encoding ASCII

# 5. 创建配置文件模板
@"
# CARLA 客户端配置文件

# 服务器配置
SERVER_HOST = "YOUR_SERVER_IP"  # 替换为服务器 IP
SERVER_PORT = 2000
TIMEOUT = 10.0

# 示例:
# 局域网: SERVER_HOST = "192.168.1.100"
# 云服务器: SERVER_HOST = "123.45.67.89"
"@ | Out-File -FilePath "D:\CARLA_Client_Package\config.py" -Encoding UTF8

# 6. 创建测试连接脚本
@"
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import carla
import sys

# 从 config.py 读取配置
try:
    from config import SERVER_HOST, SERVER_PORT, TIMEOUT
except ImportError:
    print("错误: 未找到 config.py")
    print("请先编辑 config.py 设置服务器 IP")
    sys.exit(1)

print("=" * 80)
print("CARLA 客户端连接测试")
print("=" * 80)
print(f"服务器: {SERVER_HOST}:{SERVER_PORT}")
print(f"超时: {TIMEOUT}s")
print("-" * 80)

try:
    # 连接到服务器
    print("\n正在连接到服务器...")
    client = carla.Client(SERVER_HOST, SERVER_PORT)
    client.set_timeout(TIMEOUT)

    # 获取服务器版本
    version = client.get_server_version()
    print(f"✓ 连接成功!")
    print(f"✓ 服务器版本: {version}")

    # 获取世界信息
    world = client.get_world()
    map_name = world.get_map().name
    print(f"✓ 当前地图: {map_name}")

    # 获取可用地图列表
    available_maps = client.get_available_maps()
    print(f"✓ 可用地图: {len(available_maps)} 个")
    for map_path in available_maps[:5]:  # 只显示前5个
        print(f"  - {map_path}")

    print("\n" + "=" * 80)
    print("测试成功! 客户端可以正常连接到服务器")
    print("=" * 80)

except Exception as e:
    print(f"\n✗ 连接失败: {e}")
    print("\n请检查:")
    print("  1. 服务器是否已启动")
    print("  2. config.py 中的 IP 地址是否正确")
    print("  3. 防火墙是否开放 2000 端口")
    print("  4. 网络连接是否正常")
    sys.exit(1)
"@ | Out-File -FilePath "D:\CARLA_Client_Package\test_connection.py" -Encoding UTF8

Write-Host "`n客户端打包完成!" -ForegroundColor Green
Write-Host "打包位置: D:\CARLA_Client_Package" -ForegroundColor Cyan
```

**打包后的客户端目录**:
```
D:\CARLA_Client_Package\
├── carla-0.9.15-cp310-cp310-win_amd64.whl  # Python 包 (约 50 MB)
├── carla\
│   └── agents\                              # 高级 Agent 模块
├── examples\                                # 示例脚本
│   ├── manual_control.py
│   ├── spawn_npc.py
│   └── ...
├── install.bat                              # 安装脚本
├── config.py                                # 配置文件模板
└── test_connection.py                       # 连接测试脚本
```

#### 步骤 3.2.3: 打包并分发给客户端

```powershell
# 压缩客户端包 (约 100 MB)
& "C:\Program Files\7-Zip\7z.exe" a -t7z `
    "D:\CARLA_Client.7z" `
    "D:\CARLA_Client_Package\*" `
    -mx=9

Write-Host "客户端包已压缩: D:\CARLA_Client.7z"
$size = (Get-Item "D:\CARLA_Client.7z").Length / 1MB
Write-Host ("大小: {0:N2} MB" -f $size) -ForegroundColor Cyan

# 现在可以通过邮件、网盘等方式分发给客户端用户
```

### 3.3 客户端安装与配置

**在每台客户端机器上执行以下步骤**:

#### 步骤 3.3.1: 安装 Python 环境

**Windows**:

```powershell
# 1. 下载 Python 3.10
# 地址: https://www.python.org/downloads/

# 2. 安装时勾选 "Add Python to PATH"

# 3. 验证安装
python --version
# 应该显示: Python 3.10.x

pip --version
# 应该显示: pip 23.x
```

**Ubuntu/Linux**:

```bash
# 安装 Python 3.10
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip -y

# 验证
python3.10 --version
pip3 --version
```

**macOS**:

```bash
# 使用 Homebrew 安装
brew install python@3.10

# 验证
python3.10 --version
pip3 --version
```

#### 步骤 3.3.2: 解压并安装客户端包

**Windows**:

```powershell
# 解压到用户目录
New-Item -ItemType Directory -Path "$env:USERPROFILE\CARLA_Client" -Force
& "C:\Program Files\7-Zip\7z.exe" x `
    "D:\CARLA_Client.7z" `
    -o"$env:USERPROFILE\CARLA_Client\" `
    -y

# 进入目录
cd "$env:USERPROFILE\CARLA_Client"

# 运行安装脚本
.\install.bat
```

**Linux/macOS**:

```bash
# 解压
mkdir -p ~/CARLA_Client
cd ~/CARLA_Client
7z x CARLA_Client.7z

# 创建虚拟环境 (推荐)
python3.10 -m venv venv
source venv/bin/activate

# 安装 CARLA 包
pip install carla-0.9.15-cp310-cp310-linux_x86_64.whl  # Linux
# 或
pip install carla-0.9.15-cp310-cp310-macosx_11_0_arm64.whl  # macOS

# 安装依赖
pip install numpy pygame
```

#### 步骤 3.3.3: 配置服务器连接

**编辑 `config.py`**:

```python
# CARLA 客户端配置文件

# 服务器配置
SERVER_HOST = "123.45.67.89"  # ← 替换为你的服务器 IP
SERVER_PORT = 2000
TIMEOUT = 10.0

# 示例:
# 局域网服务器: SERVER_HOST = "192.168.1.100"
# 云服务器: SERVER_HOST = "123.45.67.89"
# 本地测试: SERVER_HOST = "localhost"
```

**重要**:
- 如果是**云服务器**,填写**公网 IP**
- 如果是**局域网**,填写**内网 IP** (例如 192.168.x.x)
- 如果是**本机测试**,填写 `localhost` 或 `127.0.0.1`

#### 步骤 3.3.4: 测试连接

```bash
# 运行连接测试脚本
python test_connection.py
```

**成功的输出示例**:
```
================================================================================
CARLA 客户端连接测试
================================================================================
服务器: 123.45.67.89:2000
超时: 10.0s
--------------------------------------------------------------------------------

正在连接到服务器...
✓ 连接成功!
✓ 服务器版本: 0.9.15
✓ 当前地图: /Game/Carla/Maps/Town01
✓ 可用地图: 13 个
  - /Game/Carla/Maps/Town01
  - /Game/Carla/Maps/Town02
  - /Game/Carla/Maps/Town03
  - /Game/Carla/Maps/Town04
  - /Game/Carla/Maps/Town05

================================================================================
测试成功! 客户端可以正常连接到服务器
================================================================================
```

**失败排查**: 如果连接失败,参考 [第6章 常见问题排查](#常见问题)

---

## 4. 网络配置与防火墙 {#网络配置}

### 4.1 端口说明

| 端口 | 协议 | 用途 | 数据流向 | 带宽需求 |
|-----|------|------|---------|---------|
| **2000** | TCP | RPC 指令通道 | 双向 | 低 (<1 Mbps) |
| **2001** | TCP | 传感器数据流 | 服务器→客户端 | 高 (10-100 Mbps) |
| 2002 | TCP | 多GPU路由 (可选) | 服务器内部 | 低 |

### 4.2 防火墙配置详解

#### 服务器防火墙 (入站规则)

**Windows Server**:

```powershell
# 查看现有规则
Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*CARLA*"}

# 删除旧规则 (如果需要)
Remove-NetFirewallRule -DisplayName "CARLA RPC Port"
Remove-NetFirewallRule -DisplayName "CARLA Streaming Port"

# 添加新规则
New-NetFirewallRule -DisplayName "CARLA RPC Port" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 2000 `
    -Action Allow `
    -Profile Domain,Private,Public

New-NetFirewallRule -DisplayName "CARLA Streaming Port" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 2001 `
    -Action Allow `
    -Profile Domain,Private,Public
```

**Ubuntu Server**:

```bash
# 使用 UFW
sudo ufw allow 2000/tcp comment "CARLA RPC"
sudo ufw allow 2001/tcp comment "CARLA Streaming"
sudo ufw reload

# 查看规则
sudo ufw status numbered

# 使用 iptables (如果不用 UFW)
sudo iptables -A INPUT -p tcp --dport 2000 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 2001 -j ACCEPT
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

#### 云服务器安全组

**阿里云 ECS**:

1. 登录阿里云控制台
2. 进入 ECS → 实例 → 安全组
3. 添加入方向规则:
   - 协议: TCP
   - 端口范围: 2000/2000
   - 授权对象: 0.0.0.0/0 (或指定客户端IP)
   - 描述: CARLA RPC
4. 重复添加 2001 端口

**腾讯云 CVM**:

1. 登录腾讯云控制台
2. 进入 CVM → 安全组
3. 入站规则 → 添加规则
4. 填写类似配置

**AWS EC2**:

1. 登录 AWS Console
2. EC2 → Security Groups
3. Edit inbound rules
4. Add rule:
   - Type: Custom TCP
   - Port range: 2000, 2001
   - Source: 0.0.0.0/0 或 My IP

### 4.3 网络连通性测试

#### 服务器端测试

```powershell
# 1. 检查端口监听
netstat -ano | findstr :2000
netstat -ano | findstr :2001

# 2. 测试本地回环
Test-NetConnection -ComputerName localhost -Port 2000
Test-NetConnection -ComputerName localhost -Port 2001

# 3. 获取公网 IP
(Invoke-WebRequest -Uri "http://ifconfig.me/ip").Content
```

#### 客户端测试

```powershell
# Windows
Test-NetConnection -ComputerName SERVER_IP -Port 2000
Test-NetConnection -ComputerName SERVER_IP -Port 2001

# 如果失败,使用 telnet 测试
telnet SERVER_IP 2000
# 按 Ctrl+] 然后输入 quit 退出
```

```bash
# Linux/macOS
nc -zv SERVER_IP 2000
nc -zv SERVER_IP 2001

# 或使用 telnet
telnet SERVER_IP 2000
```

**成功示例**:
```
测试端口 2000 ...
TcpTestSucceeded : True
```

**失败示例**:
```
警告: TCP connect to (SERVER_IP : 2000) failed
```

---

## 5. 验证与测试 {#验证测试}

### 5.1 服务端验证

#### 检查清单:

```powershell
# 1. 进程是否运行
Get-Process | Where-Object {$_.ProcessName -like "*Carla*"}

# 2. 端口是否监听
netstat -ano | findstr :2000

# 3. GPU 是否被占用
nvidia-smi

# 4. 查看 CARLA 日志
Get-Content "C:\CARLA_Server\CarlaUE5\Saved\Logs\CarlaUE5.log" -Tail 50
```

**正常日志示例**:
```
[2025.01.08-10.30.15:123] LogCarlaServer: Server listening on 0.0.0.0:2000
[2025.01.08-10.30.15:234] LogCarlaServer: Streaming server listening on 2001
[2025.01.08-10.30.15:345] LogCarla: World loaded: /Game/Carla/Maps/Town01
[2025.01.08-10.30.15:456] LogCarla: Server ready to accept connections
```

### 5.2 客户端验证

#### 基础连接测试

```python
# test_basic_connection.py
import carla

SERVER_HOST = "YOUR_SERVER_IP"
SERVER_PORT = 2000

client = carla.Client(SERVER_HOST, SERVER_PORT)
client.set_timeout(10.0)

print(f"Server version: {client.get_server_version()}")
print(f"Available maps: {len(client.get_available_maps())}")
```

#### 完整功能测试

```python
# test_full_features.py
import carla
import time

SERVER_HOST = "YOUR_SERVER_IP"
SERVER_PORT = 2000

print("=" * 80)
print("CARLA 完整功能测试")
print("=" * 80)

# 1. 连接
print("\n1. 连接到服务器...")
client = carla.Client(SERVER_HOST, SERVER_PORT)
client.set_timeout(10.0)
print(f"   ✓ 连接成功 (版本: {client.get_server_version()})")

# 2. 获取世界
print("\n2. 获取世界...")
world = client.get_world()
print(f"   ✓ 当前地图: {world.get_map().name}")

# 3. 生成车辆
print("\n3. 生成车辆...")
bp_library = world.get_blueprint_library()
vehicle_bp = bp_library.find('vehicle.tesla.model3')
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
print(f"   ✓ 车辆已生成 (ID: {vehicle.id})")

# 4. 生成相机
print("\n4. 生成相机...")
camera_bp = bp_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
print(f"   ✓ 相机已生成 (ID: {camera.id})")

# 5. 测试相机数据流
print("\n5. 测试相机数据流...")
image_received = False

def on_image(image):
    global image_received
    image_received = True
    print(f"   ✓ 收到图像: {image.width}×{image.height}, 帧号: {image.frame}")

camera.listen(on_image)
time.sleep(2)  # 等待图像

if image_received:
    print("   ✓ 数据流测试成功!")
else:
    print("   ✗ 未收到图像数据")

# 6. 控制车辆
print("\n6. 控制车辆...")
vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
time.sleep(1)
velocity = vehicle.get_velocity()
speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5 * 3.6
print(f"   ✓ 车辆移动中 (速度: {speed:.1f} km/h)")

# 7. 清理
print("\n7. 清理资源...")
camera.stop()
camera.destroy()
vehicle.destroy()
print("   ✓ 资源已清理")

print("\n" + "=" * 80)
print("所有测试通过! 客户端功能正常")
print("=" * 80)
```

运行测试:
```bash
python test_full_features.py
```

### 5.3 性能测试

#### 延迟测试

```python
# test_latency.py
import carla
import time

SERVER_HOST = "YOUR_SERVER_IP"
client = carla.Client(SERVER_HOST, 2000)
client.set_timeout(10.0)

# 测试 RPC 延迟
latencies = []
for i in range(100):
    start = time.time()
    world = client.get_world()
    latency = (time.time() - start) * 1000  # ms
    latencies.append(latency)
    print(f"\r延迟测试: {i+1}/100 - {latency:.1f}ms", end='')

print(f"\n\n平均延迟: {sum(latencies)/len(latencies):.1f}ms")
print(f"最小延迟: {min(latencies):.1f}ms")
print(f"最大延迟: {max(latencies):.1f}ms")
```

**预期结果**:
- **局域网**: 平均 < 10ms
- **同城云服务器**: 平均 < 50ms
- **跨地域云服务器**: 平均 < 200ms

#### 带宽测试

```python
# test_bandwidth.py
import carla
import time

SERVER_HOST = "YOUR_SERVER_IP"
client = carla.Client(SERVER_HOST, 2000)
world = client.get_world()

# 生成车辆和相机
bp_lib = world.get_blueprint_library()
vehicle = world.spawn_actor(
    bp_lib.find('vehicle.tesla.model3'),
    world.get_map().get_spawn_points()[0]
)

camera = world.spawn_actor(
    bp_lib.find('sensor.camera.rgb'),
    carla.Transform(carla.Location(x=1.5, z=2.4)),
    attach_to=vehicle
)

# 统计接收数据
total_bytes = 0
frame_count = 0
start_time = time.time()

def on_image(image):
    global total_bytes, frame_count
    total_bytes += len(image.raw_data)
    frame_count += 1

camera.listen(on_image)

# 测试 30 秒
time.sleep(30)

camera.stop()
elapsed = time.time() - start_time
bandwidth = total_bytes / elapsed / 1024 / 1024  # MB/s

print(f"接收帧数: {frame_count}")
print(f"总数据量: {total_bytes / 1024 / 1024:.2f} MB")
print(f"平均带宽: {bandwidth:.2f} MB/s")
print(f"平均帧率: {frame_count / elapsed:.1f} FPS")

camera.destroy()
vehicle.destroy()
```

---

## 6. 常见问题排查 {#常见问题}

### 问题1: 客户端无法连接到服务器

**错误信息**:
```
RuntimeError: time-out of 10000ms while waiting for the simulator, make sure the simulator is ready and connected to localhost:2000
```

**排查步骤**:

1. **检查服务器是否运行**:
```powershell
# 服务器端
Get-Process | Where-Object {$_.ProcessName -like "*Carla*"}
```

2. **检查端口监听**:
```powershell
# 服务器端
netstat -ano | findstr :2000
```

3. **测试网络连通性**:
```powershell
# 客户端
ping SERVER_IP
Test-NetConnection -ComputerName SERVER_IP -Port 2000
```

4. **检查防火墙**:
```powershell
# 服务器端
Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*CARLA*"}
```

5. **检查服务器 IP 配置**:
```python
# 客户端 config.py
# 确认 SERVER_HOST 是正确的公网 IP 或内网 IP
```

**解决方案**:
- 确保服务器已启动: `.\start_server.bat`
- 检查防火墙规则是否正确添加
- 云服务器检查安全组配置
- 确认客户端 `config.py` 中的 IP 地址正确

---

### 问题2: 服务器启动失败

**错误信息**:
```
Fatal error: [File:Unknown] [Line: 198]
D3D12 not available
```

**原因**: 服务器没有 GPU 或 GPU 驱动未安装

**解决方案**:
```powershell
# 1. 安装 NVIDIA 驱动
# 下载: https://www.nvidia.com/Download/index.aspx

# 2. 验证
nvidia-smi

# 3. 如果是虚拟机,确保启用了 GPU 直通
```

---

### 问题3: 传感器数据未接收

**症状**: 相机 `listen()` 回调函数不执行

**排查**:

```python
# 检查回调是否注册
camera.is_listening  # 应该返回 True

# 检查世界是否在运行
world.wait_for_tick()  # 应该能正常返回
```

**解决方案**:

```python
# 确保设置了足够的超时
client.set_timeout(30.0)  # 增加到 30 秒

# 使用同步模式
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)
```

---

### 问题4: 带宽不足导致卡顿

**症状**: 图像延迟严重,帧率低

**检查带宽**:

```python
# test_bandwidth.py (见 5.3 章节)
```

**优化方案**:

1. **降低图像分辨率**:
```python
camera_bp.set_attribute('image_size_x', '640')  # 从 1920 降到 640
camera_bp.set_attribute('image_size_y', '480')  # 从 1080 降到 480
```

2. **降低帧率**:
```python
camera_bp.set_attribute('sensor_tick', '0.1')  # 10 FPS
```

3. **使用 JPEG 压缩** (虽然会损失质量):
```python
# 在服务器端设置
settings = world.get_settings()
settings.quality_level = 'Low'  # Low / Medium / High / Epic
world.apply_settings(settings)
```

4. **使用更快的网络**:
- 局域网: 使用千兆网卡
- 云服务器: 升级带宽套餐

---

### 问题5: 多客户端冲突

**症状**: 多个客户端连接后,车辆控制混乱

**原因**: 默认情况下,所有客户端共享同一个世界

**解决方案**:

```python
# 方案1: 每个客户端使用不同的 Traffic Manager 端口
tm = client.get_trafficmanager(8100)  # 客户端1 用 8100
tm = client.get_trafficmanager(8200)  # 客户端2 用 8200

# 方案2: 每个客户端加载不同的地图
client.load_world('Town01')  # 客户端1
client.load_world('Town02')  # 客户端2 (会替换 Town01)

# 方案3: 启动多个 CARLA 服务器实例 (推荐)
# 服务器1: .\CarlaUE5.exe -carla-rpc-port=2000
# 服务器2: .\CarlaUE5.exe -carla-rpc-port=2010
# 客户端连接到不同端口
```

---

## 7. 性能优化建议 {#性能优化}

### 7.1 服务器端优化

#### 图形质量设置

```bash
# 启动时设置 (推荐 Low 或 Medium)
.\CarlaUE5.exe -quality-level=Low

# 运行时设置
```

```python
# 客户端代码
settings = world.get_settings()
settings.quality_level = 'Low'  # Low / Medium / High / Epic
world.apply_settings(settings)
```

**质量级别对比**:

| 级别 | GPU占用 | 帧率 | 视觉效果 | 适用场景 |
|-----|---------|------|---------|---------|
| Low | 低 (2GB) | 60+ FPS | 基础 | 数据采集,多客户端 |
| Medium | 中 (4GB) | 30-60 FPS | 良好 | 训练场景 |
| High | 高 (6GB) | 20-30 FPS | 优秀 | 演示 |
| Epic | 很高 (8GB+) | <20 FPS | 极致 | 截图/录屏 |

#### 无头模式 (Headless)

```bash
# 不渲染窗口,节省 GPU 资源
.\CarlaUE5.exe -RenderOffScreen
```

#### 固定时间步长

```python
# 提高物理模拟稳定性
settings = world.get_settings()
settings.fixed_delta_seconds = 0.05  # 20 FPS 物理更新
settings.synchronous_mode = True     # 同步模式
world.apply_settings(settings)
```

### 7.2 客户端优化

#### 异步接收数据

```python
# 使用队列异步处理
import queue

image_queue = queue.Queue()

def on_image(image):
    if image_queue.qsize() < 10:  # 限制队列大小
        image_queue.put(image)
    else:
        print("警告: 队列已满,丢弃帧")

camera.listen(on_image)

# 在另一个线程处理
import threading

def process_images():
    while True:
        image = image_queue.get()
        # 处理图像...

thread = threading.Thread(target=process_images, daemon=True)
thread.start()
```

#### 批量操作

```python
# 一次生成多个 Actor (比循环快)
batch = [
    carla.command.SpawnActor(vehicle_bp, spawn_point)
    for spawn_point in spawn_points[:100]
]
responses = client.apply_batch_sync(batch)
```

### 7.3 网络优化

#### 局域网优化

```python
# 增加缓冲区大小
import socket
client._client.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)  # 1MB
```

#### 压缩传输 (实验性)

```python
# 在服务器配置中启用
# 编辑: C:\CARLA_Server\CarlaUE5\Config\DefaultEngine.ini
# 添加:
# [/Script/Carla.CarlaSettings]
# bUseCompression=True
```

---

## 附录A: 启动脚本完整版

### 服务器启动脚本 (start_server.bat)

```batch
@echo off
setlocal enabledelayedexpansion

REM ========================================
REM CARLA UE5 服务端启动脚本
REM ========================================

echo.
echo ========================================
echo CARLA UE5 服务端
echo ========================================
echo.

REM 检查 GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo 警告: 未检测到 NVIDIA GPU 或驱动未安装
    pause
    exit /b 1
)

REM 配置参数
set QUALITY_LEVEL=Low
set RPC_PORT=2000
set STREAMING_PORT=2001
set RENDER_OFF_SCREEN=true
set LOG_LEVEL=Info

REM 显示配置
echo 配置:
echo   质量级别: %QUALITY_LEVEL%
echo   RPC 端口: %RPC_PORT%
echo   流端口: %STREAMING_PORT%
echo   无头模式: %RENDER_OFF_SCREEN%
echo.

REM 构建启动参数
set ARGS=-quality-level=%QUALITY_LEVEL%
set ARGS=%ARGS% -carla-rpc-port=%RPC_PORT%
set ARGS=%ARGS% -carla-streaming-port=%STREAMING_PORT%

if "%RENDER_OFF_SCREEN%"=="true" (
    set ARGS=%ARGS% -RenderOffScreen
)

set ARGS=%ARGS% -log

REM 启动服务器
echo 启动服务器...
echo 命令: CarlaUE5.exe %ARGS%
echo.
echo ----------------------------------------
echo 服务器日志:
echo ----------------------------------------

CarlaUE5.exe %ARGS%

echo.
echo ========================================
echo 服务器已关闭
echo ========================================
pause
```

### 客户端连接模板 (client_template.py)

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CARLA 客户端连接模板

使用方法:
1. 编辑 config.py 设置服务器 IP
2. 运行: python client_template.py
"""

import carla
import random
import time
import numpy as np

# 从配置文件读取
try:
    from config import SERVER_HOST, SERVER_PORT, TIMEOUT
except ImportError:
    # 默认配置
    SERVER_HOST = "localhost"
    SERVER_PORT = 2000
    TIMEOUT = 10.0


def main():
    """主函数"""
    # 连接到服务器
    print(f"连接到 {SERVER_HOST}:{SERVER_PORT}...")
    client = carla.Client(SERVER_HOST, SERVER_PORT)
    client.set_timeout(TIMEOUT)

    try:
        # 获取世界
        world = client.get_world()
        print(f"✓ 连接成功! 当前地图: {world.get_map().name}")

        # 获取蓝图库
        bp_lib = world.get_blueprint_library()

        # 生成车辆
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"✓ 车辆已生成 (ID: {vehicle.id})")

        # 生成相机
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        print(f"✓ 相机已生成 (ID: {camera.id})")

        # 设置相机回调
        def on_image(image):
            # 转换为 NumPy 数组
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            rgb = array[:, :, :3]

            # 在这里处理图像...
            print(f"\r接收图像: 帧 {image.frame}, {image.width}×{image.height}", end='')

        camera.listen(on_image)

        # 启用自动驾驶
        vehicle.set_autopilot(True)
        print("✓ 自动驾驶已启用")

        # 运行 30 秒
        print("\n运行 30 秒...")
        time.sleep(30)

    except KeyboardInterrupt:
        print("\n\n用户中断")

    finally:
        # 清理资源
        print("\n清理资源...")
        if 'camera' in locals():
            camera.stop()
            camera.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()

        print("✓ 完成")


if __name__ == '__main__':
    main()
```

---

## 总结

通过本指南,你已经可以:

✅ **服务器端**:
1. 打包编译好的 CARLA UE5 服务端 (约 15-20 GB)
2. 部署到云GPU服务器或局域网服务器
3. 配置防火墙和网络
4. 启动并验证服务端运行

✅ **客户端**:
1. 打包 Python 客户端 (约 100 MB)
2. 分发给多台机器
3. 安装并配置连接
4. 测试功能和性能

✅ **网络配置**:
1. 开放端口 2000 (RPC) 和 2001 (Streaming)
2. 配置防火墙规则
3. 云服务器安全组设置

✅ **问题排查**:
1. 连接失败的诊断步骤
2. 性能优化方法
3. 常见错误解决方案

现在你可以让一台强大的服务器运行 CARLA 仿真,多台客户端机器同时连接进行数据采集、模型训练和自动驾驶测试!

---

**下一步建议**:

1. **自动化部署**: 编写 Ansible/Terraform 脚本自动化部署流程
2. **容器化**: 使用 Docker 打包服务端,简化部署
3. **负载均衡**: 启动多个 CARLA 实例,使用 Nginx 负载均衡
4. **监控**: 使用 Prometheus + Grafana 监控服务器性能

祝你部署顺利! 🚗💨

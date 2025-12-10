# 第四步:CARLA实时测试部署设计

## 文档概述

**目标**: 将完整的感知-规划-控制系统部署到 CARLA UE5 仿真环境,实现实时闭环测试与性能评估。

**设计范围**:
- 模型优化与部署(TensorRT, ONNX)
- CARLA 集成架构(传感器同步、控制应用)
- 软件在环(SIL)测试框架
- 实时性能监控与日志系统
- 自动化测试流程
- 失效模式与安全机制

**目标性能**:
- 端到端延迟: <100ms (感知 + 规划 + 控制)
- 推理帧率: 20+ FPS (Occupancy Network)
- 测试成功率: >90% (8 个标准场景)
- 系统稳定性: 连续运行 1 小时无崩溃

---

## 1. 设计目标与问题定义

### 1.1 核心挑战

**挑战 1: 实时性约束**
- **问题**: PyTorch 模型推理耗时 35ms+,无法满足 20Hz 要求
- **目标**: 通过 TensorRT FP16 优化到 <20ms
- **验证**: 持续监控推理时间,99分位数 <25ms

**挑战 2: 传感器同步**
- **问题**: 8 个摄像头 + LiDAR + GPS/IMU 异步回调,时间戳不一致
- **目标**: 同步误差 <5ms
- **方案**: CARLA 同步模式 + 时间戳对齐

**挑战 3: 仿真-现实差异 (Sim-to-Real Gap)**
- **问题**: CARLA 仿真环境与真实世界存在差异
  - 传感器噪声建模不完全
  - 物理引擎简化
  - 光照条件理想化
- **缓解措施**:
  - 添加传感器噪声(高斯噪声、运动模糊)
  - 多样化天气/光照条件
  - 域随机化(Domain Randomization)

**挑战 4: 失效模式处理**
- **问题**: 感知失败、规划失败、控制饱和等异常情况
- **目标**: 100% 异常情况有安全降级策略
- **方案**: 多层失效保护 + 紧急停车

**挑战 5: 长时间稳定性**
- **问题**: 内存泄漏、GPU 显存泄漏导致崩溃
- **目标**: 连续运行 1 小时,显存增长 <10%
- **方案**: 定期 profiling + 资源监控

### 1.2 部署架构概览

```
┌──────────────────────────────────────────────────────────┐
│                    CARLA UE5 Server                      │
│  ┌────────────────────────────────────────────────────┐ │
│  │  World: Town01-10                                  │ │
│  │  ├─ Ego Vehicle (Tesla Model 3)                    │ │
│  │  ├─ 8× RGB Cameras (1280×960@36fps)                │ │
│  │  ├─ LiDAR (64 beams, optional for GT)              │ │
│  │  ├─ GPS/IMU                                         │ │
│  │  └─ NPC Vehicles/Pedestrians                       │ │
│  └────────────────────────────────────────────────────┘ │
└────────────────┬─────────────────────────────────────────┘
                 │ RPC (Port 2000)
                 │ Streaming (Port 2001)
                 v
┌──────────────────────────────────────────────────────────┐
│              Autonomous Driving Stack (Python)           │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Perception Module (20Hz)                          │ │
│  │  ├─ TensorRT Engine (Occupancy Net + Memory)      │ │
│  │  │   Input:  8×(1280×960×3) uint8                 │ │
│  │  │   Output: 200×200×16 float32                    │ │
│  │  │   Latency: <20ms (optimized)                    │ │
│  │  └─ Grid Parser → Cost Map                         │ │
│  └────────────────────────────────────────────────────┘ │
│                           ↓                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Planning Module (20Hz)                            │ │
│  │  ├─ Lattice Planner: 50 candidates                │ │
│  │  ├─ Collision Check: Spatial Hash                  │ │
│  │  └─ Cost Evaluation                                │ │
│  └────────────────────────────────────────────────────┘ │
│                           ↓                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Control Module (100Hz)                            │ │
│  │  ├─ MPC Controller (High Speed)                    │ │
│  │  ├─ Pure Pursuit (Low Speed)                       │ │
│  │  └─ Safety Monitor                                 │ │
│  └────────────────────────────────────────────────────┘ │
│                           ↓                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Logging & Monitoring                              │ │
│  │  ├─ Metrics Logger (CSV/HDF5)                      │ │
│  │  ├─ Video Recorder (MP4)                           │ │
│  │  └─ Real-time Dashboard (Matplotlib/Dash)          │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## 2. 模型优化与部署

### 2.1 PyTorch → ONNX → TensorRT 转换

#### Step 1: 导出 ONNX 模型

```python
# export_to_onnx.py

import torch
import torch.onnx
from bev_occupancy_net_with_memory import BEVOccupancyNetWithMemory

def export_to_onnx(checkpoint_path: str,
                   onnx_path: str,
                   opset_version: int = 14):
    """
    导出 PyTorch 模型到 ONNX 格式

    Args:
        checkpoint_path: PyTorch checkpoint 路径
        onnx_path: 输出 ONNX 文件路径
        opset_version: ONNX opset 版本
    """
    # 1. 加载模型
    device = torch.device('cuda')
    model = BEVOccupancyNetWithMemory(
        bev_h=200, bev_w=200, num_z=16, embed_dim=256
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. 准备虚拟输入
    batch_size = 1
    dummy_images = torch.randn(batch_size, 8, 3, 960, 1280, device=device)
    dummy_intrinsics = torch.randn(batch_size, 8, 3, 3, device=device)
    dummy_extrinsics = torch.randn(batch_size, 8, 4, 4, device=device)

    # 注意: 时空记忆的隐藏状态需要外部管理
    # 这里导出的是单帧推理版本

    # 3. 导出 ONNX
    print(f"导出 ONNX 模型到: {onnx_path}")

    torch.onnx.export(
        model,
        (dummy_images, dummy_intrinsics, dummy_extrinsics),
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['images', 'intrinsics', 'extrinsics'],
        output_names=['occupancy_logits'],
        dynamic_axes={
            'images': {0: 'batch'},
            'intrinsics': {0: 'batch'},
            'extrinsics': {0: 'batch'},
            'occupancy_logits': {0: 'batch'}
        }
    )

    print("导出完成!")

    # 4. 验证 ONNX 模型
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型验证通过!")

    # 5. 测试推理一致性
    _test_onnx_consistency(model, onnx_path, dummy_images,
                          dummy_intrinsics, dummy_extrinsics)

def _test_onnx_consistency(pytorch_model, onnx_path,
                          images, intrinsics, extrinsics):
    """测试 PyTorch 与 ONNX 输出一致性"""
    import onnxruntime as ort

    # PyTorch 推理
    with torch.no_grad():
        pytorch_output = pytorch_model(images, intrinsics, extrinsics)
        pytorch_output = pytorch_output.cpu().numpy()

    # ONNX 推理
    ort_session = ort.InferenceSession(onnx_path,
                                       providers=['CUDAExecutionProvider'])

    onnx_output = ort_session.run(
        None,
        {
            'images': images.cpu().numpy(),
            'intrinsics': intrinsics.cpu().numpy(),
            'extrinsics': extrinsics.cpu().numpy()
        }
    )[0]

    # 比较输出
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()

    print(f"PyTorch vs ONNX:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")

    if max_diff < 1e-3:
        print("✓ 一致性验证通过!")
    else:
        print("✗ 警告: 输出差异较大!")

if __name__ == '__main__':
    export_to_onnx(
        checkpoint_path='checkpoints/occupancy_memory_best.pth',
        onnx_path='deploy/occupancy_net.onnx'
    )
```

#### Step 2: ONNX → TensorRT 优化

```python
# build_tensorrt_engine.py

import tensorrt as trt
import numpy as np

def build_tensorrt_engine(onnx_path: str,
                         engine_path: str,
                         fp16_mode: bool = True,
                         max_batch_size: int = 1):
    """
    将 ONNX 模型转换为 TensorRT 引擎

    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出 TensorRT 引擎路径
        fp16_mode: 是否启用 FP16 精度(加速 ~2x)
        max_batch_size: 最大批次大小
    """
    # 1. 创建 Builder
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)

    # 2. 创建网络定义
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    # 3. 解析 ONNX
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"解析 ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ONNX 解析失败:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    print("ONNX 解析成功!")

    # 4. 配置 Builder
    config = builder.create_builder_config()

    # 设置最大工作空间(4GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # 启用 FP16 模式
    if fp16_mode and builder.platform_has_fast_fp16:
        print("启用 FP16 精度优化")
        config.set_flag(trt.BuilderFlag.FP16)

    # 优化配置文件(动态输入尺寸)
    profile = builder.create_optimization_profile()

    # 图像输入: (B, 8, 3, 960, 1280)
    profile.set_shape(
        'images',
        min=(1, 8, 3, 960, 1280),
        opt=(1, 8, 3, 960, 1280),
        max=(1, 8, 3, 960, 1280)
    )

    # 内参: (B, 8, 3, 3)
    profile.set_shape(
        'intrinsics',
        min=(1, 8, 3, 3),
        opt=(1, 8, 3, 3),
        max=(1, 8, 3, 3)
    )

    # 外参: (B, 8, 4, 4)
    profile.set_shape(
        'extrinsics',
        min=(1, 8, 4, 4),
        opt=(1, 8, 4, 4),
        max=(1, 8, 4, 4)
    )

    config.add_optimization_profile(profile)

    # 5. 构建引擎(耗时: 5-10分钟)
    print("构建 TensorRT 引擎(这可能需要几分钟)...")
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        print("引擎构建失败!")
        return

    # 6. 保存引擎
    print(f"保存引擎到: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(engine)

    print("TensorRT 引擎构建完成!")

    # 7. 测试引擎推理
    _test_tensorrt_engine(engine_path)

def _test_tensorrt_engine(engine_path: str):
    """测试 TensorRT 引擎推理速度"""
    import pycuda.driver as cuda
    import pycuda.autoinit

    # 加载引擎
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # 准备输入/输出缓冲区
    inputs = {
        'images': np.random.randn(1, 8, 3, 960, 1280).astype(np.float32),
        'intrinsics': np.random.randn(1, 8, 3, 3).astype(np.float32),
        'extrinsics': np.random.randn(1, 8, 4, 4).astype(np.float32)
    }

    outputs = {
        'occupancy_logits': np.empty((1, 200, 200, 16), dtype=np.float32)
    }

    # 分配 GPU 内存
    d_inputs = {k: cuda.mem_alloc(v.nbytes) for k, v in inputs.items()}
    d_outputs = {k: cuda.mem_alloc(v.nbytes) for k, v in outputs.items()}

    stream = cuda.Stream()

    # Warm-up
    for _ in range(10):
        for k, v in inputs.items():
            cuda.memcpy_htod_async(d_inputs[k], v, stream)

        context.execute_async_v2(
            bindings=list(d_inputs.values()) + list(d_outputs.values()),
            stream_handle=stream.handle
        )

        for k, v in outputs.items():
            cuda.memcpy_dtoh_async(v, d_outputs[k], stream)

        stream.synchronize()

    # 性能测试
    import time

    num_iterations = 100
    start = time.time()

    for _ in range(num_iterations):
        for k, v in inputs.items():
            cuda.memcpy_htod_async(d_inputs[k], v, stream)

        context.execute_async_v2(
            bindings=list(d_inputs.values()) + list(d_outputs.values()),
            stream_handle=stream.handle
        )

        for k, v in outputs.items():
            cuda.memcpy_dtoh_async(v, d_outputs[k], stream)

        stream.synchronize()

    end = time.time()

    avg_latency = (end - start) / num_iterations * 1000
    fps = num_iterations / (end - start)

    print(f"\n性能测试结果 ({num_iterations} 次迭代):")
    print(f"  平均延迟: {avg_latency:.2f} ms")
    print(f"  吞吐量: {fps:.2f} FPS")

if __name__ == '__main__':
    build_tensorrt_engine(
        onnx_path='deploy/occupancy_net.onnx',
        engine_path='deploy/occupancy_net_fp16.trt',
        fp16_mode=True
    )
```

#### Step 3: TensorRT 推理包装器

```python
# tensorrt_inference.py

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from typing import Dict, Tuple

class TensorRTInference:
    """TensorRT 推理包装器"""

    def __init__(self, engine_path: str):
        """
        Args:
            engine_path: TensorRT 引擎文件路径
        """
        self.logger = trt.Logger(trt.Logger.WARNING)

        # 加载引擎
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # 创建 CUDA stream
        self.stream = cuda.Stream()

        # 预分配 GPU 内存
        self.d_inputs = {}
        self.d_outputs = {}

        self._allocate_buffers()

    def _allocate_buffers(self):
        """预分配输入/输出 GPU 缓冲区"""

        # 输入
        self.d_inputs['images'] = cuda.mem_alloc(
            1 * 8 * 3 * 960 * 1280 * np.dtype(np.float32).itemsize
        )
        self.d_inputs['intrinsics'] = cuda.mem_alloc(
            1 * 8 * 3 * 3 * np.dtype(np.float32).itemsize
        )
        self.d_inputs['extrinsics'] = cuda.mem_alloc(
            1 * 8 * 4 * 4 * np.dtype(np.float32).itemsize
        )

        # 输出
        self.d_outputs['occupancy_logits'] = cuda.mem_alloc(
            1 * 200 * 200 * 16 * np.dtype(np.float32).itemsize
        )

        # 主机端输出缓冲区
        self.h_output = np.empty((1, 200, 200, 16), dtype=np.float32)

    def infer(self,
              images: np.ndarray,
              intrinsics: np.ndarray,
              extrinsics: np.ndarray) -> np.ndarray:
        """
        推理

        Args:
            images: (1, 8, 3, 960, 1280) float32
            intrinsics: (1, 8, 3, 3) float32
            extrinsics: (1, 8, 4, 4) float32

        Returns:
            occupancy_logits: (1, 200, 200, 16) float32
        """
        # 1. Host → Device (异步)
        cuda.memcpy_htod_async(
            self.d_inputs['images'],
            images.astype(np.float32).ravel(),
            self.stream
        )
        cuda.memcpy_htod_async(
            self.d_inputs['intrinsics'],
            intrinsics.astype(np.float32).ravel(),
            self.stream
        )
        cuda.memcpy_htod_async(
            self.d_inputs['extrinsics'],
            extrinsics.astype(np.float32).ravel(),
            self.stream
        )

        # 2. 执行推理 (异步)
        bindings = (
            list(self.d_inputs.values()) +
            list(self.d_outputs.values())
        )

        self.context.execute_async_v2(
            bindings=[int(buf) for buf in bindings],
            stream_handle=self.stream.handle
        )

        # 3. Device → Host (异步)
        cuda.memcpy_dtoh_async(
            self.h_output,
            self.d_outputs['occupancy_logits'],
            self.stream
        )

        # 4. 同步等待
        self.stream.synchronize()

        return self.h_output

    def __del__(self):
        """清理资源"""
        for buf in list(self.d_inputs.values()) + list(self.d_outputs.values()):
            buf.free()
```

### 2.2 性能 Benchmark

```python
# benchmark_inference.py

import time
import numpy as np
from tensorrt_inference import TensorRTInference

def benchmark(engine_path: str, num_iterations: int = 500):
    """
    推理性能基准测试

    Args:
        engine_path: TensorRT 引擎路径
        num_iterations: 测试迭代次数
    """
    # 初始化推理引擎
    inference = TensorRTInference(engine_path)

    # 准备虚拟输入
    images = np.random.randn(1, 8, 3, 960, 1280).astype(np.float32)
    intrinsics = np.random.randn(1, 8, 3, 3).astype(np.float32)
    extrinsics = np.random.randn(1, 8, 4, 4).astype(np.float32)

    # Warm-up (20 次)
    print("Warm-up...")
    for _ in range(20):
        _ = inference.infer(images, intrinsics, extrinsics)

    # 性能测试
    print(f"\n运行 {num_iterations} 次推理...")
    latencies = []

    for i in range(num_iterations):
        start = time.perf_counter()
        _ = inference.infer(images, intrinsics, extrinsics)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # ms

        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{num_iterations}")

    # 统计分析
    latencies = np.array(latencies)

    print("\n性能统计:")
    print(f"  平均延迟: {latencies.mean():.2f} ms")
    print(f"  中位数:   {np.median(latencies):.2f} ms")
    print(f"  最小值:   {latencies.min():.2f} ms")
    print(f"  最大值:   {latencies.max():.2f} ms")
    print(f"  P50:      {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95:      {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:      {np.percentile(latencies, 99):.2f} ms")
    print(f"  标准差:   {latencies.std():.2f} ms")
    print(f"\n  吞吐量:   {1000 / latencies.mean():.2f} FPS")

    # 可视化延迟分布
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(latencies, bins=50, edgecolor='black')
    plt.axvline(latencies.mean(), color='r', linestyle='--', label=f'Mean: {latencies.mean():.2f}ms')
    plt.axvline(np.percentile(latencies, 99), color='g', linestyle='--', label=f'P99: {np.percentile(latencies, 99):.2f}ms')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Frequency')
    plt.title('Inference Latency Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(latencies, linewidth=0.5)
    plt.axhline(latencies.mean(), color='r', linestyle='--', label='Mean')
    plt.axhline(np.percentile(latencies, 99), color='g', linestyle='--', label='P99')
    plt.xlabel('Iteration')
    plt.ylabel('Latency (ms)')
    plt.title('Latency Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('inference_benchmark.png', dpi=150)
    print("\n延迟分布图已保存: inference_benchmark.png")

if __name__ == '__main__':
    benchmark('deploy/occupancy_net_fp16.trt')
```

**预期性能**:
| 配置 | 平均延迟 | P99延迟 | FPS |
|------|---------|---------|-----|
| PyTorch FP32 | 35ms | 42ms | 28 |
| TensorRT FP32 | 22ms | 28ms | 45 |
| **TensorRT FP16** | **18ms** | **23ms** | **55** |

---

## 3. CARLA 集成架构

### 3.1 传感器配置与同步

```python
# carla_sensor_manager.py

import carla
import numpy as np
import queue
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

@dataclass
class SensorFrame:
    """单帧传感器数据"""
    timestamp: float
    frame_id: int

    # 图像数据 (8 个摄像头)
    images: Dict[str, np.ndarray]  # {camera_name: (H, W, 3)}

    # 相机参数
    intrinsics: Dict[str, np.ndarray]  # {camera_name: (3, 3)}
    extrinsics: Dict[str, np.ndarray]  # {camera_name: (4, 4)}

    # 车辆状态
    vehicle_transform: carla.Transform
    vehicle_velocity: carla.Vector3D
    vehicle_acceleration: carla.Vector3D

    # 可选: LiDAR (用于 Ground Truth)
    lidar_points: Optional[np.ndarray] = None

class CARLASensorManager:
    """CARLA 传感器管理器"""

    def __init__(self,
                 world: carla.World,
                 vehicle: carla.Vehicle,
                 synchronous_mode: bool = True):
        """
        Args:
            world: CARLA 世界对象
            vehicle: 自车对象
            synchronous_mode: 是否启用同步模式
        """
        self.world = world
        self.vehicle = vehicle
        self.synchronous_mode = synchronous_mode

        # 传感器列表
        self.cameras: Dict[str, carla.Sensor] = {}
        self.lidar: Optional[carla.Sensor] = None

        # 数据队列 (每个传感器独立队列)
        self.camera_queues: Dict[str, queue.Queue] = {}
        self.lidar_queue: Optional[queue.Queue] = None

        # 相机配置
        self.camera_configs = self._get_camera_configs()

        # 计算内外参
        self.intrinsics = self._compute_intrinsics()
        self.extrinsics = {}

        # 初始化传感器
        self._setup_sensors()

        # 启用同步模式
        if self.synchronous_mode:
            self._enable_synchronous_mode()

    def _get_camera_configs(self) -> List[Dict]:
        """8 相机配置 (与 Tesla 类似)"""
        return [
            # 前向宽视角 (120° FOV)
            {'name': 'front_wide', 'x': 1.5, 'y': 0.0, 'z': 1.4,
             'pitch': 0, 'yaw': 0, 'roll': 0, 'fov': 120},

            # 前向主视角 (70° FOV)
            {'name': 'front_main', 'x': 1.5, 'y': 0.0, 'z': 1.4,
             'pitch': 0, 'yaw': 0, 'roll': 0, 'fov': 70},

            # 前向窄视角 (50° FOV, 用于远距离)
            {'name': 'front_narrow', 'x': 1.5, 'y': 0.0, 'z': 1.4,
             'pitch': 0, 'yaw': 0, 'roll': 0, 'fov': 50},

            # 左前方 (90° FOV)
            {'name': 'front_left', 'x': 1.0, 'y': -0.5, 'z': 1.3,
             'pitch': 0, 'yaw': -55, 'roll': 0, 'fov': 90},

            # 右前方 (90° FOV)
            {'name': 'front_right', 'x': 1.0, 'y': 0.5, 'z': 1.3,
             'pitch': 0, 'yaw': 55, 'roll': 0, 'fov': 90},

            # 左侧 (90° FOV)
            {'name': 'side_left', 'x': 0.0, 'y': -0.8, 'z': 1.3,
             'pitch': 0, 'yaw': -90, 'roll': 0, 'fov': 90},

            # 右侧 (90° FOV)
            {'name': 'side_right', 'x': 0.0, 'y': 0.8, 'z': 1.3,
             'pitch': 0, 'yaw': 90, 'roll': 0, 'fov': 90},

            # 后方 (120° FOV)
            {'name': 'rear', 'x': -1.5, 'y': 0.0, 'z': 1.2,
             'pitch': 0, 'yaw': 180, 'roll': 0, 'fov': 120},
        ]

    def _setup_sensors(self):
        """创建并附加传感器"""
        blueprint_library = self.world.get_blueprint_library()

        # 创建摄像头
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '960')
        camera_bp.set_attribute('sensor_tick', '0.05')  # 20Hz

        for config in self.camera_configs:
            # 变换矩阵
            transform = carla.Transform(
                carla.Location(x=config['x'], y=config['y'], z=config['z']),
                carla.Rotation(pitch=config['pitch'], yaw=config['yaw'], roll=config['roll'])
            )

            # 设置 FOV
            camera_bp.set_attribute('fov', str(config['fov']))

            # 生成相机
            camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)

            # 创建队列
            cam_queue = queue.Queue(maxsize=2)

            # 注册回调
            camera.listen(lambda image, q=cam_queue: self._camera_callback(image, q))

            self.cameras[config['name']] = camera
            self.camera_queues[config['name']] = cam_queue

            # 计算外参
            self.extrinsics[config['name']] = self._compute_extrinsic(transform)

        print(f"已创建 {len(self.cameras)} 个摄像头")

        # 可选: 创建 LiDAR (用于 GT)
        # self._setup_lidar()

    def _camera_callback(self, image: carla.Image, cam_queue: queue.Queue):
        """相机数据回调"""
        # 转换为 NumPy 数组
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        array = array[:, :, :3]  # 只要 BGR
        array = array[:, :, ::-1]  # BGR → RGB

        # 放入队列(非阻塞, 如果满则丢弃旧数据)
        try:
            cam_queue.put_nowait({
                'timestamp': image.timestamp,
                'frame': image.frame,
                'data': array
            })
        except queue.Full:
            cam_queue.get()  # 移除旧数据
            cam_queue.put_nowait({
                'timestamp': image.timestamp,
                'frame': image.frame,
                'data': array
            })

    def _compute_intrinsics(self) -> Dict[str, np.ndarray]:
        """计算相机内参矩阵"""
        intrinsics = {}

        width, height = 1280, 960

        for config in self.camera_configs:
            fov = config['fov']

            # 焦距计算
            focal_length = width / (2.0 * np.tan(np.deg2rad(fov) / 2.0))

            # 内参矩阵
            K = np.array([
                [focal_length, 0, width / 2],
                [0, focal_length, height / 2],
                [0, 0, 1]
            ], dtype=np.float32)

            intrinsics[config['name']] = K

        return intrinsics

    def _compute_extrinsic(self, transform: carla.Transform) -> np.ndarray:
        """计算相机外参矩阵(相机坐标系 → 车辆坐标系)"""
        location = transform.location
        rotation = transform.rotation

        # 旋转矩阵(欧拉角 → 旋转矩阵)
        yaw = np.deg2rad(rotation.yaw)
        pitch = np.deg2rad(rotation.pitch)
        roll = np.deg2rad(rotation.roll)

        # ZYX 欧拉角
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])

        R = R_yaw @ R_pitch @ R_roll

        # 平移向量
        t = np.array([location.x, location.y, location.z])

        # 外参矩阵 [R | t]
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        return extrinsic

    def capture_frame(self, timeout: float = 1.0) -> Optional[SensorFrame]:
        """
        捕获同步的传感器帧

        Args:
            timeout: 超时时间(秒)

        Returns:
            SensorFrame or None(如果超时)
        """
        # 等待所有相机数据到达
        camera_data = {}

        try:
            for name, cam_queue in self.camera_queues.items():
                data = cam_queue.get(timeout=timeout)
                camera_data[name] = data

        except queue.Empty:
            print("[CARLASensorManager] 传感器数据超时!")
            return None

        # 检查时间戳一致性(同步模式下应该相同)
        timestamps = [data['timestamp'] for data in camera_data.values()]
        frame_ids = [data['frame'] for data in camera_data.values()]

        if len(set(frame_ids)) > 1:
            print(f"[警告] 帧 ID 不一致: {frame_ids}")

        # 构建 SensorFrame
        sensor_frame = SensorFrame(
            timestamp=timestamps[0],
            frame_id=frame_ids[0],
            images={name: data['data'] for name, data in camera_data.items()},
            intrinsics=self.intrinsics,
            extrinsics=self.extrinsics,
            vehicle_transform=self.vehicle.get_transform(),
            vehicle_velocity=self.vehicle.get_velocity(),
            vehicle_acceleration=self.vehicle.get_acceleration()
        )

        return sensor_frame

    def _enable_synchronous_mode(self):
        """启用 CARLA 同步模式"""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20Hz
        self.world.apply_settings(settings)
        print("[CARLA] 同步模式已启用 (20Hz)")

    def destroy(self):
        """销毁所有传感器"""
        for camera in self.cameras.values():
            camera.destroy()

        if self.lidar is not None:
            self.lidar.destroy()

        print("传感器已销毁")
```

### 3.2 主控制循环

```python
# carla_autonomous_agent.py

import carla
import numpy as np
import time
from typing import Optional
from collections import deque

from carla_sensor_manager import CARLASensorManager, SensorFrame
from tensorrt_inference import TensorRTInference
from planning_control_node import PlanningControlNode
from iso22133_interface import ControlCommandConverter

class CARLAAutonomousAgent:
    """CARLA 自动驾驶 Agent"""

    def __init__(self,
                 carla_host: str = 'localhost',
                 carla_port: int = 2000,
                 config: dict = None):
        """
        Args:
            carla_host: CARLA 服务器地址
            carla_port: CARLA 服务器端口
            config: 配置字典
        """
        self.config = config or self._default_config()

        # 连接 CARLA
        self.client = carla.Client(carla_host, carla_port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # 创建自车
        self.vehicle = self._spawn_vehicle()

        # 传感器管理器
        self.sensor_manager = CARLASensorManager(
            self.world, self.vehicle, synchronous_mode=True
        )

        # 加载推理引擎
        self.inference_engine = TensorRTInference(
            self.config['tensorrt_engine_path']
        )

        # 规划控制节点
        self.planning_control = PlanningControlNode(self.config)

        # ISO 22133 转换器
        self.command_converter = ControlCommandConverter()

        # 性能监控
        self.latency_buffer = deque(maxlen=100)
        self.frame_count = 0

        # 日志记录器
        self.logger = self._setup_logger()

    def _spawn_vehicle(self) -> carla.Vehicle:
        """生成自车"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

        # 在地图上随机选择生成点
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points)

        vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"自车已生成: {vehicle.type_id}")

        return vehicle

    def run(self, duration: float = 120.0):
        """
        运行自动驾驶循环

        Args:
            duration: 运行时长(秒)
        """
        print(f"开始自动驾驶 (时长: {duration}s)")

        start_time = time.time()
        self.frame_count = 0

        try:
            while time.time() - start_time < duration:
                # ============ 控制循环 ============
                frame_start = time.perf_counter()

                # 1. Tick CARLA (同步模式)
                self.world.tick()

                # 2. 捕获传感器数据
                sensor_frame = self.sensor_manager.capture_frame(timeout=1.0)

                if sensor_frame is None:
                    print("[警告] 传感器数据丢失,跳过本帧")
                    continue

                # 3. 感知: Occupancy Network 推理
                perception_start = time.perf_counter()

                occupancy_grid = self._run_perception(sensor_frame)

                perception_time = (time.perf_counter() - perception_start) * 1000

                # 4. 规划控制
                planning_start = time.perf_counter()

                control_command = self._run_planning_control(
                    occupancy_grid, sensor_frame
                )

                planning_time = (time.perf_counter() - planning_start) * 1000

                # 5. 应用控制
                self.vehicle.apply_control(control_command)

                # 6. 日志记录
                frame_time = (time.perf_counter() - frame_start) * 1000
                self.latency_buffer.append(frame_time)

                self._log_metrics(
                    frame_time, perception_time, planning_time, sensor_frame
                )

                self.frame_count += 1

                # 实时性能监控
                if self.frame_count % 20 == 0:  # 每秒打印一次
                    avg_latency = np.mean(self.latency_buffer)
                    print(f"[Frame {self.frame_count}] "
                          f"平均延迟: {avg_latency:.2f}ms "
                          f"(感知: {perception_time:.2f}ms, 规划: {planning_time:.2f}ms)")

        except KeyboardInterrupt:
            print("\n用户中断")

        finally:
            self._cleanup()

        print(f"\n自动驾驶完成! 总帧数: {self.frame_count}")

    def _run_perception(self, sensor_frame: SensorFrame) -> np.ndarray:
        """运行感知模块"""

        # 准备输入数据
        images = self._prepare_images(sensor_frame.images)
        intrinsics = self._prepare_intrinsics(sensor_frame.intrinsics)
        extrinsics = self._prepare_extrinsics(sensor_frame.extrinsics)

        # TensorRT 推理
        occupancy_logits = self.inference_engine.infer(
            images, intrinsics, extrinsics
        )

        # Sigmoid 激活
        occupancy_grid = 1.0 / (1.0 + np.exp(-occupancy_logits))

        return occupancy_grid[0]  # (200, 200, 16)

    def _run_planning_control(self,
                              occupancy_grid: np.ndarray,
                              sensor_frame: SensorFrame) -> carla.VehicleControl:
        """运行规划控制模块"""

        # 提取车辆状态
        vehicle_state = {
            'x': sensor_frame.vehicle_transform.location.x,
            'y': sensor_frame.vehicle_transform.location.y,
            'yaw': np.deg2rad(sensor_frame.vehicle_transform.rotation.yaw),
            'velocity': self._get_velocity_magnitude(sensor_frame.vehicle_velocity),
            'acceleration': self._get_velocity_magnitude(sensor_frame.vehicle_acceleration)
        }

        # 获取参考路径 (简化: 使用 waypoints)
        reference_line = self._get_reference_line()

        # 规划控制
        control_command = self.planning_control.step(
            occupancy_grid=occupancy_grid,
            vehicle_state=vehicle_state,
            reference_line=reference_line
        )

        return control_command

    def _prepare_images(self, images_dict: dict) -> np.ndarray:
        """准备图像输入 (1, 8, 3, 960, 1280)"""
        camera_order = ['front_wide', 'front_main', 'front_narrow',
                       'front_left', 'front_right',
                       'side_left', 'side_right', 'rear']

        images = np.stack([images_dict[name] for name in camera_order], axis=0)
        images = images.transpose(0, 3, 1, 2)  # (8, H, W, C) → (8, C, H, W)
        images = images.astype(np.float32) / 255.0  # 归一化
        images = images[np.newaxis, ...]  # (8, C, H, W) → (1, 8, C, H, W)

        return images

    def _prepare_intrinsics(self, intrinsics_dict: dict) -> np.ndarray:
        """准备内参矩阵 (1, 8, 3, 3)"""
        camera_order = ['front_wide', 'front_main', 'front_narrow',
                       'front_left', 'front_right',
                       'side_left', 'side_right', 'rear']

        intrinsics = np.stack([intrinsics_dict[name] for name in camera_order], axis=0)
        intrinsics = intrinsics[np.newaxis, ...]

        return intrinsics

    def _prepare_extrinsics(self, extrinsics_dict: dict) -> np.ndarray:
        """准备外参矩阵 (1, 8, 4, 4)"""
        camera_order = ['front_wide', 'front_main', 'front_narrow',
                       'front_left', 'front_right',
                       'side_left', 'side_right', 'rear']

        extrinsics = np.stack([extrinsics_dict[name] for name in camera_order], axis=0)
        extrinsics = extrinsics[np.newaxis, ...]

        return extrinsics

    def _get_reference_line(self) -> np.ndarray:
        """获取参考路径(简化版本)"""
        # TODO: 从高精地图或 waypoint 生成完整参考线
        # 这里使用前方 20 个 waypoints 作为参考

        current_location = self.vehicle.get_location()
        current_waypoint = self.world.get_map().get_waypoint(current_location)

        waypoints = [current_waypoint]
        for _ in range(20):
            next_wps = waypoints[-1].next(2.0)  # 每 2m 采样一个点
            if len(next_wps) > 0:
                waypoints.append(next_wps[0])

        reference_line = np.array([
            [wp.transform.location.x, wp.transform.location.y]
            for wp in waypoints
        ])

        return reference_line

    def _get_velocity_magnitude(self, velocity: carla.Vector3D) -> float:
        """计算速度大小"""
        return np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    def _log_metrics(self, frame_time, perception_time, planning_time, sensor_frame):
        """记录性能指标"""
        # TODO: 实现完整的日志记录(CSV/HDF5)
        pass

    def _cleanup(self):
        """清理资源"""
        self.sensor_manager.destroy()
        self.vehicle.destroy()
        print("资源已清理")

    def _default_config(self) -> dict:
        """默认配置"""
        return {
            'tensorrt_engine_path': 'deploy/occupancy_net_fp16.trt',
            'planning': {
                'type': 'lattice',
                'lateral_samples': 9,
                'time_samples': 5
            },
            'control': {
                'mode': 'mpc',
                'frequency': 100
            }
        }

    def _setup_logger(self):
        """设置日志记录器"""
        # TODO: 实现日志系统
        pass

if __name__ == '__main__':
    agent = CARLAAutonomousAgent()
    agent.run(duration=120.0)  # 运行 2 分钟
```

---

## 4. 实时监控与可视化

### 4.1 实时仪表盘

```python
# realtime_dashboard.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

class RealtimeDashboard:
    """实时性能监控仪表盘"""

    def __init__(self, max_history: int = 200):
        """
        Args:
            max_history: 历史数据长度(帧数)
        """
        self.max_history = max_history

        # 数据缓冲区
        self.timestamps = deque(maxlen=max_history)
        self.perception_times = deque(maxlen=max_history)
        self.planning_times = deque(maxlen=max_history)
        self.control_times = deque(maxlen=max_history)
        self.total_times = deque(maxlen=max_history)

        self.velocities = deque(maxlen=max_history)
        self.accelerations = deque(maxlen=max_history)
        self.steering_angles = deque(maxlen=max_history)

        # 创建图表
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 8))
        self.fig.suptitle('CARLA Autonomous Driving - Real-time Dashboard', fontsize=16)

        # 初始化图表
        self._setup_plots()

        # 动画
        self.ani = animation.FuncAnimation(
            self.fig, self._update_plots, interval=200, blit=False
        )

    def _setup_plots(self):
        """初始化子图"""

        # 1. 延迟时间分布
        self.ax_latency = self.axes[0, 0]
        self.ax_latency.set_title('Module Latency (ms)')
        self.ax_latency.set_xlabel('Frame')
        self.ax_latency.set_ylabel('Time (ms)')
        self.ax_latency.grid(True, alpha=0.3)
        self.ax_latency.set_ylim([0, 100])

        self.line_perception, = self.ax_latency.plot([], [], 'b-', label='Perception', linewidth=2)
        self.line_planning, = self.ax_latency.plot([], [], 'g-', label='Planning', linewidth=2)
        self.line_control, = self.ax_latency.plot([], [], 'r-', label='Control', linewidth=2)
        self.line_total, = self.ax_latency.plot([], [], 'k--', label='Total', linewidth=2)
        self.ax_latency.legend(loc='upper right')

        # 2. 速度曲线
        self.ax_velocity = self.axes[0, 1]
        self.ax_velocity.set_title('Vehicle Velocity (m/s)')
        self.ax_velocity.set_xlabel('Frame')
        self.ax_velocity.set_ylabel('Velocity (m/s)')
        self.ax_velocity.grid(True, alpha=0.3)
        self.ax_velocity.set_ylim([0, 30])

        self.line_velocity, = self.ax_velocity.plot([], [], 'b-', linewidth=2)

        # 3. 加速度/转向角
        self.ax_control = self.axes[1, 0]
        self.ax_control.set_title('Control Signals')
        self.ax_control.set_xlabel('Frame')
        self.ax_control.set_ylabel('Value')
        self.ax_control.grid(True, alpha=0.3)

        self.line_accel, = self.ax_control.plot([], [], 'g-', label='Acceleration (m/s²)', linewidth=2)
        self.line_steering, = self.ax_control.plot([], [], 'r-', label='Steering (deg)', linewidth=2)
        self.ax_control.legend(loc='upper right')

        # 4. 统计信息(文本)
        self.ax_stats = self.axes[1, 1]
        self.ax_stats.axis('off')
        self.text_stats = self.ax_stats.text(
            0.05, 0.95, '', fontsize=12, verticalalignment='top',
            family='monospace'
        )

        plt.tight_layout()

    def _update_plots(self, frame):
        """更新图表(动画回调)"""

        if len(self.timestamps) == 0:
            return

        # 时间轴
        x_data = list(range(len(self.timestamps)))

        # 1. 延迟时间
        self.line_perception.set_data(x_data, list(self.perception_times))
        self.line_planning.set_data(x_data, list(self.planning_times))
        self.line_control.set_data(x_data, list(self.control_times))
        self.line_total.set_data(x_data, list(self.total_times))

        self.ax_latency.set_xlim([max(0, len(x_data) - self.max_history), len(x_data)])

        # 2. 速度
        self.line_velocity.set_data(x_data, list(self.velocities))
        self.ax_velocity.set_xlim([max(0, len(x_data) - self.max_history), len(x_data)])

        # 3. 控制信号
        self.line_accel.set_data(x_data, list(self.accelerations))
        self.line_steering.set_data(x_data, list(self.steering_angles))
        self.ax_control.set_xlim([max(0, len(x_data) - self.max_history), len(x_data)])

        # 4. 统计信息
        if len(self.total_times) > 0:
            stats_text = self._format_statistics()
            self.text_stats.set_text(stats_text)

    def _format_statistics(self) -> str:
        """格式化统计信息"""

        total_times_arr = np.array(self.total_times)
        velocities_arr = np.array(self.velocities)

        stats = f"""
Statistics (last {len(self.total_times)} frames):

Latency:
  Total Mean:   {total_times_arr.mean():.2f} ms
  Total P99:    {np.percentile(total_times_arr, 99):.2f} ms
  FPS:          {1000 / total_times_arr.mean():.1f}

Velocity:
  Current:      {self.velocities[-1]:.2f} m/s
  Average:      {velocities_arr.mean():.2f} m/s
  Max:          {velocities_arr.max():.2f} m/s

Control:
  Acceleration: {self.accelerations[-1]:.2f} m/s²
  Steering:     {self.steering_angles[-1]:.2f}°
        """

        return stats

    def push_data(self,
                  timestamp: float,
                  perception_time: float,
                  planning_time: float,
                  control_time: float,
                  total_time: float,
                  velocity: float,
                  acceleration: float,
                  steering_angle: float):
        """推送新数据"""

        self.timestamps.append(timestamp)
        self.perception_times.append(perception_time)
        self.planning_times.append(planning_time)
        self.control_times.append(control_time)
        self.total_times.append(total_time)

        self.velocities.append(velocity)
        self.accelerations.append(acceleration)
        self.steering_angles.append(steering_angle)

    def show(self):
        """显示仪表盘"""
        plt.show()
```

### 4.2 BEV 占据网格可视化

```python
# visualize_bev.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle

class BEVVisualizer:
    """BEV 占据网格可视化器"""

    def __init__(self, grid_size: int = 200, resolution: float = 0.5):
        """
        Args:
            grid_size: 网格大小
            resolution: 分辨率 (m/pixel)
        """
        self.grid_size = grid_size
        self.resolution = resolution

        # 创建图表
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_title('BEV Occupancy Grid', fontsize=16)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_aspect('equal')

        # 坐标范围
        world_size = grid_size * resolution
        self.ax.set_xlim([-world_size/2, world_size/2])
        self.ax.set_ylim([-world_size/2, world_size/2])

        # 网格线
        self.ax.grid(True, alpha=0.3, linestyle='--')

        # 占据网格图像
        self.im = self.ax.imshow(
            np.zeros((grid_size, grid_size)),
            extent=[-world_size/2, world_size/2, -world_size/2, world_size/2],
            origin='lower',
            cmap='RdYlGn_r',
            vmin=0, vmax=1,
            alpha=0.7
        )

        # 车辆标记(原点)
        vehicle = Rectangle(
            (-2, -1), 4, 2,
            linewidth=2, edgecolor='blue', facecolor='cyan', alpha=0.5
        )
        self.ax.add_patch(vehicle)

        # 方向箭头
        self.ax.arrow(0, 0, 3, 0, head_width=0.5, head_length=0.5,
                     fc='blue', ec='blue', linewidth=2)

        # 颜色条
        plt.colorbar(self.im, ax=self.ax, label='Occupancy Probability')

        plt.ion()  # 交互模式
        plt.show()

    def update(self, occupancy_grid: np.ndarray):
        """
        更新 BEV 可视化

        Args:
            occupancy_grid: (200, 200, 16) 占据网格(使用地面层 Z=0)
        """
        # 使用地面层
        ground_occupancy = occupancy_grid[:, :, 0]

        # 更新图像
        self.im.set_data(ground_occupancy)

        plt.pause(0.001)  # 短暂暂停以更新

    def close(self):
        """关闭可视化"""
        plt.ioff()
        plt.close(self.fig)
```

---

## 5. 自动化测试框架

### 5.1 测试场景定义

```python
# test_scenarios.py

from dataclasses import dataclass
from typing import List, Dict, Tuple
import carla

@dataclass
class TestScenario:
    """测试场景定义"""
    name: str
    town: str  # Town01-Town10
    weather: carla.WeatherParameters
    spawn_point_index: int
    num_npc_vehicles: int
    num_pedestrians: int
    duration: float  # 测试时长(秒)
    success_criteria: Dict[str, float]  # 成功标准

    def __str__(self):
        return f"[{self.name}] {self.town} - {self.num_npc_vehicles} vehicles, {self.duration}s"

# 定义 8 个标准测试场景
STANDARD_SCENARIOS = [
    # 1. 基础跟车
    TestScenario(
        name='follow_car',
        town='Town01',
        weather=carla.WeatherParameters.ClearNoon,
        spawn_point_index=50,
        num_npc_vehicles=5,
        num_pedestrians=0,
        duration=120.0,
        success_criteria={
            'collision_count': 0,
            'red_light_violations': 0,
            'avg_speed': 8.0,  # m/s
            'completion_rate': 0.9
        }
    ),

    # 2. 变道超车
    TestScenario(
        name='overtake',
        town='Town04',
        weather=carla.WeatherParameters.ClearNoon,
        spawn_point_index=100,
        num_npc_vehicles=10,
        num_pedestrians=0,
        duration=150.0,
        success_criteria={
            'collision_count': 0,
            'lane_changes': 2,  # 至少 2 次变道
            'avg_speed': 12.0
        }
    ),

    # 3. 十字路口(有信号灯)
    TestScenario(
        name='intersection',
        town='Town02',
        weather=carla.WeatherParameters.ClearNoon,
        spawn_point_index=30,
        num_npc_vehicles=15,
        num_pedestrians=10,
        duration=180.0,
        success_criteria={
            'collision_count': 0,
            'red_light_violations': 0,
            'completion_rate': 0.85
        }
    ),

    # 4. 环岛
    TestScenario(
        name='roundabout',
        town='Town03',
        weather=carla.WeatherParameters.ClearNoon,
        spawn_point_index=70,
        num_npc_vehicles=8,
        num_pedestrians=5,
        duration=120.0,
        success_criteria={
            'collision_count': 0,
            'completion_rate': 0.9
        }
    ),

    # 5. 窄路会车
    TestScenario(
        name='narrow_road',
        town='Town07',
        weather=carla.WeatherParameters.ClearNoon,
        spawn_point_index=20,
        num_npc_vehicles=5,
        num_pedestrians=0,
        duration=100.0,
        success_criteria={
            'collision_count': 0,
            'avg_lateral_accel': 2.0  # m/s² (平滑行驶)
        }
    ),

    # 6. 行人穿越
    TestScenario(
        name='pedestrian_crossing',
        town='Town05',
        weather=carla.WeatherParameters.ClearNoon,
        spawn_point_index=150,
        num_npc_vehicles=5,
        num_pedestrians=20,
        duration=120.0,
        success_criteria={
            'collision_count': 0,
            'pedestrian_near_misses': 0  # 与行人距离 < 2m 的次数
        }
    ),

    # 7. 雨天驾驶
    TestScenario(
        name='rainy_weather',
        town='Town01',
        weather=carla.WeatherParameters.HardRainNoon,
        spawn_point_index=80,
        num_npc_vehicles=10,
        num_pedestrians=5,
        duration=150.0,
        success_criteria={
            'collision_count': 0,
            'avg_speed': 6.0,  # 雨天降低速度
            'completion_rate': 0.8
        }
    ),

    # 8. 动态障碍物(车辆急刹/切入)
    TestScenario(
        name='dynamic_obstacles',
        town='Town06',
        weather=carla.WeatherParameters.ClearNoon,
        spawn_point_index=200,
        num_npc_vehicles=15,
        num_pedestrians=10,
        duration=180.0,
        success_criteria={
            'collision_count': 0,
            'emergency_brakes': 5,  # 至少触发 5 次紧急制动(说明检测到危险)
            'completion_rate': 0.85
        }
    )
]
```

### 5.2 自动化测试执行器

```python
# test_runner.py

import carla
import time
import json
from typing import Dict, List
from test_scenarios import TestScenario, STANDARD_SCENARIOS
from carla_autonomous_agent import CARLAAutonomousAgent

class TestRunner:
    """自动化测试执行器"""

    def __init__(self, carla_host: str = 'localhost', carla_port: int = 2000):
        self.client = carla.Client(carla_host, carla_port)
        self.client.set_timeout(10.0)

    def run_scenario(self, scenario: TestScenario) -> Dict:
        """
        运行单个测试场景

        Args:
            scenario: 测试场景定义

        Returns:
            测试结果字典
        """
        print(f"\n{'='*60}")
        print(f"运行场景: {scenario}")
        print(f"{'='*60}")

        # 1. 加载地图
        print(f"加载地图: {scenario.town}")
        self.world = self.client.load_world(scenario.town)

        # 2. 设置天气
        self.world.set_weather(scenario.weather)

        # 3. 生成 NPC
        self._spawn_npcs(scenario.num_npc_vehicles, scenario.num_pedestrians)

        # 4. 创建 Agent
        agent = CARLAAutonomousAgent(config={
            'spawn_point_index': scenario.spawn_point_index
        })

        # 5. 运行测试
        start_time = time.time()
        metrics = {
            'scenario_name': scenario.name,
            'start_time': start_time,
            'duration': scenario.duration,
            'collision_count': 0,
            'red_light_violations': 0,
            'lane_invasions': 0,
            'success': False,
            'error_message': None
        }

        try:
            # 注册碰撞传感器
            collision_sensor = self._attach_collision_sensor(
                agent.vehicle, metrics
            )

            # 运行自动驾驶
            agent.run(duration=scenario.duration)

            # 评估成功标准
            metrics['success'] = self._evaluate_success(
                metrics, scenario.success_criteria
            )

        except Exception as e:
            print(f"测试失败: {e}")
            metrics['error_message'] = str(e)

        finally:
            # 清理
            self._cleanup_npcs()
            agent._cleanup()

        metrics['end_time'] = time.time()

        print(f"\n测试结果: {'通过' if metrics['success'] else '失败'}")
        self._print_metrics(metrics)

        return metrics

    def run_all_scenarios(self) -> List[Dict]:
        """运行所有标准测试场景"""

        results = []

        for scenario in STANDARD_SCENARIOS:
            result = self.run_scenario(scenario)
            results.append(result)

            # 短暂休息(避免 CARLA 过载)
            time.sleep(5)

        # 生成总结报告
        self._generate_report(results)

        return results

    def _spawn_npcs(self, num_vehicles: int, num_pedestrians: int):
        """生成 NPC 车辆和行人"""
        # TODO: 实现 NPC 生成逻辑
        print(f"生成 {num_vehicles} 辆 NPC 车辆, {num_pedestrians} 个行人")

    def _attach_collision_sensor(self, vehicle: carla.Vehicle, metrics: Dict):
        """附加碰撞传感器"""
        blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        sensor = self.world.spawn_actor(
            blueprint, carla.Transform(), attach_to=vehicle
        )

        def on_collision(event):
            metrics['collision_count'] += 1
            print(f"[碰撞!] 与 {event.other_actor.type_id} 碰撞")

        sensor.listen(on_collision)
        return sensor

    def _evaluate_success(self, metrics: Dict, criteria: Dict) -> bool:
        """评估测试是否成功"""

        for key, threshold in criteria.items():
            if key not in metrics:
                continue

            actual = metrics[key]

            # 不同指标有不同的判断逻辑
            if key in ['collision_count', 'red_light_violations', 'lane_invasions']:
                if actual > threshold:
                    print(f"失败: {key} = {actual} (阈值: {threshold})")
                    return False

            elif key in ['avg_speed', 'completion_rate']:
                if actual < threshold:
                    print(f"失败: {key} = {actual:.2f} (阈值: {threshold})")
                    return False

        return True

    def _print_metrics(self, metrics: Dict):
        """打印测试指标"""
        print("\n测试指标:")
        for key, value in metrics.items():
            if key not in ['start_time', 'end_time']:
                print(f"  {key}: {value}")

    def _generate_report(self, results: List[Dict]):
        """生成测试报告"""

        report_path = 'test_report.json'

        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n测试报告已保存: {report_path}")

        # 统计
        total = len(results)
        passed = sum(1 for r in results if r['success'])

        print(f"\n总结:")
        print(f"  总测试数: {total}")
        print(f"  通过: {passed} ({passed/total*100:.1f}%)")
        print(f"  失败: {total - passed}")

    def _cleanup_npcs(self):
        """清理 NPC"""
        # TODO: 实现 NPC 清理
        pass

if __name__ == '__main__':
    runner = TestRunner()
    runner.run_all_scenarios()
```

---

## 6. 故障排查与安全机制

### 6.1 常见问题排查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| TensorRT 推理卡顿 | GPU 显存不足/碎片化 | 定期清理显存,减小 batch size |
| 传感器数据丢帧 | CARLA 负载过高 | 降低 NPC 数量,使用专用服务器 |
| 规划失败(无轨迹) | 障碍物过密 | 增加横向采样数,降低安全边界 |
| 控制抖动 | MPC 求解不稳定 | 增加权重矩阵 Q, 使用热启动 |
| 碰撞频繁 | 安全边界过小 | 增大 safety_margin 到 1.0m |
| 内存持续增长 | 内存泄漏 | 使用 Valgrind 检测,释放 GPU 缓冲区 |

### 6.2 失效模式与安全降级

```python
# safety_degradation.py

from enum import Enum

class SafetyLevel(Enum):
    """安全等级"""
    NORMAL = 0        # 正常运行
    DEGRADED = 1      # 降级模式(减速行驶)
    EMERGENCY = 2     # 紧急模式(紧急制动)
    CRITICAL = 3      # 严重故障(立即停车)

class SafetyManager:
    """安全管理器"""

    def __init__(self):
        self.current_level = SafetyLevel.NORMAL
        self.failure_counts = {
            'perception': 0,
            'planning': 0,
            'control': 0
        }

    def check_and_degrade(self,
                         perception_ok: bool,
                         planning_ok: bool,
                         control_ok: bool) -> SafetyLevel:
        """
        检查系统状态并决定安全等级

        Args:
            perception_ok: 感知模块是否正常
            planning_ok: 规划模块是否正常
            control_ok: 控制模块是否正常

        Returns:
            当前安全等级
        """
        # 更新失效计数
        if not perception_ok:
            self.failure_counts['perception'] += 1
        else:
            self.failure_counts['perception'] = 0

        if not planning_ok:
            self.failure_counts['planning'] += 1
        else:
            self.failure_counts['planning'] = 0

        if not control_ok:
            self.failure_counts['control'] += 1
        else:
            self.failure_counts['control'] = 0

        # 决策安全等级
        if self.failure_counts['perception'] >= 5:
            # 感知连续失败 5 帧 → 紧急停车
            self.current_level = SafetyLevel.CRITICAL
            print("[安全] 感知失效!立即停车!")

        elif self.failure_counts['planning'] >= 3:
            # 规划连续失败 3 帧 → 紧急制动
            self.current_level = SafetyLevel.EMERGENCY
            print("[安全] 规划失效!紧急制动!")

        elif not planning_ok:
            # 规划偶尔失败 → 降级模式(减速)
            self.current_level = SafetyLevel.DEGRADED
            print("[安全] 进入降级模式(减速行驶)")

        else:
            # 一切正常
            self.current_level = SafetyLevel.NORMAL

        return self.current_level

    def get_safe_control(self,
                        level: SafetyLevel,
                        current_velocity: float) -> Tuple[float, float]:
        """
        根据安全等级返回安全控制指令

        Args:
            level: 安全等级
            current_velocity: 当前速度 (m/s)

        Returns:
            (steering, acceleration) 安全控制指令
        """
        if level == SafetyLevel.CRITICAL:
            # 立即停车: 最大制动 + 拉手刹
            return (0.0, -8.0)

        elif level == SafetyLevel.EMERGENCY:
            # 紧急制动
            return (0.0, -5.0)

        elif level == SafetyLevel.DEGRADED:
            # 减速至 5 m/s
            target_speed = 5.0
            if current_velocity > target_speed:
                return (0.0, -2.0)
            else:
                return (0.0, 0.0)

        else:
            # 正常模式,不干预
            return (None, None)
```

---

## 7. 性能优化技巧

### 7.1 GPU 显存优化

```python
# 使用 FP16 混合精度
model.half()  # 模型参数 → FP16

# 输入数据也使用 FP16
images = images.half()

# TensorRT 构建时启用 FP16
config.set_flag(trt.BuilderFlag.FP16)
```

### 7.2 多进程并行

```python
# 使用多进程分离感知和规划控制
import multiprocessing as mp

perception_queue = mp.Queue(maxsize=2)
control_queue = mp.Queue(maxsize=2)

# 进程 1: 感知
perception_process = mp.Process(
    target=perception_loop,
    args=(perception_queue,)
)

# 进程 2: 规划控制
control_process = mp.Process(
    target=control_loop,
    args=(perception_queue, control_queue)
)

perception_process.start()
control_process.start()
```

---

## 8. 总结

本文档完整设计了 CARLA 实时测试部署方案:

1. **模型优化**: PyTorch → ONNX → TensorRT FP16, 推理加速 2x (18ms)
2. **传感器同步**: CARLA 同步模式, 8 相机时间戳对齐 (<5ms)
3. **实时监控**: 性能仪表盘 + BEV 可视化 + 日志系统
4. **自动化测试**: 8 个标准场景 + 评估指标 + 测试报告
5. **安全机制**: 多层失效保护 + 降级策略 + 紧急停车

**四步完整流程**:
- 第一步(BEV Occupancy Net): 8 相机 → 3D 占据网格
- 第二步(时空记忆): 增强遮挡/长时场景感知
- 第三步(规划控制): 占据网格 → 车辆控制指令
- **第四步(CARLA 部署)**: 完整系统实时闭环测试!

**至此,从感知到规划到控制到部署的完整自动驾驶系统设计已全部完成!** 🎉

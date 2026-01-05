# 地面全灰与坐标系修复方案

## 1. 问题现象分析
用户反馈修复后“地面完全变成了一整块灰色”，且“原来的随着运行车辆转我地面会乱，现在变成一个颜色一个分类了”。

### 核心原因分析
经过深度分析，问题根源由两个致命错误共同导致：

#### A. 坐标系旋转丢失 (XY平面问题)
*   **现象**: 地面不再闪烁，但方向固定，不随车头转动。
*   **原因**: 为了解决闪烁采用了 **World-Aligned Sampling**，但在写入 Grid 时直接使用了 `wx - ego.x` 作为 Grid 坐标，**丢失了 World -> Ego 的旋转变换**。
*   **后果**: Occupancy Grid 变成了世界朝向（North-East），而 Viewer 期望是自车朝向（Forward-Left），导致显示错乱或大部分区域无数据。

#### B. Z 坐标基准错误 (高度问题 - 关键!)
*   **现象**: 道路 (11) 和人行道 (13) 消失，只剩下基础填充的 other_flat (12)。
*   **原因**: Z 轴坐标计算使用了错误的基准。
    *   ❌ **错误代码**: `dz_world = wz - ground_z_world` (相对于路面高度)。
    *   这导致 `gz` ≈ 0 (Ego 车体中心高度)，比实际路面位置（Ego 下方约 1m）高了约 1 米。
    *   实际路面被填充到了 Z 索引 20-21 层 (Z=0m)，而基础填充 (12) 只覆盖 Z < 0.2m 的区域。
    *   虽然范围有重叠，但由于高度计算的系统性偏差，导致覆盖逻辑失效或覆盖在错误位置。
*   **正确逻辑**: 应该计算相对于 **Ego 车体中心** 的高度。
    *   ✅ **正确代码**: `dz_ego = wz - ego_location.z`。

## 2. 综合修复方案：矩阵投影法

我们需要一个方案同时解决 **旋转问题** 和 **Z 轴基准问题**。
最佳方案是使用 **World -> Ego 的逆变换矩阵**。

### 为什么矩阵变换是完美解？
1.  **解决旋转**: 逆矩阵包含了 Yaw 角旋转，能将世界坐标 `(wx, wy)` 正确变换为 Ego 坐标 `(forward, left)`。
2.  **解决 Z 基准**: 逆矩阵将世界原点变换为 Ego 原点。
    *   对于点 `P_world(wx, wy, wz)`，变换后的 `P_ego = M_inv @ P_world`。
    *   `P_ego.z` 自动就是点 P 相对于 Ego 坐标系原点（车体中心）的垂直高度。
    *   这等价于 `wz - ego_location.z` (在平地近似情况下)，且处理了 Pitch/Roll。

### 具体步骤

1.  **获取完整变换矩阵**:
    ```python
    world_to_ego = np.array(ego_transform.get_inverse_matrix())
    ```
    
2.  **世界坐标采样**:
    *   继续使用 `x_world_samples` 循环，确保采样点锁定在绝对地理位置，解决**闪烁问题**。
    
3.  **矩阵投影 (World -> Ego)**:
    *   对于每个采样点 `P_world (wx, wy, wz)`:
    *   `P_ego = world_to_ego @ P_world`
    *   `gx, gy, gz = P_ego[0], P_ego[1], P_ego[2]`
    *   这里 `gz` 将是正确的负值（例如 -1.5m），对应 Ego 下方的路面。

4.  **Grid 索引计算**:
    *   `ix = (gx - x_range[0]) / resolution`
    *   `z_idx = (gz - z_range[0]) / resolution`

## 3. 代码修改验证

修改 `d:\code\carla\dense_occupancy_collection\processing\ground_truth_voxel_generator.py`。

### 核心修改代码
```python
# 1. 预计算变换矩阵
world_to_ego = np.array(ego_transform.get_inverse_matrix())

# ... 在循环中 ...

# 2. 构建齐次坐标并变换
p_world = np.array([wx, wy, wz, 1.0])
p_ego = world_to_ego @ p_world  # 同时解决旋转和 Z 轴偏移

gx = p_ego[0]
gy = p_ego[1]
gz = p_ego[2]  # ✅ 正确的相对高度 (例如 -1.5m)
```

### 预期效果
*   **地面分类**: 恢复 11 (道路), 13 (人行道), 12 (其他) 的正确分布。
*   **稳定性**: 地面纹理不随车辆移动而闪烁。
*   **方向性**: 地面纹理随车头转动而正确旋转。

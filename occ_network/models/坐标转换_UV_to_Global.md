# 坐标系转换详解：从 Pixel UV 到 车辆全天球 XYZ

本文档详细记录了 `RayDirectionEncoding` 模块中，如何将一张 2D 图片上的像素坐标 $(u, v)$ 一步步转换成 3D 车辆坐标系下的单位方向向量 $(x_v, y_v, z_v)$。

我们使用 **等距投影 (Equidistant Projection)** 模型，这是鱼眼相机和广角相机常用的通用模型。

---

## 核心流程概览

整个转换过程分为四个步骤：

1.  **像素坐标系 $\to$ 图像平面坐标系**: $(u, v) \to (dx, dy)$
2.  **图像平面 $\to$ 球面极坐标 (Equidistant)**: $(dx, dy) \to (r, \phi) \to (\theta, \phi)$
3.  **球面极坐标 $\to$ 相机笛卡尔坐标**: $(\theta, \phi) \to (x_c, y_c, z_c)$
4.  **相机坐标 $\to$ 车辆/世界坐标**: $(x_c, y_c, z_c) \to (x_v, y_v, z_v)$

---

## 详细演算步骤

### 0. 预设参数 (Example Setup)

为了方便演算，我们设定一组具体的参数：

*   **图像尺寸**: $W=800, H=600$
*   **相机视场角 (FOV)**: $120^\circ$ (广角)
*   **相机安装角度**: Pitch=0, Roll=0, Yaw=$45^\circ$ (右前方朝向)
*   **目标像素点**: $u=600, v=450$ (位于图像右下区域)

### 第一步：像素坐标 $\to$ 图像平面坐标 (Pixel to Image Plane)

我们需要将以左上角为原点的像素坐标 $(u, v)$，转换为以图像中心为原点的物理坐标 $(dx, dy)$。

*   **光心坐标**:
    $$ c_x = W / 2 = 400 $$
    $$ c_y = H / 2 = 300 $$

*   **计算偏移量**:
    $$ dx = u - c_x = 600 - 400 = 200 $$
    $$ dy = v - c_y = 450 - 300 = 150 $$

    > 注意：这里 $dy$ 为正表示向下（因为图像坐标系通常 $y$ 轴向下）。

### 第二步：图像平面 $\to$ 入射角 $\theta$ (The Equidistant Projection)

这是最关键的一步。我们需要计算该像素距离光心有多远 ($r$)，以及它对应的入射角 ($\theta$)。

1.  **计算半径 $r$ (像素距离)**:
    $$ r = \sqrt{dx^2 + dy^2} = \sqrt{200^2 + 150^2} = \sqrt{40000 + 22500} = \sqrt{62500} = 250 $$

2.  **计算方位角 $\phi$ (Image Phi)**:
    $$ \phi = \operatorname{atan2}(dy, dx) = \operatorname{atan2}(150, 200) \approx 0.6435 \text{ rad} \approx 36.87^\circ $$

3.  **计算焦距 $f$ (Scale Factor)**:
    在等距投影模型 ($r = f \cdot \theta$) 中，我们需要根据 FOV 确定 $f$。
    假设 FOV 覆盖图像的宽度 $W$ (即当 $\theta = \text{FOV}/2$ 时，$r = W/2$)：
    $$ \text{FOV}_{\text{rad}} = 120^\circ \times \frac{\pi}{180} = \frac{2\pi}{3} \approx 2.0944 $$
    $$ f = \frac{W/2}{\text{FOV}_{\text{rad}}/2} = \frac{W}{\text{FOV}_{\text{rad}}} = \frac{800}{2.0944} \approx 381.97 $$

4.  **计算入射角 $\theta$ (Theta)**:
    $$ \theta = \frac{r}{f} = \frac{250}{381.97} \approx 0.6545 \text{ rad} \approx 37.5^\circ $$

    > **物理含义**: 这条光线与相机光轴（Z轴）的夹角是 37.5 度。

### 第三步：球面坐标 $\to$ 相机笛卡尔坐标 (Spherical to Camera XYZ)

现在我们将 $(\theta, \phi)$ 转换为相机坐标系下的单位向量 $(x_c, y_c, z_c)$。

**相机坐标系定义**:
*   **Z轴**: 光轴，指向前方。
*   **X轴**: 指向右侧。
*   **Y轴**: 指向下方。

**转换公式**:
*   光轴分量 (Forward): $z_c = \cos(\theta)$
*   径向分量 (Radial): $\sin(\theta)$
*   水平分量 (Right): $x_c = \sin(\theta) \cdot \cos(\phi)$
*   垂直分量 (Down): $y_c = \sin(\theta) \cdot \sin(\phi)$

**代入计算**:
1.  $\sin(\theta) = \sin(0.6545) \approx 0.6088$
2.  $\cos(\theta) = \cos(0.6545) \approx 0.7934$
3.  $\cos(\phi) = \cos(0.6435) = 0.8$ ($200/250$)
4.  $\sin(\phi) = \sin(0.6435) = 0.6$ ($150/250$)

**结果**:
$$ z_c = 0.7934 $$
$$ x_c = 0.6088 \times 0.8 = 0.4870 $$
$$ y_c = 0.6088 \times 0.6 = 0.3653 $$

**验证归一化**:
$$ 0.4870^2 + 0.3653^2 + 0.7934^2 \approx 0.237 + 0.133 + 0.629 = 0.999 \approx 1.0 $$

此时，我们得到了相对于**该相机**的方向向量：
$$ \vec{v}_{cam} = [0.4870, 0.3653, 0.7934] $$

### 第四步：相机坐标 $\to$ 车辆/世界坐标 (Camera to Vehicle XYZ)

最后，我们需要根据相机的安装角度，将其旋转到车辆坐标系。

**车辆坐标系定义**:
*   **X轴**: 车辆前方。
*   **Y轴**: 车辆右方。
*   **Z轴**: 车辆上方。
    *(注意：不同数据集定义不同，这里假设 CARLA/常规 自动驾驶定义，但代码中的 `_rotation_matrix` 实际上执行的是标准的 3D 旋转。我们需要确认 `_rotation_matrix` 的输出含义。)*

**代码逻辑分析 (`_rotation_matrix`)**:
代码执行的是 $R = R_z(\text{yaw}) \cdot R_y(\text{roll}) \cdot R_x(\text{pitch})$。
通常这意味着：
1.  先绕 X 轴转 (Pitch)
2.  再绕 Y 轴转 (Roll)
3.  最后绕 Z 轴转 (Yaw)

假设我们的相机是：**Yaw = 45°** (向右偏 45 度)，Pitch=0, Roll=0。

**旋转矩阵 $R_z(45^\circ)$**:
$$
R_z = \begin{bmatrix}
\cos(45^\circ) & -\sin(45^\circ) & 0 \\
\sin(45^\circ) & \cos(45^\circ) & 0 \\
0 & 0 & 1
\end{bmatrix}
= \begin{bmatrix}
0.707 & -0.707 & 0 \\
0.707 & 0.707 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

**坐标变换**:
$$ \vec{v}_{world} = R \cdot \vec{v}_{cam} $$

$$
\begin{bmatrix} x_v \\ y_v \\ z_v \end{bmatrix} =
\begin{bmatrix}
0.707 & -0.707 & 0 \\
0.707 & 0.707 & 0 \\
0 & 0 & 1
\end{bmatrix}
\cdot
\begin{bmatrix} 0.4870 \\ 0.3653 \\ 0.7934 \end{bmatrix}
$$

*   $x_v = 0.707 \times 0.4870 - 0.707 \times 0.3653 = 0.344 - 0.258 = 0.086$
*   $y_v = 0.707 \times 0.4870 + 0.707 \times 0.3653 = 0.344 + 0.258 = 0.602$
*   $z_v = 1 \times 0.7934 = 0.7934$

**结果解释**:
*   原始相机向量指向“右下方”。
*   相机本身安装方向是“向右前方 45 度”。
*   叠加后，该光线指向车辆的“右方更偏右”的位置。

---

## 总结公式表

| 步骤 | 变量 | 公式 | 说明 |
| :--- | :--- | :--- | :--- |
| **1. 归一化平面** | $dx, dy$ | $u - W/2, v - H/2$ | 移到图像中心 |
| **2. 极坐标半径** | $r$ | $\sqrt{dx^2 + dy^2}$ | 像素距离 |
| **3. 图像方位角** | $\phi$ | $\operatorname{atan2}(dy, dx)$ | 2D 平面角度 |
| **4. 焦距系数** | $f$ | $W / \text{FOV}_{\text{rad}}$ | 将像素转为弧度 |
| **5. 空间入射角** | $\theta$ | $r / f$ | **等距投影核心公式** |
| **6. 相机坐标** | $\vec{v}_{cam}$ | $[\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta]$ | $Z$为光轴 |
| **7. 世界坐标** | $\vec{v}_{world}$ | $R_{\text{yaw}} R_{\text{roll}} R_{\text{pitch}} \cdot \vec{v}_{cam}$ | 刚体旋转 |

## 代码对应关系

在 `d:\code\carla\occ_network\models\position_encoding.py` 中：

```

---

## 附录：Python 代码实现 (7步封装)

以下是严格对应上述 7 个步骤的 Python 代码封装，可以直接运行以验证逻辑。

```python
import math

class RayCaster:
    """
    分步实现的射线投射器，严格对应文档中的7个步骤。
    """
    
    @staticmethod
    def step1_pixel_to_image_plane(u: float, v: float, W: int, H: int):
        """
        步骤 1: 像素坐标系 -> 图像平面坐标系
        将左上角原点转换为图像中心原点。
        
        Args:
            u: 像素横坐标 (0 ~ W)
            v: 像素纵坐标 (0 ~ H)
            W: 图像宽度
            H: 图像高度
            
        Returns:
            dx, dy: 相对于图像中心的物理距离
        """
        cx = W / 2.0
        cy = H / 2.0
        dx = u - cx
        dy = v - cy
        return dx, dy

    @staticmethod
    def step2_compute_radius(dx: float, dy: float):
        """
        步骤 2: 计算极坐标半径 r (像素距离)
        
        Args:
            dx: x轴偏移
            dy: y轴偏移
            
        Returns:
            r: 像素距离
        """
        return math.sqrt(dx**2 + dy**2)

    @staticmethod
    def step3_compute_image_phi(dx: float, dy: float):
        """
        步骤 3: 计算图像方位角 phi
        
        Args:
            dx: x轴偏移
            dy: y轴偏移
            
        Returns:
            phi: 图像平面上的方位角 (radians)
        """
        return math.atan2(dy, dx)

    @staticmethod
    def step4_compute_focal_length(W: int, fov_deg: float):
        """
        步骤 4: 计算焦距系数 f
        根据等距投影模型: r = f * theta
        当 theta = FOV/2 时, r = W/2
        
        Args:
            W: 图像宽度
            fov_deg: 视场角 (度)
            
        Returns:
            f: 等效焦距 (pixels/radian)
        """
        fov_rad = math.radians(fov_deg)
        # f = r / theta = (W/2) / (fov/2) = W / fov
        f = (W / 2.0) / (fov_rad / 2.0)
        return f

    @staticmethod
    def step5_compute_theta(r: float, f: float):
        """
        步骤 5: 计算空间入射角 theta (等距投影核心)
        theta = r / f
        
        Args:
            r: 像素半径
            f: 焦距
            
        Returns:
            theta: 入射角 (radians)
        """
        return r / f

    @staticmethod
    def step6_spherical_to_camera(theta: float, phi: float):
        """
        步骤 6: 球面坐标 -> 相机笛卡尔坐标 (Camera XYZ)
        
        定义:
            Z轴: 光轴方向 (Forward)
            X轴: 右侧 (Right)
            Y轴: 下方 (Down)
            
        Args:
            theta: 入射角
            phi: 方位角
            
        Returns:
            (x_c, y_c, z_c): 相机坐标系下的单位向量
        """
        # Z轴是光轴方向 (theta=0时, z=1)
        z_c = math.cos(theta)
        
        # 投影到XY平面的分量
        sin_theta = math.sin(theta)
        
        # X: Right, Y: Down
        x_c = sin_theta * math.cos(phi)
        y_c = sin_theta * math.sin(phi)
        
        return x_c, y_c, z_c

    @staticmethod
    def step7_camera_to_world(v_cam: tuple, rotation_deg: list):
        """
        步骤 7: 相机坐标 -> 世界/车辆坐标
        
        执行 Rz(yaw) * Ry(roll) * Rx(pitch) 旋转。
        
        Args:
            v_cam: (x_c, y_c, z_c) 相机坐标向量
            rotation_deg: [pitch, roll, yaw] 角度制
            
        Returns:
            (x_v, y_v, z_v): 世界坐标系下的单位向量
        """
        x_c, y_c, z_c = v_cam
        pitch, roll, yaw = [math.radians(x) for x in rotation_deg]

        # 1. Rx (Pitch) - 绕X轴旋转
        # [1, 0,       0]
        # [0, cos,  -sin]
        # [0, sin,   cos]
        x1 = x_c
        y1 = y_c * math.cos(pitch) - z_c * math.sin(pitch)
        z1 = y_c * math.sin(pitch) + z_c * math.cos(pitch)

        # 2. Ry (Roll) - 绕Y轴旋转
        # [ cos, 0, sin]
        # [   0, 1,   0]
        # [-sin, 0, cos]
        x2 = x1 * math.cos(roll) + z1 * math.sin(roll)
        y2 = y1
        z2 = -x1 * math.sin(roll) + z1 * math.cos(roll)

        # 3. Rz (Yaw) - 绕Z轴旋转
        # [cos, -sin, 0]
        # [sin,  cos, 0]
        # [  0,    0, 1]
        x_v = x2 * math.cos(yaw) - y2 * math.sin(yaw)
        y_v = x2 * math.sin(yaw) + y2 * math.cos(yaw)
        z_v = z2

        return x_v, y_v, z_v

# ==========================================
# 示例运行脚本
# ==========================================
def run_example():
    print("=== 开始运行 7步坐标转换示例 ===")
    
    # 参数设置 (与文档中的 Example Setup 一致)
    W, H = 800, 600
    u, v = 600, 450
    fov = 120
    rotation = [0, 0, 45] # pitch, roll, yaw

    rc = RayCaster()

    # 1. 像素 -> 平面
    dx, dy = rc.step1_pixel_to_image_plane(u, v, W, H)
    print(f"1. Image Plane Offset: dx={dx}, dy={dy}")
    
    # 2. 半径
    r = rc.step2_compute_radius(dx, dy)
    print(f"2. Radius r: {r:.4f}")
    
    # 3. 方位角
    phi = rc.step3_compute_image_phi(dx, dy)
    print(f"3. Image Phi: {phi:.4f} rad ({math.degrees(phi):.2f} deg)")
    
    # 4. 焦距
    f = rc.step4_compute_focal_length(W, fov)
    print(f"4. Focal Length f: {f:.4f}")
    
    # 5. 入射角
    theta = rc.step5_compute_theta(r, f)
    print(f"5. Incidence Angle theta: {theta:.4f} rad ({math.degrees(theta):.2f} deg)")
    
    # 6. 相机坐标
    v_cam = rc.step6_spherical_to_camera(theta, phi)
    print(f"6. Camera Vector: [{v_cam[0]:.4f}, {v_cam[1]:.4f}, {v_cam[2]:.4f}]")
    
    # 7. 世界坐标
    v_world = rc.step7_camera_to_world(v_cam, rotation)
    print(f"7. World Vector:  [{v_world[0]:.4f}, {v_world[1]:.4f}, {v_world[2]:.4f}]")

if __name__ == "__main__":
    run_example()
```python
# 步骤 1 & 2
dx = uu - cx
dy = vv - cy
r = torch.sqrt(dx**2 + dy**2)
phi_img = torch.atan2(dy, dx)

# 步骤 4 & 5
f = W / fov_rad
theta = r / f

# 步骤 6
ray_z = torch.cos(theta)
sin_theta = torch.sin(theta)
ray_x = sin_theta * torch.cos(phi_img)
ray_y = sin_theta * torch.sin(phi_img)
rays_cam = torch.stack([ray_x, ray_y, ray_z], dim=-1)

# 步骤 7
R = self._rotation_matrix(rotation)
rays_world = torch.einsum('ij,hwj->hwi', R, rays_cam)
```

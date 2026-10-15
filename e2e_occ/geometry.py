"""
几何工具：坐标系约定、等距投影正/反投影、体素中心。

全网（射线编码、可变形注意力参考点、时序对齐）和数据加载（深度监督目标）都调用这里，
保证正投影和反投影是同一份公式，不会再出现"两处各写一套、约定不一致"的问题。

坐标系约定（2026-10 统一）：
- 车辆坐标系 / 体素坐标系：CARLA 原生，X 前、Y 右、Z 上（左手系），单位米。
  体素网格按 [X, Y, Z] 索引排列，体素中心 = vmin + (i + 0.5) * cell。
- 相机坐标系：OpenCV 约定，x 右、y 下、z 前（光轴）。网络内所有投影公式都在这个约定下。
- 外参：网络吃的是 Camera(OpenCV)→Vehicle 的 4×4 矩阵（静态安装标定），
  **不含**车辆在世界中的绝对位姿；车辆位姿只用来算帧间 ego_motion。
- 像素坐标：像素中心在整数坐标（OpenCV 约定），与 CARLA WideAngleLens.usf 一致
  （shader 用整数像素索引减主点）。
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 数值精度：几何计算一律 FP32
# ----------------------------------------------------------------------------
# 训练/推理都在 torch.autocast(bf16) 里跑，autocast 会把 einsum/matmul 降成 BF16（8 位尾数）。
# 2026-10-15 实测：细网格投影到原图的坐标误差 中位数 0.97px、P95 3.45px、最大 7.6px，18.7 万个可见点
# 里 247 个可见性判断翻转 (FP16 也有 0.12px)；grid_sample 的归一化坐标若跟着特征转成 BF16，量化步长
# 2^-8 在 stride-8 层级上约 2.5 个原图像素。所以：坐标变换、投影、射线、Fourier 位置编码、采样网格
# 全部在 no_autocast() 里用 FP32 算，采样用 grid_sample_fp32()；卷积/注意力/FFN 照常混合精度。

def no_autocast(device):
    """关闭 autocast 的上下文 (该区域内运算按输入 dtype 执行，调用方负责把输入转成 FP32)。"""
    return torch.autocast(device_type=device.type, enabled=False)


def grid_sample_fp32(inp, grid, **kwargs):
    """F.grid_sample 的 FP32 版本：特征和采样坐标都转 FP32 再采样，返回 FP32。"""
    with no_autocast(inp.device):
        return F.grid_sample(inp.float(), grid.float(), **kwargs)


# UE 相机轴（x 前、y 右、z 上）下表示的 OpenCV 相机轴：
#   cv_x(右) = ue_y,  cv_y(下) = -ue_z,  cv_z(前) = ue_x
# 即 p_ue = M_CV_TO_UE @ p_cv。推导依据 WideAngleLens.usf：像素向右 → +Y，向下 → -Z，光轴 → +X。
# det = -1：它把右手系的 OpenCV 相机系映射到 CARLA 左手系，这是正确的，不要"修正"成纯旋转。
M_CV_TO_UE = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)


def carla_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    """与 UE FRotationMatrix 完全一致的旋转矩阵（UE 局部轴 → 父坐标系），即引擎渲染时的朝向。

    UE 的 pitch 正值 = 抬头（局部 +X 指向上方）。注意当前 LibCarla 客户端的
    carla.Transform.get_matrix() 经上游 #9751 改过 pitch/roll 符号，与这里（和 UE 渲染）相反，
    不要拿它当标准。
    """
    p, y, r = (math.radians(a) for a in (pitch_deg, yaw_deg, roll_deg))
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ], dtype=np.float64)


def legacy_rotation_to_carla(R_legacy):
    """把旧版 camera_manager.get_extrinsics（2026-10 之前）写出的旋转矩阵转换为 CARLA 正确公式。

    旧公式第三行是 (-sp, cp*sr, cp*cr)，与 UE 渲染的 (sp, -cp*sr, cp*cr) 在 pitch/roll
    非零时符号相反（只影响 rear 相机 pitch=8°）。这里先按旧公式反解出 (pitch, yaw, roll)，
    再用 CARLA 公式重建。
    """
    R = np.asarray(R_legacy, dtype=np.float64)
    pitch = math.degrees(math.asin(float(np.clip(-R[2, 0], -1.0, 1.0))))
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    return carla_rotation_matrix(pitch, yaw, roll)


def cam_ue_to_vehicle_to_cv(T_ue):
    """Camera(UE 轴)→Vehicle 4×4 → Camera(OpenCV 轴)→Vehicle 4×4。"""
    T = np.array(T_ue, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3] @ M_CV_TO_UE
    out[:3, 3] = T[:3, 3]
    return out


def cam_to_vehicle_from_mount(position, rotation_pyr):
    """由安装参数 (x,y,z) 米、(pitch,yaw,roll) 度 构建 Camera(OpenCV)→Vehicle。"""
    T_ue = np.eye(4, dtype=np.float64)
    T_ue[:3, :3] = carla_rotation_matrix(*rotation_pyr)
    T_ue[:3, 3] = position
    return cam_ue_to_vehicle_to_cv(T_ue)


# ----------------------------------------------------------------------------
# 等距投影 r = f·θ（与 CARLA CameraModelUtil.cpp Equidistant 分支一致）
# ----------------------------------------------------------------------------

def equidistant_unproject(u, v, fx, fy, cx, cy):
    """像素坐标（原图分辨率）→ 相机系(OpenCV)单位方向。

    fx/fy 允许不同（各向异性）：先按各自焦距归一化，θ = |归一化坐标|。
    u, v: 任意形状张量；fx..cy 需可与之广播。返回 [..., 3]。
    """
    nx = (u - cx) / fx
    ny = (v - cy) / fy
    theta = torch.sqrt(nx * nx + ny * ny)
    phi = torch.atan2(ny, nx)
    s = torch.sin(theta)
    return torch.stack([s * torch.cos(phi), s * torch.sin(phi), torch.cos(theta)], dim=-1)


def equidistant_project(p_cam, fx, fy, cx, cy):
    """相机系(OpenCV)点 [..., 3] → 像素坐标 (u, v)（原图分辨率）和入射角 θ。"""
    x, y, z = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
    # 用 atan2(ρ, z) 而不是 acos(z/|P|)：光轴附近 cosθ≈1，float32 下 acos 的角度分辨率只有
    # ~3.5e-4 rad，对窄视场相机（~2100 px/rad）相当于 ~0.7 像素误差；atan2 在全范围数值稳定
    theta = torch.atan2(torch.sqrt(x * x + y * y), z)
    phi = torch.atan2(y, x)
    u = cx + fx * theta * torch.cos(phi)
    v = cy + fy * theta * torch.sin(phi)
    return u, v, theta


def level_pixel_to_image(i, stride):
    """特征层级像素索引 → 原图像素坐标（像素中心在整数坐标约定）。

    stride=s 的特征像素 i 覆盖原图像素 [i·s, (i+1)·s-1]，中心为 (i+0.5)·s - 0.5。
    """
    return (i + 0.5) * stride - 0.5


def image_to_grid_sample(u, v, image_hw):
    """原图像素坐标 → grid_sample(align_corners=False) 的归一化坐标。

    对任意层级特征图都成立（层级无关），因为 align_corners=False 下归一化坐标描述的是
    图像物理范围的相对位置。
    """
    H, W = image_hw
    return 2.0 * (u + 0.5) / W - 1.0, 2.0 * (v + 0.5) / H - 1.0


# ----------------------------------------------------------------------------
# 体素
# ----------------------------------------------------------------------------

def voxel_centers(size, voxel_range, device=None):
    """返回 [X*Y*Z, 3] 体素中心（米），按 [X, Y, Z] 行优先展开（与 view(B,X,Y,Z,C) 一致）。"""
    X, Y, Z = size
    xmin, ymin, zmin, xmax, ymax, zmax = voxel_range
    xs = xmin + (torch.arange(X, device=device, dtype=torch.float32) + 0.5) * (xmax - xmin) / X
    ys = ymin + (torch.arange(Y, device=device, dtype=torch.float32) + 0.5) * (ymax - ymin) / Y
    zs = zmin + (torch.arange(Z, device=device, dtype=torch.float32) + 0.5) * (zmax - zmin) / Z
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
    return torch.stack([gx, gy, gz], dim=-1).view(-1, 3)


def metric_to_grid_sample(points, voxel_range):
    """米制点 [..., 3] (x,y,z) → 3D grid_sample(align_corners=False) 归一化坐标 (x,y,z)。

    配合体积排列 [B, C, Z, Y, X] 使用：grid 的第 0/1/2 分量分别对应 W=X、H=Y、D=Z。
    """
    vr = torch.as_tensor(voxel_range, dtype=points.dtype, device=points.device)
    vmin, vmax = vr[:3], vr[3:]
    return 2.0 * (points - vmin) / (vmax - vmin) - 1.0

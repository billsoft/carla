"""
CARLA Transform → 4×4 矩阵 / 包围盒顶点，按 UE FRotationMatrix 约定（pitch 正值 = 抬头）计算。

为什么不直接用客户端的 carla.Transform.get_matrix() / BoundingBox.get_world_vertices()：
上游 #9751（2026-06，已合入本仓库 ue5-dev 并编进当前 Python API）翻转了 LibCarla 客户端
Rotation::RotateVector / Transform::GetMatrix 的 pitch/roll 符号，但 Rotation→FRotator 是原样
传递、UE 端没有跟着改——客户端算出的矩阵在 pitch/roll≠0 时与引擎实际渲染/物理不一致
（例如坡道上自车的体素网格朝向错 2θ，40m 处竖直偏差可达数米）。上游 ue58-dev 已在
7c32b194a 撤销了这次翻转；这里的实现与 UE 一致，因此在 5.5 和 5.8 客户端下结果相同。

与 e2e_occ/geometry.py::carla_rotation_matrix、sensors/camera_manager.py::get_extrinsics
是同一个公式，三处必须保持一致。
"""

import math

import numpy as np


def rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    """UE FRotationMatrix：局部轴 (x前, y右, z上) → 父坐标系。"""
    p, y, r = (math.radians(a) for a in (pitch_deg, yaw_deg, roll_deg))
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ], dtype=np.float64)


def transform_to_matrix(transform):
    """carla.Transform → 4×4（局部 → 父/世界），替代 transform.get_matrix()。"""
    rot = transform.rotation
    loc = transform.location
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rotation_matrix(rot.pitch, rot.yaw, rot.roll)
    m[:3, 3] = (loc.x, loc.y, loc.z)
    return m


_CORNER_SIGNS = np.array([
    [sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
], dtype=np.float64)


def bbox_world_vertices(bb, actor_transform=None):
    """carla.BoundingBox 的 8 个世界坐标顶点，[8, 3] numpy，替代 bb.get_world_vertices()。

    与 LibCarla BoundingBox::GetWorldVertices 相同的组合方式：
    顶点 = T_actor · (bb.location + R_bb · (±extent))；actor_transform 为 None 时视为单位变换
    （level/环境物体的 bounding_box 本身已在世界坐标）。
    """
    ext = np.array([bb.extent.x, bb.extent.y, bb.extent.z], dtype=np.float64)
    r_bb = rotation_matrix(bb.rotation.pitch, bb.rotation.yaw, bb.rotation.roll)
    local = (_CORNER_SIGNS * ext) @ r_bb.T + np.array([bb.location.x, bb.location.y, bb.location.z])
    if actor_transform is None:
        return local
    m = transform_to_matrix(actor_transform)
    return local @ m[:3, :3].T + m[:3, 3]

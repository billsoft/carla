"""
相机可见性掩码 mask_camera (Occ3D 风格)：标出 8 路相机实际"看到过"的体素。

为什么需要：体素真值是从地图/包围盒生成的"全知"结果，LiDAR 可见性过滤 (visibility_filter_simple.py)
之后仍有大量体素相机根本看不到——地面往下填充的整列、建筑/车辆内部、被前车挡住的区域。
200 帧审计里只有约 34% 的占用体素能被相机看到。拿这些体素监督一个只吃图像的网络，等于逼它
从照片里猜看不见的东西。训练时应把"相机没观测到"的体素设为 ignore (255)，只在观测到的
空间上算损失/指标；occupancy/*.npy 本身不改 (可视化和旧工具照常用)。

做法 (只用与鱼眼同位姿的针孔深度图 + 标定，采集时和离线补算走同一个函数)：
  (a) 表面命中：把每个深度像素反投影成 3D 点，落在哪个体素就标记为观测到；同时标记沿视线
      往后半个体素、以及往下半个体素处的体素——表面点常常正好落在体素边界上，被标成占用的
      往往是它后面/下面那一格。只看体素中心会在掠射角下系统性漏掉远处地面：中心比路面低
      ~0.1m，相机高 1.7m 时 20m 外视线要再往前 ~1.2m 才到那个中心的正上方，远超任何容差。
  (b) 自由空间：体素中心投影进某个深度相机画面，且沿光轴的距离 <= 该像素深度 + free_tol
      (视线在打到表面之前穿过了它) → 观测到的空的空间。
  mask = (a) ∪ (b)。体素中心在各相机里的投影只和静态标定有关，构造时预计算一次，每帧只做查表。

自由空间雕刻 carve_static()：静态环境物体 (树/栅栏/杆/交通灯/道具) 在真值生成器里是按世界 AABB
实心填充的——一棵树是一个实心长方体、斜放的栅栏变成一大块、带横臂的交通灯盒子悬在马路上。
2 帧审计：植被 74%、护栏 67% 的占用体素被相机视线明确穿过 (照片里是空的)。雕刻把这些
"视线明确穿过"的体素改成 free。判据比 (b) 严格得多，避免把真实表面误删：
  深度图先做 3×3 最小值腐蚀 (物体边缘像素不算穿过)，且体素中心要在表面前
  carve_margin + carve_rel×距离 以上 (默认 0.3m + 2%)。
只雕刻静态类别；车辆/行人不雕 (深度相机透过车窗看进车内，车按实心盒标注是常规做法)；
manmade 只雕体积小的物体 (杆/交通灯/路牌)，大楼不雕——深度同样透过玻璃幕墙，雕了立面会千疮百孔。
任何相机拍到表面的体素也不雕 (半透明的树叶/铁丝网)。

地面语义重绘 paint_ground()：地面类别 (路面/人行道/地形) 原本靠 Map API 推断 (车道中心线 + 车道宽度)，
逐像素核对发现标成 terrain 的体素 84% 在 UE 渲染里其实是路面 (路肩/停车区/路口不在 Driving 车道里)，
标成 sidewalk 的 35% 也是路面。这里用与深度相机同位姿的 UE 语义分割相机：地面类像素按深度反投影，
在所在列里投票，票数最多的地面类别覆盖该列的全部地面体素 (表面 + 向下填充)。相机看不到的列保持
Map API 结果 (训练时多半被掩码忽略)。

finalize_ground_truth() 把 重绘 → 掩码 → 雕刻 串起来，main_collection.py 和 audit_labels.py 共用。

深度相机是针孔 sensor.camera.depth (平面 z 深度)，水平 FOV = intrinsics.json 的 depth_fov
(新数据，覆盖整幅鱼眼) 或 fov (旧数据，鱼眼四角没有覆盖)。
"""
import json
import math
from pathlib import Path

import numpy as np

from config.occupancy_config import CARLA_TO_OCCUPANCY_MAPPING, GROUND_LABELS

# UE 语义分割标签 (= CityObjectLabel 编号) → 地面类 (只取映射到 GROUND_LABELS 的那些)
GROUND_TAG_TO_CLASS = {tag: c for tag, c in CARLA_TO_OCCUPANCY_MAPPING.items() if c in GROUND_LABELS}


def save_mask(path, mask):
    """(X,Y,Z) bool → 沿 Z 打包成 (X,Y,Z/8) uint8 存 .npy (400×400×32 从 5MB 降到 640KB)。"""
    np.save(path, np.packbits(np.asarray(mask, bool), axis=2))


def load_mask(path, z_size=32):
    """读 mask_camera：兼容打包格式 (X,Y,Z/8) 和未打包的 (X,Y,Z) uint8/bool。"""
    m = np.load(path)
    if m.shape[2] != z_size:
        m = np.unpackbits(m, axis=2, count=z_size)
    return m.astype(bool)


class CameraVisibility:
    def __init__(self, cameras, x_range, y_range, z_range, resolution,
                 free_tol=None, max_hit_depth=200.0):
        """
        Args:
            cameras: [{'R': (3,3) 相机(UE 轴 x前 y右 z上)→车辆 旋转, 't': (3,) 平移,
                       'hfov': 深度相机水平 FOV(度), 'width', 'height'}]，顺序即 depth 图的顺序
            free_tol: 自由空间判据的容差 (米)，默认半个体素
            max_hit_depth: 超过这个平面深度的像素 (天空 ~1000m) 不做表面命中
        """
        self.cameras = cameras
        self.x0, self.y0, self.z0 = float(x_range[0]), float(y_range[0]), float(z_range[0])
        self.res = float(resolution)
        self.shape = (int(round((x_range[1] - x_range[0]) / resolution)),
                      int(round((y_range[1] - y_range[0]) / resolution)),
                      int(round((z_range[1] - z_range[0]) / resolution)))
        self.free_tol = self.res * 0.5 if free_tol is None else float(free_tol)
        self.max_hit_depth = float(max_hit_depth)
        self._proj = [self._precompute(c) for c in cameras]
        self._rays = [self._pixel_rays(c) for c in cameras]

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_calibration(cls, calib_dir, x_range, y_range, z_range, resolution, **kw):
        """从数据集 calibration/{intrinsics,extrinsics}.json 构造 (采集时和离线补算共用)。"""
        calib_dir = Path(calib_dir)
        intr = json.load(open(calib_dir / 'intrinsics.json', encoding='utf-8'))
        extr = json.load(open(calib_dir / 'extrinsics.json', encoding='utf-8'))
        if extr.get('rotation_convention') != 'ue_frotator':
            # 旧数据 pitch/roll 符号与渲染相反 (见 e2e_occ/dataset.py 的修正逻辑)；这里不重复实现，
            # 直接拒绝，避免算出一份和图像不对应的掩码
            raise ValueError(f'{calib_dir}: extrinsics.json 缺少 rotation_convention=ue_frotator，'
                             f'旧版标定不支持计算相机可见性掩码')
        cams = []
        i = 0
        while f'cam_{i}' in intr:
            ci, ce = intr[f'cam_{i}'], extr[f'cam_{i}']
            cams.append({'R': np.asarray(ce['rotation_matrix'], np.float64),
                         't': np.asarray(ce['translation'], np.float64),
                         'hfov': float(ci.get('depth_fov', ci['fov'])),
                         'width': int(ci.get('width', 1280)), 'height': int(ci.get('height', 960))})
            i += 1
        return cls(cams, x_range, y_range, z_range, resolution, **kw)

    def _focal(self, cam):
        return (cam['width'] / 2.0) / math.tan(math.radians(cam['hfov']) / 2.0)

    def _precompute(self, cam):
        """所有体素中心在该相机里的 (像素下标, 光轴距离)，只保留落在画面内的。"""
        W, H = cam['width'], cam['height']
        f = self._focal(cam)
        X, Y, Z = self.shape
        R_T = cam['R'].T                       # 车辆 → 相机
        t = cam['t']
        vidx, pidx, depth = [], [], []
        xs = self.x0 + (np.arange(X, dtype=np.float32) + 0.5) * self.res
        ys = self.y0 + (np.arange(Y, dtype=np.float32) + 0.5) * self.res
        zs = self.z0 + (np.arange(Z, dtype=np.float32) + 0.5) * self.res
        yy, zz = np.meshgrid(ys, zs, indexing='ij')              # (Y, Z)
        base = np.arange(Y * Z, dtype=np.int64).reshape(Y, Z)
        for ix in range(X):                                      # 按 x 切片，控制临时内存
            p = np.stack([np.full_like(yy, xs[ix]), yy, zz], -1).reshape(-1, 3) - t
            pc = p @ R_T.T                                       # (N,3) 相机系 x前 y右 z上
            x = pc[:, 0]
            front = x > 0.05
            u = np.full(x.shape, -1.0, np.float32)
            v = np.full(x.shape, -1.0, np.float32)
            u[front] = W / 2.0 + f * pc[front, 1] / x[front]
            v[front] = H / 2.0 - f * pc[front, 2] / x[front]
            ok = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not ok.any():
                continue
            vidx.append(ix * Y * Z + base.reshape(-1)[ok])
            pidx.append(v[ok].astype(np.int64) * W + u[ok].astype(np.int64))
            depth.append(x[ok].astype(np.float32))
        if not vidx:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
        return np.concatenate(vidx), np.concatenate(pidx), np.concatenate(depth)

    def _pixel_rays(self, cam):
        """每个像素中心的视线 (平面深度 = 1 时的相机系坐标) 和单位方向。"""
        W, H = cam['width'], cam['height']
        f = self._focal(cam)
        u = (np.arange(W, dtype=np.float32) + 0.5 - W / 2.0) / f
        v = (np.arange(H, dtype=np.float32) + 0.5 - H / 2.0) / f
        uu, vv = np.meshgrid(u, v)                               # (H, W)
        ray = np.stack([np.ones_like(uu), uu, -vv], -1).reshape(-1, 3)   # x前 y右 z上
        ray_v = (ray @ cam['R'].T).astype(np.float32)            # 车辆系方向 (未归一化，x_cam=1)
        unit = ray_v / np.linalg.norm(ray_v, axis=1, keepdims=True)
        return ray_v, unit

    # ------------------------------------------------------------------ 每帧
    def _mark_points(self, flat_mask, pts):
        ix = np.floor((pts[:, 0] - self.x0) / self.res).astype(np.int64)
        iy = np.floor((pts[:, 1] - self.y0) / self.res).astype(np.int64)
        iz = np.floor((pts[:, 2] - self.z0) / self.res).astype(np.int64)
        X, Y, Z = self.shape
        ok = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
        flat_mask[(ix[ok] * Y + iy[ok]) * Z + iz[ok]] = True

    def compute(self, depth_maps):
        """depth_maps: 与 cameras 同序的 (H, W) 平面深度图 (米)。返回观测掩码 (X, Y, Z) bool。"""
        return self.analyze(depth_maps)[0]

    def analyze(self, depth_maps, carve_margin=0.3, carve_rel=0.02):
        """一次遍历同时返回 (观测掩码, 视线明确穿过的体素)，两者都是 (X, Y, Z) bool。"""
        flat = np.zeros(int(np.prod(self.shape)), dtype=bool)
        through = np.zeros_like(flat)
        hit = np.zeros_like(flat)                 # 某个相机的表面点落进的体素
        half = 0.5 * self.res
        for cam, (vidx, pidx, vdepth), (ray_v, unit), depth in zip(self.cameras, self._proj, self._rays,
                                                                  depth_maps):
            depth = np.asarray(depth, np.float32)
            d = depth.reshape(-1)
            # (b) 自由空间
            flat[vidx[vdepth <= d[pidx] + self.free_tol]] = True
            # 雕刻判据：3×3 腐蚀后的深度
            d_min = _erode3(depth).reshape(-1)
            through[vidx[vdepth < d_min[pidx] - (carve_margin + carve_rel * vdepth)]] = True
            # (a) 表面命中
            ok = d < self.max_hit_depth
            pts = ray_v[ok] * d[ok, None] + cam['t'].astype(np.float32)
            self._mark_points(hit, pts)
            self._mark_points(hit, pts + unit[ok] * half)            # 视线往后半格 (竖直表面)
            pts[:, 2] -= half
            self._mark_points(hit, pts)                              # 往下半格 (路面/地面)
        flat |= hit
        # 被任何一个相机拍到表面的体素不雕：树叶/铁丝网这类半透明物体，同一个体素可能一个相机拍到叶子、
        # 另一个相机从叶缝看穿；没有这条时 13% 的植被表面点落进了被雕空的体素 (逐像素核对)
        through &= ~hit
        return flat.reshape(self.shape), through.reshape(self.shape)


    def paint_ground(self, occupancy, depth_maps, sem_maps, min_votes=2):
        """用 UE 语义分割重绘地面列的类别 (原地修改 occupancy)。返回被改动的列数。"""
        X, Y, Z = self.shape
        classes = np.array(sorted(set(GROUND_TAG_TO_CLASS.values())), dtype=np.uint8)
        cls_index = np.full(256, -1, np.int64)
        lut = np.zeros(256, np.uint8)
        for tag, c in GROUND_TAG_TO_CLASS.items():
            lut[tag] = c
            cls_index[tag] = int(np.flatnonzero(classes == c)[0])
        votes = np.zeros(X * Y * len(classes), np.int32)
        for cam, (ray_v, _), depth, sem in zip(self.cameras, self._rays, depth_maps, sem_maps):
            d = np.asarray(depth, np.float32).reshape(-1)
            t = np.asarray(sem).reshape(-1)
            ok = (d < self.max_hit_depth) & (cls_index[t] >= 0)
            pts = ray_v[ok] * d[ok, None] + cam['t'].astype(np.float32)
            ix = np.floor((pts[:, 0] - self.x0) / self.res).astype(np.int64)
            iy = np.floor((pts[:, 1] - self.y0) / self.res).astype(np.int64)
            inb = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y)
            np.add.at(votes, (ix[inb] * Y + iy[inb]) * len(classes) + cls_index[t[ok][inb]], 1)
        votes = votes.reshape(X, Y, len(classes))
        total = votes.sum(2)
        new = classes[votes.argmax(2)]                                   # (X, Y)
        is_ground = np.isin(occupancy, GROUND_LABELS)
        col = (total >= min_votes) & is_ground.any(2)
        change = is_ground & col[..., None] & (occupancy != new[..., None])
        occupancy[change] = np.broadcast_to(new[..., None], occupancy.shape)[change]
        return int(np.count_nonzero(change.any(2)))


def finalize_ground_truth(vis, occupancy, actor_ids, depth_maps, sem_maps=None):
    """生成器输出之后的相机侧后处理 (原地修改 occupancy/actor_ids)：地面语义重绘 → 观测掩码 → 自由空间雕刻。
    返回 (mask_camera, 统计 dict)。"""
    stats = {}
    if sem_maps is not None:
        stats['ground_cols_repainted'] = vis.paint_ground(occupancy, depth_maps, sem_maps)
    mask, through = vis.analyze(depth_maps)
    stats['carved'] = carve_static(occupancy, actor_ids, through)
    stats['observed'] = float(mask.mean())
    return mask, stats


def _erode3(img):
    """(H, W) 3×3 最小值滤波 (边界复制)。"""
    p = np.pad(img, 1, mode='edge')
    H, W = img.shape
    out = p[1:H + 1, 1:W + 1].copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            np.minimum(out, p[dy:dy + H, dx:dx + W], out=out)
    return out


CARVE_LABELS = (1, 8, 16, 17)          # barrier / traffic_cone / vegetation / general_object：整类可雕
CARVE_SMALL_LABELS = (15,)             # manmade：只雕小物体 (杆/交通灯/路牌)，大楼不雕 (玻璃幕墙)
SMALL_OBJECT_VOXELS = 4000             # 0.2m 体素下 32m³；Town10 的楼远大于此，杆/交通灯远小于此


def carve_static(occupancy, actor_ids, through):
    """把视线明确穿过的静态物体体素改成 free (原地修改 occupancy/actor_ids)。返回被改的体素数。"""
    cand = through & np.isin(occupancy, CARVE_LABELS)
    small = through & np.isin(occupancy, CARVE_SMALL_LABELS)
    if small.any():
        ids = np.unique(actor_ids[small])
        ids = ids[ids != 0]
        if len(ids):
            vals, sizes = np.unique(actor_ids[np.isin(actor_ids, ids)], return_counts=True)
            ok_ids = vals[sizes <= SMALL_OBJECT_VOXELS]
            cand |= small & np.isin(actor_ids, ok_ids)
    occupancy[cand] = 0
    actor_ids[cand] = 0
    return int(np.count_nonzero(cand))

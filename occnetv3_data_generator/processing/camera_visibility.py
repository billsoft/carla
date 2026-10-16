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

finalize_ground_truth() 把 地面重绘(类别+高度) → 可见静态表面补建 → 掩码 → 雕刻 串起来，main_collection.py 和 audit_labels.py 共用。

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


def fisheye_from_intrinsics(ci):
    """intrinsics.json 的单相机条目 → 鱼眼投影参数 (_in_rgb 用)。支持 equidistant 和 kannala-brandt。"""
    dist = ci.get('distortion', {}) or {}
    model = dist.get('model', 'equidistant')
    if model == 'equidistant':
        k = (0.0, 0.0, 0.0, 0.0)
    elif model == 'kannala-brandt':
        k = tuple(float(x) for x in dist.get('coeffs', (0.0, 0.0, 0.0, 0.0)))
    else:
        raise ValueError(f'不支持的镜头模型 {model}')
    return {'fx': float(ci['fx']), 'fy': float(ci['fy']), 'cx': float(ci['cx']), 'cy': float(ci['cy']),
            'width': int(ci.get('width', 1280)), 'height': int(ci.get('height', 960)), 'k': k}


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
                         'width': int(ci.get('width', 1280)), 'height': int(ci.get('height', 960)),
                         'fisheye': fisheye_from_intrinsics(ci)})
            i += 1
        return cls(cams, x_range, y_range, z_range, resolution, **kw)

    def _in_rgb(self, cam, pc):
        """相机系 (x前 y右 z上) 点/方向 [N,3] 是否落在鱼眼 RGB 画面内。没有 'fisheye' 参数时全为 True。

        深度相机为了覆盖鱼眼四角，水平 FOV 比鱼眼宽 (120° 鱼眼 → 143° 针孔)，针孔画面里有大块鱼眼照片
        根本没拍到的区域；观测掩码必须再裁到鱼眼实际视野，否则这些区域被当成"相机看见了"
        (2026-10-15 评审的几何反例：空场景下 764 个网格中心被标可见却不在任何 RGB 图像内)。"""
        fe = cam.get('fisheye')
        if fe is None:
            return np.ones(len(pc), dtype=bool)
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        rho = np.sqrt(y * y + z * z)
        theta = np.arctan2(rho, x)
        k1, k2, k3, k4 = fe['k']
        t2 = theta * theta
        theta_d = theta * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)
        scale = np.where(rho > 1e-9, theta_d / np.maximum(rho, 1e-9), 0.0)
        u = fe['cx'] + fe['fx'] * scale * y
        v = fe['cy'] - fe['fy'] * scale * z
        return ((theta < np.pi / 2) & (u >= -0.5) & (u < fe['width'] - 0.5)
                & (v >= -0.5) & (v < fe['height'] - 0.5))

    def _focal(self, cam):
        return (cam['width'] / 2.0) / math.tan(math.radians(cam['hfov']) / 2.0)

    def _precompute(self, cam):
        """所有体素中心在该相机里的 (像素下标, 光轴距离)，只保留落在画面内的。"""
        W, H = cam['width'], cam['height']
        f = self._focal(cam)
        X, Y, Z = self.shape
        R_T = cam['R'].T                       # 车辆 → 相机
        t = cam['t']
        vidx, pidx, depth, rgb = [], [], [], []
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
            rgb.append(self._in_rgb(cam, pc[ok]))
        if not vidx:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, bool)
        return np.concatenate(vidx), np.concatenate(pidx), np.concatenate(depth), np.concatenate(rgb)

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
        return ray_v, unit, self._in_rgb(cam, ray)                # 第三项：该深度像素是否在鱼眼视野内

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
        hit = np.zeros_like(flat)                 # 某个深度像素的表面点落进的体素 (几何，用于雕刻保护)
        hit_rgb = np.zeros_like(flat)             # 同上，只算鱼眼视野内的像素 (用于观测掩码)
        half = 0.5 * self.res
        for cam, (vidx, pidx, vdepth, vrgb), (ray_v, unit, prgb), depth in zip(self.cameras, self._proj,
                                                                               self._rays, depth_maps):
            depth = np.asarray(depth, np.float32)
            d = depth.reshape(-1)
            # (b) 自由空间：观测掩码只认鱼眼视野内的
            flat[vidx[(vdepth <= d[pidx] + self.free_tol) & vrgb]] = True
            # 雕刻判据：3×3 腐蚀后的深度 (几何证据，不受鱼眼视野限制)
            d_min = _erode3(depth).reshape(-1)
            through[vidx[vdepth < d_min[pidx] - (carve_margin + carve_rel * vdepth)]] = True
            # (a) 表面命中：点本身 / 视线往后半格 (竖直表面) / 往下半格 (路面/地面)
            ok = d < self.max_hit_depth
            pts = ray_v[ok] * d[ok, None] + cam['t'].astype(np.float32)
            in_rgb = prgb[ok]
            for p in (pts, pts + unit[ok] * half, pts - np.array([0, 0, half], np.float32)):
                self._mark_points(hit, p)
                self._mark_points(hit_rgb, p[in_rgb])
        flat |= hit_rgb
        # 被任何一个相机拍到表面的体素不雕：树叶/铁丝网这类半透明物体，同一个体素可能一个相机拍到叶子、
        # 另一个相机从叶缝看穿；没有这条时 13% 的植被表面点落进了被雕空的体素 (逐像素核对)
        through &= ~hit
        return flat.reshape(self.shape), through.reshape(self.shape)


    LAYER_GAP_M = 1.0       # 同一列里高度差超过它的地面观测算不同层 (桥上桥下、上下层停车场)
    CREATE_DEPTH = 6        # 补建表面时最多往下填的格数 (1.2m)，不把上下两层之间的空间填成地面

    def paint_ground(self, occupancy, depth_maps, sem_maps, min_votes=2, actor_ids=None, max_shift=None,
                     through=None):
        """用 UE 语义分割重绘地面 (原地修改 occupancy / actor_ids)。返回被改动的地面层数。

        - 分层：同一 XY 列的地面观测先按高度分层 (相邻观测高差 > LAYER_GAP_M 开新层)，每层独立投票、
          独立对齐。2026-10-15 评审复现：上层人行道 4m、下层路面 0m 各有观测时，按整列平均得到 ~2m，
          结果上层被删除——平均必须在分层之后做。
        - 类别：每层只改与该层观测表面相连的那一段地面体素 (从观测高度附近的地面体素往下、到第一个
          非地面体素为止)。
        - 高度：地表高度来自 Map API (车道/人行道 waypoint 的 z、车道外取最近车道高度)，路缘石、草地起伏、
          丘陵地形与真实网格有偏差。用该层观测点的平均高度把这一段最上层地面体素对齐到"中心低于观测表面
          的最高一格" (max_shift 默认不限)：抬高时可挤掉静态物体包围盒，不动车辆/行人；降低时只清掉这一段
          连续的地面体素；该层附近根本没有地面体素时从观测高度往下补建，最多 CREATE_DEPTH 格。
        - 层匹配双向限高：观测比找到的地面段高/低出一个层间距以上，都不当成"同一层的高度误差"。低于时只有
          视线确实穿过了那段地面 (要清掉的每一格都有 through，analyze 的几何穿越证据；2026-10-16 之前只要
          一格有证据就整段删除) 才降低，整段都在观测之上时清掉后在观测高度补建——看见桥下路面不代表视线穿过了
          桥面 (2026-10-15 评审复现：只有 4m 人行道、只看到 0m 路面时整列被清空)；没有证据则在观测高度
          独立补建，上层保持不动。through 为 None 时现算。"""
        X, Y, Z = self.shape
        if through is None:
            through = self.analyze(depth_maps)[1]
        if max_shift is None:
            max_shift = Z
        # 观测比地面段低一个层间距以上的列按证据分到哪一类 (见 _paint_ground_layer)，统计写进每帧 meta
        self.far_down_counts = {'lowered': 0, 'full_detached': 0, 'partial': 0, 'none': 0}
        classes = np.array(sorted(set(GROUND_TAG_TO_CLASS.values())), dtype=np.uint8)
        cls_index = np.full(256, -1, np.int64)
        for tag, c in GROUND_TAG_TO_CLASS.items():
            cls_index[tag] = int(np.flatnonzero(classes == c)[0])
        cols_l, zf_l, cls_l = [], [], []
        for cam, (ray_v, _, _), depth, sem in zip(self.cameras, self._rays, depth_maps, sem_maps):
            d = np.asarray(depth, np.float32).reshape(-1)
            t = np.asarray(sem).reshape(-1)
            ok = (d < self.max_hit_depth) & (cls_index[t] >= 0)
            pts = ray_v[ok] * d[ok, None] + cam['t'].astype(np.float32)
            ix = np.floor((pts[:, 0] - self.x0) / self.res).astype(np.int64)
            iy = np.floor((pts[:, 1] - self.y0) / self.res).astype(np.int64)
            inb = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y)
            cols_l.append(ix[inb] * Y + iy[inb])
            zf_l.append((pts[inb, 2] - self.z0) / self.res)              # 体素单位的连续高度
            cls_l.append(cls_index[t[ok][inb]])
        col = np.concatenate(cols_l) if cols_l else np.zeros(0, np.int64)
        if len(col) == 0:
            self.last_ground_shift = np.zeros(0, np.int64)
            return 0
        zf = np.concatenate(zf_l)
        cl = np.concatenate(cls_l)
        order = np.lexsort((zf, col))
        col, zf, cl = col[order], zf[order], cl[order]
        new_layer = np.ones(len(col), bool)
        new_layer[1:] = (col[1:] != col[:-1]) | (zf[1:] - zf[:-1] > self.LAYER_GAP_M / self.res)
        layer = np.cumsum(new_layer) - 1
        n_layers = int(layer[-1]) + 1
        count = np.bincount(layer, minlength=n_layers)
        zmean = np.bincount(layer, weights=zf, minlength=n_layers) / np.maximum(count, 1)
        votes = np.zeros(n_layers * len(classes), np.int64)
        np.add.at(votes, layer * len(classes) + cl, 1)
        votes = votes.reshape(n_layers, len(classes))
        lcol = col[new_layer]                                            # 每层所属的列
        rank = np.arange(n_layers) - np.searchsorted(lcol, lcol)         # 列内第几层 (按高度从低到高)
        keep = count >= min_votes
        shifts, changed = [], 0
        for r in range(int(rank.max()) + 1):                            # 每一轮内各列只出现一次
            sel = np.flatnonzero(keep & (rank == r))
            if len(sel) == 0:
                continue
            n, sh = self._paint_ground_layer(occupancy, actor_ids, lcol[sel], zmean[sel],
                                             classes[votes[sel].argmax(1)], max_shift, through)
            changed += n
            shifts.append(sh)
        self.last_ground_shift = np.concatenate(shifts) if shifts else np.zeros(0, np.int64)
        return changed

    def _paint_ground_layer(self, occupancy, actor_ids, cols, mean_obs, new, max_shift, through):
        """对每列一个地面观测层做 类别重绘 + 高度对齐 (cols 互不重复)。返回 (改动层数, 高度调整量)。"""
        X, Y, Z = self.shape
        obs = np.floor(mean_obs).astype(np.int64)                        # 观测表面所在的体素层
        flat_occ = occupancy.reshape(X * Y, Z)                           # view
        occ_cols = flat_occ[cols]                                        # [n, Z] copy
        g = np.isin(occ_cols, GROUND_LABELS)
        # 起点：观测层附近最近的地面体素 (表面点常在地面体素顶面上，落进上面那层空体素)
        start = np.full(len(cols), -1, np.int64)
        for dz in [0, -1, 1, -2, 2] + [d for k in range(3, min(max_shift, Z) + 2) for d in (-k, k)]:
            zz = obs + dz
            cand = np.flatnonzero((start < 0) & (zz >= 0) & (zz < Z))
            hitz = g[cand, zz[cand]]
            start[cand[hitz]] = zz[cand[hitz]]
        # 再沿地面体素往上走到这段地面的顶部 (起点可能落在段中间)
        for _ in range(Z):
            up = np.flatnonzero((start >= 0) & (start + 1 < Z))
            up = up[g[up, start[up] + 1]]
            if len(up) == 0:
                break
            start[up] += 1
        # 观测表面比找到的地面段高出一个层间距以上：分不清"地图高度错了"还是"这是另一层 (桥面/上层)"，
        # 不去抬那段地面 (会把两层之间填满、还会改下层类别)，改为在观测高度处补建有限厚度的表面；
        # 下面那段地面若是错的，也被新表面挡住看不见，训练时被掩码忽略。
        # 观测表面在地面段下方一个层间距以上：只有视线穿过了这段地面 (见 paint_ground 文档) 才降低，否则同样
        # 当成另一层独立补建 (桥下路面)
        target = np.clip((np.ceil(mean_obs - 0.5) - 1).astype(np.int64), -1, Z - 1)
        gap = self.LAYER_GAP_M / self.res
        start[(start >= 0) & (target - start > gap)] = -1
        # 从起点往下连续的地面体素 = 这一层的地面段
        zz = start[:, None] - np.arange(Z)[None, :]                      # [n, Z]
        valid = (zz >= 0) & (start >= 0)[:, None]
        gk = np.zeros_like(valid)
        r, k = np.nonzero(valid)
        gk[r, k] = g[r, zz[r, k]]
        seg = np.cumprod(gk, axis=1).astype(bool)                        # 段内第 k 格 (自顶向下)
        far_down = (start >= 0) & (start - target > gap)
        if far_down.any():
            thr = through.reshape(X * Y, Z)[cols]
            above = seg & (zz > target[:, None]) & far_down[:, None]      # 降低时会被清掉的那部分
            crossed = np.zeros_like(above)
            r, k = np.nonzero(above)
            crossed[r, k] = thr[r, zz[r, k]]
            # 2026-10-16 按证据逐格处理 (之前 any()：只要一格有穿越证据就整段降低删除，上层 5 格只有 1 格被穿过时
            # 整段被删、下层也不补建)：
            # - 要清掉的每一格都被穿过、且这段一直延伸到观测高度：走下面原来的限幅降低 (同一层的高度误差)；
            # - 否则当成另一层：被视线穿过的格清成 free (有证据是空的；留着会和观测掩码"看穿了"互相矛盾)，
            #   没被穿过的格保留 (看不见，训练时被掩码忽略)，再由下面 create 在观测高度补建表面
            #   (整段都在观测之上时，原来的限幅降低最多只能清掉这一段，观测到的表面不会建出来，整列变空)
            full = above.any(1) & ~(crossed != above).any(1)
            reaches = (seg & (zz <= target[:, None])).any(1)
            other_layer = far_down & ~(full & reaches)
            r, k = np.nonzero(crossed & other_layer[:, None])
            occ_cols[r, zz[r, k]] = 0
            if actor_ids is not None:
                actor_ids.reshape(X * Y, Z)[cols[r], zz[r, k]] = 0
            start[other_layer] = -1
            seg[other_layer] = False
            fd = self.far_down_counts
            fd['lowered'] += int((far_down & full & reaches).sum())
            fd['full_detached'] += int((far_down & full & ~reaches).sum())
            fd['partial'] += int((far_down & ~full & crossed.any(1)).sum())
            fd['none'] += int((far_down & ~crossed.any(1)).sum())
        seg_len = seg.sum(1)
        change = np.zeros_like(g)
        r, k = np.nonzero(seg)
        change[r, zz[r, k]] = True
        change &= occ_cols != new[:, None]
        occ_cols[change] = np.broadcast_to(new[:, None], occ_cols.shape)[change]

        # 高度对齐：目标最上层地面体素 = 中心低于观测表面平均高度的最高一格 (target 见上)
        # 观测层附近没有地面体素：从观测高度往下补建，最多 CREATE_DEPTH 格，并在遇到下方已有体素处停止
        create = (start < 0) & (target >= 0)
        if create.any():
            zs = np.arange(Z)[None, :]
            occupied_below = (occ_cols[create] != 0) & (zs <= target[create, None])
            highest = np.where(occupied_below.any(1), (occupied_below * zs).max(1), -1)
            start[create] = np.maximum(highest, target[create] - self.CREATE_DEPTH)
        shift = np.clip(target - start, -max_shift, max_shift)
        shift[(start < 0) & ~create] = 0
        shift = np.maximum(shift, np.where(create, shift, -seg_len))     # 降低不超过这一段的厚度
        moved = np.zeros(len(cols), bool)
        aid_cols = actor_ids.reshape(X * Y, Z)[cols] if actor_ids is not None else None
        dynamic = np.isin(occ_cols, (2, 3, 4, 5, 6, 7, 9, 10))
        for k in range(1, max_shift + 1):
            # 抬高：start+1 .. start+shift 填成该层的新地面类别 (空或静态物体体素；不动车/人)
            r = np.flatnonzero((shift >= k) & (start + k < Z) & (start + k >= 0))
            r = r[~dynamic[r, start[r] + k]]
            occ_cols[r, start[r] + k] = new[r]
            if aid_cols is not None:
                aid_cols[r, start[r] + k] = -(1000 + new[r].astype(np.int32))
            moved[r] = True
            # 降低：这一段顶部往下 |shift| 格 (都在段内，见上面的限幅) 清成 free
            r = np.flatnonzero((shift <= -k) & (start - k + 1 >= 0))
            occ_cols[r, start[r] - k + 1] = 0
            if aid_cols is not None:
                aid_cols[r, start[r] - k + 1] = 0
            moved[r] = True
        # 看得见地面的层，地表正上方那一格不可能被静态包围盒占满：树冠/护栏/道具的盒子常一直伸到地面，把树下
        # 人行道上方的空气标成植被 (留出相机审计：Town10 人行道/地形表面点 24% 落进 vegetation 体素)。
        # 清掉这一格里的静态盒子类别；真正看到的树干/道具表面由随后的 fill_observed_surfaces 补回
        top = start + shift                                                # 对齐后的最上层地面体素
        r = np.flatnonzero(((start >= 0) | create) & (top >= -1) & (top + 1 < Z))
        r = r[np.isin(occ_cols[r, top[r] + 1], CARVE_LABELS)]
        occ_cols[r, top[r] + 1] = 0
        if aid_cols is not None:
            aid_cols[r, top[r] + 1] = 0
        flat_occ[cols] = occ_cols
        if aid_cols is not None:
            actor_ids.reshape(X * Y, Z)[cols] = aid_cols
        return int(np.count_nonzero(change.any(1) | moved)), shift[(start >= 0) | create]

    def fill_observed_surfaces(self, occupancy, actor_ids, depth_maps, sem_maps):
        """补建相机看到、但真值里是空的静态表面 (原地修改)。返回补建的体素数。

        深度像素反投影到 3D，若该像素的 UE 语义是静态类别 (FILL_LABELS：护栏/锥桶/人造物/植被/一般物体)
        且落进 free 体素，就把这个体素标成该类别。补的是真实可见表面 (一层)，内部看不见，训练时被掩码忽略。
        动机：Town15 等地图的地景植被 (landscape foliage) 不是 EnvironmentObject，根本没有包围盒；
        包围盒也覆盖不到网格的所有细节。车辆/行人不补 (它们的流场需要实例归属，由 Actor 包围盒负责)。"""
        X, Y, Z = self.shape
        lut = np.zeros(256, np.uint8)
        for tag, c in CARLA_TO_OCCUPANCY_MAPPING.items():
            if c in FILL_LABELS:
                lut[tag] = c
        flat_occ = occupancy.reshape(-1)
        flat_aid = actor_ids.reshape(-1)
        n = 0
        for cam, (ray_v, unit, _), depth, sem in zip(self.cameras, self._rays, depth_maps, sem_maps):
            d = np.asarray(depth, np.float32).reshape(-1)
            cls = lut[np.asarray(sem).reshape(-1)]
            ok = (d < self.max_hit_depth) & (cls > 0)
            # 表面点往视线后面挪 1/4 格再取体素，避免正好落在体素边界上时标到表面前面那格 (空气) 里
            pts = ray_v[ok] * d[ok, None] + cam['t'].astype(np.float32) + unit[ok] * (0.25 * self.res)
            ix = np.floor((pts[:, 0] - self.x0) / self.res).astype(np.int64)
            iy = np.floor((pts[:, 1] - self.y0) / self.res).astype(np.int64)
            iz = np.floor((pts[:, 2] - self.z0) / self.res).astype(np.int64)
            inb = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
            vid = (ix[inb] * Y + iy[inb]) * Z + iz[inb]
            c = cls[ok][inb]
            free = flat_occ[vid] == 0
            flat_occ[vid[free]] = c[free]
            flat_aid[vid[free]] = -(2000 + c[free].astype(np.int32))
            n += int(free.sum())
        return n


def finalize_ground_truth(vis, occupancy, actor_ids, depth_maps, sem_maps=None, object_volume=None):
    """生成器输出之后的相机侧后处理 (原地修改 occupancy/actor_ids)：地面语义重绘 → 观测掩码 → 自由空间雕刻。
    object_volume: {静态物体虚拟 ID: 原始包围盒体积 m³} (GroundTruthVoxelGenerator.static_object_volume)。
    返回 (mask_camera, 统计 dict)。"""
    stats = {}
    mask, through = vis.analyze(depth_maps)          # 只依赖深度，不依赖体素：先算一次，地面重绘和雕刻共用
    if sem_maps is not None:
        stats['ground_cols_repainted'] = vis.paint_ground(occupancy, depth_maps, sem_maps, actor_ids=actor_ids,
                                                          through=through)
        stats['surfaces_filled'] = vis.fill_observed_surfaces(occupancy, actor_ids, depth_maps, sem_maps)
        sh = getattr(vis, 'last_ground_shift', None)
        if sh is not None and len(sh):
            stats['ground_shift_cols'] = int(np.count_nonzero(sh))
            stats['ground_shift_mean_m'] = round(float(np.abs(sh[sh != 0]).mean() * vis.res), 3) if sh.any() else 0.0
        stats['ground_far_down'] = dict(getattr(vis, 'far_down_counts', {}))
    stats['carved'] = carve_static(occupancy, actor_ids, through, object_volume)
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
FILL_LABELS = (1, 8, 15, 16, 17)       # fill_observed_surfaces 可补建的静态类别
CARVE_SMALL_LABELS = (15,)             # manmade：只雕小物体 (杆/交通灯/路牌)，大楼不雕 (玻璃幕墙)
SMALL_OBJECT_M3 = 32.0                 # 包围盒体积阈值；Town10 的楼远大于此，杆/交通灯/路牌远小于此
SMALL_OBJECT_VOXELS = 4000             # 没有体积表时的退化判据 (= 32m³ / 0.2m 体素)


def carve_static(occupancy, actor_ids, through, object_volume=None):
    """把视线明确穿过的静态物体体素改成 free (原地修改 occupancy/actor_ids)。返回被改的体素数。

    "小物体"按原始包围盒体积判断 (object_volume)；只有拿不到体积表时才退化为按网格内体素数——后者会把
    被网格边界截掉大半的大楼误当成小物体雕掉。"""
    cand = through & np.isin(occupancy, CARVE_LABELS)
    small = through & np.isin(occupancy, CARVE_SMALL_LABELS)
    if small.any():
        ids = np.unique(actor_ids[small])
        ids = ids[ids != 0]
        if len(ids) and object_volume is not None:
            ok_ids = np.array([i for i in ids if object_volume.get(int(i), np.inf) <= SMALL_OBJECT_M3])
            if len(ok_ids):
                cand |= small & np.isin(actor_ids, ok_ids)
        elif len(ids):
            vals, sizes = np.unique(actor_ids[np.isin(actor_ids, ids)], return_counts=True)
            ok_ids = vals[sizes <= SMALL_OBJECT_VOXELS]
            cand |= small & np.isin(actor_ids, ok_ids)
    occupancy[cand] = 0
    actor_ids[cand] = 0
    return int(np.count_nonzero(cand))

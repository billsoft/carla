"""
OccNetV3 数据集加载（occnetv3_data_generator 格式）。

2026-10 几何约定重构（详见 geometry.py 文件头）：
- 外参返回 Camera(OpenCV)→Vehicle 静态安装标定，**不再**乘车辆世界位姿；
  旧代码把 ego_pose @ T_cam_vehicle（含绝对世界位姿）喂给网络，导致 3D 查询被当成
  世界原点附近的点，投影整体失效。
- 标定文件里的旋转矩阵是 UE 相机轴（x前 y右 z上），这里统一转换为 OpenCV 轴（x右 y下 z前）；
  旧版采集代码的旋转公式在 pitch/roll≠0 时与 CARLA 不一致（rear 相机），按标记自动修正。
- ego_motion 由车辆位姿计算：inv(pose_t) @ pose_{t-1}（上一帧车辆系 → 当前帧车辆系）。
- 时序序列不跨场景（按 scene_XXXX_frame_YYYY 的场景号和连续帧号切分）。
- 额外加载深度（深度辅助监督目标）和场景流（流场头目标）。

2026-10-14 相机可见性掩码（label_version 2 数据起，见 occnetv3_data_generator/processing/camera_visibility.py）：
- 有 mask_camera/<sid>.npy 时，相机没观测到的体素真值置为 IGNORE_INDEX(255)，损失和 mIoU 都跳过它们
  （Occ3D 约定）；occupancy/*.npy 本身是全知真值，不改。config.use_camera_mask 关闭。
- config.camera_mask_keep_ground_surface（默认开）：车/人正下方那一列最上面的地面类体素 (11/12/13/14)
  即使相机看不到也保留监督——路面在车底下连续，这是可推断的，也是此前"地面类强制保留"的约定。
  被建筑挡住、或只是在视线外的地面不保留；地面往下填充的整列始终忽略。
- 帧间隔：scenes/<scene>.json 的 frame_dt（采集抽帧后不再是 0.05s），作为 time_delta [T] 返回。

已弃用的 dense_occupancy_collection 格式（camera_params/*.npz）不再支持。
"""

import json
import math
import os
import re

import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import rawpy
except ImportError:          # 允许在没有 rawpy 的环境里做纯网络测试
    rawpy = None

try:
    from .geometry import (cam_to_vehicle_from_mount, cam_ue_to_vehicle_to_cv, legacy_rotation_to_carla,
                           equidistant_unproject, level_pixel_to_image)
except ImportError:
    from geometry import (cam_to_vehicle_from_mount, cam_ue_to_vehicle_to_cv, legacy_rotation_to_carla,
                          equidistant_unproject, level_pixel_to_image)


_SAMPLE_RE = re.compile(r'^(?P<scene>.+)_frame_(?P<frame>\d+)$')

IGNORE_INDEX = 255
GROUND_LABELS = (11, 12, 13, 14)        # driveable_surface / other_flat / sidewalk / terrain
DYNAMIC_LABELS = (2, 3, 4, 5, 6, 7, 9, 10)   # bicycle/bus/car/工程车/motorcycle/pedestrian/trailer/truck


def ground_surface_keep(occ):
    """(X,Y,Z) uint8 → bool：车/人正下方那一列最上面的地面类体素 (被它们挡住的路面，连续可推断)。
    只看"上方有动态物体"的列——上方是空的列不在这里放行，否则被建筑挡住的远处地面也会被重新监督。"""
    gnd = np.isin(occ, GROUND_LABELS)
    zs = np.arange(occ.shape[2])
    top = np.where(gnd.any(2), (gnd * zs).max(2), -1)                    # (X, Y)
    dyn_above = (np.isin(occ, DYNAMIC_LABELS) & (zs[None, None, :] > top[..., None])).any(2)
    return gnd & (zs[None, None, :] == top[..., None]) & ((top >= 0) & dyn_above)[..., None]


def load_camera_mask(path, z_size):
    """mask_camera/<sid>.npy：打包格式 (X,Y,Z/8) 或未打包 (X,Y,Z)。"""
    m = np.load(path)
    if m.shape[2] != z_size:
        m = np.unpackbits(m, axis=2, count=z_size)
    return m.astype(bool)


def default_calibration(num_cameras=8, image_size=(960, 1280)):
    """从 occnetv3_data_generator/config/camera_config.py 构建标定（无 calibration/ 目录时、
    以及测试时使用）。返回 K [N,3,3]、T_cam(cv)→vehicle [N,4,4]、pinhole 深度相机水平 FOV(度) [N]。"""
    # 按文件路径加载，避免与 e2e_occ/config.py 同名模块冲突
    import importlib.util
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, 'occnetv3_data_generator', 'config', 'camera_config.py')
    spec = importlib.util.spec_from_file_location('occnetv3_camera_config', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    TESLA_CAMERAS = mod.TESLA_CAMERAS

    H, W = image_size
    cams = sorted(TESLA_CAMERAS, key=lambda c: c['index'])[:num_cameras]
    K = np.zeros((len(cams), 3, 3), dtype=np.float32)
    T = np.zeros((len(cams), 4, 4), dtype=np.float32)
    hfov = np.zeros(len(cams), dtype=np.float32)
    for i, c in enumerate(cams):
        fov_v = c['fov_vertical']
        fov_h = c.get('fov_horizontal', fov_v * W / H)
        K[i] = [[(W / 2.0) / (math.radians(fov_h) / 2.0), 0, 0],
                [0, (H / 2.0) / (math.radians(fov_v) / 2.0), 0],
                [0, 0, 1]]
        cx, cy = c.get('principal_point', (W / 2.0, H / 2.0))
        K[i, 0, 2], K[i, 1, 2] = cx, cy
        T[i] = cam_to_vehicle_from_mount(c['position'], c['rotation'])
        hfov[i] = c.get('depth_fov', c['fov'])
    return torch.from_numpy(K), torch.from_numpy(T), hfov


class OccupancyDataset(Dataset):
    def __init__(self, data_root, split='train', config=None):
        self.data_root = data_root
        self.split = split
        self.config = config
        self.num_cameras = config.num_cameras if config else 8
        self.image_size = tuple(config.image_size) if config else (960, 1280)
        self.voxel_size = tuple(config.voxel_size) if config else (400, 400, 32)
        self.sequence_length = config.temporal_frames if (config and config.use_temporal) else 1
        self.load_depth = bool(config and config.use_depth_aux)
        self.load_flow = bool(config and config.use_flow_head)
        self.depth_stride = config.depth_stride if config else 8
        self.use_camera_mask = bool(getattr(config, 'use_camera_mask', True)) if config else True
        self.keep_ground_surface = bool(getattr(config, 'camera_mask_keep_ground_surface', True)) if config else True
        self.default_frame_dt = float(getattr(config, 'frame_dt', 0.05)) if config else 0.05

        self.samples = self._load_samples()
        self.index = self._build_sequence_index()
        self.intrinsics, self.extrinsics, self.depth_hfov = self._load_calibration()
        self.raw_bit_depth = self._load_raw_bit_depth()
        self._depth_maps = {}
        self._warned = set()
        self.scene_frame_dt = self._load_scene_frame_dt()
        self._check_label_version()

    # ------------------------------------------------------------------
    # 样本与序列索引
    # ------------------------------------------------------------------

    def _load_samples(self):
        split_file = os.path.join(self.data_root, f'{self.split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        occ_dir = os.path.join(self.data_root, 'occupancy')
        if os.path.isdir(occ_dir):
            return sorted(os.path.splitext(n)[0] for n in os.listdir(occ_dir) if n.endswith('.npy'))
        return []

    def _build_sequence_index(self):
        """序列起点列表：T 帧必须属于同一场景且帧号连续。无法解析帧号时退化为按列表顺序。"""
        T = self.sequence_length
        if T <= 1:
            return list(range(len(self.samples)))
        stride = (self.config.sequence_stride if self.config else 0) or T
        parsed = [_SAMPLE_RE.match(s) for s in self.samples]
        starts = []
        next_allowed = {}                                   # 每个场景下一个允许的起点（实现步长，场景内独立计数）
        for i in range(len(self.samples) - T + 1):
            scene = parsed[i]['scene'] if parsed[i] else None
            if i < next_allowed.get(scene, 0):
                continue
            ok = True
            for t in range(1, T):
                a, b = parsed[i + t - 1], parsed[i + t]
                if a is None or b is None:
                    continue
                if a['scene'] != b['scene'] or int(b['frame']) != int(a['frame']) + 1:
                    ok = False
                    break
            if ok:
                starts.append(i)
                next_allowed[scene] = i + stride
        return starts

    # ------------------------------------------------------------------
    # 标定
    # ------------------------------------------------------------------

    def _load_scene_frame_dt(self):
        """scenes/<scene>.json 的 frame_dt (抽帧采集后相邻保存帧的间隔)；没有就用 config.frame_dt。"""
        out = {}
        d = os.path.join(self.data_root, 'scenes')
        if os.path.isdir(d):
            for n in os.listdir(d):
                if n.endswith('.json'):
                    with open(os.path.join(d, n), encoding='utf-8') as f:
                        meta = json.load(f)
                    if 'frame_dt' in meta:
                        out[os.path.splitext(n)[0]] = float(meta['frame_dt'])
        return out

    def frame_dt(self, sample_id):
        m = _SAMPLE_RE.match(sample_id)
        return self.scene_frame_dt.get(m['scene'] if m else None, self.default_frame_dt)

    def _check_label_version(self):
        p = os.path.join(self.data_root, 'calibration', 'intrinsics.json')
        if os.path.exists(p):
            with open(p, 'r') as f:
                ver = json.load(f).get('label_version', 1)
            if ver < 2:
                warnings.warn(f'{self.data_root} 是 label_version {ver} 的旧数据：交通灯/标志被标成 traffic_cone、'
                              f'车道线→traffic_cone、路缘→barrier；与新数据类别含义不同，不要混训（mask_camera 可用 compute_camera_mask.py 补算）')

    def _load_raw_bit_depth(self):
        int_path = os.path.join(self.data_root, 'calibration', 'intrinsics.json')
        if os.path.exists(int_path):
            with open(int_path, 'r') as f:
                data = json.load(f)
            if 'raw_bit_depth' in data:
                return int(data['raw_bit_depth'])
        return 12

    def _load_calibration(self):
        calib_dir = os.path.join(self.data_root, 'calibration')
        int_path = os.path.join(calib_dir, 'intrinsics.json')
        ext_path = os.path.join(calib_dir, 'extrinsics.json')
        if not (os.path.exists(int_path) and os.path.exists(ext_path)):
            warnings.warn(f'{calib_dir} 缺少标定文件，改用 occnetv3_data_generator/config/camera_config.py 构建')
            return default_calibration(self.num_cameras, self.image_size)

        with open(int_path, 'r') as f:
            int_data = json.load(f)
        with open(ext_path, 'r') as f:
            ext_data = json.load(f)
        # 新版采集代码写入的约定标记；旧数据集没有这些字段
        rot_conv = ext_data.get('rotation_convention', 'legacy')
        axes = ext_data.get('camera_axes', 'ue')

        K_list, T_list, hfov = [], [], []
        for i in range(self.num_cameras):
            ci = int_data[f'cam_{i}']
            K = torch.eye(3)
            K[0, 0], K[1, 1], K[0, 2], K[1, 2] = ci['fx'], ci['fy'], ci['cx'], ci['cy']
            K_list.append(K)
            hfov.append(float(ci.get('depth_fov', ci.get('fov', 90.0))))   # 针孔深度相机的水平 FOV

            ce = ext_data[f'cam_{i}']
            R = np.array(ce['rotation_matrix'], dtype=np.float64)
            if rot_conv != 'ue_frotator':
                R = legacy_rotation_to_carla(R)
            T_ue = np.eye(4)
            T_ue[:3, :3] = R
            T_ue[:3, 3] = ce['translation']
            T = T_ue if axes == 'opencv' else cam_ue_to_vehicle_to_cv(T_ue)
            T_list.append(torch.from_numpy(T.astype(np.float32)))
        return torch.stack(K_list), torch.stack(T_list), np.array(hfov, dtype=np.float32)

    # ------------------------------------------------------------------
    # 深度监督目标：pinhole 平面深度 → 等距鱼眼特征像素的射线距离
    # ------------------------------------------------------------------

    def _depth_sampling_map(self, cam, depth_hw):
        """采集侧深度相机是**针孔** sensor.camera.depth（水平 FOV = 标定里的 'depth_fov'，旧数据为 'fov'；
        与 RGB 同位姿），
        值是平面 z 深度。对鱼眼 depth_stride 层级的每个特征像素：等距反投影得到射线方向 →
        投影到针孔相机取深度 → 除以方向 z 分量得到沿射线的距离。针孔覆盖不到的鱼眼边角标记无效。"""
        key = (cam, depth_hw)
        if key in self._depth_maps:
            return self._depth_maps[key]
        s = self.depth_stride
        h, w = self.image_size[0] // s, self.image_size[1] // s
        ys = level_pixel_to_image(torch.arange(h, dtype=torch.float32), s)
        xs = level_pixel_to_image(torch.arange(w, dtype=torch.float32), s)
        v, u = torch.meshgrid(ys, xs, indexing='ij')
        K = self.intrinsics[cam]
        d = equidistant_unproject(u, v, K[0, 0], K[1, 1], K[0, 2], K[1, 2])     # [h, w, 3]
        Hd, Wd = depth_hw
        f_p = (Wd / 2.0) / math.tan(math.radians(float(self.depth_hfov[cam])) / 2.0)
        dz = d[..., 2]
        safe_z = dz.clamp(min=1e-3)
        px = f_p * d[..., 0] / safe_z + Wd / 2.0
        py = f_p * d[..., 1] / safe_z + Hd / 2.0
        ix, iy = px.floor().long(), py.floor().long()
        valid = (dz > 1e-3) & (ix >= 0) & (ix < Wd) & (iy >= 0) & (iy < Hd)
        ix = ix.clamp(0, Wd - 1)
        iy = iy.clamp(0, Hd - 1)
        m = (iy * Wd + ix, valid, dz)
        self._depth_maps[key] = m
        return m

    def _load_depth(self, sample_id):
        s = self.depth_stride
        h, w = self.image_size[0] // s, self.image_size[1] // s
        out = torch.zeros(self.num_cameras, h, w)
        found = False
        for cam in range(self.num_cameras):
            p = os.path.join(self.data_root, 'depth', sample_id, f'cam_{cam}.npy')
            if not os.path.exists(p):
                continue
            planar = torch.from_numpy(np.load(p).astype(np.float32))
            flat_idx, valid, dz = self._depth_sampling_map(cam, tuple(planar.shape))
            rng = planar.view(-1)[flat_idx] / dz.clamp(min=1e-3)
            out[cam] = torch.where(valid, rng, torch.zeros_like(rng))
            found = True
        return out, found

    # ------------------------------------------------------------------
    # 单帧加载
    # ------------------------------------------------------------------

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(msg)

    def _load_dng_image(self, dng_path):
        if rawpy is None:
            self._warn_once('rawpy', '未安装 rawpy，图像以全零代替')
            return torch.zeros(1, *self.image_size)
        try:
            with rawpy.imread(dng_path) as raw:
                img = raw.raw_image_visible.astype(np.float32)
            img = img / float((1 << self.raw_bit_depth) - 1)
            return torch.from_numpy(img[np.newaxis])
        except Exception as e:
            print(f'Error loading DNG {dng_path}: {e}')
            return torch.zeros(1, *self.image_size)

    def _load_frame(self, sample_id):
        images = []
        for i in range(self.num_cameras):
            p = os.path.join(self.data_root, 'images', sample_id, f'cam_{i}.dng')
            if os.path.exists(p):
                images.append(self._load_dng_image(p))
            else:
                self._warn_once('img', f'缺少图像（如 {p}），以全零代替')
                images.append(torch.zeros(1, *self.image_size))

        occ_path = os.path.join(self.data_root, 'occupancy', f'{sample_id}.npy')
        occ = np.load(occ_path).astype(np.uint8) if os.path.exists(occ_path) \
            else np.zeros(self.voxel_size, dtype=np.uint8)
        observed = None
        if self.use_camera_mask:
            mp = os.path.join(self.data_root, 'mask_camera', f'{sample_id}.npy')
            if os.path.exists(mp):
                observed = load_camera_mask(mp, occ.shape[2])
                if self.keep_ground_surface:
                    observed |= ground_surface_keep(occ)
                occ = np.where(observed, occ, np.uint8(IGNORE_INDEX))
            else:
                self._warn_once('mask_camera', f'缺少相机可见性掩码（如 {mp}），这些帧在全部体素上监督；'
                                               f'可用 occnetv3_data_generator/compute_camera_mask.py 补算')
        voxels = torch.from_numpy(occ)

        pose_path = os.path.join(self.data_root, 'ego_pose', f'{sample_id}.npy')
        if os.path.exists(pose_path):
            pose = torch.from_numpy(np.load(pose_path).astype(np.float64))
        else:
            self._warn_once('pose', '缺少 ego_pose，时序对齐退化为单位阵')
            pose = torch.eye(4, dtype=torch.float64)

        frame = {'images': torch.stack(images).float(), 'voxels': voxels, 'pose': pose}

        if self.load_depth:
            depth, found = self._load_depth(sample_id)
            frame['depth'] = depth
            frame['has_depth'] = torch.tensor(found)
        if self.load_flow:
            fz = os.path.join(self.data_root, 'flow', f'{sample_id}.npz')       # 新数据：压缩
            fp = os.path.join(self.data_root, 'flow', f'{sample_id}.npy')
            mp = os.path.join(self.data_root, 'flow_mask', f'{sample_id}.npy')
            if (os.path.exists(fz) or os.path.exists(fp)) and os.path.exists(mp):
                if os.path.exists(fz):
                    with np.load(fz) as z:
                        flow = z['flow']
                else:
                    flow = np.load(fp)
                fmask = np.load(mp).astype(np.bool_)
                if observed is not None:
                    fmask &= observed              # 看不见的那半边车不监督流场
                frame['flow'] = torch.from_numpy(flow.astype(np.float16))
                frame['flow_mask'] = torch.from_numpy(fmask)
                frame['has_flow'] = torch.tensor(True)
            else:
                frame['flow'] = torch.zeros(3, *self.voxel_size, dtype=torch.float16)
                frame['flow_mask'] = torch.zeros(self.voxel_size, dtype=torch.bool)
                frame['has_flow'] = torch.tensor(False)
        return frame

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """始终返回带时间维 T 的字典（T = temporal_frames，非时序时 T=1）：
          images [T,N,1,H,W] float；voxels [T,X,Y,Z] uint8（GPU 上再 .long()；255 = 相机未观测，忽略）
          time_delta [T] 与上一帧的间隔秒数（场景元数据 frame_dt）
          ego_motion [T,4,4]（t=0 为单位阵）；intrinsics [N,3,3]；extrinsics [N,4,4] Camera(cv)→Vehicle
          depth [T,N,h,w]、has_depth [T]；flow [T,3,X,Y,Z] float16、flow_mask [T,X,Y,Z]、has_flow [T]
        """
        start = self.index[idx]
        T = self.sequence_length
        frames = [self._load_frame(self.samples[start + t]) for t in range(T)]
        ego_motion = [torch.eye(4)]
        for t in range(1, T):
            m = torch.linalg.inv(frames[t]['pose']) @ frames[t - 1]['pose']
            ego_motion.append(m.float())
        out = {
            'images': torch.stack([f['images'] for f in frames]),
            'voxels': torch.stack([f['voxels'] for f in frames]),
            'ego_motion': torch.stack(ego_motion),
            'time_delta': torch.tensor([self.frame_dt(self.samples[start + t]) for t in range(T)],
                                       dtype=torch.float32),
            'intrinsics': self.intrinsics.clone(),
            'extrinsics': self.extrinsics.clone(),
        }
        for k in ('depth', 'has_depth', 'flow', 'flow_mask', 'has_flow'):
            if k in frames[0]:
                out[k] = torch.stack([f[k] for f in frames])
        return out


def get_dataloader(data_root, split='train', batch_size=1, num_workers=0, config=None):
    dataset = OccupancyDataset(data_root, split, config)
    shuffle = split == 'train'
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=shuffle)

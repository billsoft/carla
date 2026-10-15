"""
数据保存器 - OccNetV3 数据集格式
按照 DATASET_FORMAT.md 规范保存
"""
import os
import json
import hashlib
import numpy as np
import concurrent.futures
# cv2 is optional
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from pathlib import Path
from typing import Dict, List, Union
# 导入配置中的 GRID_SIZE
try:
    from config.occupancy_config import GRID_SIZE, LABEL_VERSION
    from processing.camera_visibility import save_mask
except ImportError:
    # 如果作为模块导入失败，尝试相对导入
    from ..config.occupancy_config import GRID_SIZE, LABEL_VERSION
    from ..processing.camera_visibility import save_mask

# DNG支持（可选）
try:
    from PIL import Image, TiffImagePlugin
    DNG_AVAILABLE = True
except ImportError:
    DNG_AVAILABLE = False
    print("[警告] 未安装 Pillow，DNG格式不可用。安装: pip install Pillow")


def _atomic_write(path, write_fn):
    """write_fn(fileobj) 写到 <path>.tmp，成功后 os.replace 原子改名为 path；失败删除临时文件并抛出。
    进程在写入中途被杀时只会留下 .tmp，不会留下截断的正式文件被当成有效样本。"""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    try:
        with open(tmp, 'wb') as f:
            write_fn(f)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


# 数据集级写入协议文件：存在即表示这个目录由带完成标记的新代码写过。此后没有完成标记的帧一律不完整，
# 只有它创建时已经在盘上、没有完成标记的旧帧 (legacy_samples) 才按"必需文件存在且非空"兜底
PROTOCOL_FILE = 'write_protocol.json'


def _marker_files(output_dir, sample_id):
    """完成标记里的 {相对路径: 字节数}；没有标记/标记损坏/旧版标记没有 files 时返回 None。"""
    marker = Path(output_dir) / 'meta' / f'{sample_id}.json'
    try:
        with open(marker, encoding='utf-8') as f:
            return json.load(f).get('files') or None
    except (OSError, ValueError):
        return None


def _json_diff(a, b, path='', tol=1e-6):
    """两个 json 对象的差异列表 (数值按容差比较)。"""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f'{path}/{k} 只在一边')
            else:
                out += _json_diff(a[k], b[k], f'{path}/{k}', tol)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f'{path} 长度 {len(a)}≠{len(b)}']
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _json_diff(x, y, f'{path}[{i}]', tol)]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if abs(a - b) <= tol * max(1.0, abs(a)) else [f'{path}: {a}≠{b}']
    return [] if a == b else [f'{path}: {a}≠{b}']


class OccNetDataSaver:
    """
    保存OccNetV3训练数据
    目录结构:
    dataset/
        ├── calibration/
        │   ├── intrinsics.json
        │   └── extrinsics.json
        ├── images/
        │   └── scene_XXXX_frame_YYYY/
        │       ├── cam_0.dng  (Bayer RGGB, 位深可配置, 默认 12-bit)
        │       └── ...
        ├── depth/
        │   └── scene_XXXX_frame_YYYY/
        │       ├── cam_0.npy  (H, W) float16, 单位: 米 (旧数据 float32)
        │       └── ...
        ├── occupancy/
        │   └── scene_XXXX_frame_YYYY.npy  (400, 400, 32) uint8
        ├── flow/
        │   └── scene_XXXX_frame_YYYY.npz  键 'flow': (3, 400, 400, 32) float16，压缩 (旧数据是同形状 .npy)
        ├── flow_mask/
        │   └── scene_XXXX_frame_YYYY.npy  (400, 400, 32) uint8
        ├── ego_pose/
        │   └── scene_XXXX_frame_YYYY.npy  (4, 4) float32
        ├── ego_motion/
        │   └── scene_XXXX_frame_YYYY.npy  (4, 4) float32
        ├── mask_camera/
        │   └── scene_XXXX_frame_YYYY.npy  (400, 400, 4) uint8 = np.packbits(mask, axis=2)，1=相机观测到
        │                                   (读写用 processing/camera_visibility.py 的 save_mask/load_mask)
        ├── scenes/
        │   └── scene_XXXX.json            每个场景的地图/天气/种子/抽帧间隔
        ├── train.txt
        ├── val.txt
        └── test.txt
    """

    def __init__(self, output_dir: str, scene_name: str = 'scene', raw_bit_depth: int = 12,
                 save_actor_ids: bool = False, scene_index: int = 0):
        """
        Args:
            output_dir: 输出根目录 (如 D:/carla_datasets/occ_v2)
            scene_name: 场景名称前缀
            raw_bit_depth: 保存 DNG 时模拟的传感器 ADC 位深 (8/10/12/14/16)，
                默认 12-bit（常见 CMOS 传感器精度）。相机管线内部始终以 16-bit
                (0-65535) 传输，这里只是保存前的量化位深，越大保留的细节越多、
                文件也略大。必须是 8-16 之间的整数。
            save_actor_ids: 调试开关 (默认关闭)。开启后额外保存每帧的 actor_ids
                (400,400,32) int32 到 debug_actor_ids/ 目录——真实 actor 是正数
                CARLA actor.id，静态环境物体是负数虚拟ID (-(i+10000)，i 是
                ground_truth_voxel_generator.py 里 world.get_environment_objects()
                返回列表的下标)。仅用于排查体素归属问题 (点一个体素查它是哪个
                actor/环境物体生成的)，训练不需要这份数据，默认不生成，避免
                每帧多出一个 400*400*32*4 字节 (~20MB) 的文件拖慢正常采集。
            scene_index: 本次采集的场景编号 (sample_id 里的 XXXX)。多次运行追加到同一个数据集时
                每次必须不同，否则会覆盖已有样本 (main_collection.py --append 自动取下一个空闲编号)。
        """
        if not (8 <= raw_bit_depth <= 16):
            raise ValueError(f"raw_bit_depth 必须在 8-16 之间，收到: {raw_bit_depth}")
        if not DNG_AVAILABLE:
            # 以前没有 Pillow 时用 OpenCV 写普通 TIFF 再改名 .dng：没有 CFAPattern/WhiteLevel，读回白点 65535，
            # 下游拿到的是"看起来是 DNG 的错误数据"。宁可直接失败
            raise ImportError("保存 DNG 需要 Pillow：pip install Pillow")

        self.output_dir = Path(output_dir)
        self.scene_name = scene_name
        self.raw_bit_depth = raw_bit_depth
        self.save_actor_ids = save_actor_ids
        self.scene_counter = scene_index
        self.frame_counter = 0

        self.sample_ids = []  # 记录所有sample_id

        # 后台IO线程池（DNG/npy异步写入，避免阻塞采集主线程）
        self._io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        # 完成标记单独一个线程：它要等同一帧在 IO 线程池里的写入，放在同一个池里可能把 4 个线程全占住死锁
        self._marker_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending_futures = []

        # 创建目录结构
        self._create_directories()

        print(f"[OccNetDataSaver] 初始化完成")
        print(f"  输出目录: {self.output_dir}")

    def _create_directories(self):
        """创建所有必需的目录"""
        dirs = [
            'calibration',
            'images',
            'depth',       # ⭐ 新增: 深度图目录
            'occupancy',
            'flow',
            'flow_mask',
            'ego_pose',
            'ego_motion',
            'mask_camera',
            'scenes',
            'meta',
        ]

        for d in dirs:
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)

        if self.save_actor_ids:
            (self.output_dir / 'debug_actor_ids').mkdir(parents=True, exist_ok=True)

        # 写入协议：首次用新代码打开这个目录时，把盘上已有、没有完成标记的旧帧冻结成 legacy 名单
        self._marker_gen = {}
        if (self.output_dir / PROTOCOL_FILE).exists():
            self._legacy = self._legacy_samples()
        else:
            self._legacy = {q.stem for q in (self.output_dir / 'occupancy').glob('*.npy')
                            if _marker_files(self.output_dir, q.stem) is None}
            self._write_protocol()

        print(f"  ✓ 目录结构已创建 (含 depth 目录{', 含 debug_actor_ids 目录' if self.save_actor_ids else ''})")

    def save_calibration(
        self,
        camera_intrinsics: Dict[str, np.ndarray],
        camera_extrinsics: Dict[str, np.ndarray],
        camera_configs: List[Dict],
        must_match_existing: bool = False,
    ):
        """
        保存相机标定文件

        Args:
            camera_intrinsics: {cam_id: (3, 3) 内参矩阵}
            camera_extrinsics: {cam_id: (4, 4) 外参矩阵}
            camera_configs: 相机配置列表
        """
        # intrinsics.json
        intrinsics_data = {}
        for cam_config in camera_configs:
            cam_id = f"cam_{cam_config['index']}"
            K = camera_intrinsics[cam_config['id']]

            intrinsics_data[cam_id] = {
                'fx': float(K[0, 0]),
                'fy': float(K[1, 1]),
                'cx': float(K[0, 2]),
                'cy': float(K[1, 2]),
                'width': 1280,
                'height': 960,
                # fov: 水平 FOV，仅作文档参考；fov_vertical 才是实际传给等距投影
                # 相机、并用于计算 fx/fy 的值（focal = (height/2)/(fov_vertical/2 弧度)）
                'fov': float(cam_config['fov']),
                'fov_vertical': float(cam_config['fov_vertical']),
                # 同位姿针孔深度相机的水平 FOV（大于 fov，覆盖鱼眼四角）。没有这个字段的旧数据
                # 深度相机 FOV 就是 'fov'
                'depth_fov': float(cam_config.get('depth_fov', cam_config['fov'])),
                # 等距投影 (equidistant, r=f*theta)，与 CARLA sensor.camera.rgb_fisheye
                # (camera_model=equidistant) 采集时用的投影模型一致，不是针孔模型，
                # 没有径向/切向畸变系数的概念
                # 与 camera_manager 实际传给传感器的镜头模型一致 (之前固定写 equidistant，配了
                # Kannala-Brandt 的相机标定会被错误解释)
                'distortion': {
                    'model': cam_config.get('lens_model', 'equidistant'),
                    'coeffs': [float(k) for k in cam_config.get('distortion_coeffs', (0.0, 0.0, 0.0, 0.0))],
                }
            }

        # 顶层字段（不是相机 ID，e2e_occ/dataset.py 按 cam_i 精确查找，不会被这个
        # 额外键干扰）：DNG 实际保存的位深。DNG 的 BitsPerSample EXIF 标签靠不住——
        # PIL 的 TIFF writer 按存储容器宽度（固定 uint16）覆盖它，不反映真实 ADC 位深，
        # 所以位深必须在这里单独记一份，供 dataset.py 加载 DNG 时做正确的归一化。
        intrinsics_data['raw_bit_depth'] = self.raw_bit_depth
        # 真值语义版本 (见 config/occupancy_config.py LABEL_VERSION)，dataset.py 据此拒绝混用
        intrinsics_data['label_version'] = LABEL_VERSION

        # 追加采集 (must_match_existing)：数据集里所有场景共用一份标定，新场景的相机配置必须与已有标定完全一致，
        # 否则旧图片会被新标定解释 (之前追加时只比 label_version/位深，然后直接覆盖公共标定)
        self._write_or_check_json('intrinsics.json', intrinsics_data, must_match_existing)

        # extrinsics.json
        extrinsics_data = {}
        for cam_config in camera_configs:
            cam_id = f"cam_{cam_config['index']}"
            T = camera_extrinsics[cam_config['id']]

            # 分解为平移和旋转
            translation = T[:3, 3].tolist()

            R = T[:3, :3]
            rotation_matrix = R.tolist()
            # camera_config 的 rotation 顺序是 (pitch, yaw, roll)，单位度
            pitch, yaw, roll = (float(a) for a in cam_config['rotation'])

            extrinsics_data[cam_id] = {
                'translation': [float(x) for x in translation],
                'rotation': {'pitch': pitch, 'yaw': yaw, 'roll': roll},
                'rotation_matrix': [[float(x) for x in row] for row in rotation_matrix]
            }

        # 约定标记（e2e_occ/dataset.py 据此区分新旧数据）：
        # rotation_convention='ue_frotator'：rotation_matrix 与 UE FRotationMatrix 一致（pitch 正值=抬头），
        #   即引擎实际渲染时的相机朝向。注意它和当前 LibCarla 客户端 carla.Transform.get_matrix()
        #   不同——上游 #9751 改了客户端的 pitch/roll 符号，但 UE 端 FRotator 没改。
        #   旧数据没有此字段，其 pitch/roll 符号与渲染相反，加载时自动修正。
        # camera_axes='ue'：矩阵的列是 UE 相机轴（x前 y右 z上），Camera→Vehicle
        extrinsics_data['rotation_convention'] = 'ue_frotator'
        extrinsics_data['camera_axes'] = 'ue'

        self._write_or_check_json('extrinsics.json', extrinsics_data, must_match_existing)

        print(f"  ✓ 标定文件已保存")

    def _write_or_check_json(self, name, data, must_match_existing):
        path = self.output_dir / 'calibration' / name
        if must_match_existing and path.exists():
            with open(path, encoding='utf-8') as f:
                old = json.load(f)
            new = json.loads(json.dumps(data))          # 与写盘后读回的形式一致 (tuple→list 等)
            diff = _json_diff(old, new)
            if diff:
                raise ValueError(f'{path} 与本次相机配置不一致，不能追加到同一个数据集 (换个输出目录)：'
                                 + '; '.join(diff[:5]))
            return
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_sample(
        self,
        sample_id: str,
        images: Dict[str, Union[np.ndarray, Dict]],
        occupancy: np.ndarray,
        flow: np.ndarray = None,
        flow_mask: np.ndarray = None,
        ego_pose: np.ndarray = None,
        ego_motion: np.ndarray = None,
        depth: Dict[str, np.ndarray] = None, # ⭐ 新增
        actor_ids: np.ndarray = None,  # 调试用，仅 save_actor_ids=True 时写盘
        mask_camera: np.ndarray = None,
        meta: Dict = None,
    ):
        """
        保存一个训练样本

        Args:
            sample_id: 样本ID (如 'scene_0001_frame_0000')
            images: {cam_id: data}
                    data可以是 np.ndarray (float16)
                    也可以是 {'data': np.ndarray, 'raw_type': str} (Bayer)
            occupancy: (400, 400, 32) uint8 语义标签
            flow: (3, 400, 400, 32) float16 流场 (可选)
            flow_mask: (400, 400, 32) uint8 流场掩码 (可选)
            ego_pose: (4, 4) float32 全局位姿 (可选)
            ego_motion: (4, 4) float32 帧间运动 (可选)
            depth: {cam_id: (H, W) float32} 深度图 (可选, meters)，存成 float16
            actor_ids: (400, 400, 32) int32，只在 self.save_actor_ids 为 True 时才写盘 (可选)
            mask_camera: (400, 400, 32) bool，相机观测到的体素 (可选，见 processing/camera_visibility.py)
        """
        # 写入协议 (2026-10-15)：每个文件先写 <name>.tmp 再 os.replace 原子改名 (进程被杀时不会留下截断的
        # 正式文件)；这一帧全部文件 (含后台线程写的 DNG/深度/流场) 都成功后，最后写完成标记
        # meta/<sid>.json，其中 files 列出每个文件的相对路径和字节数。sample_complete() 以完成标记为准逐个核对，
        # 必需的输出 (掩码、深度...) 由这一帧实际写了什么决定，而不是从目录里有没有同类文件推断。
        # 覆盖同名样本：先让旧完成标记失效，否则这一帧写到一半失败时，旧标记会把"新体素 + 旧图像"放行
        # (2026-10-15 评审复现：文件大小恰好一致)。旧帧被覆盖后也不再享受 legacy 兜底
        marker = self.output_dir / 'meta' / f'{sample_id}.json'
        try:
            marker.unlink()
        except FileNotFoundError:
            pass
        if sample_id in self._legacy:
            self._legacy.discard(sample_id)
            self._write_protocol()
        gen = self._marker_gen[sample_id] = self._marker_gen.get(sample_id, 0) + 1
        written = {}                       # 相对路径 -> 同步写完的字节数 (异步文件在完成标记任务里补)
        futures = []

        def rel(p):
            return p.relative_to(self.output_dir).as_posix()

        def sync_npy(path, arr):
            _atomic_write(path, lambda f: np.save(f, arr))
            written[rel(path)] = path.stat().st_size

        def async_write(path, fn):
            futures.append((rel(path), self._io_executor.submit(_atomic_write, path, fn)))

        # 1. 图像
        img_dir = self.output_dir / 'images' / sample_id
        img_dir.mkdir(parents=True, exist_ok=True)
        for cam_id, img_info in images.items():
            cam_index = self._get_cam_index(cam_id)
            if isinstance(img_info, dict) and 'raw_type' in img_info:
                data = img_info['data'].copy()  # 拷贝避免主线程修改
                if img_info['raw_type'] == 'bayer_rggb':
                    path = img_dir / f'cam_{cam_index}.dng'
                    async_write(path, lambda f, d=data: self._save_bayer_dng(d, f))
                else:
                    async_write(img_dir / f'cam_{cam_index}.npy', lambda f, d=data: np.save(f, d))
            else:
                sync_npy(img_dir / f'cam_{cam_index}.npy', img_info)      # 兼容旧格式 (直接传入 array)

        # 2. occupancy
        assert occupancy.shape == GRID_SIZE, f"occupancy形状错误: {occupancy.shape} (expected {GRID_SIZE})"
        sync_npy(self.output_dir / 'occupancy' / f'{sample_id}.npy', occupancy.astype(np.uint8))

        # 3. flow (只有动态物体所在体素非零，压缩后从 ~30MB 降到几十 KB)
        if flow is not None:
            expected_flow_shape = (3,) + GRID_SIZE
            assert flow.shape == expected_flow_shape, f"flow形状错误: {flow.shape} (expected {expected_flow_shape})"
            f16 = flow.astype(np.float16)
            async_write(self.output_dir / 'flow' / f'{sample_id}.npz', lambda f: np.savez_compressed(f, flow=f16))
        if flow_mask is not None:
            assert flow_mask.shape == GRID_SIZE, f"flow_mask形状错误: {flow_mask.shape} (expected {GRID_SIZE})"
            sync_npy(self.output_dir / 'flow_mask' / f'{sample_id}.npy', flow_mask.astype(np.uint8))

        # 4. ego
        if ego_pose is not None:
            assert ego_pose.shape == (4, 4), f"ego_pose形状错误: {ego_pose.shape}"
            sync_npy(self.output_dir / 'ego_pose' / f'{sample_id}.npy', ego_pose.astype(np.float32))
        if ego_motion is not None:
            assert ego_motion.shape == (4, 4), f"ego_motion形状错误: {ego_motion.shape}"
            sync_npy(self.output_dir / 'ego_motion' / f'{sample_id}.npy', ego_motion.astype(np.float32))

        # 5. 深度图：float16 (米)，40m 处量化步长 0.03m，对深度监督和可见性掩码 (0.1m 容差) 都够用
        if depth is not None:
            depth_dir = self.output_dir / 'depth' / sample_id
            depth_dir.mkdir(parents=True, exist_ok=True)
            for cam_id, depth_data in depth.items():
                d16 = depth_data.astype(np.float16)
                async_write(depth_dir / f'cam_{self._get_cam_index(cam_id)}.npy', lambda f, d=d16: np.save(f, d))

        # 6. actor_ids (可选，调试用)
        if self.save_actor_ids and actor_ids is not None:
            assert actor_ids.shape == GRID_SIZE, f"actor_ids形状错误: {actor_ids.shape} (expected {GRID_SIZE})"
            sync_npy(self.output_dir / 'debug_actor_ids' / f'{sample_id}.npy', actor_ids.astype(np.int32))

        if mask_camera is not None:
            assert mask_camera.shape == GRID_SIZE, f"mask_camera形状错误: {mask_camera.shape}"
            path = self.output_dir / 'mask_camera' / f'{sample_id}.npy'
            _atomic_write(path, lambda f: save_mask(f, mask_camera))
            written[rel(path)] = path.stat().st_size

        # 7. 完成标记：等这一帧的后台写入全部成功后再写 (单线程，不占 IO 线程池)
        meta = dict(meta or {})

        def finish():
            if self._marker_gen.get(sample_id) != gen:                   # 同名样本又被覆盖了：这份标记已过期
                return False
            files = dict(written)
            for name, fut in futures:
                try:
                    fut.result()
                except Exception as e:                                   # 任何一个失败：不写完成标记
                    print(f"  [错误] {sample_id} 写 {name} 失败: {e}；该帧不会进入划分")
                    return False
                files[name] = (self.output_dir / name).stat().st_size
            meta['files'] = files
            _atomic_write(marker, lambda f: f.write(json.dumps(meta, ensure_ascii=False).encode('utf-8')))
            return True

        self._pending_futures.append(self._marker_executor.submit(finish))

        # 记录sample_id
        self.sample_ids.append(sample_id)

    def _save_bayer_dng(self, bayer_data, fileobj):
        """Bayer RGGB 数据写成 DNG 到 fileobj (由 _atomic_write 提供临时文件)；失败直接抛出，该帧不写完成标记"""
        # 转换为 uint16
        bayer_u16 = bayer_data.astype(np.uint16)

        # 相机管线内部始终以 16-bit (0-65535) 传输，这里右移到目标位深模拟
        # 真实传感器的 ADC 精度，例如 12-bit: [0, 65535] → [0, 4095]。
        shift = 16 - self.raw_bit_depth
        # 四舍五入而不是直接右移截断（截断会让整幅图系统性偏暗约 0.5 LSB），
        # 并限幅到目标位深的最大值，避免 65535 舍入后溢出到 2^bits。
        if shift > 0:
            max_val = (1 << self.raw_bit_depth) - 1
            bayer_quantized = np.minimum(
                (bayer_u16.astype(np.uint32) + (1 << (shift - 1))) >> shift, max_val
            ).astype(np.uint16)
        else:
            bayer_quantized = bayer_u16

        # Pillow 写单通道 16-bit TIFF（DNG 兼容）；Pillow 缺失时 __init__ 已报错
        # DNG = 带 DNG 标签的 TIFF。标签必须走 tiffinfo 直接写进 IFD0：以前用 piexif 构造
        # exif 传给 PIL，那些标签根本没进 IFD，文件里没有 CFAPattern/WhiteLevel/DNGVersion，
        # rawpy 读出来 white_level=65535、Bayer 排列靠猜。BitsPerSample 仍是容器宽度 16
        # (PIL 不可改)，真实位深看 WhiteLevel / intrinsics.json 的 raw_bit_depth。
        ifd = TiffImagePlugin.ImageFileDirectory_v2()
        for tag, val, typ in (
            (262, 32803, 3),                              # PhotometricInterpretation = CFA
            (271, 'CARLA Simulator', 2),                  # Make
            (272, 'Bayer RGGB Camera', 2),                # Model
            (305, 'OccNetV3 Data Generator', 2),          # Software
            (50706, b'\x01\x04\x00\x00', 1),             # DNGVersion 1.4
            (50707, b'\x01\x01\x00\x00', 1),             # DNGBackwardVersion 1.1
            (50708, 'CARLA Bayer RGGB', 2),               # UniqueCameraModel
            (33421, (2, 2), 3),                           # CFARepeatPatternDim
            (33422, b'\x00\x01\x01\x02', 1),             # CFAPattern = R G / G B
            (50714, 0, 4),                                # BlackLevel (仿真无暗电流偏置)
            (50717, (1 << self.raw_bit_depth) - 1, 4),    # WhiteLevel = 2^bits - 1
        ):
            ifd[tag] = val
            ifd.tagtype[tag] = typ
        Image.fromarray(bayer_quantized).save(fileobj, format='TIFF', compression='raw', tiffinfo=ifd)

    def _generate_png_thumbnail(self, bayer_data: np.ndarray, dng_path: Path):
        """
        从 Bayer RAW 数据生成 PNG 缩略图

        Args:
            bayer_data: (H, W) uint16 Bayer RGGB 数据
            dng_path: DNG 文件路径 (用于生成 PNG 路径)
        """
        try:
            # PNG 路径: cam_0.dng → cam_0.png
            png_path = dng_path.with_suffix('.png')

            # 简单 Bayer → RGB 转换
            if CV2_AVAILABLE:
                # 方法1: 使用 OpenCV (更快)
                bayer_u16 = bayer_data.astype(np.uint16)
                rgb = cv2.cvtColor(bayer_u16, cv2.COLOR_BAYER_RGGB2RGB)

                # 归一化到 8-bit
                max_val = np.max(rgb)
                if max_val > 0:
                    rgb_u8 = (rgb / max_val * 255).astype(np.uint8)
                else:
                    rgb_u8 = rgb.astype(np.uint8)

                # 缩放到 640×480 (原始 1280×960 的一半)
                rgb_small = cv2.resize(rgb_u8, (640, 480), interpolation=cv2.INTER_AREA)

                # 保存 PNG
                cv2.imwrite(str(png_path), cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR), [
                    cv2.IMWRITE_PNG_COMPRESSION, 3  # 压缩等级 3 (快速)
                ])

            elif DNG_AVAILABLE:
                # 方法2: 使用 PIL (降级方案)
                # 直接将 Bayer 数据归一化为灰度图
                bayer_u16 = bayer_data.astype(np.uint16)
                max_val = np.max(bayer_u16)
                if max_val > 0:
                    gray_u8 = (bayer_u16 / max_val * 255).astype(np.uint8)
                else:
                    gray_u8 = bayer_u16.astype(np.uint8)

                # 创建 PIL 图像并缩放
                img_pil = Image.fromarray(gray_u8, mode='L')
                img_small = img_pil.resize((640, 480), Image.Resampling.LANCZOS)

                # 保存 PNG
                img_small.save(str(png_path), format='PNG', compress_level=3)

            else:
                # 无可用库，跳过缩略图生成
                return

        except Exception as e:
            # 缩略图生成失败不影响主流程
            print(f"  [警告] PNG 缩略图生成失败: {e}")

    def _get_cam_index(self, cam_id: str) -> int:
        """从相机ID提取索引"""
        mapping = {
            'front_main': 0,
            'front_wide': 1,
            'front_narrow': 2,
            'left_pillar': 3,
            'right_pillar': 4,
            'left_repeater': 5,
            'right_repeater': 6,
            'rear': 7,
        }
        return mapping.get(cam_id, 0)

    def generate_sample_id(self) -> str:
        """生成sample_id"""
        sample_id = f"{self.scene_name}_{self.scene_counter:04d}_frame_{self.frame_counter:04d}"
        self.frame_counter += 1
        return sample_id

    def next_scene(self):
        """切换到下一个场景"""
        self.scene_counter += 1
        self.frame_counter = 0
        print(f"[OccNetDataSaver] 切换到场景 {self.scene_counter}")

    def save_scene_meta(self, meta: Dict):
        """保存本场景的元数据 (地图/天气/种子/抽帧间隔等) 到 scenes/<scene_id>.json"""
        scene_id = f"{self.scene_name}_{self.scene_counter:04d}"
        meta = dict(meta, scene_id=scene_id, label_version=LABEL_VERSION, raw_bit_depth=self.raw_bit_depth)
        with open(self.output_dir / 'scenes' / f'{scene_id}.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def sample_complete(self, sample_id: str, num_cameras: int = 8) -> bool:
        """一个样本是否完整写完 (与 e2e_occ/dataset.py 的严格校验对应)。

        新数据 (2026-10-15 起)：以完成标记 meta/<sid>.json 为准——它在这一帧所有文件都原子写完后才写，
        files 字段记录每个文件的字节数，逐个核对存在且大小一致 (零字节/截断/被删都能识别)。
        旧数据没有完成标记 (或标记里没有 files)：退化为检查必需文件存在且非空。"""
        d = self.output_dir
        files = _marker_files(d, sample_id)
        if files:
            for name, size in files.items():
                p = d / name
                if not p.exists() or p.stat().st_size != size or size == 0:
                    return False
            return f'occupancy/{sample_id}.npy' in files
        legacy = self._legacy_samples()
        if legacy is not None and sample_id not in legacy:     # 新协议写过的目录：缺完成标记 = 不完整
            return False
        files = [d / 'occupancy' / f'{sample_id}.npy', d / 'ego_pose' / f'{sample_id}.npy',
                 d / 'flow_mask' / f'{sample_id}.npy']
        files += [d / 'images' / sample_id / f'cam_{i}.dng' for i in range(num_cameras)]
        files += [d / 'depth' / sample_id / f'cam_{i}.npy' for i in range(num_cameras)]
        if getattr(self, '_has_masks', None) is None:      # 关了相机掩码 (--no-camera-mask) 的数据集不要求
            self._has_masks = any((d / 'mask_camera').glob('*.npy'))
        if self._has_masks:
            files.append(d / 'mask_camera' / f'{sample_id}.npy')
        flow = [p for p in (d / 'flow' / f'{sample_id}.npz', d / 'flow' / f'{sample_id}.npy') if p.exists()]
        return bool(flow) and flow[0].stat().st_size > 0 and all(f.exists() and f.stat().st_size > 0 for f in files)

    def _legacy_samples(self):
        """写入协议文件里的旧帧名单；目录从未被新代码写过 (没有协议文件) 时返回 None。
        batch_collection 用 __new__ 构造只设了 output_dir 的检查器，这里按需从磁盘读。"""
        if getattr(self, '_legacy', None) is not None:
            return self._legacy
        try:
            with open(self.output_dir / PROTOCOL_FILE, encoding='utf-8') as f:
                self._legacy = set(json.load(f).get('legacy_samples', []))
        except FileNotFoundError:
            return None
        return self._legacy

    def _write_protocol(self):
        body = {'marker_protocol': 1, 'legacy_samples': sorted(self._legacy)}
        _atomic_write(self.output_dir / PROTOCOL_FILE,
                      lambda f: f.write(json.dumps(body, ensure_ascii=False, indent=1).encode('utf-8')))

    @staticmethod
    def scene_split(scene_id: str, train_ratio=0.8, val_ratio=0.1) -> str:
        """按场景名的哈希确定性地分 train/val/test：追加新场景不会改变已有场景的归属，
        同一场景的帧 (相邻帧几乎重复) 永远在同一个划分里，不会跨 train/val 泄漏。"""
        h = int(hashlib.md5(scene_id.encode('utf-8')).hexdigest()[:8], 16) / 0xFFFFFFFF
        return 'train' if h < train_ratio else ('val' if h < train_ratio + val_ratio else 'test')

    def finalize(self, train_ratio=0.8, val_ratio=0.1):
        """
        结束采集,生成数据集划分文件

        划分覆盖输出目录里**所有**场景 (occupancy/ 下的全部样本，不只本次运行采的)，所以多次
        --append 采集到同一个数据集后，最后一次 finalize 写出的 train/val/test.txt 就是完整的。

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        # 等待所有后台IO完成
        if self._pending_futures:
            print(f"[OccNetDataSaver] 等待 {len(self._pending_futures)} 个后台IO任务完成...")
            for fut in self._pending_futures:
                try:
                    fut.result()
                except Exception as e:
                    print(f"  [警告] 后台IO任务失败: {e}")
            self._pending_futures.clear()
        self._io_executor.shutdown(wait=False)
        self._marker_executor.shutdown(wait=True)

        all_ids = sorted(p.stem for p in (self.output_dir / 'occupancy').glob('*.npy'))
        # 只把文件齐全的样本写进划分：采集进程被杀/崩溃时，最后几帧可能只写了一部分 (体素已存、DNG 还在
        # 后台线程里)。e2e_occ/dataset.py 默认严格校验，残缺样本进了 train.txt 会让训练直接报错
        complete = [s for s in all_ids if self.sample_complete(s)]
        if len(complete) < len(all_ids):
            bad = sorted(set(all_ids) - set(complete))
            print(f"[OccNetDataSaver] ⚠️ {len(bad)} 个样本文件不全，不写入划分文件 (例如 {bad[:3]})")
        all_ids = complete
        total = len(all_ids)
        if total == 0:
            print(f"[OccNetDataSaver] 警告: 没有数据")
            return

        # 按场景分组
        scenes = {}
        for sample_id in all_ids:
            scenes.setdefault(sample_id.split('_frame_')[0], []).append(sample_id)
        scene_list = sorted(scenes)

        n_scenes = len(scene_list)
        # 划分持久化在 splits.json：已经分过的场景永远不再移动 (之前每次 finalize 重新计算，为了"保证验证集
        # 非空"会把老场景挪来挪去，追加采集后验证集悄悄换了一批)。新场景按场景名哈希分，只在新场景里补位。
        split_path = self.output_dir / 'splits.json'
        record = {}
        if split_path.exists():
            with open(split_path, encoding='utf-8') as f:
                record = json.load(f).get('scenes', {})
        new = [sc for sc in scene_list if sc not in record]
        for sc in new:
            record[sc] = self.scene_split(sc, train_ratio, val_ratio)
        assigned = {k: v for k, v in record.items() if k in scenes}
        if new and 'train' not in assigned.values():
            record[new[0]] = assigned[new[0]] = 'train'
        if new and 'val' not in assigned.values():
            spare = [sc for sc in new if assigned[sc] == 'train']
            if len(spare) > 1 or (spare and list(assigned.values()).count('train') > 1):
                record[spare[-1]] = assigned[spare[-1]] = 'val'
        train_scenes = [sc for sc in scene_list if assigned[sc] == 'train']
        val_scenes = [sc for sc in scene_list if assigned[sc] == 'val']
        test_scenes = [sc for sc in scene_list if assigned[sc] == 'test']
        debug_shared = False
        if not val_scenes:
            # 只有一个场景等调试情况：验证集复用训练场景，在 splits.json 里显式标记，正式指标不能用它
            print("[OccNetDataSaver] ⚠️ 没有独立的验证场景，验证集复用训练场景 (仅供调试，指标无参考价值)")
            val_scenes = list(train_scenes)
            debug_shared = True
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump({'scenes': record, 'debug_val_reuses_train': debug_shared}, f, indent=2, ensure_ascii=False)

        # 展开为样本列表
        train_samples = [s for sc in train_scenes for s in scenes[sc]]
        val_samples = [s for sc in val_scenes for s in scenes[sc]]
        test_samples = [s for sc in test_scenes for s in scenes[sc]]

        # 保存文件
        with open(self.output_dir / 'train.txt', 'w') as f:
            f.write('\n'.join(train_samples))

        with open(self.output_dir / 'val.txt', 'w') as f:
            f.write('\n'.join(val_samples))

        with open(self.output_dir / 'test.txt', 'w') as f:
            f.write('\n'.join(test_samples))

        print(f"\n[OccNetDataSaver] 数据集划分完成:")
        print(f"  训练集: {len(train_samples)} 样本")
        print(f"  验证集: {len(val_samples)} 样本")
        print(f"  测试集: {len(test_samples)} 样本")
        print(f"  总计: {total} 样本 / {n_scenes} 个场景 (train {len(train_scenes)} / val {len(val_scenes)} / "
              f"test {len(test_scenes)})")

import os
import json
import argparse
from pathlib import Path
import numpy as np
from flask import Flask, jsonify, send_file, request, abort, Response
import io
from PIL import Image
import logging
import time
import sys
from functools import lru_cache

app = Flask(__name__)

# 配置详细的 HTTP 日志 (类似 occupancy_viewer 风格)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger('DatasetViewer')

# 禁用 Werkzeug 默认日志，使用自定义格式
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)

@app.before_request
def before_request():
    request.start_time = time.time()
    logger.info(f"→ {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def after_request(response):
    latency = (time.time() - request.start_time) * 1000  # ms
    status_icon = "✓" if 200 <= response.status_code < 300 else "✗"
    logger.info(f"← {status_icon} {request.method} {request.path} - {response.status_code} ({latency:.1f}ms)")
    return response

# LRU Cache for processed images (Max 32 images ~ 4 frames of 8 cams)
@lru_cache(maxsize=32)
def process_dng_cached(dng_path_str, half_size):
    """
    缓存的 DNG 处理函数
    输入必须是字符串(hashable)，不能是 Path 对象
    half_size=True: 1/4 分辨率缩略图 (预览网格用)；False: 全分辨率 (lightbox 大图用)
    """
    import rawpy
    with rawpy.imread(dng_path_str) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=half_size, no_auto_bright=False)
        return rgb

def create_placeholder_image(text="Error", size=(640, 480)):
    img = Image.new('RGB', size, color=(30, 30, 30))
    return img

# 默认配置
DEFAULT_DATASET_DIR = r"d:\code\carla\dataset_10k_bak"
CURRENT_DATASET_DIR = DEFAULT_DATASET_DIR
CURRENT_PREDICTION_DIR = None  # 推理结果目录 (e2e_occ/inference.py 输出)，可选

# 缓存
FRAMES_CACHE = []
PRED_FRAMES_CACHE = None  # None = 未设置预测目录; set() = 预测目录里实际有 occupancy 的帧号集合

# ------------------------------------------------------------------
# 帧列表 / 数据集元信息
# ------------------------------------------------------------------

def get_frames():
    """获取数据集中的帧列表"""
    global FRAMES_CACHE
    if FRAMES_CACHE:
        return FRAMES_CACHE

    dataset_path = Path(CURRENT_DATASET_DIR)
    if not dataset_path.exists():
        return []

    frames = []
    for txt_file in ['test.txt', 'train.txt', 'val.txt']:
        txt_path = dataset_path / txt_file
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                frames.extend(lines)

    if not frames:
        occ_dir = dataset_path / 'occupancy'
        if occ_dir.exists():
            frames = [f.stem for f in occ_dir.glob('*.npy')]
            frames.sort()

    frames = sorted(list(set(frames)))
    FRAMES_CACHE = frames
    return frames


def get_prediction_frames():
    """获取当前预测目录里实际存在 occupancy 结果的帧号集合"""
    global PRED_FRAMES_CACHE
    if CURRENT_PREDICTION_DIR is None:
        return None
    if PRED_FRAMES_CACHE is not None:
        return PRED_FRAMES_CACHE

    pred_occ_dir = Path(CURRENT_PREDICTION_DIR) / 'occupancy'
    if not pred_occ_dir.exists():
        PRED_FRAMES_CACHE = set()
    else:
        PRED_FRAMES_CACHE = {f.stem for f in pred_occ_dir.glob('*.npy')}
    return PRED_FRAMES_CACHE


@app.route('/')
def index():
    return send_file('templates/index.html')


@app.route('/api/dataset_info')
def dataset_info():
    frames = get_frames()
    dataset_path = Path(CURRENT_DATASET_DIR)
    pred_frames = get_prediction_frames()

    return jsonify({
        'path': CURRENT_DATASET_DIR,
        'count': len(frames),
        'frames': frames,
        'has_depth': (dataset_path / 'depth').exists(),
        'has_ego_pose': (dataset_path / 'ego_pose').exists(),
        'has_calibration': (dataset_path / 'calibration' / 'intrinsics.json').exists(),
        'prediction_path': CURRENT_PREDICTION_DIR,
        'prediction_count': len(pred_frames) if pred_frames is not None else 0,
    })


@app.route('/api/set_dataset', methods=['POST'])
def set_dataset():
    global CURRENT_DATASET_DIR, FRAMES_CACHE
    data = request.json
    new_path = data.get('path')
    if new_path and os.path.exists(new_path):
        CURRENT_DATASET_DIR = new_path
        FRAMES_CACHE = []
        return jsonify({'success': True, 'count': len(get_frames())})
    return jsonify({'success': False, 'message': 'Path does not exist'})


@app.route('/api/set_prediction', methods=['POST'])
def set_prediction():
    """设置推理结果目录 (e2e_occ/inference.py 输出格式：<dir>/occupancy/<id>.npy)。
    传空字符串/null 表示清除。"""
    global CURRENT_PREDICTION_DIR, PRED_FRAMES_CACHE
    data = request.json or {}
    new_path = data.get('path')

    if not new_path:
        CURRENT_PREDICTION_DIR = None
        PRED_FRAMES_CACHE = None
        return jsonify({'success': True, 'count': 0})

    if not os.path.exists(new_path):
        return jsonify({'success': False, 'message': 'Path does not exist'})

    CURRENT_PREDICTION_DIR = new_path
    PRED_FRAMES_CACHE = None
    frames = get_prediction_frames()
    return jsonify({'success': True, 'count': len(frames)})


# ------------------------------------------------------------------
# 目录浏览 (给"数据集"/"推理结果"输入框配一个可点选的目录选择器，
# 不用手动拷贝粘贴绝对路径)
# ------------------------------------------------------------------

DATASET_MARKER_DIRS = ('occupancy', 'images')  # 出现任一个就判定为"看起来像数据集目录"


def _looks_like_dataset(dir_path):
    try:
        return any((dir_path / marker).is_dir() for marker in DATASET_MARKER_DIRS)
    except OSError:
        return False


def _list_windows_drives():
    import string
    drives = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root):
            drives.append({'name': root, 'path': root, 'is_dataset': False})
    return drives


@app.route('/api/browse_dir')
def browse_dir():
    """
    列出给定路径下的子目录，供前端目录选择器使用。
    不传 path (或传空) 时，Windows 下返回盘符列表作为根；非 Windows 返回 '/'。
    每个子目录附带 is_dataset (是否含 occupancy/ 或 images/ 子目录)，前端用它高亮"这是个数据集"。
    """
    raw_path = request.args.get('path', '').strip()

    if not raw_path:
        if os.name == 'nt':
            return jsonify({'path': None, 'parent': None, 'entries': _list_windows_drives()})
        raw_path = '/'

    target = Path(raw_path)
    if not target.exists() or not target.is_dir():
        return jsonify({'error': 'Path does not exist or is not a directory'}), 400

    try:
        subdirs = [p for p in target.iterdir() if p.is_dir() and not p.name.startswith('.')]
    except PermissionError:
        return jsonify({'error': 'Permission denied'}), 403

    entries = sorted(
        [{'name': p.name, 'path': str(p), 'is_dataset': _looks_like_dataset(p)} for p in subdirs],
        key=lambda e: e['name'].lower()
    )

    parent = str(target.parent) if target.parent != target else None
    # Windows 盘符根 (如 D:\) 的 .parent 还是它自己，用上面这个判断已经处理了；
    # 非根目录正常返回上一级路径即可，前端遇到 parent=None 就回退到盘符列表。
    if os.name == 'nt' and len(str(target)) <= 3:  # "D:\" / "D:/" 这类根
        parent = None

    return jsonify({
        'path': str(target),
        'parent': parent,
        'entries': entries,
        'is_dataset': _looks_like_dataset(target),
    })


# ------------------------------------------------------------------
# 相机图像 (RGB, 缩略图/高清)
# ------------------------------------------------------------------

@app.route('/api/image/<frame_id>/<int:cam_idx>')
def get_image(frame_id, cam_idx):
    """获取指定帧和相机的图像 (优先 PNG > DNG > NPY)。?hires=1 返回全分辨率(跳过缓存)。"""
    dataset_path = Path(CURRENT_DATASET_DIR)
    img_dir = dataset_path / 'images' / frame_id
    hires = request.args.get('hires') == '1'

    if not hires:
        png_path = img_dir / f"cam_{cam_idx}.png"
        if png_path.exists():
            try:
                return send_file(str(png_path), mimetype='image/png')
            except Exception as e:
                logger.warning(f"PNG loading failed: {e}, falling back to DNG")

    dng_path = img_dir / f"cam_{cam_idx}.dng"

    if dng_path.exists():
        try:
            if hires:
                # 高清大图：不落盘缓存 (点开 lightbox 才会请求，量不大)，直接实时解码
                import rawpy
                with rawpy.imread(str(dng_path)) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False)
                img_pil = Image.fromarray(rgb)
                img_io = io.BytesIO()
                img_pil.save(img_io, 'JPEG', quality=92)
                img_io.seek(0)
                return send_file(img_io, mimetype='image/jpeg')

            # 缩略图路径：DNG-to-PNG 持久化缓存
            cache_dir = dataset_path / '.png_cache' / 'images' / frame_id
            cache_png = cache_dir / f"cam_{cam_idx}.png"

            use_cache = False
            if cache_png.exists():
                dng_mtime = dng_path.stat().st_mtime
                png_mtime = cache_png.stat().st_mtime
                if png_mtime >= dng_mtime:
                    use_cache = True

            if use_cache:
                return send_file(str(cache_png), mimetype='image/png')

            try:
                import rawpy
                rgb = process_dng_cached(str(dng_path), True)
                if rgb is None:
                    process_dng_cached.cache_clear()
                    raise Exception("Cached processing failed")
            except ImportError as e:
                logger.warning(f"rawpy not available: {e}, trying cv2")
                import cv2
                img = cv2.imread(str(dng_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    logger.error(f"cv2 failed to load DNG: {dng_path}")
                    return abort(404, description="Failed to load DNG with cv2")
                rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_RGGB2RGB)
                if rgb.dtype == np.uint16:
                    max_val = np.max(rgb)
                    rgb = (rgb / max_val * 255).astype(np.uint8) if max_val > 0 else rgb.astype(np.uint8)

            img_pil = Image.fromarray(rgb)
            cache_dir.mkdir(parents=True, exist_ok=True)
            img_pil.save(str(cache_png), 'PNG', optimize=True)
            return send_file(str(cache_png), mimetype='image/png')

        except Exception as e:
            logger.error(f"❌ Error loading DNG {dng_path.name}: {type(e).__name__}: {e}")
            img = create_placeholder_image(text=f"Error: {dng_path.name}")
            img_io = io.BytesIO()
            img.save(img_io, 'JPEG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')

    # NPY 兜底 (灰度)
    npy_path = img_dir / f"cam_{cam_idx}.npy"
    if not npy_path.exists():
        return abort(404, description="Image not found")

    try:
        data = np.load(npy_path)
        img_data = data[0] if data.ndim == 3 and data.shape[0] == 1 else data
        if not np.isfinite(img_data).all():
            img_data = np.nan_to_num(img_data)
        img_data = np.clip(img_data, 0, 1)
        img_uint8 = (img_data * 255).astype(np.uint8)
        img = Image.fromarray(img_uint8, mode='L')
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error loading NPY {npy_path}: {e}")
        return abort(500)


# ------------------------------------------------------------------
# 深度图 (colorized)
# ------------------------------------------------------------------

@app.route('/api/depth/<frame_id>/<int:cam_idx>')
def get_depth(frame_id, cam_idx):
    """深度图上色预览。depth/*.npy 是 (H,W) float32，单位米。?max_depth=80 可调裁剪范围。"""
    dataset_path = Path(CURRENT_DATASET_DIR)
    depth_path = dataset_path / 'depth' / frame_id / f"cam_{cam_idx}.npy"
    if not depth_path.exists():
        return abort(404, description="Depth not found")

    try:
        max_depth = float(request.args.get('max_depth', 80.0))
    except ValueError:
        max_depth = 80.0

    cache_dir = dataset_path / '.png_cache' / 'depth' / frame_id
    cache_png = cache_dir / f"cam_{cam_idx}_{int(max_depth)}.png"

    if cache_png.exists() and cache_png.stat().st_mtime >= depth_path.stat().st_mtime:
        return send_file(str(cache_png), mimetype='image/png')

    try:
        import cv2
        depth = np.load(depth_path).astype(np.float32)
        depth = np.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0)
        depth_clipped = np.clip(depth, 0.0, max_depth)
        depth_u8 = (depth_clipped / max_depth * 255.0).astype(np.uint8)
        colormap = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)
        colored_bgr = cv2.applyColorMap(depth_u8, colormap)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(colored_rgb)
        cache_dir.mkdir(parents=True, exist_ok=True)
        img_pil.save(str(cache_png), 'PNG', optimize=True)
        return send_file(str(cache_png), mimetype='image/png')
    except Exception as e:
        logger.error(f"Error rendering depth {depth_path}: {e}")
        return abort(500)


# ------------------------------------------------------------------
# 体素 (二进制传输)
# ------------------------------------------------------------------

def _load_occupancy_grid(frame_id, source):
    """source: 'gt' | 'pred' -> (400,400,32) uint8 ndarray，找不到返回 None"""
    if source == 'pred':
        if CURRENT_PREDICTION_DIR is None:
            return None
        path = Path(CURRENT_PREDICTION_DIR) / 'occupancy' / f"{frame_id}.npy"
    else:
        path = Path(CURRENT_DATASET_DIR) / 'occupancy' / f"{frame_id}.npy"
    if not path.exists():
        return None
    return np.load(path)


def _pack_voxels(indices, labels):
    """
    indices: [N,3] int, labels: [N] uint8 -> 6 字节/体素，但按列拼接 (不是逐体素交织)：
    [x u16 * N][y u16 * N][z u8 * N][label u8 * N]。
    列式布局是为了让前端能用 `new Uint16Array(buf, offset, N)` 直接在 ArrayBuffer 上开视图
    零拷贝解码，不用逐体素跑一遍 DataView 循环——真实数据一帧几百万体素时这个循环本身就是
    卡顿来源之一。
    """
    n = len(indices)
    if n == 0:
        return b''
    return (indices[:, 0].astype('<u2').tobytes()
            + indices[:, 1].astype('<u2').tobytes()
            + indices[:, 2].astype('u1').tobytes()
            + labels.astype('u1').tobytes())


def _z_mask(indices, z_min, z_max):
    if z_min is None and z_max is None:
        return None
    z = indices[:, 2]
    mask = np.ones(len(indices), dtype=bool)
    if z_min is not None:
        mask &= (z >= z_min)
    if z_max is not None:
        mask &= (z <= z_max)
    return mask


# 默认不限制体素数量——之前默认限流到 35 万会对整块的地面/建筑做等步长跨步采样，
# np.argwhere 按 (x,y,z) 行优先顺序排列，跨步采样会周期性地漏采某些 x/y 列，肉眼看就是
# "实心块变成一条条离散栅栏"的走样条纹，不是数据问题，是这个降采样策略本身有问题。
# 配合二进制列式布局 (零拷贝解码) + 直接写 InstancedMesh 缓冲区 (跳过 Object3D)，
# 全量数据本身已经能流畅渲染，不再需要默认限流。max_voxels 参数保留，只在显式传入时生效
# (比如以后要给低配设备加一个"性能模式"开关)。
DEFAULT_MAX_VOXELS = None


def _decimate(indices, labels_tuple, max_count):
    """均匀跨步降采样。labels_tuple 是要跟着 indices 一起降采样的若干 label 数组。
    返回 (indices, labels_tuple, total_before_decimate)。"""
    total = len(indices)
    if max_count is None or total <= max_count:
        return indices, labels_tuple, total
    stride = int(np.ceil(total / max_count))
    sl = slice(None, None, stride)
    return indices[sl], tuple(a[sl] for a in labels_tuple), total


@app.route('/api/occupancy/<frame_id>')
def get_occupancy(frame_id):
    """
    体素二进制流。查询参数：
      source=gt|pred (默认 gt)
      include_free=0|1 (默认 0，排除 label==0)
      z_min, z_max (可选，按 Z 层裁切，闭区间)
      max_voxels (可选，默认 DEFAULT_MAX_VOXELS，超过则均匀降采样；传 0 表示不限制)
    响应头 X-Voxel-Count 是实际发送的体素数 (body = count*6 字节)，
    X-Voxel-Total 是降采样前的真实体素数。
    """
    source = request.args.get('source', 'gt')
    include_free = request.args.get('include_free', '0') == '1'
    z_min = request.args.get('z_min', type=int)
    z_max = request.args.get('z_max', type=int)
    max_voxels = request.args.get('max_voxels', default=DEFAULT_MAX_VOXELS, type=int)
    if max_voxels == 0:
        max_voxels = None

    grid = _load_occupancy_grid(frame_id, source)
    if grid is None:
        return abort(404, description=f"Occupancy not found (source={source})")

    indices = np.argwhere(grid >= 0) if include_free else np.argwhere(grid != 0)
    if len(indices) == 0:
        resp = Response(b'', mimetype='application/octet-stream')
        resp.headers['X-Voxel-Count'] = '0'
        resp.headers['X-Voxel-Total'] = '0'
        return resp

    labels = grid[indices[:, 0], indices[:, 1], indices[:, 2]]

    mask = _z_mask(indices, z_min, z_max)
    if mask is not None:
        indices = indices[mask]
        labels = labels[mask]

    indices, (labels,), total = _decimate(indices, (labels,), max_voxels)

    body = _pack_voxels(indices, labels)
    resp = Response(body, mimetype='application/octet-stream')
    resp.headers['X-Voxel-Count'] = str(len(indices))
    resp.headers['X-Voxel-Total'] = str(total)
    resp.headers['X-Grid-Shape'] = ','.join(str(s) for s in grid.shape)
    return resp


@app.route('/api/occupancy_diff/<frame_id>')
def get_occupancy_diff(frame_id):
    """
    GT vs Pred 逐体素比较，只返回不一致的体素。
    8 字节/体素: u16 x, u16 y, u8 z, u8 gt_label, u8 pred_label, u8 category
    category: 0=confusion(两边都非空但类别不同) 1=miss(GT有Pred空) 2=false_positive(GT空Pred有)
    """
    z_min = request.args.get('z_min', type=int)
    z_max = request.args.get('z_max', type=int)
    max_voxels = request.args.get('max_voxels', default=DEFAULT_MAX_VOXELS, type=int)
    if max_voxels == 0:
        max_voxels = None

    gt = _load_occupancy_grid(frame_id, 'gt')
    pred = _load_occupancy_grid(frame_id, 'pred')
    if gt is None or pred is None:
        return abort(404, description="GT or Pred occupancy not found")
    if gt.shape != pred.shape:
        return abort(500, description="GT/Pred shape mismatch")

    mismatch = gt != pred
    indices = np.argwhere(mismatch)
    if len(indices) == 0:
        resp = Response(b'', mimetype='application/octet-stream')
        resp.headers['X-Voxel-Count'] = '0'
        resp.headers['X-Voxel-Total'] = '0'
        return resp

    gt_labels = gt[indices[:, 0], indices[:, 1], indices[:, 2]]
    pred_labels = pred[indices[:, 0], indices[:, 1], indices[:, 2]]

    mask = _z_mask(indices, z_min, z_max)
    if mask is not None:
        indices = indices[mask]
        gt_labels = gt_labels[mask]
        pred_labels = pred_labels[mask]

    indices, (gt_labels, pred_labels), total = _decimate(indices, (gt_labels, pred_labels), max_voxels)

    category = np.zeros(len(indices), dtype=np.uint8)
    category[(gt_labels == 0) & (pred_labels != 0)] = 2   # false_positive
    category[(gt_labels != 0) & (pred_labels == 0)] = 1   # miss
    # 其余 (两边都非空但不同类) 保持默认 0 = confusion

    # 列式布局 (同 _pack_voxels 的理由): x,y,z,gt,pred,cat 各自连续存放，前端零拷贝开视图
    n = len(indices)
    body = (indices[:, 0].astype('<u2').tobytes()
            + indices[:, 1].astype('<u2').tobytes()
            + indices[:, 2].astype('u1').tobytes()
            + gt_labels.astype('u1').tobytes()
            + pred_labels.astype('u1').tobytes()
            + category.astype('u1').tobytes())

    resp = Response(body, mimetype='application/octet-stream')
    resp.headers['X-Voxel-Count'] = str(n)
    resp.headers['X-Voxel-Total'] = str(total)
    return resp


@app.route('/api/occupancy_diff_summary/<frame_id>')
def get_occupancy_diff_summary(frame_id):
    """GT vs Pred 整帧质量摘要 (JSON，轻量，用于控制面板展示一行统计)"""
    gt = _load_occupancy_grid(frame_id, 'gt')
    pred = _load_occupancy_grid(frame_id, 'pred')
    if gt is None or pred is None:
        return abort(404, description="GT or Pred occupancy not found")
    if gt.shape != pred.shape:
        return abort(500, description="GT/Pred shape mismatch")

    total = gt.size
    exact_match = int(np.count_nonzero(gt == pred))
    occ_gt = gt != 0
    occ_pred = pred != 0
    intersection = int(np.count_nonzero(occ_gt & occ_pred))
    union = int(np.count_nonzero(occ_gt | occ_pred))

    miss = int(np.count_nonzero(occ_gt & ~occ_pred))
    false_positive = int(np.count_nonzero(~occ_gt & occ_pred))
    confusion = int(np.count_nonzero(occ_gt & occ_pred & (gt != pred)))

    return jsonify({
        'total_voxels': int(total),
        'gt_nonfree': int(np.count_nonzero(occ_gt)),
        'pred_nonfree': int(np.count_nonzero(occ_pred)),
        'exact_match': exact_match,
        'accuracy': exact_match / total if total else 0.0,
        'occupancy_iou': (intersection / union) if union else 1.0,
        'miss_count': miss,
        'false_positive_count': false_positive,
        'confusion_count': confusion,
    })


# ------------------------------------------------------------------
# 标定 / 轨迹
# ------------------------------------------------------------------

@app.route('/api/calibration')
def get_calibration():
    """返回当前数据集 calibration/intrinsics.json + extrinsics.json 的合并内容"""
    dataset_path = Path(CURRENT_DATASET_DIR)
    int_path = dataset_path / 'calibration' / 'intrinsics.json'
    ext_path = dataset_path / 'calibration' / 'extrinsics.json'

    if not int_path.exists() or not ext_path.exists():
        return abort(404, description="Calibration not found")

    with open(int_path, 'r') as f:
        intrinsics = json.load(f)
    with open(ext_path, 'r') as f:
        extrinsics = json.load(f)

    cameras = {}
    cam_keys = sorted([k for k in intrinsics.keys() if k.startswith('cam_')],
                       key=lambda k: int(k.split('_')[1]))
    for key in cam_keys:
        cameras[key] = {
            **intrinsics.get(key, {}),
            'extrinsics': extrinsics.get(key, {}),
        }

    return jsonify({
        'cameras': cameras,
        'raw_bit_depth': intrinsics.get('raw_bit_depth'),
    })


@app.route('/api/trajectory')
def get_trajectory():
    """
    扫描 ego_pose/*.npy，返回按帧顺序排列的 (x,y) 位置，供 3D 视图画自车轨迹。
    体素视图始终是"当前帧车辆自身坐标系"（车在原点），所以轨迹默认按 ?relative_to=<frame_id>
    转换到该帧的自车坐标系下（用完整 4x4 位姿做旋转+平移变换），不传则返回原始世界坐标
    （直接画意义不大，只有 relative_to 模式才能叠加到当前体素视图里）。
    """
    dataset_path = Path(CURRENT_DATASET_DIR)
    ego_dir = dataset_path / 'ego_pose'
    if not ego_dir.exists():
        return jsonify({'points': []})

    relative_to = request.args.get('relative_to')
    frames = get_frames()

    poses = {}
    for frame_id in frames:
        pose_path = ego_dir / f"{frame_id}.npy"
        if not pose_path.exists():
            continue
        try:
            poses[frame_id] = np.load(pose_path).astype(np.float64)  # (4,4) Vehicle->World
        except Exception as e:
            logger.warning(f"Failed to read ego_pose for {frame_id}: {e}")

    ref_pose = poses.get(relative_to) if relative_to else None

    points = []
    for frame_id in frames:
        pose = poses.get(frame_id)
        if pose is None:
            continue
        if ref_pose is not None:
            R_ref = ref_pose[:3, :3]
            t_ref = ref_pose[:3, 3]
            rel = R_ref.T @ (pose[:3, 3] - t_ref)
            points.append([float(rel[0]), float(rel[1])])
        else:
            points.append([float(pose[0, 3]), float(pose[1, 3])])

    return jsonify({'points': points, 'relative_to': relative_to})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument('--prediction', type=str, default=None, help='推理结果目录 (可选)')
    args = parser.parse_args()

    if args.dataset and os.path.exists(args.dataset):
        CURRENT_DATASET_DIR = args.dataset

    if args.prediction and os.path.exists(args.prediction):
        CURRENT_PREDICTION_DIR = args.prediction

    print("=" * 60)
    print("Dataset Viewer v2 Server")
    print("=" * 60)
    print(f"Dataset:    {CURRENT_DATASET_DIR}")
    print(f"Prediction: {CURRENT_PREDICTION_DIR or '(未设置)'}")
    print(f"Port:       {args.port}")
    print(f"URL:        http://127.0.0.1:{args.port}/")
    print("(用 127.0.0.1 而不是 localhost 打开 —— 本机 localhost 解析会先尝试 IPv6 再回退")
    print(" IPv4，每个请求多出约 2 秒延迟，直接用 127.0.0.1 可以完全避开这个坑)")
    print("=" * 60)

    dataset_path = Path(CURRENT_DATASET_DIR)
    if not dataset_path.exists():
        logger.warning(f"⚠️ Dataset directory does not exist: {CURRENT_DATASET_DIR}")
    else:
        frames = get_frames()
        logger.info(f"✓ Found {len(frames)} frames")

    if CURRENT_PREDICTION_DIR:
        pred_frames = get_prediction_frames()
        logger.info(f"✓ Found {len(pred_frames)} prediction frames")

    import logging as flask_logging
    flask_logging.getLogger('werkzeug').setLevel(flask_logging.ERROR)

    print(f"Server starting...")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)

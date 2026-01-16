import os
import json
import argparse
from pathlib import Path
import numpy as np
from flask import Flask, jsonify, send_file, request, abort
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
    # 记录请求开始
    logger.info(f"→ {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def after_request(response):
    latency = (time.time() - request.start_time) * 1000  # ms

    # 状态码颜色标记 (虽然在终端不显示颜色，但便于阅读)
    status_icon = "✓" if 200 <= response.status_code < 300 else "✗"

    logger.info(f"← {status_icon} {request.method} {request.path} - {response.status_code} ({latency:.1f}ms)")
    return response

# LRU Cache for processed images (Max 32 images ~ 4 frames of 8 cams)
@lru_cache(maxsize=32)
def process_dng_cached(dng_path_str):
    """
    缓存的 DNG 处理函数
    输入必须是字符串(hashable)，不能是 Path 对象
    """
    try:
        import rawpy
        with rawpy.imread(dng_path_str) as raw:
            # 性能优化: half_size=True (1/4 分辨率, 4x 速度)
            # Web 预览不需要全分辨率 (1280x960 -> 640x480)
            rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=False)
            return rgb
    except Exception as e:
        # 这里不要 print，交给调用者处理
        raise e

def create_placeholder_image(text="Error", size=(640, 480)):
    """生成占位图"""
    img = Image.new('RGB', size, color=(30, 30, 30))
    return img

# 默认配置
DEFAULT_DATASET_DIR = r"d:\code\carla\dataset_10k_bak"
CURRENT_DATASET_DIR = DEFAULT_DATASET_DIR

# 缓存帧列表
FRAMES_CACHE = []

def get_frames():
    """获取数据集中的帧列表"""
    global FRAMES_CACHE
    if FRAMES_CACHE:
        return FRAMES_CACHE
    
    dataset_path = Path(CURRENT_DATASET_DIR)
    if not dataset_path.exists():
        return []

    # 优先读取 test.txt / train.txt / val.txt
    frames = []
    for txt_file in ['test.txt', 'train.txt', 'val.txt']:
        txt_path = dataset_path / txt_file
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                frames.extend(lines)
    
    # 如果没有 txt，扫描 occupancy 目录
    if not frames:
        occ_dir = dataset_path / 'occupancy'
        if occ_dir.exists():
            frames = [f.stem for f in occ_dir.glob('*.npy')]
            frames.sort()
    
    # 去重并排序
    frames = sorted(list(set(frames)))
    FRAMES_CACHE = frames
    return frames

@app.route('/')
def index():
    return send_file('templates/index.html') # Flask send_file doesn't process templates
    # To use templates properly we should use render_template, but let's just stick to static file for now
    # and use client-side cache busting


@app.route('/api/dataset_info')
def dataset_info():
    frames = get_frames()
    return jsonify({
        'path': CURRENT_DATASET_DIR,
        'count': len(frames),
        'frames': frames
    })

@app.route('/api/set_dataset', methods=['POST'])
def set_dataset():
    global CURRENT_DATASET_DIR, FRAMES_CACHE
    data = request.json
    new_path = data.get('path')
    if new_path and os.path.exists(new_path):
        CURRENT_DATASET_DIR = new_path
        FRAMES_CACHE = [] # 清空缓存
        return jsonify({'success': True, 'count': len(get_frames())})
    return jsonify({'success': False, 'message': 'Path does not exist'})

@app.route('/api/image/<frame_id>/<int:cam_idx>')
def get_image(frame_id, cam_idx):
    """获取指定帧和相机的图像 (优先 PNG > DNG > NPY)"""
    dataset_path = Path(CURRENT_DATASET_DIR)
    img_dir = dataset_path / 'images' / frame_id

    # 🔥 0. 优先加载 PNG 缩略图 (最快, 无需处理)
    png_path = img_dir / f"cam_{cam_idx}.png"
    if png_path.exists():
        try:
            logger.debug(f"✓ Loading PNG thumbnail: {png_path.name}")
            return send_file(str(png_path), mimetype='image/png')
        except Exception as e:
            logger.warning(f"PNG loading failed: {e}, falling back to DNG")

    # 1. 尝试加载 DNG (需要实时处理)
    dng_path = img_dir / f"cam_{cam_idx}.dng"

    if dng_path.exists():
        try:
            # ⭐⭐⭐ 性能优化: DNG-to-PNG 持久化缓存 ⭐⭐⭐
            # 为每个 DNG 生成对应的 PNG 缓存文件
            # 缓存路径: dataset/.png_cache/images/frame_id/cam_idx.png

            cache_dir = dataset_path / '.png_cache' / 'images' / frame_id
            cache_png = cache_dir / f"cam_{cam_idx}.png"

            # 检查缓存是否存在且比 DNG 新
            use_cache = False
            if cache_png.exists():
                dng_mtime = dng_path.stat().st_mtime
                png_mtime = cache_png.stat().st_mtime
                if png_mtime >= dng_mtime:
                    use_cache = True
                    logger.debug(f"✓ Using PNG cache: {cache_png.relative_to(dataset_path)}")

            # 如果缓存可用，直接返回
            if use_cache:
                return send_file(str(cache_png), mimetype='image/png')

            # 缓存不可用，需要加载 DNG 并生成缓存
            logger.debug(f"⚙ Generating PNG cache for: {dng_path.name}")

            # 尝试使用 rawpy (最佳质量 + 缓存 + 性能优化)
            try:
                import rawpy
                # 使用缓存处理
                rgb = process_dng_cached(str(dng_path))

                if rgb is None:
                    # 如果 rawpy 失败, 清除缓存并抛出异常以降级
                    logger.warning(f"rawpy returned None for {dng_path.name}, clearing cache")
                    process_dng_cached.cache_clear()
                    raise Exception("Cached processing failed")

            except ImportError as e:
                # rawpy 未安装，降级使用 OpenCV
                logger.warning(f"rawpy not available: {e}, trying cv2")
                import cv2
                img = cv2.imread(str(dng_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    logger.error(f"cv2 failed to load DNG: {dng_path}")
                    return abort(404, description="Failed to load DNG with cv2")

                # Bayer RGGB -> RGB
                rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_RGGB2RGB)

                # 关键修复: 如果是 16-bit/12-bit (uint16)，必须归一化到 8-bit
                if rgb.dtype == np.uint16:
                    max_val = np.max(rgb)
                    if max_val > 0:
                        rgb = (rgb / max_val * 255).astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)

            # 转换为 PIL Image
            img_pil = Image.fromarray(rgb)

            # 保存 PNG 缓存到磁盘
            cache_dir.mkdir(parents=True, exist_ok=True)
            img_pil.save(str(cache_png), 'PNG', optimize=True)
            logger.debug(f"✓ PNG cache saved: {cache_png.relative_to(dataset_path)}")

            # 返回 PNG (直接发送缓存文件，避免二次编码)
            return send_file(str(cache_png), mimetype='image/png')

        except Exception as e:
            logger.error(f"❌ Error loading DNG {dng_path.name}: {type(e).__name__}: {e}")
            # 返回占位图，防止前端图标破碎
            img = create_placeholder_image(text=f"Error: {dng_path.name}")
            img_io = io.BytesIO()
            img.save(img_io, 'JPEG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')

    # 2. 尝试加载 NPY
    npy_path = img_dir / f"cam_{cam_idx}.npy"
    if not npy_path.exists():
        return abort(404, description="Image not found")
        
    try:
        # 加载 npy: (1, H, W) float16, range [0, 1]
        data = np.load(npy_path)
        
        # 转换为 (H, W)
        if data.ndim == 3 and data.shape[0] == 1:
            img_data = data[0]
        else:
            img_data = data
            
        # 检查是否包含 NaN 或 Inf
        if not np.isfinite(img_data).all():
             img_data = np.nan_to_num(img_data)
            
        # 归一化并转 uint8
        img_data = np.clip(img_data, 0, 1)
        img_uint8 = (img_data * 255).astype(np.uint8)
        
        # 转换为 PIL Image (灰度)
        img = Image.fromarray(img_uint8, mode='L')
        
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
        
    except Exception as e:
        print(f"Error loading NPY {npy_path}: {e}")
        return abort(500)

@app.route('/api/occupancy/<frame_id>')
def get_occupancy(frame_id):
    """获取指定帧的体素数据 (稀疏格式)"""
    dataset_path = Path(CURRENT_DATASET_DIR)
    occ_path = dataset_path / 'occupancy' / f"{frame_id}.npy"
    
    if not occ_path.exists():
        return abort(404)
        
    try:
        # 加载 npy: (400, 400, 32) uint8
        grid = np.load(occ_path)

        # 提取非空体素 (排除 free=0)
        # ⚠️ 但我们的数据集使用 nuScenes 17 类,需要可视化所有类 (包括 free)
        # 为了性能,只显示 label != 0 (但这会丢失 free 体素)
        # 如果要显示 free,改为: indices = np.argwhere(grid >= 0)
        indices = np.argwhere(grid != 0)  # 排除 free (0)
        
        if len(indices) == 0:
             return jsonify({'points': [], 'labels': []})
             
        labels = grid[indices[:,0], indices[:,1], indices[:,2]]
        
        # 返回稀疏数据
        # 为了减少传输量，可以将 points 和 labels 分开
        # 或者使用简单的 list of lists
        return jsonify({
            'points': indices.tolist(),
            'labels': labels.tolist(),
            'shape': grid.shape
        })
        
    except Exception as e:
        print(f"Error loading occupancy {occ_path}: {e}")
        return abort(500)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()

    if args.dataset and os.path.exists(args.dataset):
        CURRENT_DATASET_DIR = args.dataset

    # 打印启动横幅 (类似 occupancy_viewer 风格)
    print("=" * 60)
    print("Dataset Viewer v2 Server")
    print("=" * 60)
    print(f"Dataset:  {CURRENT_DATASET_DIR}")
    print(f"Port:     {args.port}")
    print(f"URL:      http://localhost:{args.port}/")
    print("=" * 60)

    # 验证数据集
    dataset_path = Path(CURRENT_DATASET_DIR)
    if not dataset_path.exists():
        logger.warning(f"⚠️ Dataset directory does not exist: {CURRENT_DATASET_DIR}")
    else:
        frames = get_frames()
        logger.info(f"✓ Found {len(frames)} frames")

    # 显式开启多线程，禁用 Debugger (防止干扰线程)
    # 禁用 Flask 自带的请求日志 (使用我们的自定义日志)
    import logging as flask_logging
    flask_logging.getLogger('werkzeug').setLevel(flask_logging.ERROR)

    print(f"Server starting...")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)

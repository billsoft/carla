"""
测试 PNG 缓存优化效果

比较 DNG 首次加载 vs PNG 缓存加载的性能差异
"""

import time
import requests
from pathlib import Path

# 配置
BASE_URL = "http://localhost:8085"
DATASET_PATH = r"d:\code\carla\dataset_10k_bak"

def test_cache_performance():
    """测试缓存性能"""

    print("=" * 60)
    print("PNG 缓存性能测试")
    print("=" * 60)

    # 1. 获取数据集信息
    print("\n[1/4] 获取数据集信息...")
    resp = requests.get(f"{BASE_URL}/api/dataset_info")
    data = resp.json()

    if data['count'] == 0:
        print("❌ 数据集为空，请设置正确的数据集路径")
        return

    print(f"  ✓ 数据集路径: {data['path']}")
    print(f"  ✓ 帧数: {data['count']}")

    # 选择第一帧进行测试
    frame_id = data['frames'][0]
    print(f"  ✓ 测试帧: {frame_id}")

    # 2. 清理缓存（如果存在）
    print("\n[2/4] 清理旧缓存...")
    cache_dir = Path(DATASET_PATH) / '.png_cache'
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print(f"  ✓ 已删除缓存目录: {cache_dir}")
    else:
        print(f"  ✓ 缓存目录不存在，跳过")

    # 3. 首次加载（无缓存，需要处理 DNG）
    print("\n[3/4] 首次加载（DNG 处理 + 生成 PNG 缓存）...")

    load_times_first = []
    for cam_idx in range(8):
        start = time.time()
        resp = requests.get(f"{BASE_URL}/api/image/{frame_id}/{cam_idx}")
        elapsed = (time.time() - start) * 1000  # ms
        load_times_first.append(elapsed)
        print(f"  相机 {cam_idx}: {elapsed:.1f} ms")

    avg_first = sum(load_times_first) / len(load_times_first)
    print(f"\n  平均加载时间: {avg_first:.1f} ms")

    # 4. 第二次加载（使用 PNG 缓存）
    print("\n[4/4] 第二次加载（直接使用 PNG 缓存）...")

    load_times_cached = []
    for cam_idx in range(8):
        start = time.time()
        resp = requests.get(f"{BASE_URL}/api/image/{frame_id}/{cam_idx}")
        elapsed = (time.time() - start) * 1000  # ms
        load_times_cached.append(elapsed)
        print(f"  相机 {cam_idx}: {elapsed:.1f} ms")

    avg_cached = sum(load_times_cached) / len(load_times_cached)
    print(f"\n  平均加载时间: {avg_cached:.1f} ms")

    # 5. 性能对比
    print("\n" + "=" * 60)
    print("性能对比结果")
    print("=" * 60)
    print(f"首次加载（DNG 处理）: {avg_first:.1f} ms")
    print(f"缓存加载（PNG 读取）: {avg_cached:.1f} ms")

    speedup = avg_first / avg_cached if avg_cached > 0 else 0
    print(f"性能提升: {speedup:.1f}x 倍")

    # 6. 缓存大小统计
    print("\n" + "=" * 60)
    print("缓存统计")
    print("=" * 60)

    cache_files = list(cache_dir.glob('**/*.png'))
    total_size = sum(f.stat().st_size for f in cache_files)

    print(f"缓存文件数: {len(cache_files)}")
    print(f"缓存总大小: {total_size / 1024 / 1024:.2f} MB")
    print(f"平均文件大小: {total_size / len(cache_files) / 1024:.2f} KB")

if __name__ == '__main__':
    print("⚠️ 请确保 server.py 正在运行 (python server.py)\n")
    time.sleep(2)

    try:
        test_cache_performance()
    except requests.exceptions.ConnectionError:
        print("\n❌ 无法连接到服务器，请先启动: python server.py")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

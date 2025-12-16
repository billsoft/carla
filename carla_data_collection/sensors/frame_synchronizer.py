"""多传感器帧同步模块

负责:
1. 同步8个摄像头的数据
2. 基于时间戳对齐帧
3. 处理丢帧和超时情况
"""

import time
from collections import defaultdict
import numpy as np


class FrameSynchronizer:
    """多传感器帧同步器"""

    def __init__(self, camera_ids, timeout=1.0, time_tolerance=0.01):
        """
        初始化同步器

        参数:
            camera_ids: 相机ID列表 (例如 ['front_main', 'front_narrow', ...])
            timeout: 等待同步超时时间 (秒)
            time_tolerance: 时间戳容差 (秒), 认为在此范围内的帧是同步的
        """
        self.camera_ids = camera_ids
        self.num_cameras = len(camera_ids)
        self.timeout = timeout
        self.time_tolerance = time_tolerance

        # 每个相机的最新数据缓存
        self.latest_data = {}

        # 统计信息
        self.stats = {
            'synced_frames': 0,
            'timeout_count': 0,
            'timestamp_misalignment': [],
        }

        # 时间戳缓存
        self.frame_timestamps = defaultdict(dict)

    def push_camera_data(self, camera_id, data):
        """
        推送相机数据到同步器

        参数:
            camera_id: 相机标识符
            data: 相机数据字典 (必须包含 'timestamp' 和 'frame' 字段)
        """
        if camera_id not in self.camera_ids:
            print(f"警告: 未知相机ID {camera_id}")
            return

        # 更新最新数据
        self.latest_data[camera_id] = data

        # 缓存时间戳
        frame_id = data['frame']
        self.frame_timestamps[frame_id][camera_id] = data['timestamp']

    def get_synced_frame(self):
        """
        获取一帧同步的多相机数据

        阻塞等待直到所有相机数据就绪或超时

        返回:
            synced_data: 字典
                {
                    'cameras': {camera_id: data, ...},  # 8个相机的数据
                    'timestamp': float,  # 平均时间戳
                    'frame': int,  # 帧号
                    'time_std': float,  # 时间戳标准差
                }
            或 None (如果超时)
        """
        start_time = time.time()

        while True:
            # 检查是否所有相机都有数据
            if len(self.latest_data) == self.num_cameras:
                # 检查时间戳是否对齐
                timestamps = [data['timestamp'] for data in self.latest_data.values()]
                frames = [data['frame'] for data in self.latest_data.values()]

                # 时间戳统计
                ts_mean = np.mean(timestamps)
                ts_std = np.std(timestamps)
                ts_max_diff = np.max(timestamps) - np.min(timestamps)

                # 帧号统计
                frame_ids = set(frames)

                # 判断是否同步
                if ts_max_diff <= self.time_tolerance and len(frame_ids) == 1:
                    # 同步成功
                    synced_data = {
                        'cameras': self.latest_data.copy(),
                        'timestamp': ts_mean,
                        'frame': frames[0],
                        'time_std': ts_std,
                        'time_max_diff': ts_max_diff
                    }

                    # 更新统计
                    self.stats['synced_frames'] += 1
                    self.stats['timestamp_misalignment'].append(ts_max_diff)

                    # 清空缓存,准备下一帧
                    self.latest_data.clear()

                    return synced_data

                # 如果时间戳不对齐,清除最旧的数据
                elif ts_max_diff > self.time_tolerance:
                    # 找到最旧的相机数据
                    oldest_camera = min(self.latest_data.items(),
                                       key=lambda x: x[1]['timestamp'])[0]
                    print(f"警告: 相机 {oldest_camera} 时间戳过旧 "
                          f"(diff={ts_max_diff:.3f}s), 丢弃")
                    del self.latest_data[oldest_camera]

                # 如果帧号不一致,清除最旧的帧
                elif len(frame_ids) > 1:
                    oldest_frame = min(frames)
                    cameras_to_remove = [cam_id for cam_id, data in self.latest_data.items()
                                        if data['frame'] == oldest_frame]
                    for cam_id in cameras_to_remove:
                        print(f"警告: 相机 {cam_id} 帧号过旧 (frame={oldest_frame}), 丢弃")
                        del self.latest_data[cam_id]

            # 检查超时
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                print(f"警告: 帧同步超时 ({elapsed:.2f}s)")
                print(f"  已收到 {len(self.latest_data)}/{self.num_cameras} 个相机数据")
                print(f"  缺失相机: {set(self.camera_ids) - set(self.latest_data.keys())}")

                self.stats['timeout_count'] += 1

                # 清空缓存
                self.latest_data.clear()

                return None

            # 短暂休眠,避免CPU空转
            time.sleep(0.001)

    def print_stats(self):
        """打印同步统计信息"""
        print("\n=== 帧同步统计 ===")
        print(f"成功同步帧数: {self.stats['synced_frames']}")
        print(f"超时次数: {self.stats['timeout_count']}")

        if self.stats['timestamp_misalignment']:
            misalign = np.array(self.stats['timestamp_misalignment'])
            print(f"时间戳对齐精度:")
            print(f"  平均: {misalign.mean()*1000:.2f} ms")
            print(f"  最大: {misalign.max()*1000:.2f} ms")
            print(f"  标准差: {misalign.std()*1000:.2f} ms")

        success_rate = 100.0 * self.stats['synced_frames'] / (
            self.stats['synced_frames'] + self.stats['timeout_count']
        ) if (self.stats['synced_frames'] + self.stats['timeout_count']) > 0 else 0

        print(f"同步成功率: {success_rate:.1f}%")

    def reset(self):
        """重置同步器状态"""
        self.latest_data.clear()
        self.frame_timestamps.clear()
        self.stats = {
            'synced_frames': 0,
            'timeout_count': 0,
            'timestamp_misalignment': [],
        }

"""
数据采集主类
采集 8 相机 RGB + 语义激光雷达点云 → 生成训练数据对
"""

import carla
import h5py
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional
import json

from core.hero_vehicle import HeroVehicleManager
from core.npc_manager import NPCManager
from sensors.frame_synchronizer import FrameSynchronizer
from utils.image_processing import convert_to_12bit_raw
from utils.coordinate_transform import world_to_ego
from data.occupancy_generator import OccupancyGenerator


class DataCollector:
    """
    数据采集主类

    采集内容:
    - 8 相机 RGB 图像 (1280×960, 12-bit)
    - 语义激光雷达点云
    - 车辆位姿

    生成内容:
    - images: 8×RGB (训练输入)
    - occupancy: 3D 体素 (训练标签/GT)
    - metadata: 内外参、位姿等
    """

    def __init__(self,
                 carla_host: str = 'localhost',
                 carla_port: int = 2000,
                 output_dir: str = 'data/collected',
                 dataset_name: str = 'carla_occupancy'):
        """
        Args:
            carla_host: CARLA 服务器地址
            carla_port: CARLA 服务器端口
            output_dir: 输出目录
            dataset_name: 数据集名称
        """
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 连接 CARLA
        self.client = carla.Client(carla_host, carla_port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # 管理器
        self.hero_manager: Optional[HeroVehicleManager] = None
        self.npc_manager: Optional[NPCManager] = None
        self.frame_synchronizer: Optional[FrameSynchronizer] = None
        self.occupancy_generator: Optional[OccupancyGenerator] = None

        # HDF5 文件
        self.hdf5_file: Optional[h5py.File] = None

        # 帧计数
        self.frame_count = 0

    def setup(self,
              num_npc_vehicles: int = 50,
              num_pedestrians: int = 30):
        """
        初始化数据采集环境

        Args:
            num_npc_vehicles: NPC 车辆数量
            num_pedestrians: 行人数量
        """
        print("="*60)
        print("数据采集环境初始化")
        print("="*60)

        # 1. 启用同步模式
        self._enable_synchronous_mode()

        # 2. 创建 Hero 车辆
        self.hero_manager = HeroVehicleManager(self.world)

        # 3. 附加 8 个相机
        cameras = self.hero_manager.attach_cameras()

        # 4. 附加语义激光雷达 (用于 GT 生成)
        lidar = self.hero_manager.attach_semantic_lidar()

        # 5. 创建帧同步器 (8 相机 + 1 激光雷达)
        sensor_ids = list(cameras.keys()) + ['semantic_lidar']
        self.frame_synchronizer = FrameSynchronizer(sensor_ids)

        # 6. 注册传感器回调
        for cam_id, camera_sensor in cameras.items():
            camera_sensor.listen(
                lambda image, cid=cam_id: self._camera_callback(cid, image)
            )

        lidar.listen_to_queue()  # 激光雷达使用队列模式

        # 7. 启用自动驾驶
        self.hero_manager.enable_autopilot()

        # 8. 生成 NPC
        self.npc_manager = NPCManager(self.world)
        self.npc_manager.spawn_vehicles(num_npc_vehicles)
        self.npc_manager.spawn_pedestrians(num_pedestrians)

        # 9. 创建 Occupancy 生成器
        self.occupancy_generator = OccupancyGenerator()

        # 10. 创建 HDF5 文件
        self._create_hdf5_file()

        print("\n✓ 环境初始化完成!")

    def _enable_synchronous_mode(self):
        """启用 CARLA 同步模式"""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 20.0  # 20Hz (语义激光雷达频率)
        self.world.apply_settings(settings)
        print("[CARLA] 同步模式已启用 (20Hz)")

    def _camera_callback(self, camera_id: str, image: carla.Image):
        """相机数据回调"""
        # 转换图像数据
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA

        # 推送到同步器
        self.frame_synchronizer.push_camera_data(camera_id, {
            'timestamp': image.timestamp,
            'frame': image.frame,
            'data': array,
            'width': image.width,
            'height': image.height
        })

    def _create_hdf5_file(self):
        """创建 HDF5 数据集文件"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.dataset_name}_{timestamp}.h5"
        filepath = self.output_dir / filename

        self.hdf5_file = h5py.File(filepath, 'w')

        # 预分配空间 (假设采集 10000 帧)
        max_frames = 10000

        # ========== 训练输入: 8 相机图像 ==========
        self.hdf5_file.create_dataset(
            'images',
            shape=(max_frames, 8, 960, 1280, 3),
            dtype=np.uint16,  # 12-bit 存储为 uint16
            compression='gzip',
            compression_opts=4
        )

        # ========== 训练标签: Occupancy 体素 ==========
        self.hdf5_file.create_dataset(
            'occupancy',
            shape=(max_frames, 200, 200, 16),
            dtype=np.uint8,  # 语义类别 [0-17]
            compression='gzip',
            compression_opts=4
        )

        self.hdf5_file.create_dataset(
            'occupancy_mask',
            shape=(max_frames, 200, 200, 16),
            dtype=np.bool_,  # 有效观测掩码
            compression='gzip',
            compression_opts=4
        )

        # ========== 元数据 ==========
        self.hdf5_file.create_dataset(
            'timestamps',
            shape=(max_frames,),
            dtype=np.float64
        )

        self.hdf5_file.create_dataset(
            'frame_ids',
            shape=(max_frames,),
            dtype=np.int32
        )

        # 车辆位姿
        self.hdf5_file.create_dataset(
            'vehicle_location',
            shape=(max_frames, 3),
            dtype=np.float32
        )

        self.hdf5_file.create_dataset(
            'vehicle_rotation',
            shape=(max_frames, 3),
            dtype=np.float32
        )

        self.hdf5_file.create_dataset(
            'vehicle_velocity',
            shape=(max_frames, 3),
            dtype=np.float32
        )

        # 相机内外参 (固定值,只存一次)
        camera_names = [cam.config['id'] for cam in self.hero_manager.cameras.values()]
        intrinsics = np.stack([
            cam.intrinsic_matrix for cam in self.hero_manager.cameras.values()
        ])
        extrinsics = np.stack([
            cam.extrinsic_matrix for cam in self.hero_manager.cameras.values()
        ])

        self.hdf5_file.create_dataset('camera_names', data=[n.encode() for n in camera_names])
        self.hdf5_file.create_dataset('camera_intrinsics', data=intrinsics)
        self.hdf5_file.create_dataset('camera_extrinsics', data=extrinsics)

        # Occupancy 配置
        self.hdf5_file.attrs['occupancy_x_range'] = self.occupancy_generator.x_range
        self.hdf5_file.attrs['occupancy_y_range'] = self.occupancy_generator.y_range
        self.hdf5_file.attrs['occupancy_z_range'] = self.occupancy_generator.z_range
        self.hdf5_file.attrs['occupancy_resolution'] = self.occupancy_generator.resolution

        print(f"[HDF5] 数据集文件已创建: {filepath}")

    def collect(self, num_frames: int = 1000):
        """
        开始数据采集

        Args:
            num_frames: 采集帧数
        """
        print(f"\n{'='*60}")
        print(f"开始采集 {num_frames} 帧数据")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            for frame_idx in range(num_frames):
                # Tick 仿真
                self.world.tick()

                # ========== 1. 采集相机数据 ==========
                synced_frame = self._collect_camera_data(timeout=2.0)

                if synced_frame is None:
                    print(f"[警告] 帧 {frame_idx} 相机数据同步失败,跳过")
                    continue

                # ========== 2. 采集激光雷达数据 ==========
                lidar_data = self._collect_lidar_data(timeout=2.0)

                if lidar_data is None:
                    print(f"[警告] 帧 {frame_idx} 激光雷达数据缺失,跳过")
                    continue

                # ========== 3. 生成 Occupancy GT ==========
                occupancy, mask = self._generate_occupancy_gt(lidar_data)

                # ========== 4. 保存数据 ==========
                self._save_frame(synced_frame, occupancy, mask)

                # 进度显示
                if (frame_idx + 1) % 20 == 0:
                    elapsed = time.time() - start_time
                    fps = (frame_idx + 1) / elapsed
                    eta = (num_frames - frame_idx - 1) / fps if fps > 0 else 0

                    print(f"进度: {frame_idx + 1}/{num_frames} "
                          f"| {fps:.2f} fps "
                          f"| ETA: {eta/60:.1f}min")

        except KeyboardInterrupt:
            print("\n用户中断采集")

        finally:
            self._finalize()

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"采集完成!")
        print(f"  总帧数: {self.frame_count}")
        print(f"  总耗时: {total_time/60:.2f} 分钟")
        print(f"  平均帧率: {self.frame_count / total_time:.2f} fps")
        print(f"{'='*60}")

    def _collect_camera_data(self, timeout: float = 2.0) -> Optional[Dict]:
        """采集同步的相机数据"""
        # 等待所有相机数据 (除了 semantic_lidar)
        camera_ids = [cid for cid in self.frame_synchronizer.camera_ids
                     if cid != 'semantic_lidar']

        synced_data = {}
        start_time = time.time()

        for camera_id in camera_ids:
            try:
                remaining_timeout = timeout - (time.time() - start_time)
                if remaining_timeout <= 0:
                    return None

                data = self.frame_synchronizer.camera_queues[camera_id].get(
                    timeout=remaining_timeout
                )
                synced_data[camera_id] = data

            except:
                return None

        return synced_data

    def _collect_lidar_data(self, timeout: float = 2.0) -> Optional[Dict]:
        """采集激光雷达数据"""
        try:
            lidar_data = self.hero_manager.lidar.data_queue.get(timeout=timeout)
            return lidar_data
        except:
            return None

    def _generate_occupancy_gt(self, lidar_data: Dict) -> tuple:
        """
        从激光雷达点云生成 Occupancy GT

        Args:
            lidar_data: 激光雷达数据字典

        Returns:
            occupancy: (200, 200, 16) 体素标签
            mask: (200, 200, 16) 有效掩码
        """
        # 解析点云
        xyz_world, semantic_tags = self.hero_manager.lidar.parse_lidar_data(
            lidar_data['raw_data']
        )

        # 坐标转换: 世界 → 车辆
        vehicle_state = self.hero_manager.get_vehicle_state()
        xyz_ego = world_to_ego(xyz_world, vehicle_state['transform'])

        # 生成 Occupancy
        occupancy, mask = self.occupancy_generator.generate(xyz_ego, semantic_tags)

        return occupancy, mask

    def _save_frame(self,
                   synced_frame: Dict,
                   occupancy: np.ndarray,
                   mask: np.ndarray):
        """保存单帧数据到 HDF5"""

        # ========== 1. 转换图像为 12-bit RAW ==========
        images_12bit = []
        camera_order = [
            'cam_front_ultra_wide', 'cam_front_wide', 'cam_front_narrow',
            'cam_front_left', 'cam_front_right',
            'cam_rear_left', 'cam_rear_right', 'cam_rear'
        ]

        for camera_id in camera_order:
            bgra_image = synced_frame[camera_id]['data']
            rgb_12bit = convert_to_12bit_raw(bgra_image)
            images_12bit.append(rgb_12bit)

        images_12bit = np.stack(images_12bit, axis=0)  # (8, 960, 1280, 3)

        # ========== 2. 获取车辆状态 ==========
        vehicle_state = self.hero_manager.get_vehicle_state()

        # ========== 3. 写入 HDF5 ==========
        idx = self.frame_count

        self.hdf5_file['images'][idx] = images_12bit
        self.hdf5_file['occupancy'][idx] = occupancy
        self.hdf5_file['occupancy_mask'][idx] = mask

        # 时间戳
        first_cam = list(synced_frame.values())[0]
        self.hdf5_file['timestamps'][idx] = first_cam['timestamp']
        self.hdf5_file['frame_ids'][idx] = first_cam['frame']

        # 车辆状态
        self.hdf5_file['vehicle_location'][idx] = [
            vehicle_state['location']['x'],
            vehicle_state['location']['y'],
            vehicle_state['location']['z']
        ]

        self.hdf5_file['vehicle_rotation'][idx] = [
            vehicle_state['rotation']['pitch'],
            vehicle_state['rotation']['yaw'],
            vehicle_state['rotation']['roll']
        ]

        self.hdf5_file['vehicle_velocity'][idx] = [
            vehicle_state['velocity']['x'],
            vehicle_state['velocity']['y'],
            vehicle_state['velocity']['z']
        ]

        self.frame_count += 1

    def _finalize(self):
        """结束采集,清理资源"""

        # 裁剪 HDF5 数据集到实际大小
        if self.hdf5_file is not None and self.frame_count > 0:
            for key in ['images', 'occupancy', 'occupancy_mask',
                       'timestamps', 'frame_ids',
                       'vehicle_location', 'vehicle_rotation', 'vehicle_velocity']:
                self.hdf5_file[key].resize((self.frame_count,) + self.hdf5_file[key].shape[1:])

            self.hdf5_file.close()
            print(f"[HDF5] 数据集已保存 ({self.frame_count} 帧)")

        # 打印同步统计
        if self.frame_synchronizer is not None:
            self.frame_synchronizer.print_stats()

        # 销毁 NPC
        if self.npc_manager is not None:
            self.npc_manager.destroy_all()

        # 销毁 Hero 车辆
        if self.hero_manager is not None:
            self.hero_manager.destroy()

        print("\n✓ 资源已清理")

    def cleanup(self):
        """外部调用的清理接口(与 _finalize 相同,但可重复调用)"""
        if hasattr(self, 'hdf5_file') and self.hdf5_file is not None:
            self._finalize()

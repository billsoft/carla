"""
数据保存器 - OccNetV3 数据集格式
按照 DATASET_FORMAT.md 规范保存
"""
import os
import json
import numpy as np
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
    from config.occupancy_config import GRID_SIZE
except ImportError:
    # 如果作为模块导入失败，尝试相对导入
    try:
        from ..config.occupancy_config import GRID_SIZE
    except ImportError:
         # Fallback default
         GRID_SIZE = (512, 512, 40)
         print(f"[Warning] Could not import GRID_SIZE, using default: {GRID_SIZE}")

# DNG支持（可选）
try:
    from PIL import Image
    import piexif
    DNG_AVAILABLE = True
except ImportError:
    DNG_AVAILABLE = False
    print("[警告] 未安装 Pillow/piexif，DNG格式不可用。安装: pip install Pillow piexif")


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
        │       ├── cam_0.dng  (12-bit Bayer RGGB)
        │       └── ...
        ├── occupancy/
        │   └── scene_XXXX_frame_YYYY.npy  (512, 512, 32) uint8
        ├── flow/
        │   └── scene_XXXX_frame_YYYY.npy  (3, 512, 512, 32) float16
        ├── flow_mask/
        │   └── scene_XXXX_frame_YYYY.npy  (512, 512, 32) uint8
        ├── ego_pose/
        │   └── scene_XXXX_frame_YYYY.npy  (4, 4) float32
        ├── ego_motion/
        │   └── scene_XXXX_frame_YYYY.npy  (4, 4) float32
        ├── train.txt
        ├── val.txt
        └── test.txt
    """

    def __init__(self, output_dir: str, scene_name: str = 'scene'):
        """
        Args:
            output_dir: 输出根目录 (如 D:/code/carla/dataset_10k_bak)
            scene_name: 场景名称前缀
        """
        self.output_dir = Path(output_dir)
        self.scene_name = scene_name
        self.scene_counter = 0
        self.frame_counter = 0

        self.sample_ids = []  # 记录所有sample_id

        # 创建目录结构
        self._create_directories()

        print(f"[OccNetDataSaver] 初始化完成")
        print(f"  输出目录: {self.output_dir}")

    def _create_directories(self):
        """创建所有必需的目录"""
        dirs = [
            'calibration',
            'images',
            'occupancy',
            'flow',
            'flow_mask',
            'ego_pose',
            'ego_motion',
        ]

        for d in dirs:
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)

        print(f"  ✓ 目录结构已创建")

    def save_calibration(
        self,
        camera_intrinsics: Dict[str, np.ndarray],
        camera_extrinsics: Dict[str, np.ndarray],
        camera_configs: List[Dict]
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
                'fov': float(cam_config['fov']),
                'distortion': {
                    'model': 'pinhole',
                    'k1': 0.0,
                    'k2': 0.0,
                    'p1': 0.0,
                    'p2': 0.0,
                }
            }

        with open(self.output_dir / 'calibration' / 'intrinsics.json', 'w') as f:
            json.dump(intrinsics_data, f, indent=2)

        # extrinsics.json
        extrinsics_data = {}
        for cam_config in camera_configs:
            cam_id = f"cam_{cam_config['index']}"
            T = camera_extrinsics[cam_config['id']]

            # 分解为平移和旋转
            translation = T[:3, 3].tolist()

            # 从旋转矩阵提取欧拉角 (简化,假设无roll)
            R = T[:3, :3]
            rotation_matrix = R.tolist()

            extrinsics_data[cam_id] = {
                'translation': [float(x) for x in translation],
                'rotation': {
                    'roll': 0.0,
                    'pitch': 0.0,
                    'yaw': float(cam_config['rotation'][2]),  # yaw从config读取
                },
                'rotation_matrix': [[float(x) for x in row] for row in rotation_matrix]
            }

        with open(self.output_dir / 'calibration' / 'extrinsics.json', 'w') as f:
            json.dump(extrinsics_data, f, indent=2)

        print(f"  ✓ 标定文件已保存")

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
            depth: {cam_id: (H, W) float32} 深度图 (可选, meters)
        """
        # 1. 保存图像
        img_dir = self.output_dir / 'images' / sample_id
        img_dir.mkdir(parents=True, exist_ok=True)

        for cam_id, img_info in images.items():
            # 提取相机索引 (如 'front_main' → 0)
            cam_index = self._get_cam_index(cam_id)
            
            # 检查数据类型
            if isinstance(img_info, dict) and 'raw_type' in img_info:
                # 处理 Bayer 数据
                raw_type = img_info['raw_type']
                data = img_info['data']
                
                if raw_type == 'bayer_rggb':
                    self._save_bayer_dng(data, img_dir / f'cam_{cam_index}.dng')
                else:
                    # Fallback for other types
                    np.save(img_dir / f'cam_{cam_index}.npy', data)
            else:
                # 兼容旧格式 (直接传入 array)
                img = img_info
                # assert img.shape == (1, 960, 1280), f"图像形状错误: {img.shape}"
                # assert img.dtype == np.float16, f"图像类型错误: {img.dtype}"
                np.save(img_dir / f'cam_{cam_index}.npy', img)

        # 2. 保存occupancy
        assert occupancy.shape == GRID_SIZE, f"occupancy形状错误: {occupancy.shape} (expected {GRID_SIZE})"
        np.save(self.output_dir / 'occupancy' / f'{sample_id}.npy', occupancy.astype(np.uint8))

        # 3. 保存flow (可选)
        if flow is not None:
            expected_flow_shape = (3,) + GRID_SIZE
            assert flow.shape == expected_flow_shape, f"flow形状错误: {flow.shape} (expected {expected_flow_shape})"
            np.save(self.output_dir / 'flow' / f'{sample_id}.npy', flow.astype(np.float16))

        if flow_mask is not None:
            assert flow_mask.shape == GRID_SIZE, f"flow_mask形状错误: {flow_mask.shape} (expected {GRID_SIZE})"
            np.save(self.output_dir / 'flow_mask' / f'{sample_id}.npy', flow_mask.astype(np.uint8))

        # 4. 保存ego数据 (可选)
        if ego_pose is not None:
            assert ego_pose.shape == (4, 4), f"ego_pose形状错误: {ego_pose.shape}"
            np.save(self.output_dir / 'ego_pose' / f'{sample_id}.npy', ego_pose.astype(np.float32))

        if ego_motion is not None:
            assert ego_motion.shape == (4, 4), f"ego_motion形状错误: {ego_motion.shape}"
            np.save(self.output_dir / 'ego_motion' / f'{sample_id}.npy', ego_motion.astype(np.float32))

        # 记录sample_id
        self.sample_ids.append(sample_id)

    def _save_bayer_dng(self, bayer_data, output_path):
        """保存 Bayer RGGB 数据为 DNG 格式"""
        # 转换为 uint16
        bayer_u16 = bayer_data.astype(np.uint16)

        # 16-bit → 12-bit: 右移 4 位
        # [0, 65535] → [0, 4095]
        bayer_12bit = (bayer_u16 >> 4).astype(np.uint16)

        try:
            # 使用 PIL/Pillow 保存单通道 16-bit TIFF（DNG 兼容）
            if DNG_AVAILABLE:
                # 创建 PIL Image（单通道灰度）
                img_pil = Image.fromarray(bayer_12bit, mode='I;16')  # 16-bit grayscale

                # 构建 EXIF/TIFF 元数据
                exif_dict = {
                    "0th": {
                        piexif.ImageIFD.Make: b"CARLA Simulator",
                        piexif.ImageIFD.Model: b"Bayer RGGB Camera",
                        piexif.ImageIFD.Software: b"OccNetV3 Data Generator",
                        piexif.ImageIFD.PhotometricInterpretation: 32803,  # CFA (Color Filter Array)
                        piexif.ImageIFD.SamplesPerPixel: 1,
                        piexif.ImageIFD.BitsPerSample: (12,),  # 12-bit
                    }
                }
                exif_bytes = piexif.dump(exif_dict)

                # 保存为 TIFF（DNG 本质上是特殊的 TIFF）
                img_pil.save(str(output_path), format='TIFF', compression='none', exif=exif_bytes)

            else:
                # 降级: 使用 OpenCV 保存为 TIFF 后重命名
                path_tif = output_path.with_suffix('.tif')
                success = cv2.imwrite(str(path_tif), bayer_12bit, [
                    cv2.IMWRITE_TIFF_COMPRESSION, 1  # 无压缩
                ])
                if not success:
                    raise IOError("cv2.imwrite returned False")

                # 重命名为 .dng
                if output_path.exists():
                    output_path.unlink()
                path_tif.rename(output_path)

        except Exception as e:
            print(f"  [错误] DNG保存失败: {e}")
            # Fallback: 保存为 NPY
            path_npy = output_path.with_suffix('.npy')
            np.save(path_npy, bayer_data)

        # 注意: 缩略图生成已移至外部调用，避免重复逻辑
        height, width = bayer_data.shape

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

    def finalize(self, train_ratio=0.8, val_ratio=0.1):
        """
        结束采集,生成数据集划分文件

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        total = len(self.sample_ids)
        if total == 0:
            print(f"[OccNetDataSaver] 警告: 没有数据")
            return

        # 按场景分组
        scenes = {}
        for sample_id in self.sample_ids:
            scene_id = sample_id.split('_frame_')[0]
            if scene_id not in scenes:
                scenes[scene_id] = []
            scenes[scene_id].append(sample_id)

        # 随机打乱场景
        scene_list = list(scenes.keys())
        np.random.shuffle(scene_list)

        # 划分
        n_scenes = len(scene_list)
        if n_scenes == 1:
            # 特殊处理：只有一个场景时，用于训练和验证，以便调试
            print("[OccNetDataSaver] 提示: 只有一个场景，将同时用于训练和验证")
            train_scenes = scene_list
            val_scenes = scene_list
            test_scenes = []
        else:
            n_train = int(n_scenes * train_ratio)
            n_val = int(n_scenes * val_ratio)
            
            # 确保至少分配一个给训练集
            if n_train == 0 and n_scenes > 0:
                n_train = 1
                n_val = 0
            
            train_scenes = scene_list[:n_train]
            val_scenes = scene_list[n_train:n_train+n_val]
            test_scenes = scene_list[n_train+n_val:]

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
        print(f"  总计: {total} 样本")

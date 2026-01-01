"""
数据保存工具 (Data Saver)
"""
import numpy as np
import cv2
import json
from pathlib import Path

# DNG支持（可选）
try:
    from PIL import Image
    import piexif
    DNG_AVAILABLE = True
except ImportError:
    DNG_AVAILABLE = False
    print("[警告] 未安装 Pillow/piexif，DNG格式不可用。安装: pip install Pillow piexif")

class DataSaver:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.cameras_dir = self.output_dir / 'cameras'
        self.occupancy_dir = self.output_dir / 'occupancy'
        self.camera_params_dir = self.output_dir / 'camera_params'

        self.cameras_dir.mkdir(parents=True, exist_ok=True)
        self.occupancy_dir.mkdir(parents=True, exist_ok=True)
        self.camera_params_dir.mkdir(parents=True, exist_ok=True)

    def save_rgb(self, frame_idx, data_dict):
        """
        保存 RGB 图像（支持 uint8/uint16/float32）
        data_dict: {cam_id: {'data': rgb_array, 'raw_type': str, ...}}
        """
        for cam_id, info in data_dict.items():
            cam_dir = self.cameras_dir / cam_id
            cam_dir.mkdir(exist_ok=True)

            img_data = info['data']
            raw_type = info.get('raw_type', 'uint8')

            # ========== 根据数据类型选择保存格式 ==========

            if raw_type == 'float32':
                # HDR 格式: 保存为 EXR (OpenEXR)
                path = cam_dir / f"{frame_idx:06d}.exr"
                # OpenCV 支持 EXR 写入 (需要编译时启用 OpenEXR)
                try:
                    # BGR 顺序（OpenCV 要求）
                    bgr_float = img_data[:, :, ::-1].astype(np.float32)
                    cv2.imwrite(str(path), bgr_float, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                except Exception as e:
                    print(f"  [警告] EXR 保存失败 ({e})，降级为 16bit PNG")
                    # 降级: 转换为 uint16
                    img_u16 = np.clip(img_data * 65535.0, 0, 65535).astype(np.uint16)
                    bgr_u16 = img_u16[:, :, ::-1]
                    path = cam_dir / f"{frame_idx:06d}.png"
                    cv2.imwrite(str(path), bgr_u16)

            elif raw_type == 'uint16':
                # 16bit PNG 格式
                path = cam_dir / f"{frame_idx:06d}.png"
                bgr_u16 = img_data[:, :, ::-1]  # RGB -> BGR
                cv2.imwrite(str(path), bgr_u16)

            else:  # 'uint8' (默认)
                # 8bit PNG 格式
                path = cam_dir / f"{frame_idx:06d}.png"
                bgr_u8 = img_data[:, :, ::-1].astype(np.uint8)  # RGB -> BGR
                cv2.imwrite(str(path), bgr_u8)

    def save_rgb_preview(self, frame_idx, data_dict):
        """
        保存彩色 RGB PNG 预览图（用于 occ_network_lite）
        
        说明：
        - 从 Bayer 单通道数据去马赛克生成彩色 RGB 图像
        - 保存为 8-bit PNG（标准格式）
        """
        for cam_id, info in data_dict.items():
            cam_dir = self.cameras_dir / cam_id
            cam_dir.mkdir(exist_ok=True)

            bayer_data = info['data']
            raw_type = info.get('raw_type', 'uint8')

            # 只处理 Bayer 数据
            if raw_type == 'bayer_rggb':
                # Bayer RGGB -> RGB (Demosaic)
                # OpenCV cvtColor 支持 uint16
                # COLOR_BayerBG2BGR (因为 CARLA RGGB 对应 OpenCV BayerBG? 
                # 通常: CARLA (0,0) is R -> OpenCV BayerBG (0,0) is B? No.
                # OpenCV Pattern naming: 
                # BayerBG: Line 0: B G ... Line 1: G R ...
                # BayerRG: Line 0: R G ... Line 1: G B ...
                # CARLA is RGGB. So we use COLOR_BayerRG2BGR.
                
                rgb_u16 = cv2.cvtColor(bayer_data, cv2.COLOR_BayerRG2BGR)
                
                # 转换为 8-bit (12-bit range [0, 4095] -> 8-bit [0, 255])
                # scale = 255 / 4095 = 1 / 16.05... approx 1/16
                rgb_u8 = (rgb_u16 / 16).astype(np.uint8)
                
                path_png = cam_dir / f"{frame_idx:06d}.png"
                cv2.imwrite(str(path_png), rgb_u8)

    def save_bayer_as_dng(self, frame_idx, data_dict, camera_configs=None):
        """
        保存单通道 Bayer RGGB 为 12-bit DNG/TIFF 格式

        说明：
        - 单通道 uint16 Bayer 数据降采样到 12-bit（右移 4 位）
        - 保存为单通道 16-bit TIFF 格式（DNG 兼容）
        - 实际数据范围: [0, 4095] (12-bit)
        - 数据量：比 RGB 少 66%

        Args:
            frame_idx: 帧索引
            data_dict: {cam_id: {'data': bayer_array, 'raw_type': str, ...}}
                bayer_array: (H, W) uint16 单通道
            camera_configs: 相机配置列表（用于获取 bit_depth）
        """
        for cam_id, info in data_dict.items():
            cam_dir = self.cameras_dir / cam_id
            cam_dir.mkdir(exist_ok=True)

            bayer_data = info['data']  # 单通道 (H, W)
            raw_type = info.get('raw_type', 'uint8')

            # ========== 跳过非 Bayer 数据 ==========
            if raw_type != 'bayer_rggb':
                print(f"  [跳过] {cam_id}: 不是 Bayer 数据（raw_type='{raw_type}'）")
                continue

            # ========== 检查形状 ==========
            if len(bayer_data.shape) != 2:
                print(f"  [错误] {cam_id}: Bayer 数据必须是单通道 (H, W)，实际 {bayer_data.shape}")
                continue

            # ========== 转换为 uint16 ==========
            bayer_u16 = bayer_data.astype(np.uint16)

            # ========== 降采样到 12-bit ==========
            # 获取目标位深度（从 camera_configs 中查找）
            bit_depth = 12  # 默认 12-bit
            if camera_configs:
                for cfg in camera_configs:
                    if cfg['id'] == cam_id:
                        bit_depth = cfg.get('bit_depth', 12)
                        break

            if bit_depth == 12:
                # 16-bit → 12-bit: 右移 4 位
                # [0, 65535] → [0, 4095]
                bayer_12bit = (bayer_u16 >> 4).astype(np.uint16)
            elif bit_depth == 16:
                # 保持 16-bit
                bayer_12bit = bayer_u16
            else:
                print(f"  [警告] {cam_id}: 不支持的位深度 {bit_depth}，使用 12-bit")
                bayer_12bit = (bayer_u16 >> 4).astype(np.uint16)

            # ========== 保存为 DNG 格式（带完整 TIFF/DNG 头）==========
            path_dng = cam_dir / f"{frame_idx:06d}.dng"

            try:
                # 使用 PIL/Pillow 保存单通道 16-bit TIFF（DNG 兼容）
                if DNG_AVAILABLE:
                    from PIL import Image
                    import piexif

                    # 创建 PIL Image（单通道灰度）
                    img_pil = Image.fromarray(bayer_12bit, mode='I;16')  # 16-bit grayscale

                    # 构建 EXIF/TIFF 元数据（可选）
                    # 这里添加基本的 DNG 识别标签
                    exif_dict = {
                        "0th": {
                            piexif.ImageIFD.Make: b"CARLA Simulator",
                            piexif.ImageIFD.Model: b"Bayer RGGB Camera",
                            piexif.ImageIFD.Software: b"CARLA Dense Occupancy Collection",
                            piexif.ImageIFD.PhotometricInterpretation: 32803,  # CFA (Color Filter Array)
                            piexif.ImageIFD.SamplesPerPixel: 1,
                            piexif.ImageIFD.BitsPerSample: (bit_depth,),  # 12 or 16
                        }
                    }
                    exif_bytes = piexif.dump(exif_dict)

                    # 保存为 TIFF（DNG 本质上是特殊的 TIFF）
                    img_pil.save(str(path_dng), format='TIFF', compression='none', exif=exif_bytes)

                else:
                    # 降级: 使用 OpenCV 保存为 TIFF 后重命名
                    path_tif = cam_dir / f"{frame_idx:06d}.tif"
                    success = cv2.imwrite(str(path_tif), bayer_12bit, [
                        cv2.IMWRITE_TIFF_COMPRESSION, 1  # 无压缩
                    ])
                    if not success:
                        raise IOError("cv2.imwrite returned False")

                    # 重命名为 .dng
                    if path_dng.exists():
                        path_dng.unlink()
                    path_tif.rename(path_dng)

            except Exception as e:
                print(f"  [错误] {cam_id} Bayer DNG保存失败: {e}")
                # Fallback: 保存为 PNG
                path_png = cam_dir / f"{frame_idx:06d}.png"
                cv2.imwrite(str(path_png), bayer_12bit)

    def save_rgb_as_dng(self, frame_idx, data_dict, camera_configs=None):
        """
        保存 RGB 图像为 12-bit DNG/TIFF 格式

        说明：
        - 从 16-bit uint16 数据降采样到 12-bit（右移 4 位）
        - 保存为 16-bit TIFF 格式（DNG 兼容）
        - 实际数据范围: [0, 4095] (12-bit)
        - 需要安装: pip install Pillow piexif (可选)

        Args:
            frame_idx: 帧索引
            data_dict: {cam_id: {'data': rgb_array, 'raw_type': str, ...}}
            camera_configs: 相机配置列表（用于获取 bit_depth）
        """
        for cam_id, info in data_dict.items():
            cam_dir = self.cameras_dir / cam_id
            cam_dir.mkdir(exist_ok=True)

            img_data = info['data']  # RGB格式
            raw_type = info.get('raw_type', 'uint8')

            # ========== 跳过 uint8 数据 ==========
            if raw_type == 'uint8':
                print(f"  [跳过] {cam_id}: uint8 数据不适合 DNG 格式（请使用 raw_type='uint16'）")
                continue

            # ========== 转换为 uint16 ==========
            if raw_type == 'float32':
                # HDR → 线性 uint16 (假设 HDR 范围 [0, 1])
                img_u16 = np.clip(img_data * 65535.0, 0, 65535).astype(np.uint16)
            else:  # uint16
                img_u16 = img_data.astype(np.uint16)

            # ========== 降采样到 12-bit ==========
            # 获取目标位深度（从 camera_configs 中查找）
            bit_depth = 12  # 默认 12-bit
            if camera_configs:
                for cfg in camera_configs:
                    if cfg['id'] == cam_id:
                        bit_depth = cfg.get('bit_depth', 12)
                        break

            if bit_depth == 12:
                # 16-bit → 12-bit: 右移 4 位
                # [0, 65535] → [0, 4095]
                img_12bit = (img_u16 >> 4).astype(np.uint16)
            elif bit_depth == 16:
                # 保持 16-bit
                img_12bit = img_u16
            else:
                print(f"  [警告] {cam_id}: 不支持的位深度 {bit_depth}，使用 12-bit")
                img_12bit = (img_u16 >> 4).astype(np.uint16)

            # ========== 保存为 TIFF 格式（DNG 兼容）==========
            # 文件扩展名使用 .tif（因为 OpenCV 4.9 可能不支持 .dng 写入）
            # 用户可以后续批量重命名为 .dng
            path_tif = cam_dir / f"{frame_idx:06d}.tif"
            
            # RGB → BGR (OpenCV 要求)
            bgr_12bit = img_12bit[:, :, ::-1]

            # 保存为 16-bit TIFF（无压缩）
            # 注意：虽然数据是 12-bit [0, 4095]，但仍存储为 uint16 容器
            try:
                success = cv2.imwrite(str(path_tif), bgr_12bit, [
                    cv2.IMWRITE_TIFF_COMPRESSION, 1  # 无压缩
                ])
                if not success:
                    raise IOError("cv2.imwrite returned False")
                
                # 重命名为 .dng (如果需要)
                path_dng = cam_dir / f"{frame_idx:06d}.dng"
                if path_dng.exists():
                    path_dng.unlink() # 删除旧文件
                path_tif.rename(path_dng)
                
            except Exception as e:
                print(f"  [错误] {cam_id} DNG保存失败: {e}")
                # Fallback: 保存为 PNG
                path_png = cam_dir / f"{frame_idx:06d}.png"
                cv2.imwrite(str(path_png), bgr_12bit)

    def save_depth(self, frame_idx, depth_data):
        """
        保存深度图 (16-bit PNG)
        depth_data: {'depth_maps': (6, H, W) float32, ...}
        """
        depth_maps = depth_data['depth_maps']
        # Config 顺序: Front, Right, Back, Left, Up, Down
        cam_ids = ['depth_front', 'depth_right', 'depth_back', 'depth_left', 'depth_up', 'depth_down']
        
        depth_dir_root = self.output_dir / 'depth'
        depth_dir_root.mkdir(parents=True, exist_ok=True)
        
        for i, cam_id in enumerate(cam_ids):
            # 保存每个相机的深度图
            # 格式: 16-bit PNG (单位: 毫米)
            # float meters -> uint16 millimeters
            d_map = depth_maps[i] # (H, W)
            d_mm = (d_map * 1000.0).astype(np.uint16)
            
            cam_dir = depth_dir_root / cam_id
            cam_dir.mkdir(exist_ok=True)
            
            path = cam_dir / f"{frame_idx:06d}.png"
            cv2.imwrite(str(path), d_mm)

    def save_voxel(self, frame_idx, occupancy, actor_ids, metadata=None):
        """
        保存体素数据 (npz)

        注意: mask 字段已移除，使用 Label 0 (Free) 表示不可见/空白区域
        """
        path = self.occupancy_dir / f"{frame_idx:06d}.npz"

        save_dict = {
            'occupancy': occupancy,
            'actor_ids': actor_ids,
        }
        if metadata:
            save_dict.update(metadata)

        np.savez_compressed(path, **save_dict)
        return path

    def save_camera_params(self, frame_idx, camera_configs, intrinsics, extrinsics):
        """
        保存相机参数

        Args:
            frame_idx: 帧索引
            camera_configs: 相机配置列表（8个相机）
            intrinsics: [8, 3, 3] 内参矩阵
            extrinsics: [8, 4, 4] 外参矩阵（世界 -> 相机）
        """
        path = self.camera_params_dir / f"{frame_idx:06d}.npz"

        # 保存配置为 JSON 字符串（方便阅读）
        configs_json = json.dumps(camera_configs, indent=2)

        np.savez_compressed(
            path,
            intrinsics=intrinsics.astype(np.float32),
            extrinsics=extrinsics.astype(np.float32),
            configs=configs_json  # JSON 字符串
        )

        return path

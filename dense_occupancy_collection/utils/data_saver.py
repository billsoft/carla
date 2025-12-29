"""
数据保存工具 (Data Saver)
"""
import numpy as np
import cv2
import json
from pathlib import Path

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
        保存 RGB 图像
        data_dict: {cam_id: {'data': rgb_array, ...}}
        """
        for cam_id, info in data_dict.items():
            cam_dir = self.cameras_dir / cam_id
            cam_dir.mkdir(exist_ok=True)
            
            # RGB -> BGR
            bgr = info['data'][:, :, ::-1]
            path = cam_dir / f"{frame_idx:06d}.png"
            cv2.imwrite(str(path), bgr)

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

    def save_voxel(self, frame_idx, occupancy, actor_ids, mask, metadata=None):
        """保存体素数据 (npz)"""
        path = self.occupancy_dir / f"{frame_idx:06d}.npz"

        save_dict = {
            'occupancy': occupancy,
            'actor_ids': actor_ids,
            'mask': mask
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

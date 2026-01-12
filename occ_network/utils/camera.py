import torch
import torch.nn as nn
import math
import numpy as np

def build_intrinsic(fov, width, height):
    fx = width / (2 * math.tan(math.radians(fov / 2)))
    fy = fx
    cx, cy = width / 2, height / 2
    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

def build_extrinsic(position, rotation):
    x, y, z = position
    pitch, roll, yaw = [math.radians(r) for r in rotation]
    Rx = torch.tensor([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]], dtype=torch.float32)
    Ry = torch.tensor([[math.cos(roll), 0, math.sin(roll)], [0, 1, 0], [-math.sin(roll), 0, math.cos(roll)]], dtype=torch.float32)
    Rz = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]], dtype=torch.float32)
    R = Rz @ Ry @ Rx
    t = torch.tensor([[x], [y], [z]], dtype=torch.float32)
    extrinsic = torch.eye(4, dtype=torch.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3:4] = t
    return extrinsic

def build_camera_matrices(camera_configs, image_size):
    intrinsics, extrinsics = {}, {}
    for cam_name, cfg in camera_configs.items():
        intrinsics[cam_name] = build_intrinsic(cfg['fov'], image_size[1], image_size[0])
        extrinsics[cam_name] = build_extrinsic(cfg['position'], cfg['rotation'])
    return intrinsics, extrinsics

def project_points_to_image(points_3d, intrinsic, extrinsic):
    points_homo = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=1)
    points_cam = (torch.inverse(extrinsic) @ points_homo.T).T[:, :3]
    points_img = (intrinsic @ points_cam.T).T
    points_2d = points_img[:, :2] / (points_img[:, 2:3] + 1e-6)
    depth = points_cam[:, 2]
    return points_2d, depth

def unproject_image_to_3d(points_2d, depth, intrinsic, extrinsic):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (points_2d[:, 0] - cx) * depth / fx
    y = (points_2d[:, 1] - cy) * depth / fy
    z = depth
    points_cam = torch.stack([x, y, z], dim=1)
    points_homo = torch.cat([points_cam, torch.ones_like(points_cam[:, :1])], dim=1)
    points_world = (extrinsic @ points_homo.T).T[:, :3]
    return points_world

def get_ray_directions(height, width, intrinsic):
    i, j = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
    i, j = i.float(), j.float()
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    directions = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], dim=-1)
    return directions / directions.norm(dim=-1, keepdim=True)

def voxel_to_world(voxel_coords, pc_range, voxel_resolution):
    x = voxel_coords[:, 0] * voxel_resolution + pc_range[0]
    y = voxel_coords[:, 1] * voxel_resolution + pc_range[1]
    z = voxel_coords[:, 2] * voxel_resolution + pc_range[2]
    return torch.stack([x, y, z], dim=1)

def world_to_voxel(world_coords, pc_range, voxel_resolution):
    x = (world_coords[:, 0] - pc_range[0]) / voxel_resolution
    y = (world_coords[:, 1] - pc_range[1]) / voxel_resolution
    z = (world_coords[:, 2] - pc_range[2]) / voxel_resolution
    return torch.stack([x, y, z], dim=1).long()

class CameraModel:
    def __init__(self, camera_configs, image_size, pc_range, voxel_resolution):
        self.camera_configs = camera_configs
        self.image_size = image_size
        self.pc_range = pc_range
        self.voxel_resolution = voxel_resolution
        self.intrinsics, self.extrinsics = build_camera_matrices(camera_configs, image_size)

    def project_voxels_to_cameras(self, voxel_coords):
        world_coords = voxel_to_world(voxel_coords, self.pc_range, self.voxel_resolution)
        projections = {}
        for cam_name in self.camera_configs:
            points_2d, depth = project_points_to_image(world_coords, self.intrinsics[cam_name], self.extrinsics[cam_name])
            valid = (depth > 0) & (points_2d[:, 0] >= 0) & (points_2d[:, 0] < self.image_size[1]) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < self.image_size[0])
            projections[cam_name] = {'points_2d': points_2d, 'depth': depth, 'valid': valid}
        return projections

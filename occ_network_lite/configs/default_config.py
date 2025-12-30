# configs/default_config.py
"""
Occupancy Network 默认配置
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CameraConfig:
    """相机配置"""
    # 相机ID列表（固定顺序）
    camera_ids: List[str] = field(default_factory=lambda: [
        'cam_front_main',      # 前主摄 50° FOV
        'cam_front_wide',      # 前广角 120° FOV
        'cam_front_narrow',    # 前长焦 35° FOV
        'cam_left_pillar',     # 左B柱 80° FOV
        'cam_right_pillar',    # 右B柱 80° FOV
        'cam_left_repeater',   # 左后视镜 100° FOV
        'cam_right_repeater',  # 右后视镜 100° FOV
        'cam_rear',            # 后摄 120° FOV
    ])

    num_cameras: int = 8

    # 图像输入尺寸
    input_size: Tuple[int, int] = (384, 640)  # (H, W)

    # Tesla FSD 相机配置（默认值，实际从数据集加载）
    camera_specs: List[dict] = field(default_factory=lambda: [
        {'id': 'cam_front_main',    'fov': 50,  'x': 1.0, 'y': 0.0,  'z': 1.6, 'pitch': 0, 'yaw': 0,    'roll': 0},
        {'id': 'cam_front_wide',    'fov': 120, 'x': 1.0, 'y': 0.0,  'z': 1.6, 'pitch': 0, 'yaw': 0,    'roll': 0},
        {'id': 'cam_front_narrow',  'fov': 35,  'x': 1.0, 'y': 0.0,  'z': 1.6, 'pitch': 0, 'yaw': 0,    'roll': 0},
        {'id': 'cam_left_pillar',   'fov': 80,  'x': 0.0, 'y': -0.9, 'z': 1.7, 'pitch': 0, 'yaw': -60,  'roll': 0},
        {'id': 'cam_right_pillar',  'fov': 80,  'x': 0.0, 'y': 0.9,  'z': 1.7, 'pitch': 0, 'yaw': 60,   'roll': 0},
        {'id': 'cam_left_repeater', 'fov': 100, 'x': 1.2, 'y': -0.9, 'z': 1.0, 'pitch': 0, 'yaw': -160, 'roll': 0},
        {'id': 'cam_right_repeater','fov': 100, 'x': 1.2, 'y': 0.9,  'z': 1.0, 'pitch': 0, 'yaw': 160,  'roll': 0},
        {'id': 'cam_rear',          'fov': 120, 'x': -2.5,'y': 0.0,  'z': 1.2, 'pitch': -5,'yaw': 180,  'roll': 0}
    ])

    # 注意: 内参和外参矩阵在数据集中已保存，训练时直接加载
    # 这里的 camera_specs 仅作为参考，不用于实际计算


@dataclass
class OccupancyConfig:
    """体素空间配置"""
    # 空间范围（米）
    x_range: Tuple[float, float] = (-50.0, 50.0)  # 前后 100m
    y_range: Tuple[float, float] = (-50.0, 50.0)  # 左右 100m
    z_range: Tuple[float, float] = (-4.0, 4.0)    # 上下 8m
    
    # 原始分辨率 0.2m → 500×500×40
    # 训练时使用降采样分辨率以节省显存
    train_resolution: float = 0.5    # 训练分辨率 0.5m → 200×200×16
    full_resolution: float = 0.2     # 完整分辨率 0.2m → 500×500×40
    
    # 训练时的网格尺寸
    train_grid_size: Tuple[int, int, int] = (200, 200, 16)
    
    # 完整网格尺寸（用于评估）
    full_grid_size: Tuple[int, int, int] = (500, 500, 40)
    
    # 类别数
    num_classes: int = 18


@dataclass
class BackboneConfig:
    """Backbone 配置"""
    type: str = 'resnet50'  # 'resnet50', 'resnet101', 'efficientnet_b4'
    pretrained: bool = True
    out_indices: Tuple[int, ...] = (1, 2, 3)  # 输出 C3, C4, C5
    frozen_stages: int = 1  # 冻结前几个 stage


@dataclass
class NeckConfig:
    """FPN Neck 配置"""
    in_channels: List[int] = field(default_factory=lambda: [256, 512, 1024])
    out_channels: int = 256
    num_outs: int = 1  # 只输出一个尺度用于 BEV 变换


@dataclass
class ViewTransformerConfig:
    """View Transformer 配置"""
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    
    # BEV 网格配置
    bev_h: int = 200
    bev_w: int = 200
    
    # 特征图尺寸（来自 Backbone）
    feature_h: int = 48
    feature_w: int = 80


@dataclass
class BEVEncoderConfig:
    """BEV Encoder 配置"""
    in_channels: int = 256
    out_channels: int = 256
    num_layers: int = 4


@dataclass
class OccDecoderConfig:
    """Occupancy Decoder 配置"""
    in_channels: int = 256
    hidden_channels: int = 128
    num_classes: int = 18
    num_heights: int = 16  # Z 方向的离散高度数
    use_3d_conv: bool = True


@dataclass
class LossConfig:
    """损失函数配置"""
    ce_weight: float = 0.7
    lovasz_weight: float = 0.3
    
    # 类别权重（处理类别不平衡）
    class_weights: List[float] = field(default_factory=lambda: [
        0.5,   # 0: free
        2.0,   # 1: barrier
        5.0,   # 2: bicycle
        3.0,   # 3: bus
        1.0,   # 4: car
        10.0,  # 5: construction_vehicle
        5.0,   # 6: motorcycle
        3.0,   # 7: pedestrian
        8.0,   # 8: traffic_cone
        10.0,  # 9: trailer
        2.0,   # 10: truck
        0.8,   # 11: driveable_surface
        1.5,   # 12: other_flat
        1.5,   # 13: sidewalk
        1.0,   # 14: terrain
        1.0,   # 15: manmade
        1.0,   # 16: vegetation
        2.0,   # 17: general_object
    ])


@dataclass
class TrainConfig:
    """训练配置"""
    # 数据
    batch_size: int = 2
    num_workers: int = 4
    
    # 优化器
    optimizer: str = 'adamw'
    lr: float = 2e-4
    weight_decay: float = 0.01
    
    # 学习率调度
    lr_scheduler: str = 'cosine'
    warmup_epochs: int = 1
    epochs: int = 24
    
    # 梯度
    grad_clip: float = 35.0
    accumulate_grad_batches: int = 1
    
    # 保存
    save_dir: str = 'checkpoints'
    log_interval: int = 50
    val_interval: int = 1  # 每几个 epoch 验证一次


@dataclass
class Config:
    """完整配置"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    occupancy: OccupancyConfig = field(default_factory=OccupancyConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    view_transformer: ViewTransformerConfig = field(default_factory=ViewTransformerConfig)
    bev_encoder: BEVEncoderConfig = field(default_factory=BEVEncoderConfig)
    occ_decoder: OccDecoderConfig = field(default_factory=OccDecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    # 设备
    device: str = 'cuda'
    
    # 随机种子
    seed: int = 42


def get_config():
    """获取默认配置"""
    return Config()


# nuScenes 17类标签名称
CLASS_NAMES = [
    'free',                  # 0
    'barrier',               # 1
    'bicycle',               # 2
    'bus',                   # 3
    'car',                   # 4
    'construction_vehicle',  # 5
    'motorcycle',            # 6
    'pedestrian',            # 7
    'traffic_cone',          # 8
    'trailer',               # 9
    'truck',                 # 10
    'driveable_surface',     # 11
    'other_flat',            # 12
    'sidewalk',              # 13
    'terrain',               # 14
    'manmade',               # 15
    'vegetation',            # 16
    'general_object',        # 17
]


# 类别颜色（用于可视化）
CLASS_COLORS = [
    (0, 0, 0),         # 0: free - 黑色
    (200, 200, 200),   # 1: barrier - 灰白
    (128, 128, 0),     # 2: bicycle - 深黄
    (0, 0, 128),       # 3: bus - 深蓝
    (0, 128, 0),       # 4: car - 绿色
    (128, 0, 128),     # 5: construction_vehicle - 紫色
    (128, 0, 0),       # 6: motorcycle - 深红
    (255, 0, 0),       # 7: pedestrian - 红色
    (255, 165, 0),     # 8: traffic_cone - 橙色
    (0, 128, 128),     # 9: trailer - 青色
    (0, 0, 255),       # 10: truck - 蓝色
    (100, 100, 100),   # 11: driveable_surface - 深灰
    (150, 150, 150),   # 12: other_flat - 浅灰
    (255, 192, 203),   # 13: sidewalk - 粉色
    (0, 255, 0),       # 14: terrain - 亮绿
    (255, 255, 0),     # 15: manmade - 黄色
    (0, 255, 128),     # 16: vegetation - 春绿
    (255, 0, 255),     # 17: general_object - 洋红
]

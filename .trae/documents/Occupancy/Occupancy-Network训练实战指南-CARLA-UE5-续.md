# Occupancy Network 训练实战指南（续）：训练、验证与部署

> 接续前文：完整的训练流程、超参数调优、模型部署

---

## 5. Occupancy Network 完整实现 {#网络实现}

### 5.1 完整网络架构整合

```python
# models/occupancy_network_full.py

import torch
import torch.nn as nn
from .regnet_backbone import RegNetY16GF
from .bifpn import BiFPN
from .occupancy_lifting import AttentionBasedLifting
from .temporal_fusion import TemporalFusion
from .occupancy_heads import OccupancyPredictionHead

class CARLAOccupancyNetwork(nn.Module):
    """
    完整的 Occupancy Network 实现

    针对 CARLA 仿真环境优化
    """
    def __init__(
        self,
        backbone='regnet_y_16gf',
        feature_dim=256,
        num_cameras=8,
        voxel_config=None
    ):
        super().__init__()

        self.num_cameras = num_cameras

        # 默认体素配置
        if voxel_config is None:
            voxel_config = {
                'grid_size': (200, 200, 16),
                'voxel_size': 0.5,
                'x_range': (-50, 50),
                'y_range': (-50, 50),
                'z_range': (-2, 6)
            }

        # ===== 1. Backbone =====
        self.backbone = RegNetY16GF()

        # ===== 2. BiFPN =====
        self.bifpn = BiFPN(channels=feature_dim, num_layers=3)

        # ===== 3. 3D Lifting =====
        self.lifting = AttentionBasedLifting(
            feature_dim=feature_dim,
            **voxel_config
        )

        # ===== 4. 时序融合 =====
        self.temporal_fusion = TemporalFusion(
            feature_dim=feature_dim,
            hidden_dim=512
        )

        # ===== 5. 预测头 =====
        self.prediction_head = OccupancyPredictionHead(in_channels=512)

        # 隐藏状态缓存
        self.hidden_state = None

    def forward(
        self,
        cameras,
        camera_params=None,
        ego_motion=None,
        reset_hidden=False
    ):
        """
        前向传播

        输入:
            cameras: (B, N_cams, 3, H, W) - 8 相机图像
            camera_params: dict (可选)
            ego_motion: (B, 4, 4) (可选)
            reset_hidden: bool

        输出:
            dict {
                'occupancy': (B, 200, 200, 16),
                'flow': (B, 200, 200, 16, 3)
            }
        """
        B, N_cams, C, H, W = cameras.shape

        if reset_hidden:
            self.hidden_state = None

        # ===== 1. Backbone 特征提取 =====
        # 将所有相机展平为 batch
        cameras_flat = cameras.view(B * N_cams, C, H, W)

        # 提取多尺度特征
        features = self.backbone(cameras_flat)
        # features = {'P2': ..., 'P3': ..., 'P4': ..., 'P5': ...}

        # ===== 2. BiFPN 融合 =====
        features_fused = self.bifpn(features)

        # 重塑回多相机格式
        for level in features_fused.keys():
            feat = features_fused[level]
            _, C_feat, H_feat, W_feat = feat.shape
            features_fused[level] = feat.view(B, N_cams, C_feat, H_feat, W_feat)

        # ===== 3. 3D 特征提升 =====
        if camera_params is None:
            # 使用默认相机参数 (简化)
            camera_params = self._get_default_camera_params(B, N_cams, cameras.device)

        occupancy_volume = self.lifting(features_fused, camera_params)
        # Shape: (B, 200, 200, 16, 256)

        # ===== 4. 时序融合 =====
        occupancy_sequence = [occupancy_volume]
        if ego_motion is None:
            ego_motion = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(cameras.device)
        ego_motions = [ego_motion]

        fused_occupancy, self.hidden_state = self.temporal_fusion(
            occupancy_sequence,
            ego_motions,
            self.hidden_state
        )

        # ===== 5. 占据预测 =====
        occupancy_prob, occupancy_flow = self.prediction_head(fused_occupancy)

        # 调整形状
        occupancy_prob = occupancy_prob.squeeze(1)  # (B, 200, 200, 16)
        occupancy_flow = occupancy_flow.permute(0, 2, 3, 4, 1)  # (B, 200, 200, 16, 3)

        return {
            'occupancy': occupancy_prob,
            'flow': occupancy_flow
        }

    def _get_default_camera_params(self, B, N_cams, device):
        """生成默认相机参数 (用于快速测试)"""
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N_cams, 1, 1).to(device)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N_cams, 1, 1).to(device)

        return {
            'intrinsics': intrinsics,
            'extrinsics': extrinsics
        }
```

---

## 6. 训练流程与超参数调优 {#训练流程}

### 6.1 训练器实现

```python
# training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
from tqdm import tqdm

from .losses import OccupancyLoss
from .metrics import OccupancyMetrics

class OccupancyTrainer:
    """
    Occupancy Network 训练器

    功能:
    - 管理训练循环
    - 损失计算
    - 指标评估
    - 模型保存
    - W&B 日志
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # 损失函数
        self.criterion = OccupancyLoss(
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
            lovasz_weight=config.get('lovasz_weight', 0.5),
            flow_weight=config.get('flow_weight', 0.1)
        )

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.get('max_lr', 1e-3),
            epochs=config.get('epochs', 100),
            steps_per_epoch=len(train_loader)
        )

        # 评估指标
        self.metrics = OccupancyMetrics()

        # 混合精度训练
        self.use_amp = config.get('mixed_precision', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # W&B 日志
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project=config.get('wandb_project', 'carla-occupancy'),
                config=config
            )
            wandb.watch(model, log_freq=100)

        # 保存路径
        self.save_dir = Path(config.get('save_dir', './checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 当前 epoch
        self.current_epoch = 0
        self.best_val_iou = 0.0

    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()

        epoch_loss = 0.0
        epoch_metrics = {'iou': 0.0, 'precision': 0.0, 'recall': 0.0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 将数据移到 GPU
            cameras = batch['cameras'].to(self.device)
            occupancy_gt = batch['occupancy'].to(self.device)
            flow_gt = batch['flow'].to(self.device)

            # 前向传播
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(cameras)

                # 计算损失
                losses = self.criterion(
                    {
                        'occupancy': outputs['occupancy'],
                        'flow': outputs['flow']
                    },
                    {
                        'occupancy': occupancy_gt,
                        'flow': flow_gt
                    }
                )

                total_loss = losses['total']

            # 反向传播
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            # 计算指标
            with torch.no_grad():
                metrics = self.metrics.compute(
                    outputs['occupancy'],
                    occupancy_gt
                )

            # 累积
            epoch_loss += total_loss.item()
            for key in epoch_metrics.keys():
                epoch_metrics[key] += metrics[key]

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'iou': f"{metrics['iou']:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # W&B 日志
            if self.use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    'train/loss': total_loss.item(),
                    'train/focal_loss': losses['focal'].item(),
                    'train/lovasz_loss': losses['lovasz'].item(),
                    'train/flow_loss': losses['flow'].item(),
                    'train/iou': metrics['iou'],
                    'train/lr': self.scheduler.get_last_lr()[0]
                })

        # 平均指标
        n_batches = len(self.train_loader)
        epoch_loss /= n_batches
        for key in epoch_metrics.keys():
            epoch_metrics[key] /= n_batches

        return epoch_loss, epoch_metrics

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()

        val_loss = 0.0
        val_metrics = {'iou': 0.0, 'precision': 0.0, 'recall': 0.0}

        for batch in tqdm(self.val_loader, desc="Validation"):
            cameras = batch['cameras'].to(self.device)
            occupancy_gt = batch['occupancy'].to(self.device)
            flow_gt = batch['flow'].to(self.device)

            # 前向传播
            outputs = self.model(cameras)

            # 损失
            losses = self.criterion(
                {
                    'occupancy': outputs['occupancy'],
                    'flow': outputs['flow']
                },
                {
                    'occupancy': occupancy_gt,
                    'flow': flow_gt
                }
            )

            # 指标
            metrics = self.metrics.compute(
                outputs['occupancy'],
                occupancy_gt
            )

            val_loss += losses['total'].item()
            for key in val_metrics.keys():
                val_metrics[key] += metrics[key]

        # 平均
        n_batches = len(self.val_loader)
        val_loss /= n_batches
        for key in val_metrics.keys():
            val_metrics[key] /= n_batches

        # W&B 日志
        if self.use_wandb:
            wandb.log({
                'val/loss': val_loss,
                'val/iou': val_metrics['iou'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall']
            })

        return val_loss, val_metrics

    def train(self, num_epochs):
        """完整训练流程"""
        print(f"开始训练 {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # 训练
            train_loss, train_metrics = self.train_epoch()

            # 验证
            val_loss, val_metrics = self.validate()

            # 打印统计
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} | IoU: {train_metrics['iou']:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | IoU: {val_metrics['iou']:.4f}")

            # 保存最佳模型
            if val_metrics['iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['iou']
                self.save_checkpoint('best.pth')
                print(f"  ✓ 保存最佳模型 (IoU: {self.best_val_iou:.4f})")

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pth')

    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_iou': self.best_val_iou
        }

        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_iou = checkpoint['best_val_iou']

        print(f"✓ 加载检查点: {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}, Best IoU: {self.best_val_iou:.4f}")
```

### 6.2 评估指标

```python
# training/metrics.py

import torch
import torch.nn.functional as F

class OccupancyMetrics:
    """
    占据预测评估指标

    指标:
    1. IoU (Intersection over Union)
    2. Precision
    3. Recall
    4. F1 Score
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def compute(self, pred, target):
        """
        计算所有指标

        输入:
            pred: (B, 200, 200, 16) - 预测概率
            target: (B, 200, 200, 16) - Ground Truth (0/1)

        输出: dict
        """
        # 二值化预测
        pred_binary = (pred > self.threshold).float()

        # 计算 TP, FP, FN
        tp = (pred_binary * target).sum().item()
        fp = (pred_binary * (1 - target)).sum().item()
        fn = ((1 - pred_binary) * target).sum().item()

        # IoU
        intersection = tp
        union = tp + fp + fn
        iou = intersection / (union + 1e-6)

        # Precision
        precision = tp / (tp + fp + 1e-6)

        # Recall
        recall = tp / (tp + fn + 1e-6)

        # F1
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
```

### 6.3 训练配置

```yaml
# configs/occupancy_training.yaml

# ===== 模型配置 =====
model:
  backbone: regnet_y_16gf
  feature_dim: 256
  num_cameras: 8

  voxel:
    grid_size: [200, 200, 16]
    voxel_size: 0.5
    x_range: [-50.0, 50.0]
    y_range: [-50.0, 50.0]
    z_range: [-2.0, 6.0]

# ===== 训练配置 =====
training:
  # 优化器
  optimizer: AdamW
  lr: 0.0001
  weight_decay: 0.01

  # 学习率调度
  lr_scheduler: OneCycleLR
  max_lr: 0.001
  epochs: 100

  # 批次大小
  batch_size: 2  # 每块 GPU (V100 32GB)
  accumulation_steps: 16  # 梯度累积,有效 batch_size = 32

  # 混合精度
  mixed_precision: true

  # 损失权重
  loss:
    focal_alpha: 0.25
    focal_gamma: 2.0
    lovasz_weight: 0.5
    flow_weight: 0.1

# ===== 数据配置 =====
data:
  dataset_path: ./data/occupancy
  train_split: 0.9
  val_split: 0.1

  # 数据增强
  augmentation:
    brightness: 0.2
    contrast: 0.2
    gaussian_noise: 0.02
    camera_dropout: 0.1

  # DataLoader
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2

# ===== 日志配置 =====
logging:
  use_wandb: true
  wandb_project: carla-occupancy
  log_interval: 50
  save_dir: ./checkpoints
  save_interval: 10

# ===== 硬件配置 =====
hardware:
  num_gpus: 4  # 多 GPU 训练
  backend: nccl
  gradient_checkpointing: true  # 节省显存
```

### 6.4 训练脚本

```python
# scripts/train.py

import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from dataset.occupancy_dataset import CARLAOccupancyDataset
from models.occupancy_network_full import CARLAOccupancyNetwork
from training.trainer import OccupancyTrainer

def main():
    # ===== 1. 加载配置 =====
    config_path = Path('./configs/occupancy_training.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=== 配置 ===")
    print(yaml.dump(config, default_flow_style=False))

    # ===== 2. 创建数据集 =====
    train_dataset = CARLAOccupancyDataset(
        data_root=config['data']['dataset_path'],
        split='train',
        augment=True
    )

    val_dataset = CARLAOccupancyDataset(
        data_root=config['data']['dataset_path'],
        split='val',
        augment=False
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        prefetch_factor=config['data'].get('prefetch_factor', 2)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    print(f"\n✓ 数据集加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")

    # ===== 3. 创建模型 =====
    model = CARLAOccupancyNetwork(
        backbone=config['model']['backbone'],
        feature_dim=config['model']['feature_dim'],
        num_cameras=config['model']['num_cameras'],
        voxel_config=config['model']['voxel']
    )

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✓ 模型创建完成:")
    print(f"  总参数: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")

    # ===== 4. 创建训练器 =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ 使用设备: {device}")

    trainer = OccupancyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'] | config['logging'],
        device=device
    )

    # ===== 5. 开始训练 =====
    print("\n" + "="*50)
    print("开始训练...")
    print("="*50 + "\n")

    trainer.train(num_epochs=config['training']['epochs'])

    print("\n✓ 训练完成!")


if __name__ == '__main__':
    main()
```

---

## 7. 验证与可视化 {#验证可视化}

### 7.1 3D 占据可视化

```python
# visualization/visualize_occupancy.py

import numpy as np
import open3d as o3d
import torch

class OccupancyVisualizer:
    """
    3D 占据网格可视化

    使用 Open3D 进行 3D 渲染
    """
    def __init__(self, voxel_size=0.5):
        self.voxel_size = voxel_size

    def visualize_occupancy_grid(
        self,
        occupancy_grid,
        flow_grid=None,
        threshold=0.5,
        color_mode='occupancy'
    ):
        """
        可视化 3D 占据网格

        输入:
            occupancy_grid: (200, 200, 16) - 占据概率
            flow_grid: (200, 200, 16, 3) - 运动流 (可选)
            threshold: float - 占据阈值
            color_mode: 'occupancy' | 'flow' | 'height'
        """
        # 找到被占据的体素
        occupied_mask = occupancy_grid > threshold
        occupied_indices = np.argwhere(occupied_mask)

        if len(occupied_indices) == 0:
            print("⚠️ 没有被占据的体素")
            return

        # 转换为世界坐标
        voxel_coords = self._indices_to_coords(occupied_indices)

        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxel_coords)

        # 根据模式设置颜色
        if color_mode == 'occupancy':
            # 根据占据概率着色
            occupied_probs = occupancy_grid[occupied_mask]
            colors = self._prob_to_color(occupied_probs)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        elif color_mode == 'flow' and flow_grid is not None:
            # 根据运动速度着色
            flow_vectors = flow_grid[occupied_mask]
            speeds = np.linalg.norm(flow_vectors, axis=1)
            colors = self._speed_to_color(speeds)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        elif color_mode == 'height':
            # 根据高度着色
            heights = voxel_coords[:, 2]
            colors = self._height_to_color(heights)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # 创建坐标轴
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)

        # 可视化
        o3d.visualization.draw_geometries(
            [pcd, coord_frame],
            window_name="Occupancy Grid Visualization",
            width=1280,
            height=720
        )

    def visualize_flow_field(self, occupancy_grid, flow_grid, threshold=0.5):
        """
        可视化占据流场 (带箭头)
        """
        # 被占据的体素
        occupied_mask = occupancy_grid > threshold
        occupied_indices = np.argwhere(occupied_mask)

        if len(occupied_indices) == 0:
            return

        # 体素坐标
        voxel_coords = self._indices_to_coords(occupied_indices)

        # 流向量
        flow_vectors = flow_grid[occupied_mask]

        # 创建点云 (起点)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxel_coords)

        # 创建箭头
        arrows = []
        for i, (start, vec) in enumerate(zip(voxel_coords, flow_vectors)):
            speed = np.linalg.norm(vec)

            if speed < 0.1:  # 忽略静止体素
                continue

            # 箭头终点
            end = start + vec * 0.5  # 缩放以便可视化

            # 创建圆柱体箭头
            arrow = self._create_arrow(start, end, radius=0.1)
            arrows.append(arrow)

        # 可视化
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        o3d.visualization.draw_geometries(
            [pcd, coord_frame] + arrows,
            window_name="Occupancy Flow Field",
            width=1280,
            height=720
        )

    def _indices_to_coords(self, indices, x_min=-50, y_min=-50, z_min=-2):
        """体素索引 → 世界坐标"""
        coords = np.zeros((len(indices), 3), dtype=np.float32)
        coords[:, 0] = x_min + (indices[:, 0] + 0.5) * self.voxel_size
        coords[:, 1] = y_min + (indices[:, 1] + 0.5) * self.voxel_size
        coords[:, 2] = z_min + (indices[:, 2] + 0.5) * self.voxel_size
        return coords

    def _prob_to_color(self, probs):
        """概率 → 颜色 (绿色到红色)"""
        colors = np.zeros((len(probs), 3))
        colors[:, 0] = probs  # R
        colors[:, 1] = 1 - probs  # G
        return colors

    def _speed_to_color(self, speeds):
        """速度 → 颜色 (蓝色到红色)"""
        # 归一化速度
        max_speed = 20.0  # m/s
        normalized = np.clip(speeds / max_speed, 0, 1)

        colors = np.zeros((len(speeds), 3))
        colors[:, 0] = normalized  # R
        colors[:, 2] = 1 - normalized  # B
        return colors

    def _height_to_color(self, heights):
        """高度 → 颜色 (渐变)"""
        # 归一化高度 [-2, 6] → [0, 1]
        normalized = (heights + 2) / 8.0
        normalized = np.clip(normalized, 0, 1)

        # 使用 jet colormap
        colors = plt.cm.jet(normalized)[:, :3]
        return colors

    def _create_arrow(self, start, end, radius=0.05):
        """创建箭头"""
        # 计算方向和长度
        vec = end - start
        length = np.linalg.norm(vec)
        direction = vec / length

        # 创建圆柱体
        cylinder_height = length * 0.8
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius,
            height=cylinder_height
        )

        # 旋转到正确方向
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.dot(z_axis, direction))
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
                rotation_axis * angle
            )
            cylinder.rotate(rotation_matrix, center=(0, 0, 0))

        # 移动到起点
        cylinder.translate(start + direction * cylinder_height / 2)

        # 设置颜色
        cylinder.paint_uniform_color([1, 0, 0])  # 红色

        return cylinder


# ===== 使用示例 =====
if __name__ == '__main__':
    # 加载模型预测结果
    occupancy = np.load('output_occupancy.npy')  # (200, 200, 16)
    flow = np.load('output_flow.npy')  # (200, 200, 16, 3)

    # 可视化
    visualizer = OccupancyVisualizer(voxel_size=0.5)

    # 方式1: 仅占据网格
    visualizer.visualize_occupancy_grid(
        occupancy,
        color_mode='height'
    )

    # 方式2: 占据 + 流场
    visualizer.visualize_flow_field(occupancy, flow)
```

---

## 8. 模型部署与实时推理 {#模型部署}

### 8.1 ONNX 导出

```python
# deployment/export_onnx.py

import torch
import onnx
from models.occupancy_network_full import CARLAOccupancyNetwork

def export_to_onnx(model_path, output_path, batch_size=1):
    """
    导出模型为 ONNX 格式

    输入:
        model_path: PyTorch 模型路径
        output_path: ONNX 输出路径
        batch_size: 批次大小
    """
    # 加载模型
    model = CARLAOccupancyNetwork()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 准备示例输入
    dummy_cameras = torch.randn(batch_size, 8, 3, 960, 1280)

    # 导出
    torch.onnx.export(
        model,
        dummy_cameras,
        output_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['cameras'],
        output_names=['occupancy', 'flow'],
        dynamic_axes={
            'cameras': {0: 'batch_size'},
            'occupancy': {0: 'batch_size'},
            'flow': {0: 'batch_size'}
        }
    )

    # 验证
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"✓ ONNX 模型已导出: {output_path}")
```

### 8.2 CARLA 实时推理部署

```python
# scripts/deploy_carla.py

import carla
import torch
import numpy as np
import time
from pathlib import Path

from models.occupancy_network_full import CARLAOccupancyNetwork
from carla_interface.sensors.camera_array import CameraArray
from visualization.visualize_occupancy import OccupancyVisualizer

class CARLAOccupancyInference:
    """
    CARLA 实时推理部署

    功能:
    - 实时采集 8 相机图像
    - 运行 Occupancy Network 推理
    - 在 CARLA 中可视化 3D 占据网格
    """
    def __init__(
        self,
        model_path,
        host='localhost',
        port=2000,
        device='cuda'
    ):
        # 连接 CARLA
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # 加载模型
        self.model = CARLAOccupancyNetwork().to(device)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.device = device

        # 车辆和传感器
        self.vehicle = None
        self.camera_array = None

        # 可视化器
        self.visualizer = OccupancyVisualizer()

        print("✓ Occupancy Network 推理系统已初始化")

    def setup_vehicle(self):
        """生成车辆和传感器"""
        # 生成车辆
        vehicle_bp = self.world.get_blueprint_library().filter('model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])

        # 8 相机
        camera_configs = [...]  # 与训练时相同的配置
        self.camera_array = CameraArray(
            self.world,
            self.vehicle,
            camera_configs
        )

        print("✓ 车辆和传感器已生成")

    @torch.no_grad()
    def inference_step(self):
        """单步推理"""
        # 1. 采集相机图像
        camera_frames = self.camera_array.get_latest_frame()
        if camera_frames is None:
            return None

        # 2. 预处理
        cameras_tensor = self._preprocess_cameras(camera_frames)
        cameras_tensor = cameras_tensor.unsqueeze(0).to(self.device)

        # 3. 推理
        start_time = time.time()
        outputs = self.model(cameras_tensor)
        inference_time = time.time() - start_time

        # 4. 后处理
        occupancy = outputs['occupancy'].squeeze(0).cpu().numpy()
        flow = outputs['flow'].squeeze(0).cpu().numpy()

        return {
            'occupancy': occupancy,
            'flow': flow,
            'inference_time': inference_time
        }

    def run_realtime(self, duration=60):
        """实时运行"""
        self.setup_vehicle()

        # 启用自动驾驶
        self.vehicle.set_autopilot(True)

        # 等待传感器稳定
        time.sleep(2.0)

        print(f"\n开始实时推理 ({duration} 秒)...")

        start_time = time.time()
        frame_count = 0

        try:
            while time.time() - start_time < duration:
                # 推理
                result = self.inference_step()

                if result is not None:
                    frame_count += 1

                    # 打印统计
                    if frame_count % 10 == 0:
                        fps = frame_count / (time.time() - start_time)
                        occupied_voxels = (result['occupancy'] > 0.5).sum()

                        print(f"帧 {frame_count} | "
                              f"FPS: {fps:.1f} | "
                              f"推理时间: {result['inference_time']*1000:.1f}ms | "
                              f"占据体素: {occupied_voxels}")

                    # 可视化 (每 30 帧一次)
                    if frame_count % 30 == 0:
                        self.visualizer.visualize_occupancy_grid(
                            result['occupancy'],
                            color_mode='height'
                        )

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n✓ 用户中断")

        finally:
            self.cleanup()

    def _preprocess_cameras(self, camera_frames):
        """预处理相机图像"""
        cameras_list = []
        for cam_name in camera_frames.keys():
            img = camera_frames[cam_name]
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC → CHW
            cameras_list.append(img)

        cameras = np.stack(cameras_list, axis=0)  # (8, 3, H, W)
        return torch.from_numpy(cameras).float()

    def cleanup(self):
        """清理"""
        if self.camera_array:
            self.camera_array.destroy()
        if self.vehicle:
            self.vehicle.destroy()


# ===== 使用示例 =====
if __name__ == '__main__':
    inference_system = CARLAOccupancyInference(
        model_path='./checkpoints/best.pth',
        host='localhost',
        port=2000,
        device='cuda'
    )

    # 实时运行 60 秒
    inference_system.run_realtime(duration=60)
```

---

## 9. 常见问题与调试 {#常见问题}

### 9.1 显存不足

**问题**: `CUDA out of memory`

**解决方案**:
```yaml
# 1. 减小 batch size
batch_size: 1  # 从 2 减到 1

# 2. 启用梯度累积
accumulation_steps: 32  # 增加累积步数

# 3. 启用梯度检查点
gradient_checkpointing: true

# 4. 减小模型尺寸
model:
  feature_dim: 128  # 从 256 减到 128

# 5. 使用混合精度 FP16
mixed_precision: true
```

### 9.2 训练不收敛

**问题**: Loss 不下降或震荡

**检查清单**:
```python
# 1. 检查数据标签是否正确
occupancy_gt = dataset[0]['occupancy']
print(f"占据率: {(occupancy_gt > 0.5).mean():.2%}")  # 应该在 1-10%

# 2. 检查学习率
# 过大 → 震荡, 过小 → 不动
lr: 0.0001  # 起始学习率

# 3. 检查损失权重
loss:
  focal_alpha: 0.25  # 平衡正负样本
  lovasz_weight: 0.5

# 4. 检查数据增强
# 过强的增强可能破坏几何关系
augmentation:
  camera_dropout: 0.0  # 先关闭相机丢弃

# 5. 检查 Batch Normalization
# 确保 batch size > 1
```

### 9.3 推理速度慢

**问题**: FPS < 10

**优化方案**:
```python
# 1. TensorRT 加速
model_trt = convert_to_tensorrt(model)  # 2-3x 加速

# 2. 减小输入分辨率
camera_size: (480, 640)  # 从 (960, 1280) 减半

# 3. 减小体素分辨率
voxel_grid: [100, 100, 8]  # 从 [200, 200, 16] 减半

# 4. 使用更小的 backbone
backbone: regnet_y_8gf  # 从 16gf 减到 8gf
```

---

## 总结

本指南涵盖了 Occupancy Network 的完整训练流程:

1. ✅ **数据采集**: LiDAR 点云 → 体素化 → 占据标签
2. ✅ **模型实现**: RegNet + BiFPN + Attention Lifting + 时序融合
3. ✅ **训练流程**: 损失函数、优化器、学习率调度、混合精度
4. ✅ **验证评估**: IoU、Precision、Recall、3D 可视化
5. ✅ **部署推理**: ONNX 导出、TensorRT 加速、CARLA 实时推理

**与 HydraNet 的对比**:
- HydraNet: 目标检测范式,依赖预定义类别
- Occupancy Network: 空间占据范式,类别无关,检测任何障碍物

**下一步**:
- 在 CARLA 中采集 10000+ 帧数据
- 训练完整模型 (100 epochs)
- 部署到车辆,测试实时性能

---

_接续文档: [拆解特斯拉占位网络Occupancy-Network架构](./拆解特斯拉占位网络Occupancy-Network架构.md)_

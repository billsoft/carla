import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedWeightedCELoss(nn.Module):
    """
    带掩码和类别权重的 Cross Entropy Loss
    
    特点:
    1. 支持 mask: 只计算 mask=True 的体素损失
    2. 支持 class_weights: 对类别不平衡进行加权
    3. 支持忽略索引: ignore_index (默认 255)
    """
    def __init__(self, class_weights=None, ignore_index=255):
        super().__init__()
        
        # 类别权重
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights).float())
        else:
            self.class_weights = None
            
        self.ignore_index = ignore_index

    def forward(self, logits, target, mask=None):
        """
        Args:
            logits: [B, C, X, Y, Z] 预测值 (未经过 softmax)
            target: [B, X, Y, Z] 真实标签 (long)
            mask:   [B, X, Y, Z] 有效区域掩码 (bool/float), 1=有效, 0=无效
            
        Returns:
            loss: 标量损失
        """
        # 如果有 mask，将无效区域的 target 设为 ignore_index
        if mask is not None:
            # 确保 mask 和 target 形状一致
            if mask.shape != target.shape:
                raise ValueError(f"Mask shape {mask.shape} != Target shape {target.shape}")
                
            # 克隆 target 以免修改原始数据
            target = target.clone()
            
            # 将 mask=0 的位置设为 ignore_index
            # mask 可能是 bool 或 float (0.0/1.0)
            if mask.dtype == torch.bool:
                target[~mask] = self.ignore_index
            else:
                target[mask < 0.5] = self.ignore_index

        # 计算 Cross Entropy Loss
        # F.cross_entropy 已经内置了 softmax
        loss = F.cross_entropy(
            logits, 
            target, 
            weight=self.class_weights, 
            ignore_index=self.ignore_index, 
            reduction='mean'
        )
        
        return loss

def get_default_class_weights():
    """
    获取默认的类别权重 (基于 occ_loss.py)
    
    背景类 (0) 权重较低 (0.5)
    稀有类 (自行车、摩托车等) 权重较高 (5.0-10.0)
    常见类 (汽车、路面) 权重中等 (1.0-2.0)
    """
    # 假设 18 个类别 (0-17)
    # 0: Unlabeled/Free (背景)
    # 1: Building
    # 2: Fence
    # 3: Other
    # 4: Pedestrian
    # 5: Pole
    # 6: RoadLine
    # 7: Road
    # 8: Sidewalk
    # 9: Vegetation
    # 10: Vehicles (Car)
    # 11: Wall
    # 12: TrafficSign
    # 13: Sky (通常被过滤掉)
    # 14: Ground
    # 15: Bridge
    # 16: RailTrack
    # 17: GuardRail
    # ... 以及其他可能映射的类别
    
    # 这里定义一个简化的权重列表，基于常见频率
    # 修复: Class 0 权重从 0.1 提高到 1.0, 避免网络完全忽略空白类
    weights = [
        1.0,  # 0: Free/Unlabeled (空白类,重要的几何信息!)
        1.0,  # 1: Building
        1.0,  # 2: Fence
        1.0,  # 3: Other
        5.0,  # 4: Pedestrian (稀有且重要)
        2.0,  # 5: Pole
        2.0,  # 6: RoadLine
        1.0,  # 7: Road
        1.0,  # 8: Sidewalk
        1.0,  # 9: Vegetation
        2.0,  # 10: Vehicles (重要)
        1.0,  # 11: Wall
        5.0,  # 12: TrafficSign
        0.5,  # 13: Sky (降低权重,通常在视野外)
        1.0,  # 14: Ground
        1.0,  # 15: Bridge
        1.0,  # 16: RailTrack
        1.0,  # 17: GuardRail
    ]
    
    return weights

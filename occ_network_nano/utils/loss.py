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
    获取默认的类别权重 (基于 nuScenes 18类标准)

    权重策略:
    - 空白类 (free): 权重 1.0 (重要! 提供场景几何信息,不可忽略)
    - 稀有但关键的类 (自行车、摩托车、行人、交通锥): 权重 3.0-5.0
    - 中等重要的类 (车辆、护栏): 权重 2.0
    - 常见类 (路面、植被): 权重 1.0

    ⚠️ 修改历史:
    - 2025-12-30: 对齐 dense_occupancy_collection 的 18类映射 (nuScenes 标准)
    - 2025-12-30: Class 0 权重 0.1 → 1.0 (修复网络完全忽略空白类的问题)

    类别定义表 (基于 dense_occupancy_collection/config/actor_occupancy_mapping.py):
    ┌──────┬──────────────────────┬────────┬─────────────────────────────────┐
    │ ID   │ 类别名称             │ 权重   │ 说明                            │
    ├──────┼──────────────────────┼────────┼─────────────────────────────────┤
    │  0   │ free                 │  1.0   │ 自由空间 (几何信息)             │
    │  1   │ barrier              │  2.0   │ 隔离栏/护栏                     │
    │  2   │ bicycle              │  5.0   │ 自行车 (稀有且重要)             │
    │  3   │ bus                  │  2.0   │ 公交车                          │
    │  4   │ car                  │  2.0   │ 小汽车 (重要目标)               │
    │  5   │ construction_vehicle │  3.0   │ 工程车 (较稀有)                 │
    │  6   │ motorcycle           │  5.0   │ 摩托车 (稀有且重要)             │
    │  7   │ pedestrian           │  5.0   │ 行人 (最重要的目标)             │
    │  8   │ traffic_cone         │  3.0   │ 交通锥桶 (较稀有)               │
    │  9   │ trailer              │  2.0   │ 拖车                            │
    │ 10   │ truck                │  2.0   │ 卡车                            │
    │ 11   │ driveable_surface    │  1.0   │ 可行驶路面                      │
    │ 12   │ other_flat           │  1.0   │ 其他平坦表面                    │
    │ 13   │ sidewalk             │  1.0   │ 人行道                          │
    │ 14   │ terrain              │  1.0   │ 地形 (草地、泥地)               │
    │ 15   │ manmade              │  1.0   │ 人造物体 (建筑、标志)           │
    │ 16   │ vegetation           │  1.0   │ 植被 (树、草)                   │
    │ 17   │ general_object       │  1.0   │ 通用障碍物/其他                 │
    └──────┴──────────────────────┴────────┴─────────────────────────────────┘

    Returns:
        List[float]: 18个类别的权重列表
    """
    weights = [
        1.0,  # 0: free - 自由空间 (重要几何信息!)
        2.0,  # 1: barrier - 隔离栏/护栏
        5.0,  # 2: bicycle - 自行车 (稀有且重要)
        2.0,  # 3: bus - 公交车
        2.0,  # 4: car - 小汽车 (重要目标)
        3.0,  # 5: construction_vehicle - 工程车 (较稀有)
        5.0,  # 6: motorcycle - 摩托车 (稀有且重要)
        5.0,  # 7: pedestrian - 行人 (最重要!)
        3.0,  # 8: traffic_cone - 交通锥桶 (较稀有)
        2.0,  # 9: trailer - 拖车
        2.0,  # 10: truck - 卡车
        1.0,  # 11: driveable_surface - 可行驶路面
        1.0,  # 12: other_flat - 其他平坦表面
        1.0,  # 13: sidewalk - 人行道
        1.0,  # 14: terrain - 地形 (草地、泥地)
        1.0,  # 15: manmade - 人造物体 (建筑、标志)
        1.0,  # 16: vegetation - 植被
        1.0,  # 17: general_object - 通用障碍物/其他
    ]

    return weights

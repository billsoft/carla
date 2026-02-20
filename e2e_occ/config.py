from dataclasses import dataclass
from typing import Tuple

@dataclass
class E2EOccConfig:
    """
    端到端占用网络(End-to-End Occupancy)配置
    """
    num_cameras: int = 8                    # 相机数量
    image_size: Tuple[int, int] = (960, 1280) # 图像尺寸 (高, 宽)
    raw_channels: int = 1                   # 原始图像通道数 (例如, 1表示灰度图)
    
    # --- 模型容量与性能平衡配置 (目标显存占用: 18-20GB) ---
    embed_dim: int = 256                    # Transformer的嵌入维度 (从 384 降为 256 以优化性能)
    num_heads: int = 8                      # 多头注意力机制的头数
    encoder_layers: int = 2                 # Transformer编码器层数 (从 4 降为 2 以提高效率)
    decoder_layers: int = 2                 # Transformer解码器层数 (从 3 降为 2)
    
    # --- 分辨率优化配置 ---
    coarse_size: Tuple[int, int, int] = (25, 25, 8)    # 粗查询网格尺寸 (长, 宽, 高), 将查询点从20K减少到5K, 修复自注意力机制中的内存溢出(OOM)问题
    fine_size: Tuple[int, int, int] = (80, 80, 16)     # 精细查询网格尺寸 (长, 宽, 高), 对应102.4K个查询点
    voxel_size: Tuple[int, int, int] = (400, 400, 32)  # 体素尺寸
    
    # --- 模型通用超参数 ---
    num_classes: int = 18                   # 语义分割的类别数量
    num_sample_points: int = 4              # 可变形注意力中的采样点数
    dropout: float = 0.1                    # Dropout比率, 用于防止过拟合
    
    # --- 功能开关 ---
    use_ray_encoding: bool = True           # 是否使用光线位置编码
    use_self_attention: bool = True         # 粗查询阶段是否使用自注意力机制
    use_fine_self_attention: bool = False   # 精细查询阶段是否使用自注意力机制 (102K queries，默认关闭防止 OOM)
    
    # --- 时序融合设置 ---
    use_temporal: bool = True               # 是否启用时序融合
    use_ego_motion: bool = True             # 是否启用自车运动补偿对齐
    temporal_frames: int = 2                # 用于时序融合的帧数 (从 4 降为 2 以减少内存使用)
    memory_dim: int = 256                   # 记忆模块的维度, 应与 embed_dim 保持一致
    
    # --- 几何与体素定义 ---
    voxel_range: Tuple[float, ...] = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4) # 感知范围/体素范围 (xmin, ymin, zmin, xmax, ymax, zmax)，单位为米
    voxel_resolution: float = 0.2           # 体素分辨率, 每个体素的大小 (米)
    
    @property
    def feat_size(self) -> Tuple[int, int]:
        """
        计算并返回由图像主干网络输出的特征图尺寸 (高, 宽)。
        通常是输入图像尺寸除以主干网络的步长 (这里是16)。
        """
        return (self.image_size[0] // 16, self.image_size[1] // 16)
    
    @property
    def num_coarse_queries(self) -> int:
        """计算并返回粗查询的总数。"""
        return self.coarse_size[0] * self.coarse_size[1] * self.coarse_size[2]
    
    @property
    def num_fine_queries(self) -> int:
        """计算并返回精细查询的总数。"""
        return self.fine_size[0] * self.fine_size[1] * self.fine_size[2]

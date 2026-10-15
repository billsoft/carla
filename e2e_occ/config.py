from dataclasses import dataclass
from typing import Tuple

@dataclass
class E2EOccConfig:
    """
    端到端占用网络(End-to-End Occupancy)配置

    2026-10 重构说明：几何坐标系修正（P0）无开关、始终生效；其余优化（P1 多尺度/深度辅助/
    流场/可见性归一化，P2 稀疏细化/局部时序注意力/隐式头，P3 RMSNorm/SwiGLU/3D RoPE/移位窗口）
    都有独立开关，便于和基线做消融对比。旧 checkpoint 的 pickled config 缺少新字段时，
    dataclass 的类属性默认值会自动兜底（getattr 能取到），但旧权重与新结构不兼容，需重新训练。
    """
    num_cameras: int = 8                    # 相机数量
    image_size: Tuple[int, int] = (960, 1280) # 图像尺寸 (高, 宽)
    raw_channels: int = 1                   # 原始图像通道数 (例如, 1表示灰度图)

    # --- 模型容量与性能平衡配置 ---
    embed_dim: int = 256                    # Transformer的嵌入维度
    num_heads: int = 8                      # 多头注意力机制的头数
    encoder_layers: int = 2                 # 图像编码器窗口注意力层数
    decoder_layers: int = 2                 # 粗/细解码器各自的层数

    # --- 分辨率配置 ---
    coarse_size: Tuple[int, int, int] = (25, 25, 8)    # 粗查询网格 (X, Y, Z)，5,000 个查询
    fine_size: Tuple[int, int, int] = (80, 80, 16)     # 细查询网格 (X, Y, Z)，102,400 个查询
    voxel_size: Tuple[int, int, int] = (400, 400, 32)  # 输出体素网格 (X, Y, Z)

    # --- 模型通用超参数 ---
    num_classes: int = 18                   # 语义类别数
    num_sample_points: int = 4              # 可变形注意力每个(相机,头,层级)的采样点数
    deform_offset_px: float = 4.0           # 可变形采样偏移的最大幅度（单位：该层级特征图像素）
    dropout: float = 0.1

    # --- 基础功能开关 ---
    use_ray_encoding: bool = True           # 射线方向编码
    use_self_attention: bool = True         # 粗阶段查询自注意力
    use_fine_self_attention: bool = False   # 细阶段查询自注意力（稠密 102K 查询会 OOM；稀疏模式下可考虑开启）

    # --- 时序融合 ---
    use_temporal: bool = True
    use_ego_motion: bool = True
    temporal_frames: int = 4                # 训练序列长度（原 2；序列不跨场景）
    sequence_stride: int = 0                # 相邻训练序列起点间隔（帧）；0 = 等于 temporal_frames（不重叠，
                                            # 每帧每 epoch 只处理一次）；1 = 全部重叠（每帧处理 T 次，epoch 耗时×T）
    memory_dim: int = 256
    frame_dt: float = 0.05                  # 相邻帧时间间隔（秒）的默认值；数据集 scenes/<scene>.json 有 frame_dt
                                            # (抽帧采集) 时以数据集为准，按样本作为 time_delta 传入

    # --- 数据 ---
    data_strict: bool = True                # 数据缺件直接报错 (见 dataset.py 文件头)；调试旧数据时设 False 降级并告警

    # --- 真值监督范围（相机可见性掩码，见 dataset.py 文件头）---
    use_camera_mask: bool = True            # 相机没观测到的体素不监督（Occ3D 约定），需要 mask_camera/
    camera_mask_keep_ground_surface: bool = True   # 被车/人挡住的最上层地面体素仍监督（路面连续可推断）

    # --- 几何与体素定义（车辆坐标系：X前 Y右 Z上，CARLA 左手系，原点为车辆 transform 原点）---
    voxel_range: Tuple[float, ...] = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)
    voxel_resolution: float = 0.2

    # ===================== P1 =====================
    use_multiscale: bool = True             # 多尺度图像特征（stride 8/16/32），可变形注意力跨层级采样
    use_depth_aux: bool = True              # 深度辅助监督（训练时生效，推理无开销）
    depth_bins: int = 64                    # 深度分类 bin 数（对数均匀）
    depth_min: float = 1.0                  # 米
    depth_max: float = 60.0                 # 米
    depth_loss_weight: float = 1.0
    use_flow_head: bool = True              # 占用流（场景流）回归头
    flow_loss_weight: float = 1.0
    cam_visibility_norm: bool = True        # 跨相机融合按可见相机数归一化

    # ===================== P2 =====================
    use_sparse_fine: bool = True            # 稀疏细化：只对粗阶段预测占用概率最高的 top-k 细体素做交叉注意力
    fine_keep_ratio: float = 0.25           # 稀疏细化保留比例
    # 稀疏细化的选点方式：'topk' = 只按粗占用概率取前 K (基线)；'mixed' = 名额按 fine_select_quota 拆成
    # (占用概率最高, 占用最不确定, 离自车最近) 三份。路面/建筑占满 top-k 名额、粗阶段拿不准的小物体
    # 反而没被细化是 'topk' 的已知风险，'mixed' 用于对照实验，先用验证集的小目标召回/近场漏报决定是否换默认
    fine_select_mode: str = 'topk'
    fine_select_quota: Tuple[float, ...] = (0.5, 0.25, 0.25)
    coarse_occ_loss_weight: float = 1.0     # 粗阶段占用概率（稀疏选择依据）的 BCE 损失权重
    temporal_attn_type: str = 'deformable'  # 'deformable'（对齐后局部采样）| 'global'（原 5000×5000 全局注意力）
    temporal_points: int = 8                # 局部时序注意力每头采样点数
    temporal_offset_cells: float = 2.0      # 局部时序采样偏移的最大幅度（粗体素格）
    use_time_embedding: bool = True         # 帧间隔时间嵌入
    head_type: str = 'dense'                # 'dense'（3D 卷积上采样头）| 'implicit'（可查询 MLP 隐式头）
    checkpoint_head: bool = True            # 稠密头训练时梯度检查点（400×400×32 激活单帧数 GB）
    lovasz_max_points: int = 1000000        # Lovász 损失最多使用的体素数（超出时保留全部非空体素 + 随机空体素）
    implicit_train_points: int = 262144     # 隐式头训练时每帧采样点数
    implicit_occupied_ratio: float = 0.5    # 采样点中来自"非空"体素的比例
    implicit_hidden: int = 128

    # ===================== P3 =====================
    norm_type: str = 'rmsnorm'              # 'rmsnorm' | 'layernorm'
    ffn_type: str = 'swiglu'                # 'swiglu' | 'gelu'
    use_rope_3d: bool = True                # 粗阶段自注意力使用 3D RoPE
    use_shifted_window: bool = True         # 图像编码器交替使用移位窗口（Swin）
    window_size: int = 7

    @property
    def feat_size(self) -> Tuple[int, int]:
        """stride-16 主层级特征图尺寸 (高, 宽)。"""
        return (self.image_size[0] // 16, self.image_size[1] // 16)

    @property
    def level_strides(self) -> Tuple[int, ...]:
        return (8, 16, 32) if self.use_multiscale else (16,)

    def level_size(self, stride: int) -> Tuple[int, int]:
        return (self.image_size[0] // stride, self.image_size[1] // stride)

    @property
    def depth_stride(self) -> int:
        """深度辅助监督所在层级的 stride（多尺度时用最细的 stride 8）。"""
        return self.level_strides[0]

    @property
    def num_coarse_queries(self) -> int:
        return self.coarse_size[0] * self.coarse_size[1] * self.coarse_size[2]

    @property
    def num_fine_queries(self) -> int:
        return self.fine_size[0] * self.fine_size[1] * self.fine_size[2]


def upgrade_config(cfg):
    """checkpoint 里存的是 pickle 的配置对象：旧 checkpoint 缺少后来新增的字段 (如 use_camera_mask)，
    --set 覆盖或 inference 读字段时会 AttributeError。按当前字段表重建，缺的字段用默认值。"""
    if cfg is None:
        return E2EOccConfig()
    default = E2EOccConfig()
    return E2EOccConfig(**{k: getattr(cfg, k, getattr(default, k)) for k in E2EOccConfig.__dataclass_fields__})

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelHead(nn.Module):
    """
    体素输出头（交织降维+上采样方案）

    信息流：
        256@80 → 128@80 → 64@80
               → 64→32@200（插值+3×3精化+降通道）
               → 32→18@400（插值+3×3精化+最终分类）

    关键改进：
    - 把分类决策推迟到最高分辨率（400×400）上做
    - 每次上采样后用高通道数（64/32）做 3×3 空间精化，再降通道
    - 最终分类在 400×400 全分辨率上完成，边缘判断能力更强
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim  # 256
        nc = config.num_classes  # 18

        # 阶段一：低分辨率降维（80×80×16）
        self.conv1 = nn.Sequential(
            nn.Conv3d(dim, dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(dim // 4),
            nn.GELU(),
        )

        # 阶段二：插值到中间分辨率（200×200×32），64通道精化并降至32通道
        # 3×3 卷积同时完成空间精化和通道降维
        self.refine1 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(dim // 8),
            nn.GELU(),
        )
        self.skip1 = nn.Conv3d(dim // 4, dim // 8, kernel_size=1)  # 残差通道对齐 64→32

        # 阶段三：插值到最终分辨率（400×400×32），32通道精化并直接分类到18类
        # 最终分类在全分辨率上完成，不加激活函数
        self.refine2 = nn.Sequential(
            nn.Conv3d(dim // 8, nc, kernel_size=3, padding=1),
            nn.BatchNorm3d(nc),
        )
        self.skip2 = nn.Conv3d(dim // 8, nc, kernel_size=1)  # 残差通道对齐 32→18

    def forward(self, x):
        # x: [B, fx, fy, fz, C]（来自 OccupancyDecoder）
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # → [B, C, fx, fy, fz]

        vx, vy, vz = self.config.voxel_size          # (400, 400, 32)
        mid_size = (vx // 2, vy // 2, vz)            # (200, 200, 32)

        # 阶段一：80×80×16 降维
        x = self.conv1(x)   # [B, 128, 80, 80, 16]
        x = self.conv2(x)   # [B,  64, 80, 80, 16]

        # 阶段二：插值到 200×200×32，64通道精化→32通道
        x = F.interpolate(x, size=mid_size, mode='trilinear', align_corners=False)
        x = self.refine1(x) + self.skip1(x)   # [B, 32, 200, 200, 32]

        # 阶段三：插值到 400×400×32，32通道精化→18类（全分辨率分类）
        x = F.interpolate(x, size=(vx, vy, vz), mode='trilinear', align_corners=False)
        x = self.refine2(x) + self.skip2(x)   # [B, 18, 400, 400, 32]

        return x

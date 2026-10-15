import torch.nn as nn


class RAWPatchEmbed(nn.Module):
    """Bayer RAW → stride 8 / stride 16 两个层级的特征图（共享权重处理 8 路相机）。

    rggb_conv 是可学习的 2×2/stride-2 卷积：每次恰好覆盖一个完整 RGGB 单元，相当于
    可学习的去马赛克；不加 bias 以保持对亮度的线性响应。
    """

    def __init__(self, embed_dim=256):
        super().__init__()
        self.rggb_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)   # stride 2
        self.stage1 = nn.Sequential(                                              # stride 4
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(                                              # stride 8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(                                              # stride 16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x):
        x = self.rggb_conv(x)
        x = self.stage1(x)
        f8 = self.stage2(x)
        f16 = self.stage3(f8)
        return f8, f16


class MultiCameraPatchEmbed(nn.Module):
    """输入 [B, N, 1, H, W] → 字典 {stride: [B, N, C, H/s, W/s]}。

    use_multiscale=False 时只输出 stride 16（与重构前一致）。
    """

    def __init__(self, config):
        super().__init__()
        self.use_multiscale = config.use_multiscale
        self.patch_embed = RAWPatchEmbed(embed_dim=config.embed_dim)
        self.norm16 = nn.LayerNorm(config.embed_dim)
        if self.use_multiscale:
            self.proj8 = nn.Conv2d(128, config.embed_dim, kernel_size=1)
            self.norm8 = nn.LayerNorm(config.embed_dim)

    @staticmethod
    def _channel_norm(norm, x):
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, images):
        B, N, C, H, W = images.shape
        f8, f16 = self.patch_embed(images.view(B * N, C, H, W))
        out = {16: self._channel_norm(self.norm16, f16)}
        if self.use_multiscale:
            out[8] = self._channel_norm(self.norm8, self.proj8(f8))
        return {s: f.view(B, N, *f.shape[1:]) for s, f in out.items()}

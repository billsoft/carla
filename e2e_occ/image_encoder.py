import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .position_encoding import RayDirectionEncoding
    from .layers import make_norm, FeedForward
except ImportError:
    from position_encoding import RayDirectionEncoding
    from layers import make_norm, FeedForward


class WindowAttention(nn.Module):
    """窗口自注意力（可选 Swin 式循环移位窗口）。

    特征图 pad 到 window_size 的倍数后切成互不重叠的 ws×ws 窗口，窗口内做多头自注意力。
    shift>0 时先循环移位 shift 个像素，让相邻窗口之间交换信息；被移位拼接到一起的
    不相邻区域、以及 pad 出来的位置，通过注意力掩码屏蔽。
    """

    def __init__(self, dim, num_heads, window_size=7, shift=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift = shift
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self._mask_cache = {}

    def _build_mask(self, H, W, Hp, Wp, device):
        key = (H, W, str(device))
        if key in self._mask_cache:
            return self._mask_cache[key]
        ws, s = self.window_size, self.shift
        ids = torch.zeros(Hp, Wp, device=device)
        valid = torch.zeros(Hp, Wp, dtype=torch.bool, device=device)
        valid[:H, :W] = True
        if s > 0:
            valid = torch.roll(valid, shifts=(-s, -s), dims=(0, 1))
            cnt = 0
            for hs in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                for wsl in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                    ids[hs, wsl] = cnt
                    cnt += 1
        ids = torch.where(valid, ids, torch.full_like(ids, -1.0))
        win = ids.view(Hp // ws, ws, Wp // ws, ws).permute(0, 2, 1, 3).reshape(-1, ws * ws)
        diff = win.unsqueeze(1) != win.unsqueeze(2)                       # [nW, L, L]
        pad_key = (win == -1).unsqueeze(1).expand_as(diff)
        masked = diff | pad_key
        # 用有限大负数而不是 -inf：pad 查询的整行都会被屏蔽，-inf 会产生 NaN（其输出随后被裁掉）
        mask = torch.zeros(masked.shape, device=device).masked_fill(masked, -1e4)
        self._mask_cache[key] = mask
        return mask

    def forward(self, x):
        B, H, W, C = x.shape
        ws, s = self.window_size, self.shift
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        if s > 0:
            x = torch.roll(x, shifts=(-s, -s), dims=(1, 2))
        nWh, nWw = Hp // ws, Wp // ws
        x = x.view(B, nWh, ws, nWw, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        hd = C // self.num_heads
        qkv = self.qkv(x).reshape(-1, ws * ws, 3, self.num_heads, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                 # [B*nW, h, L, hd]
        mask = None
        if s > 0 or pad_h > 0 or pad_w > 0:
            mask = self._build_mask(H, W, Hp, Wp, x.device)              # [nW, L, L]
            mask = mask.unsqueeze(1).repeat(B, 1, 1, 1).to(q.dtype)      # [B*nW, 1, L, L]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = self.proj(out.transpose(1, 2).reshape(-1, ws * ws, C))
        out = out.view(B, nWh, nWw, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        if s > 0:
            out = torch.roll(out, shifts=(s, s), dims=(1, 2))
        return out[:, :H, :W, :]


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, window_size=7, shift=0, dropout=0.1,
                 norm_type='layernorm', ffn_type='gelu'):
        super().__init__()
        self.norm1 = make_norm(dim, norm_type)
        self.attn = WindowAttention(dim, num_heads, window_size, shift)
        self.norm2 = make_norm(dim, norm_type)
        self.mlp = FeedForward(dim, mlp_ratio, dropout, ffn_type)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DepthHead(nn.Module):
    """深度分类头（BEVDepth 式辅助监督）：特征 → depth_bins 个对数均匀深度区间的 logits。"""

    def __init__(self, dim, bins):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, bins, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class ImageEncoder(nn.Module):
    """多尺度图像编码器。

    输入：PatchEmbed 输出的 {8: [B,N,C,120,160], 16: [B,N,C,60,80]}（单尺度时只有 16）
    输出：{'feats': {8:..., 16:..., 32:...}, 'depth_logits': [B,N,bins,120,160] 或 None}

    - 每个层级都加射线方向编码（共享 MLP）
    - stride 16 主层级过 encoder_layers 个窗口注意力块（奇数块用移位窗口）
    - stride 32 由编码后的 stride 16 下采样得到，带上下文
    - 深度头接在最细层级（stride 8）上
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        if config.use_ray_encoding:
            self.ray_embed = RayDirectionEncoding(dim)
        ws = config.window_size
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, config.num_heads, window_size=ws,
                         shift=(ws // 2 if (config.use_shifted_window and i % 2 == 1) else 0),
                         dropout=config.dropout, norm_type=config.norm_type, ffn_type=config.ffn_type)
            for i in range(config.encoder_layers)
        ])
        self.norm = make_norm(dim, config.norm_type)
        if config.use_multiscale:
            self.down32 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dim),
                nn.GELU(),
            )
        if config.use_depth_aux:
            self.depth_head = DepthHead(dim, config.depth_bins)

    def _add_rays(self, feat, stride, intrinsics, extrinsics):
        if not self.config.use_ray_encoding or intrinsics is None:
            return feat
        B, N, C, H, W = feat.shape
        return feat + self.ray_embed(intrinsics, extrinsics, H, W, stride)

    def forward(self, feats, intrinsics=None, extrinsics=None, need_depth=True):
        f16 = self._add_rays(feats[16], 16, intrinsics, extrinsics)
        B, N, C, H, W = f16.shape

        # 逐相机串行过窗口注意力（显存友好）
        x = f16.permute(0, 1, 3, 4, 2)                                   # [B, N, H, W, C]
        outs = []
        for i in range(N):
            xc = x[:, i]
            for block in self.blocks:
                xc = block(xc)
            outs.append(xc)
        f16 = self.norm(torch.stack(outs, dim=1)).permute(0, 1, 4, 2, 3)  # [B, N, C, H, W]

        out = {16: f16}
        depth_logits = None
        if self.config.use_multiscale:
            f8 = self._add_rays(feats[8], 8, intrinsics, extrinsics)
            f32 = self.down32(f16.flatten(0, 1))
            f32 = f32.view(B, N, *f32.shape[1:])
            f32 = self._add_rays(f32, 32, intrinsics, extrinsics)
            out[8] = f8
            out[32] = f32
        if self.config.use_depth_aux and need_depth:      # 深度头只是训练辅助监督，推理不算 (~14% 运算量)
            ds = self.config.depth_stride
            fd = out[ds]
            d = self.depth_head(fd.flatten(0, 1))
            depth_logits = d.view(B, N, *d.shape[1:])
        return {'feats': out, 'depth_logits': depth_logits}

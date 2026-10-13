"""
公共网络构件：归一化 / 前馈网络 / 3D RoPE 自注意力。

这些构件借鉴现代 LLM 的标配设计（LLaMA 系）：
- RMSNorm：去掉均值中心化的 LayerNorm，计算更省、训练更稳
- SwiGLU：门控前馈网络，同参数量下表达能力强于 GELU-MLP
- RoPE（旋转位置编码）：把位置信息以"旋转"方式注入 Q/K，注意力分数只依赖相对位置，
  这里扩展成 3D 版本（x/y/z 各占一段 head_dim），用于 3D 体素查询之间的自注意力

全部由 E2EOccConfig 的 norm_type / ffn_type / use_rope_3d 开关控制，
关掉时退化为原来的 LayerNorm / GELU-MLP / 无 RoPE，方便做消融对比。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(dim, norm_type='layernorm'):
    if norm_type == 'rmsnorm':
        return nn.RMSNorm(dim, eps=1e-6)
    return nn.LayerNorm(dim)


class FeedForward(nn.Module):
    """前馈网络：'gelu' = Linear→GELU→Linear；'swiglu' = (SiLU(xW1) ⊙ xW3)W2。

    SwiGLU 隐层取 2/3·ratio·dim（对齐到 8 的倍数），使参数量与同 ratio 的 GELU-MLP 基本一致。
    """

    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1, ffn_type='gelu'):
        super().__init__()
        self.ffn_type = ffn_type
        if ffn_type == 'swiglu':
            hidden = int(dim * mlp_ratio * 2 / 3)
            hidden = (hidden + 7) // 8 * 8
            self.w12 = nn.Linear(dim, hidden * 2)
            self.w2 = nn.Linear(hidden, dim)
        else:
            hidden = int(dim * mlp_ratio)
            self.w1 = nn.Linear(dim, hidden)
            self.w2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.ffn_type == 'swiglu':
            a, b = self.w12(x).chunk(2, dim=-1)
            h = F.silu(a) * b
        else:
            h = F.gelu(self.w1(x))
        return self.drop(self.w2(self.drop(h)))


class RoPE3D(nn.Module):
    """3D 旋转位置编码。

    head_dim 拆成 3 段（每段偶数维）分别编码 x/y/z 坐标；坐标单位为米，
    base 取 100（感知范围 ~80m，频率覆盖 0.1m~几十米的尺度）。
    """

    def __init__(self, head_dim, base=100.0):
        super().__init__()
        d_xy = (head_dim // 3) // 2 * 2
        dims = [d_xy, d_xy, head_dim - 2 * d_xy]
        assert all(d % 2 == 0 and d > 0 for d in dims), f'head_dim={head_dim} 无法均分给 3 轴'
        self.dims = dims
        for i, d in enumerate(dims):
            inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
            self.register_buffer(f'inv_freq_{i}', inv_freq, persistent=False)

    def forward(self, coords):
        """coords: [B, Q, 3]（米）→ (cos, sin)，形状 [B, 1, Q, head_dim/2]"""
        angles = []
        for i in range(3):
            inv_freq = getattr(self, f'inv_freq_{i}')
            angles.append(coords[..., i:i + 1].float() * inv_freq)   # [B, Q, d_i/2]
        ang = torch.cat(angles, dim=-1).unsqueeze(1)                   # [B, 1, Q, hd/2]
        return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: [B, H, Q, hd]；按相邻两维一对做旋转。"""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class MultiHeadSelfAttention(nn.Module):
    """基于 F.scaled_dot_product_attention（自动选 FlashAttention/MemEfficient kernel）的
    多头自注意力，可选 3D RoPE。"""

    def __init__(self, dim, num_heads, dropout=0.1, use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout
        self.rope = RoPE3D(self.head_dim) if use_rope else None

    def forward(self, x, coords=None, attn_mask=None):
        B, Q, C = x.shape
        qkv = self.qkv(x).view(B, Q, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                               # [B, H, Q, hd]
        if self.rope is not None and coords is not None:
            cos, sin = self.rope(coords)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, Q, C)
        return self.proj(out)

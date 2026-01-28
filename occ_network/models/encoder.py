import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .attention import FlashWindowAttention

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim=None, drop=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class WindowTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., drop=0., attn_drop=0., shift=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashWindowAttention(dim, num_heads, window_size, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def _window_partition(self, x, h, w):
        B, L, C = x.shape
        x = x.view(B, h, w, C)
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x = x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C), Hp, Wp

    def _window_reverse(self, windows, Hp, Wp, h, w, B):
        x = windows.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if Hp > h or Wp > w:
            x = x[:, :h, :w, :]
        return x.view(B, h * w, -1)

    def forward(self, x, h, w):
        """
        Args:
            x: [B, L, C] 输入特征 (已包含射线方向编码)
            h, w: 空间尺寸
        """
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        windows, Hp, Wp = self._window_partition(x, h, w)
        windows = self.attn(windows)
        x = self._window_reverse(windows, Hp, Wp, h, w, B)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.block1 = WindowTransformerBlock(dim, num_heads, window_size, mlp_ratio, drop, attn_drop, shift=False)
        self.block2 = WindowTransformerBlock(dim, num_heads, window_size, mlp_ratio, drop, attn_drop, shift=True)

    def forward(self, x, h, w):
        x = self.block1(x, h, w)
        return self.block2(x, h, w)

class SingleCameraEncoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers, window_size=8, mlp_ratio=4., drop=0., attn_drop=0., use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([TransformerEncoderLayer(dim, num_heads, window_size, mlp_ratio, drop, attn_drop) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        """
        Args:
            x: [B, L, C] 输入特征 (已包含射线方向编码)
            h, w: 空间尺寸
        """
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, h, w, use_reentrant=False)
            else:
                x = layer(x, h, w)
        return self.norm(x)

class MultiCameraEncoder(nn.Module):
    """
    多相机编码器 (简化版)

    位置编码架构 (重构后):
    1. RayDirectionEncoding: 射线方向编码 (通过 tokens += ray_enc 注入)
       - 支持多种投影模型: pinhole / equidistant / stereographic
       - 包含完整的 6-DoF 信息 (内参 + 外参)
    2. Swin 相对位置偏置: 在 FlashWindowAttention 内部处理

    已移除:
    - HyperbolicFOVEncoding: FOV 信息已在射线方向中编码
    - CameraRoPE: Yaw 信息已在射线方向中编码

    详见: occ_network/球面位置编码.md
    """
    def __init__(self, dim, num_heads, num_layers, window_size=8, mlp_ratio=4., drop=0., attn_drop=0., use_checkpoint=True):
        super().__init__()
        self.encoder = SingleCameraEncoder(dim, num_heads, num_layers, window_size, mlp_ratio, drop, attn_drop, use_checkpoint)

    def forward(self, camera_tokens, spatial_shape, position_encoder=None):
        """
        Args:
            camera_tokens: list of [B, L, C] 每个相机的 patch tokens
            spatial_shape: (H, W) patch 网格尺寸
            position_encoder: CameraPositionEncoding 实例
        """
        H, W = spatial_shape
        encoded_list = []
        for cam_idx, tokens in enumerate(camera_tokens):
            # 添加射线方向编码 (唯一的位置编码注入点)
            if position_encoder is not None and hasattr(position_encoder, 'get_ray_encoding'):
                ray_enc = position_encoder.get_ray_encoding(cam_idx, tokens.shape[0], tokens.device)
                if ray_enc is not None:
                    tokens = tokens + ray_enc
            # Transformer 编码 (不再传递 position_encoder)
            encoded = self.encoder(tokens, H, W)
            encoded_list.append(encoded)
        return encoded_list

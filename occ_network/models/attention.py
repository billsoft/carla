import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FlashWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, attn_drop=0., proj_drop=0., use_rope_fov=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope_fov = use_rope_fov
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords, coords], indexing='ij'))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None, position_encoder=None, camera_id=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.use_rope_fov and position_encoder is not None and camera_id is not None:
            q_flat = q.permute(0, 2, 1, 3).reshape(B_, N, C)
            k_flat = k.permute(0, 2, 1, 3).reshape(B_, N, C)
            q_enc, k_enc = position_encoder.encode_qk_single_camera(q_flat, k_flat, camera_id)
            q = q_enc.reshape(B_, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k_enc.reshape(B_, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        use_flash = hasattr(F, 'scaled_dot_product_attention') and mask is None
        if use_flash:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size ** 2, self.window_size ** 2, -1).permute(2, 0, 1)
            attn = attn + rpb.unsqueeze(0)
            if mask is not None:
                attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))

class DeformableAttention(nn.Module):
    def __init__(self, dim, num_heads, num_levels=1, num_points=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, value, spatial_shapes, level_start_index):
        B, Lq, _ = query.shape
        B, Lv, _ = value.shape
        value = self.value_proj(value).view(B, Lv, self.num_heads, self.dim // self.num_heads)
        sampling_offsets = self.sampling_offsets(query).view(B, Lq, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = F.softmax(self.attention_weights(query).view(B, Lq, self.num_heads, self.num_levels * self.num_points), -1).view(B, Lq, self.num_heads, self.num_levels, self.num_points)
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, None, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        output = self._sampling(value, spatial_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)

    def _sampling(self, value, spatial_shapes, sampling_locations, attention_weights):
        B, Lv, num_heads, head_dim = value.shape
        B, Lq, num_heads, num_levels, num_points, _ = sampling_locations.shape
        value_list = value.split([h * w for h, w in spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        output = torch.zeros(B, Lq, num_heads, head_dim, device=value.device, dtype=value.dtype)
        for level, (h, w) in enumerate(spatial_shapes):
            value_l = value_list[level].view(B, h, w, num_heads, head_dim).permute(0, 3, 4, 1, 2).contiguous().view(B * num_heads, head_dim, h, w)
            sampling_grid_l = sampling_grids[:, :, :, level].view(B * num_heads, Lq, num_points, 2)
            sampled_value = F.grid_sample(value_l, sampling_grid_l, mode='bilinear', padding_mode='zeros', align_corners=False)
            sampled_value = sampled_value.view(B, num_heads, head_dim, Lq, num_points).permute(0, 3, 1, 4, 2)
            output += (sampled_value * attention_weights[:, :, :, level, :, None]).sum(-2)
        return output.view(B, Lq, num_heads * head_dim)

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value, mask=None):
        B, Nq, C = query.shape
        Nkv = key_value.shape[1]
        q = self.q(query).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(key_value).reshape(B, Nkv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        if hasattr(F, 'scaled_dot_product_attention'):
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            if mask is not None:
                attn = attn + mask
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, Nq, C)))

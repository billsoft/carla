# e2e_occ 逐层 shape 报告

由 `verify_network.py` 自动生成，记录每个子模块第一次调用时的输入/输出张量形状与 dtype。

## 配置: 默认（全部优化开启）

参数量 13.45M，两帧(第二帧带时序记忆)前向，batch=1，bf16 autocast。"调用次数">1 表示该模块在相机/层级/帧循环中被多次调用，形状取第一次调用。

| 模块 | 类型 | 调用次数 | 输入 | 输出 |
|---|---|---|---|---|
| `patch_embed.patch_embed.rggb_conv` | Conv2d | 2 | (8x1x960x1280 float32) | 8x4x480x640 bfloat16 |
| `patch_embed.patch_embed.stage1.0` | Conv2d | 2 | (8x4x480x640 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1.1` | BatchNorm2d | 2 | (8x64x240x320 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1.2` | GELU | 2 | (8x64x240x320 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1` | Sequential | 2 | (8x4x480x640 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage2.0` | Conv2d | 2 | (8x64x240x320 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2.1` | BatchNorm2d | 2 | (8x128x120x160 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2.2` | GELU | 2 | (8x128x120x160 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2` | Sequential | 2 | (8x64x240x320 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage3.0` | Conv2d | 2 | (8x128x120x160 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.1` | BatchNorm2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.2` | GELU | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.3` | Conv2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.4` | BatchNorm2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3` | Sequential | 2 | (8x128x120x160 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed` | RAWPatchEmbed | 2 | (8x1x960x1280 float32) | (8x128x120x160 bfloat16, 8x256x60x80 bfloat16) |
| `patch_embed.norm16` | LayerNorm | 2 | (8x60x80x256 bfloat16) | 8x60x80x256 float32 |
| `patch_embed.proj8` | Conv2d | 2 | (8x128x120x160 bfloat16) | 8x256x120x160 bfloat16 |
| `patch_embed.norm8` | LayerNorm | 2 | (8x120x160x256 bfloat16) | 8x120x160x256 float32 |
| `patch_embed` | MultiCameraPatchEmbed | 2 | (1x8x1x960x1280 float32) | {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32} |
| `encoder.ray_embed.proj.0` | Linear | 6 | (1x8x60x80x63 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed.proj.1` | LayerNorm | 6 | (1x8x60x80x256 bfloat16) | 1x8x60x80x256 float32 |
| `encoder.ray_embed.proj.2` | ReLU | 6 | (1x8x60x80x256 float32) | 1x8x60x80x256 float32 |
| `encoder.ray_embed.proj.3` | Linear | 6 | (1x8x60x80x256 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed.proj` | Sequential | 6 | (1x8x60x80x63 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed` | RayDirectionEncoding | 6 | (1x8x3x3 float32, 1x8x4x4 float32) | 1x8x256x60x80 bfloat16 |
| `encoder.blocks.0.norm1` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.0.attn.qkv` | Linear | 16 | (108x49x256 float32) | 108x49x768 bfloat16 |
| `encoder.blocks.0.attn.proj` | Linear | 16 | (108x49x256 bfloat16) | 108x49x256 bfloat16 |
| `encoder.blocks.0.attn` | WindowAttention | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0.norm2` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.0.mlp.w12` | Linear | 16 | (1x60x80x256 float32) | 1x60x80x1376 bfloat16 |
| `encoder.blocks.0.mlp.drop` | Dropout | 32 | (1x60x80x688 bfloat16) | 1x60x80x688 bfloat16 |
| `encoder.blocks.0.mlp.w2` | Linear | 16 | (1x60x80x688 bfloat16) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0.mlp` | FeedForward | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0` | EncoderBlock | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.norm1` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.attn.qkv` | Linear | 16 | (108x49x256 float32) | 108x49x768 bfloat16 |
| `encoder.blocks.1.attn.proj` | Linear | 16 | (108x49x256 bfloat16) | 108x49x256 bfloat16 |
| `encoder.blocks.1.attn` | WindowAttention | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1.norm2` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.mlp.w12` | Linear | 16 | (1x60x80x256 float32) | 1x60x80x1376 bfloat16 |
| `encoder.blocks.1.mlp.drop` | Dropout | 32 | (1x60x80x688 bfloat16) | 1x60x80x688 bfloat16 |
| `encoder.blocks.1.mlp.w2` | Linear | 16 | (1x60x80x688 bfloat16) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1.mlp` | FeedForward | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1` | EncoderBlock | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.norm` | RMSNorm | 2 | (1x8x60x80x256 float32) | 1x8x60x80x256 float32 |
| `encoder.down32.0` | Conv2d | 2 | (8x256x60x80 float32) | 8x256x30x40 bfloat16 |
| `encoder.down32.1` | BatchNorm2d | 2 | (8x256x30x40 bfloat16) | 8x256x30x40 bfloat16 |
| `encoder.down32.2` | GELU | 2 | (8x256x30x40 bfloat16) | 8x256x30x40 bfloat16 |
| `encoder.down32` | Sequential | 2 | (8x256x60x80 float32) | 8x256x30x40 bfloat16 |
| `encoder.depth_head.net.0` | Conv2d | 2 | (8x256x120x160 float32) | 8x256x120x160 bfloat16 |
| `encoder.depth_head.net.1` | BatchNorm2d | 2 | (8x256x120x160 bfloat16) | 8x256x120x160 bfloat16 |
| `encoder.depth_head.net.2` | GELU | 2 | (8x256x120x160 bfloat16) | 8x256x120x160 bfloat16 |
| `encoder.depth_head.net.3` | Conv2d | 2 | (8x256x120x160 bfloat16) | 8x64x120x160 bfloat16 |
| `encoder.depth_head.net` | Sequential | 2 | (8x256x120x160 float32) | 8x64x120x160 bfloat16 |
| `encoder.depth_head` | DepthHead | 2 | (8x256x120x160 float32) | 8x64x120x160 bfloat16 |
| `encoder` | ImageEncoder | 2 | ({16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32}, 1x8x3x3 float32, 1x8x4x4 float32) | {feats: {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, depth_logits: 1x8x64x120x160 bfloat16} |
| `decoder.pos_3d` | SineCosinePositionEncoding3D | 4 |  | 5000x256 float32 |
| `decoder.coarse_layers.0.norm_self` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.self_attn.qkv` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.0.self_attn.rope` | RoPE3D | 2 | (1x5000x3 float32) | (1x1x5000x16 float32, 1x1x5000x16 float32) |
| `decoder.coarse_layers.0.self_attn.proj` | Linear | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.self_attn` | MultiHeadSelfAttention | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.norm_cross` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.cross_attn.sampling_offsets` | Linear | 2 | (1x5000x256 float32) | 1x5000x1536 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.attention_weights` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.output_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.dropout` | Dropout | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn` | DeformableCrossAttention | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.norm_mlp` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.mlp.w12` | Linear | 2 | (1x5000x256 float32) | 1x5000x1376 bfloat16 |
| `decoder.coarse_layers.0.mlp.drop` | Dropout | 4 | (1x5000x688 bfloat16) | 1x5000x688 bfloat16 |
| `decoder.coarse_layers.0.mlp.w2` | Linear | 2 | (1x5000x688 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.mlp` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0` | DeformableDecoderLayer | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.norm_self` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.self_attn.qkv` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.1.self_attn.rope` | RoPE3D | 2 | (1x5000x3 float32) | (1x1x5000x16 float32, 1x1x5000x16 float32) |
| `decoder.coarse_layers.1.self_attn.proj` | Linear | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.self_attn` | MultiHeadSelfAttention | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.norm_cross` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.cross_attn.sampling_offsets` | Linear | 2 | (1x5000x256 float32) | 1x5000x1536 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.attention_weights` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.output_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.dropout` | Dropout | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn` | DeformableCrossAttention | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.norm_mlp` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.mlp.w12` | Linear | 2 | (1x5000x256 float32) | 1x5000x1376 bfloat16 |
| `decoder.coarse_layers.1.mlp.drop` | Dropout | 4 | (1x5000x688 bfloat16) | 1x5000x688 bfloat16 |
| `decoder.coarse_layers.1.mlp.w2` | Linear | 2 | (1x5000x688 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.mlp` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1` | DeformableDecoderLayer | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.norm_q` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.norm_kv` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.temporal_attn.value_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.sampling_offsets` | Linear | 2 | (1x5000x256 float32) | 1x5000x192 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.attention_weights` | Linear | 2 | (1x5000x256 float32) | 1x5000x64 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.output_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.temporal_attn` | DeformableTemporalAttention | 2 | (1x5000x256 float32, 1x5000x256 float32, 1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.norm2` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.ffn.w12` | Linear | 2 | (1x5000x256 float32) | 1x5000x1376 bfloat16 |
| `decoder.temporal_fusion.ffn.drop` | Dropout | 4 | (1x5000x688 bfloat16) | 1x5000x688 bfloat16 |
| `decoder.temporal_fusion.ffn.w2` | Linear | 2 | (1x5000x688 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.ffn` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate.1` | Sigmoid | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate.1` | Sigmoid | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate.1` | Tanh | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate` | GRUGate | 2 | (1x5000x256 float32, 1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.mem_out` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion` | TemporalFusionModule | 2 | (1x5000x256 float32) | (1x5000x256 float32, 1x5000x256 float32) |
| `decoder.coarse_occ_head.0` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_occ_head.1` | Linear | 2 | (1x5000x256 float32) | 1x5000x128 bfloat16 |
| `decoder.coarse_occ_head.2` | GELU | 2 | (1x5000x128 bfloat16) | 1x5000x128 bfloat16 |
| `decoder.coarse_occ_head.3` | Linear | 2 | (1x5000x128 bfloat16) | 1x5000x1 bfloat16 |
| `decoder.coarse_occ_head` | Sequential | 2 | (1x5000x256 float32) | 1x5000x1 bfloat16 |
| `decoder.coarse_to_fine.0` | RMSNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.coarse_to_fine.1.w12` | Linear | 2 | (1x102400x256 float32) | 1x102400x688 bfloat16 |
| `decoder.coarse_to_fine.1.drop` | Dropout | 4 | (1x102400x344 bfloat16) | 1x102400x344 bfloat16 |
| `decoder.coarse_to_fine.1.w2` | Linear | 2 | (1x102400x344 bfloat16) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine.1` | FeedForward | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine` | Sequential | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine_norm` | RMSNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.0.norm_cross` | RMSNorm | 2 | (1x25600x256 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.0.cross_attn.sampling_offsets` | Linear | 2 | (1x25600x256 float32) | 1x25600x1536 bfloat16 |
| `decoder.fine_layers.0.cross_attn.attention_weights` | Linear | 2 | (1x25600x256 float32) | 1x25600x768 bfloat16 |
| `decoder.fine_layers.0.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn.output_proj` | Linear | 2 | (1x25600x256 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn.dropout` | Dropout | 2 | (1x25600x256 bfloat16) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn` | DeformableCrossAttention | 2 | (1x25600x256 float32, 1x25600x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0.norm_mlp` | RMSNorm | 2 | (1x25600x256 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.0.mlp.w12` | Linear | 2 | (1x25600x256 float32) | 1x25600x1376 bfloat16 |
| `decoder.fine_layers.0.mlp.drop` | Dropout | 4 | (1x25600x688 bfloat16) | 1x25600x688 bfloat16 |
| `decoder.fine_layers.0.mlp.w2` | Linear | 2 | (1x25600x688 bfloat16) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0.mlp` | FeedForward | 2 | (1x25600x256 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0` | DeformableDecoderLayer | 2 | (1x25600x256 float32, 1x25600x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.1.norm_cross` | RMSNorm | 2 | (1x25600x256 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.1.cross_attn.sampling_offsets` | Linear | 2 | (1x25600x256 float32) | 1x25600x1536 bfloat16 |
| `decoder.fine_layers.1.cross_attn.attention_weights` | Linear | 2 | (1x25600x256 float32) | 1x25600x768 bfloat16 |
| `decoder.fine_layers.1.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn.output_proj` | Linear | 2 | (1x25600x256 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn.dropout` | Dropout | 2 | (1x25600x256 bfloat16) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn` | DeformableCrossAttention | 2 | (1x25600x256 float32, 1x25600x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1.norm_mlp` | RMSNorm | 2 | (1x25600x256 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.1.mlp.w12` | Linear | 2 | (1x25600x256 float32) | 1x25600x1376 bfloat16 |
| `decoder.fine_layers.1.mlp.drop` | Dropout | 4 | (1x25600x688 bfloat16) | 1x25600x688 bfloat16 |
| `decoder.fine_layers.1.mlp.w2` | Linear | 2 | (1x25600x688 bfloat16) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1.mlp` | FeedForward | 2 | (1x25600x256 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1` | DeformableDecoderLayer | 2 | (1x25600x256 float32, 1x25600x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x25600x256 float32 |
| `decoder.fine_spatial_conv.0` | Conv3d | 2 | (1x256x80x80x16 float32) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv.1` | BatchNorm3d | 2 | (1x256x80x80x16 bfloat16) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv.2` | GELU | 2 | (1x256x80x80x16 bfloat16) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv` | Sequential | 2 | (1x256x80x80x16 float32) | 1x256x80x80x16 bfloat16 |
| `decoder` | OccupancyDecoder | 2 | ({16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | {fine_feats: 1x80x80x16x256 float32, memory: 1x5000x256 float32, coarse_occ_logits: 1x25x25x8 bfloat16, fine_selected: 1x25600 int64} |
| `head.conv1.0` | Conv3d | 2 | (1x256x80x80x16 float32) | 1x128x80x80x16 bfloat16 |
| `head.conv1.1` | BatchNorm3d | 2 | (1x128x80x80x16 bfloat16) | 1x128x80x80x16 bfloat16 |
| `head.conv1.2` | GELU | 2 | (1x128x80x80x16 bfloat16) | 1x128x80x80x16 bfloat16 |
| `head.conv1` | Sequential | 2 | (1x256x80x80x16 float32) | 1x128x80x80x16 bfloat16 |
| `head.conv2.0` | Conv3d | 2 | (1x128x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.conv2.1` | BatchNorm3d | 2 | (1x64x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.conv2.2` | GELU | 2 | (1x64x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.conv2` | Sequential | 2 | (1x128x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.refine1.0` | Conv3d | 2 | (1x64x200x200x32 float32) | 1x32x200x200x32 bfloat16 |
| `head.refine1.1` | BatchNorm3d | 2 | (1x32x200x200x32 bfloat16) | 1x32x200x200x32 bfloat16 |
| `head.refine1.2` | GELU | 2 | (1x32x200x200x32 bfloat16) | 1x32x200x200x32 bfloat16 |
| `head.refine1` | Sequential | 2 | (1x64x200x200x32 float32) | 1x32x200x200x32 bfloat16 |
| `head.skip1` | Conv3d | 2 | (1x64x200x200x32 float32) | 1x32x200x200x32 bfloat16 |
| `head.flow.0` | Conv3d | 2 | (1x32x200x200x32 bfloat16) | 1x16x200x200x32 bfloat16 |
| `head.flow.1` | GELU | 2 | (1x16x200x200x32 bfloat16) | 1x16x200x200x32 bfloat16 |
| `head.flow.2` | Conv3d | 2 | (1x16x200x200x32 bfloat16) | 1x3x200x200x32 bfloat16 |
| `head.flow` | Sequential | 2 | (1x32x200x200x32 bfloat16) | 1x3x200x200x32 bfloat16 |
| `head.refine2.0` | Conv3d | 2 | (1x32x400x400x32 float32) | 1x18x400x400x32 bfloat16 |
| `head.refine2.1` | BatchNorm3d | 2 | (1x18x400x400x32 bfloat16) | 1x18x400x400x32 bfloat16 |
| `head.refine2` | Sequential | 2 | (1x32x400x400x32 float32) | 1x18x400x400x32 bfloat16 |
| `head.skip2` | Conv3d | 2 | (1x32x400x400x32 float32) | 1x18x400x400x32 bfloat16 |
| `head` | VoxelHead | 2 | (1x80x80x16x256 float32) | {semantic: 1x18x400x400x32 bfloat16, flow: 1x3x400x400x32 float32} |
| `decoder.temporal_fusion.time_embed.mlp.0` | Linear | 1 | (1x16 float32) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed.mlp.1` | SiLU | 1 | (1x256 bfloat16) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed.mlp.2` | Linear | 1 | (1x256 bfloat16) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed.mlp` | Sequential | 1 | (1x16 float32) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed` | TimeEmbedding | 1 | (1 float32) | 1x1x256 bfloat16 |

## 配置: 稠密细化（use_sparse_fine=False）

参数量 13.45M，两帧(第二帧带时序记忆)前向，batch=1，bf16 autocast。"调用次数">1 表示该模块在相机/层级/帧循环中被多次调用，形状取第一次调用。

| 模块 | 类型 | 调用次数 | 输入 | 输出 |
|---|---|---|---|---|
| `patch_embed.patch_embed.rggb_conv` | Conv2d | 2 | (8x1x960x1280 float32) | 8x4x480x640 bfloat16 |
| `patch_embed.patch_embed.stage1.0` | Conv2d | 2 | (8x4x480x640 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1.1` | BatchNorm2d | 2 | (8x64x240x320 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1.2` | GELU | 2 | (8x64x240x320 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1` | Sequential | 2 | (8x4x480x640 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage2.0` | Conv2d | 2 | (8x64x240x320 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2.1` | BatchNorm2d | 2 | (8x128x120x160 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2.2` | GELU | 2 | (8x128x120x160 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2` | Sequential | 2 | (8x64x240x320 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage3.0` | Conv2d | 2 | (8x128x120x160 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.1` | BatchNorm2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.2` | GELU | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.3` | Conv2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.4` | BatchNorm2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3` | Sequential | 2 | (8x128x120x160 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed` | RAWPatchEmbed | 2 | (8x1x960x1280 float32) | (8x128x120x160 bfloat16, 8x256x60x80 bfloat16) |
| `patch_embed.norm16` | LayerNorm | 2 | (8x60x80x256 bfloat16) | 8x60x80x256 float32 |
| `patch_embed.proj8` | Conv2d | 2 | (8x128x120x160 bfloat16) | 8x256x120x160 bfloat16 |
| `patch_embed.norm8` | LayerNorm | 2 | (8x120x160x256 bfloat16) | 8x120x160x256 float32 |
| `patch_embed` | MultiCameraPatchEmbed | 2 | (1x8x1x960x1280 float32) | {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32} |
| `encoder.ray_embed.proj.0` | Linear | 6 | (1x8x60x80x63 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed.proj.1` | LayerNorm | 6 | (1x8x60x80x256 bfloat16) | 1x8x60x80x256 float32 |
| `encoder.ray_embed.proj.2` | ReLU | 6 | (1x8x60x80x256 float32) | 1x8x60x80x256 float32 |
| `encoder.ray_embed.proj.3` | Linear | 6 | (1x8x60x80x256 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed.proj` | Sequential | 6 | (1x8x60x80x63 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed` | RayDirectionEncoding | 6 | (1x8x3x3 float32, 1x8x4x4 float32) | 1x8x256x60x80 bfloat16 |
| `encoder.blocks.0.norm1` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.0.attn.qkv` | Linear | 16 | (108x49x256 float32) | 108x49x768 bfloat16 |
| `encoder.blocks.0.attn.proj` | Linear | 16 | (108x49x256 bfloat16) | 108x49x256 bfloat16 |
| `encoder.blocks.0.attn` | WindowAttention | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0.norm2` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.0.mlp.w12` | Linear | 16 | (1x60x80x256 float32) | 1x60x80x1376 bfloat16 |
| `encoder.blocks.0.mlp.drop` | Dropout | 32 | (1x60x80x688 bfloat16) | 1x60x80x688 bfloat16 |
| `encoder.blocks.0.mlp.w2` | Linear | 16 | (1x60x80x688 bfloat16) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0.mlp` | FeedForward | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0` | EncoderBlock | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.norm1` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.attn.qkv` | Linear | 16 | (108x49x256 float32) | 108x49x768 bfloat16 |
| `encoder.blocks.1.attn.proj` | Linear | 16 | (108x49x256 bfloat16) | 108x49x256 bfloat16 |
| `encoder.blocks.1.attn` | WindowAttention | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1.norm2` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.mlp.w12` | Linear | 16 | (1x60x80x256 float32) | 1x60x80x1376 bfloat16 |
| `encoder.blocks.1.mlp.drop` | Dropout | 32 | (1x60x80x688 bfloat16) | 1x60x80x688 bfloat16 |
| `encoder.blocks.1.mlp.w2` | Linear | 16 | (1x60x80x688 bfloat16) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1.mlp` | FeedForward | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1` | EncoderBlock | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.norm` | RMSNorm | 2 | (1x8x60x80x256 float32) | 1x8x60x80x256 float32 |
| `encoder.down32.0` | Conv2d | 2 | (8x256x60x80 float32) | 8x256x30x40 bfloat16 |
| `encoder.down32.1` | BatchNorm2d | 2 | (8x256x30x40 bfloat16) | 8x256x30x40 bfloat16 |
| `encoder.down32.2` | GELU | 2 | (8x256x30x40 bfloat16) | 8x256x30x40 bfloat16 |
| `encoder.down32` | Sequential | 2 | (8x256x60x80 float32) | 8x256x30x40 bfloat16 |
| `encoder.depth_head.net.0` | Conv2d | 2 | (8x256x120x160 float32) | 8x256x120x160 bfloat16 |
| `encoder.depth_head.net.1` | BatchNorm2d | 2 | (8x256x120x160 bfloat16) | 8x256x120x160 bfloat16 |
| `encoder.depth_head.net.2` | GELU | 2 | (8x256x120x160 bfloat16) | 8x256x120x160 bfloat16 |
| `encoder.depth_head.net.3` | Conv2d | 2 | (8x256x120x160 bfloat16) | 8x64x120x160 bfloat16 |
| `encoder.depth_head.net` | Sequential | 2 | (8x256x120x160 float32) | 8x64x120x160 bfloat16 |
| `encoder.depth_head` | DepthHead | 2 | (8x256x120x160 float32) | 8x64x120x160 bfloat16 |
| `encoder` | ImageEncoder | 2 | ({16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32}, 1x8x3x3 float32, 1x8x4x4 float32) | {feats: {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, depth_logits: 1x8x64x120x160 bfloat16} |
| `decoder.pos_3d` | SineCosinePositionEncoding3D | 4 |  | 5000x256 float32 |
| `decoder.coarse_layers.0.norm_self` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.self_attn.qkv` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.0.self_attn.rope` | RoPE3D | 2 | (1x5000x3 float32) | (1x1x5000x16 float32, 1x1x5000x16 float32) |
| `decoder.coarse_layers.0.self_attn.proj` | Linear | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.self_attn` | MultiHeadSelfAttention | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.norm_cross` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.cross_attn.sampling_offsets` | Linear | 2 | (1x5000x256 float32) | 1x5000x1536 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.attention_weights` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.output_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.dropout` | Dropout | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn` | DeformableCrossAttention | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.norm_mlp` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.mlp.w12` | Linear | 2 | (1x5000x256 float32) | 1x5000x1376 bfloat16 |
| `decoder.coarse_layers.0.mlp.drop` | Dropout | 4 | (1x5000x688 bfloat16) | 1x5000x688 bfloat16 |
| `decoder.coarse_layers.0.mlp.w2` | Linear | 2 | (1x5000x688 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.mlp` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0` | DeformableDecoderLayer | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.norm_self` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.self_attn.qkv` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.1.self_attn.rope` | RoPE3D | 2 | (1x5000x3 float32) | (1x1x5000x16 float32, 1x1x5000x16 float32) |
| `decoder.coarse_layers.1.self_attn.proj` | Linear | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.self_attn` | MultiHeadSelfAttention | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.norm_cross` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.cross_attn.sampling_offsets` | Linear | 2 | (1x5000x256 float32) | 1x5000x1536 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.attention_weights` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.output_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.dropout` | Dropout | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn` | DeformableCrossAttention | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.norm_mlp` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.mlp.w12` | Linear | 2 | (1x5000x256 float32) | 1x5000x1376 bfloat16 |
| `decoder.coarse_layers.1.mlp.drop` | Dropout | 4 | (1x5000x688 bfloat16) | 1x5000x688 bfloat16 |
| `decoder.coarse_layers.1.mlp.w2` | Linear | 2 | (1x5000x688 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.mlp` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1` | DeformableDecoderLayer | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.norm_q` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.norm_kv` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.temporal_attn.value_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.sampling_offsets` | Linear | 2 | (1x5000x256 float32) | 1x5000x192 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.attention_weights` | Linear | 2 | (1x5000x256 float32) | 1x5000x64 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.output_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.temporal_attn` | DeformableTemporalAttention | 2 | (1x5000x256 float32, 1x5000x256 float32, 1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.norm2` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.ffn.w12` | Linear | 2 | (1x5000x256 float32) | 1x5000x1376 bfloat16 |
| `decoder.temporal_fusion.ffn.drop` | Dropout | 4 | (1x5000x688 bfloat16) | 1x5000x688 bfloat16 |
| `decoder.temporal_fusion.ffn.w2` | Linear | 2 | (1x5000x688 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.ffn` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate.1` | Sigmoid | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate.1` | Sigmoid | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate.1` | Tanh | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate` | GRUGate | 2 | (1x5000x256 float32, 1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.mem_out` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion` | TemporalFusionModule | 2 | (1x5000x256 float32) | (1x5000x256 float32, 1x5000x256 float32) |
| `decoder.coarse_occ_head.0` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_occ_head.1` | Linear | 2 | (1x5000x256 float32) | 1x5000x128 bfloat16 |
| `decoder.coarse_occ_head.2` | GELU | 2 | (1x5000x128 bfloat16) | 1x5000x128 bfloat16 |
| `decoder.coarse_occ_head.3` | Linear | 2 | (1x5000x128 bfloat16) | 1x5000x1 bfloat16 |
| `decoder.coarse_occ_head` | Sequential | 2 | (1x5000x256 float32) | 1x5000x1 bfloat16 |
| `decoder.coarse_to_fine.0` | RMSNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.coarse_to_fine.1.w12` | Linear | 2 | (1x102400x256 float32) | 1x102400x688 bfloat16 |
| `decoder.coarse_to_fine.1.drop` | Dropout | 4 | (1x102400x344 bfloat16) | 1x102400x344 bfloat16 |
| `decoder.coarse_to_fine.1.w2` | Linear | 2 | (1x102400x344 bfloat16) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine.1` | FeedForward | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine` | Sequential | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine_norm` | RMSNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.0.norm_cross` | RMSNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.0.cross_attn.sampling_offsets` | Linear | 2 | (1x102400x256 float32) | 1x102400x1536 bfloat16 |
| `decoder.fine_layers.0.cross_attn.attention_weights` | Linear | 2 | (1x102400x256 float32) | 1x102400x768 bfloat16 |
| `decoder.fine_layers.0.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn.output_proj` | Linear | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn.dropout` | Dropout | 2 | (1x102400x256 bfloat16) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn` | DeformableCrossAttention | 2 | (1x102400x256 float32, 1x102400x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.0.norm_mlp` | RMSNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.0.mlp.w12` | Linear | 2 | (1x102400x256 float32) | 1x102400x1376 bfloat16 |
| `decoder.fine_layers.0.mlp.drop` | Dropout | 4 | (1x102400x688 bfloat16) | 1x102400x688 bfloat16 |
| `decoder.fine_layers.0.mlp.w2` | Linear | 2 | (1x102400x688 bfloat16) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.0.mlp` | FeedForward | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.0` | DeformableDecoderLayer | 2 | (1x102400x256 float32, 1x102400x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.1.norm_cross` | RMSNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.1.cross_attn.sampling_offsets` | Linear | 2 | (1x102400x256 float32) | 1x102400x1536 bfloat16 |
| `decoder.fine_layers.1.cross_attn.attention_weights` | Linear | 2 | (1x102400x256 float32) | 1x102400x768 bfloat16 |
| `decoder.fine_layers.1.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn.output_proj` | Linear | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn.dropout` | Dropout | 2 | (1x102400x256 bfloat16) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn` | DeformableCrossAttention | 2 | (1x102400x256 float32, 1x102400x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.1.norm_mlp` | RMSNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.1.mlp.w12` | Linear | 2 | (1x102400x256 float32) | 1x102400x1376 bfloat16 |
| `decoder.fine_layers.1.mlp.drop` | Dropout | 4 | (1x102400x688 bfloat16) | 1x102400x688 bfloat16 |
| `decoder.fine_layers.1.mlp.w2` | Linear | 2 | (1x102400x688 bfloat16) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.1.mlp` | FeedForward | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.1` | DeformableDecoderLayer | 2 | (1x102400x256 float32, 1x102400x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x102400x256 float32 |
| `decoder.fine_spatial_conv.0` | Conv3d | 2 | (1x256x80x80x16 float32) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv.1` | BatchNorm3d | 2 | (1x256x80x80x16 bfloat16) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv.2` | GELU | 2 | (1x256x80x80x16 bfloat16) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv` | Sequential | 2 | (1x256x80x80x16 float32) | 1x256x80x80x16 bfloat16 |
| `decoder` | OccupancyDecoder | 2 | ({16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | {fine_feats: 1x80x80x16x256 float32, memory: 1x5000x256 float32, coarse_occ_logits: 1x25x25x8 bfloat16} |
| `head.conv1.0` | Conv3d | 2 | (1x256x80x80x16 float32) | 1x128x80x80x16 bfloat16 |
| `head.conv1.1` | BatchNorm3d | 2 | (1x128x80x80x16 bfloat16) | 1x128x80x80x16 bfloat16 |
| `head.conv1.2` | GELU | 2 | (1x128x80x80x16 bfloat16) | 1x128x80x80x16 bfloat16 |
| `head.conv1` | Sequential | 2 | (1x256x80x80x16 float32) | 1x128x80x80x16 bfloat16 |
| `head.conv2.0` | Conv3d | 2 | (1x128x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.conv2.1` | BatchNorm3d | 2 | (1x64x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.conv2.2` | GELU | 2 | (1x64x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.conv2` | Sequential | 2 | (1x128x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.refine1.0` | Conv3d | 2 | (1x64x200x200x32 float32) | 1x32x200x200x32 bfloat16 |
| `head.refine1.1` | BatchNorm3d | 2 | (1x32x200x200x32 bfloat16) | 1x32x200x200x32 bfloat16 |
| `head.refine1.2` | GELU | 2 | (1x32x200x200x32 bfloat16) | 1x32x200x200x32 bfloat16 |
| `head.refine1` | Sequential | 2 | (1x64x200x200x32 float32) | 1x32x200x200x32 bfloat16 |
| `head.skip1` | Conv3d | 2 | (1x64x200x200x32 float32) | 1x32x200x200x32 bfloat16 |
| `head.flow.0` | Conv3d | 2 | (1x32x200x200x32 bfloat16) | 1x16x200x200x32 bfloat16 |
| `head.flow.1` | GELU | 2 | (1x16x200x200x32 bfloat16) | 1x16x200x200x32 bfloat16 |
| `head.flow.2` | Conv3d | 2 | (1x16x200x200x32 bfloat16) | 1x3x200x200x32 bfloat16 |
| `head.flow` | Sequential | 2 | (1x32x200x200x32 bfloat16) | 1x3x200x200x32 bfloat16 |
| `head.refine2.0` | Conv3d | 2 | (1x32x400x400x32 float32) | 1x18x400x400x32 bfloat16 |
| `head.refine2.1` | BatchNorm3d | 2 | (1x18x400x400x32 bfloat16) | 1x18x400x400x32 bfloat16 |
| `head.refine2` | Sequential | 2 | (1x32x400x400x32 float32) | 1x18x400x400x32 bfloat16 |
| `head.skip2` | Conv3d | 2 | (1x32x400x400x32 float32) | 1x18x400x400x32 bfloat16 |
| `head` | VoxelHead | 2 | (1x80x80x16x256 float32) | {semantic: 1x18x400x400x32 bfloat16, flow: 1x3x400x400x32 float32} |
| `decoder.temporal_fusion.time_embed.mlp.0` | Linear | 1 | (1x16 float32) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed.mlp.1` | SiLU | 1 | (1x256 bfloat16) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed.mlp.2` | Linear | 1 | (1x256 bfloat16) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed.mlp` | Sequential | 1 | (1x16 float32) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed` | TimeEmbedding | 1 | (1 float32) | 1x1x256 bfloat16 |

## 配置: 隐式输出头（head_type=implicit）

参数量 13.39M，两帧(第二帧带时序记忆)前向，batch=1，bf16 autocast。"调用次数">1 表示该模块在相机/层级/帧循环中被多次调用，形状取第一次调用。

| 模块 | 类型 | 调用次数 | 输入 | 输出 |
|---|---|---|---|---|
| `patch_embed.patch_embed.rggb_conv` | Conv2d | 2 | (8x1x960x1280 float32) | 8x4x480x640 bfloat16 |
| `patch_embed.patch_embed.stage1.0` | Conv2d | 2 | (8x4x480x640 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1.1` | BatchNorm2d | 2 | (8x64x240x320 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1.2` | GELU | 2 | (8x64x240x320 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1` | Sequential | 2 | (8x4x480x640 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage2.0` | Conv2d | 2 | (8x64x240x320 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2.1` | BatchNorm2d | 2 | (8x128x120x160 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2.2` | GELU | 2 | (8x128x120x160 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2` | Sequential | 2 | (8x64x240x320 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage3.0` | Conv2d | 2 | (8x128x120x160 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.1` | BatchNorm2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.2` | GELU | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.3` | Conv2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.4` | BatchNorm2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3` | Sequential | 2 | (8x128x120x160 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed` | RAWPatchEmbed | 2 | (8x1x960x1280 float32) | (8x128x120x160 bfloat16, 8x256x60x80 bfloat16) |
| `patch_embed.norm16` | LayerNorm | 2 | (8x60x80x256 bfloat16) | 8x60x80x256 float32 |
| `patch_embed.proj8` | Conv2d | 2 | (8x128x120x160 bfloat16) | 8x256x120x160 bfloat16 |
| `patch_embed.norm8` | LayerNorm | 2 | (8x120x160x256 bfloat16) | 8x120x160x256 float32 |
| `patch_embed` | MultiCameraPatchEmbed | 2 | (1x8x1x960x1280 float32) | {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32} |
| `encoder.ray_embed.proj.0` | Linear | 6 | (1x8x60x80x63 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed.proj.1` | LayerNorm | 6 | (1x8x60x80x256 bfloat16) | 1x8x60x80x256 float32 |
| `encoder.ray_embed.proj.2` | ReLU | 6 | (1x8x60x80x256 float32) | 1x8x60x80x256 float32 |
| `encoder.ray_embed.proj.3` | Linear | 6 | (1x8x60x80x256 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed.proj` | Sequential | 6 | (1x8x60x80x63 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed` | RayDirectionEncoding | 6 | (1x8x3x3 float32, 1x8x4x4 float32) | 1x8x256x60x80 bfloat16 |
| `encoder.blocks.0.norm1` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.0.attn.qkv` | Linear | 16 | (108x49x256 float32) | 108x49x768 bfloat16 |
| `encoder.blocks.0.attn.proj` | Linear | 16 | (108x49x256 bfloat16) | 108x49x256 bfloat16 |
| `encoder.blocks.0.attn` | WindowAttention | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0.norm2` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.0.mlp.w12` | Linear | 16 | (1x60x80x256 float32) | 1x60x80x1376 bfloat16 |
| `encoder.blocks.0.mlp.drop` | Dropout | 32 | (1x60x80x688 bfloat16) | 1x60x80x688 bfloat16 |
| `encoder.blocks.0.mlp.w2` | Linear | 16 | (1x60x80x688 bfloat16) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0.mlp` | FeedForward | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0` | EncoderBlock | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.norm1` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.attn.qkv` | Linear | 16 | (108x49x256 float32) | 108x49x768 bfloat16 |
| `encoder.blocks.1.attn.proj` | Linear | 16 | (108x49x256 bfloat16) | 108x49x256 bfloat16 |
| `encoder.blocks.1.attn` | WindowAttention | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1.norm2` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.mlp.w12` | Linear | 16 | (1x60x80x256 float32) | 1x60x80x1376 bfloat16 |
| `encoder.blocks.1.mlp.drop` | Dropout | 32 | (1x60x80x688 bfloat16) | 1x60x80x688 bfloat16 |
| `encoder.blocks.1.mlp.w2` | Linear | 16 | (1x60x80x688 bfloat16) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1.mlp` | FeedForward | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1` | EncoderBlock | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.norm` | RMSNorm | 2 | (1x8x60x80x256 float32) | 1x8x60x80x256 float32 |
| `encoder.down32.0` | Conv2d | 2 | (8x256x60x80 float32) | 8x256x30x40 bfloat16 |
| `encoder.down32.1` | BatchNorm2d | 2 | (8x256x30x40 bfloat16) | 8x256x30x40 bfloat16 |
| `encoder.down32.2` | GELU | 2 | (8x256x30x40 bfloat16) | 8x256x30x40 bfloat16 |
| `encoder.down32` | Sequential | 2 | (8x256x60x80 float32) | 8x256x30x40 bfloat16 |
| `encoder.depth_head.net.0` | Conv2d | 2 | (8x256x120x160 float32) | 8x256x120x160 bfloat16 |
| `encoder.depth_head.net.1` | BatchNorm2d | 2 | (8x256x120x160 bfloat16) | 8x256x120x160 bfloat16 |
| `encoder.depth_head.net.2` | GELU | 2 | (8x256x120x160 bfloat16) | 8x256x120x160 bfloat16 |
| `encoder.depth_head.net.3` | Conv2d | 2 | (8x256x120x160 bfloat16) | 8x64x120x160 bfloat16 |
| `encoder.depth_head.net` | Sequential | 2 | (8x256x120x160 float32) | 8x64x120x160 bfloat16 |
| `encoder.depth_head` | DepthHead | 2 | (8x256x120x160 float32) | 8x64x120x160 bfloat16 |
| `encoder` | ImageEncoder | 2 | ({16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32}, 1x8x3x3 float32, 1x8x4x4 float32) | {feats: {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, depth_logits: 1x8x64x120x160 bfloat16} |
| `decoder.pos_3d` | SineCosinePositionEncoding3D | 4 |  | 5000x256 float32 |
| `decoder.coarse_layers.0.norm_self` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.self_attn.qkv` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.0.self_attn.rope` | RoPE3D | 2 | (1x5000x3 float32) | (1x1x5000x16 float32, 1x1x5000x16 float32) |
| `decoder.coarse_layers.0.self_attn.proj` | Linear | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.self_attn` | MultiHeadSelfAttention | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.norm_cross` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.cross_attn.sampling_offsets` | Linear | 2 | (1x5000x256 float32) | 1x5000x1536 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.attention_weights` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.output_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.dropout` | Dropout | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn` | DeformableCrossAttention | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.norm_mlp` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.mlp.w12` | Linear | 2 | (1x5000x256 float32) | 1x5000x1376 bfloat16 |
| `decoder.coarse_layers.0.mlp.drop` | Dropout | 4 | (1x5000x688 bfloat16) | 1x5000x688 bfloat16 |
| `decoder.coarse_layers.0.mlp.w2` | Linear | 2 | (1x5000x688 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.mlp` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0` | DeformableDecoderLayer | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.norm_self` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.self_attn.qkv` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.1.self_attn.rope` | RoPE3D | 2 | (1x5000x3 float32) | (1x1x5000x16 float32, 1x1x5000x16 float32) |
| `decoder.coarse_layers.1.self_attn.proj` | Linear | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.self_attn` | MultiHeadSelfAttention | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.norm_cross` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.cross_attn.sampling_offsets` | Linear | 2 | (1x5000x256 float32) | 1x5000x1536 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.attention_weights` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.output_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.dropout` | Dropout | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn` | DeformableCrossAttention | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.norm_mlp` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.mlp.w12` | Linear | 2 | (1x5000x256 float32) | 1x5000x1376 bfloat16 |
| `decoder.coarse_layers.1.mlp.drop` | Dropout | 4 | (1x5000x688 bfloat16) | 1x5000x688 bfloat16 |
| `decoder.coarse_layers.1.mlp.w2` | Linear | 2 | (1x5000x688 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.mlp` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1` | DeformableDecoderLayer | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.norm_q` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.norm_kv` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.temporal_attn.value_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.sampling_offsets` | Linear | 2 | (1x5000x256 float32) | 1x5000x192 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.attention_weights` | Linear | 2 | (1x5000x256 float32) | 1x5000x64 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.output_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.temporal_attn` | DeformableTemporalAttention | 2 | (1x5000x256 float32, 1x5000x256 float32, 1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.norm2` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.ffn.w12` | Linear | 2 | (1x5000x256 float32) | 1x5000x1376 bfloat16 |
| `decoder.temporal_fusion.ffn.drop` | Dropout | 4 | (1x5000x688 bfloat16) | 1x5000x688 bfloat16 |
| `decoder.temporal_fusion.ffn.w2` | Linear | 2 | (1x5000x688 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.ffn` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate.1` | Sigmoid | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate.1` | Sigmoid | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate.1` | Tanh | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate` | GRUGate | 2 | (1x5000x256 float32, 1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.mem_out` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion` | TemporalFusionModule | 2 | (1x5000x256 float32) | (1x5000x256 float32, 1x5000x256 float32) |
| `decoder.coarse_occ_head.0` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_occ_head.1` | Linear | 2 | (1x5000x256 float32) | 1x5000x128 bfloat16 |
| `decoder.coarse_occ_head.2` | GELU | 2 | (1x5000x128 bfloat16) | 1x5000x128 bfloat16 |
| `decoder.coarse_occ_head.3` | Linear | 2 | (1x5000x128 bfloat16) | 1x5000x1 bfloat16 |
| `decoder.coarse_occ_head` | Sequential | 2 | (1x5000x256 float32) | 1x5000x1 bfloat16 |
| `decoder.coarse_to_fine.0` | RMSNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.coarse_to_fine.1.w12` | Linear | 2 | (1x102400x256 float32) | 1x102400x688 bfloat16 |
| `decoder.coarse_to_fine.1.drop` | Dropout | 4 | (1x102400x344 bfloat16) | 1x102400x344 bfloat16 |
| `decoder.coarse_to_fine.1.w2` | Linear | 2 | (1x102400x344 bfloat16) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine.1` | FeedForward | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine` | Sequential | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine_norm` | RMSNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.0.norm_cross` | RMSNorm | 2 | (1x25600x256 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.0.cross_attn.sampling_offsets` | Linear | 2 | (1x25600x256 float32) | 1x25600x1536 bfloat16 |
| `decoder.fine_layers.0.cross_attn.attention_weights` | Linear | 2 | (1x25600x256 float32) | 1x25600x768 bfloat16 |
| `decoder.fine_layers.0.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn.output_proj` | Linear | 2 | (1x25600x256 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn.dropout` | Dropout | 2 | (1x25600x256 bfloat16) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn` | DeformableCrossAttention | 2 | (1x25600x256 float32, 1x25600x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0.norm_mlp` | RMSNorm | 2 | (1x25600x256 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.0.mlp.w12` | Linear | 2 | (1x25600x256 float32) | 1x25600x1376 bfloat16 |
| `decoder.fine_layers.0.mlp.drop` | Dropout | 4 | (1x25600x688 bfloat16) | 1x25600x688 bfloat16 |
| `decoder.fine_layers.0.mlp.w2` | Linear | 2 | (1x25600x688 bfloat16) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0.mlp` | FeedForward | 2 | (1x25600x256 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0` | DeformableDecoderLayer | 2 | (1x25600x256 float32, 1x25600x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.1.norm_cross` | RMSNorm | 2 | (1x25600x256 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.1.cross_attn.sampling_offsets` | Linear | 2 | (1x25600x256 float32) | 1x25600x1536 bfloat16 |
| `decoder.fine_layers.1.cross_attn.attention_weights` | Linear | 2 | (1x25600x256 float32) | 1x25600x768 bfloat16 |
| `decoder.fine_layers.1.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn.output_proj` | Linear | 2 | (1x25600x256 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn.dropout` | Dropout | 2 | (1x25600x256 bfloat16) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn` | DeformableCrossAttention | 2 | (1x25600x256 float32, 1x25600x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1.norm_mlp` | RMSNorm | 2 | (1x25600x256 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.1.mlp.w12` | Linear | 2 | (1x25600x256 float32) | 1x25600x1376 bfloat16 |
| `decoder.fine_layers.1.mlp.drop` | Dropout | 4 | (1x25600x688 bfloat16) | 1x25600x688 bfloat16 |
| `decoder.fine_layers.1.mlp.w2` | Linear | 2 | (1x25600x688 bfloat16) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1.mlp` | FeedForward | 2 | (1x25600x256 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1` | DeformableDecoderLayer | 2 | (1x25600x256 float32, 1x25600x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x25600x256 float32 |
| `decoder.fine_spatial_conv.0` | Conv3d | 2 | (1x256x80x80x16 float32) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv.1` | BatchNorm3d | 2 | (1x256x80x80x16 bfloat16) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv.2` | GELU | 2 | (1x256x80x80x16 bfloat16) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv` | Sequential | 2 | (1x256x80x80x16 float32) | 1x256x80x80x16 bfloat16 |
| `decoder` | OccupancyDecoder | 2 | ({16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | {fine_feats: 1x80x80x16x256 float32, memory: 1x5000x256 float32, coarse_occ_logits: 1x25x25x8 bfloat16, fine_selected: 1x25600 int64} |
| `head.reduce.0` | Conv3d | 2 | (1x256x80x80x16 float32) | 1x128x80x80x16 bfloat16 |
| `head.reduce.1` | BatchNorm3d | 2 | (1x128x80x80x16 bfloat16) | 1x128x80x80x16 bfloat16 |
| `head.reduce.2` | GELU | 2 | (1x128x80x80x16 bfloat16) | 1x128x80x80x16 bfloat16 |
| `head.reduce.3` | Conv3d | 2 | (1x128x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.reduce.4` | BatchNorm3d | 2 | (1x64x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.reduce.5` | GELU | 2 | (1x64x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.reduce` | Sequential | 2 | (1x256x80x80x16 float32) | 1x64x80x80x16 bfloat16 |
| `head.mlp.0` | Linear | 20 | (1x524288x103 float32) | 1x524288x128 bfloat16 |
| `head.mlp.1` | GELU | 20 | (1x524288x128 bfloat16) | 1x524288x128 bfloat16 |
| `head.mlp.2` | Linear | 20 | (1x524288x128 bfloat16) | 1x524288x128 bfloat16 |
| `head.mlp.3` | GELU | 20 | (1x524288x128 bfloat16) | 1x524288x128 bfloat16 |
| `head.mlp.4` | Linear | 20 | (1x524288x128 bfloat16) | 1x524288x21 bfloat16 |
| `head.mlp` | Sequential | 20 | (1x524288x103 float32) | 1x524288x21 bfloat16 |
| `head` | ImplicitVoxelHead | 2 | (1x80x80x16x256 float32) | {semantic: 1x18x400x400x32 bfloat16, flow: 1x3x400x400x32 bfloat16} |
| `decoder.temporal_fusion.time_embed.mlp.0` | Linear | 1 | (1x16 float32) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed.mlp.1` | SiLU | 1 | (1x256 bfloat16) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed.mlp.2` | Linear | 1 | (1x256 bfloat16) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed.mlp` | Sequential | 1 | (1x16 float32) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed` | TimeEmbedding | 1 | (1 float32) | 1x1x256 bfloat16 |

## 配置: 全局时序注意力（temporal_attn_type=global）

参数量 13.51M，两帧(第二帧带时序记忆)前向，batch=1，bf16 autocast。"调用次数">1 表示该模块在相机/层级/帧循环中被多次调用，形状取第一次调用。

| 模块 | 类型 | 调用次数 | 输入 | 输出 |
|---|---|---|---|---|
| `patch_embed.patch_embed.rggb_conv` | Conv2d | 2 | (8x1x960x1280 float32) | 8x4x480x640 bfloat16 |
| `patch_embed.patch_embed.stage1.0` | Conv2d | 2 | (8x4x480x640 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1.1` | BatchNorm2d | 2 | (8x64x240x320 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1.2` | GELU | 2 | (8x64x240x320 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1` | Sequential | 2 | (8x4x480x640 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage2.0` | Conv2d | 2 | (8x64x240x320 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2.1` | BatchNorm2d | 2 | (8x128x120x160 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2.2` | GELU | 2 | (8x128x120x160 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2` | Sequential | 2 | (8x64x240x320 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage3.0` | Conv2d | 2 | (8x128x120x160 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.1` | BatchNorm2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.2` | GELU | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.3` | Conv2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.4` | BatchNorm2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3` | Sequential | 2 | (8x128x120x160 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed` | RAWPatchEmbed | 2 | (8x1x960x1280 float32) | (8x128x120x160 bfloat16, 8x256x60x80 bfloat16) |
| `patch_embed.norm16` | LayerNorm | 2 | (8x60x80x256 bfloat16) | 8x60x80x256 float32 |
| `patch_embed.proj8` | Conv2d | 2 | (8x128x120x160 bfloat16) | 8x256x120x160 bfloat16 |
| `patch_embed.norm8` | LayerNorm | 2 | (8x120x160x256 bfloat16) | 8x120x160x256 float32 |
| `patch_embed` | MultiCameraPatchEmbed | 2 | (1x8x1x960x1280 float32) | {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32} |
| `encoder.ray_embed.proj.0` | Linear | 6 | (1x8x60x80x63 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed.proj.1` | LayerNorm | 6 | (1x8x60x80x256 bfloat16) | 1x8x60x80x256 float32 |
| `encoder.ray_embed.proj.2` | ReLU | 6 | (1x8x60x80x256 float32) | 1x8x60x80x256 float32 |
| `encoder.ray_embed.proj.3` | Linear | 6 | (1x8x60x80x256 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed.proj` | Sequential | 6 | (1x8x60x80x63 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed` | RayDirectionEncoding | 6 | (1x8x3x3 float32, 1x8x4x4 float32) | 1x8x256x60x80 bfloat16 |
| `encoder.blocks.0.norm1` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.0.attn.qkv` | Linear | 16 | (108x49x256 float32) | 108x49x768 bfloat16 |
| `encoder.blocks.0.attn.proj` | Linear | 16 | (108x49x256 bfloat16) | 108x49x256 bfloat16 |
| `encoder.blocks.0.attn` | WindowAttention | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0.norm2` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.0.mlp.w12` | Linear | 16 | (1x60x80x256 float32) | 1x60x80x1376 bfloat16 |
| `encoder.blocks.0.mlp.drop` | Dropout | 32 | (1x60x80x688 bfloat16) | 1x60x80x688 bfloat16 |
| `encoder.blocks.0.mlp.w2` | Linear | 16 | (1x60x80x688 bfloat16) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0.mlp` | FeedForward | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0` | EncoderBlock | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.norm1` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.attn.qkv` | Linear | 16 | (108x49x256 float32) | 108x49x768 bfloat16 |
| `encoder.blocks.1.attn.proj` | Linear | 16 | (108x49x256 bfloat16) | 108x49x256 bfloat16 |
| `encoder.blocks.1.attn` | WindowAttention | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1.norm2` | RMSNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.mlp.w12` | Linear | 16 | (1x60x80x256 float32) | 1x60x80x1376 bfloat16 |
| `encoder.blocks.1.mlp.drop` | Dropout | 32 | (1x60x80x688 bfloat16) | 1x60x80x688 bfloat16 |
| `encoder.blocks.1.mlp.w2` | Linear | 16 | (1x60x80x688 bfloat16) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1.mlp` | FeedForward | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1` | EncoderBlock | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.norm` | RMSNorm | 2 | (1x8x60x80x256 float32) | 1x8x60x80x256 float32 |
| `encoder.down32.0` | Conv2d | 2 | (8x256x60x80 float32) | 8x256x30x40 bfloat16 |
| `encoder.down32.1` | BatchNorm2d | 2 | (8x256x30x40 bfloat16) | 8x256x30x40 bfloat16 |
| `encoder.down32.2` | GELU | 2 | (8x256x30x40 bfloat16) | 8x256x30x40 bfloat16 |
| `encoder.down32` | Sequential | 2 | (8x256x60x80 float32) | 8x256x30x40 bfloat16 |
| `encoder.depth_head.net.0` | Conv2d | 2 | (8x256x120x160 float32) | 8x256x120x160 bfloat16 |
| `encoder.depth_head.net.1` | BatchNorm2d | 2 | (8x256x120x160 bfloat16) | 8x256x120x160 bfloat16 |
| `encoder.depth_head.net.2` | GELU | 2 | (8x256x120x160 bfloat16) | 8x256x120x160 bfloat16 |
| `encoder.depth_head.net.3` | Conv2d | 2 | (8x256x120x160 bfloat16) | 8x64x120x160 bfloat16 |
| `encoder.depth_head.net` | Sequential | 2 | (8x256x120x160 float32) | 8x64x120x160 bfloat16 |
| `encoder.depth_head` | DepthHead | 2 | (8x256x120x160 float32) | 8x64x120x160 bfloat16 |
| `encoder` | ImageEncoder | 2 | ({16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32}, 1x8x3x3 float32, 1x8x4x4 float32) | {feats: {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, depth_logits: 1x8x64x120x160 bfloat16} |
| `decoder.pos_3d` | SineCosinePositionEncoding3D | 4 |  | 5000x256 float32 |
| `decoder.coarse_layers.0.norm_self` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.self_attn.qkv` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.0.self_attn.rope` | RoPE3D | 2 | (1x5000x3 float32) | (1x1x5000x16 float32, 1x1x5000x16 float32) |
| `decoder.coarse_layers.0.self_attn.proj` | Linear | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.self_attn` | MultiHeadSelfAttention | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.norm_cross` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.cross_attn.sampling_offsets` | Linear | 2 | (1x5000x256 float32) | 1x5000x1536 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.attention_weights` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.output_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.dropout` | Dropout | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn` | DeformableCrossAttention | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.norm_mlp` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.mlp.w12` | Linear | 2 | (1x5000x256 float32) | 1x5000x1376 bfloat16 |
| `decoder.coarse_layers.0.mlp.drop` | Dropout | 4 | (1x5000x688 bfloat16) | 1x5000x688 bfloat16 |
| `decoder.coarse_layers.0.mlp.w2` | Linear | 2 | (1x5000x688 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.mlp` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0` | DeformableDecoderLayer | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.norm_self` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.self_attn.qkv` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.1.self_attn.rope` | RoPE3D | 2 | (1x5000x3 float32) | (1x1x5000x16 float32, 1x1x5000x16 float32) |
| `decoder.coarse_layers.1.self_attn.proj` | Linear | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.self_attn` | MultiHeadSelfAttention | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.norm_cross` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.cross_attn.sampling_offsets` | Linear | 2 | (1x5000x256 float32) | 1x5000x1536 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.attention_weights` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.output_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.dropout` | Dropout | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn` | DeformableCrossAttention | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.norm_mlp` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.mlp.w12` | Linear | 2 | (1x5000x256 float32) | 1x5000x1376 bfloat16 |
| `decoder.coarse_layers.1.mlp.drop` | Dropout | 4 | (1x5000x688 bfloat16) | 1x5000x688 bfloat16 |
| `decoder.coarse_layers.1.mlp.w2` | Linear | 2 | (1x5000x688 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.mlp` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1` | DeformableDecoderLayer | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.norm_q` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.norm_kv` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.temporal_attn.q_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.k_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.v_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.o_proj` | Linear | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.temporal_attn` | EfficientTemporalAttention | 2 | (1x5000x256 float32, 1x5000x256 float32, 1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.norm2` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.ffn.w12` | Linear | 2 | (1x5000x256 float32) | 1x5000x1376 bfloat16 |
| `decoder.temporal_fusion.ffn.drop` | Dropout | 4 | (1x5000x688 bfloat16) | 1x5000x688 bfloat16 |
| `decoder.temporal_fusion.ffn.w2` | Linear | 2 | (1x5000x688 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.ffn` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate.1` | Sigmoid | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate.1` | Sigmoid | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate.1` | Tanh | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate` | GRUGate | 2 | (1x5000x256 float32, 1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.mem_out` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion` | TemporalFusionModule | 2 | (1x5000x256 float32) | (1x5000x256 float32, 1x5000x256 float32) |
| `decoder.coarse_occ_head.0` | RMSNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_occ_head.1` | Linear | 2 | (1x5000x256 float32) | 1x5000x128 bfloat16 |
| `decoder.coarse_occ_head.2` | GELU | 2 | (1x5000x128 bfloat16) | 1x5000x128 bfloat16 |
| `decoder.coarse_occ_head.3` | Linear | 2 | (1x5000x128 bfloat16) | 1x5000x1 bfloat16 |
| `decoder.coarse_occ_head` | Sequential | 2 | (1x5000x256 float32) | 1x5000x1 bfloat16 |
| `decoder.coarse_to_fine.0` | RMSNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.coarse_to_fine.1.w12` | Linear | 2 | (1x102400x256 float32) | 1x102400x688 bfloat16 |
| `decoder.coarse_to_fine.1.drop` | Dropout | 4 | (1x102400x344 bfloat16) | 1x102400x344 bfloat16 |
| `decoder.coarse_to_fine.1.w2` | Linear | 2 | (1x102400x344 bfloat16) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine.1` | FeedForward | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine` | Sequential | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine_norm` | RMSNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.0.norm_cross` | RMSNorm | 2 | (1x25600x256 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.0.cross_attn.sampling_offsets` | Linear | 2 | (1x25600x256 float32) | 1x25600x1536 bfloat16 |
| `decoder.fine_layers.0.cross_attn.attention_weights` | Linear | 2 | (1x25600x256 float32) | 1x25600x768 bfloat16 |
| `decoder.fine_layers.0.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn.output_proj` | Linear | 2 | (1x25600x256 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn.dropout` | Dropout | 2 | (1x25600x256 bfloat16) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn` | DeformableCrossAttention | 2 | (1x25600x256 float32, 1x25600x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0.norm_mlp` | RMSNorm | 2 | (1x25600x256 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.0.mlp.w12` | Linear | 2 | (1x25600x256 float32) | 1x25600x1376 bfloat16 |
| `decoder.fine_layers.0.mlp.drop` | Dropout | 4 | (1x25600x688 bfloat16) | 1x25600x688 bfloat16 |
| `decoder.fine_layers.0.mlp.w2` | Linear | 2 | (1x25600x688 bfloat16) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0.mlp` | FeedForward | 2 | (1x25600x256 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.0` | DeformableDecoderLayer | 2 | (1x25600x256 float32, 1x25600x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.1.norm_cross` | RMSNorm | 2 | (1x25600x256 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.1.cross_attn.sampling_offsets` | Linear | 2 | (1x25600x256 float32) | 1x25600x1536 bfloat16 |
| `decoder.fine_layers.1.cross_attn.attention_weights` | Linear | 2 | (1x25600x256 float32) | 1x25600x768 bfloat16 |
| `decoder.fine_layers.1.cross_attn.value_proj` | Linear | 6 | (1x8x120x160x256 float32) | 1x8x120x160x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn.output_proj` | Linear | 2 | (1x25600x256 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn.dropout` | Dropout | 2 | (1x25600x256 bfloat16) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn` | DeformableCrossAttention | 2 | (1x25600x256 float32, 1x25600x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1.norm_mlp` | RMSNorm | 2 | (1x25600x256 float32) | 1x25600x256 float32 |
| `decoder.fine_layers.1.mlp.w12` | Linear | 2 | (1x25600x256 float32) | 1x25600x1376 bfloat16 |
| `decoder.fine_layers.1.mlp.drop` | Dropout | 4 | (1x25600x688 bfloat16) | 1x25600x688 bfloat16 |
| `decoder.fine_layers.1.mlp.w2` | Linear | 2 | (1x25600x688 bfloat16) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1.mlp` | FeedForward | 2 | (1x25600x256 float32) | 1x25600x256 bfloat16 |
| `decoder.fine_layers.1` | DeformableDecoderLayer | 2 | (1x25600x256 float32, 1x25600x3 float32, {16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x25600x256 float32 |
| `decoder.fine_spatial_conv.0` | Conv3d | 2 | (1x256x80x80x16 float32) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv.1` | BatchNorm3d | 2 | (1x256x80x80x16 bfloat16) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv.2` | GELU | 2 | (1x256x80x80x16 bfloat16) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv` | Sequential | 2 | (1x256x80x80x16 float32) | 1x256x80x80x16 bfloat16 |
| `decoder` | OccupancyDecoder | 2 | ({16: 1x8x256x60x80 float32, 8: 1x8x256x120x160 float32, 32: 1x8x256x30x40 bfloat16}, 1x8x3x3 float32, 1x8x4x4 float32) | {fine_feats: 1x80x80x16x256 float32, memory: 1x5000x256 float32, coarse_occ_logits: 1x25x25x8 bfloat16, fine_selected: 1x25600 int64} |
| `head.conv1.0` | Conv3d | 2 | (1x256x80x80x16 float32) | 1x128x80x80x16 bfloat16 |
| `head.conv1.1` | BatchNorm3d | 2 | (1x128x80x80x16 bfloat16) | 1x128x80x80x16 bfloat16 |
| `head.conv1.2` | GELU | 2 | (1x128x80x80x16 bfloat16) | 1x128x80x80x16 bfloat16 |
| `head.conv1` | Sequential | 2 | (1x256x80x80x16 float32) | 1x128x80x80x16 bfloat16 |
| `head.conv2.0` | Conv3d | 2 | (1x128x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.conv2.1` | BatchNorm3d | 2 | (1x64x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.conv2.2` | GELU | 2 | (1x64x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.conv2` | Sequential | 2 | (1x128x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.refine1.0` | Conv3d | 2 | (1x64x200x200x32 float32) | 1x32x200x200x32 bfloat16 |
| `head.refine1.1` | BatchNorm3d | 2 | (1x32x200x200x32 bfloat16) | 1x32x200x200x32 bfloat16 |
| `head.refine1.2` | GELU | 2 | (1x32x200x200x32 bfloat16) | 1x32x200x200x32 bfloat16 |
| `head.refine1` | Sequential | 2 | (1x64x200x200x32 float32) | 1x32x200x200x32 bfloat16 |
| `head.skip1` | Conv3d | 2 | (1x64x200x200x32 float32) | 1x32x200x200x32 bfloat16 |
| `head.flow.0` | Conv3d | 2 | (1x32x200x200x32 bfloat16) | 1x16x200x200x32 bfloat16 |
| `head.flow.1` | GELU | 2 | (1x16x200x200x32 bfloat16) | 1x16x200x200x32 bfloat16 |
| `head.flow.2` | Conv3d | 2 | (1x16x200x200x32 bfloat16) | 1x3x200x200x32 bfloat16 |
| `head.flow` | Sequential | 2 | (1x32x200x200x32 bfloat16) | 1x3x200x200x32 bfloat16 |
| `head.refine2.0` | Conv3d | 2 | (1x32x400x400x32 float32) | 1x18x400x400x32 bfloat16 |
| `head.refine2.1` | BatchNorm3d | 2 | (1x18x400x400x32 bfloat16) | 1x18x400x400x32 bfloat16 |
| `head.refine2` | Sequential | 2 | (1x32x400x400x32 float32) | 1x18x400x400x32 bfloat16 |
| `head.skip2` | Conv3d | 2 | (1x32x400x400x32 float32) | 1x18x400x400x32 bfloat16 |
| `head` | VoxelHead | 2 | (1x80x80x16x256 float32) | {semantic: 1x18x400x400x32 bfloat16, flow: 1x3x400x400x32 float32} |
| `decoder.temporal_fusion.time_embed.mlp.0` | Linear | 1 | (1x16 float32) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed.mlp.1` | SiLU | 1 | (1x256 bfloat16) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed.mlp.2` | Linear | 1 | (1x256 bfloat16) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed.mlp` | Sequential | 1 | (1x16 float32) | 1x256 bfloat16 |
| `decoder.temporal_fusion.time_embed` | TimeEmbedding | 1 | (1 float32) | 1x1x256 bfloat16 |

## 配置: 接近重构前基线（单尺度/无深度/无流场/LayerNorm/GELU/无RoPE/无移位窗口）

参数量 10.59M，两帧(第二帧带时序记忆)前向，batch=1，bf16 autocast。"调用次数">1 表示该模块在相机/层级/帧循环中被多次调用，形状取第一次调用。

| 模块 | 类型 | 调用次数 | 输入 | 输出 |
|---|---|---|---|---|
| `patch_embed.patch_embed.rggb_conv` | Conv2d | 2 | (8x1x960x1280 float32) | 8x4x480x640 bfloat16 |
| `patch_embed.patch_embed.stage1.0` | Conv2d | 2 | (8x4x480x640 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1.1` | BatchNorm2d | 2 | (8x64x240x320 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1.2` | GELU | 2 | (8x64x240x320 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage1` | Sequential | 2 | (8x4x480x640 bfloat16) | 8x64x240x320 bfloat16 |
| `patch_embed.patch_embed.stage2.0` | Conv2d | 2 | (8x64x240x320 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2.1` | BatchNorm2d | 2 | (8x128x120x160 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2.2` | GELU | 2 | (8x128x120x160 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage2` | Sequential | 2 | (8x64x240x320 bfloat16) | 8x128x120x160 bfloat16 |
| `patch_embed.patch_embed.stage3.0` | Conv2d | 2 | (8x128x120x160 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.1` | BatchNorm2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.2` | GELU | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.3` | Conv2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3.4` | BatchNorm2d | 2 | (8x256x60x80 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed.stage3` | Sequential | 2 | (8x128x120x160 bfloat16) | 8x256x60x80 bfloat16 |
| `patch_embed.patch_embed` | RAWPatchEmbed | 2 | (8x1x960x1280 float32) | (8x128x120x160 bfloat16, 8x256x60x80 bfloat16) |
| `patch_embed.norm16` | LayerNorm | 2 | (8x60x80x256 bfloat16) | 8x60x80x256 float32 |
| `patch_embed` | MultiCameraPatchEmbed | 2 | (1x8x1x960x1280 float32) | {16: 1x8x256x60x80 float32} |
| `encoder.ray_embed.proj.0` | Linear | 2 | (1x8x60x80x63 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed.proj.1` | LayerNorm | 2 | (1x8x60x80x256 bfloat16) | 1x8x60x80x256 float32 |
| `encoder.ray_embed.proj.2` | ReLU | 2 | (1x8x60x80x256 float32) | 1x8x60x80x256 float32 |
| `encoder.ray_embed.proj.3` | Linear | 2 | (1x8x60x80x256 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed.proj` | Sequential | 2 | (1x8x60x80x63 float32) | 1x8x60x80x256 bfloat16 |
| `encoder.ray_embed` | RayDirectionEncoding | 2 | (1x8x3x3 float32, 1x8x4x4 float32) | 1x8x256x60x80 bfloat16 |
| `encoder.blocks.0.norm1` | LayerNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.0.attn.qkv` | Linear | 16 | (108x49x256 float32) | 108x49x768 bfloat16 |
| `encoder.blocks.0.attn.proj` | Linear | 16 | (108x49x256 bfloat16) | 108x49x256 bfloat16 |
| `encoder.blocks.0.attn` | WindowAttention | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0.norm2` | LayerNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.0.mlp.w1` | Linear | 16 | (1x60x80x256 float32) | 1x60x80x1024 bfloat16 |
| `encoder.blocks.0.mlp.drop` | Dropout | 32 | (1x60x80x1024 bfloat16) | 1x60x80x1024 bfloat16 |
| `encoder.blocks.0.mlp.w2` | Linear | 16 | (1x60x80x1024 bfloat16) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0.mlp` | FeedForward | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.0` | EncoderBlock | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.norm1` | LayerNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.attn.qkv` | Linear | 16 | (108x49x256 float32) | 108x49x768 bfloat16 |
| `encoder.blocks.1.attn.proj` | Linear | 16 | (108x49x256 bfloat16) | 108x49x256 bfloat16 |
| `encoder.blocks.1.attn` | WindowAttention | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1.norm2` | LayerNorm | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.blocks.1.mlp.w1` | Linear | 16 | (1x60x80x256 float32) | 1x60x80x1024 bfloat16 |
| `encoder.blocks.1.mlp.drop` | Dropout | 32 | (1x60x80x1024 bfloat16) | 1x60x80x1024 bfloat16 |
| `encoder.blocks.1.mlp.w2` | Linear | 16 | (1x60x80x1024 bfloat16) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1.mlp` | FeedForward | 16 | (1x60x80x256 float32) | 1x60x80x256 bfloat16 |
| `encoder.blocks.1` | EncoderBlock | 16 | (1x60x80x256 float32) | 1x60x80x256 float32 |
| `encoder.norm` | LayerNorm | 2 | (1x8x60x80x256 float32) | 1x8x60x80x256 float32 |
| `encoder` | ImageEncoder | 2 | ({16: 1x8x256x60x80 float32}, 1x8x3x3 float32, 1x8x4x4 float32) | {feats: {16: 1x8x256x60x80 float32}} |
| `decoder.pos_3d` | SineCosinePositionEncoding3D | 4 |  | 5000x256 float32 |
| `decoder.coarse_layers.0.norm_self` | LayerNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.self_attn.qkv` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.0.self_attn.proj` | Linear | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.self_attn` | MultiHeadSelfAttention | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.norm_cross` | LayerNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.cross_attn.sampling_offsets` | Linear | 2 | (1x5000x256 float32) | 1x5000x512 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.attention_weights` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.value_proj` | Linear | 2 | (1x8x60x80x256 float32) | 1x8x60x80x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.output_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn.dropout` | Dropout | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.cross_attn` | DeformableCrossAttention | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.norm_mlp` | LayerNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.0.mlp.w1` | Linear | 2 | (1x5000x256 float32) | 1x5000x1024 bfloat16 |
| `decoder.coarse_layers.0.mlp.drop` | Dropout | 4 | (1x5000x1024 bfloat16) | 1x5000x1024 bfloat16 |
| `decoder.coarse_layers.0.mlp.w2` | Linear | 2 | (1x5000x1024 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0.mlp` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.0` | DeformableDecoderLayer | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.norm_self` | LayerNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.self_attn.qkv` | Linear | 2 | (1x5000x256 float32) | 1x5000x768 bfloat16 |
| `decoder.coarse_layers.1.self_attn.proj` | Linear | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.self_attn` | MultiHeadSelfAttention | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.norm_cross` | LayerNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.cross_attn.sampling_offsets` | Linear | 2 | (1x5000x256 float32) | 1x5000x512 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.attention_weights` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.value_proj` | Linear | 2 | (1x8x60x80x256 float32) | 1x8x60x80x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.output_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn.dropout` | Dropout | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.cross_attn` | DeformableCrossAttention | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.norm_mlp` | LayerNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_layers.1.mlp.w1` | Linear | 2 | (1x5000x256 float32) | 1x5000x1024 bfloat16 |
| `decoder.coarse_layers.1.mlp.drop` | Dropout | 4 | (1x5000x1024 bfloat16) | 1x5000x1024 bfloat16 |
| `decoder.coarse_layers.1.mlp.w2` | Linear | 2 | (1x5000x1024 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1.mlp` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.coarse_layers.1` | DeformableDecoderLayer | 2 | (1x5000x256 float32, 1x5000x3 float32, {16: 1x8x256x60x80 float32}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.norm_q` | LayerNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.norm_kv` | LayerNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.temporal_attn.q_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.k_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.v_proj` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.temporal_attn.o_proj` | Linear | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.temporal_attn` | EfficientTemporalAttention | 2 | (1x5000x256 float32, 1x5000x256 float32, 1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.norm2` | LayerNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.ffn.w1` | Linear | 2 | (1x5000x256 float32) | 1x5000x1024 bfloat16 |
| `decoder.temporal_fusion.ffn.drop` | Dropout | 4 | (1x5000x1024 bfloat16) | 1x5000x1024 bfloat16 |
| `decoder.temporal_fusion.ffn.w2` | Linear | 2 | (1x5000x1024 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.ffn` | FeedForward | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate.1` | Sigmoid | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.update_gate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate.1` | Sigmoid | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.reset_gate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate.0` | Linear | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate.1` | Tanh | 2 | (1x5000x256 bfloat16) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate.candidate` | Sequential | 2 | (1x5000x512 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion.gate` | GRUGate | 2 | (1x5000x256 float32, 1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.temporal_fusion.mem_out` | Linear | 2 | (1x5000x256 float32) | 1x5000x256 bfloat16 |
| `decoder.temporal_fusion` | TemporalFusionModule | 2 | (1x5000x256 float32) | (1x5000x256 float32, 1x5000x256 float32) |
| `decoder.coarse_occ_head.0` | LayerNorm | 2 | (1x5000x256 float32) | 1x5000x256 float32 |
| `decoder.coarse_occ_head.1` | Linear | 2 | (1x5000x256 float32) | 1x5000x128 bfloat16 |
| `decoder.coarse_occ_head.2` | GELU | 2 | (1x5000x128 bfloat16) | 1x5000x128 bfloat16 |
| `decoder.coarse_occ_head.3` | Linear | 2 | (1x5000x128 bfloat16) | 1x5000x1 bfloat16 |
| `decoder.coarse_occ_head` | Sequential | 2 | (1x5000x256 float32) | 1x5000x1 bfloat16 |
| `decoder.coarse_to_fine.0` | LayerNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.coarse_to_fine.1.w1` | Linear | 2 | (1x102400x256 float32) | 1x102400x512 bfloat16 |
| `decoder.coarse_to_fine.1.drop` | Dropout | 4 | (1x102400x512 bfloat16) | 1x102400x512 bfloat16 |
| `decoder.coarse_to_fine.1.w2` | Linear | 2 | (1x102400x512 bfloat16) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine.1` | FeedForward | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine` | Sequential | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.coarse_to_fine_norm` | LayerNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.0.norm_cross` | LayerNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.0.cross_attn.sampling_offsets` | Linear | 2 | (1x102400x256 float32) | 1x102400x512 bfloat16 |
| `decoder.fine_layers.0.cross_attn.attention_weights` | Linear | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn.value_proj` | Linear | 2 | (1x8x60x80x256 float32) | 1x8x60x80x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn.output_proj` | Linear | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn.dropout` | Dropout | 2 | (1x102400x256 bfloat16) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.0.cross_attn` | DeformableCrossAttention | 2 | (1x102400x256 float32, 1x102400x3 float32, {16: 1x8x256x60x80 float32}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.0.norm_mlp` | LayerNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.0.mlp.w1` | Linear | 2 | (1x102400x256 float32) | 1x102400x1024 bfloat16 |
| `decoder.fine_layers.0.mlp.drop` | Dropout | 4 | (1x102400x1024 bfloat16) | 1x102400x1024 bfloat16 |
| `decoder.fine_layers.0.mlp.w2` | Linear | 2 | (1x102400x1024 bfloat16) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.0.mlp` | FeedForward | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.0` | DeformableDecoderLayer | 2 | (1x102400x256 float32, 1x102400x3 float32, {16: 1x8x256x60x80 float32}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.1.norm_cross` | LayerNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.1.cross_attn.sampling_offsets` | Linear | 2 | (1x102400x256 float32) | 1x102400x512 bfloat16 |
| `decoder.fine_layers.1.cross_attn.attention_weights` | Linear | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn.value_proj` | Linear | 2 | (1x8x60x80x256 float32) | 1x8x60x80x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn.output_proj` | Linear | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn.dropout` | Dropout | 2 | (1x102400x256 bfloat16) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.1.cross_attn` | DeformableCrossAttention | 2 | (1x102400x256 float32, 1x102400x3 float32, {16: 1x8x256x60x80 float32}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.1.norm_mlp` | LayerNorm | 2 | (1x102400x256 float32) | 1x102400x256 float32 |
| `decoder.fine_layers.1.mlp.w1` | Linear | 2 | (1x102400x256 float32) | 1x102400x1024 bfloat16 |
| `decoder.fine_layers.1.mlp.drop` | Dropout | 4 | (1x102400x1024 bfloat16) | 1x102400x1024 bfloat16 |
| `decoder.fine_layers.1.mlp.w2` | Linear | 2 | (1x102400x1024 bfloat16) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.1.mlp` | FeedForward | 2 | (1x102400x256 float32) | 1x102400x256 bfloat16 |
| `decoder.fine_layers.1` | DeformableDecoderLayer | 2 | (1x102400x256 float32, 1x102400x3 float32, {16: 1x8x256x60x80 float32}, 1x8x3x3 float32, 1x8x4x4 float32) | 1x102400x256 float32 |
| `decoder.fine_spatial_conv.0` | Conv3d | 2 | (1x256x80x80x16 float32) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv.1` | BatchNorm3d | 2 | (1x256x80x80x16 bfloat16) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv.2` | GELU | 2 | (1x256x80x80x16 bfloat16) | 1x256x80x80x16 bfloat16 |
| `decoder.fine_spatial_conv` | Sequential | 2 | (1x256x80x80x16 float32) | 1x256x80x80x16 bfloat16 |
| `decoder` | OccupancyDecoder | 2 | ({16: 1x8x256x60x80 float32}, 1x8x3x3 float32, 1x8x4x4 float32) | {fine_feats: 1x80x80x16x256 float32, memory: 1x5000x256 float32, coarse_occ_logits: 1x25x25x8 bfloat16} |
| `head.conv1.0` | Conv3d | 2 | (1x256x80x80x16 float32) | 1x128x80x80x16 bfloat16 |
| `head.conv1.1` | BatchNorm3d | 2 | (1x128x80x80x16 bfloat16) | 1x128x80x80x16 bfloat16 |
| `head.conv1.2` | GELU | 2 | (1x128x80x80x16 bfloat16) | 1x128x80x80x16 bfloat16 |
| `head.conv1` | Sequential | 2 | (1x256x80x80x16 float32) | 1x128x80x80x16 bfloat16 |
| `head.conv2.0` | Conv3d | 2 | (1x128x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.conv2.1` | BatchNorm3d | 2 | (1x64x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.conv2.2` | GELU | 2 | (1x64x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.conv2` | Sequential | 2 | (1x128x80x80x16 bfloat16) | 1x64x80x80x16 bfloat16 |
| `head.refine1.0` | Conv3d | 2 | (1x64x200x200x32 float32) | 1x32x200x200x32 bfloat16 |
| `head.refine1.1` | BatchNorm3d | 2 | (1x32x200x200x32 bfloat16) | 1x32x200x200x32 bfloat16 |
| `head.refine1.2` | GELU | 2 | (1x32x200x200x32 bfloat16) | 1x32x200x200x32 bfloat16 |
| `head.refine1` | Sequential | 2 | (1x64x200x200x32 float32) | 1x32x200x200x32 bfloat16 |
| `head.skip1` | Conv3d | 2 | (1x64x200x200x32 float32) | 1x32x200x200x32 bfloat16 |
| `head.refine2.0` | Conv3d | 2 | (1x32x400x400x32 float32) | 1x18x400x400x32 bfloat16 |
| `head.refine2.1` | BatchNorm3d | 2 | (1x18x400x400x32 bfloat16) | 1x18x400x400x32 bfloat16 |
| `head.refine2` | Sequential | 2 | (1x32x400x400x32 float32) | 1x18x400x400x32 bfloat16 |
| `head.skip2` | Conv3d | 2 | (1x32x400x400x32 float32) | 1x18x400x400x32 bfloat16 |
| `head` | VoxelHead | 2 | (1x80x80x16x256 float32) | {semantic: 1x18x400x400x32 bfloat16} |

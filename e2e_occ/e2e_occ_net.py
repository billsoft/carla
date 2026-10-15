import torch.nn as nn

try:
    from .config import E2EOccConfig
    from .raw_embed import MultiCameraPatchEmbed
    from .image_encoder import ImageEncoder
    from .occ_decoder import OccupancyDecoder
    from .voxel_head import build_head
except ImportError:
    from config import E2EOccConfig
    from raw_embed import MultiCameraPatchEmbed
    from image_encoder import ImageEncoder
    from occ_decoder import OccupancyDecoder
    from voxel_head import build_head


class E2EOccNet(nn.Module):
    """
    输入:
      images      [B, N, 1, H, W]  Bayer RAW，归一化到 [0,1]
      intrinsics  [B, N, 3, 3]     等距投影内参（原图分辨率）
      extrinsics  [B, N, 4, 4]     Camera(OpenCV)→Vehicle 静态安装标定（不含车辆世界位姿）
      memory      [B, 5000, C]     上一帧时序记忆（可选）
      ego_motion  [B, 4, 4]        上一帧车辆系 → 当前帧车辆系（可选）
      time_delta  [B]              帧间隔秒数（可选）
      query_points [B, P, 3]       仅隐式头训练时使用：需要输出 logits 的点（grid_sample 归一化坐标）
      need_aux     bool            是否计算辅助输出 (深度 logits)；默认 = self.training。验证时要算深度损失就传 True
      need_flow    bool            是否计算流场 (默认 True；只要占用时传 False 省掉流场分支)
    输出 dict:
      semantic [B, 18, 400, 400, 32]（隐式头训练时为 semantic_points [B, P, 18]）
      flow     [B, 3, 400, 400, 32] 或 None（隐式头训练时为 flow_points）
      memory, depth_logits, coarse_occ_logits, fine_selected
    """

    def __init__(self, config: E2EOccConfig = None):
        super().__init__()
        self.config = config or E2EOccConfig()
        self.patch_embed = MultiCameraPatchEmbed(self.config)
        self.encoder = ImageEncoder(self.config)
        self.decoder = OccupancyDecoder(self.config)
        self.head = build_head(self.config)
        self._init_weights()

    def _init_weights(self):
        # 可变形注意力（空间/时序）的 sampling_offsets / attention_weights 在各自构造函数里
        # 做了专门初始化（偏移从几何参考点附近起步），这里按属性名跳过，不覆盖。
        skip = set()
        for m in self.modules():
            for name in ('sampling_offsets', 'attention_weights'):
                sub = getattr(m, name, None)
                if isinstance(sub, nn.Linear):
                    skip.add(id(sub))
        for m in self.modules():
            if id(m) in skip:
                continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self._init_output_layers()

    def _init_output_layers(self):
        """各预测头的最后一层初始化为接近 0 的输出：初始 logits ≈ 均匀分布（CE ≈ ln C），
        流场 ≈ 0。否则 kaiming 初始化的最后一层会给出很大的初始 logits（实测初始 CE ≈ 7，
        远高于 ln18 ≈ 2.9），训练初期先要花大量步数把错误的高置信度压下去。"""
        last = []
        head = self.head
        if hasattr(head, 'skip2'):
            last += [head.skip2, head.refine2[0]]
            nn.init.zeros_(head.refine2[1].weight)            # refine2 的 BN gamma=0，起步只走 1×1 skip
        if hasattr(head, 'mlp'):
            last.append(head.mlp[-1])
        if getattr(head, 'flow', None) is not None:
            last.append(head.flow[-1])
        if getattr(self.encoder, 'depth_head', None) is not None:
            last.append(self.encoder.depth_head.net[-1])
        last.append(self.decoder.coarse_occ_head[-1])
        for m in last:
            nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        tf = getattr(self.decoder, 'temporal_fusion', None)
        if tf is not None:
            # 记忆回注分支零初始化：起步时时序模块输出与不回注时一致
            nn.init.zeros_(tf.mem_out.weight)
            nn.init.zeros_(tf.mem_out.bias)

    def forward(self, images, intrinsics, extrinsics, memory=None, ego_motion=None, time_delta=None,
                query_points=None, need_aux=None, need_flow=True):
        if need_aux is None:
            need_aux = self.training
        feats = self.patch_embed(images)
        enc = self.encoder(feats, intrinsics, extrinsics, need_depth=need_aux)
        dec = self.decoder(enc['feats'], intrinsics, extrinsics, memory, ego_motion, time_delta)
        out = self.head(dec['fine_feats'], query_points=query_points, need_flow=need_flow)
        out.update({
            'memory': dec['memory'],
            'depth_logits': enc['depth_logits'],
            'coarse_occ_logits': dec['coarse_occ_logits'],
            'fine_selected': dec['fine_selected'],
        })
        return out

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config=None):
    config = config or E2EOccConfig()
    return E2EOccNet(config)

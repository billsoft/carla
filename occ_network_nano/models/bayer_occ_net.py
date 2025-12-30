"""
完整的 Bayer Occupancy Network

整合所有模块：
1. BayerMobileNetV2 Backbone (特征提取)
2. FPN Neck (多尺度融合)
3. View Transformer (2D→BEV)
4. BEV Encoder (BEV 特征增强)
5. Occupancy Decoder (BEV→3D 体素)
"""

import torch
import torch.nn as nn

from .backbone.mobilenet_v2_bayer import build_bayer_mobilenetv2
from .neck.fpn_lite import LiteFPN
from .transformer.view_transformer_lite import LiteViewTransformer
from .encoder.bev_encoder_lite import LiteBEVEncoder
from .decoder.occ_decoder_lite import LiteOccDecoder


class BayerOccNet(nn.Module):
    """
    完整的 Bayer RAW Occupancy Network

    从单通道 Bayer RGGB 图像预测 3D 占用网格。

    Args:
        num_classes: 占用类别数（包括空类）
        grid_size: 3D 网格尺寸 (X, Y, Z)
        img_size: 输入图像尺寸 (H, W)
        backbone_width_mult: Backbone 宽度乘数
        fpn_channels: FPN 输出通道数
        bev_size: BEV 网格尺寸 (H, W)
        hidden_channels: 中间隐藏层通道数
    """

    def __init__(
        self,
        num_classes=18,
        grid_size=(200, 200, 16),
        img_size=(384, 640),
        backbone_width_mult=1.0,
        fpn_channels=128,
        bev_size=(100, 100),
        hidden_channels=64,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.grid_size = grid_size
        self.img_size = img_size
        self.bev_size = bev_size

        # 1. Backbone: BayerMobileNetV2
        self.backbone = build_bayer_mobilenetv2(width_mult=backbone_width_mult)

        # 2. FPN Neck: 融合多尺度特征
        self.fpn = LiteFPN(
            in_channels=self.backbone.out_channels,
            out_channels=fpn_channels
        )

        # 3. View Transformer: 2D → BEV
        feat_h, feat_w = img_size[0] // 8, img_size[1] // 8  # 1/8 分辨率
        self.view_transformer = LiteViewTransformer(
            in_channels=fpn_channels,
            out_channels=fpn_channels,
            feat_height=feat_h,
            feat_width=feat_w,
            bev_height=bev_size[0],
            bev_width=bev_size[1],
            num_depth_bins=32
        )

        # 4. BEV Encoder: 增强 BEV 特征
        self.bev_encoder = LiteBEVEncoder(
            in_channels=fpn_channels,
            out_channels=fpn_channels,
            num_blocks=2
        )

        # 5. Occupancy Decoder: BEV → 3D 体素
        self.occ_decoder = LiteOccDecoder(
            in_channels=fpn_channels,
            num_classes=num_classes,
            grid_size=grid_size,
            hidden_channels=hidden_channels
        )

    def forward(self, images, camera_extrinsics=None):
        """
        前向传播

        Args:
            images: [B, N_cam, 1, H, W] Bayer 图像
            camera_extrinsics: [B, N_cam, 4, 4] 相机外参（可选）

        Returns:
            occ_logits: [B, num_classes, X, Y, Z] 占用 logits
        """
        B, N_cam, C, H, W = images.shape

        # 1. Backbone: 提取多尺度特征
        # 展平批次和相机维度
        images_flat = images.view(B * N_cam, C, H, W)
        features = self.backbone(images_flat)  # {'C3', 'C4', 'C5'}

        # 2. FPN: 融合多尺度特征
        fpn_feat = self.fpn(features)  # [B*N_cam, fpn_C, H/8, W/8]

        # 3. Reshape 回多相机维度
        _, fpn_C, feat_H, feat_W = fpn_feat.shape
        fpn_feat = fpn_feat.view(B, N_cam, fpn_C, feat_H, feat_W)

        # 4. View Transformer: 2D → BEV
        bev_feat = self.view_transformer(fpn_feat, camera_extrinsics)  # [B, fpn_C, BEV_H, BEV_W]

        # 5. BEV Encoder: 增强 BEV 特征
        bev_feat = self.bev_encoder(bev_feat)  # [B, fpn_C, BEV_H, BEV_W]

        # 6. Occupancy Decoder: BEV → 3D 体素
        occ_logits = self.occ_decoder(bev_feat)  # [B, num_classes, X, Y, Z]

        return occ_logits

    def get_params_summary(self):
        """获取参数量统计"""
        total_params = sum(p.numel() for p in self.parameters())

        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        fpn_params = sum(p.numel() for p in self.fpn.parameters())
        view_trans_params = sum(p.numel() for p in self.view_transformer.parameters())
        bev_encoder_params = sum(p.numel() for p in self.bev_encoder.parameters())
        occ_decoder_params = sum(p.numel() for p in self.occ_decoder.parameters())

        return {
            'total': total_params / 1e6,
            'backbone': backbone_params / 1e6,
            'fpn': fpn_params / 1e6,
            'view_transformer': view_trans_params / 1e6,
            'bev_encoder': bev_encoder_params / 1e6,
            'occ_decoder': occ_decoder_params / 1e6,
        }


def build_bayer_occ_net(num_classes=18, **kwargs):
    """
    构建 Bayer Occupancy Network

    Args:
        num_classes: 占用类别数
        **kwargs: 其他参数传递给 BayerOccNet

    Returns:
        BayerOccNet 模型
    """
    model = BayerOccNet(num_classes=num_classes, **kwargs)
    return model


if __name__ == '__main__':
    print("=" * 80)
    print("完整 Bayer Occupancy Network 测试".center(80))
    print("=" * 80)

    # 模拟输入
    B, N_cam = 2, 8
    H, W = 384, 640
    images = torch.randn(B, N_cam, 1, H, W)

    # 创建模型
    model = build_bayer_occ_net(
        num_classes=18,
        grid_size=(200, 200, 16),
        img_size=(H, W),
        backbone_width_mult=1.0,
        fpn_channels=128,
        bev_size=(100, 100),
        hidden_channels=64
    )

    print(f"\n输入:")
    print(f"  Images: {images.shape} (Bayer RAW)")

    # 前向传播
    print(f"\n前向传播...")
    with torch.no_grad():
        occ_logits = model(images)

    print(f"\n输出:")
    print(f"  Occupancy Logits: {occ_logits.shape}")
    print(f"  预期: [B={B}, num_classes=18, X=200, Y=200, Z=16]")

    # 参数量统计
    params_summary = model.get_params_summary()
    print(f"\n参数量统计:")
    print(f"  {'模块':<20} {'参数量 (M)':>12}")
    print(f"  {'-'*20} {'-'*12}")
    print(f"  {'Backbone':<20} {params_summary['backbone']:>12.2f}")
    print(f"  {'FPN':<20} {params_summary['fpn']:>12.2f}")
    print(f"  {'View Transformer':<20} {params_summary['view_transformer']:>12.2f}")
    print(f"  {'BEV Encoder':<20} {params_summary['bev_encoder']:>12.2f}")
    print(f"  {'Occ Decoder':<20} {params_summary['occ_decoder']:>12.2f}")
    print(f"  {'-'*20} {'-'*12}")
    print(f"  {'总计':<20} {params_summary['total']:>12.2f}")

    print("\n" + "=" * 80)
    print("✅ 完整网络测试通过！".center(80))
    print("=" * 80)

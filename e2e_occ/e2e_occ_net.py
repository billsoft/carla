import torch
import torch.nn as nn
from config import E2EOccConfig
from raw_embed import MultiCameraPatchEmbed
from image_encoder import ImageEncoder
from occ_decoder import OccupancyDecoder
from voxel_head import VoxelHead

class E2EOccNet(nn.Module):
    def __init__(self, config: E2EOccConfig = None):
        super().__init__()
        self.config = config or E2EOccConfig()
        self.patch_embed = MultiCameraPatchEmbed(self.config)
        self.encoder = ImageEncoder(self.config)
        self.decoder = OccupancyDecoder(self.config)
        self.head = VoxelHead(self.config)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, images, intrinsics=None, extrinsics=None, memory=None):
        feats = self.patch_embed(images)
        feats = self.encoder(feats, intrinsics, extrinsics)
        voxel_feats, new_memory = self.decoder(feats, intrinsics, extrinsics, memory)
        logits = self.head(voxel_feats)
        return {
            'semantic': logits,
            'memory': new_memory
        }
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_model(config=None):
    config = config or E2EOccConfig()
    return E2EOccNet(config)

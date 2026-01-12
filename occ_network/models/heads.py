import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, 1, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ChunkedConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, chunk_size_z=10):
        super().__init__()
        self.chunk_size_z = chunk_size_z
        self.conv = ConvBlock3D(in_ch, out_ch)
        self.out_conv = nn.Conv3d(out_ch, out_ch, 1)

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        if not self.training or Z <= self.chunk_size_z:
            return self.out_conv(self.conv(x))
        outputs = []
        for z_start in range(0, Z, self.chunk_size_z):
            z_end = min(z_start + self.chunk_size_z, Z)
            chunk = x[:, :, :, :, z_start:z_end]
            chunk = self.conv(chunk)
            chunk = self.out_conv(chunk)
            outputs.append(chunk)
        return torch.cat(outputs, dim=4)

class ChunkedSemanticHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels=64, chunk_size_z=10):
        super().__init__()
        self.chunk_size_z = chunk_size_z
        self.conv1 = ConvBlock3D(in_channels, hidden_channels)
        self.conv2 = nn.Conv3d(hidden_channels, num_classes, 1)

    def _forward_chunk(self, x):
        return self.conv2(self.conv1(x))

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        if not self.training or Z <= self.chunk_size_z:
            return self._forward_chunk(x)
        outputs = []
        for z_start in range(0, Z, self.chunk_size_z):
            z_end = min(z_start + self.chunk_size_z, Z)
            chunk = x[:, :, :, :, z_start:z_end]
            if self.training:
                out = checkpoint(self._forward_chunk, chunk, use_reentrant=False)
            else:
                out = self._forward_chunk(chunk)
            outputs.append(out)
        return torch.cat(outputs, dim=4)

class ChunkedFlowHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, chunk_size_z=10):
        super().__init__()
        self.chunk_size_z = chunk_size_z
        self.conv1 = ConvBlock3D(in_channels, hidden_channels)
        self.conv2 = nn.Conv3d(hidden_channels, 3, 1)

    def _forward_chunk(self, x):
        return self.conv2(self.conv1(x))

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        if not self.training or Z <= self.chunk_size_z:
            return self._forward_chunk(x)
        outputs = []
        for z_start in range(0, Z, self.chunk_size_z):
            z_end = min(z_start + self.chunk_size_z, Z)
            chunk = x[:, :, :, :, z_start:z_end]
            if self.training:
                out = checkpoint(self._forward_chunk, chunk, use_reentrant=False)
            else:
                out = self._forward_chunk(chunk)
            outputs.append(out)
        return torch.cat(outputs, dim=4)

class CoarseToFineHead(nn.Module):
    def __init__(self, in_channels, num_classes, coarse_size, fine_size, use_flow=True, chunk_size_z=10):
        super().__init__()
        self.coarse_size = coarse_size
        self.fine_size = fine_size
        self.use_flow = use_flow
        self.chunk_size_z = chunk_size_z
        self.coarse_semantic = nn.Sequential(ConvBlock3D(in_channels, in_channels // 2), nn.Conv3d(in_channels // 2, num_classes, 1))
        self.refine_conv = ConvBlock3D(num_classes + in_channels, in_channels // 2)
        self.refine_out = nn.Conv3d(in_channels // 2, num_classes, 1)
        if use_flow:
            self.coarse_flow = nn.Sequential(ConvBlock3D(in_channels, in_channels // 4), nn.Conv3d(in_channels // 4, 3, 1))
            self.refine_flow_conv = ConvBlock3D(3 + in_channels, in_channels // 4)
            self.refine_flow_out = nn.Conv3d(in_channels // 4, 3, 1)

    def _refine_chunk(self, x, coarse_up):
        concat = torch.cat([coarse_up, x], dim=1)
        return self.refine_out(self.refine_conv(concat))

    def _refine_flow_chunk(self, x, coarse_up):
        concat = torch.cat([coarse_up, x], dim=1)
        return self.refine_flow_out(self.refine_flow_conv(concat))

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        x_coarse = F.interpolate(x, size=self.coarse_size, mode='trilinear', align_corners=False)
        coarse_sem = self.coarse_semantic(x_coarse)
        coarse_sem_up = F.interpolate(coarse_sem, size=self.fine_size, mode='trilinear', align_corners=False)
        x_fine = F.interpolate(x, size=self.fine_size, mode='trilinear', align_corners=False)
        if not self.training or self.fine_size[2] <= self.chunk_size_z:
            fine_sem = self._refine_chunk(x_fine, coarse_sem_up)
        else:
            chunks = []
            Z_fine = self.fine_size[2]
            for z_start in range(0, Z_fine, self.chunk_size_z):
                z_end = min(z_start + self.chunk_size_z, Z_fine)
                x_chunk = x_fine[:, :, :, :, z_start:z_end]
                coarse_chunk = coarse_sem_up[:, :, :, :, z_start:z_end]
                out = checkpoint(self._refine_chunk, x_chunk, coarse_chunk, use_reentrant=False)
                chunks.append(out)
            fine_sem = torch.cat(chunks, dim=4)
        outputs = {'semantic': fine_sem, 'coarse_semantic': coarse_sem}
        if self.use_flow:
            coarse_flow = self.coarse_flow(x_coarse)
            coarse_flow_up = F.interpolate(coarse_flow, size=self.fine_size, mode='trilinear', align_corners=False)
            if not self.training or self.fine_size[2] <= self.chunk_size_z:
                fine_flow = self._refine_flow_chunk(x_fine, coarse_flow_up)
            else:
                chunks = []
                for z_start in range(0, Z_fine, self.chunk_size_z):
                    z_end = min(z_start + self.chunk_size_z, Z_fine)
                    x_chunk = x_fine[:, :, :, :, z_start:z_end]
                    flow_chunk = coarse_flow_up[:, :, :, :, z_start:z_end]
                    out = checkpoint(self._refine_flow_chunk, x_chunk, flow_chunk, use_reentrant=False)
                    chunks.append(out)
                fine_flow = torch.cat(chunks, dim=4)
            outputs['flow'] = fine_flow
            outputs['coarse_flow'] = coarse_flow
        return outputs

class MultiTaskHead(nn.Module):
    def __init__(self, in_channels, num_classes, use_flow=True, chunk_size_z=10):
        super().__init__()
        self.use_flow = use_flow
        self.shared = ConvBlock3D(in_channels, in_channels // 2)
        self.semantic_head = ChunkedSemanticHead(in_channels // 2, num_classes, in_channels // 4, chunk_size_z)
        if use_flow:
            self.flow_head = ChunkedFlowHead(in_channels // 2, in_channels // 4, chunk_size_z)

    def forward(self, x):
        shared = self.shared(x)
        outputs = {'semantic': self.semantic_head(shared)}
        if self.use_flow:
            outputs['flow'] = self.flow_head(shared)
        return outputs

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import spconv.pytorch as spconv
    SPCONV_AVAILABLE = True
except ImportError:
    SPCONV_AVAILABLE = False
    spconv = None

class DenseToSparse(nn.Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, dense_features, occupancy_mask=None):
        B, C, X, Y, Z = dense_features.shape
        if occupancy_mask is None:
            occupancy_mask = dense_features.abs().sum(dim=1) > self.threshold
        coords_list = []
        feats_list = []
        for b in range(B):
            mask_b = occupancy_mask[b]
            coords = torch.nonzero(mask_b, as_tuple=False)
            batch_idx = torch.full((coords.shape[0], 1), b, dtype=torch.int32, device=coords.device)
            coords = torch.cat([batch_idx, coords], dim=1)
            feats = dense_features[b, :, mask_b].t()
            coords_list.append(coords)
            feats_list.append(feats)
        if len(coords_list) > 0:
            coords = torch.cat(coords_list, dim=0)
            feats = torch.cat(feats_list, dim=0)
        else:
            coords = torch.zeros((0, 4), dtype=torch.int32, device=dense_features.device)
            feats = torch.zeros((0, C), dtype=dense_features.dtype, device=dense_features.device)
        return feats, coords, (B, X, Y, Z)

class SparseToDense(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats, coords, spatial_shape, batch_size):
        B, X, Y, Z = batch_size, *spatial_shape
        C = feats.shape[1]
        dense = torch.zeros(B, C, X, Y, Z, dtype=feats.dtype, device=feats.device)
        if coords.shape[0] > 0:
            b_idx = coords[:, 0].long()
            x_idx = coords[:, 1].long()
            y_idx = coords[:, 2].long()
            z_idx = coords[:, 3].long()
            dense[b_idx, :, x_idx, y_idx, z_idx] = feats
        return dense

class PseudoSparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.threshold = 0.1

    def forward(self, x, mask=None):
        if mask is None:
            mask = (x.abs().sum(dim=1, keepdim=True) > self.threshold).float()
        out = self.conv(x)
        return out * mask

class SparseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        if SPCONV_AVAILABLE:
            self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.conv = PseudoSparseConv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()
        self.use_spconv = SPCONV_AVAILABLE

    def forward(self, x):
        if self.use_spconv:
            x = self.conv(x)
            x = x.replace_feature(self.act(self.bn(x.features)))
            return x
        else:
            return self.act(self.bn(self.conv(x)))

class SparseSemanticHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels=64, use_sparse=True):
        super().__init__()
        self.use_sparse = use_sparse and SPCONV_AVAILABLE
        if self.use_sparse:
            self.conv1 = spconv.SubMConv3d(in_channels, hidden_channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.conv2 = spconv.SubMConv3d(hidden_channels, num_classes, 1, bias=True)
        else:
            self.conv1 = nn.Conv3d(in_channels, hidden_channels, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm3d(hidden_channels)
            self.conv2 = nn.Conv3d(hidden_channels, num_classes, 1)
        self.act = nn.GELU()

    def forward(self, x):
        if self.use_sparse:
            x = self.conv1(x)
            x = x.replace_feature(self.act(self.bn1(x.features)))
            x = self.conv2(x)
            return x
        else:
            x = self.act(self.bn1(self.conv1(x)))
            return self.conv2(x)

class AdaptiveSparseProcessor(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels=64, sparsity_threshold=0.1):
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        self.dense_to_sparse = DenseToSparse(threshold=sparsity_threshold)
        self.sparse_to_dense = SparseToDense()
        self.use_sparse = SPCONV_AVAILABLE
        if self.use_sparse:
            self.sparse_head = SparseSemanticHead(in_channels, num_classes, hidden_channels, use_sparse=True)
        else:
            self.dense_head = nn.Sequential(
                nn.Conv3d(in_channels, hidden_channels, 3, 1, 1, bias=False),
                nn.BatchNorm3d(hidden_channels),
                nn.GELU(),
                nn.Conv3d(hidden_channels, num_classes, 1)
            )

    def forward(self, x, coarse_pred=None):
        B, C, X, Y, Z = x.shape
        if coarse_pred is not None:
            occupancy_mask = (coarse_pred.argmax(dim=1) != 0)
        else:
            occupancy_mask = x.abs().sum(dim=1) > self.sparsity_threshold
        sparsity = 1.0 - occupancy_mask.float().mean().item()
        if self.use_sparse and sparsity > 0.5:
            feats, coords, spatial_info = self.dense_to_sparse(x, occupancy_mask)
            if feats.shape[0] == 0:
                return torch.zeros(B, self.sparse_head.conv2.out_channels if hasattr(self.sparse_head, 'conv2') else 18, X, Y, Z, device=x.device, dtype=x.dtype)
            sparse_tensor = spconv.SparseConvTensor(feats, coords.int(), (X, Y, Z), B)
            sparse_out = self.sparse_head(sparse_tensor)
            dense_out = sparse_out.dense()
            return dense_out
        else:
            if hasattr(self, 'dense_head'):
                return self.dense_head(x)
            else:
                return self.sparse_head(x)

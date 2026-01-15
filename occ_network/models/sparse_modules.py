import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.default import config

# Backend Availability Flags
SPCONV_AVAILABLE = False
TORCHSPARSE_AVAILABLE = False

try:
    import spconv.pytorch as spconv
    SPCONV_AVAILABLE = True
except ImportError:
    pass

try:
    import torchsparse
    from torchsparse import SparseTensor
    import torchsparse.nn as tsnn
    TORCHSPARSE_AVAILABLE = True
except ImportError:
    pass

def get_backend():
    """Determine which sparse backend to use based on config and availability."""
    cfg_backend = getattr(config, 'sparse_backend', 'auto')
    
    if cfg_backend == 'spconv':
        return 'spconv' if SPCONV_AVAILABLE else 'dense'
    elif cfg_backend == 'torchsparse':
        return 'torchsparse' if TORCHSPARSE_AVAILABLE else 'dense'
    elif cfg_backend == 'dense':
        return 'dense'
    else: # auto
        if SPCONV_AVAILABLE:
            return 'spconv'
        elif TORCHSPARSE_AVAILABLE:
            return 'torchsparse'
        else:
            return 'dense'

class DenseToSparse(nn.Module):
    def __init__(self, threshold=0.1, backend='auto'):
        super().__init__()
        self.threshold = threshold
        self.backend = backend if backend != 'auto' else get_backend()

    def forward(self, dense_features, occupancy_mask=None):
        B, C, X, Y, Z = dense_features.shape
        if occupancy_mask is None:
            occupancy_mask = dense_features.abs().sum(dim=1) > self.threshold
            
        coords_list = []
        feats_list = []
        
        for b in range(B):
            mask_b = occupancy_mask[b]
            # coords: (N, 3) -> (x, y, z)
            coords = torch.nonzero(mask_b, as_tuple=False)
            
            # Add batch index
            # (N, 1)
            batch_idx = torch.full((coords.shape[0], 1), b, dtype=torch.int32, device=coords.device)
            
            # (N, 4) -> (b, x, y, z)
            coords_b = torch.cat([batch_idx, coords], dim=1)
            
            feats = dense_features[b, :, mask_b].t() # (N, C)
            
            coords_list.append(coords_b)
            feats_list.append(feats)
            
        if len(coords_list) > 0:
            coords = torch.cat(coords_list, dim=0)
            feats = torch.cat(feats_list, dim=0)
        else:
            coords = torch.zeros((0, 4), dtype=torch.int32, device=dense_features.device)
            feats = torch.zeros((0, C), dtype=dense_features.dtype, device=dense_features.device)
            
        # Backend specific tensor construction
        if self.backend == 'torchsparse' and TORCHSPARSE_AVAILABLE:
            # torchsparse expects int coordinates
            return SparseTensor(feats, coords.int()), coords, (B, X, Y, Z)
        elif self.backend == 'spconv' and SPCONV_AVAILABLE:
            # spconv expects indices (batch, z, y, x) usually, but here we used (b, x, y, z)
            # We will use spconv.SparseConvTensor later which takes (b, x, y, z) if spatial_shape matches
            return feats, coords.int(), (B, X, Y, Z)
        else:
            return feats, coords, (B, X, Y, Z)

class SparseToDense(nn.Module):
    def __init__(self, backend='auto'):
        super().__init__()
        self.backend = backend if backend != 'auto' else get_backend()

    def forward(self, sparse_tensor, spatial_shape, batch_size):
        B, X, Y, Z = batch_size, *spatial_shape
        
        if self.backend == 'torchsparse' and TORCHSPARSE_AVAILABLE:
            if isinstance(sparse_tensor, SparseTensor):
                feats = sparse_tensor.feats
                coords = sparse_tensor.coords
            else:
                feats, coords = sparse_tensor
        elif self.backend == 'spconv' and SPCONV_AVAILABLE:
            if hasattr(sparse_tensor, 'features'):
                feats = sparse_tensor.features
                coords = sparse_tensor.indices
            else:
                feats, coords = sparse_tensor
        else:
            feats, coords = sparse_tensor

        C = feats.shape[1]
        dense = torch.zeros(B, C, X, Y, Z, dtype=feats.dtype, device=feats.device)
        
        if coords.shape[0] > 0:
            # Assumes coords are (b, x, y, z)
            b_idx = coords[:, 0].long()
            x_idx = coords[:, 1].long()
            y_idx = coords[:, 2].long()
            z_idx = coords[:, 3].long()
            
            # Safe indexing
            mask = (b_idx < B) & (x_idx < X) & (y_idx < Y) & (z_idx < Z)
            if mask.sum() < coords.shape[0]:
                b_idx = b_idx[mask]
                x_idx = x_idx[mask]
                y_idx = y_idx[mask]
                z_idx = z_idx[mask]
                feats = feats[mask]
                
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
        self.backend = get_backend()
        
        if self.backend == 'spconv' and SPCONV_AVAILABLE:
            self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
            self.bn = nn.BatchNorm1d(out_channels)
        elif self.backend == 'torchsparse' and TORCHSPARSE_AVAILABLE:
            self.conv = tsnn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=False)
            self.bn = tsnn.BatchNorm(out_channels)
        else:
            self.conv = PseudoSparseConv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            self.bn = nn.BatchNorm3d(out_channels)
            self.backend = 'dense' # Fallback
            
        self.act = nn.GELU()

    def forward(self, x):
        if self.backend == 'spconv':
            x = self.conv(x)
            x = x.replace_feature(self.act(self.bn(x.features)))
            return x
        elif self.backend == 'torchsparse':
            x = self.conv(x)
            x = self.bn(x)
            # torchsparse 1.x/2.x differentiation might be needed for activation
            # assuming v2+ where we can apply act to feats or tensor supports it
            # Safest is to modify features
            if hasattr(x, 'feats'):
                x.feats = self.act(x.feats)
            return x
        else:
            return self.act(self.bn(self.conv(x)))

class SparseSemanticHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels=64, use_sparse=True):
        super().__init__()
        self.backend = get_backend() if use_sparse else 'dense'
        
        if self.backend == 'spconv' and SPCONV_AVAILABLE:
            self.conv1 = spconv.SubMConv3d(in_channels, hidden_channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.conv2 = spconv.SubMConv3d(hidden_channels, num_classes, 1, bias=True)
        elif self.backend == 'torchsparse' and TORCHSPARSE_AVAILABLE:
            self.conv1 = tsnn.Conv3d(in_channels, hidden_channels, kernel_size=3, stride=1, bias=False)
            self.bn1 = tsnn.BatchNorm(hidden_channels)
            self.conv2 = tsnn.Conv3d(hidden_channels, num_classes, kernel_size=1, stride=1, bias=True)
        else:
            self.conv1 = nn.Conv3d(in_channels, hidden_channels, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm3d(hidden_channels)
            self.conv2 = nn.Conv3d(hidden_channels, num_classes, 1)
            self.backend = 'dense'
            
        self.act = nn.GELU()

    def forward(self, x):
        if self.backend == 'spconv':
            x = self.conv1(x)
            x = x.replace_feature(self.act(self.bn1(x.features)))
            x = self.conv2(x)
            return x
        elif self.backend == 'torchsparse':
            x = self.conv1(x)
            x = self.bn1(x)
            if hasattr(x, 'feats'):
                x.feats = self.act(x.feats)
            x = self.conv2(x)
            return x
        else:
            x = self.act(self.bn1(self.conv1(x)))
            return self.conv2(x)

class AdaptiveSparseProcessor(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels=64, sparsity_threshold=0.1):
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        self.backend = get_backend()
        
        self.dense_to_sparse = DenseToSparse(threshold=sparsity_threshold, backend=self.backend)
        self.sparse_to_dense = SparseToDense(backend=self.backend)
        
        self.use_sparse = (self.backend in ['spconv', 'torchsparse'])
        
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
            # feats: (N, C), coords: (N, 4)
            # output depends on backend
            res = self.dense_to_sparse(x, occupancy_mask)
            
            if self.backend == 'torchsparse':
                sparse_tensor, coords, _ = res
                if sparse_tensor.feats.shape[0] == 0:
                     return torch.zeros(B, 18, X, Y, Z, device=x.device, dtype=x.dtype)
                
                sparse_out = self.sparse_head(sparse_tensor)
                dense_out = self.sparse_to_dense(sparse_out, (X, Y, Z), B)
                return dense_out
                
            elif self.backend == 'spconv':
                feats, coords, _ = res
                if feats.shape[0] == 0:
                    return torch.zeros(B, 18, X, Y, Z, device=x.device, dtype=x.dtype)
                    
                sparse_tensor = spconv.SparseConvTensor(feats, coords.int(), (X, Y, Z), B)
                sparse_out = self.sparse_head(sparse_tensor)
                dense_out = sparse_out.dense()
                return dense_out
        
        # Fallback to dense
        if hasattr(self, 'dense_head'):
            return self.dense_head(x)
        else:
            # Should not happen if init logic is correct, but safe fallback
            return self.sparse_head(x) # Will fail if sparse_head is actually sparse

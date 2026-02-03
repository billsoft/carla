import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        
        # Classification First Approach (Low Memory)
        # Input: [B, dim, 80, 80, 16]
        
        # 1. Reduce channels first
        self.conv1 = nn.Sequential(
            nn.Conv3d(dim, dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.GELU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(dim // 4),
            nn.GELU(),
        )
        
        # 2. Classify at low resolution
        # [B, 64, 80, 80, 16] -> [B, 18, 80, 80, 16]
        self.cls_head = nn.Conv3d(dim // 4, config.num_classes, kernel_size=1)
        
        # 3. Upsampling parameters (Learnable upsampling for logits)
        # We use ConvTranspose3d but on 18 channels only!
        
        # Step 1: 80x80x16 -> 160x160x32 (Stride 2)
        # Wait, target is 400x400x32.
        # 80 -> 400 is 5x.
        # 16 -> 32 is 2x.
        # Stride (5, 5, 2)? ConvTranspose with stride 5 is aggressive but possible.
        # Or interpolate + refine?
        # The user plan says: "F.interpolate (trilinear, 无参数) ... 上采样不存梯度!"
        # But also: "VoxelHead | 改为低分辨率卷积 + 分类 + 三线性上采样"
        # However, simply interpolating 18 channels might be too coarse?
        # Let's look at the user's specific instruction table:
        # "Step 2: Upsample 1 | ConvTranspose3d(18->18) | [B, 18, 200, 200, 32]"
        # "Step 3: Upsample 2 | ConvTranspose3d(18->18) | [B, 18, 400, 400, 32]"
        
        # So we need intermediate steps.
        # 80 -> 200 -> 400?
        # 80 -> 200 is 2.5x.
        # 200 -> 400 is 2x.
        # ConvTranspose3d supports integer strides easily. 2.5x is hard.
        # Maybe config.fine_size=(80,80,16) and voxel_size=(400,400,32) implies we need to be careful.
        # If we use trilinear interpolation, we can do any size.
        # The user plan says: "Step 2: Upsample 1 ... ConvTranspose3d(18->18) ... [B, 18, 200, 200, 32]"
        # This implies we *should* go through 200.
        # 80 -> 200?
        # Maybe the user meant 100x100 fine size?
        # But user explicitly changed fine_size to 80x80.
        # If fine_size is 80, and we want 400. That's exactly 5x.
        # If we want 200 intermediate, that's 2.5x.
        
        # Let's use F.interpolate for the resizing, and then a Conv3d to refine.
        # Or use ConvTranspose3d with specific kernel/stride/padding to achieve 2.5x? No, that's messy.
        
        # User plan explicitly listed:
        # "Step 2: Upsample 1 | ConvTranspose3d(18->18) | [B, 18, 200, 200, 32]"
        # If input is 80, how do we get 200 with ConvTranspose?
        # Maybe the user assumed 100x100 fine size in that table row?
        # "fine_size | 100x100x16 -> 80x80x16"
        # If 80x80, then 5x upsampling is needed.
        
        # I will implement a robust upsampling strategy:
        # 1. Logits at 80x80x16.
        # 2. Interpolate to 200x200x32 -> Conv3d(18->18)
        # 3. Interpolate to 400x400x32 -> Conv3d(18->18)
        # This is safer and cleaner than weird strided transpose convs.
        # Also "上采样不存梯度" suggests interpolate.
        
        # Wait, look at the "VoxelHead 修改" block in user input:
        # "x [18, 80, 80, 16] -> F.interpolate (trilinear) -> out [18, 400, 400, 32]"
        # It says "F.interpolate (trilinear, 无参数)".
        # This is the simplest and most memory efficient way.
        
        # But the table below it says:
        # "Step 2: Upsample 1 | ConvTranspose3d(18->18)"
        # These are contradictory.
        # Given "OOM" is the main concern, Interpolate is better.
        # But ConvTranspose3d(18->18) is very cheap (18*18*3*3*3 parameters).
        # And activations for 18 channels are small (18 * 400 * 400 * 32 * 4 bytes = 370MB).
        # So using ConvTranspose or Conv after interpolate is fine.
        
        # I'll stick to the "Classification First" and then upsample.
        # I will use Interpolate -> Refine Block.
        
        self.refine1 = nn.Sequential(
            nn.Conv3d(config.num_classes, config.num_classes, kernel_size=3, padding=1),
            nn.BatchNorm3d(config.num_classes),
            nn.ReLU(inplace=True)
        )
        self.refine2 = nn.Sequential(
            nn.Conv3d(config.num_classes, config.num_classes, kernel_size=3, padding=1),
             # No BN/Act on final output? Usually logits shouldn't be activated?
             # But if this is intermediate...
             # The final output should be logits.
        )
        
    def forward(self, x):
        # x: [B, fx, fy, fz, C] (from OccDecoder)
        # Permute to [B, C, fx, fy, fz] for Conv3d/Interpolate
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        
        # 1. Down-channel
        x = self.conv1(x) # -> 128
        x = self.conv2(x) # -> 64
        
        # 2. Classify
        logits_small = self.cls_head(x) # -> [B, 18, 80, 80, 16]
        
        # 3. Upsample to Target
        # Target: 400x400x32
        
        # We can do it in one shot or two steps.
        # Two steps allows for some refinement.
        
        # Step 1: Upsample to 200x200x32
        logits_mid = F.interpolate(logits_small, size=(200, 200, 32), mode='trilinear', align_corners=False)
        logits_mid = self.refine1(logits_mid) + logits_mid # Residual connection?
        
        # Step 2: Upsample to 400x400x32
        logits_final = F.interpolate(logits_mid, size=(400, 400, 32), mode='trilinear', align_corners=False)
        # Optional final refinement?
        # logits_final = self.refine2(logits_final) 
        
        # Actually, if we just interpolate, we don't learn much spatial refinement.
        # But at 18 channels, it's cheap.
        
        return logits_final


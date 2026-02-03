import torch
import torch.nn as nn
import sys
import os
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import E2EOccConfig
from e2e_occ_net import E2EOccNet

def print_mem(tag):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem = torch.cuda.memory_allocated() / 1024**3
        max_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[{tag}] Mem: {mem:.2f} GB (Max: {max_mem:.2f} GB)")

def verify_layers():
    print("--- Layer-wise Memory Verification (After Fix) ---")
    config = E2EOccConfig()
    # Explicitly set the new safe values
    config.embed_dim = 256
    config.coarse_size = (25, 25, 8) # 5000 queries
    config.fine_size = (80, 80, 16)
    config.decoder_layers = 2
    
    print(f"Config: Dim={config.embed_dim}, Coarse={config.coarse_size}, Fine={config.fine_size}")
    
    model = E2EOccNet(config).cuda()
    model.train() 
    
    print_mem("Model Loaded")
    
    # Inputs
    B = 1
    N = 8
    imgs = torch.randn(B, N, 1, 960, 1280).cuda()
    intrinsics = torch.eye(3).cuda().unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
    extrinsics = torch.eye(4).cuda().unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
    
    print_mem("Inputs Created")
    
    # 1. Patch Embed
    feats = model.patch_embed(imgs)
    print_mem("After Patch Embed")
    
    # 2. Encoder
    feats = model.encoder(feats, intrinsics, extrinsics)
    print_mem("After Encoder")
    
    # 3. Decoder Steps
    decoder = model.decoder
    image_feats = feats
    
    cx, cy, cz = decoder.config.coarse_size
    coarse_pos = decoder.pos_3d(cx, cy, cz, feats.device)
    query = decoder.coarse_query.expand(B, -1, -1) + coarse_pos.unsqueeze(0)
    ref = decoder.coarse_ref.unsqueeze(0).expand(B, -1, -1)
    
    print_mem("Before Coarse Loop")
    
    for i, layer in enumerate(decoder.coarse_layers):
        query = layer(query, ref, image_feats, intrinsics, extrinsics)
        print_mem(f"After Coarse Layer {i}")
        
    coarse_feats = query.view(B, cx, cy, cz, -1).permute(0, 4, 1, 2, 3)
    
    # Fine Setup
    fx, fy, fz = decoder.config.fine_size
    fine_feats = torch.nn.functional.interpolate(coarse_feats, size=(fx, fy, fz), mode='trilinear', align_corners=False)
    fine_feats = fine_feats.permute(0, 2, 3, 4, 1).reshape(B, -1, decoder.config.embed_dim)
    fine_feats = decoder.coarse_to_fine(fine_feats)
    
    fine_pos = decoder.pos_3d(fx, fy, fz, feats.device)
    query = fine_feats + fine_pos.unsqueeze(0)
    ref = decoder.fine_ref.unsqueeze(0).expand(B, -1, -1)
    
    print_mem("Before Fine Loop")
    
    for i, layer in enumerate(decoder.fine_layers):
        query = layer(query, ref, image_feats, intrinsics, extrinsics)
        print_mem(f"After Fine Layer {i}")
        
    output = query.view(B, fx, fy, fz, -1).permute(0, 4, 1, 2, 3)
    
    # Head
    logits = model.head(output)
    print_mem("After Head")
    
    # Backward
    loss = logits.sum()
    loss.backward()
    print_mem("After Backward")
    
    if torch.cuda.max_memory_allocated() / 1024**3 < 20.0:
        print("\nSUCCESS: Memory is within 20GB limit!")
    else:
        print("\nWARNING: Memory still high.")

if __name__ == "__main__":
    try:
        verify_layers()
    except Exception as e:
        print(f"\nCRASHED: {e}")


import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from models.transformer_occ.transformer_occ_nano_net import TransformerOccNetNano

def verify_nano_net():
    print("=" * 60)
    print("TransformerOccNetNano 全面验证")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Model Initialization
    print("\n[1] 初始化模型...")
    model = TransformerOccNetNano(
        num_cameras=8,
        img_size=(960, 1280),
        embed_dim=128,
        bev_size=(25, 25),
        num_height_levels=4,
        output_grid_size=(200, 200, 16)
    ).to(device)
    
    params = model.get_params_summary()
    print(f"  Total Params: {params['total']:.2f}M")
    
    # 2. Inference Memory Test (eval mode)
    print("\n[2] 推理显存测试 (eval mode, no_grad)...")
    model.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated()
        
    inputs = torch.randn(1, 8, 1, 960, 1280, device=device)
    
    with torch.no_grad():
        start_time = time.time()
        out = model(inputs)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
    print(f"  Input: {inputs.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Time: {(end_time - start_time)*1000:.2f} ms")
    
    if device.type == 'cuda':
        peak_mem = (torch.cuda.max_memory_allocated() - start_mem) / 1024**2
        print(f"  Peak Memory Increase: {peak_mem:.2f} MB")
        
    # 3. Training Memory Test (train mode, with grad)
    print("\n[3] 训练显存测试 (train mode, with grad, optimizer)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base_mem = torch.cuda.memory_allocated()
        
    inputs = torch.randn(1, 8, 1, 960, 1280, device=device)
    labels = torch.randint(0, 18, (1, 200, 200, 16), device=device)
    criterion = nn.CrossEntropyLoss()
    
    try:
        optimizer.zero_grad()
        out = model(inputs)
        
        # Reshape output for loss: [B, C, X, Y, Z] -> [B, C, X*Y*Z]
        # But CrossEntropyLoss expects [B, C, d1, d2, ...]
        # Output is [B, C, X, Y, Z] = [1, 18, 200, 200, 16]
        
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
        print(f"  Loss: {loss.item():.4f}")
        
        if device.type == 'cuda':
            peak_mem = (torch.cuda.max_memory_allocated() - base_mem) / 1024**2
            print(f"  Peak Memory Increase (Training): {peak_mem:.2f} MB")
            
            # Check if < 1.2GB (1228 MB)
            if peak_mem < 1228:
                print("  ✅ 显存占用符合要求 (< 1.2GB)")
            else:
                print(f"  ⚠️ 显存占用较高 ({peak_mem:.2f} MB)")
                
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("  ❌ OOM (Out of Memory)")
        else:
            print(f"  ❌ Runtime Error: {e}")
            
    print("\n验证完成。")

if __name__ == "__main__":
    verify_nano_net()


import torch
import sys
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from models.transformer_occ.transformer_occ_nano_net import TransformerOccNetNano
from models.transformer_occ.transformer_occ_net import TransformerOccNetLite

def verify_fixes():
    print("=" * 60)
    print("验证修复 (Sequential Processing & Decoder Shape)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. 验证 TransformerOccNetNano (Sequential Processing + Checkpointing)
    print("\n[1] 验证 TransformerOccNetNano (Sequential + Checkpoint)")
    model_nano = TransformerOccNetNano(
        num_cameras=8,
        img_size=(960, 1280),
        embed_dim=128,
        bev_size=(25, 25),
        output_grid_size=(200, 200, 16),
        use_checkpoint=True # 开启 Checkpointing
    ).to(device)
    
    inputs = torch.randn(1, 8, 1, 960, 1280, device=device)
    
    # 显存测试 (Train mode with grad)
    model_nano.train()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
    try:
        out_nano = model_nano(inputs)
        print(f"  Nano Output: {out_nano.shape}")
        
        loss = out_nano.sum()
        loss.backward()
        
        if device.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  Nano Peak Memory (Train): {peak_mem:.2f} MB")
            
    except Exception as e:
        print(f"  ❌ Nano Failed: {e}")

    # 2. 验证 TransformerOccNetLite (Decoder Shape Fix)
    print("\n[2] 验证 TransformerOccNetLite (Decoder Shape Fix)")
    model_lite = TransformerOccNetLite(
        num_cameras=8,
        img_size=(960, 1280),
        output_grid_size=(200, 200, 16),
        bev_size=(100, 100),
        num_height_levels=16
    ).to(device)
    
    try:
        with torch.no_grad():
            out_lite = model_lite(inputs)
        print(f"  Lite Output: {out_lite.shape}")
        
        expected_shape = (1, 18, 200, 200, 16)
        if out_lite.shape == expected_shape:
            print("  ✅ Lite Shape Correct!")
        else:
            print(f"  ❌ Lite Shape Incorrect! Expected {expected_shape}")
            
    except Exception as e:
        print(f"  ❌ Lite Failed: {e}")
        
    print("\n验证完成。")

if __name__ == "__main__":
    verify_fixes()

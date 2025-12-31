
import torch
import torch.nn as nn
import time
from models.transformer_occ import TransformerOccNetBalanced

def verify_balanced_net():
    print("=" * 60)
    print("Verifying TransformerOccNetBalanced")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Instantiate Model
    print("\n[1] Instantiating Model (Balanced-Pro Optimized)...")
    model = TransformerOccNetBalanced(
        num_cameras=8,
        img_size=(960, 1280),
        patch_size=16,
        embed_dim=256,
        encoder_layers=5,      # 4 -> 5
        decoder_layers=4,      # 3 -> 4
        num_heads=8,
        bev_size=(75, 75),     # 保持 75x75
        num_height_levels=10,  # 8 -> 10
        num_deform_points=6,   # 4 -> 6
        output_grid_size=(200, 200, 16),
        num_classes=18,
        dropout=0.1,
        use_checkpoint=True
    ).to(device)
    
    # 2. Check Parameters
    print("\n[2] Model Parameters:")
    summary = model.get_params_summary()
    for k, v in summary.items():
        print(f"  {k:<20}: {v:.2f}M")
        
    # 3. Forward Pass Test
    print("\n[3] Forward Pass Test:")
    B = 1
    dummy_input = torch.randn(B, 8, 1, 960, 1280).to(device)
    print(f"  Input Shape: {dummy_input.shape}")
    
    # Memory before forward
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**3
    
    start_time = time.time()
    with torch.amp.autocast('cuda'):
        output = model(dummy_input)
    end_time = time.time()
    
    print(f"  Output Shape: {output.shape}")
    print(f"  Time: {end_time - start_time:.4f}s")
    
    if torch.cuda.is_available():
        mem_after = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak Memory (BS={B}): {mem_after:.2f} GB")
        
    # 4. Batch Size 2 Test (Training Simulation)
    if torch.cuda.is_available():
        print("\n[4] Batch Size 2 Test (Training Simulation with Checkpointing):")
        try:
            B = 2
            # 模拟训练输入 (requires_grad=True 激活 Checkpointing)
            dummy_input = torch.randn(B, 8, 1, 960, 1280, device=device, requires_grad=True)
            model.train() # 确保是训练模式
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            with torch.amp.autocast('cuda'):
                output = model(dummy_input)
                # 模拟 Loss 计算和 Backward (才会释放中间变量)
                loss = output.sum()
                loss.backward()
                
            end_time = time.time()
            
            print(f"  Batch Size 2 Success!")
            print(f"  Time: {end_time - start_time:.4f}s")
            mem_after = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Peak Memory (Training BS={B}): {mem_after:.2f} GB (Should be ~4-5GB)")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("  Batch Size 2 OOM!")
            else:
                print(f"  Error: {e}")

    # 5. Batch Size 2 Inference Test
    if torch.cuda.is_available():
        print("\n[5] Batch Size 2 Inference Test (no_grad):")
        try:
            B = 2
            dummy_input = torch.randn(B, 8, 1, 960, 1280, device=device)
            model.eval() # 推理模式
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            with torch.no_grad(): # 不构建计算图
                with torch.amp.autocast('cuda'):
                    output = model(dummy_input)
            end_time = time.time()
            
            mem_after = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Peak Memory (Inference BS={B}): {mem_after:.2f} GB")
            
        except RuntimeError as e:
            print(f"  Error: {e}")

if __name__ == '__main__':
    verify_balanced_net()

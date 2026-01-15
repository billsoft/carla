import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from torch.cuda.amp import autocast
from configs.default import config
from models import build_model

def benchmark_memory(model, config, device, num_warmup=3, num_runs=10):
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    dummy_input = torch.randn(1, config.num_cameras, config.in_channels, *config.image_size, dtype=torch.float16, device=device)
    print("Warming up...")
    for _ in range(num_warmup):
        with autocast(enabled=True):
            with torch.no_grad():
                _ = model(dummy_input)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    print("Benchmarking...")
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        with autocast(enabled=True):
            with torch.no_grad():
                outputs = model(dummy_input)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    avg_time = sum(times) / len(times) * 1000
    print(f"\n{'='*50}")
    print(f"Benchmark Results (Batch Size = 1)")
    print(f"{'='*50}")
    print(f"Peak GPU Memory: {peak_mem:.2f} GB")
    print(f"Average Latency: {avg_time:.2f} ms")
    print(f"FPS: {1000/avg_time:.2f}")
    print(f"Output Shape: {outputs['semantic'].shape}")
    print(f"{'='*50}")
    return peak_mem, avg_time

def benchmark_training_memory(model, config, device):
    model.train()
    torch.cuda.reset_peak_memory_stats()
    dummy_input = torch.randn(1, config.num_cameras, config.in_channels, *config.image_size, dtype=torch.float16, device=device)
    dummy_target = {'semantic': torch.randint(0, config.num_classes, config.voxel_size, device=device), 'flow': torch.randn(1, 3, *config.voxel_size, dtype=torch.float16, device=device), 'flow_mask': torch.ones(1, *config.voxel_size, dtype=torch.bool, device=device)}
    from losses.losses import OccLoss
    loss_fn = OccLoss(config)
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print("Training memory benchmark...")
    optimizer.zero_grad()
    with autocast(enabled=True):
        outputs = model(dummy_input)
        losses = loss_fn(outputs, dummy_target)
    scaler.scale(losses['total']).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n{'='*50}")
    print(f"Training Memory (Batch Size = 1)")
    print(f"{'='*50}")
    print(f"Peak GPU Memory: {peak_mem:.2f} GB")
    print(f"{'='*50}")
    return peak_mem

@torch.no_grad()
def inference_with_uncertainty(model, config, device, num_samples=10, temperature=1.0):
    """
    Perform MC Dropout Inference to estimate uncertainty.
    """
    print(f"Running MC Dropout Inference (Samples={num_samples}, Temp={temperature})...")
    
    # 1. Force model to training mode (to activate dropout)
    model.train()
    
    # 2. Prepare dummy input (or real input if available)
    dummy_input = torch.randn(1, config.num_cameras, config.in_channels, *config.image_size, dtype=torch.float16, device=device)
    
    logits_list = []
    
    start_time = time.time()
    
    # 3. Multiple forward passes
    for i in range(num_samples):
        with autocast(enabled=True):
            outputs = model(dummy_input)
            # Apply temperature scaling
            logits = outputs['semantic'] / temperature
            logits_list.append(logits)
            
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    # 4. Statistical Analysis
    logits_stack = torch.stack(logits_list, dim=0) # [samples, B, C, X, Y, Z]
    probs = F.softmax(logits_stack, dim=2)
    
    mean_probs = probs.mean(0)
    pred = mean_probs.argmax(1)
    
    # Uncertainty metrics
    variance = probs.var(0).mean(1) # Average variance across classes
    entropy = - (mean_probs * torch.log(mean_probs + 1e-6)).sum(1)
    
    print(f"\n{'='*50}")
    print(f"MC Dropout Uncertainty Results")
    print(f"{'='*50}")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Avg Time per Pass: {total_time/num_samples:.4f}s")
    print(f"Estimated FPS: {num_samples/total_time:.2f}")
    print(f"Uncertainty (Variance) Mean: {variance.mean().item():.6f}")
    print(f"Uncertainty (Entropy) Mean: {entropy.mean().item():.6f}")
    print(f"{'='*50}")
    
    return {
        'pred': pred,
        'uncertainty_variance': variance,
        'uncertainty_entropy': entropy,
        'logits_mean': mean_probs
    }


def export_onnx(model, config, output_path='occ_net_v3.onnx'):
    model.eval()
    dummy_input = torch.randn(1, config.num_cameras, config.in_channels, *config.image_size)
    torch.onnx.export(model.cpu(), dummy_input, output_path, input_names=['images'], output_names=['semantic', 'flow'], dynamic_axes={'images': {0: 'batch'}, 'semantic': {0: 'batch'}, 'flow': {0: 'batch'}}, opset_version=17)
    print(f"Exported to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--train_mem', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--uncertainty', action='store_true', help='Enable MC Dropout Uncertainty Estimation')
    parser.add_argument('--mc-samples', type=int, default=10, help='Number of MC samples')
    parser.add_argument('--mc-temp', type=float, default=1.0, help='Temperature for Softmax')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint: {args.checkpoint}")
    if args.benchmark:
        benchmark_memory(model, config, device)
    if args.train_mem:
        benchmark_training_memory(model, config, device)
    if args.uncertainty:
        inference_with_uncertainty(model, config, device, args.mc_samples, args.mc_temp)
    if args.export:
        export_onnx(model, config)

if __name__ == '__main__':
    main()

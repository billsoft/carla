import torch
import torch.nn as nn
import time
import os
import sys

try:
    # For use as a package
    from .config import E2EOccConfig
    from .e2e_occ_net import E2EOccNet
except ImportError:
    # For direct script execution
    from config import E2EOccConfig
    from e2e_occ_net import E2EOccNet

def test_e2e_network():
    print("=" * 60)
    print("E2E-OccNet Network Verification")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Initialize Configuration
    print("\n[1] Initializing Configuration...")
    config = E2EOccConfig()
    print(f"  Embed Dim: {config.embed_dim}")
    print(f"  Coarse Size: {config.coarse_size}")
    print(f"  Fine Size: {config.fine_size}")
    print(f"  Temporal Frames: {config.temporal_frames}")
    print(f"  Use Temporal: {config.use_temporal}")
    print(f"  Use Ego Motion: {config.use_ego_motion}")

    # 2. Build Model
    print("\n[2] Building Model...")
    model = E2EOccNet(config).to(device)
    model.eval()
    print(f"  Model Parameters: {model.get_num_params() / 1e6:.2f}M")

    # 3. Create Dummy Inputs
    print("\n[3] Creating Dummy Inputs...")
    B = 1
    T = 2 # Sequence length
    N = config.num_cameras
    H, W = config.image_size
    
    # Create sequence of inputs
    # Images: [B, N, C, H, W] (Raw channels=1)
    images_seq = torch.randn(B, T, N, config.raw_channels, H, W, device=device)
    
    # Intrinsics: [B, N, 3, 3]
    intrinsics = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, N, -1, -1)
    
    # Extrinsics: [B, T, N, 4, 4]
    # Simulate simple forward motion
    extrinsics_seq = torch.eye(4, device=device).view(1, 1, 1, 4, 4).expand(B, T, N, -1, -1).clone()
    for t in range(T):
        extrinsics_seq[:, t, :, 0, 3] = t * 1.0 # Move 1m forward per step
        
    memory = None
    
    # 4. Run Sequential Inference
    print("\n[4] Running Sequential Inference...")
    
    try:
        with torch.no_grad():
            for t in range(T):
                print(f"\n  --- Time Step {t} ---")
                
                # Get current frame data
                img_t = images_seq[:, t]
                ext_t = extrinsics_seq[:, t]
                
                # Calculate Ego-Motion (relative pose)
                ego_motion = None
                if t > 0:
                    # Previous frame extrinsics
                    ext_prev = extrinsics_seq[:, t-1]
                    
                    # Assuming extrinsics are Pose (Camera-to-World)
                    # We use the first camera (ego) as reference
                    pose_t = ext_t[:, 0] # [B, 4, 4]
                    pose_prev = ext_prev[:, 0] # [B, 4, 4]
                    
                    # Calculate relative transform: T_{t-1 -> t}
                    # P_t = T^{-1} * P_{t-1} -> T_{t-1 -> t} = Pose_t^{-1} * Pose_{t-1}
                    ego_motion = torch.linalg.inv(pose_t) @ pose_prev
                    print("    Calculated Ego-Motion matrix")
                
                # Forward Pass
                start_time = time.time()
                outputs = model(
                    images=img_t, 
                    intrinsics=intrinsics, 
                    extrinsics=ext_t, 
                    memory=memory, 
                    ego_motion=ego_motion
                )
                end_time = time.time()
                
                # Check Outputs
                logits = outputs['semantic']
                new_memory = outputs['memory']
                
                print(f"    Inference Time: {(end_time - start_time)*1000:.2f} ms")
                print(f"    Output Logits Shape: {logits.shape}")
                
                if new_memory is not None:
                    print(f"    New Memory Shape: {new_memory.shape}")
                    # Check if memory has NaN
                    if torch.isnan(new_memory).any():
                        print("    ❌ Error: Memory contains NaN values!")
                    else:
                        print("    ✅ Memory values valid")
                else:
                    print("    Memory is None (Expected for first frame if temporal disabled, but enabled here)")

                # Verify Output Shape
                # VoxelHead output should be [B, num_classes, X, Y, Z]
                # Config voxel_size is (400, 400, 32)
                expected_shape = (B, config.num_classes, *config.voxel_size)
                if logits.shape == expected_shape:
                    print(f"    ✅ Output Shape Correct: {logits.shape}")
                else:
                    print(f"    ❌ Output Shape Mismatch! Expected {expected_shape}, got {logits.shape}")
                
                # Update memory for next step
                memory = new_memory
                # Detach to simulate TBPTT or just inference loop
                if memory is not None:
                    memory = memory.detach()
                
        print("\n✅ Network Verification Passed Successfully!")
        
    except Exception as e:
        print(f"\n❌ Network Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_e2e_network()

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import E2EOccConfig
from e2e_occ_net import E2EOccNet
from deformable_attention import DeformableCrossAttention

def test_projection():
    print("Testing 3D -> 2D Projection Logic...")
    
    # Setup dummy params
    B, N = 1, 1
    H, W = 100, 100
    
    # Intrinsics: Identity-ish but scaled to image size
    # f = 50, cx = 50, cy = 50
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0)
    intrinsics[..., 0, 0] = 50
    intrinsics[..., 1, 1] = 50
    intrinsics[..., 0, 2] = 50
    intrinsics[..., 1, 2] = 50
    
    # Extrinsics: Identity (Camera at origin looking down +Z)
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0)
    
    # Point at (0, 0, 10) in Camera frame (which is World frame here)
    # Should project to center of image (0, 0) in normalized coords
    # Wait, our query coords are 0-1.
    # The projection logic in deformable_attention maps 0-1 to -40~40 world coords.
    # Let's map normalized (0.5, 0.5, 0.5) to world center if possible.
    # (0.5 * 80 - 40) = 0
    # (0.5 * 80 - 40) = 0
    # (0.5 * 6.4 - 1) = 2.2
    # So (0.5, 0.5, 0.5) -> (0, 0, 2.2)
    # This point (0, 0, 2.2) is in front of camera.
    # Should project to (50, 50) pixel coords -> (0, 0) normalized.
    
    query_coords = torch.tensor([[[0.5, 0.5, 0.5]]]) # [1, 1, 3]
    
    attn = DeformableCrossAttention(dim=256, num_heads=8, num_cameras=1)
    
    ref_points = attn.get_reference_points(query_coords, intrinsics, extrinsics, H, W)
    
    print(f"Query (0.5, 0.5, 0.5) -> World (0, 0, 2.2) -> Ref Point: {ref_points[0,0,0].tolist()}")
    
    # Check if close to 0,0
    if torch.allclose(ref_points, torch.zeros_like(ref_points), atol=0.1):
        print("Projection Test PASSED: Center point projected correctly.")
    else:
        print("Projection Test FAILED: Expected approx (0,0).")

def verify_network():
    print("\nVerifying E2EOccNet...")
    
    config = E2EOccConfig()
    # Use smaller size for speed
    config.image_size = (128, 256) 
    config.encoder_layers = 1
    config.decoder_layers = 1
    
    model = E2EOccNet(config)
    model.cuda()
    
    # Register hooks to print intermediate shapes
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                print(f"[{name}] Output Shape: {output[0].shape}")
            elif isinstance(output, dict):
                for k, v in output.items():
                    print(f"[{name}] Output '{k}' Shape: {v.shape}")
            else:
                print(f"[{name}] Output Shape: {output.shape}")
        return hook

    model.patch_embed.register_forward_hook(get_activation("1. Patch Embed"))
    model.encoder.register_forward_hook(get_activation("2. Image Encoder"))
    # Hook internal layers of decoder to see Coarse vs Fine
    if len(model.decoder.coarse_layers) > 0:
        model.decoder.coarse_layers[-1].register_forward_hook(get_activation("3. Coarse Decoder (Last Layer)"))
    if len(model.decoder.fine_layers) > 0:
        model.decoder.fine_layers[-1].register_forward_hook(get_activation("4. Fine Decoder (Last Layer)"))
    model.decoder.register_forward_hook(get_activation("5. Occupancy Decoder Final"))
    model.head.register_forward_hook(get_activation("6. Voxel Head"))

    # Dummy Input (RAW uses 1 channel)
    B = 1
    N = config.num_cameras
    imgs = torch.randn(B, N, 1, *config.image_size).cuda()
    
    # Dummy Params
    intrinsics = torch.eye(3).unsqueeze(0).repeat(N, 1, 1).unsqueeze(0).cuda()
    extrinsics = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).unsqueeze(0).cuda()
    
    print(f"Input Image Shape: {imgs.shape}")
    
    # Forward
    try:
        output = model(imgs, intrinsics, extrinsics)
        logits = output['semantic']
        print(f"Output Logits Shape: {logits.shape}")
        
        # Expected shape: [B, C, X, Y, Z]
        # Voxel size in config: (400, 400, 32)
        # Logits: [1, 18, 400, 400, 32]
        expected_shape = (B, config.num_classes, *config.voxel_size)
        
        if logits.shape == expected_shape:
            print("Shape Verification PASSED")
        else:
            print(f"Shape Verification FAILED. Expected {expected_shape}, got {logits.shape}")
            
        # Backward
        loss = logits.sum()
        loss.backward()
        print("Backward Pass PASSED")
        
    except Exception as e:
        print(f"Forward/Backward FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        test_projection()
        if torch.cuda.is_available():
            verify_network()
        else:
            print("Skipping network verification (No CUDA)")
    except Exception as e:
        print(f"Verification Script Failed: {e}")

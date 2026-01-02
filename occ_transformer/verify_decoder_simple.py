
import sys
import os
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from occ_transformer.models.transformer_occ.decoder import VoxelDecoder, BalancedDecoder, SimplifiedDecoder

def test_decoder():
    print("=" * 60)
    print("Transformer Decoder Verification")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test VoxelDecoder (Standard/Base)
    # print("\n[1] VoxelDecoder (Standard):")
    # voxel_decoder = VoxelDecoder(
    #     embed_dim=256,
    #     num_layers=2, # Reduce layers for speed
    #     query_grid_size=(50, 50, 8),
    #     output_grid_size=(512, 512, 40),
    #     num_classes=18
    # ).to(device)
    
    # query = torch.randn(2, 50*50*8, 256, device=device)
    # memory = torch.randn(2, 1000, 256, device=device) # Mock memory
    
    # try:
    #     occ = voxel_decoder(query, memory)
    #     print(f"  Input Query: {query.shape}")
    #     print(f"  Output Occ:  {occ.shape}")
        
    #     expected_shape = (2, 18, 512, 512, 40)
    #     if occ.shape == expected_shape:
    #         print("  ✅ Shape Correct")
    #     else:
    #         print(f"  ❌ Shape Mismatch! Expected {expected_shape}")
    # except Exception as e:
    #     print(f"  ❌ Error: {e}")

    # Test BalancedDecoder
    print("\n[2] BalancedDecoder:")
    balanced_decoder = BalancedDecoder(
        embed_dim=256,
        num_layers=2,
        bev_size=(100, 100),
        num_height_levels=16,
        output_grid_size=(512, 512, 40),
        num_classes=18
    ).to(device)
    
    # Balanced takes BEV query [B, H*W, C]
    # Set batch size to 1 as requested
    query_balanced = torch.randn(1, 100*100, 256, device=device)
    memory = torch.randn(1, 1200, 256, device=device) # Mock memory (e.g. 30x40)
    spatial_shapes = torch.tensor([[30, 40]], device=device)
    ref_points_balanced = torch.rand(1, 100*100, 2, device=device) # 2D BEV points
    
    expected_shape = (1, 18, 512, 512, 40)
    
    try:
        occ_balanced = balanced_decoder(
            query_balanced, 
            memory, 
            reference_points=ref_points_balanced,
            memory_spatial_shapes=spatial_shapes
        )
        print(f"  Input Query: {query_balanced.shape}")
        print(f"  Output Occ:  {occ_balanced.shape}")
        
        if occ_balanced.shape == expected_shape:
            print("  ✅ Shape Correct")
        else:
            print(f"  ❌ Shape Mismatch! Expected {expected_shape}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test SimplifiedDecoder (Nano/Mini)
    print("\n[3] SimplifiedDecoder (Nano):")
    nano_decoder = SimplifiedDecoder(
        embed_dim=256,
        num_heads=8,
        query_grid_size=(50, 50, 8),
        output_grid_size=(512, 512, 40),
        num_classes=18
    ).to(device)
    
    query_nano = torch.randn(1, 50*50*8, 256, device=device)
    ref_points_nano = torch.rand(1, 50*50*8, 2, device=device) # DeformableAttn here seems to take 2D ref points usually (projected), need to check
    # But usually 3D query might need 3D ref points if 3D deformable attention.
    # Let's assume 2D for now as standard DeformableDETR uses 2D.
    # If SimplifiedDecoder uses 3D points, it might fail.
    # However, standard DeformableAttn is 2D.
    
    try:
        occ_nano = nano_decoder(
            query_nano, 
            memory,
            reference_points=ref_points_nano,
            memory_spatial_shapes=spatial_shapes
        )
        print(f"  Input Query: {query_nano.shape}")
        print(f"  Output Occ:  {occ_nano.shape}")
        
        if occ_nano.shape == expected_shape:
            print("  ✅ Shape Correct")
        else:
            print(f"  ❌ Shape Mismatch! Expected {expected_shape}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Test ProgressiveUpsample (Nano)
    print("\n[4] ProgressiveUpsample (Nano):")
    from occ_transformer.models.transformer_occ.transformer_occ_nano_net import ProgressiveUpsample
    
    upsample = ProgressiveUpsample(
        in_dim=128,
        num_classes=18,
        bev_start_size=50,
        target_size=(512, 512, 40),
        start_height=4
    ).to(device)
    
    # Input: [B, D, H, W, Z_start]
    # D=128, H=50, W=50, Z_start=4
    feat_nano = torch.randn(1, 128, 50, 50, 4, device=device)
    
    try:
        occ_upsampled = upsample(feat_nano)
        print(f"  Input Feat: {feat_nano.shape}")
        print(f"  Output Occ: {occ_upsampled.shape}")
        
        expected_shape = (1, 18, 512, 512, 40)
        if occ_upsampled.shape == expected_shape:
            print("  ✅ Shape Correct")
        else:
            print(f"  ❌ Shape Mismatch! Expected {expected_shape}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_decoder()

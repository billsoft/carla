import torch
import torch.nn as nn
import sys
import os

# Add the current directory to sys.path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.default import Config
from models.occ_net import OccNetV3

def get_shape_str(x):
    if isinstance(x, torch.Tensor):
        return str(list(x.shape))
    elif isinstance(x, (list, tuple)):
        return f"List/Tuple of len {len(x)}: " + ", ".join([get_shape_str(i) for i in x])
    elif x is None:
        return "None"
    else:
        return str(type(x))

def hook_fn(name):
    def hook(module, input, output):
        print(f"\n[Stage: {name}]")
        print(f"  Input:  {get_shape_str(input)}")
        if isinstance(output, tuple):
            print(f"  Output: Tuple of len {len(output)}")
            for i, o in enumerate(output):
                print(f"    Out[{i}]: {get_shape_str(o)}")
        elif isinstance(output, dict):
             print(f"  Output: Dict with keys {list(output.keys())}")
             for k, v in output.items():
                 print(f"    Out[{k}]: {get_shape_str(v)}")
        else:
            print(f"  Output: {get_shape_str(output)}")
    return hook

def verify_pipeline():
    print("Initializing OccNetV3 with default config...")
    config = Config()
    
    # Enable all new features for verification
    config.use_depth_aware_fusion = True
    config.use_memory_cell = True
    config.use_multi_scale_bev = True
    
    model = OccNetV3(config)
    model.eval()
    
    # Register hooks
    hooks = []
    hooks.append(model.patch_embed.register_forward_hook(hook_fn("Patch Embedding")))
    hooks.append(model.encoder.register_forward_hook(hook_fn("Encoder")))
    
    if hasattr(model, 'depth_fusion'):
        hooks.append(model.depth_fusion.register_forward_hook(hook_fn("Depth Fusion (LiftSplat)")))
    elif hasattr(model, 'fusion_proj'):
        hooks.append(model.fusion_proj.register_forward_hook(hook_fn("Simple Fusion")))
        
    hooks.append(model.decoder.register_forward_hook(hook_fn("BEV Decoder")))
    hooks.append(model.temporal.register_forward_hook(hook_fn("Temporal Fusion")))
    hooks.append(model.height_expand.register_forward_hook(hook_fn("Height Expand")))
    hooks.append(model.upsampler.register_forward_hook(hook_fn("Upsampler")))
    hooks.append(model.head.register_forward_hook(hook_fn("Head")))

    # Prepare Dummy Data
    B = 1
    N = config.num_cameras
    C = config.in_channels
    H, W = config.image_size
    
    print(f"\nRunning Forward Pass with Input: [B={B}, N={N}, C={C}, H={H}, W={W}]")
    
    images = torch.randn(B, N, C, H, W)
    
    # Dummy Intrinsics (Identity-like)
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
    # Scale K for image size (just to be somewhat realistic, though not strictly needed for shape check)
    intrinsics[..., 0, 0] = W / 2
    intrinsics[..., 1, 1] = H / 2
    intrinsics[..., 0, 2] = W / 2
    intrinsics[..., 1, 2] = H / 2
    
    # Dummy Extrinsics (Identity)
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
    
    # Ego Motion (Identity)
    ego_motion = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    
    # Ego Pose (Identity)
    ego_pose = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    
    timestamp = 0.0
    scene_id = "scene_001"
    
    with torch.no_grad():
        outputs = model(
            images=images,
            ego_motion=ego_motion,
            ego_pose=ego_pose,
            timestamp=timestamp,
            scene_id=scene_id,
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
        
    print("\nFinal Output Keys and Shapes:")
    for k, v in outputs.items():
        print(f"  {k}: {get_shape_str(v)}")
        
    # Remove hooks
    for h in hooks:
        h.remove()

if __name__ == "__main__":
    verify_pipeline()

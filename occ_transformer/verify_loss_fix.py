import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.loss import DistanceAwareLoss, get_default_class_weights

def verify_fix():
    print("Verifying DistanceAwareLoss fix...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. 实例化 (此时 self.dist_weights 应为 None)
    criterion = DistanceAwareLoss(class_weights=get_default_class_weights()).to(device)
    print("Instantiation successful.")
    
    if hasattr(criterion, 'dist_weights'):
        print(f"Initial dist_weights: {criterion.dist_weights}")
    else:
        print("Initial dist_weights attribute missing!")

    # 2. 构造虚拟数据
    B, C, H, W, Z = 2, 18, 50, 50, 8
    logits = torch.randn(B, C, H, W, Z, device=device)
    target = torch.randint(0, C, (B, H, W, Z), device=device)
    mask = torch.ones(B, H, W, Z, dtype=torch.bool, device=device)
    
    # 3. 第一次 Forward (应触发 _make_dist_weights)
    print("Running 1st forward pass...")
    try:
        loss = criterion(logits, target, mask)
        print(f"1st forward successful. Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ 1st forward FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 检查 dist_weights 是否已生成
    if criterion.dist_weights is not None:
        print(f"dist_weights generated. Shape: {criterion.dist_weights.shape}")
    else:
        print("❌ dist_weights is still None!")
        return

    # 5. 第二次 Forward (应使用缓存)
    print("Running 2nd forward pass...")
    try:
        loss = criterion(logits, target, mask)
        print(f"2nd forward successful. Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ 2nd forward FAILED: {e}")
        return

    print("\n✅ Verification PASSED!")

if __name__ == "__main__":
    verify_fix()

import torch
import time
import os
import sys
import traceback

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import E2EOccConfig
    from e2e_occ_net import E2EOccNet
    from voxel_head import VoxelHead
except (ImportError, ValueError):
    print("无法从相对路径导入，尝试作为包导入。")
    from e2e_occ.config import E2EOccConfig
    from e2e_occ.e2e_occ_net import E2EOccNet
    from e2e_occ.voxel_head import VoxelHead


def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_section(title):
    print(f"\n--- {title} ---")


def verify_voxel_head(config, device):
    """
    专项验证 VoxelHead 新结构：
    - 输入输出形状正确
    - 中间分辨率的通道数正确（64→32 in 200, 32→18 in 400）
    - 无 NaN/Inf
    - 参数量合理
    """
    print_section("VoxelHead 新结构专项验证")

    head = VoxelHead(config).to(device).eval()
    num_params = sum(p.numel() for p in head.parameters())
    print(f"  VoxelHead 参数量: {num_params / 1e6:.3f}M")

    # 检查关键子模块通道数
    refine1_in  = head.refine1[0].in_channels
    refine1_out = head.refine1[0].out_channels
    refine2_in  = head.refine2[0].in_channels
    refine2_out = head.refine2[0].out_channels
    skip1_in    = head.skip1.in_channels
    skip1_out   = head.skip1.out_channels
    skip2_in    = head.skip2.in_channels
    skip2_out   = head.skip2.out_channels

    dim = config.embed_dim
    nc  = config.num_classes

    checks = [
        (refine1_in,  dim // 4, "refine1 输入通道"),
        (refine1_out, dim // 8, "refine1 输出通道"),
        (refine2_in,  dim // 8, "refine2 输入通道"),
        (refine2_out, nc,       "refine2 输出通道（等于 num_classes）"),
        (skip1_in,    dim // 4, "skip1 输入通道"),
        (skip1_out,   dim // 8, "skip1 输出通道"),
        (skip2_in,    dim // 8, "skip2 输入通道"),
        (skip2_out,   nc,       "skip2 输出通道（等于 num_classes）"),
    ]
    all_ok = True
    for actual, expected, name in checks:
        ok = actual == expected
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name}: {actual}（期望 {expected}）")
        if not ok:
            all_ok = False

    # 前向传播形状验证
    fx, fy, fz = config.fine_size
    dummy = torch.randn(1, fx, fy, fz, dim, device=device)
    with torch.no_grad():
        out = head(dummy)

    expected_shape = (1, nc, *config.voxel_size)
    shape_ok = out.shape == expected_shape
    mark = "✅" if shape_ok else "❌"
    print(f"  {mark} 输出形状: {tuple(out.shape)}（期望 {expected_shape}）")
    if not shape_ok:
        all_ok = False

    nan_ok = not (torch.isnan(out).any() or torch.isinf(out).any())
    mark = "✅" if nan_ok else "❌"
    print(f"  {mark} 输出数值: {'无 NaN/Inf' if nan_ok else '含 NaN/Inf！'}")
    if not nan_ok:
        all_ok = False

    return all_ok


def run_verification():
    print_header("E2E-OccNet 网络验证（含新输出头）")

    try:
        # 1. 初始化
        print_section("1. 环境与配置")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"运行设备: {device}")

        config = E2EOccConfig()
        print(f"  embed_dim={config.embed_dim}, num_classes={config.num_classes}")
        print(f"  fine_size={config.fine_size}, voxel_size={config.voxel_size}")
        print(f"  时序融合: {config.use_temporal} (帧数: {config.temporal_frames})")

        # 2. VoxelHead 专项验证
        print_section("2. VoxelHead 新结构专项验证")
        head_ok = verify_voxel_head(config, device)

        # 3. 完整模型构建
        print_section("3. 完整模型构建")
        model = E2EOccNet(config).to(device).eval()
        num_params = model.get_num_params()
        print(f"  总参数量: {num_params / 1e6:.2f}M")

        # 4. 序列前向传播验证
        print_section("4. 序列前向传播（含时序记忆）")
        B = 1
        T = config.temporal_frames
        N = config.num_cameras
        H, W = config.image_size

        images_seq    = torch.randn(B, T, N, config.raw_channels, H, W, device=device)
        intrinsics    = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, N, -1, -1)
        extrinsics_seq = torch.eye(4, device=device).view(1, 1, 1, 4, 4).expand(B, T, N, -1, -1).clone()
        for t in range(1, T):
            extrinsics_seq[:, t, :, 0, 3] = float(t)  # X 方向每帧前进 1m

        memory = None
        seq_ok = True
        with torch.no_grad():
            for t in range(T):
                img_t = images_seq[:, t]
                ext_t = extrinsics_seq[:, t]

                ego_motion = None
                if config.use_ego_motion and t > 0:
                    pose_curr = ext_t[:, 0]
                    pose_prev = extrinsics_seq[:, t - 1, 0]
                    ego_motion = torch.linalg.inv(pose_curr) @ pose_prev

                t0 = time.time()
                outputs = model(images=img_t, intrinsics=intrinsics,
                                extrinsics=ext_t, memory=memory, ego_motion=ego_motion)
                ms = (time.time() - t0) * 1000

                logits = outputs.get('semantic')
                expected = (B, config.num_classes, *config.voxel_size)
                shape_ok = logits is not None and logits.shape == expected
                nan_ok   = logits is not None and not (torch.isnan(logits).any() or torch.isinf(logits).any())

                print(f"  t={t}: {ms:.0f}ms  "
                      f"{'✅' if shape_ok else '❌'} shape={tuple(logits.shape) if logits is not None else 'None'}  "
                      f"{'✅' if nan_ok else '❌ NaN/Inf'}")
                if not (shape_ok and nan_ok):
                    seq_ok = False

                if config.use_temporal:
                    memory = outputs.get('memory')
                    if memory is not None:
                        mem_ok = not (torch.isnan(memory).any() or torch.isinf(memory).any())
                        expected_mem = (B, config.num_coarse_queries, config.memory_dim)
                        mem_shape_ok = memory.shape == expected_mem
                        print(f"       memory shape={'✅' if mem_shape_ok else '❌'} {tuple(memory.shape)}  "
                              f"数值={'✅' if mem_ok else '❌ NaN/Inf'}")
                        if not (mem_ok and mem_shape_ok):
                            seq_ok = False
                        memory = memory.detach()

        # 5. 显存报告（仅 GPU）
        if device.type == 'cuda':
            print_section("5. 显存报告")
            peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            print(f"  推理峰值显存: {peak_gb:.2f} GB")

        # 汇总
        all_pass = head_ok and seq_ok
        if all_pass:
            print_header("✅ 全部验证通过")
        else:
            print_header("❌ 部分验证失败，请查看上方日志")

    except Exception as e:
        print_header("❌ 验证异常中止")
        print(f"错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    run_verification()

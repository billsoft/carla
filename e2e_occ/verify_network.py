import torch
import time
import os
import sys
import traceback

# 确保根目录在路径中，以便进行类似包的导入
# 这使得脚本可以从任何地方运行
try:
    # 假设脚本在 e2e_occ/ 目录中
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import E2EOccConfig
    from e2e_occ_net import E2EOccNet
except (ImportError, ValueError):
    # 针对不同执行上下文的回退
    print("无法从相对路径导入，尝试作为包导入。")
    from e2e_occ.config import E2EOccConfig
    from e2e_occ.e2e_occ_net import E2EOccNet


def print_header(title):
    """打印格式化的标题。"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_section(title):
    """打印格式化的段落标题。"""
    print(f"\n--- {title} ---")


def run_verification():
    """
    对 E2EOccNet 模型进行全面验证，
    重点关注序列前向传播、形状正确性和数值稳定性。
    """
    print_header("E2E-OccNet 网络验证")
    
    try:
        # --- 1. 设置环境和配置 ---
        print_section("1. 设置环境和配置")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"运行设备: {device}")

        config = E2EOccConfig()
        print("配置已加载:")
        print(f"  - 时序融合: {config.use_temporal} (帧数: {config.temporal_frames})")
        print(f"  - 自车运动对齐: {config.use_ego_motion}")
        print(f"  - 粗查询数: {config.num_coarse_queries}")
        print(f"  - 精细查询数: {config.num_fine_queries}")
        print(f"  - 最终体素网格: {config.voxel_size}")

        # --- 2. 构建模型 ---
        print_section("2. 构建模型")
        model = E2EOccNet(config).to(device).eval() # 设置为评估模式
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型构建成功，参数量: {num_params / 1e6:.2f}M")

        # --- 3. 为序列创建伪输入 ---
        print_section("3. 为序列创建伪输入")
        B = 1  # 批量大小
        T = config.temporal_frames # 使用配置中的序列长度
        N = config.num_cameras
        C, H, W = config.raw_channels, config.image_size[0], config.image_size[1]

        images_seq = torch.randn(B, T, N, C, H, W, device=device)
        intrinsics = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, N, -1, -1)
        
        # 模拟简单的前进自车运动作为外参
        extrinsics_seq = torch.eye(4, device=device).view(1, 1, 1, 4, 4).expand(B, T, N, -1, -1).clone()
        for t in range(1, T):
            # 每个时间步沿x轴前进1米
            extrinsics_seq[:, t, :, 0, 3] = t * 1.0
        
        print(f"为 {T} 帧的序列创建了伪数据。")
        print(f"  - 图像形状: {images_seq.shape}")
        print(f"  - 外参形状: {extrinsics_seq.shape}")

        # --- 4. 运行序列推理 ---
        print_section("4. 运行序列推理")
        
        memory = None
        with torch.no_grad():
            for t in range(T):
                print(f"\n处理时间步 {t+1}/{T}...")
                
                img_t = images_seq[:, t]
                ext_t = extrinsics_seq[:, t]
                
                ego_motion = None
                if config.use_ego_motion and t > 0:
                    # 自车运动是从前一帧到当前帧的变换。
                    # T_{t-1 -> t} = (T_{world -> t}) @ T_{t-1 -> world}
                    # 假设外参是相机到世界的位姿 (T_c->w)
                    pose_curr = ext_t[:, 0]  # 以第一个相机为参考
                    pose_prev = extrinsics_seq[:, t-1, 0]
                    ego_motion = torch.linalg.inv(pose_curr) @ pose_prev
                    print("  - 已为对齐计算自车运动矩阵。")

                # 前向传播
                start_time = time.time()
                outputs = model(
                    images=img_t, 
                    intrinsics=intrinsics, 
                    extrinsics=ext_t, 
                    memory=memory, 
                    ego_motion=ego_motion
                )
                duration_ms = (time.time() - start_time) * 1000
                
                # --- 5. 验证当前步骤的输出 ---
                print(f"  - 前向传播完成，耗时 {duration_ms:.2f} ms。")

                # 检查语义logits
                logits = outputs.get('semantic')
                assert logits is not None, "模型输出缺少 'semantic' 键。"
                expected_shape = (B, config.num_classes, *config.voxel_size)
                if logits.shape == expected_shape:
                    print(f"  ✅ 语义 logits 形状正确: {logits.shape}")
                else:
                    raise AssertionError(f"语义 logits 形状不匹配！期望 {expected_shape}, 得到 {logits.shape}")
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise ValueError("语义 logits 包含 NaN 或 Inf 值！")
                else:
                    print("  ✅ 语义 logits 数值有效 (无 NaN/Inf)。")

                # 检查时序记忆
                if config.use_temporal:
                    memory = outputs.get('memory')
                    assert memory is not None, "时序融合已启用，但模型输出缺少 'memory' 键。"
                    
                    expected_mem_shape = (B, config.num_coarse_queries, config.memory_dim)
                    if memory.shape == expected_mem_shape:
                        print(f"  ✅ Memory 形状正确: {memory.shape}")
                    else:
                         raise AssertionError(f"Memory 形状不匹配！期望 {expected_mem_shape}, 得到 {memory.shape}")

                    if torch.isnan(memory).any() or torch.isinf(memory).any():
                        raise ValueError("Memory 包含 NaN 或 Inf 值！")
                    else:
                        print("  ✅ Memory 数值有效 (无 NaN/Inf)。")
                    
                    # 为下一次迭代分离 memory，模拟推理循环
                    memory = memory.detach()
        
        print_header("✅ 验证成功通过！")

    except Exception as e:
        print_header("❌ 验证失败")
        print(f"发生错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    run_verification()

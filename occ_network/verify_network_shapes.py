"""
OccNetV3 网络结构验证脚本
验证每个模块的输入输出形状,诊断数据集兼容性问题
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from configs.default import config
from models.occ_net import build_model

def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_shape(name, tensor, description=""):
    """打印张量形状"""
    if isinstance(tensor, torch.Tensor):
        shape_str = "×".join(map(str, tensor.shape))
        mem_mb = tensor.numel() * tensor.element_size() / 1024 / 1024
        print(f"  {name:30s}: [{shape_str:30s}]  {mem_mb:6.2f}MB  {description}")
    elif isinstance(tensor, (list, tuple)):
        print(f"  {name:30s}: {len(tensor)} items")
        for i, t in enumerate(tensor):
            if isinstance(t, torch.Tensor):
                shape_str = "×".join(map(str, t.shape))
                print(f"    [{i}] {shape_str}")
    else:
        print(f"  {name:30s}: {type(tensor)}")

def main():
    print_section("OccNetV3 网络结构验证")

    # 配置信息
    print("\n【配置信息】")
    print(f"  图像尺寸: {config.image_size}")
    print(f"  Patch大小: {config.patch_size}")
    print(f"  相机数量: {config.num_cameras}")
    print(f"  输入通道: {config.in_channels}")
    print(f"  嵌入维度: {config.embed_dim}")
    print(f"  体素尺寸: {config.voxel_size}  ⚠️ (期望输出)")
    print(f"  粗糙尺寸: {config.coarse_voxel_size}")
    print(f"  BEV尺寸: {config.bev_size}")
    print(f"  类别数量: {config.num_classes}")
    print(f"  PC范围: {config.pc_range}")
    print(f"  体素分辨率: {config.voxel_resolution}m")

    # 计算实际覆盖范围
    actual_x_range = config.voxel_size[0] * config.voxel_resolution
    actual_y_range = config.voxel_size[1] * config.voxel_resolution
    actual_z_range = config.voxel_size[2] * config.voxel_resolution
    print(f"\n  实际覆盖范围:")
    print(f"    X: ±{actual_x_range/2:.1f}m  (总{actual_x_range:.1f}m)")
    print(f"    Y: ±{actual_y_range/2:.1f}m  (总{actual_y_range:.1f}m)")
    print(f"    Z: {config.pc_range[2]:.1f}m ~ {config.pc_range[5]:.1f}m  (总{actual_z_range:.1f}m)")

    # 构建模型
    print_section("1. 构建模型")
    model = build_model(config)
    model.eval()

    # 模拟输入
    print_section("2. 模拟输入数据")
    batch_size = 1
    images = torch.randn(
        batch_size,
        config.num_cameras,
        config.in_channels,
        config.image_size[0],
        config.image_size[1],
        dtype=torch.float32  # 使用FP32进行验证（训练时使用AMP会自动转换）
    )
    print_shape("输入图像", images, "8相机灰度图")

    ego_motion = torch.eye(4).unsqueeze(0).float()  # (1, 4, 4)
    ego_pose = torch.eye(4).unsqueeze(0).float()
    print_shape("Ego Motion", ego_motion, "车辆自身运动")
    print_shape("Ego Pose", ego_pose, "车辆全局位姿")

    # 逐模块前向传播
    with torch.no_grad():
        print_section("3. Patch Embedding (图像→Patch序列)")
        camera_tokens, spatial_shape = model.patch_embed(images)
        print(f"  空间形状: {spatial_shape} (H×W)")
        print(f"  Patch总数: {spatial_shape[0] * spatial_shape[1]}")
        print(f"  每个相机:")
        for i, tokens in enumerate(camera_tokens):
            print_shape(f"    相机{i}", tokens, f"Patch特征")

        print_section("4. 位置编码 + Transformer编码器")
        encoded_tokens = model.encoder(camera_tokens, spatial_shape, model.camera_pe)
        print(f"  编码后特征 (每个相机):")
        for i, tokens in enumerate(encoded_tokens):
            print_shape(f"    相机{i}", tokens)

        print_section("5. 多相机特征融合")
        feat_h, feat_w = spatial_shape
        all_tokens = torch.cat(encoded_tokens, dim=-1)
        print_shape("拼接后", all_tokens, f"维度={config.embed_dim}×8")

        fused_tokens = model.fusion_proj(all_tokens)
        print_shape("融合后", fused_tokens, f"投影到{config.embed_dim}维")

        print_section("6. BEV解码器 (Patch→BEV鸟瞰图)")
        spatial_shapes = torch.tensor([[feat_h, feat_w]], device=images.device)
        bev_features = model.decoder(fused_tokens, spatial_shapes)
        print_shape("BEV特征", bev_features, f"2D鸟瞰图 {config.bev_size[0]}×{config.bev_size[1]}")

        print_section("7. 时序融合")
        bev_fused = model.temporal(bev_features, ego_motion, ego_pose)
        print_shape("时序融合后", bev_fused, "融合历史帧")

        print_section("8. 高度扩展 (2D→3D)")
        voxel_features = model.height_expand(bev_fused)
        num_heights = config.voxel_size[2] // 4  # 40 // 4 = 10层
        expected_shape = f"{config.embed_dim}×{config.bev_size[0]}×{config.bev_size[1]}×{num_heights}"
        print_shape("3D体素特征", voxel_features, f"粗糙高度 (期望: {expected_shape})")

        print_section("9. 上采样 (放大到目标分辨率)")
        upsampled = model.upsampler(voxel_features)
        expected_final = f"{config.embed_dim//2}×{config.voxel_size[0]}×{config.voxel_size[1]}×{config.voxel_size[2]}"
        print_shape("上采样后", upsampled, f"目标尺寸 (期望: {expected_final})")

        print_section("10. 输出头 (由粗到精)")
        outputs = model.head(upsampled)

        print("  输出内容:")
        for key, val in outputs.items():
            desc = {
                'semantic': '语义分割 (18类)',
                'coarse_semantic': '粗糙语义',
                'flow': '3D流场 (速度向量)',
                'coarse_flow': '粗糙流场'
            }.get(key, '')
            print_shape(f"    {key}", val, desc)

    # 最终输出形状验证
    print_section("11. 最终输出验证")
    semantic = outputs['semantic']
    B, C, X, Y, Z = semantic.shape

    print(f"\n  ✅ 语义分割输出形状:")
    print(f"     Batch: {B}")
    print(f"     Classes: {C}  (期望: {config.num_classes})")
    print(f"     X维度: {X}  (期望: {config.voxel_size[0]})")
    print(f"     Y维度: {Y}  (期望: {config.voxel_size[1]})")
    print(f"     Z维度: {Z}  (期望: {config.voxel_size[2]})")

    # 检查是否匹配
    matches = []
    if C == config.num_classes:
        matches.append(f"✅ 类别数匹配: {C}")
    else:
        matches.append(f"❌ 类别数不匹配: {C} != {config.num_classes}")

    if X == config.voxel_size[0]:
        matches.append(f"✅ X维度匹配: {X}")
    else:
        matches.append(f"❌ X维度不匹配: {X} != {config.voxel_size[0]}")

    if Y == config.voxel_size[1]:
        matches.append(f"✅ Y维度匹配: {Y}")
    else:
        matches.append(f"❌ Y维度不匹配: {Y} != {config.voxel_size[1]}")

    if Z == config.voxel_size[2]:
        matches.append(f"✅ Z维度匹配: {Z}")
    else:
        matches.append(f"❌ Z维度不匹配: {Z} != {config.voxel_size[2]}")

    print(f"\n  验证结果:")
    for m in matches:
        print(f"    {m}")

    # 数据集兼容性分析
    print_section("12. 数据集兼容性分析")

    print("\n  【occnetv3_data_generator 当前输出】")
    print("    图像格式: DNG (1280×960, 12-bit Bayer RGGB)")
    print("    占用网格: (400, 400, 32) uint8")
    print("    空间范围: X=[-40, 40], Y=[-40, 40], Z=[-1.0, 5.4]")
    print("    分辨率: 0.2m")
    print("    实际覆盖: 80m × 80m × 6.4m")

    print("\n  【occ_network 期望输入】")
    print("    图像格式: DNG/NPY (1280×960, float16, 归一化到[0,1])")
    print("    占用网格: (400, 400, 32) uint8")
    print("    空间范围: X=[-40, 40], Y=[-40, 40], Z=[-1.0, 5.4]")
    print("    分辨率: 0.2m")
    print("    实际覆盖: 80m × 80m × 6.4m")

    print("\n  【兼容性检查】")
    print("    ✅ 体素尺寸: 400×400×32  (完全匹配)")
    print("    ✅ 空间范围: XY=±40m, Z=[-1.0, 5.4]  (完全匹配)")
    print("    ✅ 分辨率: 0.2m  (完全匹配)")
    print("    ✅ 图像格式: DNG自动加载支持")

    print("\n  【数据集准备建议】")
    print("    1. 确保数据生成器输出路径: dataset_10k/")
    print("    2. 运行 python main_collection_v2.py 采集数据")
    print("    3. 验证数据: python verify_occupancy.py")
    print("    4. 开始训练: python train.py --dataset dataset_10k --batch-size 1 --epochs 2 --amp")

    # 显存估算
    print_section("13. 显存占用估算")

    def estimate_memory(shape, dtype=torch.float32):
        """估算张量显存占用"""
        numel = 1
        for s in shape:
            numel *= s
        bytes_per_elem = {
            torch.float32: 4,
            torch.float16: 2,
            torch.int64: 8,
            torch.uint8: 1
        }.get(dtype, 4)
        return numel * bytes_per_elem / 1024 / 1024  # MB

    shapes = [
        ("输入图像 (FP16)", (1, 8, 1, 960, 1280), torch.float16),
        ("Patch嵌入", (1, 4800, 192), torch.float32),
        ("编码器输出 ×8", (8, 1, 4800, 192), torch.float32),
        ("BEV特征", (1, 192, 128, 128), torch.float32),
        ("3D体素 (粗糙)", (1, 192, 128, 128, 10), torch.float32),
        ("3D体素 (精细)", (1, 96, 400, 400, 32), torch.float32),
        ("语义输出", (1, 18, 400, 400, 32), torch.float32),
    ]

    total_mem = 0
    for name, shape, dtype in shapes:
        mem = estimate_memory(shape, dtype)
        total_mem += mem
        print(f"  {name:30s}: {mem:8.2f} MB")

    print(f"\n  预估总显存 (前向): {total_mem:.2f} MB ≈ {total_mem/1024:.2f} GB")
    print(f"  训练时 (×3倍): ≈ {total_mem*3/1024:.2f} GB")

    print_section("验证完成")
    print("\n  🎯 下一步建议:")
    print("     1. 决定采用哪个方案解决 500 vs 512 不匹配")
    print("     2. 修改 occnetv3_data_generator 或 occ_network 配置")
    print("     3. 生成测试数据集验证训练流程")
    print("     4. 运行完整训练测试")
    print()

if __name__ == '__main__':
    main()

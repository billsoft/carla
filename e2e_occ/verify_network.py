"""
e2e_occ 网络自检（2026-10 重构后）

1. 逐层 shape 报告：对每个子模块挂 forward hook，记录输入/输出形状和调用次数，
   完整表写入 e2e_occ/shape_report.md，并对各阶段关键形状做断言。
   覆盖多种配置组合（默认 / 稠密细化 / 隐式头 / 全局时序 / 接近重构前的基线）。
2. 几何测试（使用真实相机标定，非单位外参）：
   - 等距投影往返一致性
   - 语义检查：车前 10m 地面点落在 front_main 下半部中央；左侧点落在左侧相机；车后点落在 rear
   - 反例：用旧约定（UE 相机轴 / 含世界绝对位姿的外参）同样的点投影失败——证明测试有区分度
3. 时序对齐方向测试：车辆前进/右移一个粗格，记忆中的标记应沿 -X / -Y 平移
4. 训练冒烟：T=2 序列前向 + 反向，检查梯度有限、关键参数有梯度，报告峰值显存

用法: python e2e_occ/verify_network.py [--skip_train] [--quick]
"""

import argparse

import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import E2EOccConfig
from e2e_occ_net import build_model
from dataset import default_calibration
from deformable_attention import get_reference_points
from position_encoding import pixel_rays_vehicle
from geometry import equidistant_project, equidistant_unproject, M_CV_TO_UE
from temporal_fusion import TemporalFusionModule
from loss import E2ELoss, miou_from_confusion

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FAILS = []


def check(cond, msg):
    print(('  [PASS] ' if cond else '  [FAIL] ') + msg)
    if not cond:
        FAILS.append(msg)


def header(t):
    print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)


def shape_of(x):
    if torch.is_tensor(x):
        return 'x'.join(str(s) for s in x.shape) + f' {str(x.dtype).replace("torch.", "")}'
    if isinstance(x, dict):
        return '{' + ', '.join(f'{k}: {shape_of(v)}' for k, v in x.items() if v is not None) + '}'
    if isinstance(x, (list, tuple)):
        inner = [shape_of(v) for v in x if v is not None]
        inner = [s for s in inner if s]
        return '(' + ', '.join(inner) + ')' if inner else ''
    return ''


# ----------------------------------------------------------------------------
# 1. 逐层 shape
# ----------------------------------------------------------------------------

def collect_shapes(model, run):
    records = {}
    hooks = []
    for name, mod in model.named_modules():
        if name == '':
            continue

        def hook(m, inp, out, name=name):
            r = records.setdefault(name, {'type': type(m).__name__, 'in': shape_of(inp),
                                          'out': shape_of(out), 'calls': 0})
            r['calls'] += 1
        hooks.append(mod.register_forward_hook(hook))
    try:
        run()
    finally:
        for h in hooks:
            h.remove()
    return records


def make_inputs(cfg, B=1):
    K, T, _ = default_calibration(cfg.num_cameras, cfg.image_size)
    K = K.unsqueeze(0).expand(B, -1, -1, -1).to(DEVICE)
    T = T.unsqueeze(0).expand(B, -1, -1, -1).to(DEVICE)
    img = torch.rand(B, cfg.num_cameras, 1, *cfg.image_size, device=DEVICE)
    return img, K, T




def verify_shapes(name, cfg, write_table=None):
    header(f'1. 逐层 shape 验证 —— 配置: {name}')
    model = build_model(cfg).to(DEVICE).eval()
    print(f'  参数量: {model.get_num_params() / 1e6:.2f}M')
    img, K, T = make_inputs(cfg)
    B, N, C = 1, cfg.num_cameras, cfg.embed_dim
    em = torch.eye(4, device=DEVICE).unsqueeze(0)
    em[:, 0, 3] = -0.5
    outs = {}

    def run():
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
            o1 = model(img, K, T, need_aux=True)            # eval 默认不算深度辅助头，这里要验证其形状
            o2 = model(img, K, T, memory=o1['memory'], ego_motion=em,
                       time_delta=torch.full((B,), cfg.frame_dt, device=DEVICE), need_aux=True)
        outs['o1'], outs['o2'] = o1, o2

    if DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    rec = collect_shapes(model, run)
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == 'cuda' else 0
    print(f'  两帧前向耗时 {time.time() - t0:.1f}s，推理峰值显存 {peak:.2f}GB')

    # ---- 关键阶段形状断言 ----
    def rec_out(n):
        return rec[n]['out'] if n in rec else None

    lv = {s: cfg.level_size(s) for s in cfg.level_strides}
    pe = rec_out('patch_embed')
    for s, (h, w) in lv.items():
        if s == 32:
            continue
        check(f'{s}: {B}x{N}x{C}x{h}x{w}' in pe, f'patch_embed stride{s} 输出 [{B},{N},{C},{h},{w}]')
    enc = rec_out('encoder')
    for s, (h, w) in lv.items():
        check(f'{s}: {B}x{N}x{C}x{h}x{w}' in enc, f'encoder stride{s} 输出 [{B},{N},{C},{h},{w}]')
    if cfg.use_depth_aux:
        dh, dw = cfg.level_size(cfg.depth_stride)
        check(f'depth_logits: {B}x{N}x{cfg.depth_bins}x{dh}x{dw}' in enc,
              f'深度 logits [{B},{N},{cfg.depth_bins},{dh},{dw}]')
    cq = cfg.num_coarse_queries
    check(rec_out('decoder.coarse_layers.0').startswith(f'{B}x{cq}x{C}'), f'粗解码层输出 [{B},{cq},{C}]')
    if cfg.use_temporal:
        check(rec_out('decoder.temporal_fusion') == f'({B}x{cq}x{C} float32, {B}x{cq}x{C} float32)'
              or rec_out('decoder.temporal_fusion').startswith(f'({B}x{cq}x{C}'),
              f'时序融合输出 (fused, memory) 均为 [{B},{cq},{C}]')
    K_fine = int(cfg.num_fine_queries * cfg.fine_keep_ratio) if cfg.use_sparse_fine else cfg.num_fine_queries
    check(rec_out('decoder.fine_layers.0').startswith(f'{B}x{K_fine}x{C}'),
          f'细解码层输出 [{B},{K_fine},{C}]（{"稀疏 top-k" if cfg.use_sparse_fine else "稠密"}）')
    fx, fy, fz = cfg.fine_size
    check(f'fine_feats: {B}x{fx}x{fy}x{fz}x{C}' in rec_out('decoder'), f'解码器输出 [{B},{fx},{fy},{fz},{C}]')
    vx, vy, vz = cfg.voxel_size
    o2 = outs['o2']
    check(tuple(o2['semantic'].shape) == (B, cfg.num_classes, vx, vy, vz),
          f'语义输出 [{B},{cfg.num_classes},{vx},{vy},{vz}]')
    if cfg.use_flow_head:
        check(tuple(o2['flow'].shape) == (B, 3, vx, vy, vz), f'流场输出 [{B},3,{vx},{vy},{vz}]')
    check(tuple(o2['coarse_occ_logits'].shape) == (B, *cfg.coarse_size), f'粗占用 logits [{B},{cfg.coarse_size}]')
    check(torch.isfinite(o2['semantic'].float()).all().item(), '语义输出无 NaN/Inf')

    if write_table:
        with open(write_table, 'a', encoding='utf-8') as f:
            f.write(f'\n## 配置: {name}\n\n参数量 {model.get_num_params() / 1e6:.2f}M，'
                    f'两帧(第二帧带时序记忆)前向，batch=1，bf16 autocast。'
                    f'"调用次数">1 表示该模块在相机/层级/帧循环中被多次调用，形状取第一次调用。\n\n')
            f.write('| 模块 | 类型 | 调用次数 | 输入 | 输出 |\n|---|---|---|---|---|\n')
            for n, r in rec.items():
                f.write(f"| `{n}` | {r['type']} | {r['calls']} | {r['in']} | {r['out']} |\n")
    # 控制台打印前两级模块
    print('  —— 主要模块（完整逐层表见 shape_report.md）——')
    for n, r in rec.items():
        if n.count('.') <= 1 and not n.split('.')[-1].isdigit():
            print(f"  {n:<42s} in={r['in'][:70]:<70s} out={r['out'][:90]}")
    del model
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# 2. 几何
# ----------------------------------------------------------------------------

def verify_geometry():
    header('2. 几何测试（真实相机标定）')
    cfg = E2EOccConfig()
    K, T, _ = default_calibration(cfg.num_cameras, cfg.image_size)
    K, T = K.unsqueeze(0), T.unsqueeze(0)
    H, W = cfg.image_size
    names = ['front_main', 'front_wide', 'front_narrow', 'left_pillar', 'right_pillar',
             'left_repeater', 'right_repeater', 'rear']

    # 2.1 往返一致性（固定种子；额外加入主点附近 ±2px 的点——光轴附近是 acos 式实现的精度死角）
    errs = []
    gen = torch.Generator().manual_seed(0)
    for c in range(cfg.num_cameras):
        cx, cy = K[0, c, 0, 2].item(), K[0, c, 1, 2].item()
        u = torch.cat([torch.rand(2000, generator=gen) * (W - 1), cx + (torch.rand(200, generator=gen) - 0.5) * 4])
        v = torch.cat([torch.rand(2000, generator=gen) * (H - 1), cy + (torch.rand(200, generator=gen) - 0.5) * 4])
        d = equidistant_unproject(u, v, K[0, c, 0, 0], K[0, c, 1, 1], K[0, c, 0, 2], K[0, c, 1, 2])
        u2, v2, _ = equidistant_project(d * 7.3, K[0, c, 0, 0], K[0, c, 1, 1], K[0, c, 0, 2], K[0, c, 1, 2])
        errs.append(torch.maximum((u2 - u).abs(), (v2 - v).abs()).max().item())
    check(max(errs) < 1e-2, f'等距投影 像素→射线→像素 往返误差 {max(errs):.2e} px')

    rays = pixel_rays_vehicle(K, T, H // 16, W // 16, 16)                  # [1, N, h, w, 3]
    c0 = rays[0, 0, H // 32, W // 32]
    check(c0[0] > 0.99, f'front_main 图像中心射线指向车辆 +X（前）: {c0.numpy().round(3)}')
    c7 = rays[0, 7, H // 32, W // 32]
    check(c7[0] < -0.95, f'rear 图像中心射线指向车辆 -X（后）: {c7.numpy().round(3)}')

    def proj(points, ext):
        ref, valid = get_reference_points(points.view(1, -1, 3), K, ext, cfg.image_size)
        u = (ref[..., 0] + 1) * W / 2 - 0.5
        v = (ref[..., 1] + 1) * H / 2 - 0.5
        return u[0], v[0], valid[0]

    # 2.2 语义检查
    pts = torch.tensor([[10.0, 0.0, 0.0],      # 车前 10m 地面
                        [0.0, -10.0, 0.5],     # 左侧 10m
                        [0.0, 10.0, 0.5],      # 右侧 10m
                        [-10.0, 0.0, 0.5]])    # 车后 10m
    u, v, valid = proj(pts, T)
    check(valid[0, 0] and abs(u[0, 0] - W / 2) < 20 and v[0, 0] > H / 2,
          f'车前10m地面点 → front_main 下半部中央 (u={u[0, 0]:.0f}, v={v[0, 0]:.0f})')
    left_cams = [names[i] for i in range(8) if valid[i, 1]]
    right_cams = [names[i] for i in range(8) if valid[i, 2]]
    rear_cams = [names[i] for i in range(8) if valid[i, 3]]
    check(len(left_cams) > 0 and all('left' in n for n in left_cams), f'左侧点只出现在左侧相机: {left_cams}')
    check(len(right_cams) > 0 and all('right' in n for n in right_cams), f'右侧点只出现在右侧相机: {right_cams}')
    check('rear' in rear_cams and not any(n.startswith('front') for n in rear_cams),
          f'车后点出现在 rear 且不在前视相机: {rear_cams}')

    # 2.3 反例：旧约定应当失败（证明上面的测试有区分度）
    M = torch.eye(4, dtype=torch.float32)
    M[:3, :3] = torch.from_numpy(M_CV_TO_UE).float()
    T_ue_axes = T @ torch.linalg.inv(M)                                    # 去掉轴置换 = 旧代码用的 UE 轴外参
    u, v, valid = proj(pts, T_ue_axes)
    old_ok = bool(valid[0, 0] and abs(u[0, 0] - W / 2) < 20 and v[0, 0] > H / 2)
    check(not old_ok, f'[反例] 旧约定（UE 相机轴当 OpenCV 轴）下车前点不再落在 front_main 正确位置 '
          f'(valid={bool(valid[0, 0])}, u={u[0, 0]:.0f}, v={v[0, 0]:.0f})')
    pose = torch.eye(4)
    pose[0, 3], pose[1, 3] = 100.0, -50.0                                 # 车辆不在世界原点
    T_world = pose @ T                                                    # 旧代码喂给网络的外参
    u, v, valid = proj(pts, T_world)
    # 旧约定把查询点当成世界原点附近的点：车在 (100,-50) 时，"车前 10m" 实际被投影到
    # 车后约 90m、偏右 50m 处，落进 rear/右侧相机而不是 front_main
    check(not valid[0, 0], f'[反例] 旧约定（外参含世界绝对位姿）下车前点不在 front_main 中，而被投影到: '
          f'{[names[i] for i in range(8) if valid[i, 0]]}')


# ----------------------------------------------------------------------------
# 3. 时序对齐方向
# ----------------------------------------------------------------------------

def verify_temporal_alignment():
    header('3. 时序记忆对齐方向测试')
    cfg = E2EOccConfig()
    tf = TemporalFusionModule(cfg, use_checkpoint=False)
    X, Y, Z = cfg.coarse_size
    cell = (cfg.voxel_range[3] - cfg.voxel_range[0]) / X                  # 3.2m
    mem = torch.zeros(1, X, Y, Z, 4)
    mem[0, 12, 12, 4, 0] = 1.0
    mem = mem.view(1, -1, 4)

    def peak_after(dx, dy, yaw_deg=0.0):
        # ego_motion = 上一帧车辆系 → 当前帧车辆系；车辆前进 dx 时，静止物体在新坐标系里 x 减小 dx
        a = math.radians(yaw_deg)
        pose_prev = torch.eye(4)
        pose_t = torch.eye(4)
        pose_t[0, 0], pose_t[0, 1], pose_t[1, 0], pose_t[1, 1] = math.cos(a), -math.sin(a), math.sin(a), math.cos(a)
        pose_t[0, 3], pose_t[1, 3] = dx, dy
        em = (torch.linalg.inv(pose_t) @ pose_prev).unsqueeze(0)
        aligned, valid = tf.align_memory(mem, em)
        idx = aligned[0, :, 0].view(X, Y, Z).argmax()
        return [int(i) for i in torch.unravel_index(idx, (X, Y, Z))], valid

    p, _ = peak_after(0.0, 0.0)
    check(p == [12, 12, 4], f'静止：标记保持在 (12,12,4)，实际 {p}')
    p, valid = peak_after(cell, 0.0)
    check(p == [11, 12, 4], f'车辆前进 {cell}m：标记沿 -X 移动到 (11,12,4)，实际 {p}')
    vv = valid.view(X, Y, Z)
    check(vv[-1].mean() < 0.5 and vv[0].mean() > 0.99, '前进后最前一排为新进入视野（valid≈0），最后一排有效')
    p, _ = peak_after(0.0, cell)
    check(p == [12, 11, 4], f'车辆右移 {cell}m：标记沿 -Y 移动到 (12,11,4)，实际 {p}')
    # 原地右转 90°：车前方 (x=+d) 的点在新坐标系里变成左侧 (y=-d)
    mem2 = torch.zeros(1, X, Y, Z, 4)
    mem2[0, 18, 12, 4, 0] = 1.0                                          # 车前约 19m
    mem_bak = mem.clone()
    mem.copy_(mem2.view(1, -1, 4))
    p, _ = peak_after(0.0, 0.0, yaw_deg=90.0)
    mem.copy_(mem_bak)
    check(p[0] == 12 and p[1] < 12, f'原地右转 90°：车前标记转到车辆左侧（y 减小），实际 {p}')


# ----------------------------------------------------------------------------
# 4. 训练冒烟
# ----------------------------------------------------------------------------

def verify_train_step(name, cfg):
    header(f'4. 训练冒烟（T=2 前向+反向，bf16）—— 配置: {name}')
    model = build_model(cfg).to(DEVICE).train()
    crit = E2ELoss(cfg)
    img, K, T = make_inputs(cfg)
    X, Y, Z = cfg.voxel_size
    vox = torch.zeros(1, X, Y, Z, dtype=torch.long, device=DEVICE)
    vox[:, :, :, 4] = 11
    vox[:, 220:240, 190:210, 5:12] = 4
    tg = {'voxels': vox,
          'depth': torch.full((1, cfg.num_cameras, *cfg.level_size(cfg.depth_stride)), 10.0, device=DEVICE),
          'has_depth': torch.tensor([True], device=DEVICE),
          'flow': torch.zeros(1, 3, X, Y, Z, device=DEVICE),
          'flow_mask': (vox == 4),
          'has_flow': torch.tensor([True], device=DEVICE)}
    if DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    memory = None
    # 与 train.py 默认 --tbptt 1 一致：每帧单独反向，记忆逐帧 detach
    for t in range(2):
        qp = None
        if cfg.head_type == 'implicit':
            from loss import sample_implicit_points
            tg['point_idx'], qp = sample_implicit_points(vox, cfg, cfg.implicit_train_points)
        em = torch.eye(4, device=DEVICE).unsqueeze(0) if t > 0 else None
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
            out = model(img, K, T, memory=memory.detach() if memory is not None else None, ego_motion=em,
                        time_delta=torch.full((1,), cfg.frame_dt, device=DEVICE) if t > 0 else None,
                        query_points=qp)
            losses = crit(out, tg)
        memory = out['memory']
        losses['total'].backward()
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == 'cuda' else 0
    print('  损失项: ' + ', '.join(f'{k}={float(v):.3f}' for k, v in losses.items()))
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check(all(torch.isfinite(g).all() for g in grads), '所有梯度有限')
    key = {
        'patch_embed.patch_embed.rggb_conv.weight': '图像骨干',
        'decoder.coarse_layers.0.cross_attn.sampling_offsets.weight': '粗阶段可变形采样偏移',
        'decoder.fine_layers.0.cross_attn.value_proj.weight': '细阶段交叉注意力',
        'decoder.coarse_occ_head.1.weight': '粗占用头',
    }
    if cfg.use_temporal:
        key['decoder.temporal_fusion.mem_out.weight'] = '记忆回注分支（GRU 输出 → 当前帧，GRU 的梯度入口）'
    if cfg.use_depth_aux:
        key['encoder.depth_head.net.0.weight'] = '深度头'
    params = dict(model.named_parameters())
    for k, desc in key.items():
        g = params[k].grad
        check(g is not None and g.abs().sum() > 0, f'{desc} 有梯度 ({k})')
    print(f'  两帧前向+反向耗时 {dt:.1f}s，训练峰值显存 {peak:.2f}GB')

    if cfg.use_temporal:
        # mem_out 零初始化，第一步时 GRU 门控梯度为 0；模拟一次参数更新（mem_out 变为非零）后，
        # 在流式训练（记忆 detach）下 GRU 门控必须能从当前帧损失拿到梯度
        model.zero_grad(set_to_none=True)
        nn.init.normal_(model.decoder.temporal_fusion.mem_out.weight, std=1e-2)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
            out = model(img, K, T, memory=memory.detach(), ego_motion=torch.eye(4, device=DEVICE).unsqueeze(0),
                        time_delta=torch.full((1,), cfg.frame_dt, device=DEVICE),
                        query_points=qp)
            crit(out, tg)['total'].backward()
        g = dict(model.named_parameters())['decoder.temporal_fusion.gate.update_gate.0.weight'].grad
        check(g is not None and g.abs().sum() > 0, 'mem_out 非零后，记忆 detach 的流式训练下 GRU 门控有梯度')
    del model
    torch.cuda.empty_cache()
    return peak


def verify_regressions():
    """2026-10-15 评审发现并修复的问题，各自留一个回归检查。"""
    header('5. 回归检查（混合精度几何 / BN 检查点 / 多 batch 时序 / 推理开销 / mIoU）')
    import copy
    cfg = E2EOccConfig()
    K, T, _ = default_calibration(cfg.num_cameras, cfg.image_size)
    K, T = K.unsqueeze(0).to(DEVICE), T.unsqueeze(0).to(DEVICE)
    from geometry import voxel_centers
    pts = voxel_centers(cfg.fine_size, cfg.voxel_range).unsqueeze(0).to(DEVICE)
    r32, v32 = get_reference_points(pts, K, T, cfg.image_size)
    ray32 = pixel_rays_vehicle(K, T, 120, 160, 8)
    with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
        r, v = get_reference_points(pts, K, T, cfg.image_size)
        ray = pixel_rays_vehicle(K, T, 120, 160, 8)
    err = ((r - r32) * torch.tensor([cfg.image_size[1] / 2, cfg.image_size[0] / 2], device=DEVICE)).norm(dim=-1)
    check(r.dtype == torch.float32 and err[v & v32].max().item() < 1e-3 and bool((v == v32).all()),
          f'bf16 autocast 下投影保持 FP32 (最大误差 {err[v & v32].max().item():.2e}px，可见性一致)')
    check((ray - ray32).abs().max().item() < 1e-6, 'bf16 autocast 下射线方向保持 FP32')

    model = build_model(cfg).to(DEVICE)
    head = model.head.train()
    if hasattr(head, 'conv1'):
        ref = copy.deepcopy(head)
        ref.config = copy.copy(cfg)
        ref.config.checkpoint_head = False
        bns = [m for m in head.modules() if isinstance(m, nn.BatchNorm3d)]
        bns_ref = [m for m in ref.modules() if isinstance(m, nn.BatchNorm3d)]
        n0 = int(bns[0].num_batches_tracked)
        x = torch.randn(1, 20, 20, 4, cfg.embed_dim, device=DEVICE)
        for h in (head, ref):
            xi = x.clone().requires_grad_()
            h(xi)['semantic'].float().mean().backward()
        diff = max((a.running_var - b.running_var).abs().max().item() for a, b in zip(bns, bns_ref))
        check(int(bns[0].num_batches_tracked) - n0 == 1 and diff < 1e-6,
              f'梯度检查点重算不重复更新 BN 统计 (计数 +{int(bns[0].num_batches_tracked) - n0}，running_var 差 {diff:.1e})')

    model.eval()
    imgs = torch.rand(2, 2, cfg.num_cameras, 1, *cfg.image_size, device=DEVICE)
    try:
        with torch.no_grad(), torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
            out = model(imgs[:, 0], K.expand(2, -1, -1, -1), T.expand(2, -1, -1, -1), need_flow=False)
        check(out['semantic'].shape[0] == 2, 'B=2 不连续时间切片 images[:, t] 前向正常')
        check(out['depth_logits'] is None and out['flow'] is None, 'eval 默认不算深度辅助头；need_flow=False 不算流场')
    except RuntimeError as e:
        check(False, f'B=2 不连续时间切片前向: {str(e)[:80]}')
    del model, imgs
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    c = torch.zeros(cfg.num_classes, cfg.num_classes, dtype=torch.long)
    c[0, 0], c[1, 1], c[0, 2] = 1000, 100, 50          # 类别 1 全对，free 被错报成真值里没有的类别 2
    check(abs(miou_from_confusion(c)[0] - 0.5) < 1e-6, 'mIoU 计入凭空预测出的类别 (=0.5，旧实现为 1.0)')


def variant(**kw):
    c = E2EOccConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip_train', action='store_true')
    ap.add_argument('--quick', action='store_true', help='只跑默认配置')
    args = ap.parse_args()

    verify_geometry()
    verify_temporal_alignment()
    verify_regressions()

    report = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shape_report.md')
    with open(report, 'w', encoding='utf-8') as f:
        f.write('# e2e_occ 逐层 shape 报告\n\n由 `verify_network.py` 自动生成，'
                '记录每个子模块第一次调用时的输入/输出张量形状与 dtype。\n')

    configs = [('默认（全部优化开启）', E2EOccConfig())]
    if not args.quick:
        configs += [
            ('稠密细化（use_sparse_fine=False）', variant(use_sparse_fine=False)),
            ('隐式输出头（head_type=implicit）', variant(head_type='implicit')),
            ('全局时序注意力（temporal_attn_type=global）', variant(temporal_attn_type='global')),
            ('接近重构前基线（单尺度/无深度/无流场/LayerNorm/GELU/无RoPE/无移位窗口）',
             variant(use_multiscale=False, use_depth_aux=False, use_flow_head=False, cam_visibility_norm=False,
                     use_sparse_fine=False, temporal_attn_type='global', use_time_embedding=False,
                     norm_type='layernorm', ffn_type='gelu', use_rope_3d=False, use_shifted_window=False)),
        ]
    for name, cfg in configs:
        verify_shapes(name, cfg, write_table=report)
    print(f'\n逐层 shape 表已写入 {report}')

    if not args.skip_train and DEVICE.type == 'cuda':
        verify_train_step('默认', E2EOccConfig())
        if not args.quick:
            verify_train_step('隐式输出头', variant(head_type='implicit'))

    header('结果')
    if FAILS:
        print(f'{len(FAILS)} 项失败:')
        for m in FAILS:
            print('  - ' + m)
        sys.exit(1)
    print('全部通过')


if __name__ == '__main__':
    main()

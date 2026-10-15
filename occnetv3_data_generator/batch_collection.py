"""
批量采集：多地图 × 多场景 (每个场景不同出生点/交通/随机天气) 追加到同一个数据集。

每个场景单独起一个 main_collection.py 子进程 (--append --scene-index i --seed base+i)，进程之间
互不影响：某个场景崩溃/超时只丢这一个场景，重新运行本脚本会跳过已采完的场景继续。
train/val/test 按场景名哈希划分 (data_saver.scene_split)，每个场景采完都会重写一次，覆盖全部场景。

    python batch_collection.py --host 192.168.100.81 --output D:/carla_datasets/occ_v2 \\
        --towns Town10HD_Opt Town15 --scenes-per-town 10 --frames 150 --frame-skip 5

每个场景 150 帧 × 抽帧 5 = 37.5 秒行驶；10 场景 × 2 地图 = 3000 帧。实测 (test_v2)：每帧约 48MB
(8 DNG ~20MB + 8 张 float16 深度图 ~20MB + 体素/掩码/压缩流场 ~8MB)，3000 帧约 145GB；速度 Town10HD_Opt 约 13s/帧、
Town15 约 25s/帧 (抽帧时中间步的各路相机照样渲染)，两图各半 3000 帧约 16 小时。输出务必放在空间足够的盘 (不要放 C 盘)。
每个场景开始前都用 remote/reset_carla_world.py 复位常驻服务端。
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def reset_world(host, port):
    """remote/reset_carla_world.py：关同步模式、销毁残留的车/人/传感器 (失败只告警)。
    子进程被超时杀掉时它的 finally 不会执行，ego/NPC/16 个传感器和同步模式都会留在常驻服务端上，
    同地图的下一个场景不重载地图会全部继承，越跑越慢、连环超时。"""
    script = HERE.parent / 'remote' / 'reset_carla_world.py'
    try:
        r = subprocess.run([sys.executable, str(script), '--host', host, '--port', str(port), '--keep-weather'],
                           capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            print(f'[batch] ⚠️ 复位服务端失败: {r.stdout[-300:]}{r.stderr[-300:]}')
    except subprocess.TimeoutExpired:
        print('[batch] ⚠️ 复位服务端超时')


def scene_frames(output, scene_name, index):
    return len(list((Path(output) / 'occupancy').glob(f'{scene_name}_{index:04d}_frame_*.npy')))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--output', required=True)
    ap.add_argument('--towns', nargs='+', default=['Town10HD_Opt', 'Town15'])
    ap.add_argument('--scenes-per-town', type=int, default=10)
    ap.add_argument('--frames', type=int, default=150, help='每个场景保存的帧数')
    ap.add_argument('--frame-skip', type=int, default=5, help='每隔多少仿真步 (0.05s) 保存一帧')
    ap.add_argument('--warmup-ticks', type=int, default=100, help='每个场景正式采集前空跑的步数 (5 秒)')
    ap.add_argument('--num-vehicles', type=int, default=40)
    ap.add_argument('--num-walkers', type=int, default=30)
    ap.add_argument('--weather', default='random')
    ap.add_argument('--seed', type=int, default=1000, help='场景 i 用 seed+i，整批可复现')
    ap.add_argument('--raw-bit-depth', type=int, default=12)
    ap.add_argument('--scene-name', default='scene')
    ap.add_argument('--sec-per-tick', type=float, default=8.0,
                    help='超时估算用的每仿真步秒数 (实测 Town10 约 2.6s/步、Town15 约 5s/步，留余量)；'
                         '子进程超时 = (frames×frame_skip + warmup)×此值 + 600s')
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(town, i) for i, town in enumerate(t for t in args.towns for _ in range(args.scenes_per_town))]
    # 场景编号连续分配：Town10HD_Opt 0..N-1，Town15 N..2N-1 ...
    log_dir = out / 'logs'
    log_dir.mkdir(exist_ok=True)
    t_all = time.time()
    failed = []
    for town, idx in jobs:
        have = scene_frames(out, args.scene_name, idx)
        if have >= args.frames:
            print(f'[batch] 场景 {idx:04d} ({town}) 已有 {have} 帧，跳过')
            continue
        if have:
            # 中途断掉的场景：时序不完整，整段重采 (main_collection 会覆盖同名样本，残留的多余帧删掉)
            for sub in ('occupancy', 'flow', 'flow_mask', 'ego_pose', 'ego_motion', 'mask_camera', 'debug_actor_ids'):
                for p in (out / sub).glob(f'{args.scene_name}_{idx:04d}_frame_*.npy'):
                    p.unlink()
            for sub in ('images', 'depth'):
                for d in (out / sub).glob(f'{args.scene_name}_{idx:04d}_frame_*'):
                    shutil.rmtree(d, ignore_errors=True)
            print(f'[batch] 场景 {idx:04d} 只有 {have} 帧 (上次中断)，重采')
        cmd = [sys.executable, str(HERE / 'main_collection.py'),
               '--host', args.host, '--port', str(args.port), '--town', town,
               '--output', str(out), '--append', '--scene', args.scene_name, '--scene-index', str(idx),
               '--frames', str(args.frames), '--frame-skip', str(args.frame_skip),
               '--warmup-ticks', str(args.warmup_ticks),
               '--num-vehicles', str(args.num_vehicles), '--num-walkers', str(args.num_walkers),
               '--weather', args.weather, '--seed', str(args.seed + idx),
               '--raw-bit-depth', str(args.raw_bit_depth)]
        print(f'\n[batch] === 场景 {idx:04d} / {len(jobs)}  {town}  seed={args.seed + idx} ===')
        reset_world(args.host, args.port)
        t0 = time.time()
        log_path = log_dir / f'{args.scene_name}_{idx:04d}.log'
        try:
            with open(log_path, 'w', encoding='utf-8') as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE),
                                    timeout=(args.frames * args.frame_skip + args.warmup_ticks) * args.sec_per_tick
                                    + 600).returncode
        except subprocess.TimeoutExpired:
            rc = 'timeout'
        got = scene_frames(out, args.scene_name, idx)
        print(f'[batch] 场景 {idx:04d} 结束: rc={rc}, {got}/{args.frames} 帧, {time.time() - t0:.0f}s, 日志 {log_path}')
        if got < args.frames:
            failed.append(idx)
    reset_world(args.host, args.port)
    print(f'\n[batch] 全部完成，耗时 {(time.time() - t_all) / 3600:.2f}h；未采满的场景: {failed or "无"}')
    if failed:
        print('[batch] 重新运行同一条命令会只重采这些场景')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Multi-GPU Launcher for PIDS Renderer
=====================================

這個腳本只做調度，不 import 任何 CUDA 相關模組。
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Launcher for PIDS Renderer')
    parser.add_argument('--input_dir', type=str, required=True, help='OBJ 場景目錄')
    parser.add_argument('--output', type=str, required=True, help='輸出目錄')
    parser.add_argument('--spp', type=int, default=16384, help='SPP')
    parser.add_argument('--max_scenes', type=int, default=None, help='最大場景數')
    parser.add_argument('--skip', type=int, default=0, help='跳過前 N 個場景')
    parser.add_argument('--no_preview', action='store_true', help='不保存預覽 PNG')
    parser.add_argument('--num_gpus', type=int, default=4, help='GPU 數量')
    args = parser.parse_args()

    # 渲染腳本路徑
    renderer_script = Path(__file__).parent / 'pids_renderer_random.py'

    # 收集場景（排除分離文件）
    all_objs = sorted(Path(args.input_dir).glob('*.obj'))
    scenes = [f for f in all_objs
              if not any(x in f.stem for x in ['_glass', '_ceiling', '_other'])]

    if args.skip > 0:
        scenes = scenes[args.skip:]
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    print(f"[Multi-GPU Launcher]")
    print(f"  GPU 數量: {args.num_gpus}")
    print(f"  待渲染: {len(scenes)} 個場景")
    print(f"  渲染腳本: {renderer_script}")

    if len(scenes) == 0:
        print("沒有場景需要渲染！")
        return

    # 分配場景
    scenes_per_gpu = len(scenes) // args.num_gpus

    print(f"\n場景分配:")
    processes = []

    for gpu_id in range(args.num_gpus):
        start_idx = gpu_id * scenes_per_gpu
        if gpu_id == args.num_gpus - 1:
            num_scenes_gpu = len(scenes) - start_idx
        else:
            num_scenes_gpu = scenes_per_gpu

        if num_scenes_gpu <= 0:
            continue

        print(f"  GPU {gpu_id}: {num_scenes_gpu} 個場景 (skip={args.skip + start_idx})")

        # 單 GPU 子進程命令
        cmd = [
            sys.executable, str(renderer_script),
            '--input_dir', args.input_dir,
            '--output', args.output,
            '--spp', str(args.spp),
            '--skip', str(args.skip + start_idx),
            '--max_scenes', str(num_scenes_gpu),
        ]
        if args.no_preview:
            cmd.append('--no_preview')

        # 設置環境變量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # 啟動子進程
        log_file = open(f'gpu{gpu_id}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((p, log_file, gpu_id))
        print(f"[啟動] GPU {gpu_id} PID: {p.pid}")

    print(f"\n所有 GPU 已啟動！")
    print(f"監控: tail -f gpu0.log gpu1.log gpu2.log gpu3.log")
    print(f"\n等待完成...")

    # 等待所有進程完成
    for p, log_file, gpu_id in processes:
        p.wait()
        log_file.close()
        print(f"[完成] GPU {gpu_id}")

    print(f"\n所有 GPU 渲染完成！")


if __name__ == '__main__':
    main()

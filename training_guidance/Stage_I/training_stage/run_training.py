#!/usr/bin/env python
"""
PIDS Training Launcher
快速啟動訓練的腳本，提供預設配置

Usage:
    # 小規模測試 (快速驗證)
    python run_training.py --mode test

    # 正式訓練
    python run_training.py --mode full

    # 自定義
    python run_training.py --data_dir /path/to/data --batch_size 8
"""

import os
import sys
import subprocess
from pathlib import Path


def get_default_data_dir():
    """獲取預設數據目錄"""
    script_dir = Path(__file__).parent
    # 使用整理後的 dataset 目錄
    data_dir = script_dir / 'dataset' / 'stereo_pairs'
    return data_dir.resolve()


def run_training(mode='full', **kwargs):
    """
    運行訓練

    Args:
        mode: 'test' (快速測試), 'full' (完整訓練), 'custom' (自定義)
        **kwargs: 額外的命令行參數
    """
    # 預設配置 (不使用 crop，保持全圖以維持全局幾何上下文)
    configs = {
        'test': {
            'batch_size': 2,
            'num_steps': 500,
            'iters': 6,
            'print_freq': 50,
            'val_freq': 1,
            'save_freq': 1,
        },
        'full': {
            'batch_size': 16,
            'accumulation_steps': 1,
            'num_steps': 50000,
            'iters': 12,
            'lr': 0.0001,
            'scheduler': 'cosine',
            'print_freq': 100,
            'val_freq': 500,
            'save_freq': 5,
        },
        'small': {
            'batch_size': 4,
            'num_steps': 20000,
            'iters': 12,
            'lr': 0.0002,
            'print_freq': 100,
            'val_freq': 1,
            'save_freq': 2,
        },
    }

    # 選擇配置
    config = configs.get(mode, configs['full']).copy()
    config.update(kwargs)

    # 設置路徑
    data_dir = config.pop('data_dir', None) or get_default_data_dir()
    output_dir = config.pop('output_dir', None) or Path(__file__).parent / 'checkpoints'

    # 確認數據目錄存在
    if not Path(data_dir).exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("\nPlease specify the correct path with --data_dir")
        sys.exit(1)

    # 創建輸出目錄
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 構建命令
    cmd = [
        sys.executable, 'train_pids.py',
        '--data_dir', str(data_dir),
        '--output_dir', str(output_dir),
    ]

    # 添加配置參數
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    # 添加混合精度
    cmd.append('--mixed_precision')

    print("=" * 60)
    print("PIDS Training Launcher")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Config: {config}")
    print("=" * 60)
    print()

    # 運行
    os.chdir(Path(__file__).parent)
    subprocess.run(cmd)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='PIDS Training Launcher')
    parser.add_argument('--mode', type=str, default='small',
                        choices=['test', 'small', 'full', 'custom'],
                        help='Training mode: test (500 steps), small (20k steps), full (100k steps)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_steps', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    # 收集自定義參數
    kwargs = {}
    if args.data_dir:
        kwargs['data_dir'] = args.data_dir
    if args.output_dir:
        kwargs['output_dir'] = args.output_dir
    if args.batch_size:
        kwargs['batch_size'] = args.batch_size
    if args.num_steps:
        kwargs['num_steps'] = args.num_steps
    if args.lr:
        kwargs['lr'] = args.lr
    if args.resume:
        kwargs['resume'] = args.resume

    run_training(mode=args.mode, **kwargs)


if __name__ == '__main__':
    main()

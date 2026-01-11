"""
TensorBoard Log to CSV Exporter
===============================

從 TensorBoard logs 目錄批量匯出 CSV 檔案，每個 metric 一個檔案。

Usage:
    python export_tensorboard_csv.py --logdir ./checkpoints/logs --output ./csv_exports

    # 只匯出特定 metrics
    python export_tensorboard_csv.py --logdir ./checkpoints/logs --output ./csv_exports \
        --metrics val/glass_epe val/loss train/loss

    # 匯出所有 metrics
    python export_tensorboard_csv.py --logdir ./checkpoints/logs --output ./csv_exports --all

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import csv

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: tensorboard not installed. Run: pip install tensorboard")
    exit(1)


def find_event_files(logdir: str) -> List[str]:
    """找到所有 TensorBoard event 檔案"""
    event_files = []
    logdir = Path(logdir)

    # 搜尋 events.out.tfevents.* 檔案
    for pattern in ['**/events.out.tfevents.*', '**/events.tfevents.*']:
        event_files.extend(logdir.glob(pattern))

    return [str(f) for f in event_files]


def load_tensorboard_logs(logdir: str) -> Dict[str, List[tuple]]:
    """
    載入 TensorBoard logs 並提取所有 scalar metrics

    Returns:
        Dict[metric_name, List[(wall_time, step, value)]]
    """
    print(f"Loading TensorBoard logs from: {logdir}")

    # 使用 EventAccumulator 載入
    ea = EventAccumulator(logdir, size_guidance={
        'scalars': 0,  # 0 = 載入全部
    })
    ea.Reload()

    # 獲取所有 scalar tags
    tags = ea.Tags().get('scalars', [])

    if not tags:
        print("Warning: No scalar metrics found in logs")
        return {}

    print(f"Found {len(tags)} metrics: {tags}")

    # 提取每個 metric 的數據
    metrics = {}
    for tag in tags:
        events = ea.Scalars(tag)
        metrics[tag] = [(e.wall_time, e.step, e.value) for e in events]
        print(f"  {tag}: {len(events)} data points")

    return metrics


def sanitize_filename(name: str) -> str:
    """將 metric 名稱轉換為安全的檔案名"""
    # 替換 / 為 _
    safe_name = name.replace('/', '_')
    # 移除其他不安全字元
    safe_name = ''.join(c if c.isalnum() or c in '_-.' else '_' for c in safe_name)
    return safe_name


def export_to_csv(
    metrics: Dict[str, List[tuple]],
    output_dir: str,
    prefix: str = '',
    selected_metrics: Optional[List[str]] = None,
) -> List[str]:
    """
    將 metrics 匯出為 CSV 檔案

    Args:
        metrics: Dict[metric_name, List[(wall_time, step, value)]]
        output_dir: 輸出目錄
        prefix: 檔案名前綴
        selected_metrics: 只匯出指定的 metrics (None = 全部)

    Returns:
        List of exported file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_files = []

    for metric_name, data in metrics.items():
        # 過濾指定的 metrics
        if selected_metrics and metric_name not in selected_metrics:
            continue

        if not data:
            print(f"  Skipping {metric_name}: no data")
            continue

        # 生成檔案名
        safe_name = sanitize_filename(metric_name)
        if prefix:
            filename = f"{prefix}_{safe_name}.csv"
        else:
            filename = f"{safe_name}.csv"

        filepath = output_dir / filename

        # 寫入 CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Wall time', 'Step', 'Value'])
            for wall_time, step, value in data:
                writer.writerow([wall_time, step, value])

        print(f"  Exported: {filepath} ({len(data)} rows)")
        exported_files.append(str(filepath))

    return exported_files


def list_available_metrics(logdir: str):
    """列出所有可用的 metrics"""
    metrics = load_tensorboard_logs(logdir)

    print("\n" + "=" * 50)
    print("Available Metrics:")
    print("=" * 50)

    for metric_name, data in sorted(metrics.items()):
        print(f"  {metric_name}: {len(data)} data points")

    print("=" * 50)
    print(f"\nTotal: {len(metrics)} metrics")
    print("\nUsage example:")
    print(f"  python export_tensorboard_csv.py --logdir {logdir} --output ./csv_exports --all")
    print(f"  python export_tensorboard_csv.py --logdir {logdir} --output ./csv_exports --metrics val/glass_epe val/loss")


def main():
    parser = argparse.ArgumentParser(
        description='Export TensorBoard logs to CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 列出所有可用 metrics
  python export_tensorboard_csv.py --logdir ./checkpoints/logs --list

  # 匯出所有 metrics
  python export_tensorboard_csv.py --logdir ./checkpoints/logs --output ./csv_exports --all

  # 只匯出指定 metrics
  python export_tensorboard_csv.py --logdir ./checkpoints/logs --output ./csv_exports \\
      --metrics val/glass_epe val/loss train/loss

  # 加上前綴
  python export_tensorboard_csv.py --logdir ./checkpoints/logs --output ./csv_exports \\
      --all --prefix exp18_cross_attention
        """
    )

    parser.add_argument('--logdir', type=str, required=True,
                        help='TensorBoard logs directory (contains events.out.tfevents.*)')
    parser.add_argument('--output', type=str, default='./csv_exports',
                        help='Output directory for CSV files')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                        help='Specific metrics to export (e.g., val/glass_epe val/loss)')
    parser.add_argument('--all', action='store_true',
                        help='Export all available metrics')
    parser.add_argument('--prefix', type=str, default='',
                        help='Prefix for output filenames')
    parser.add_argument('--list', action='store_true',
                        help='List available metrics without exporting')

    args = parser.parse_args()

    # 檢查 logdir 存在
    if not os.path.exists(args.logdir):
        print(f"Error: Log directory not found: {args.logdir}")
        return 1

    # 列出 metrics 模式
    if args.list:
        list_available_metrics(args.logdir)
        return 0

    # 需要指定 --all 或 --metrics
    if not args.all and not args.metrics:
        print("Error: Please specify --all or --metrics")
        print("Use --list to see available metrics")
        return 1

    # 載入 logs
    metrics = load_tensorboard_logs(args.logdir)

    if not metrics:
        print("Error: No metrics found in logs")
        return 1

    # 匯出
    print(f"\nExporting to: {args.output}")
    selected = None if args.all else args.metrics

    exported = export_to_csv(
        metrics=metrics,
        output_dir=args.output,
        prefix=args.prefix,
        selected_metrics=selected,
    )

    print(f"\n{'=' * 50}")
    print(f"Exported {len(exported)} CSV files to {args.output}")
    print('=' * 50)

    return 0


if __name__ == '__main__':
    exit(main())

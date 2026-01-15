#!/usr/bin/env python3
"""
TensorBoard Log Reader
讀取 TensorBoard event 檔案並匯出為 CSV

Usage:
    python read_tensorboard_logs.py --log_dir ./checkpoints_dual_stream_exp16
    python read_tensorboard_logs.py --log_dir ./checkpoints_dual_stream_exp16 --tags glass_epe val_loss
    python read_tensorboard_logs.py --log_dir ./checkpoints_dual_stream_exp16 --output ./exported_logs
"""

import argparse
import os
from pathlib import Path

def read_with_tbparse(log_dir: str, output_dir: str, tags: list = None):
    """使用 tbparse 讀取 (推薦，更簡潔)"""
    try:
        from tbparse import SummaryReader
        import pandas as pd
    except ImportError:
        print("tbparse not found. Install with: pip install tbparse pandas")
        return False

    print(f"Reading TensorBoard logs from: {log_dir}")
    reader = SummaryReader(log_dir)

    # 讀取所有 scalars
    df = reader.scalars

    if df.empty:
        print("No scalar data found!")
        return False

    # 顯示所有可用的 tags
    available_tags = df['tag'].unique()
    print(f"\nAvailable tags ({len(available_tags)}):")
    for tag in available_tags:
        count = len(df[df['tag'] == tag])
        print(f"  - {tag} ({count} points)")

    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)

    # 如果指定了特定 tags，只匯出那些
    if tags:
        export_tags = [t for t in tags if t in available_tags]
        if not export_tags:
            print(f"\nWarning: None of the specified tags found: {tags}")
            export_tags = available_tags
    else:
        export_tags = available_tags

    print(f"\nExporting {len(export_tags)} tags to: {output_dir}")

    # 匯出每個 tag 為獨立 CSV
    for tag in export_tags:
        tag_df = df[df['tag'] == tag][['step', 'value']].copy()
        tag_df = tag_df.sort_values('step')

        # 清理檔名 (移除特殊字元)
        safe_name = tag.replace('/', '_').replace('\\', '_')
        csv_path = os.path.join(output_dir, f"{safe_name}.csv")

        tag_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path} ({len(tag_df)} rows)")

    # 匯出合併版本
    all_csv_path = os.path.join(output_dir, "all_metrics.csv")
    df.to_csv(all_csv_path, index=False)
    print(f"\n  Saved combined: {all_csv_path}")

    return True


def read_with_event_accumulator(log_dir: str, output_dir: str, tags: list = None):
    """使用 tensorboard.backend.event_processing 讀取"""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        import pandas as pd
    except ImportError:
        print("tensorboard not found. Install with: pip install tensorboard pandas")
        return False

    print(f"Reading TensorBoard logs from: {log_dir}")

    # 載入 event 檔案
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # 取得所有 scalar tags
    available_tags = ea.Tags().get('scalars', [])

    if not available_tags:
        print("No scalar data found!")
        return False

    print(f"\nAvailable tags ({len(available_tags)}):")
    for tag in available_tags:
        count = len(ea.Scalars(tag))
        print(f"  - {tag} ({count} points)")

    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)

    # 如果指定了特定 tags，只匯出那些
    if tags:
        export_tags = [t for t in tags if t in available_tags]
        if not export_tags:
            print(f"\nWarning: None of the specified tags found: {tags}")
            export_tags = available_tags
    else:
        export_tags = available_tags

    print(f"\nExporting {len(export_tags)} tags to: {output_dir}")

    all_data = []

    # 匯出每個 tag 為獨立 CSV
    for tag in export_tags:
        events = ea.Scalars(tag)
        tag_df = pd.DataFrame([
            {'step': e.step, 'value': e.value, 'wall_time': e.wall_time}
            for e in events
        ])
        tag_df = tag_df.sort_values('step')

        # 清理檔名
        safe_name = tag.replace('/', '_').replace('\\', '_')
        csv_path = os.path.join(output_dir, f"{safe_name}.csv")

        tag_df[['step', 'value']].to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path} ({len(tag_df)} rows)")

        # 加入合併資料
        tag_df['tag'] = tag
        all_data.append(tag_df)

    # 匯出合併版本
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        all_csv_path = os.path.join(output_dir, "all_metrics.csv")
        all_df.to_csv(all_csv_path, index=False)
        print(f"\n  Saved combined: {all_csv_path}")

    return True


def find_event_files(log_dir: str) -> list:
    """找到所有 event 檔案"""
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, f))
    return event_files


def main():
    parser = argparse.ArgumentParser(description='Read TensorBoard logs and export to CSV')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Path to TensorBoard log directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for CSV files (default: log_dir/exported)')
    parser.add_argument('--tags', nargs='+', default=None,
                        help='Specific tags to export (default: all)')
    parser.add_argument('--method', type=str, choices=['tbparse', 'tensorboard'], default='tbparse',
                        help='Method to read logs (default: tbparse)')

    args = parser.parse_args()

    # 驗證路徑
    if not os.path.exists(args.log_dir):
        print(f"Error: Log directory not found: {args.log_dir}")
        return 1

    # 找 event 檔案
    event_files = find_event_files(args.log_dir)
    if not event_files:
        print(f"Error: No event files found in: {args.log_dir}")
        print("Looking for files like: events.out.tfevents.*")
        return 1

    print(f"Found {len(event_files)} event file(s):")
    for f in event_files[:5]:  # 只顯示前 5 個
        print(f"  - {f}")
    if len(event_files) > 5:
        print(f"  ... and {len(event_files) - 5} more")

    # 設定輸出目錄
    output_dir = args.output or os.path.join(args.log_dir, 'exported')

    # 讀取並匯出
    if args.method == 'tbparse':
        success = read_with_tbparse(args.log_dir, output_dir, args.tags)
        if not success:
            print("\nFalling back to tensorboard method...")
            success = read_with_event_accumulator(args.log_dir, output_dir, args.tags)
    else:
        success = read_with_event_accumulator(args.log_dir, output_dir, args.tags)

    if success:
        print("\nDone!")
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit(main())

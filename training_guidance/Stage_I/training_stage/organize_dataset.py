"""
PIDS 數據集整理腳本（精簡版）
只保留訓練必要的 EXR 檔案

輸出結構:
    dataset/
    ├── stereo_pairs/     # left_parallel.exr, right_cross.exr
    ├── ground_truth/     # disparity.exr
    └── masks/            # glass_mask.exr

Usage:
    python organize_dataset.py --input_dir ./output --output_dir ./dataset
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from typing import Set
from collections import defaultdict


def parse_quality_report(report_path: Path) -> Set[str]:
    """解析 quality_report.md，提取不合格場景名稱"""
    failed_scenes = set()

    if not report_path.exists():
        print(f"Warning: Quality report not found: {report_path}")
        return failed_scenes

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 尋找表格中標記為 ✗ 的場景
    pattern = r'\|\s*(scene_\d+)\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*✗\s*\|'
    matches = re.findall(pattern, content)
    for scene_name in matches:
        failed_scenes.add(scene_name)

    # 也檢查詳細報告區塊
    pattern2 = r'###\s+(scene_\d+)\s+✗'
    matches2 = re.findall(pattern2, content)
    for scene_name in matches2:
        failed_scenes.add(scene_name)

    print(f"Found {len(failed_scenes)} failed scenes in quality report")
    return failed_scenes


def get_scene_name(filename: str) -> str:
    """從檔名提取場景名稱"""
    match = re.match(r'(scene_\d+)', filename)
    return match.group(1) if match else None


def is_training_file(filename: str) -> tuple:
    """
    判斷是否為訓練必要檔案

    Returns:
        (category, bool): 類別名稱和是否保留
    """
    # 只保留這些 EXR 檔案
    if filename.endswith('.exr'):
        if '_left_parallel.exr' in filename or '_right_cross.exr' in filename:
            return ('stereo_pairs', True)
        if '_disparity.exr' in filename:
            return ('disparity', True)
        if '_glass_mask.exr' in filename:
            return ('masks', True)

    return (None, False)


def find_quality_report(input_dir: Path, report_path: Path = None) -> Path:
    """尋找 quality_report.md"""
    # 優先使用指定路徑
    if report_path and report_path.exists():
        return report_path

    # 搜尋常見位置
    search_paths = [
        input_dir / 'quality_report.md',
        input_dir.parent / 'quality_report.md',
        input_dir.parent / 'Quality_Assurance' / 'quality_report.md',
        input_dir / 'Quality_Assurance' / 'quality_report.md',
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def organize_dataset(
    input_dir: Path,
    output_dir: Path,
    report_path: Path = None,
    dry_run: bool = False,
    copy_mode: bool = False,
):
    """整理數據集，只保留訓練必要檔案"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    # 尋找品質報告
    report_path = find_quality_report(input_dir, report_path)

    if report_path:
        print(f"Found quality report: {report_path}")
        failed_scenes = parse_quality_report(report_path)
    else:
        print("Warning: No quality_report.md found, no scenes will be filtered!")
        print("  Use --report to specify path")
        failed_scenes = set()

    # 創建輸出目錄 (與 pids_dataset.py 一致)
    subdirs = {
        'stereo_pairs': output_dir / 'stereo_pairs',
        'disparity': output_dir / 'ground_truth',  # pids_dataset.py 期望 ground_truth/
        'masks': output_dir / 'masks',
    }

    if not dry_run:
        for subdir in subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)

    # 統計
    stats = defaultdict(int)
    skipped = 0
    failed_count = 0

    print(f"\n{'=' * 60}")
    print(f"PIDS Dataset Organizer (Training Files Only)")
    print(f"{'=' * 60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode:   {'Copy' if copy_mode else 'Move'}")
    print(f"{'=' * 60}\n")

    # 處理每個檔案
    for filepath in sorted(input_dir.iterdir()):
        if not filepath.is_file():
            continue

        filename = filepath.name
        scene_name = get_scene_name(filename)
        category, keep = is_training_file(filename)

        # 跳過非訓練檔案
        if not keep:
            skipped += 1
            continue

        # 跳過不合格場景
        if scene_name and scene_name in failed_scenes:
            failed_count += 1
            if dry_run:
                print(f"  [SKIP FAILED] {filename}")
            continue

        target_dir = subdirs[category]
        target_path = target_dir / filename

        if dry_run:
            print(f"  {category}: {filename}")
        else:
            if copy_mode:
                shutil.copy2(filepath, target_path)
            else:
                shutil.move(filepath, target_path)

        stats[category] += 1

    # 顯示統計
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    total = sum(stats.values())
    scenes = stats['stereo_pairs'] // 2  # 每個場景有 2 個 stereo 檔案

    print(f"\nKept (training files):")
    for category in ['stereo_pairs', 'disparity', 'masks']:
        count = stats[category]
        folder = 'ground_truth' if category == 'disparity' else category
        if count > 0:
            print(f"  {folder:15s}: {count:4d} files")

    print(f"\nSkipped:")
    print(f"  Non-training files: {skipped}")
    print(f"  Failed scenes:      {failed_count}")

    print(f"\nTotal: {total} files ({scenes} scenes)")

    # 複製 quality_report.md 到輸出目錄
    if report_path and report_path.exists() and not dry_run:
        shutil.copy2(report_path, output_dir / 'quality_report.md')
        print(f"\nCopied quality_report.md to output")

    if not dry_run:
        print(f"\nOutput structure:")
        for name, path in subdirs.items():
            if path.exists():
                count = len(list(path.glob('*.exr')))
                print(f"  {path.name}/  ({count} EXR files)")


def main():
    parser = argparse.ArgumentParser(
        description='PIDS 數據集整理（只保留訓練必要 EXR）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 預覽
  python organize_dataset.py --input_dir ./output --dry_run

  # 整理（移動檔案）
  python organize_dataset.py --input_dir ./output --output_dir ./dataset

  # 整理（複製檔案）
  python organize_dataset.py --input_dir ./output --output_dir ./dataset --copy

Output structure:
  dataset/
  ├── stereo_pairs/     # *_left_parallel.exr, *_right_cross.exr
  ├── ground_truth/     # *_disparity.exr
  └── masks/            # *_glass_mask.exr
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='輸入目錄（渲染輸出）')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                        help='輸出目錄')
    parser.add_argument('--report', type=str, default=None,
                        help='quality_report.md 路徑')
    parser.add_argument('--dry_run', action='store_true',
                        help='預覽模式')
    parser.add_argument('--copy', action='store_true',
                        help='複製而非移動')

    args = parser.parse_args()

    report_path = Path(args.report) if args.report else None

    organize_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        report_path=report_path,
        dry_run=args.dry_run,
        copy_mode=args.copy,
    )

    if args.dry_run:
        print("\n[Dry run - no files moved]")


if __name__ == '__main__':
    main()

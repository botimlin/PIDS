"""
PIDS 數據集整理腳本（精簡版）
只保留訓練必要的 EXR 檔案，支援訓練/測試分割

輸出結構:
    dataset/
    ├── train/
    │   ├── stereo_pairs/     # left_parallel.exr, right_cross.exr
    │   ├── ground_truth/     # disparity.exr
    │   └── masks/            # glass_mask.exr
    └── test/
        ├── stereo_pairs/
        ├── ground_truth/
        └── masks/

Usage:
    python organize_dataset.py --input_dir ./output --output_dir ./dataset --train_size 3500

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import os
import re
import random
import shutil
import argparse
from pathlib import Path
from typing import Set, List
from collections import defaultdict


def parse_quality_report(report_path: Path) -> Set[str]:
    """解析 quality_report.md，提取不合格場景名稱"""
    failed_scenes = set()

    if not report_path.exists():
        print(f"Warning: Quality report not found: {report_path}")
        return failed_scenes

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 尋找表格中標記為 ✗ 的場景（匹配包含 ✗ 的任何行）
    # V5 格式: | scene_0001 | 🟢 excellent | 85 | ... | 95.0% ✗ | 1.05x ✓ | ... |
    for line in content.split('\n'):
        if '✗' in line:
            match = re.search(r'\|\s*(scene_\d+)\s*\|', line)
            if match:
                failed_scenes.add(match.group(1))

    # 也檢查失敗場景區塊的表格
    # 格式: | scene_0001 | 失敗原因 |
    fail_section = re.search(r'## 失敗場景.*?\n\n(.*?)(?=\n---|\n##|\Z)', content, re.DOTALL)
    if fail_section:
        fail_table = fail_section.group(1)
        for match in re.finditer(r'\|\s*(scene_\d+)\s*\|', fail_table):
            failed_scenes.add(match.group(1))

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
        # pol: _left_parallel.exr, _right_cross.exr
        # nopol: _left.exr, _right.exr
        if '_left_parallel.exr' in filename or '_right_cross.exr' in filename:
            return ('stereo_pairs', True)
        if filename.endswith('_left.exr') or filename.endswith('_right.exr'):
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


def collect_valid_scenes(input_dir: Path, failed_scenes: Set[str]) -> List[str]:
    """收集所有合格場景名稱"""
    scenes = set()
    for filepath in input_dir.iterdir():
        if not filepath.is_file():
            continue
        scene_name = get_scene_name(filepath.name)
        if scene_name and scene_name not in failed_scenes:
            _, keep = is_training_file(filepath.name)
            if keep:
                scenes.add(scene_name)
    return sorted(scenes)


def organize_dataset(
    input_dir: Path,
    output_dir: Path,
    report_path: Path = None,
    dry_run: bool = False,
    copy_mode: bool = False,
    train_size: int = None,
    seed: int = 42,
    scene_list_dir: Path = None,
):
    """整理數據集，只保留訓練必要檔案，支援訓練/測試分割"""
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

    # 使用指定的場景列表（用於對齊不同數據集）
    if scene_list_dir:
        scene_list_dir = Path(scene_list_dir)
        train_list_file = scene_list_dir / 'train_scenes.txt'
        test_list_file = scene_list_dir / 'test_scenes.txt'

        if not train_list_file.exists():
            print(f"Error: train_scenes.txt not found in {scene_list_dir}")
            return

        train_scenes = set(train_list_file.read_text().strip().split('\n'))
        test_scenes = set(test_list_file.read_text().strip().split('\n')) if test_list_file.exists() else set()

        print(f"Using scene list from: {scene_list_dir}")
        print(f"  Train scenes: {len(train_scenes)}")
        print(f"  Test scenes: {len(test_scenes)}")

        # 檢查輸入目錄中有多少場景可用
        available_scenes = set()
        for filepath in input_dir.iterdir():
            if filepath.is_file():
                scene_name = get_scene_name(filepath.name)
                if scene_name:
                    available_scenes.add(scene_name)

        train_available = train_scenes & available_scenes
        test_available = test_scenes & available_scenes

        if len(train_available) < len(train_scenes):
            missing = len(train_scenes) - len(train_available)
            print(f"  Warning: {missing} train scenes not found in input")
        if len(test_available) < len(test_scenes):
            missing = len(test_scenes) - len(test_available)
            print(f"  Warning: {missing} test scenes not found in input")

        train_scenes = train_available
        test_scenes = test_available
    else:
        # 收集所有合格場景
        all_scenes = collect_valid_scenes(input_dir, failed_scenes)
        total_scenes = len(all_scenes)
        print(f"Total valid scenes: {total_scenes}")

        # 分割訓練/測試集
        if train_size and train_size < total_scenes:
            random.seed(seed)
            random.shuffle(all_scenes)
            train_scenes = set(all_scenes[:train_size])
            test_scenes = set(all_scenes[train_size:])
            print(f"Train/Test split: {len(train_scenes)} / {len(test_scenes)} (seed={seed})")
        else:
            train_scenes = set(all_scenes)
            test_scenes = set()
            if train_size:
                print(f"Warning: train_size ({train_size}) >= total scenes ({total_scenes}), no test split")

    # 創建輸出目錄結構
    if test_scenes:
        # 有分割時使用 train/ 和 test/ 子目錄
        train_subdirs = {
            'stereo_pairs': output_dir / 'train' / 'stereo_pairs',
            'disparity': output_dir / 'train' / 'ground_truth',
            'masks': output_dir / 'train' / 'masks',
        }
        test_subdirs = {
            'stereo_pairs': output_dir / 'test' / 'stereo_pairs',
            'disparity': output_dir / 'test' / 'ground_truth',
            'masks': output_dir / 'test' / 'masks',
        }
    else:
        # 無分割時直接使用根目錄
        train_subdirs = {
            'stereo_pairs': output_dir / 'stereo_pairs',
            'disparity': output_dir / 'ground_truth',
            'masks': output_dir / 'masks',
        }
        test_subdirs = {}

    if not dry_run:
        for subdir in train_subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        for subdir in test_subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)

    # 統計
    train_stats = defaultdict(int)
    test_stats = defaultdict(int)
    skipped = 0
    failed_count = 0

    print(f"\n{'=' * 60}")
    print(f"PIDS Dataset Organizer (Training Files Only)")
    print(f"{'=' * 60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode:   {'Copy' if copy_mode else 'Move'}")
    if train_size:
        print(f"Train size: {train_size}")
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

        # 決定是訓練還是測試
        if scene_name in train_scenes:
            target_dir = train_subdirs[category]
            stats = train_stats
            split_label = "train"
        elif scene_name in test_scenes:
            target_dir = test_subdirs[category]
            stats = test_stats
            split_label = "test"
        else:
            continue

        target_path = target_dir / filename

        if dry_run:
            print(f"  [{split_label}] {category}: {filename}")
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

    train_total = sum(train_stats.values())
    train_scene_count = train_stats['stereo_pairs'] // 2

    print(f"\nTrain set ({train_scene_count} scenes, {train_total} files):")
    for category in ['stereo_pairs', 'disparity', 'masks']:
        count = train_stats[category]
        folder = 'ground_truth' if category == 'disparity' else category
        if count > 0:
            print(f"  {folder:15s}: {count:4d} files")

    if test_scenes:
        test_total = sum(test_stats.values())
        test_scene_count = test_stats['stereo_pairs'] // 2
        print(f"\nTest set ({test_scene_count} scenes, {test_total} files):")
        for category in ['stereo_pairs', 'disparity', 'masks']:
            count = test_stats[category]
            folder = 'ground_truth' if category == 'disparity' else category
            if count > 0:
                print(f"  {folder:15s}: {count:4d} files")

    print(f"\nSkipped:")
    print(f"  Non-training files: {skipped}")
    print(f"  Failed scenes:      {failed_count}")

    # 複製 quality_report.md 到輸出目錄
    if report_path and report_path.exists() and not dry_run:
        shutil.copy2(report_path, output_dir / 'quality_report.md')
        print(f"\nCopied quality_report.md to output")

    # 輸出場景列表（方便追溯）
    if not dry_run and test_scenes:
        with open(output_dir / 'train_scenes.txt', 'w') as f:
            for scene in sorted(train_scenes):
                f.write(f"{scene}\n")
        with open(output_dir / 'test_scenes.txt', 'w') as f:
            for scene in sorted(test_scenes):
                f.write(f"{scene}\n")
        print(f"Saved train_scenes.txt and test_scenes.txt")

    if not dry_run:
        print(f"\nOutput structure:")
        if test_scenes:
            print(f"  train/")
            for name, path in train_subdirs.items():
                if path.exists():
                    count = len(list(path.glob('*.exr')))
                    print(f"    {path.name}/  ({count} EXR files)")
            print(f"  test/")
            for name, path in test_subdirs.items():
                if path.exists():
                    count = len(list(path.glob('*.exr')))
                    print(f"    {path.name}/  ({count} EXR files)")
        else:
            for name, path in train_subdirs.items():
                if path.exists():
                    count = len(list(path.glob('*.exr')))
                    print(f"  {path.name}/  ({count} EXR files)")


def main():
    parser = argparse.ArgumentParser(
        description='PIDS 數據集整理（只保留訓練必要 EXR），支援訓練/測試分割',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 預覽
  python organize_dataset.py --input_dir ./output --dry_run

  # 整理（全部作為訓練集）
  python organize_dataset.py --input_dir ./output --output_dir ./dataset

  # 整理並分割（3500訓練，剩餘測試）
  python organize_dataset.py --input_dir ./output --output_dir ./dataset --train_size 3500

  # 複製模式（保留原始檔案）
  python organize_dataset.py --input_dir ./output --output_dir ./dataset --train_size 3500 --copy

Output structure (with split):
  dataset/
  ├── train/
  │   ├── stereo_pairs/     # *_left_parallel.exr, *_right_cross.exr
  │   ├── ground_truth/     # *_disparity.exr
  │   └── masks/            # *_glass_mask.exr
  ├── test/
  │   ├── stereo_pairs/
  │   ├── ground_truth/
  │   └── masks/
  ├── train_scenes.txt      # 訓練場景列表
  └── test_scenes.txt       # 測試場景列表
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='輸入目錄（渲染輸出）')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                        help='輸出目錄')
    parser.add_argument('--report', type=str, default=None,
                        help='quality_report.md 路徑')
    parser.add_argument('--train_size', type=int, default=None,
                        help='訓練集場景數量（剩餘作為測試集）')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子（預設: 42）')
    parser.add_argument('--dry_run', action='store_true',
                        help='預覽模式')
    parser.add_argument('--copy', action='store_true',
                        help='複製而非移動')
    parser.add_argument('--scene_list', type=str, default=None,
                        help='場景列表目錄（包含 train_scenes.txt 和 test_scenes.txt，用於對齊不同數據集）')

    args = parser.parse_args()

    report_path = Path(args.report) if args.report else None

    scene_list_dir = Path(args.scene_list) if args.scene_list else None

    organize_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        report_path=report_path,
        dry_run=args.dry_run,
        copy_mode=args.copy,
        train_size=args.train_size,
        seed=args.seed,
        scene_list_dir=scene_list_dir,
    )

    if args.dry_run:
        print("\n[Dry run - no files moved]")


if __name__ == '__main__':
    main()

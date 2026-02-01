"""
PIDS 數據集整理腳本（百分比分割版）
按 5%, 10%, 25%, 50%, 75%, 100% 分別複製訓練場景
支援 pol (偏振) 和 nopol (無偏振) 兩種模式

檔案命名差異:
  pol:   *_left_parallel.exr, *_right_cross.exr
  nopol: *_left.exr, *_right.exr

輸出結構:
    dataset_pol/                    或 dataset_nopol/
    ├── train_5pct/
    │   ├── stereo_pairs/
    │   ├── ground_truth/
    │   └── masks/
    ├── train_10pct/
    ├── train_25pct/
    ├── train_50pct/
    ├── train_75pct/
    ├── train_100pct/
    └── test/
        ├── stereo_pairs/
        ├── ground_truth/
        └── masks/

Usage:
    # Pol 模式
    python organize_dataset_percentages.py --input_dir ./output_pol --output_dir ./dataset_pol --train_size 3500 --mode pol

    # Nopol 模式
    python organize_dataset_percentages.py --input_dir ./output_nopol --output_dir ./dataset_nopol --train_size 3500 --mode nopol

    # 自動偵測模式
    python organize_dataset_percentages.py --input_dir ./output --output_dir ./dataset --train_size 3500

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# 百分比設定
PERCENTAGES = [5, 10, 25, 50, 75, 100]


def parse_quality_report(report_path: Path) -> Set[str]:
    """解析 quality_report.md 或 failed_scenes.txt，提取不合格場景名稱"""
    failed_scenes = set()

    # 優先讀取 failed_scenes.txt（pids_qa.py v2.3.0+ 生成）
    failed_txt = report_path.parent / "failed_scenes.txt"
    if failed_txt.exists():
        print(f"Found failed_scenes.txt: {failed_txt}")
        with open(failed_txt, 'r', encoding='utf-8') as f:
            for line in f:
                scene_name = line.strip()
                if scene_name:
                    failed_scenes.add(scene_name)
        print(f"Found {len(failed_scenes)} failed scenes in failed_scenes.txt")
        return failed_scenes

    # 退回到解析 quality_report.md
    if not report_path.exists():
        print(f"Warning: Quality report not found: {report_path}")
        return failed_scenes

    print(f"Parsing quality_report.md (legacy mode)...")
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    for line in content.split('\n'):
        if '✗' in line:
            match = re.search(r'\|\s*(scene_\d+)\s*\|', line)
            if match:
                failed_scenes.add(match.group(1))

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


def detect_mode(input_dir: Path) -> str:
    """
    自動偵測數據集模式 (pol 或 nopol)

    pol:   *_left_parallel.exr, *_right_cross.exr
    nopol: *_left.exr, *_right.exr
    """
    for filepath in input_dir.iterdir():
        if not filepath.is_file():
            continue
        name = filepath.name
        if '_left_parallel.exr' in name or '_right_cross.exr' in name:
            return 'pol'
        if name.endswith('_left.exr') or name.endswith('_right.exr'):
            # 確認不是 _left_parallel 的誤判
            if '_parallel' not in name and '_cross' not in name:
                return 'nopol'
    return 'unknown'


def is_training_file(filename: str, mode: str = 'auto') -> tuple:
    """
    判斷是否為訓練必要檔案

    Args:
        filename: 檔案名稱
        mode: 'pol', 'nopol', 或 'auto' (兩者都接受)

    Returns:
        (category, bool): 類別名稱和是否保留
    """
    if filename.endswith('.exr'):
        # Pol 模式: _left_parallel.exr, _right_cross.exr
        if mode in ('pol', 'auto'):
            if '_left_parallel.exr' in filename or '_right_cross.exr' in filename:
                return ('stereo_pairs', True)

        # Nopol 模式: _left.exr, _right.exr (但不是 _left_parallel)
        if mode in ('nopol', 'auto'):
            if filename.endswith('_left.exr') and '_parallel' not in filename:
                return ('stereo_pairs', True)
            if filename.endswith('_right.exr') and '_cross' not in filename:
                return ('stereo_pairs', True)

        # 共用檔案
        if '_disparity.exr' in filename:
            return ('disparity', True)
        if '_glass_mask.exr' in filename and '_strict' not in filename:
            return ('masks', True)

    return (None, False)


def find_quality_report(input_dir: Path, report_path: Path = None) -> Path:
    """尋找 quality_report.md"""
    if report_path and report_path.exists():
        return report_path

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


def collect_valid_scenes(input_dir: Path, failed_scenes: Set[str], mode: str = 'auto') -> List[str]:
    """收集所有合格場景名稱"""
    scenes = set()
    for filepath in input_dir.iterdir():
        if not filepath.is_file():
            continue
        scene_name = get_scene_name(filepath.name)
        if scene_name and scene_name not in failed_scenes:
            _, keep = is_training_file(filepath.name, mode)
            if keep:
                scenes.add(scene_name)
    return sorted(scenes)


def copy_file(src: Path, dst: Path):
    """複製單個檔案"""
    shutil.copy2(src, dst)
    return src.name


def organize_dataset_percentages(
    input_dir: Path,
    output_dir: Path,
    report_path: Path = None,
    dry_run: bool = False,
    train_size: int = None,
    seed: int = 42,
    scene_list_dir: Path = None,
    num_workers: int = 8,
    mode: str = 'auto',
):
    """整理數據集，按百分比分割訓練集"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    # 自動偵測或使用指定模式
    if mode == 'auto':
        detected_mode = detect_mode(input_dir)
        if detected_mode == 'unknown':
            print("Warning: Cannot detect mode (pol/nopol), using 'auto'")
            mode = 'auto'
        else:
            mode = detected_mode
            print(f"Detected mode: {mode}")
    else:
        print(f"Using mode: {mode}")

    # 尋找品質報告
    report_path = find_quality_report(input_dir, report_path)

    if report_path:
        print(f"Found quality report: {report_path}")
        failed_scenes = parse_quality_report(report_path)
    else:
        print("Warning: No quality_report.md found, no scenes will be filtered!")
        failed_scenes = set()

    # 使用指定的場景列表
    if scene_list_dir:
        scene_list_dir = Path(scene_list_dir)
        train_list_file = scene_list_dir / 'train_scenes.txt'
        test_list_file = scene_list_dir / 'test_scenes.txt'

        if not train_list_file.exists():
            print(f"Error: train_scenes.txt not found in {scene_list_dir}")
            return

        train_scenes = list(train_list_file.read_text().strip().split('\n'))
        test_scenes = list(test_list_file.read_text().strip().split('\n')) if test_list_file.exists() else []

        print(f"Using scene list from: {scene_list_dir}")
        print(f"  Train scenes: {len(train_scenes)}")
        print(f"  Test scenes: {len(test_scenes)}")

        # 檢查可用場景
        available_scenes = set()
        for filepath in input_dir.iterdir():
            if filepath.is_file():
                scene_name = get_scene_name(filepath.name)
                if scene_name:
                    available_scenes.add(scene_name)

        train_scenes = [s for s in train_scenes if s in available_scenes]
        test_scenes = [s for s in test_scenes if s in available_scenes]

    else:
        # 收集所有合格場景
        all_scenes = collect_valid_scenes(input_dir, failed_scenes, mode)
        total_scenes = len(all_scenes)
        print(f"Total valid scenes: {total_scenes}")

        # 分割訓練/測試集
        random.seed(seed)
        random.shuffle(all_scenes)

        if train_size and train_size < total_scenes:
            train_scenes = all_scenes[:train_size]
            test_scenes = all_scenes[train_size:]
            print(f"Train/Test split: {len(train_scenes)} / {len(test_scenes)} (seed={seed})")
        else:
            train_scenes = all_scenes
            test_scenes = []
            if train_size:
                print(f"Warning: train_size ({train_size}) >= total scenes ({total_scenes}), no test split")

    # 計算各百分比的場景數量
    total_train = len(train_scenes)
    percentage_scenes = {}

    for pct in PERCENTAGES:
        count = max(1, int(total_train * pct / 100))
        if pct == 100:
            count = total_train
        percentage_scenes[pct] = train_scenes[:count]

    print(f"\n{'=' * 60}")
    print(f"PIDS Dataset Organizer (Percentage Split)")
    print(f"{'=' * 60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode:   {mode}")
    print(f"Total train scenes: {total_train}")
    print(f"\nPercentage splits:")
    for pct in PERCENTAGES:
        print(f"  {pct:3d}%: {len(percentage_scenes[pct]):5d} scenes")
    if test_scenes:
        print(f"  Test: {len(test_scenes):5d} scenes")
    print(f"{'=' * 60}\n")

    if dry_run:
        print("[Dry run - no files will be copied]")
        return

    # 建立目錄結構
    subdirs = ['stereo_pairs', 'ground_truth', 'masks']

    for pct in PERCENTAGES:
        for subdir in subdirs:
            (output_dir / f'train_{pct}pct' / subdir).mkdir(parents=True, exist_ok=True)

    if test_scenes:
        for subdir in subdirs:
            (output_dir / 'test' / subdir).mkdir(parents=True, exist_ok=True)

    # 收集所有要複製的檔案
    category_map = {
        'stereo_pairs': 'stereo_pairs',
        'disparity': 'ground_truth',
        'masks': 'masks',
    }

    copy_tasks = []  # (src, dst, pct_label)

    # 處理各百分比的訓練集
    for pct in PERCENTAGES:
        scenes_set = set(percentage_scenes[pct])

        for filepath in input_dir.iterdir():
            if not filepath.is_file():
                continue

            filename = filepath.name
            scene_name = get_scene_name(filename)
            category, keep = is_training_file(filename, mode)

            if not keep or not scene_name:
                continue

            if scene_name not in scenes_set:
                continue

            if scene_name in failed_scenes:
                continue

            target_subdir = category_map[category]
            target_path = output_dir / f'train_{pct}pct' / target_subdir / filename
            copy_tasks.append((filepath, target_path, f'{pct}%'))

    # 處理測試集
    if test_scenes:
        test_set = set(test_scenes)

        for filepath in input_dir.iterdir():
            if not filepath.is_file():
                continue

            filename = filepath.name
            scene_name = get_scene_name(filename)
            category, keep = is_training_file(filename, mode)

            if not keep or not scene_name:
                continue

            if scene_name not in test_set:
                continue

            if scene_name in failed_scenes:
                continue

            target_subdir = category_map[category]
            target_path = output_dir / 'test' / target_subdir / filename
            copy_tasks.append((filepath, target_path, 'test'))

    # 並行複製檔案
    print(f"Copying {len(copy_tasks)} files with {num_workers} workers...")

    stats = defaultdict(int)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(copy_file, src, dst): (src, dst, label)
                   for src, dst, label in copy_tasks}

        with tqdm(total=len(copy_tasks), desc="Copying") as pbar:
            for future in as_completed(futures):
                src, dst, label = futures[future]
                try:
                    future.result()
                    stats[label] += 1
                except Exception as e:
                    print(f"Error copying {src}: {e}")
                pbar.update(1)

    # 顯示統計
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    for pct in PERCENTAGES:
        label = f'{pct}%'
        count = stats[label]
        scene_count = len(percentage_scenes[pct])
        print(f"  train_{pct:3d}pct/: {scene_count:5d} scenes, {count:6d} files")

    if test_scenes:
        test_count = stats['test']
        print(f"  test/:        {len(test_scenes):5d} scenes, {test_count:6d} files")

    # 複製 quality_report.md
    if report_path and report_path.exists():
        shutil.copy2(report_path, output_dir / 'quality_report.md')
        print(f"\nCopied quality_report.md to output")

    # 輸出場景列表
    for pct in PERCENTAGES:
        with open(output_dir / f'train_{pct}pct_scenes.txt', 'w') as f:
            for scene in sorted(percentage_scenes[pct]):
                f.write(f"{scene}\n")

    if test_scenes:
        with open(output_dir / 'test_scenes.txt', 'w') as f:
            for scene in sorted(test_scenes):
                f.write(f"{scene}\n")

    # 保存模式資訊
    with open(output_dir / 'dataset_info.txt', 'w') as f:
        f.write(f"mode: {mode}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"total_train: {total_train}\n")
        f.write(f"total_test: {len(test_scenes)}\n")
        for pct in PERCENTAGES:
            f.write(f"train_{pct}pct: {len(percentage_scenes[pct])}\n")

    print(f"Saved scene lists to output directory")
    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description='PIDS 數據集整理（按百分比分割訓練集）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 預覽 (自動偵測 pol/nopol)
  python organize_dataset_percentages.py --input_dir ./output --dry_run

  # Pol 模式 - 整理偏振數據集
  python organize_dataset_percentages.py --input_dir ./output_pol --output_dir ./dataset_pol --train_size 3500 --mode pol

  # Nopol 模式 - 整理無偏振數據集
  python organize_dataset_percentages.py --input_dir ./output_nopol --output_dir ./dataset_nopol --train_size 3500 --mode nopol

  # 使用現有場景列表 (確保 pol 和 nopol 使用相同場景分割)
  python organize_dataset_percentages.py --input_dir ./output_nopol --output_dir ./dataset_nopol --scene_list ./dataset_pol --mode nopol

File naming:
  pol:   *_left_parallel.exr, *_right_cross.exr
  nopol: *_left.exr, *_right.exr

Output structure:
  dataset_pol/                      或 dataset_nopol/
  ├── train_5pct/
  │   ├── stereo_pairs/     # pol: *_left_parallel.exr, *_right_cross.exr
  │   │                     # nopol: *_left.exr, *_right.exr
  │   ├── ground_truth/     # *_disparity.exr
  │   └── masks/            # *_glass_mask.exr
  ├── train_10pct/
  ├── train_25pct/
  ├── train_50pct/
  ├── train_75pct/
  ├── train_100pct/
  ├── test/
  │   ├── stereo_pairs/
  │   ├── ground_truth/
  │   └── masks/
  ├── train_5pct_scenes.txt
  ├── train_10pct_scenes.txt
  ├── ...
  └── test_scenes.txt
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
    parser.add_argument('--scene_list', type=str, default=None,
                        help='場景列表目錄（包含 train_scenes.txt 和 test_scenes.txt）')
    parser.add_argument('--workers', type=int, default=8,
                        help='並行複製的 worker 數量（預設: 8）')
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'pol', 'nopol'],
                        help='數據集模式: pol (偏振), nopol (無偏振), auto (自動偵測)')

    args = parser.parse_args()

    report_path = Path(args.report) if args.report else None
    scene_list_dir = Path(args.scene_list) if args.scene_list else None

    organize_dataset_percentages(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        report_path=report_path,
        dry_run=args.dry_run,
        train_size=args.train_size,
        seed=args.seed,
        scene_list_dir=scene_list_dir,
        num_workers=args.workers,
        mode=args.mode,
    )


if __name__ == '__main__':
    main()

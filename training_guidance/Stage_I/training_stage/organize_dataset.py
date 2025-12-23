"""
PIDS 數據集整理腳本
將渲染輸出整理成訓練所需的目錄結構

功能:
1. 將 JSON 報告移到 reports/ 資料夾
2. 將左右眼圖像移到 stereo_pairs/ 資料夾
3. 將深度/視差圖移到 depth/ 資料夾
4. 將玻璃遮罩移到 masks/ 資料夾
5. 將 DoLP 視覺化移到 visualization/ 資料夾
6. 讀取 quality_report.md，將不合格場景移到 failed/ 資料夾

Usage:
    python organize_dataset.py --input_dir ./output/output --output_dir ./organized
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from typing import List, Set, Dict
from collections import defaultdict


def parse_quality_report(report_path: Path) -> Set[str]:
    """
    解析 quality_report.md，提取不合格場景名稱

    Args:
        report_path: quality_report.md 路徑

    Returns:
        不合格場景名稱集合
    """
    failed_scenes = set()

    if not report_path.exists():
        print(f"Warning: Quality report not found: {report_path}")
        return failed_scenes

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 尋找表格中標記為 ✗ 的場景
    # 格式: | scene_0057 | - | ✓ | ✓ | ✗ | ✓ | 87 | ✗ |
    pattern = r'\|\s*(scene_\d+)\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*✗\s*\|'
    matches = re.findall(pattern, content)

    for scene_name in matches:
        failed_scenes.add(scene_name)

    # 也檢查詳細報告區塊
    # 格式: ### scene_0057 ✗
    pattern2 = r'###\s+(scene_\d+)\s+✗'
    matches2 = re.findall(pattern2, content)

    for scene_name in matches2:
        failed_scenes.add(scene_name)

    print(f"Found {len(failed_scenes)} failed scenes in quality report")
    if failed_scenes:
        print(f"  Failed: {sorted(failed_scenes)}")

    return failed_scenes


def categorize_file(filename: str) -> str:
    """
    根據檔名判斷檔案類別

    Returns:
        類別名稱: 'reports', 'stereo', 'depth', 'masks', 'visualization', 'other'
    """
    name_lower = filename.lower()

    # JSON 報告
    if filename.endswith('.json'):
        return 'reports'

    # 左右眼圖像 (EXR)
    if '_left_parallel.exr' in filename or '_right_cross.exr' in filename or '_right_parallel.exr' in filename:
        return 'stereo'

    # 深度和視差圖
    if '_depth.exr' in filename or '_disparity.exr' in filename:
        return 'depth'
    if '_depth.png' in filename:
        return 'depth'

    # 玻璃遮罩
    if '_glass_mask' in filename or '_mask.exr' in filename or '_mask.png' in filename:
        return 'masks'

    # DoLP 視覺化
    if '_DoLP' in filename or '_polarization_diff' in filename:
        return 'visualization'

    # 左右眼 PNG 預覽
    if '_left_parallel.png' in filename or '_right_cross.png' in filename:
        return 'visualization'

    return 'other'


def get_scene_name(filename: str) -> str:
    """從檔名提取場景名稱"""
    # 匹配 scene_XXXX 格式
    match = re.match(r'(scene_\d+)', filename)
    if match:
        return match.group(1)
    return None


def organize_dataset(
    input_dir: Path,
    output_dir: Path,
    report_path: Path = None,
    dry_run: bool = False,
    copy_mode: bool = False,
):
    """
    整理數據集

    Args:
        input_dir: 輸入目錄
        output_dir: 輸出目錄
        report_path: quality_report.md 路徑
        dry_run: 只顯示操作，不實際執行
        copy_mode: 使用複製而非移動
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    # 解析品質報告
    if report_path is None:
        # 嘗試在上層目錄找 quality_report.md
        report_path = input_dir.parent / 'quality_report.md'

    failed_scenes = parse_quality_report(report_path)

    # 創建輸出目錄結構
    subdirs = {
        'reports': output_dir / 'reports',
        'stereo': output_dir / 'stereo_pairs',
        'depth': output_dir / 'ground_truth',
        'masks': output_dir / 'masks',
        'visualization': output_dir / 'visualization',
        'other': output_dir / 'other',
        'failed': output_dir / 'failed_scenes',
    }

    if not dry_run:
        for subdir in subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)

    # 統計
    stats = defaultdict(int)
    file_list = list(input_dir.iterdir())
    total_files = len([f for f in file_list if f.is_file()])

    print(f"\nProcessing {total_files} files...")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode:   {'Copy' if copy_mode else 'Move'}")
    print(f"Dry run: {dry_run}")
    print("-" * 60)

    # 處理每個檔案
    for filepath in sorted(file_list):
        if not filepath.is_file():
            continue

        filename = filepath.name
        scene_name = get_scene_name(filename)
        category = categorize_file(filename)

        # 判斷是否為不合格場景
        is_failed = scene_name in failed_scenes if scene_name else False

        # 決定目標目錄
        if is_failed:
            # 不合格場景：按類別放入 failed/ 子目錄
            target_dir = subdirs['failed'] / category
        else:
            target_dir = subdirs[category]

        target_path = target_dir / filename

        # 執行操作
        if dry_run:
            status = "FAILED -> " if is_failed else ""
            print(f"  {status}{category}: {filename}")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)

            if copy_mode:
                shutil.copy2(filepath, target_path)
            else:
                shutil.move(filepath, target_path)

        # 更新統計
        if is_failed:
            stats[f'failed_{category}'] += 1
        else:
            stats[category] += 1

    # 複製 quality_report.md 到輸出目錄
    if report_path.exists() and not dry_run:
        shutil.copy2(report_path, output_dir / 'quality_report.md')

    # 顯示統計
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("\nPassed scenes:")
    for category in ['stereo', 'depth', 'masks', 'reports', 'visualization', 'other']:
        if stats[category] > 0:
            print(f"  {category:15s}: {stats[category]:4d} files -> {subdirs[category].name}/")

    failed_total = sum(v for k, v in stats.items() if k.startswith('failed_'))
    if failed_total > 0:
        print(f"\nFailed scenes ({len(failed_scenes)} scenes):")
        for category in ['stereo', 'depth', 'masks', 'reports', 'visualization', 'other']:
            key = f'failed_{category}'
            if stats[key] > 0:
                print(f"  {category:15s}: {stats[key]:4d} files -> failed_scenes/{category}/")

    total_processed = sum(stats.values())
    print(f"\nTotal: {total_processed} files processed")

    if not dry_run:
        print(f"\nOutput directory structure:")
        for name, path in subdirs.items():
            if path.exists() and any(path.iterdir()):
                count = len(list(path.rglob('*')))
                print(f"  {path.relative_to(output_dir)}/  ({count} items)")


def main():
    parser = argparse.ArgumentParser(
        description='PIDS 數據集整理腳本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 預覽整理結果（不實際移動）
  python organize_dataset.py --input_dir ./output/output --dry_run

  # 整理到新目錄（移動檔案）
  python organize_dataset.py --input_dir ./output/output --output_dir ./organized

  # 複製模式（保留原始檔案）
  python organize_dataset.py --input_dir ./output/output --output_dir ./organized --copy
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='輸入目錄（包含渲染輸出的目錄）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='輸出目錄（預設為 input_dir 同層的 organized/）')
    parser.add_argument('--report', type=str, default=None,
                        help='quality_report.md 路徑（預設自動尋找）')
    parser.add_argument('--dry_run', action='store_true',
                        help='只顯示操作，不實際執行')
    parser.add_argument('--copy', action='store_true',
                        help='複製檔案而非移動（保留原始檔案）')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # 設定輸出目錄
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 預設輸出到 training_stage/dataset
        script_dir = Path(__file__).parent
        output_dir = script_dir / 'dataset'

    # 設定報告路徑
    report_path = Path(args.report) if args.report else None

    print("=" * 60)
    print("PIDS Dataset Organizer")
    print("=" * 60)

    organize_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        report_path=report_path,
        dry_run=args.dry_run,
        copy_mode=args.copy,
    )

    if args.dry_run:
        print("\n[Dry run mode - no files were moved]")
        print("Remove --dry_run to actually organize files")


if __name__ == '__main__':
    main()

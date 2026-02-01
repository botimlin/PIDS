"""
PIDS 場景刪除腳本
根據 scenes.txt 刪除所有符合場景名稱的檔案

Usage:
    # 預覽要刪除的檔案
    python delete_scenes.py --scenes_txt failed_scenes.txt --input_dir ./output --dry_run

    # 實際刪除
    python delete_scenes.py --scenes_txt failed_scenes.txt --input_dir ./output

    # 從 quality_report.md 自動提取失敗場景並刪除
    python delete_scenes.py --quality_report ./output/quality_report.md --input_dir ./output

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import os
import re
import argparse
from pathlib import Path
from typing import Set, List


def parse_scenes_txt(scenes_txt: Path) -> Set[str]:
    """讀取 scenes.txt，每行一個場景名稱"""
    if not scenes_txt.exists():
        print(f"Error: File not found: {scenes_txt}")
        return set()

    scenes = set()
    with open(scenes_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # 支援 scene_0001 或純數字 0001
                if line.startswith('scene_'):
                    scenes.add(line)
                elif line.isdigit():
                    scenes.add(f"scene_{int(line):04d}")
                else:
                    scenes.add(line)
    return scenes


def parse_quality_report(report_path: Path) -> Set[str]:
    """從 quality_report.md 提取失敗場景"""
    if not report_path.exists():
        print(f"Error: File not found: {report_path}")
        return set()

    failed_scenes = set()

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 尋找表格中標記為 ✗ 的場景
    for line in content.split('\n'):
        if '✗' in line:
            match = re.search(r'\|\s*(scene_\d+)\s*\|', line)
            if match:
                failed_scenes.add(match.group(1))

    # 也檢查失敗場景區塊
    fail_section = re.search(r'## 失敗場景.*?\n\n(.*?)(?=\n---|\n##|\Z)', content, re.DOTALL)
    if fail_section:
        fail_table = fail_section.group(1)
        for match in re.finditer(r'\|\s*(scene_\d+)\s*\|', fail_table):
            failed_scenes.add(match.group(1))

    return failed_scenes


def find_files_to_delete(input_dir: Path, scenes: Set[str]) -> List[Path]:
    """找出所有要刪除的檔案"""
    files_to_delete = []

    for filepath in input_dir.iterdir():
        if not filepath.is_file():
            continue

        filename = filepath.name
        # 提取場景名稱
        match = re.match(r'(scene_\d+)', filename)
        if match:
            scene_name = match.group(1)
            if scene_name in scenes:
                files_to_delete.append(filepath)

    return sorted(files_to_delete)


def delete_scenes(
    input_dir: Path,
    scenes: Set[str],
    dry_run: bool = False,
):
    """刪除指定場景的所有檔案"""

    print(f"\n{'=' * 60}")
    print(f"PIDS Scene Deletion Tool")
    print(f"{'=' * 60}")
    print(f"Input directory: {input_dir}")
    print(f"Scenes to delete: {len(scenes)}")
    print(f"Mode: {'DRY RUN (preview only)' if dry_run else 'DELETE'}")
    print(f"{'=' * 60}\n")

    if not scenes:
        print("No scenes to delete.")
        return

    # 找出所有要刪除的檔案
    files_to_delete = find_files_to_delete(input_dir, scenes)

    if not files_to_delete:
        print("No files found matching the specified scenes.")
        return

    # 統計
    total_size = sum(f.stat().st_size for f in files_to_delete)
    size_mb = total_size / (1024 * 1024)

    print(f"Files to delete: {len(files_to_delete)}")
    print(f"Total size: {size_mb:.2f} MB")
    print()

    # 按場景分組顯示
    scenes_found = set()
    for f in files_to_delete:
        match = re.match(r'(scene_\d+)', f.name)
        if match:
            scenes_found.add(match.group(1))

    print(f"Scenes found: {len(scenes_found)}")

    if dry_run:
        print("\n[DRY RUN] Files that would be deleted:")
        for f in files_to_delete[:20]:  # 只顯示前 20 個
            print(f"  {f.name}")
        if len(files_to_delete) > 20:
            print(f"  ... and {len(files_to_delete) - 20} more files")
    else:
        # 確認刪除
        print(f"\nWARNING: This will permanently delete {len(files_to_delete)} files!")
        confirm = input("Type 'DELETE' to confirm: ")

        if confirm != 'DELETE':
            print("Aborted.")
            return

        # 刪除檔案
        deleted = 0
        errors = 0

        for f in files_to_delete:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                print(f"Error deleting {f}: {e}")
                errors += 1

        print(f"\nDeleted: {deleted} files")
        if errors:
            print(f"Errors: {errors}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description='PIDS 場景刪除工具 - 根據場景列表刪除檔案',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 預覽要刪除的檔案
  python delete_scenes.py --scenes_txt failed_scenes.txt --input_dir ./output --dry_run

  # 實際刪除
  python delete_scenes.py --scenes_txt failed_scenes.txt --input_dir ./output

  # 從 quality_report.md 自動提取失敗場景
  python delete_scenes.py --quality_report ./output/quality_report.md --input_dir ./output --dry_run

scenes.txt 格式:
  scene_0001
  scene_0002
  scene_0003
  # 註解行會被忽略
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--scenes_txt', type=str,
                             help='場景列表檔案 (每行一個場景名稱)')
    input_group.add_argument('--quality_report', type=str,
                             help='quality_report.md 路徑 (自動提取失敗場景)')

    parser.add_argument('--input_dir', type=str, required=True,
                        help='要刪除檔案的目錄')
    parser.add_argument('--dry_run', action='store_true',
                        help='預覽模式，不實際刪除')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return

    # 取得要刪除的場景列表
    if args.scenes_txt:
        scenes = parse_scenes_txt(Path(args.scenes_txt))
        print(f"Loaded {len(scenes)} scenes from {args.scenes_txt}")
    else:
        scenes = parse_quality_report(Path(args.quality_report))
        print(f"Found {len(scenes)} failed scenes in quality report")

    if not scenes:
        print("No scenes to delete.")
        return

    # 顯示場景列表
    print("\nScenes to delete:")
    scene_list = sorted(scenes)
    for s in scene_list[:10]:
        print(f"  {s}")
    if len(scene_list) > 10:
        print(f"  ... and {len(scene_list) - 10} more")

    delete_scenes(
        input_dir=input_dir,
        scenes=scenes,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()

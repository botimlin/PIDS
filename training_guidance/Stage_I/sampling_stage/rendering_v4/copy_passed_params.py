#!/usr/bin/env python3
"""
複製通過 QA 的 params.json 到指定資料夾

用法:
  python copy_passed_params.py --input_dir ./output_pol --output_dir ./passed_params --exclude failed.txt

說明:
  從渲染輸出目錄中，排除不通過的場景，複製剩餘的 params.json。
  之後 nopol 渲染可以用 --params_dir 指向這個資料夾。
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='複製通過 QA 的 params.json（排除失敗的）'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='偏振版輸出目錄（包含 *_params.json）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='輸出目錄（存放篩選後的 params.json）')
    parser.add_argument('--exclude', type=str, default=None,
                        help='不通過 QA 的場景名稱列表（每行一個場景名，會被忽略）')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 params.json
    all_params = list(input_dir.glob('*_params.json'))
    print(f"找到 {len(all_params)} 個 params.json")

    # 讀取排除列表
    exclude_scenes = set()
    if args.exclude:
        with open(args.exclude, 'r') as f:
            exclude_scenes = set(line.strip() for line in f if line.strip())
        print(f"排除列表中有 {len(exclude_scenes)} 個場景")

    # 複製非排除的場景
    copied = 0
    skipped = 0
    for params_path in all_params:
        # 從 scene_0001_params.json 提取 scene_0001
        scene_name = params_path.stem.replace('_params', '')
        if scene_name in exclude_scenes:
            skipped += 1
            continue
        shutil.copy(params_path, output_dir / params_path.name)
        copied += 1

    print(f"已複製 {copied} 個 params.json 到 {output_dir}")
    print(f"已跳過 {skipped} 個不通過的場景")


if __name__ == '__main__':
    main()

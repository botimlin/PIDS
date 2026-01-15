#!/usr/bin/env python3
"""
檢查場景亮度，找出過暗的場景

從 *_params.json 讀取 I_parallel_range 和 I_cross_range，
檢查場景是否過暗（可能導致 SNR 虛高）。

Usage:
    python check_brightness.py --input_dir /root/output_pol
    python check_brightness.py --input_dir /root/output_pol --threshold 0.1 --output dark_scenes.txt
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='檢查場景亮度')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='包含 *_params.json 的目錄')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='亮度閾值（max intensity < threshold 視為過暗，預設 0.1）')
    parser.add_argument('--output', type=str, default='dark_scenes.txt',
                        help='輸出過暗場景列表')
    parser.add_argument('--top', type=int, default=20,
                        help='顯示最暗的 N 個場景（預設 20）')

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    threshold = args.threshold

    params_files = sorted(input_dir.glob('*_params.json'))
    print(f"找到 {len(params_files)} 個 params.json")

    dark_scenes = []
    brightness_data = []

    for pf in params_files:
        try:
            with open(pf, 'r') as f:
                data = json.load(f)

            scene_name = pf.stem.replace('_params', '')
            stats = data.get('stats', {})

            i_par_range = stats.get('I_parallel_range', [0, 1])
            i_cross_range = stats.get('I_cross_range', [0, 1])

            # 取 max 值作為亮度指標
            max_parallel = i_par_range[1] if len(i_par_range) > 1 else 0
            max_cross = i_cross_range[1] if len(i_cross_range) > 1 else 0
            max_intensity = max(max_parallel, max_cross)

            brightness_data.append({
                'scene': scene_name,
                'max_parallel': max_parallel,
                'max_cross': max_cross,
                'max_intensity': max_intensity,
            })

            if max_intensity < threshold:
                dark_scenes.append((scene_name, max_intensity))

        except Exception as e:
            print(f"警告: 無法讀取 {pf}: {e}")

    # 統計
    if brightness_data:
        max_vals = [d['max_intensity'] for d in brightness_data]
        avg_max = sum(max_vals) / len(max_vals)
        min_max = min(max_vals)
        max_max = max(max_vals)

        print(f"\n亮度統計 (max intensity):")
        print(f"  平均: {avg_max:.4f}")
        print(f"  最小: {min_max:.4f}")
        print(f"  最大: {max_max:.4f}")
        print(f"  閾值: {threshold}")

        # 顯示最暗的 N 個場景
        sorted_by_brightness = sorted(brightness_data, key=lambda x: x['max_intensity'])
        print(f"\n最暗的 {args.top} 個場景:")
        print(f"{'場景':<15} {'max_parallel':>12} {'max_cross':>12} {'max_total':>12}")
        print("-" * 55)
        for d in sorted_by_brightness[:args.top]:
            print(f"{d['scene']:<15} {d['max_parallel']:>12.4f} {d['max_cross']:>12.4f} {d['max_intensity']:>12.4f}")

    print(f"\n過暗場景 (max < {threshold}): {len(dark_scenes)}/{len(params_files)}")

    if dark_scenes:
        # 按亮度排序
        dark_scenes.sort(key=lambda x: x[1])

        print(f"\n最暗的 10 個場景:")
        for scene, intensity in dark_scenes[:10]:
            print(f"  {scene}: max={intensity:.4f}")

        # 輸出到檔案
        with open(args.output, 'w') as f:
            for scene, intensity in dark_scenes:
                f.write(f"{scene}\n")
        print(f"\n已輸出到: {args.output}")

    else:
        print("所有場景亮度正常 ✓")


if __name__ == '__main__':
    main()

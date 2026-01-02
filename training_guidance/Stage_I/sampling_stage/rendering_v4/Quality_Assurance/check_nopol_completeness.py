#!/usr/bin/env python3
"""
nopol 渲染输出完整性检查

用法:
  python check_nopol_completeness.py --input_dir /root/output_nopol

检查每个场景是否有完整的5个输出文件。
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='检查 nopol 渲染输出完整性')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='nopol 输出目录')
    parser.add_argument('--failed', type=str, default='failed_nopol.txt',
                        help='输出不完整场景列表 (预设: failed_nopol.txt)')

    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    # 找出所有场景（以 _left.exr 为基准）
    left_files = sorted(input_dir.glob('*_left.exr'))
    scene_names = [f.stem.replace('_left', '') for f in left_files]

    print(f"找到 {len(scene_names)} 个场景")

    # 检查每个场景的文件完整性
    required_suffixes = ['_left.exr', '_right.exr', '_disparity.exr', '_depth.exr', '_mask.png']

    complete = 0
    incomplete = []

    for scene in scene_names:
        missing = []
        for suffix in required_suffixes:
            if not (input_dir / f"{scene}{suffix}").exists():
                missing.append(suffix)

        if missing:
            incomplete.append((scene, missing))
        else:
            complete += 1

    # 输出结果
    print(f"\n完整: {complete}")
    print(f"不完整: {len(incomplete)}")
    print(f"完整率: {complete / len(scene_names) * 100:.1f}%")

    if incomplete:
        print(f"\n不完整场景:")
        for scene, missing in incomplete[:10]:  # 只显示前10个
            print(f"  {scene}: 缺少 {missing}")
        if len(incomplete) > 10:
            print(f"  ... 还有 {len(incomplete) - 10} 个")

        # 写入failed文件
        with open(args.failed, 'w') as f:
            for scene, _ in incomplete:
                f.write(f"{scene}\n")
        print(f"\n已写入: {args.failed}")
    else:
        print("\n所有场景文件完整!")


if __name__ == '__main__':
    main()

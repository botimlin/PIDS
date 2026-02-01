"""
Mixed Pol/Nopol Dataset Organizer for PIDS Training
====================================================

組織混合 pol/nopol 數據集，支援 Curriculum Learning 訓練。

規則:
1. Scene Isolation: pol 和 nopol 場景無重疊
2. Val = pol only: 驗證集只包含 pol 場景
3. Nopol = train only: nopol 場景只用於訓練

數據結構:
- pol scenes: scene_0001 ~ scene_15000 (經過 QA 篩選)
- nopol scenes: scene_15001 ~ scene_18000 (3000 個新場景)

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random


def load_scene_list(filepath: str) -> List[str]:
    """載入場景列表"""
    scenes = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                scenes.append(line)
    return scenes


def save_scene_list(scenes: List[str], filepath: str, header: str = ""):
    """儲存場景列表"""
    with open(filepath, 'w') as f:
        if header:
            f.write(f"# {header}\n")
            f.write(f"# Total: {len(scenes)} scenes\n\n")
        for scene in sorted(scenes):
            f.write(f"{scene}\n")


def discover_nopol_scenes(nopol_data_dir: str) -> List[str]:
    """
    自動發現 nopol 場景

    Args:
        nopol_data_dir: nopol 數據目錄

    Returns:
        nopol 場景名稱列表
    """
    scenes = []
    if not os.path.exists(nopol_data_dir):
        print(f"[Warning] Nopol data directory not found: {nopol_data_dir}")
        return scenes

    # 尋找場景 (支援 nopol 命名: _left.exr 或 pol 命名: _left_parallel.exr)
    for f in os.listdir(nopol_data_dir):
        if f.endswith('_left.exr'):
            scene_name = f.replace('_left.exr', '')
            scenes.append(scene_name)
        elif f.endswith('_left_parallel.exr'):
            scene_name = f.replace('_left_parallel.exr', '')
            scenes.append(scene_name)

    return sorted(list(set(scenes)))  # 去重


def split_pol_scenes(
    pol_scenes: List[str],
    val_ratio: float = 0.1,
    seed: int = 42,
    train_count: Optional[int] = None,
    val_count: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """
    分割 pol 場景為訓練集和驗證集

    Args:
        pol_scenes: pol 場景列表
        val_ratio: 驗證集比例 (當 train_count/val_count 未指定時使用)
        seed: 隨機種子
        train_count: 指定訓練集數量 (優先於 val_ratio)
        val_count: 指定驗證集數量 (優先於 val_ratio)

    Returns:
        (train_pol, val_pol)
    """
    random.seed(seed)
    scenes = pol_scenes.copy()
    random.shuffle(scenes)

    # 優先使用指定數量
    if train_count is not None and val_count is not None:
        total_needed = train_count + val_count
        if total_needed > len(scenes):
            raise ValueError(
                f"Requested {train_count} train + {val_count} val = {total_needed}, "
                f"but only {len(scenes)} pol scenes available"
            )
        val_scenes = scenes[:val_count]
        train_scenes = scenes[val_count:val_count + train_count]
    elif train_count is not None:
        # 只指定 train_count，剩餘作為 val
        if train_count > len(scenes):
            raise ValueError(f"Requested {train_count} train, but only {len(scenes)} available")
        train_scenes = scenes[:train_count]
        val_scenes = scenes[train_count:]
    elif val_count is not None:
        # 只指定 val_count，剩餘作為 train
        if val_count > len(scenes):
            raise ValueError(f"Requested {val_count} val, but only {len(scenes)} available")
        val_scenes = scenes[:val_count]
        train_scenes = scenes[val_count:]
    else:
        # 使用比例
        n_val = int(len(scenes) * val_ratio)
        val_scenes = scenes[:n_val]
        train_scenes = scenes[n_val:]

    return sorted(train_scenes), sorted(val_scenes)


def organize_mixed_dataset(
    pol_scenes_file: str,
    nopol_data_dir: str,
    output_dir: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    nopol_scene_prefix: str = "scene_",
    nopol_scene_start: int = 15001,
    nopol_scene_count: int = 3000,
    train_pol_count: Optional[int] = None,
    val_count: Optional[int] = None,
) -> Dict:
    """
    組織混合數據集

    Args:
        pol_scenes_file: pol 場景列表文件 (經過 QA 的)
        nopol_data_dir: nopol 數據目錄
        output_dir: 輸出目錄
        val_ratio: 驗證集比例 (當 train_pol_count/val_count 未指定時使用)
        seed: 隨機種子
        nopol_scene_prefix: nopol 場景名前綴
        nopol_scene_start: nopol 場景起始編號
        nopol_scene_count: nopol 場景數量
        train_pol_count: 指定訓練集 pol 數量 (優先於 val_ratio)
        val_count: 指定驗證集數量 (優先於 val_ratio)

    Returns:
        組織結果統計
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 載入 pol 場景 (經過 QA 篩選的)
    print(f"[1/5] Loading pol scenes from: {pol_scenes_file}")
    pol_scenes = load_scene_list(pol_scenes_file)
    print(f"      Found {len(pol_scenes)} QA-passed pol scenes")

    # 2. 發現或生成 nopol 場景列表
    print(f"[2/5] Discovering nopol scenes from: {nopol_data_dir}")
    if os.path.exists(nopol_data_dir):
        all_nopol_scenes = discover_nopol_scenes(nopol_data_dir)
        print(f"      Found {len(all_nopol_scenes)} total nopol scenes in directory")

        # 過濾：只選擇指定範圍的場景 (nopol_start ~ nopol_start + nopol_count)
        nopol_scenes = []
        for scene in all_nopol_scenes:
            try:
                # 解析場景編號 (假設格式為 scene_XXXXX)
                scene_num = int(scene.split('_')[1])
                if nopol_scene_start <= scene_num < nopol_scene_start + nopol_scene_count:
                    nopol_scenes.append(scene)
            except (IndexError, ValueError):
                continue
        nopol_scenes = sorted(nopol_scenes)
        print(f"      Filtered to {len(nopol_scenes)} scenes (range: {nopol_scene_start} ~ {nopol_scene_start + nopol_scene_count - 1})")
    else:
        # 生成預期的 nopol 場景名稱 (用於規劃階段)
        nopol_scenes = [
            f"{nopol_scene_prefix}{i:05d}"
            for i in range(nopol_scene_start, nopol_scene_start + nopol_scene_count)
        ]
        print(f"      Generated {len(nopol_scenes)} expected nopol scene names")
        print(f"      (Actual data not found - this is for planning)")

    # 3. 檢查場景無重疊
    print(f"[3/5] Checking scene isolation...")
    pol_set = set(pol_scenes)
    nopol_set = set(nopol_scenes)
    overlap = pol_set & nopol_set
    if overlap:
        raise ValueError(f"Scene overlap detected: {list(overlap)[:5]}...")
    print(f"      No overlap - OK")

    # 4. 分割 pol 場景
    if train_pol_count is not None or val_count is not None:
        print(f"[4/5] Splitting pol scenes (train_pol={train_pol_count}, val={val_count})...")
    else:
        print(f"[4/5] Splitting pol scenes (val_ratio={val_ratio})...")
    train_pol, val_pol = split_pol_scenes(
        pol_scenes, val_ratio, seed,
        train_count=train_pol_count,
        val_count=val_count
    )
    print(f"      Train pol: {len(train_pol)}")
    print(f"      Val pol: {len(val_pol)}")

    # 5. 組合訓練集 (pol + nopol)
    print(f"[5/5] Organizing datasets...")
    train_all = train_pol + nopol_scenes  # nopol 只在訓練集

    # 建立場景命名映射 (用於 CurriculumSampler)
    scene_naming = {}
    for scene in train_pol:
        scene_naming[scene] = 'pol'
    for scene in nopol_scenes:
        scene_naming[scene] = 'nopol'
    for scene in val_pol:
        scene_naming[scene] = 'pol'

    # 儲存各種列表
    save_scene_list(
        train_pol,
        os.path.join(output_dir, "train_pol_scenes.txt"),
        "Training set - Pol scenes only"
    )
    save_scene_list(
        nopol_scenes,
        os.path.join(output_dir, "train_nopol_scenes.txt"),
        "Training set - Nopol scenes only"
    )
    save_scene_list(
        train_all,
        os.path.join(output_dir, "train_mixed_scenes.txt"),
        "Training set - Mixed pol + nopol"
    )
    save_scene_list(
        val_pol,
        os.path.join(output_dir, "val_scenes.txt"),
        "Validation set - Pol only"
    )

    # 儲存場景命名映射 (JSON)
    naming_file = os.path.join(output_dir, "scene_naming.json")
    with open(naming_file, 'w') as f:
        json.dump(scene_naming, f, indent=2)

    # 統計結果
    stats = {
        'total_pol_scenes': len(pol_scenes),
        'total_nopol_scenes': len(nopol_scenes),
        'train_pol': len(train_pol),
        'train_nopol': len(nopol_scenes),
        'train_total': len(train_all),
        'val_total': len(val_pol),
        'pol_ratio_train': len(train_pol) / len(train_all) if train_all else 0,
        'nopol_ratio_train': len(nopol_scenes) / len(train_all) if train_all else 0,
    }

    # 儲存統計
    stats_file = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


def print_summary(stats: Dict, output_dir: str):
    """印出摘要"""
    print("\n" + "=" * 60)
    print("Mixed Dataset Organization Complete")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nDataset Statistics:")
    print(f"  Pol scenes (QA passed): {stats['total_pol_scenes']}")
    print(f"  Nopol scenes:           {stats['total_nopol_scenes']}")
    print(f"\nTraining Set:")
    print(f"  Pol:   {stats['train_pol']} ({stats['pol_ratio_train']:.1%})")
    print(f"  Nopol: {stats['train_nopol']} ({stats['nopol_ratio_train']:.1%})")
    print(f"  Total: {stats['train_total']}")
    print(f"\nValidation Set:")
    print(f"  Pol only: {stats['val_total']}")
    print(f"\nGenerated Files:")
    print(f"  - train_pol_scenes.txt")
    print(f"  - train_nopol_scenes.txt")
    print(f"  - train_mixed_scenes.txt")
    print(f"  - val_scenes.txt")
    print(f"  - scene_naming.json")
    print(f"  - dataset_stats.json")
    print("=" * 60)


def create_symlinks_or_copy(
    pol_data_dir: str,
    nopol_data_dir: str,
    output_data_dir: str,
    train_scenes: List[str],
    val_scenes: List[str],
    scene_naming: Dict[str, str],
    use_symlinks: bool = False
):
    """
    創建符號連結或複製數據文件到統一目錄

    支援兩種目錄結構:
    1. Flat: 所有檔案在同一目錄
    2. Organized: stereo_pairs/, ground_truth/, masks/ 子目錄

    Args:
        pol_data_dir: pol 數據來源目錄
        nopol_data_dir: nopol 數據來源目錄
        output_data_dir: 輸出數據目錄
        train_scenes: 訓練場景列表
        val_scenes: 驗證場景列表
        scene_naming: 場景命名映射 (scene_name -> 'pol' or 'nopol')
        use_symlinks: 是否使用符號連結 (Windows 需要管理員權限)
    """
    train_dir = os.path.join(output_data_dir, "train")
    val_dir = os.path.join(output_data_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 檢測 pol 目錄結構
    pol_is_organized = os.path.exists(os.path.join(pol_data_dir, 'stereo_pairs'))
    nopol_is_organized = os.path.exists(os.path.join(nopol_data_dir, 'stereo_pairs'))

    print(f"  Pol structure: {'organized' if pol_is_organized else 'flat'}")
    print(f"  Nopol structure: {'organized' if nopol_is_organized else 'flat'}")

    # 檔案映射: (suffix, subdir_for_organized)
    pol_files = [
        ('_left_parallel.exr', 'stereo_pairs'),
        ('_right_cross.exr', 'stereo_pairs'),
        ('_disparity.exr', 'ground_truth'),
        ('_depth.exr', 'ground_truth'),
        ('_glass_mask.exr', 'masks'),
        ('_glass_mask_strict.exr', 'masks'),
        ('_mask.png', 'masks'),
        ('_params.json', ''),
    ]
    nopol_files = [
        ('_left.exr', 'stereo_pairs'),
        ('_right.exr', 'stereo_pairs'),
        ('_disparity.exr', 'ground_truth'),
        ('_depth.exr', 'ground_truth'),
        ('_glass_mask.exr', 'masks'),
        ('_glass_mask_strict.exr', 'masks'),
        ('_mask.png', 'masks'),
        ('_params.json', ''),
    ]

    def link_or_copy(src: str, dst: str):
        if os.path.exists(dst):
            return
        if use_symlinks:
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)

    def get_src_path(base_dir: str, scene: str, suffix: str, subdir: str, is_organized: bool) -> str:
        if is_organized and subdir:
            return os.path.join(base_dir, subdir, f"{scene}{suffix}")
        else:
            return os.path.join(base_dir, f"{scene}{suffix}")

    print(f"\nLinking/copying data files...")

    # 處理訓練集
    linked_count = 0
    for scene in train_scenes:
        is_nopol = scene_naming.get(scene, 'pol') == 'nopol'
        if is_nopol:
            base_dir = nopol_data_dir
            files = nopol_files
            is_organized = nopol_is_organized
        else:
            base_dir = pol_data_dir
            files = pol_files
            is_organized = pol_is_organized

        for suffix, subdir in files:
            src = get_src_path(base_dir, scene, suffix, subdir, is_organized)
            dst = os.path.join(train_dir, f"{scene}{suffix}")
            if os.path.exists(src):
                link_or_copy(src, dst)
                linked_count += 1

    print(f"  Train: linked {linked_count} files")

    # 處理驗證集 (只有 pol)
    linked_count = 0
    for scene in val_scenes:
        for suffix, subdir in pol_files:
            src = get_src_path(pol_data_dir, scene, suffix, subdir, pol_is_organized)
            dst = os.path.join(val_dir, f"{scene}{suffix}")
            if os.path.exists(src):
                link_or_copy(src, dst)
                linked_count += 1

    print(f"  Val: linked {linked_count} files")

    print(f"  Train data: {train_dir}")
    print(f"  Val data: {val_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Organize mixed pol/nopol dataset for PIDS training"
    )
    parser.add_argument(
        '--pol_scenes_file',
        type=str,
        default='./train_100pct_scenes.txt',
        help='QA-passed pol scenes list file'
    )
    parser.add_argument(
        '--nopol_data_dir',
        type=str,
        default='./data/nopol',
        help='Nopol rendered data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./mixed_dataset',
        help='Output directory for organized dataset'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (from pol scenes only)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--nopol_start',
        type=int,
        default=15001,
        help='Starting scene number for nopol scenes'
    )
    parser.add_argument(
        '--nopol_count',
        type=int,
        default=3000,
        help='Number of nopol scenes'
    )
    parser.add_argument(
        '--train_pol_count',
        type=int,
        default=None,
        help='Fixed number of pol scenes for training (overrides val_ratio)'
    )
    parser.add_argument(
        '--val_count',
        type=int,
        default=None,
        help='Fixed number of pol scenes for validation (overrides val_ratio)'
    )
    parser.add_argument(
        '--link_data',
        action='store_true',
        help='Create symlinks/copy data files to unified directory'
    )
    parser.add_argument(
        '--pol_data_dir',
        type=str,
        default='./data/pol',
        help='Pol rendered data directory (for --link_data)'
    )
    parser.add_argument(
        '--output_data_dir',
        type=str,
        default='./mixed_dataset/data',
        help='Output data directory (for --link_data)'
    )

    args = parser.parse_args()

    # 組織數據集
    stats = organize_mixed_dataset(
        pol_scenes_file=args.pol_scenes_file,
        nopol_data_dir=args.nopol_data_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        nopol_scene_start=args.nopol_start,
        nopol_scene_count=args.nopol_count,
        train_pol_count=args.train_pol_count,
        val_count=args.val_count,
    )

    print_summary(stats, args.output_dir)

    # 如果需要，連結數據文件
    if args.link_data:
        train_scenes = load_scene_list(
            os.path.join(args.output_dir, "train_mixed_scenes.txt")
        )
        val_scenes = load_scene_list(
            os.path.join(args.output_dir, "val_scenes.txt")
        )
        # 載入 scene_naming
        naming_file = os.path.join(args.output_dir, "scene_naming.json")
        with open(naming_file, 'r') as f:
            scene_naming = json.load(f)

        # 默認 output_data_dir 為 output_dir/data
        output_data_dir = args.output_data_dir
        if output_data_dir == './mixed_dataset/data':  # 未指定，使用默認
            output_data_dir = os.path.join(args.output_dir, "data")

        create_symlinks_or_copy(
            pol_data_dir=args.pol_data_dir,
            nopol_data_dir=args.nopol_data_dir,
            output_data_dir=output_data_dir,
            train_scenes=train_scenes,
            val_scenes=val_scenes,
            scene_naming=scene_naming,
        )


if __name__ == '__main__':
    main()

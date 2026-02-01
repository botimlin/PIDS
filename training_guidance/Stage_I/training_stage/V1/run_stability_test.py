"""
Exp #42: Numerical Stability Test
=================================

在同一台機器上，改變精度設定（TF32, cuDNN benchmark, AMP）來測試
Baseline 和 V2-E 對浮點精度差異的敏感度。

主指標: Glass EPE Median (px)
報告: 各條件下的值、σ、CV(%)

Usage:
    python run_stability_test.py \
        --pol_data_dir ./dataset_pol_test \
        --nopol_data_dir ./dataset_nopol_test \
        --baseline_ckpt ./checkpoints_baseline_exp22/baseline_best.pth \
        --v2e_ckpt ./checkpoints_exp39_v2e/checkpoint_best.pth \
        --output_dir ./eval_results/stability_test
"""

import argparse
import json
import subprocess
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime


# 5 組精度設定
CONDITIONS = {
    "C1": {
        "label": "Default (TF32=ON, Bench=ON, FP32)",
        "tf32": True,
        "benchmark": True,
        "amp": False,
    },
    "C2": {
        "label": "TF32=OFF, Bench=ON, FP32",
        "tf32": False,
        "benchmark": True,
        "amp": False,
    },
    "C3": {
        "label": "TF32=ON, Bench=OFF, FP32",
        "tf32": True,
        "benchmark": False,
        "amp": False,
    },
    "C4": {
        "label": "Full Deterministic (TF32=OFF, Bench=OFF, FP32)",
        "tf32": False,
        "benchmark": False,
        "amp": False,
    },
    "C5": {
        "label": "AMP (TF32=ON, Bench=ON, FP16)",
        "tf32": True,
        "benchmark": True,
        "amp": True,
    },
}


def run_single_evaluation(args, arch: str, condition_name: str, condition: dict) -> dict:
    """執行單一 evaluation 並回傳 Glass EPE Median"""

    # 構建環境變數來控制精度設定
    env_vars = {
        "STABILITY_TEST_TF32": "1" if condition["tf32"] else "0",
        "STABILITY_TEST_BENCHMARK": "1" if condition["benchmark"] else "0",
        "STABILITY_TEST_AMP": "1" if condition["amp"] else "0",
    }

    # 選擇 checkpoint、架構參數和對應的資料集
    if arch == "baseline":
        ckpt = args.baseline_ckpt
        data_dir = args.nopol_data_dir
        arch_args = ["--baseline"]
    else:
        ckpt = args.v2e_ckpt
        data_dir = args.pol_data_dir
        arch_args = ["--pol_volume_v2e", "--pol_levels", "4", "--pol_radius", "4"]

    output_subdir = os.path.join(args.output_dir, f"{arch}_{condition_name}")

    cmd = [
        sys.executable, "evaluate_pids.py",
        "--data_dir", data_dir,
        "--checkpoint", ckpt,
        "--iters", str(args.iters),
        "--output_dir", output_subdir,
    ] + arch_args

    # 設定環境
    env = os.environ.copy()
    env.update(env_vars)

    print(f"\n{'='*60}")
    print(f"  Running: {arch.upper()} | {condition_name}: {condition['label']}")
    print(f"  TF32={condition['tf32']}, Benchmark={condition['benchmark']}, AMP={condition['amp']}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))

    # 顯示輸出
    if result.stdout:
        print(result.stdout[-500:])  # 只顯示最後 500 字元
    if result.returncode != 0:
        print(f"  [ERROR] Return code: {result.returncode}")
        if result.stderr:
            print(result.stderr[-500:])
        return None

    # 從 statistics json 讀取 Glass EPE Median
    stats = parse_statistics(output_subdir)
    return stats


def parse_statistics(output_dir: str) -> dict:
    """從 evaluation 輸出目錄讀取統計數據"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    # 找到最新的 statistics json
    json_files = sorted(output_path.glob("statistics_*.json"), reverse=True)
    if not json_files:
        return None

    with open(json_files[0], 'r') as f:
        stats = json.load(f)

    return stats


def extract_glass_epe_median(stats: dict) -> float:
    """從 statistics dict 提取 Glass EPE Median"""
    if stats is None:
        return float('nan')

    # 嘗試不同的 key 結構
    try:
        return stats['glass_epe']['median']
    except (KeyError, TypeError):
        pass

    try:
        return stats['glass_epe_median']
    except (KeyError, TypeError):
        pass

    return float('nan')


def print_results_table(results: dict):
    """列印最終結果表格"""

    print("\n" + "=" * 80)
    print("  STABILITY TEST RESULTS - Exp #42")
    print("  Metric: Glass EPE Median (px)")
    print("=" * 80)

    # 表頭
    cond_names = list(CONDITIONS.keys())
    header = f"{'Method':<20}"
    for c in cond_names:
        header += f" {c:>8}"
    header += f" {'σ':>8} {'CV(%)':>8}"
    print(header)
    print("-" * 80)

    # 每個架構一行
    for arch in ["baseline", "v2e"]:
        arch_label = "RAFT-Stereo" if arch == "baseline" else "PIDS (Ours)"
        row = f"{arch_label:<20}"

        values = []
        for c in cond_names:
            key = f"{arch}_{c}"
            val = results.get(key, float('nan'))
            values.append(val)
            if np.isnan(val):
                row += f" {'N/A':>8}"
            else:
                row += f" {val:>8.3f}"

        # 計算 σ 和 CV
        valid_values = [v for v in values if not np.isnan(v)]
        if len(valid_values) >= 2:
            sigma = np.std(valid_values, ddof=1)
            mean = np.mean(valid_values)
            cv = (sigma / mean * 100) if mean > 0 else float('nan')
            row += f" {sigma:>8.3f} {cv:>7.2f}%"
        else:
            row += f" {'N/A':>8} {'N/A':>8}"

        print(row)

    print("-" * 80)

    # 條件說明
    print("\nConditions:")
    for name, cond in CONDITIONS.items():
        print(f"  {name}: {cond['label']}")


def main():
    parser = argparse.ArgumentParser(description='Exp #42: Numerical Stability Test')

    parser.add_argument('--pol_data_dir', type=str, default=None,
                        help='Path to pol test dataset (for V2-E)')
    parser.add_argument('--nopol_data_dir', type=str, required=True,
                        help='Path to nopol test dataset (for Baseline)')
    parser.add_argument('--baseline_ckpt', type=str, required=True,
                        help='Path to baseline RAFT-Stereo checkpoint')
    parser.add_argument('--v2e_ckpt', type=str, default=None,
                        help='Path to V2-E PIDS checkpoint')
    parser.add_argument('--output_dir', type=str, default='./eval_results/stability_test',
                        help='Output directory')
    parser.add_argument('--iters', type=int, default=32,
                        help='Number of inference iterations')
    parser.add_argument('--baseline_only', action='store_true',
                        help='Only run baseline tests (skip V2-E)')

    args = parser.parse_args()

    # 自動偵測: 沒有 V2-E checkpoint 就只跑 baseline
    if args.v2e_ckpt is None or args.pol_data_dir is None:
        args.baseline_only = True

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("  Exp #42: Numerical Stability Test")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    archs = ["baseline"] if args.baseline_only else ["baseline", "v2e"]
    n_runs = len(CONDITIONS) * len(archs)

    print(f"  Baseline: {args.baseline_ckpt}")
    if not args.baseline_only:
        print(f"  V2-E:     {args.v2e_ckpt}")
        print(f"  Pol Data:   {args.pol_data_dir}")
    print(f"  Nopol Data: {args.nopol_data_dir}")
    print(f"  Mode: {'Baseline Only' if args.baseline_only else 'Baseline + V2-E'}")
    print(f"  Conditions: {len(CONDITIONS)} x {len(archs)} = {n_runs} runs")
    print("=" * 80)

    # 收集結果
    results = {}

    for cond_name, cond in CONDITIONS.items():
        for arch in archs:
            stats = run_single_evaluation(args, arch, cond_name, cond)
            median_val = extract_glass_epe_median(stats)
            key = f"{arch}_{cond_name}"
            results[key] = median_val
            print(f"  -> {arch.upper()} {cond_name}: Glass EPE Median = {median_val:.3f} px")

    # 列印結果表格
    print_results_table(results)

    # 保存結果
    results_path = os.path.join(args.output_dir, "stability_test_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "conditions": {k: v["label"] for k, v in CONDITIONS.items()},
            "results": {k: v if not np.isnan(v) else None for k, v in results.items()},
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

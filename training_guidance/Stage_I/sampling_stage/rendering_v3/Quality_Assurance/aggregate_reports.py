"""
PIDS 場景報告匯總腳本
讀取所有場景的 JSON 報告，匯總成一個 Markdown 文件

Usage:
    python aggregate_reports.py --input_dir ./output/output --output summary_report.md
    python aggregate_reports.py --input_dir ./output/output --output summary_report.md --sort score

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import re


def load_reports(input_dir: Path) -> List[Dict]:
    """
    載入所有場景報告

    自動過濾:
    - *_glass_report.json (OBJ 分離中間產物)
    - *_other_report.json (OBJ 分離中間產物)
    - *_report_nopol.json (無偏振版本，分開處理)
    """
    reports = []

    for json_file in sorted(input_dir.glob("*_report.json")):
        filename = json_file.name

        # 過濾中間產物
        if '_glass_report.json' in filename or '_other_report.json' in filename:
            continue
        if '_report_nopol.json' in filename:
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_file_path'] = str(json_file)
                reports.append(data)
        except Exception as e:
            print(f"警告: 無法讀取 {json_file}: {e}")

    return reports


def to_percent(value: float, decimals: int = 1) -> str:
    """轉換為百分比字串"""
    return f"{value * 100:.{decimals}f}%"


def generate_summary_table(reports: List[Dict], sort_by: str = 'name') -> str:
    """生成匯總表格"""

    # 排序
    if sort_by == 'score':
        reports = sorted(reports, key=lambda x: x.get('quality', {}).get('score', 0), reverse=True)
    elif sort_by == 'dolp':
        reports = sorted(reports, key=lambda x: x.get('polarization', {}).get('glass_region', {}).get('dolp_mean', 0), reverse=True)
    elif sort_by == 'name':
        reports = sorted(reports, key=lambda x: x.get('scene_name', ''))

    # 表頭
    lines = [
        "| 場景 | 品質 | 分數 | 玻璃DoLP | 背景DoLP | 玻璃佔比 | 深度有效率 | 背景平衡 | SNR | 警告 |",
        "|------|------|------|----------|----------|----------|------------|----------|-----|------|",
    ]

    for r in reports:
        scene_name = r.get('scene_name', 'unknown')

        # 品質
        quality = r.get('quality', {})
        level = quality.get('level', '-')
        score = quality.get('score', 0)

        # 偏振
        pol = r.get('polarization', {})
        glass_dolp = pol.get('glass_region', {}).get('dolp_mean', 0)
        bg_dolp = pol.get('background_region', {}).get('dolp_mean', 0)
        glass_ratio = pol.get('glass_region', {}).get('pixel_ratio', 0)

        # 深度有效率
        gdv = r.get('glass_depth_validity', {})
        validity_rate = gdv.get('validity_rate', 0)
        validity_pass = gdv.get('pass', False)

        # 背景平衡
        balance = r.get('intensity_balance', {})
        bg_ratio = balance.get('background_ratio_mean', 1.0)
        is_balanced = balance.get('is_balanced', False)

        # SNR
        noise = r.get('noise', {})
        snr = noise.get('snr_polarization', 0)

        # 警告數
        warnings = r.get('warnings', [])
        warn_count = len(warnings)

        # 格式化
        level_emoji = {'excellent': '🟢', 'good': '🟡', 'acceptable': '🟠', 'poor': '🔴'}.get(level, '⚪')
        validity_emoji = '✓' if validity_pass else '✗'
        balance_emoji = '✓' if is_balanced else '✗'

        line = f"| {scene_name} | {level_emoji} {level} | {score} | {to_percent(glass_dolp)} | {to_percent(bg_dolp)} | {to_percent(glass_ratio)} | {to_percent(validity_rate)} {validity_emoji} | {bg_ratio:.2f}x {balance_emoji} | {snr:.1f} | {warn_count} |"
        lines.append(line)

    return "\n".join(lines)


def generate_statistics(reports: List[Dict]) -> str:
    """生成統計摘要"""

    if not reports:
        return "無數據"

    n = len(reports)

    # 收集數據
    scores = [r.get('quality', {}).get('score', 0) for r in reports]
    glass_dolps = [r.get('polarization', {}).get('glass_region', {}).get('dolp_mean', 0) for r in reports]
    bg_dolps = [r.get('polarization', {}).get('background_region', {}).get('dolp_mean', 0) for r in reports]
    validity_rates = [r.get('glass_depth_validity', {}).get('validity_rate', 0) for r in reports]
    snrs = [r.get('noise', {}).get('snr_polarization', 0) for r in reports]

    # 品質等級分佈
    levels = [r.get('quality', {}).get('level', 'unknown') for r in reports]
    level_counts = {
        'excellent': levels.count('excellent'),
        'good': levels.count('good'),
        'acceptable': levels.count('acceptable'),
        'poor': levels.count('poor'),
    }

    # 通過率
    validity_pass = sum(1 for r in reports if r.get('glass_depth_validity', {}).get('pass', False))
    balance_pass = sum(1 for r in reports if r.get('intensity_balance', {}).get('is_balanced', False))

    def avg(lst): return sum(lst) / len(lst) if lst else 0
    def min_val(lst): return min(lst) if lst else 0
    def max_val(lst): return max(lst) if lst else 0

    lines = [
        "## 統計摘要",
        "",
        f"**場景總數**: {n}",
        "",
        "### 品質分佈",
        "",
        f"| 等級 | 數量 | 佔比 |",
        f"|------|------|------|",
        f"| 🟢 Excellent | {level_counts['excellent']} | {to_percent(level_counts['excellent']/n)} |",
        f"| 🟡 Good | {level_counts['good']} | {to_percent(level_counts['good']/n)} |",
        f"| 🟠 Acceptable | {level_counts['acceptable']} | {to_percent(level_counts['acceptable']/n)} |",
        f"| 🔴 Poor | {level_counts['poor']} | {to_percent(level_counts['poor']/n)} |",
        "",
        "### 關鍵指標統計",
        "",
        "| 指標 | 平均 | 最小 | 最大 |",
        "|------|------|------|------|",
        f"| 品質分數 | {avg(scores):.1f} | {min_val(scores)} | {max_val(scores)} |",
        f"| 玻璃區域 DoLP | {to_percent(avg(glass_dolps))} | {to_percent(min_val(glass_dolps))} | {to_percent(max_val(glass_dolps))} |",
        f"| 背景區域 DoLP | {to_percent(avg(bg_dolps))} | {to_percent(min_val(bg_dolps))} | {to_percent(max_val(bg_dolps))} |",
        f"| 玻璃深度有效率 | {to_percent(avg(validity_rates))} | {to_percent(min_val(validity_rates))} | {to_percent(max_val(validity_rates))} |",
        f"| SNR | {avg(snrs):.2f} | {min_val(snrs):.2f} | {max_val(snrs):.2f} |",
        "",
        "### 通過率",
        "",
        f"| 檢查項目 | 通過數 | 通過率 |",
        f"|----------|--------|--------|",
        f"| 深度有效率 (>90%) | {validity_pass}/{n} | {to_percent(validity_pass/n)} |",
        f"| 背景平衡 (0.5~2.0x) | {balance_pass}/{n} | {to_percent(balance_pass/n)} |",
    ]

    return "\n".join(lines)


def generate_warnings_summary(reports: List[Dict]) -> str:
    """生成警告摘要"""

    warning_counts = {}
    scenes_with_warnings = []

    for r in reports:
        scene_name = r.get('scene_name', 'unknown')
        warnings = r.get('warnings', [])

        if warnings:
            scenes_with_warnings.append((scene_name, warnings))

        for w in warnings:
            # 簡化警告訊息用於統計
            w_key = w.split('（')[0].split('，')[0]  # 取主要訊息
            warning_counts[w_key] = warning_counts.get(w_key, 0) + 1

    lines = [
        "## 警告摘要",
        "",
        f"**有警告的場景數**: {len(scenes_with_warnings)}/{len(reports)}",
        "",
    ]

    if warning_counts:
        lines.extend([
            "### 警告類型統計",
            "",
            "| 警告類型 | 次數 | 影響說明 |",
            "|----------|------|----------|",
        ])

        # 警告影響說明對照表
        warning_impacts = {
            "I∥/I⊥ 比值較低": "偏振對比度較弱，但 DoLP 仍 >10% 可用於訓練。玻璃區域仍有明顯偏振信號。",
            "高偏振區域佔比過大": "場景中玻璃佔比過高，可能導致模型過度依賴偏振特徵。",
            "高偏振區域過小": "玻璃物體太小或偏振效果弱，訓練效果可能受限。",
            "背景 DoLP 偏高": "背景區域出現非預期偏振，可能影響玻璃/背景區分。",
            "SNR 較低": "信噪比不足，建議增加 SPP 或檢查光源配置。",
            "玻璃區域深度有效率不足": "玻璃區域深度值缺失過多，違反 Criterion 5。",
            "背景區域強度不平衡": "I∥/I⊥ 在背景區域不平衡，可能導致立體匹配失敗。",
        }

        for w, count in sorted(warning_counts.items(), key=lambda x: -x[1]):
            impact = warning_impacts.get(w, "請檢查渲染配置")
            lines.append(f"| {w} | {count} | {impact} |")

    return "\n".join(lines)


def generate_failed_scenes(reports: List[Dict]) -> str:
    """生成失敗場景清單"""

    failed = []

    for r in reports:
        scene_name = r.get('scene_name', 'unknown')
        quality = r.get('quality', {})

        # 收集失敗原因
        reasons = []

        # 品質太低
        if quality.get('score', 100) < 40:
            reasons.append(f"品質分數低 ({quality.get('score')})")

        # 深度有效率未通過
        gdv = r.get('glass_depth_validity', {})
        if not gdv.get('pass', True):
            reasons.append(f"深度有效率不足 ({to_percent(gdv.get('validity_rate', 0))})")

        # 背景不平衡
        balance = r.get('intensity_balance', {})
        if not balance.get('is_balanced', True):
            reasons.append(f"背景不平衡 ({balance.get('background_ratio_mean', 0):.2f}x)")

        if reasons:
            failed.append((scene_name, reasons))

    if not failed:
        return "## 失敗場景\n\n無失敗場景 🎉"

    lines = [
        "## 失敗場景",
        "",
        f"**失敗數**: {len(failed)}/{len(reports)} ({to_percent(len(failed)/len(reports))})",
        "",
        "| 場景 | 失敗原因 |",
        "|------|----------|",
    ]

    for scene_name, reasons in failed:
        lines.append(f"| {scene_name} | {'; '.join(reasons)} |")

    return "\n".join(lines)


def generate_markdown_report(reports: List[Dict], sort_by: str = 'name') -> str:
    """生成完整 Markdown 報告"""

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = [
        f"# PIDS 渲染場景匯總報告",
        "",
        f"**生成時間**: {now}",
        f"**場景總數**: {len(reports)}",
        "",
        "---",
        "",
        generate_statistics(reports),
        "",
        "---",
        "",
        generate_failed_scenes(reports),
        "",
        "---",
        "",
        generate_warnings_summary(reports),
        "",
        "---",
        "",
        "## 完整場景列表",
        "",
        generate_summary_table(reports, sort_by),
        "",
    ]

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description='PIDS 場景報告匯總腳本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python aggregate_reports.py --input_dir ./output/output
  python aggregate_reports.py --input_dir ./output/output --output summary.md --sort score
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='包含 *_report.json 的目錄')
    parser.add_argument('--output', type=str, default='summary_report.md',
                        help='輸出 Markdown 檔案路徑 (預設: summary_report.md)')
    parser.add_argument('--sort', type=str, default='name',
                        choices=['name', 'score', 'dolp'],
                        help='排序方式: name (場景名), score (品質分數), dolp (玻璃DoLP)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"錯誤: 目錄不存在 {input_dir}")
        return 1

    print(f"讀取報告: {input_dir}")
    reports = load_reports(input_dir)

    if not reports:
        print("錯誤: 沒有找到有效的報告檔案")
        return 1

    print(f"找到 {len(reports)} 個場景報告")

    # 生成報告
    md_content = generate_markdown_report(reports, args.sort)

    # 寫入檔案
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"報告已生成: {output_path}")

    # 顯示摘要
    levels = [r.get('quality', {}).get('level', 'unknown') for r in reports]
    print(f"\nQuality Distribution:")
    print(f"  Excellent: {levels.count('excellent')}")
    print(f"  Good: {levels.count('good')}")
    print(f"  Acceptable: {levels.count('acceptable')}")
    print(f"  Poor: {levels.count('poor')}")

    return 0


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
"""
PIDS 報告匯總腳本
=================

讀取渲染器生成的各場景 JSON 報告，生成總結報告。

用法：
    python pids_summarize_reports.py --input_dir ./output_stage1
    python pids_summarize_reports.py --input_dir ./output_stage1 --output ./summary.md
    python pids_summarize_reports.py --input_dir ./output_stage1 --format html

作者: PIDS Project
"""

import numpy as np
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def load_reports(input_dir: str) -> List[Dict]:
    """載入所有場景報告"""
    report_files = sorted(Path(input_dir).glob("*_report.json"))
    
    reports = []
    for f in report_files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                report = json.load(fp)
                reports.append(report)
        except Exception as e:
            print(f"[WARNING] 無法載入 {f}: {e}")
    
    return reports


def generate_summary(reports: List[Dict]) -> Dict:
    """生成統計摘要"""
    if not reports:
        return {'total': 0, 'valid': False}
    
    summary = {
        'total_scenes': len(reports),
        'timestamp': datetime.now().isoformat(),
    }
    
    # 品質統計
    quality_scores = [r['quality']['score'] for r in reports if 'quality' in r]
    quality_levels = [r['quality']['level'] for r in reports if 'quality' in r]
    valid_count = sum(1 for r in reports if r.get('quality', {}).get('valid', False))
    
    summary['quality'] = {
        'valid_scenes': valid_count,
        'pass_rate': valid_count / len(reports) * 100,
        'score_avg': float(np.mean(quality_scores)) if quality_scores else 0,
        'score_min': float(np.min(quality_scores)) if quality_scores else 0,
        'score_max': float(np.max(quality_scores)) if quality_scores else 0,
    }
    
    # 品質等級分佈
    level_dist = defaultdict(int)
    for level in quality_levels:
        level_dist[level] += 1
    summary['quality']['level_distribution'] = dict(level_dist)
    
    # 偏振統計
    glass_dolp = [r['polarization']['glass_region']['dolp_mean'] for r in reports 
                  if 'polarization' in r and r['polarization']['glass_region']['pixel_count'] > 0]
    bg_dolp = [r['polarization']['background_region']['dolp_mean'] for r in reports
               if 'polarization' in r and r['polarization']['background_region']['pixel_count'] > 0]
    ratios = [r['polarization']['intensity_ratio']['mean'] for r in reports
              if 'polarization' in r and r['polarization']['intensity_ratio']['mean'] > 0]
    
    summary['polarization'] = {
        'glass_dolp_avg': float(np.mean(glass_dolp)) if glass_dolp else 0,
        'glass_dolp_min': float(np.min(glass_dolp)) if glass_dolp else 0,
        'glass_dolp_max': float(np.max(glass_dolp)) if glass_dolp else 0,
        'background_dolp_avg': float(np.mean(bg_dolp)) if bg_dolp else 0,
        'intensity_ratio_avg': float(np.mean(ratios)) if ratios else 0,
        'intensity_ratio_min': float(np.min(ratios)) if ratios else 0,
        'intensity_ratio_max': float(np.max(ratios)) if ratios else 0,
    }
    
    # 噪點統計
    snr_values = [r['noise']['snr_polarization'] for r in reports if 'noise' in r]
    summary['noise'] = {
        'snr_avg': float(np.mean(snr_values)) if snr_values else 0,
        'snr_min': float(np.min(snr_values)) if snr_values else 0,
        'snr_max': float(np.max(snr_values)) if snr_values else 0,
    }
    
    # 🔥 新增：強度平衡統計
    balance_ratios = [r['intensity_balance']['background_ratio_mean'] for r in reports 
                      if 'intensity_balance' in r]
    balanced_count = sum(1 for r in reports 
                        if r.get('intensity_balance', {}).get('is_balanced', False))
    summary['intensity_balance'] = {
        'balanced_count': balanced_count,
        'balanced_ratio': balanced_count / len(reports) * 100 if reports else 0,
        'bg_ratio_avg': float(np.mean(balance_ratios)) if balance_ratios else 1.0,
        'bg_ratio_min': float(np.min(balance_ratios)) if balance_ratios else 1.0,
        'bg_ratio_max': float(np.max(balance_ratios)) if balance_ratios else 1.0,
    }
    
    # 警告統計
    all_warnings = []
    for r in reports:
        all_warnings.extend(r.get('warnings', []))
    
    warning_counts = defaultdict(int)
    for w in all_warnings:
        warning_counts[w] += 1
    summary['warnings'] = {
        'total': len(all_warnings),
        'by_type': dict(warning_counts),
    }
    
    # 渲染配置（取第一個）
    if reports and 'render_config' in reports[0]:
        summary['render_config'] = reports[0]['render_config']
    
    return summary


def generate_markdown(reports: List[Dict], summary: Dict, input_dir: str) -> str:
    """生成 Markdown 報告"""
    md = f"""# PIDS 訓練數據品質報告

- **數據目錄**: `{input_dir}`
- **生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **場景數量**: {summary['total_scenes']}

---

## 📊 總覽

| 指標 | 數值 |
|------|------|
| 有效場景 | {summary['quality']['valid_scenes']} / {summary['total_scenes']} |
| 通過率 | {summary['quality']['pass_rate']:.1f}% |
| 平均分數 | {summary['quality']['score_avg']:.1f} |
| 分數範圍 | {summary['quality']['score_min']:.0f} ~ {summary['quality']['score_max']:.0f} |

### 品質等級分佈

| 等級 | 數量 | 比例 |
|------|------|------|
"""
    
    level_order = ['excellent', 'good', 'acceptable', 'poor']
    level_emoji = {'excellent': '🟢', 'good': '🟢', 'acceptable': '🟡', 'poor': '🔴'}
    
    for level in level_order:
        count = summary['quality']['level_distribution'].get(level, 0)
        pct = count / summary['total_scenes'] * 100 if summary['total_scenes'] > 0 else 0
        emoji = level_emoji.get(level, '')
        md += f"| {emoji} {level} | {count} | {pct:.1f}% |\n"
    
    md += f"""
---

## 🔬 偏振統計

### 玻璃區域（高偏振）

| 指標 | 數值 |
|------|------|
| DoLP 平均 | {summary['polarization']['glass_dolp_avg']:.4f} |
| DoLP 範圍 | {summary['polarization']['glass_dolp_min']:.4f} ~ {summary['polarization']['glass_dolp_max']:.4f} |

### 背景區域（低偏振）

| 指標 | 數值 |
|------|------|
| DoLP 平均 | {summary['polarization']['background_dolp_avg']:.4f} |

### 強度比值 (I∥/I⊥)

| 指標 | 數值 |
|------|------|
| 平均 | {summary['polarization']['intensity_ratio_avg']:.1f}x |
| 範圍 | {summary['polarization']['intensity_ratio_min']:.1f}x ~ {summary['polarization']['intensity_ratio_max']:.1f}x |

---

## 📉 噪點統計

| 指標 | 數值 |
|------|------|
| SNR 平均 | {summary['noise']['snr_avg']:.1f}x |
| SNR 範圍 | {summary['noise']['snr_min']:.1f}x ~ {summary['noise']['snr_max']:.1f}x |

---

## ⚖️ 強度平衡（背景區域 I∥/I⊥）

| 指標 | 數值 |
|------|------|
| 平衡場景數 | {summary.get('intensity_balance', {}).get('balanced_count', 'N/A')} / {summary['total_scenes']} |
| 平衡率 | {summary.get('intensity_balance', {}).get('balanced_ratio', 0):.1f}% |
| 背景 I∥/I⊥ 平均 | {summary.get('intensity_balance', {}).get('bg_ratio_avg', 1.0):.2f}x |
| 背景 I∥/I⊥ 範圍 | {summary.get('intensity_balance', {}).get('bg_ratio_min', 1.0):.2f}x ~ {summary.get('intensity_balance', {}).get('bg_ratio_max', 1.0):.2f}x |

> **說明**：背景區域的 I∥/I⊥ 應接近 1.0（0.5~2.0 為平衡）。如果偏離太多，RAFT-Stereo 的雙目對齊會失敗。
> 
> **解決方案**：啟用非偏振背景光（fill_light）讓漫反射表面正常照明。

---

## ⚠️ 警告統計

"""
    
    if summary['warnings']['total'] > 0:
        md += f"共 {summary['warnings']['total']} 個警告：\n\n"
        md += "| 警告類型 | 次數 |\n"
        md += "|---------|------|\n"
        for warning, count in sorted(summary['warnings']['by_type'].items(), key=lambda x: -x[1]):
            md += f"| {warning} | {count} |\n"
    else:
        md += "✅ 無警告\n"
    
    md += """
---

## 💡 建議

"""
    
    pass_rate = summary['quality']['pass_rate']
    if pass_rate >= 90:
        md += """🟢 **數據品質優秀！**

可以直接開始 RAFT-Stereo fine-tuning。
"""
    elif pass_rate >= 70:
        md += """🟡 **數據品質良好**

建議：
1. 檢查未通過的場景，考慮排除或重新渲染
2. 確認警告項目是否需要處理
"""
    elif pass_rate >= 50:
        md += """🟠 **數據品質一般**

建議：
1. 增加 SPP（目前: {spp}）減少噪點
2. 檢查偏振光源設置
3. 確認玻璃材質正確
""".format(spp=summary.get('render_config', {}).get('spp', 'N/A'))
    else:
        md += """🔴 **數據品質不足**

需要處理：
1. 檢查 Mitsuba variant 是否為 polarized
2. 確認 max_depth 足夠（建議 16+）
3. 檢查偏振光源設置
4. 確認玻璃材質為 dielectric/roughdielectric
"""
    
    md += """
---

## 📋 各場景詳情

| 場景 | 分數 | 等級 | 玻璃DoLP | 背景DoLP | I∥/I⊥(玻璃) | I∥/I⊥(背景) | SNR | 警告 |
|------|------|------|----------|----------|-------------|-------------|-----|------|
"""
    
    for r in sorted(reports, key=lambda x: x.get('quality', {}).get('score', 0), reverse=True):
        name = r.get('scene_name', 'N/A')
        score = r.get('quality', {}).get('score', 0)
        level = r.get('quality', {}).get('level', 'N/A')
        glass_dolp = r.get('polarization', {}).get('glass_region', {}).get('dolp_mean', 0)
        bg_dolp = r.get('polarization', {}).get('background_region', {}).get('dolp_mean', 0)
        ratio = r.get('polarization', {}).get('intensity_ratio', {}).get('mean', 0)
        bg_ratio = r.get('intensity_balance', {}).get('background_ratio_mean', 1.0)
        is_balanced = r.get('intensity_balance', {}).get('is_balanced', True)
        snr = r.get('noise', {}).get('snr_polarization', 0)
        warnings = len(r.get('warnings', []))
        
        emoji = level_emoji.get(level, '')
        balance_mark = '✓' if is_balanced else '⚠️'
        md += f"| {name} | {score} | {emoji}{level} | {glass_dolp:.4f} | {bg_dolp:.4f} | {ratio:.1f}x | {bg_ratio:.2f}x{balance_mark} | {snr:.1f}x | {warnings} |\n"
    
    md += """
---

## 📖 判斷標準

### DoLP (偏振度)

| 區域 | 預期值 | 說明 |
|------|--------|------|
| 玻璃區域 | > 0.3 | Fresnel 反射產生偏振 |
| 背景區域 | < 0.1 | 漫反射無偏振 |

### SNR (信噪比)

| SNR | 等級 | 說明 |
|-----|------|------|
| ≥ 10x | 優秀 | 直接訓練 |
| ≥ 2x | 良好 | 直接訓練 |
| ≥ 1x | 可用 | CNN可提取信號 |
| < 1x | 較差 | 建議增加SPP |

### 強度比值 (I∥/I⊥)

| 區域 | 預期值 | 說明 |
|------|--------|------|
| 玻璃區域 | > 2x | 偏振效果明顯 |
| 背景區域 | 0.5~2.0x | 🔥 重要！應接近 1，否則 RAFT 無法對齊 |

### 強度平衡（背景區域）

| 背景 I∥/I⊥ | 狀態 | 說明 |
|------------|------|------|
| 0.5~2.0x | ✓ 平衡 | RAFT 可正常雙目對齊 |
| 0.3~3.0x | ⚠️ 警告 | 可能有問題 |
| < 0.3 或 > 3.0 | ❌ 不平衡 | 需要啟用非偏振背景光 |

---

*報告由 PIDS 工具集自動生成*
"""
    
    return md


def generate_html(reports: List[Dict], summary: Dict, input_dir: str) -> str:
    """生成 HTML 報告"""
    # 簡化版，基於 Markdown 轉換
    md = generate_markdown(reports, summary, input_dir)
    
    # 基本 HTML 包裝
    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PIDS 訓練數據品質報告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1, h2, h3 {{ color: #2c3e50; }}
        h1 {{ border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; margin-top: 30px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 15px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 10px 15px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        code {{
            background: #f1f1f1;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }}
        hr {{ border: none; border-top: 1px solid #ecf0f1; margin: 20px 0; }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 0;
            padding: 10px 20px;
            background: #e8f4f8;
        }}
    </style>
</head>
<body>
"""
    
    # 簡單的 Markdown 轉 HTML
    lines = md.split('\n')
    in_table = False
    in_code = False
    
    for line in lines:
        # 標題
        if line.startswith('# '):
            html += f"<h1>{line[2:]}</h1>\n"
        elif line.startswith('## '):
            html += f"<h2>{line[3:]}</h2>\n"
        elif line.startswith('### '):
            html += f"<h3>{line[4:]}</h3>\n"
        # 表格
        elif line.startswith('|'):
            if not in_table:
                html += "<table>\n"
                in_table = True
            
            cells = [c.strip() for c in line.split('|')[1:-1]]
            
            if all(c.replace('-', '').replace(':', '') == '' for c in cells):
                # 分隔行，跳過
                continue
            elif not in_code and line.count('|') > 2:
                # 判斷是否為表頭
                if html.rstrip().endswith('<table>\n') or html.rstrip().endswith('<table>'):
                    html += "<thead><tr>"
                    for cell in cells:
                        html += f"<th>{cell}</th>"
                    html += "</tr></thead>\n<tbody>\n"
                else:
                    html += "<tr>"
                    for cell in cells:
                        html += f"<td>{cell}</td>"
                    html += "</tr>\n"
        elif in_table and not line.startswith('|'):
            html += "</tbody></table>\n"
            in_table = False
            if line.strip():
                html += f"<p>{line}</p>\n"
        # 水平線
        elif line.startswith('---'):
            if in_table:
                html += "</tbody></table>\n"
                in_table = False
            html += "<hr>\n"
        # 列表
        elif line.startswith('- '):
            html += f"<p>{line}</p>\n"
        elif line.startswith('1. ') or line.startswith('2. ') or line.startswith('3. '):
            html += f"<p>{line}</p>\n"
        # 普通段落
        elif line.strip():
            # 處理內聯代碼
            while '`' in line:
                start = line.find('`')
                end = line.find('`', start + 1)
                if end > start:
                    line = line[:start] + '<code>' + line[start+1:end] + '</code>' + line[end+1:]
                else:
                    break
            html += f"<p>{line}</p>\n"
    
    if in_table:
        html += "</tbody></table>\n"
    
    html += """
</body>
</html>
"""
    
    return html


def main():
    parser = argparse.ArgumentParser(
        description='PIDS 報告匯總腳本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python pids_summarize_reports.py --input_dir ./output_stage1
  python pids_summarize_reports.py --input_dir ./output_stage1 --output ./summary.md
  python pids_summarize_reports.py --input_dir ./output_stage1 --format html
        """
    )
    
    parser.add_argument('--input_dir', '-i', type=str, required=True, 
                        help='包含 *_report.json 的目錄')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='輸出報告路徑')
    parser.add_argument('--format', '-f', type=str, default='markdown',
                        choices=['markdown', 'md', 'html', 'json'],
                        help='輸出格式 (預設: markdown)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"[ERROR] 目錄不存在: {args.input_dir}")
        return 1
    
    print("="*60)
    print("PIDS 報告匯總")
    print("="*60)
    
    # 載入報告
    print(f"\n載入報告中...")
    reports = load_reports(args.input_dir)
    
    if not reports:
        print("[ERROR] 找不到任何 *_report.json 文件")
        print("       請先執行渲染器生成報告")
        return 1
    
    print(f"找到 {len(reports)} 個場景報告")
    
    # 生成摘要
    summary = generate_summary(reports)
    
    # 打印摘要
    print(f"\n{'='*40}")
    print(f"摘要")
    print(f"{'='*40}")
    print(f"總場景: {summary['total_scenes']}")
    print(f"有效場景: {summary['quality']['valid_scenes']}")
    print(f"通過率: {summary['quality']['pass_rate']:.1f}%")
    print(f"平均分數: {summary['quality']['score_avg']:.1f}")
    print(f"平均 SNR: {summary['noise']['snr_avg']:.1f}x")
    print(f"I∥/I⊥ 比值: {summary['polarization']['intensity_ratio_avg']:.1f}x")
    
    # 生成報告
    output_path = args.output
    
    if args.format in ['markdown', 'md']:
        content = generate_markdown(reports, summary, args.input_dir)
        if output_path is None:
            output_path = os.path.join(args.input_dir, 'SUMMARY.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    elif args.format == 'html':
        content = generate_html(reports, summary, args.input_dir)
        if output_path is None:
            output_path = os.path.join(args.input_dir, 'SUMMARY.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    elif args.format == 'json':
        if output_path is None:
            output_path = os.path.join(args.input_dir, 'SUMMARY.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'scenes': reports,
            }, f, indent=2, ensure_ascii=False)
    
    print(f"\n報告已保存: {output_path}")
    
    # 結論
    print(f"\n{'='*40}")
    pass_rate = summary['quality']['pass_rate']
    if pass_rate >= 90:
        print("🟢 結論: 數據品質優秀，可以開始訓練！")
    elif pass_rate >= 70:
        print("🟡 結論: 數據品質良好，建議檢查未通過場景")
    elif pass_rate >= 50:
        print("🟠 結論: 數據品質一般，建議增加 SPP 或檢查設置")
    else:
        print("🔴 結論: 數據品質不足，需要檢查渲染設置")
    print(f"{'='*40}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

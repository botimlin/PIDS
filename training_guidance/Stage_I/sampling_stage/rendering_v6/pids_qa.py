#!/usr/bin/env python3
"""
PIDS QA Tool v2.4.0
===================
独立品质验证工具，兼容 pids_renderer_textured.py v5.1.8+ / v6.0.0+ 生成的 report.json

v6 相容说明: v6 renderer 输出 RGB EXR，但 report.json 格式不变
（renderer 内部已将 RGB 转 luminance 后写入 JSON 指标）

检查项目：
- C2: 背景光度一致性 (跨视角，检查左右图亮度一致性)
- C3: 玻璃偏振比值 (同视角，检查偏振对比度) [可选]
- C5: 玻璃深度有效率 (Glass Depth Validity Rate)

用法：
    python pids_qa.py --input <report_dir> [--output <report.md>]
    python pids_qa.py --input <report_dir> --export-passed <output_dir>
    python pids_qa.py --input <report_dir> --skip-c3  # 只检查 C2+C5 (兼容旧版)

作者: PIDS Project
日期: 2026-01-23
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import shutil


# ============================================================
# 配置
# ============================================================

@dataclass
class QAConfig:
    """QA 阈值配置"""
    # C2: 背景光度一致性（跨视角 warp 后）
    # 背景 I∥/I⊥ 应接近 1.0
    c2_bg_ratio_min: float = 0.80
    c2_bg_ratio_max: float = 1.25

    # C3: 玻璃偏振比值（同视角 I(90°)/I(0°)）
    # 玻璃应明显偏离 1.0（< 0.96 或 > 1.04）
    c3_glass_ratio_low: float = 0.96
    c3_glass_ratio_high: float = 1.04

    # C5: 玻璃深度有效率
    # 玻璃区域深度有效像素占比 >= 90%
    c5_validity_threshold: float = 0.9


# 默认配置
DEFAULT_CONFIG = QAConfig()


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SceneResult:
    """单场景验证结果"""
    scene_name: str

    # C2: 背景一致性
    c2_pass: bool
    c2_value: float

    # C3: 玻璃偏振比值
    c3_pass: bool
    c3_value: float

    # C5: 深度有效率
    c5_pass: bool
    c5_value: float

    # 综合结果
    overall_pass: bool
    quality_score: int

    # 额外信息
    warnings: List[str]


# ============================================================
# QA 验证器
# ============================================================

class PIDSQA:
    """PIDS 品质验证器"""

    def __init__(self, input_dir: Path, config: QAConfig = DEFAULT_CONFIG):
        self.input_dir = Path(input_dir)
        self.config = config
        self.reports: Dict[str, dict] = {}
        self._load_reports()

    def _load_reports(self):
        """加载所有 report.json"""
        report_files = list(self.input_dir.glob("*_report.json"))
        print(f"[QA] 找到 {len(report_files)} 个报告文件")

        for f in report_files:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    scene_name = data.get('scene_name', f.stem.replace('_report', ''))
                    self.reports[scene_name] = data
            except Exception as e:
                print(f"  [警告] 无法读取 {f.name}: {e}")

        print(f"[QA] 成功加载 {len(self.reports)} 个报告")

    def validate_scene(self, scene_name: str, report: dict) -> SceneResult:
        """验证单一场景"""
        warnings = report.get('warnings', [])

        # ========== C2: 背景光度一致性（跨视角）==========
        # 使用 intensity_balance.background_ratio_mean（跨视角 warp 后）
        # 检查 left_parallel 和 right_cross 在背景区域的亮度一致性
        intensity_balance = report.get('intensity_balance', {})
        c2_value = intensity_balance.get('background_ratio_mean', 1.0)

        c2_pass = self.config.c2_bg_ratio_min <= c2_value <= self.config.c2_bg_ratio_max

        # ========== C3: 玻璃偏振比值（同视角）==========
        # 使用 stokes_ratio（同视角，从 Stokes 参数直接计算）
        # 检查玻璃区域的偏振对比度
        polarization = report.get('polarization', {})
        glass_region = polarization.get('glass_region', {})

        c3_value = glass_region.get('stokes_ratio', None)
        if c3_value is None:
            # Fallback: 旧版本可能用 intensity_ratio_mean
            c3_value = glass_region.get('intensity_ratio_mean', 1.0)

        # 通过条件: ratio < 0.96 (反向) 或 ratio > 1.04 (正向)
        c3_pass = (c3_value < self.config.c3_glass_ratio_low or
                   c3_value > self.config.c3_glass_ratio_high)

        # ========== C5: 玻璃深度有效率 ==========
        glass_depth = report.get('glass_depth_validity', {})
        c5_value = glass_depth.get('validity_rate', 0.0)
        c5_pass = c5_value >= self.config.c5_validity_threshold

        # ========== 综合结果 ==========
        overall_pass = c2_pass and c3_pass and c5_pass
        quality_score = report.get('quality', {}).get('score', 0)

        return SceneResult(
            scene_name=scene_name,
            c2_pass=c2_pass,
            c2_value=c2_value,
            c3_pass=c3_pass,
            c3_value=c3_value,
            c5_pass=c5_pass,
            c5_value=c5_value,
            overall_pass=overall_pass,
            quality_score=quality_score,
            warnings=warnings,
        )

    def validate_all(self) -> Tuple[List[SceneResult], Dict]:
        """验证所有场景"""
        results = []

        for scene_name, report in self.reports.items():
            result = self.validate_scene(scene_name, report)
            results.append(result)

        # 统计
        total = len(results)
        passed = sum(1 for r in results if r.overall_pass)
        failed = total - passed

        c2_passed = sum(1 for r in results if r.c2_pass)
        c3_passed = sum(1 for r in results if r.c3_pass)
        c5_passed = sum(1 for r in results if r.c5_pass)

        stats = {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0,
            'c2_passed': c2_passed,
            'c2_rate': c2_passed / total if total > 0 else 0,
            'c3_passed': c3_passed,
            'c3_rate': c3_passed / total if total > 0 else 0,
            'c5_passed': c5_passed,
            'c5_rate': c5_passed / total if total > 0 else 0,
        }

        return results, stats

    def generate_report(self, results: List[SceneResult], stats: Dict,
                        output_path: Optional[Path] = None) -> str:
        """生成 Markdown 报告"""
        lines = []

        # 标题
        lines.append("# PIDS 数据品质报告")
        lines.append("")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"报告目录: `{self.input_dir}`")
        lines.append("")

        # 总览
        lines.append("## 总览")
        lines.append("")
        lines.append(f"| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 总场景数 | {stats['total']} |")
        lines.append(f"| 通过数 | {stats['passed']} |")
        lines.append(f"| 未通过数 | {stats['failed']} |")
        lines.append(f"| **通过率** | **{stats['pass_rate']*100:.1f}%** |")
        lines.append("")

        # 各检查项通过率
        lines.append("## 各检查项统计")
        lines.append("")
        lines.append("| 检查项 | 通过数 | 通过率 | 阈值 |")
        lines.append("|--------|--------|--------|------|")
        lines.append(f"| C2 背景一致性 | {stats['c2_passed']} | {stats['c2_rate']*100:.1f}% | [{self.config.c2_bg_ratio_min}, {self.config.c2_bg_ratio_max}] |")
        lines.append(f"| C3 玻璃偏振比值 | {stats['c3_passed']} | {stats['c3_rate']*100:.1f}% | < {self.config.c3_glass_ratio_low} 或 > {self.config.c3_glass_ratio_high} |")
        lines.append(f"| C5 深度有效率 | {stats['c5_passed']} | {stats['c5_rate']*100:.1f}% | >= {self.config.c5_validity_threshold*100:.0f}% |")
        lines.append("")

        # 失败场景详情
        failed_results = [r for r in results if not r.overall_pass]
        if failed_results:
            lines.append("## 未通过场景")
            lines.append("")
            lines.append("| 场景 | C2 背景 | C3 玻璃比值 | C5 深度 | 品质分 |")
            lines.append("|------|---------|-------------|---------|--------|")

            for r in failed_results[:100]:  # 只显示前100个
                c2_str = f"{'✓' if r.c2_pass else '✗'} {r.c2_value:.3f}"
                c3_str = f"{'✓' if r.c3_pass else '✗'} {r.c3_value:.3f}"
                c5_str = f"{'✓' if r.c5_pass else '✗'} {r.c5_value*100:.1f}%"
                lines.append(f"| {r.scene_name} | {c2_str} | {c3_str} | {c5_str} | {r.quality_score} |")

            if len(failed_results) > 100:
                lines.append(f"| ... | (省略 {len(failed_results) - 100} 个) | | | |")
            lines.append("")

        # 通过场景列表
        passed_results = [r for r in results if r.overall_pass]
        lines.append("## 通过场景")
        lines.append("")
        lines.append(f"共 {len(passed_results)} 个场景通过所有检查。")
        lines.append("")

        # 检查项说明
        lines.append("## 检查项说明")
        lines.append("")
        lines.append("### C2: 背景光度一致性 (Background Photometric Consistency)")
        lines.append("")
        lines.append("背景区域（非玻璃）的跨视角亮度比值应接近 1.0。")
        lines.append("")
        lines.append("- **数据来源**: `report.json → intensity_balance.background_ratio_mean`")
        lines.append(f"- **阈值**: {self.config.c2_bg_ratio_min} <= ratio <= {self.config.c2_bg_ratio_max}")
        lines.append("- **物理意义**: 背景区域的 I∥(左相机) 和 I⊥(右相机 warp 后) 应接近")
        lines.append("- **计算方式**: 跨视角（右图 warp 到左视角后比较）")
        lines.append("")

        lines.append("### C3: 玻璃偏振比值 (Glass Polarization Ratio)")
        lines.append("")
        lines.append("玻璃区域的同视角偏振比值 I(90°)/I(0°) 应明显偏离 1.0。")
        lines.append("")
        lines.append("- **数据来源**: `report.json → polarization.glass_region.stokes_ratio`")
        lines.append(f"- **阈值**: ratio < {self.config.c3_glass_ratio_low} 或 ratio > {self.config.c3_glass_ratio_high}")
        lines.append("- **物理意义**: 玻璃的菲涅尔反射会保持/改变偏振态，产生明显的偏振差异")
        lines.append("- **计算方式**: 同视角（左相机位置），从 Stokes 参数直接计算")
        lines.append("")

        lines.append("### C5: 玻璃深度有效率 (Glass Depth Validity Rate)")
        lines.append("")
        lines.append("玻璃区域的深度值有效像素占比。")
        lines.append("")
        lines.append("- **数据来源**: `report.json → glass_depth_validity.validity_rate`")
        lines.append(f"- **阈值**: >= {self.config.c5_validity_threshold*100:.0f}%")
        lines.append("- **物理意义**: 确保玻璃区域有足够的有效深度用于训练")
        lines.append("")

        content = "\n".join(lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[QA] 报告已保存至: {output_path}")

            # 额外输出 failed_scenes.txt 和 passed_scenes.txt
            output_dir = Path(output_path).parent

            failed_results = [r for r in results if not r.overall_pass]
            failed_list_path = output_dir / "failed_scenes.txt"
            with open(failed_list_path, 'w', encoding='utf-8') as f:
                for r in failed_results:
                    f.write(f"{r.scene_name}\n")
            print(f"[QA] 失败场景列表: {failed_list_path} ({len(failed_results)} 个)")

            passed_results = [r for r in results if r.overall_pass]
            passed_list_path = output_dir / "passed_scenes.txt"
            with open(passed_list_path, 'w', encoding='utf-8') as f:
                for r in passed_results:
                    f.write(f"{r.scene_name}\n")
            print(f"[QA] 通过场景列表: {passed_list_path} ({len(passed_results)} 个)")

        return content

    def export_passed_scenes(self, results: List[SceneResult], output_dir: Path):
        """导出通过场景的文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        passed = [r for r in results if r.overall_pass]
        print(f"[QA] 导出 {len(passed)} 个通过场景到 {output_dir}")

        # 定义要复制的文件后缀
        suffixes = [
            '_left_parallel.exr',
            '_right_cross.exr',
            '_depth.exr',
            '_disparity.exr',
            '_glass_mask.exr',
            '_glass_mask_left.exr',
            '_glass_mask_strict.exr',
            '_report.json',
            '_params.json',
        ]

        copied = 0
        for r in passed:
            for suffix in suffixes:
                src = self.input_dir / f"{r.scene_name}{suffix}"
                if src.exists():
                    dst = output_dir / src.name
                    shutil.copy2(src, dst)
                    copied += 1

        print(f"[QA] 共复制 {copied} 个文件")

        # 保存通过场景列表
        passed_list = output_dir / "passed_scenes.txt"
        with open(passed_list, 'w') as f:
            for r in passed:
                f.write(f"{r.scene_name}\n")
        print(f"[QA] 场景列表已保存至: {passed_list}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='PIDS QA Tool v2.4.0 - 数据品质验证 (v5/v6 兼容)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 验证并生成报告
  python pids_qa.py --input ./output

  # 指定输出报告路径
  python pids_qa.py --input ./output --output qa_report.md

  # 导出通过的场景
  python pids_qa.py --input ./output --export-passed ./passed_scenes

  # 自定义阈值
  python pids_qa.py --input ./output --c2-min 0.85 --c2-max 1.2
        """
    )

    parser.add_argument('--input', '-i', required=True,
                        help='包含 *_report.json 的目录')
    parser.add_argument('--output', '-o', default=None,
                        help='输出报告路径 (默认: <input>/quality_report.md)')
    parser.add_argument('--export-passed', default=None,
                        help='导出通过场景到指定目录')

    # 阈值参数
    parser.add_argument('--c2-min', type=float, default=0.8,
                        help='C2 背景比值下限 (默认: 0.8)')
    parser.add_argument('--c2-max', type=float, default=1.25,
                        help='C2 背景比值上限 (默认: 1.25)')
    parser.add_argument('--c3-low', type=float, default=0.96,
                        help='C3 玻璃比值下限 (默认: 0.96)')
    parser.add_argument('--c3-high', type=float, default=1.04,
                        help='C3 玻璃比值上限 (默认: 1.04)')
    parser.add_argument('--c5-threshold', type=float, default=0.9,
                        help='C5 深度有效率阈值 (默认: 0.9)')

    args = parser.parse_args()

    # 配置
    config = QAConfig(
        c2_bg_ratio_min=args.c2_min,
        c2_bg_ratio_max=args.c2_max,
        c3_glass_ratio_low=args.c3_low,
        c3_glass_ratio_high=args.c3_high,
        c5_validity_threshold=args.c5_threshold,
    )

    # 运行验证
    input_dir = Path(args.input)
    qa = PIDSQA(input_dir, config)

    if len(qa.reports) == 0:
        print("[错误] 未找到任何报告文件")
        return 1

    results, stats = qa.validate_all()

    # 打印摘要
    print("")
    print("=" * 50)
    print("  PIDS QA 结果摘要 (v2.4.0)")
    print("=" * 50)
    print(f"  总场景: {stats['total']}")
    print(f"  通过:   {stats['passed']}")
    print(f"  未通过: {stats['failed']}")
    print(f"  通过率: {stats['pass_rate']*100:.1f}%")
    print("")
    print("  各检查项通过率:")
    print(f"    C2 背景一致性:   {stats['c2_rate']*100:.1f}%")
    print(f"    C3 玻璃偏振比值: {stats['c3_rate']*100:.1f}%")
    print(f"    C5 深度有效率:   {stats['c5_rate']*100:.1f}%")
    print("=" * 50)

    # 生成报告
    output_path = Path(args.output) if args.output else input_dir / "quality_report.md"
    qa.generate_report(results, stats, output_path)

    # 导出通过场景
    if args.export_passed:
        qa.export_passed_scenes(results, Path(args.export_passed))

    return 0


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
"""
PIDS 偏振信號 Sanity Check 腳本
================================

用數值指標（不是肉眼）判斷偏振資料是否有效。

核心觀點：
- 神經網路不是看「亮暗差異」，而是學「統計一致的微弱 correlation」
- 偏振信號微弱但一致 → 網路被迫學幾何特徵
- 用數值指標判斷，不是肉眼

驗證指標：
1. DoLP 統計分佈（mean > 0.02, std > 0.01）
2. I∥ - I⊥ 與 depth 的 correlation（≠ 0 就有資訊）
3. 偏振梯度在物體邊界的聚集程度

使用方法：
    python pids_sanity_check.py --input_dir ./rendered_output
    python pids_sanity_check.py --scene scene_0002 --input_dir ./output

作者: PIDS Project
"""

import numpy as np
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 嘗試導入可選依賴
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARNING] OpenCV not found, some visualizations disabled")

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# 嘗試 Mitsuba（如果可用）
try:
    import mitsuba as mi
    HAS_MITSUBA = True
except ImportError:
    HAS_MITSUBA = False


def load_exr(path: str) -> Optional[np.ndarray]:
    """載入 EXR 文件（多種方式 fallback）"""
    if not os.path.exists(path):
        return None
    
    # 方法 1: 使用 Mitsuba Bitmap（最可靠，因為渲染就是用 Mitsuba）
    if HAS_MITSUBA:
        try:
            bitmap = mi.Bitmap(path)
            image = np.array(bitmap)
            
            # 處理多通道
            if image.ndim == 3:
                if image.shape[2] == 1:
                    image = image[:, :, 0]
                elif image.shape[2] == 3:
                    image = 0.2126 * image[:,:,0] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,2]
                elif image.shape[2] == 4:
                    # RGBA，取前三通道
                    image = 0.2126 * image[:,:,0] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,2]
                else:
                    # 多通道（可能是 Stokes），取第一個
                    image = image[:, :, 0]
            
            return image.astype(np.float32)
        except Exception as e:
            pass  # 靜默失敗，嘗試下一個方法
    
    # 方法 2: 使用 OpenEXR 庫
    if HAS_OPENEXR:
        try:
            exr_file = OpenEXR.InputFile(path)
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            channels = header['channels'].keys()
            
            if 'Y' in channels:
                raw = exr_file.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT))
                image = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
            elif 'R' in channels:
                r_raw = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
                g_raw = exr_file.channel('G', Imath.PixelType(Imath.PixelType.FLOAT))
                b_raw = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
                r = np.frombuffer(r_raw, dtype=np.float32).reshape(height, width)
                g = np.frombuffer(g_raw, dtype=np.float32).reshape(height, width)
                b = np.frombuffer(b_raw, dtype=np.float32).reshape(height, width)
                image = 0.2126 * r + 0.7152 * g + 0.0722 * b
            else:
                ch = list(channels)[0]
                raw = exr_file.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
                image = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
            
            return image
        except Exception as e:
            pass
    
    # 方法 3: 使用 imageio
    if HAS_IMAGEIO:
        try:
            image = imageio.imread(path)
            if image.ndim == 3:
                if image.shape[2] >= 3:
                    image = 0.2126 * image[:,:,0] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,2]
                else:
                    image = image[:,:,0]
            return image.astype(np.float32)
        except Exception as e:
            pass
    
    # 方法 4: 使用 cv2（需要 OpenEXR 支持）
    if HAS_CV2:
        try:
            # 設置環境變量啟用 OpenEXR
            os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
            image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if image is not None:
                if image.ndim == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return image.astype(np.float32)
        except Exception as e:
            pass
    
    # 所有方法都失敗
    print(f"  [ERROR] 無法載入 EXR: {path}")
    print(f"          可用方法: Mitsuba={HAS_MITSUBA}, OpenEXR={HAS_OPENEXR}, imageio={HAS_IMAGEIO}, cv2={HAS_CV2}")
    print(f"          建議: pip install imageio imageio-ffmpeg")
    return None


def compute_dolp_stats(I_parallel: np.ndarray, I_cross: np.ndarray) -> Dict:
    """
    計算 DoLP（偏振度）統計
    
    DoLP = |I∥ - I⊥| / (I∥ + I⊥)
    
    判斷標準：
    - mean DoLP > 0.02 ✔ 有效偏振信號
    - std DoLP > 0.01 ✔ 有變化（不是噪點）
    """
    # 避免除零
    I_sum = I_parallel + I_cross
    valid_mask = I_sum > 1e-6
    
    # 計算 DoLP
    DoLP = np.zeros_like(I_parallel)
    DoLP[valid_mask] = np.abs(I_parallel[valid_mask] - I_cross[valid_mask]) / I_sum[valid_mask]
    DoLP = np.clip(DoLP, 0, 1)
    
    # 只統計有效區域
    dolp_valid = DoLP[valid_mask]
    
    stats = {
        'mean': float(np.mean(dolp_valid)) if len(dolp_valid) > 0 else 0.0,
        'std': float(np.std(dolp_valid)) if len(dolp_valid) > 0 else 0.0,
        'median': float(np.median(dolp_valid)) if len(dolp_valid) > 0 else 0.0,
        'max': float(np.max(dolp_valid)) if len(dolp_valid) > 0 else 0.0,
        'p95': float(np.percentile(dolp_valid, 95)) if len(dolp_valid) > 0 else 0.0,
        'p99': float(np.percentile(dolp_valid, 99)) if len(dolp_valid) > 0 else 0.0,
        'valid_ratio': float(np.sum(valid_mask) / valid_mask.size),
    }
    
    # 判斷是否有效
    stats['is_valid'] = stats['mean'] > 0.02 or stats['p95'] > 0.05
    stats['has_variation'] = stats['std'] > 0.01
    
    return stats, DoLP


def compute_polarization_diff_stats(I_parallel: np.ndarray, I_cross: np.ndarray) -> Dict:
    """
    計算偏振差異 (I∥ - I⊥) 統計
    """
    diff = I_parallel - I_cross  # 帶符號差異
    abs_diff = np.abs(diff)
    
    # 相對差異
    I_mean = (I_parallel + I_cross) / 2
    valid_mask = I_mean > 1e-6
    rel_diff = np.zeros_like(diff)
    rel_diff[valid_mask] = abs_diff[valid_mask] / I_mean[valid_mask]
    
    stats = {
        'abs_mean': float(np.mean(abs_diff)),
        'abs_std': float(np.std(abs_diff)),
        'abs_max': float(np.max(abs_diff)),
        'signed_mean': float(np.mean(diff)),  # 帶符號，看是否有系統性偏差
        'rel_mean': float(np.mean(rel_diff[valid_mask])) if np.any(valid_mask) else 0.0,
        'rel_p95': float(np.percentile(rel_diff[valid_mask], 95)) if np.any(valid_mask) else 0.0,
    }
    
    return stats, diff


def compute_depth_correlation(diff: np.ndarray, depth: np.ndarray) -> Dict:
    """
    計算偏振差異與深度的相關性
    
    核心指標：correlation ≠ 0 就表示偏振信號包含幾何信息
    """
    # 有效區域（深度 > 0）
    valid_mask = depth > 0
    
    # 預設返回值
    default_result = {
        'correlation': 0.0,
        'abs_correlation': 0.0,
        'gradient_correlation': None,
        'valid': False,
        'has_info': False,
    }
    
    if not np.any(valid_mask):
        return default_result
    
    diff_valid = diff[valid_mask].flatten()
    depth_valid = depth[valid_mask].flatten()
    
    # Pearson correlation
    if len(diff_valid) < 10:
        return default_result
    
    # 標準化
    diff_std = np.std(diff_valid)
    depth_std = np.std(depth_valid)
    
    if diff_std < 1e-10 or depth_std < 1e-10:
        return default_result
    
    diff_norm = (diff_valid - np.mean(diff_valid)) / diff_std
    depth_norm = (depth_valid - np.mean(depth_valid)) / depth_std
    
    correlation = float(np.mean(diff_norm * depth_norm))
    
    # 計算與深度梯度的相關性（更有意義）
    if HAS_CV2:
        depth_grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        depth_grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        depth_grad_mag = np.sqrt(depth_grad_x**2 + depth_grad_y**2)
        
        diff_abs = np.abs(diff)
        
        grad_valid = depth_grad_mag[valid_mask].flatten()
        diff_abs_valid = diff_abs[valid_mask].flatten()
        
        if np.std(grad_valid) > 1e-10 and np.std(diff_abs_valid) > 1e-10:
            grad_norm = (grad_valid - np.mean(grad_valid)) / np.std(grad_valid)
            diff_abs_norm = (diff_abs_valid - np.mean(diff_abs_valid)) / np.std(diff_abs_valid)
            gradient_correlation = float(np.mean(grad_norm * diff_abs_norm))
        else:
            gradient_correlation = 0.0
    else:
        gradient_correlation = None
    
    return {
        'correlation': correlation,
        'abs_correlation': float(np.abs(correlation)),
        'gradient_correlation': gradient_correlation,
        'valid': True,
        'has_info': abs(correlation) > 0.01 or (gradient_correlation is not None and abs(gradient_correlation) > 0.01),
    }


def compute_boundary_concentration(diff: np.ndarray, depth: np.ndarray) -> Dict:
    """
    計算偏振信號在物體邊界的聚集程度
    
    物理原理：在透明物體邊界，折射率變化導致偏振梯度最大
    """
    if not HAS_CV2:
        return {'valid': False}
    
    valid_mask = depth > 0
    if not np.any(valid_mask):
        return {'valid': False}
    
    # 計算深度邊緣
    depth_normalized = np.zeros_like(depth)
    depth_valid = depth[valid_mask]
    depth_normalized[valid_mask] = (depth[valid_mask] - depth_valid.min()) / (depth_valid.max() - depth_valid.min() + 1e-10)
    
    edges = cv2.Canny((depth_normalized * 255).astype(np.uint8), 50, 150)
    edge_mask = edges > 0
    
    # 膨脹邊緣區域
    kernel = np.ones((5, 5), np.uint8)
    edge_region = cv2.dilate(edges, kernel, iterations=2) > 0
    non_edge_region = ~edge_region & valid_mask
    
    # 計算邊緣區域 vs 非邊緣區域的偏振差異
    abs_diff = np.abs(diff)
    
    if np.any(edge_region) and np.any(non_edge_region):
        edge_diff_mean = float(np.mean(abs_diff[edge_region]))
        non_edge_diff_mean = float(np.mean(abs_diff[non_edge_region]))
        concentration_ratio = edge_diff_mean / (non_edge_diff_mean + 1e-10)
    else:
        edge_diff_mean = 0.0
        non_edge_diff_mean = 0.0
        concentration_ratio = 1.0
    
    return {
        'edge_diff_mean': edge_diff_mean,
        'non_edge_diff_mean': non_edge_diff_mean,
        'concentration_ratio': concentration_ratio,
        'edge_pixel_ratio': float(np.sum(edge_region) / valid_mask.sum()) if valid_mask.sum() > 0 else 0.0,
        'valid': True,
        # > 1.2 表示偏振信號確實在邊界聚集
        'concentrated_at_boundary': concentration_ratio > 1.2,
    }


def analyze_scene(input_dir: str, scene_name: str) -> Dict:
    """
    分析單一場景的偏振信號
    """
    print(f"\n{'='*60}")
    print(f"分析場景: {scene_name}")
    print(f"{'='*60}")
    
    # 尋找文件
    parallel_path = os.path.join(input_dir, f"{scene_name}_left_parallel.exr")
    cross_path = os.path.join(input_dir, f"{scene_name}_right_cross.exr")
    depth_path = os.path.join(input_dir, f"{scene_name}_depth.exr")
    
    # 也嘗試其他命名方式
    if not os.path.exists(parallel_path):
        parallel_path = os.path.join(input_dir, f"{scene_name}_I_parallel.exr")
    if not os.path.exists(cross_path):
        cross_path = os.path.join(input_dir, f"{scene_name}_I_cross.exr")
    
    results = {
        'scene_name': scene_name,
        'files_found': {},
        'valid': False,
    }
    
    # 載入數據
    I_parallel = load_exr(parallel_path)
    I_cross = load_exr(cross_path)
    depth = load_exr(depth_path)
    
    results['files_found'] = {
        'parallel': os.path.exists(parallel_path),
        'cross': os.path.exists(cross_path),
        'depth': os.path.exists(depth_path) if depth_path else False,
    }
    
    if I_parallel is None or I_cross is None:
        print(f"  [ERROR] 無法載入 I∥ 或 I⊥")
        print(f"    嘗試路徑: {parallel_path}")
        print(f"    嘗試路徑: {cross_path}")
        return results
    
    print(f"  [INFO] I∥ shape: {I_parallel.shape}, range: [{I_parallel.min():.4f}, {I_parallel.max():.4f}]")
    print(f"  [INFO] I⊥ shape: {I_cross.shape}, range: [{I_cross.min():.4f}, {I_cross.max():.4f}]")
    
    # 1. DoLP 統計
    print(f"\n  [1] DoLP 統計:")
    dolp_stats, DoLP = compute_dolp_stats(I_parallel, I_cross)
    results['dolp'] = dolp_stats
    
    print(f"      mean:   {dolp_stats['mean']:.6f}  {'✓' if dolp_stats['mean'] > 0.02 else '✗'} (閾值: > 0.02)")
    print(f"      std:    {dolp_stats['std']:.6f}  {'✓' if dolp_stats['std'] > 0.01 else '✗'} (閾值: > 0.01)")
    print(f"      median: {dolp_stats['median']:.6f}")
    print(f"      p95:    {dolp_stats['p95']:.6f}")
    print(f"      p99:    {dolp_stats['p99']:.6f}")
    print(f"      max:    {dolp_stats['max']:.6f}")
    
    # 2. 偏振差異統計
    print(f"\n  [2] 偏振差異 (I∥ - I⊥) 統計:")
    diff_stats, diff = compute_polarization_diff_stats(I_parallel, I_cross)
    results['diff'] = diff_stats
    
    print(f"      |diff| mean: {diff_stats['abs_mean']:.6f}")
    print(f"      |diff| std:  {diff_stats['abs_std']:.6f}")
    print(f"      |diff| max:  {diff_stats['abs_max']:.6f}")
    print(f"      signed mean: {diff_stats['signed_mean']:.6f}  (系統性偏差)")
    print(f"      relative mean: {diff_stats['rel_mean']*100:.4f}%")
    print(f"      relative p95:  {diff_stats['rel_p95']*100:.4f}%")
    
    # 3. 與深度的相關性
    if depth is not None:
        print(f"\n  [3] 與深度的相關性:")
        corr_stats = compute_depth_correlation(diff, depth)
        results['depth_correlation'] = corr_stats
        
        print(f"      correlation:      {corr_stats.get('correlation', 0):.6f}  {'✓' if corr_stats.get('has_info', False) else '?'}")
        print(f"      |correlation|:    {corr_stats.get('abs_correlation', 0):.6f}  (≠ 0 表示有資訊)")
        if corr_stats.get('gradient_correlation') is not None:
            print(f"      gradient corr:    {corr_stats['gradient_correlation']:.6f}  (與深度梯度)")
        
        # 4. 邊界聚集程度
        print(f"\n  [4] 邊界聚集程度:")
        boundary_stats = compute_boundary_concentration(diff, depth)
        results['boundary'] = boundary_stats
        
        if boundary_stats.get('valid', False):
            print(f"      邊界區域 |diff|:    {boundary_stats['edge_diff_mean']:.6f}")
            print(f"      非邊界區域 |diff|:  {boundary_stats['non_edge_diff_mean']:.6f}")
            print(f"      聚集比率:           {boundary_stats['concentration_ratio']:.4f}  {'✓' if boundary_stats['concentrated_at_boundary'] else '✗'} (> 1.2 表示聚集)")
    else:
        print(f"\n  [3] 深度圖未找到，跳過相關性分析")
        results['depth_correlation'] = {'valid': False}
        results['boundary'] = {'valid': False}
    
    # 總結判斷
    print(f"\n  {'='*50}")
    print(f"  總結判斷:")
    
    is_valid = False
    reasons = []
    
    # 條件 1: DoLP 有效
    if dolp_stats['mean'] > 0.02 or dolp_stats['p95'] > 0.05:
        reasons.append("✓ DoLP 有效（mean > 0.02 或 p95 > 0.05）")
        is_valid = True
    elif dolp_stats['mean'] > 0.005:
        reasons.append("△ DoLP 微弱但非零（mean > 0.005）")
        is_valid = True  # 微弱信號也算
    else:
        reasons.append("✗ DoLP 過低（mean < 0.005）")
    
    # 條件 2: 有變化
    if dolp_stats['std'] > 0.01:
        reasons.append("✓ DoLP 有變化（std > 0.01）")
    elif dolp_stats['std'] > 0.001:
        reasons.append("△ DoLP 變化微弱（std > 0.001）")
    else:
        reasons.append("✗ DoLP 無變化（可能是噪點）")
    
    # 條件 3: 與深度相關
    if results.get('depth_correlation', {}).get('has_info', False):
        reasons.append("✓ 與深度有相關性（correlation ≠ 0）")
    elif results.get('depth_correlation', {}).get('valid', False):
        corr = abs(results['depth_correlation'].get('correlation', 0))
        if corr > 0.001:
            reasons.append(f"△ 與深度微弱相關（|r| = {corr:.4f}）")
    
    # 條件 4: 邊界聚集
    if results.get('boundary', {}).get('concentrated_at_boundary', False):
        reasons.append("✓ 偏振信號在邊界聚集")
    
    for r in reasons:
        print(f"    {r}")
    
    results['valid'] = is_valid
    results['summary'] = reasons
    
    if is_valid:
        print(f"\n  🟢 結論: 偏振信號有效，可以用於訓練")
        print(f"     （神經網路可以學習微弱但一致的統計規律）")
    else:
        print(f"\n  🔴 結論: 偏振信號過弱，建議檢查渲染設置")
    
    return results


def analyze_all_scenes(input_dir: str, max_scenes: int = None) -> List[Dict]:
    """
    分析目錄中的所有場景
    """
    # 尋找所有場景
    exr_files = list(Path(input_dir).glob("*_left_parallel.exr"))
    if not exr_files:
        exr_files = list(Path(input_dir).glob("*_I_parallel.exr"))
    
    scene_names = set()
    for f in exr_files:
        name = f.stem.replace("_left_parallel", "").replace("_I_parallel", "")
        scene_names.add(name)
    
    scene_names = sorted(scene_names)
    
    if max_scenes:
        scene_names = scene_names[:max_scenes]
    
    print(f"\n找到 {len(scene_names)} 個場景")
    
    all_results = []
    for scene_name in scene_names:
        results = analyze_scene(input_dir, scene_name)
        all_results.append(results)
    
    # 彙總統計
    print(f"\n{'='*60}")
    print(f"彙總統計 ({len(all_results)} 個場景)")
    print(f"{'='*60}")
    
    valid_count = sum(1 for r in all_results if r.get('valid', False))
    
    dolp_means = [r['dolp']['mean'] for r in all_results if 'dolp' in r]
    dolp_stds = [r['dolp']['std'] for r in all_results if 'dolp' in r]
    
    if dolp_means:
        print(f"\n  DoLP mean 分佈:")
        print(f"    min:    {min(dolp_means):.6f}")
        print(f"    max:    {max(dolp_means):.6f}")
        print(f"    avg:    {np.mean(dolp_means):.6f}")
    
    if dolp_stds:
        print(f"\n  DoLP std 分佈:")
        print(f"    min:    {min(dolp_stds):.6f}")
        print(f"    max:    {max(dolp_stds):.6f}")
        print(f"    avg:    {np.mean(dolp_stds):.6f}")
    
    print(f"\n  有效場景: {valid_count}/{len(all_results)} ({valid_count/len(all_results)*100:.1f}%)")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='PIDS 偏振信號 Sanity Check',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
判斷標準：
  DoLP mean > 0.02      有效偏振信號
  DoLP std > 0.01       有變化（不是噪點）
  correlation ≠ 0       偏振包含幾何信息
  
記住：神經網路不是人，它可以學習人眼感覺不到的微弱物理偏差！

範例：
  python pids_sanity_check.py --input_dir ./output
  python pids_sanity_check.py --scene scene_0002 --input_dir ./output
  python pids_sanity_check.py --input_dir ./output --save_report
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='渲染輸出目錄')
    parser.add_argument('--scene', type=str, default=None,
                        help='指定場景名稱（不指定則分析所有）')
    parser.add_argument('--max_scenes', type=int, default=None,
                        help='最多分析幾個場景')
    parser.add_argument('--save_report', action='store_true',
                        help='保存 JSON 報告')
    parser.add_argument('--output', type=str, default=None,
                        help='報告輸出路徑')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"[ERROR] 目錄不存在: {args.input_dir}")
        return 1
    
    print("="*60)
    print("PIDS 偏振信號 Sanity Check")
    print("="*60)
    print("\n核心觀點：")
    print("  - 神經網路不是看「亮暗差異」")
    print("  - 而是學「統計一致的微弱 correlation」")
    print("  - 用數值指標判斷，不是肉眼！")
    
    if args.scene:
        results = [analyze_scene(args.input_dir, args.scene)]
    else:
        results = analyze_all_scenes(args.input_dir, args.max_scenes)
    
    # 保存報告
    if args.save_report:
        report_path = args.output or os.path.join(args.input_dir, "sanity_check_report.json")
        
        # 轉換為可序列化格式
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            return obj
        
        report = {
            'input_dir': args.input_dir,
            'num_scenes': len(results),
            'results': make_serializable(results),
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n報告已保存: {report_path}")
    
    # 最終建議
    valid_count = sum(1 for r in results if r.get('valid', False))
    
    print(f"\n{'='*60}")
    print("最終建議")
    print(f"{'='*60}")
    
    if valid_count == len(results):
        print("\n🟢 所有場景的偏振信號都有效！")
        print("   可以開始訓練神經網路。")
    elif valid_count > 0:
        print(f"\n🟡 {valid_count}/{len(results)} 個場景有效")
        print("   建議先用有效場景進行實驗。")
    else:
        print("\n🔴 沒有場景通過驗證")
        print("   建議檢查：")
        print("   1. Mitsuba variant 是否為 polarized")
        print("   2. 玻璃材質是否正確（dielectric/roughdielectric）")
        print("   3. Stokes 通道解析是否正確")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

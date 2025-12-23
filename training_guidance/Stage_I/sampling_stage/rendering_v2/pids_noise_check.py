#!/usr/bin/env python3
"""
PIDS 渲染噪聲檢測腳本
=====================

分析 Monte Carlo 渲染的噪點水平，判斷是否會影響偏振信號。

核心問題：
  噪點強度 vs 偏振信號強度
  如果噪點 >> 信號 → 需要增加 SPP

檢測指標：
1. 噪點標準差估算
2. SNR (Signal-to-Noise Ratio)
3. 噪點 vs DoLP 比值
4. 建議的 SPP 倍數

使用方法：
    python pids_noise_check.py --input_dir ./rendered_output
    python pids_noise_check.py --scene scene_0002 --input_dir ./output

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

# 嘗試導入依賴
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mitsuba as mi
    HAS_MITSUBA = True
except ImportError:
    HAS_MITSUBA = False


def load_exr(path: str) -> Optional[np.ndarray]:
    """載入 EXR 文件"""
    if not os.path.exists(path):
        return None
    
    # 優先使用 Mitsuba
    if HAS_MITSUBA:
        try:
            bitmap = mi.Bitmap(path)
            image = np.array(bitmap)
            
            if image.ndim == 3:
                if image.shape[2] == 1:
                    image = image[:, :, 0]
                elif image.shape[2] >= 3:
                    image = 0.2126 * image[:,:,0] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,2]
            
            return image.astype(np.float32)
        except Exception as e:
            pass
    
    # Fallback: cv2
    if HAS_CV2:
        try:
            os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
            image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if image is not None:
                if image.ndim == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return image.astype(np.float32)
        except:
            pass
    
    return None


def estimate_noise_laplacian(image: np.ndarray) -> float:
    """
    使用 Laplacian 方法估算噪點標準差
    
    原理：Laplacian 算子對噪點敏感，對平滑區域響應小
    σ_noise ≈ σ(Laplacian) / √(4/3)
    """
    if not HAS_CV2:
        return estimate_noise_local_variance(image)
    
    # 確保圖像是 2D
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    
    # 轉換為 float64 以確保兼容性
    image_f64 = image.astype(np.float64)
    
    # Laplacian
    try:
        laplacian = cv2.Laplacian(image_f64, cv2.CV_64F)
    except cv2.error:
        # Fallback: 使用手動 Laplacian kernel
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        laplacian = cv2.filter2D(image_f64, cv2.CV_64F, kernel)
    
    # 使用 MAD (Median Absolute Deviation) 估算，比 std 更穩健
    laplacian_flat = laplacian.flatten()
    mad = np.median(np.abs(laplacian_flat - np.median(laplacian_flat)))
    
    # 轉換為標準差估算（假設高斯噪點）
    # σ ≈ MAD / 0.6745
    sigma = mad / 0.6745
    
    # Laplacian 的噪點放大係數
    sigma_noise = sigma / np.sqrt(4/3)
    
    return float(sigma_noise)


def estimate_noise_local_variance(image: np.ndarray, block_size: int = 8) -> float:
    """
    使用局部方差方法估算噪點
    
    原理：在「平滑」區域，局部方差主要來自噪點
    """
    # 確保圖像是 2D
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    
    h, w = image.shape[:2]
    
    # 計算局部方差
    local_vars = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = image[i:i+block_size, j:j+block_size]
            local_vars.append(np.var(block))
    
    if len(local_vars) == 0:
        return 0.0
    
    local_vars = np.array(local_vars)
    
    # 取最小的 10% 局部方差（假設這些是平滑區域）
    n_smooth = max(1, len(local_vars) // 10)
    smooth_vars = np.sort(local_vars)[:n_smooth]
    
    # 噪點方差 ≈ 平滑區域的局部方差
    noise_var = np.mean(smooth_vars)
    noise_std = np.sqrt(noise_var)
    
    return float(noise_std)


def estimate_noise_difference(image: np.ndarray) -> float:
    """
    使用相鄰像素差異估算噪點
    
    原理：相鄰像素應該相似，差異主要來自噪點
    """
    # 確保圖像是 2D
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    
    # 水平差異
    diff_h = np.abs(image[:, 1:] - image[:, :-1])
    
    # 垂直差異
    diff_v = np.abs(image[1:, :] - image[:-1, :])
    
    # 使用 MAD 估算（更穩健）
    mad_h = np.median(diff_h)
    mad_v = np.median(diff_v)
    
    # 相鄰像素差異的期望值 ≈ √2 * σ_noise
    sigma_h = mad_h / (np.sqrt(2) * 0.6745)
    sigma_v = mad_v / (np.sqrt(2) * 0.6745)
    
    return float((sigma_h + sigma_v) / 2)


def compute_snr(image: np.ndarray, noise_std: float) -> Dict:
    """
    計算信噪比 (Signal-to-Noise Ratio)
    """
    # 確保圖像是 2D
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    
    valid_mask = image > 0
    
    if not np.any(valid_mask):
        return {'snr_db': 0, 'snr_linear': 0, 'signal_mean': 0, 'signal_std': 0}
    
    signal_mean = np.mean(image[valid_mask])
    signal_std = np.std(image[valid_mask])
    
    # SNR = signal / noise
    snr_linear = signal_mean / (noise_std + 1e-10)
    snr_db = 20 * np.log10(snr_linear + 1e-10)
    
    return {
        'snr_linear': float(snr_linear),
        'snr_db': float(snr_db),
        'signal_mean': float(signal_mean),
        'signal_std': float(signal_std),
    }


def compute_polarization_snr(I_parallel: np.ndarray, I_cross: np.ndarray, 
                              noise_std: float) -> Dict:
    """
    計算偏振信號的 SNR
    
    這是最重要的指標：偏振差異 vs 噪點
    """
    # 確保圖像是 2D
    if I_parallel.ndim == 3:
        I_parallel = np.mean(I_parallel, axis=2)
    if I_cross.ndim == 3:
        I_cross = np.mean(I_cross, axis=2)
    
    # 偏振差異
    diff = I_parallel - I_cross
    abs_diff = np.abs(diff)
    
    # DoLP
    I_sum = I_parallel + I_cross
    valid_mask = I_sum > 1e-6
    
    dolp = np.zeros_like(I_parallel)
    dolp[valid_mask] = abs_diff[valid_mask] / I_sum[valid_mask]
    
    # 偏振信號強度
    signal_dolp_mean = np.mean(dolp[valid_mask]) if np.any(valid_mask) else 0
    signal_diff_mean = np.mean(abs_diff[valid_mask]) if np.any(valid_mask) else 0
    
    # 噪點傳播到 DoLP
    # DoLP = |I∥ - I⊥| / (I∥ + I⊥)
    # σ(DoLP) ≈ √2 * σ_noise / I_mean （一階近似）
    I_mean = np.mean(I_sum[valid_mask]) / 2 if np.any(valid_mask) else 1
    noise_dolp = np.sqrt(2) * noise_std / (I_mean + 1e-10)
    
    # 噪點傳播到差異
    # σ(I∥ - I⊥) = √2 * σ_noise
    noise_diff = np.sqrt(2) * noise_std
    
    # SNR
    snr_dolp = signal_dolp_mean / (noise_dolp + 1e-10)
    snr_diff = signal_diff_mean / (noise_diff + 1e-10)
    
    return {
        'signal_dolp_mean': float(signal_dolp_mean),
        'signal_diff_mean': float(signal_diff_mean),
        'noise_dolp': float(noise_dolp),
        'noise_diff': float(noise_diff),
        'snr_dolp': float(snr_dolp),
        'snr_diff': float(snr_diff),
        'noise_dominates': snr_dolp < 1.0,  # 噪點比信號大
    }


def estimate_required_spp_multiplier(current_snr: float, target_snr: float = 2.0) -> float:
    """
    估算需要增加多少倍 SPP
    
    噪點 σ ∝ 1/√SPP
    所以要將 SNR 提高 k 倍，需要 SPP 增加 k² 倍
    """
    if current_snr >= target_snr:
        return 1.0
    
    ratio = target_snr / (current_snr + 1e-10)
    spp_multiplier = ratio ** 2
    
    return float(min(spp_multiplier, 100))  # 最多建議 100 倍


def analyze_noise(input_dir: str, scene_name: str) -> Dict:
    """
    分析單一場景的噪點水平
    """
    print(f"\n{'='*60}")
    print(f"噪聲分析: {scene_name}")
    print(f"{'='*60}")
    
    # 尋找文件
    parallel_path = os.path.join(input_dir, f"{scene_name}_left_parallel.exr")
    cross_path = os.path.join(input_dir, f"{scene_name}_right_cross.exr")
    
    if not os.path.exists(parallel_path):
        parallel_path = os.path.join(input_dir, f"{scene_name}_I_parallel.exr")
    if not os.path.exists(cross_path):
        cross_path = os.path.join(input_dir, f"{scene_name}_I_cross.exr")
    
    results = {
        'scene_name': scene_name,
        'valid': False,
    }
    
    # 載入數據
    I_parallel = load_exr(parallel_path)
    I_cross = load_exr(cross_path)
    
    if I_parallel is None or I_cross is None:
        print(f"  [ERROR] 無法載入圖像")
        return results
    
    print(f"  [INFO] 圖像尺寸: {I_parallel.shape}")
    print(f"  [INFO] I∥ 範圍: [{I_parallel.min():.4f}, {I_parallel.max():.4f}]")
    print(f"  [INFO] I⊥ 範圍: [{I_cross.min():.4f}, {I_cross.max():.4f}]")
    
    # ============================================================
    # 1. 估算噪點水平
    # ============================================================
    print(f"\n  [1] 噪點估算:")
    
    # 多種方法估算，取中位數
    noise_estimates = []
    
    # Laplacian 方法
    noise_lap_parallel = estimate_noise_laplacian(I_parallel)
    noise_lap_cross = estimate_noise_laplacian(I_cross)
    noise_laplacian = (noise_lap_parallel + noise_lap_cross) / 2
    noise_estimates.append(noise_laplacian)
    print(f"      Laplacian 方法:    σ = {noise_laplacian:.6f}")
    
    # 局部方差方法
    noise_var_parallel = estimate_noise_local_variance(I_parallel)
    noise_var_cross = estimate_noise_local_variance(I_cross)
    noise_local_var = (noise_var_parallel + noise_var_cross) / 2
    noise_estimates.append(noise_local_var)
    print(f"      局部方差方法:      σ = {noise_local_var:.6f}")
    
    # 相鄰像素方法
    noise_diff_parallel = estimate_noise_difference(I_parallel)
    noise_diff_cross = estimate_noise_difference(I_cross)
    noise_neighbor = (noise_diff_parallel + noise_diff_cross) / 2
    noise_estimates.append(noise_neighbor)
    print(f"      相鄰像素方法:      σ = {noise_neighbor:.6f}")
    
    # 使用中位數作為最終估算
    noise_std = float(np.median(noise_estimates))
    print(f"      ---------------------------")
    print(f"      最終估算:          σ = {noise_std:.6f}")
    
    results['noise'] = {
        'laplacian': noise_laplacian,
        'local_variance': noise_local_var,
        'neighbor_diff': noise_neighbor,
        'estimated_std': noise_std,
    }
    
    # ============================================================
    # 2. 計算圖像 SNR
    # ============================================================
    print(f"\n  [2] 圖像 SNR:")
    
    snr_parallel = compute_snr(I_parallel, noise_std)
    snr_cross = compute_snr(I_cross, noise_std)
    
    print(f"      I∥ SNR: {snr_parallel['snr_db']:.1f} dB ({snr_parallel['snr_linear']:.1f}x)")
    print(f"      I⊥ SNR: {snr_cross['snr_db']:.1f} dB ({snr_cross['snr_linear']:.1f}x)")
    
    results['image_snr'] = {
        'parallel': snr_parallel,
        'cross': snr_cross,
    }
    
    # ============================================================
    # 3. 計算偏振信號 SNR（最重要！）
    # ============================================================
    print(f"\n  [3] 偏振信號 SNR（關鍵指標）:")
    
    pol_snr = compute_polarization_snr(I_parallel, I_cross, noise_std)
    results['polarization_snr'] = pol_snr
    
    print(f"      偏振信號 (DoLP):   {pol_snr['signal_dolp_mean']:.6f}")
    print(f"      噪點 (DoLP):       {pol_snr['noise_dolp']:.6f}")
    print(f"      SNR (DoLP):        {pol_snr['snr_dolp']:.2f}x")
    print(f"      ")
    print(f"      偏振信號 (|diff|): {pol_snr['signal_diff_mean']:.6f}")
    print(f"      噪點 (diff):       {pol_snr['noise_diff']:.6f}")
    print(f"      SNR (diff):        {pol_snr['snr_diff']:.2f}x")
    
    # ============================================================
    # 4. 判斷與建議
    # ============================================================
    print(f"\n  [4] 判斷:")
    
    snr_dolp = pol_snr['snr_dolp']
    
    if snr_dolp >= 2.0:
        status = "🟢 優秀"
        advice = "噪點水平良好，偏振信號清晰可辨"
        spp_advice = "不需要增加 SPP"
    elif snr_dolp >= 1.0:
        status = "🟡 可用"
        advice = "噪點與信號相當，Network 可能需要更多努力學習"
        spp_multiplier = estimate_required_spp_multiplier(snr_dolp, 2.0)
        spp_advice = f"建議 SPP 增加 {spp_multiplier:.1f}x 以達到 SNR=2"
    elif snr_dolp >= 0.5:
        status = "🟠 勉強"
        advice = "噪點較大，但系統性信號可能還在"
        spp_multiplier = estimate_required_spp_multiplier(snr_dolp, 2.0)
        spp_advice = f"建議 SPP 增加 {spp_multiplier:.1f}x"
    else:
        status = "🔴 噪點過大"
        advice = "偏振信號被噪點淹沒"
        spp_multiplier = estimate_required_spp_multiplier(snr_dolp, 2.0)
        spp_advice = f"強烈建議 SPP 增加 {spp_multiplier:.1f}x"
    
    print(f"      狀態: {status}")
    print(f"      說明: {advice}")
    print(f"      建議: {spp_advice}")
    
    results['judgment'] = {
        'status': status,
        'advice': advice,
        'spp_advice': spp_advice,
        'snr_dolp': snr_dolp,
        'recommended_spp_multiplier': estimate_required_spp_multiplier(snr_dolp, 2.0),
    }
    
    # ============================================================
    # 5. 但是！Network 可能還是學得到
    # ============================================================
    print(f"\n  [5] Network 學習能力評估:")
    
    # 即使 SNR < 1，如果有以下特徵，Network 還是可能學得到：
    # - 噪點是隨機的（會被 convolution 平均）
    # - 信號有空間結構（與幾何相關）
    
    # 確保圖像是 2D
    I_par_2d = np.mean(I_parallel, axis=2) if I_parallel.ndim == 3 else I_parallel
    I_cro_2d = np.mean(I_cross, axis=2) if I_cross.ndim == 3 else I_cross
    
    # 檢查信號是否有空間結構
    if HAS_CV2:
        # 用大 kernel 平均，如果偏振差異仍然存在，說明有結構
        diff = np.abs(I_par_2d - I_cro_2d).astype(np.float32)
        
        try:
            diff_smoothed = cv2.GaussianBlur(diff, (15, 15), 0)
        except cv2.error:
            # Fallback: 使用簡單平均
            kernel = np.ones((15, 15), np.float32) / 225
            diff_smoothed = cv2.filter2D(diff, -1, kernel)
        
        # 平滑後的信號
        valid_smooth = diff_smoothed > 0
        signal_smoothed = np.mean(diff_smoothed[valid_smooth]) if np.any(valid_smooth) else 0
        # 平滑後噪點大幅減少
        noise_smoothed = noise_std / np.sqrt(15*15)  # 約 15x15 = 225 像素平均
        
        snr_smoothed = signal_smoothed / (noise_smoothed + 1e-10)
        
        print(f"      平滑後 SNR (15x15): {snr_smoothed:.2f}x")
        
        if snr_smoothed > 2.0:
            print(f"      → CNN 的 convolution 可以有效提取信號")
            results['network_can_learn'] = True
        else:
            print(f"      → 信號可能過弱，建議增加 SPP 或檢查渲染設置")
            results['network_can_learn'] = False
    else:
        results['network_can_learn'] = snr_dolp > 0.5
        if results['network_can_learn']:
            print(f"      → 信號存在，Network 可能學得到（通過 convolution 降噪）")
    
    results['valid'] = True
    return results


def analyze_all_scenes(input_dir: str, max_scenes: int = None) -> List[Dict]:
    """分析所有場景"""
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
        results = analyze_noise(input_dir, scene_name)
        all_results.append(results)
    
    # 彙總
    print(f"\n{'='*60}")
    print(f"彙總 ({len(all_results)} 個場景)")
    print(f"{'='*60}")
    
    valid_results = [r for r in all_results if r.get('valid', False)]
    
    if valid_results:
        snr_values = [r['polarization_snr']['snr_dolp'] for r in valid_results]
        noise_values = [r['noise']['estimated_std'] for r in valid_results]
        
        print(f"\n  噪點 σ: min={min(noise_values):.6f}, max={max(noise_values):.6f}, avg={np.mean(noise_values):.6f}")
        print(f"  SNR:    min={min(snr_values):.2f}x, max={max(snr_values):.2f}x, avg={np.mean(snr_values):.2f}x")
        
        # 統計狀態
        excellent = sum(1 for s in snr_values if s >= 2.0)
        good = sum(1 for s in snr_values if 1.0 <= s < 2.0)
        marginal = sum(1 for s in snr_values if 0.5 <= s < 1.0)
        poor = sum(1 for s in snr_values if s < 0.5)
        
        print(f"\n  🟢 優秀 (SNR≥2):  {excellent}")
        print(f"  🟡 可用 (SNR≥1):  {good}")
        print(f"  🟠 勉強 (SNR≥0.5): {marginal}")
        print(f"  🔴 過噪 (SNR<0.5): {poor}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='PIDS 渲染噪聲檢測',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
判斷標準：
  SNR ≥ 2.0    優秀，偏振信號清晰
  SNR ≥ 1.0    可用，信號與噪點相當
  SNR ≥ 0.5    勉強，噪點較大但可能還有結構
  SNR < 0.5    過噪，建議增加 SPP

記住：
  - 噪點是隨機的，CNN convolution 會平均掉
  - 偏振信號是系統性的，會被保留
  - 所以即使 SNR < 1，Network 可能還是學得到！

範例：
  python pids_noise_check.py --input_dir ./output
  python pids_noise_check.py --scene scene_0002 --input_dir ./output
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True, help='渲染輸出目錄')
    parser.add_argument('--scene', type=str, default=None, help='指定場景名稱')
    parser.add_argument('--max_scenes', type=int, default=None, help='最多分析幾個場景')
    parser.add_argument('--save_report', action='store_true', help='保存 JSON 報告')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"[ERROR] 目錄不存在: {args.input_dir}")
        return 1
    
    print("="*60)
    print("PIDS 渲染噪聲檢測")
    print("="*60)
    print("\n核心問題: 噪點強度 vs 偏振信號強度")
    print("如果噪點 >> 信號 → 需要增加 SPP")
    print("\n但記住: CNN 的 convolution 會平均掉隨機噪點！")
    
    if args.scene:
        results = [analyze_noise(args.input_dir, args.scene)]
    else:
        results = analyze_all_scenes(args.input_dir, args.max_scenes)
    
    if args.save_report:
        report_path = os.path.join(args.input_dir, "noise_check_report.json")
        
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
        
        with open(report_path, 'w') as f:
            json.dump(make_serializable(results), f, indent=2)
        print(f"\n報告已保存: {report_path}")
    
    # 最終建議
    print(f"\n{'='*60}")
    print("最終建議")
    print(f"{'='*60}")
    
    valid_results = [r for r in results if r.get('valid', False)]
    if valid_results:
        avg_snr = np.mean([r['polarization_snr']['snr_dolp'] for r in valid_results])
        
        if avg_snr >= 2.0:
            print("\n🟢 噪點水平良好，可以開始訓練！")
        elif avg_snr >= 1.0:
            print("\n🟡 噪點與信號相當")
            print("   可以嘗試訓練，但建議：")
            print("   1. 增加 SPP 2-4 倍")
            print("   2. 或訓練時加入 noise augmentation")
        elif avg_snr >= 0.5:
            print("\n🟠 噪點較大")
            print("   建議增加 SPP 4-10 倍再訓練")
        else:
            print("\n🔴 噪點過大")
            print("   強烈建議增加 SPP 10+ 倍")
            print("   或檢查渲染設置是否正確")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

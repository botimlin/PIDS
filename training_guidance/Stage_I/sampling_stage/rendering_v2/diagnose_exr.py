#!/usr/bin/env python3
"""
PIDS EXR 數值診斷腳本
檢查背景區域和玻璃區域的實際數值
"""

import numpy as np
import sys
import os

def load_exr(path):
    """載入 EXR 文件"""
    try:
        import OpenEXR
        import Imath
        
        exr = OpenEXR.InputFile(path)
        header = exr.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        channels = header['channels'].keys()
        if 'Y' in channels:
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            y_str = exr.channel('Y', pt)
            y = np.frombuffer(y_str, dtype=np.float32).reshape(height, width)
            return y
        elif 'R' in channels:
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            r_str = exr.channel('R', pt)
            r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
            return r
    except ImportError:
        pass
    
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"無法載入: {path}")
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        return img.astype(np.float32)
    except:
        pass
    
    raise ImportError("需要 OpenEXR 或 OpenCV")


def analyze_regions(I_parallel, I_cross):
    """分析不同區域的數值"""
    
    print("="*60)
    print("EXR 數值診斷報告")
    print("="*60)
    
    # 全局統計
    print("\n📊 全局統計:")
    print(f"  I∥ (parallel): min={I_parallel.min():.4f}, max={I_parallel.max():.4f}, mean={I_parallel.mean():.4f}")
    print(f"  I⊥ (cross):    min={I_cross.min():.4f}, max={I_cross.max():.4f}, mean={I_cross.mean():.4f}")
    print(f"  全局比值:      I∥/I⊥ = {I_parallel.mean() / (I_cross.mean() + 1e-10):.2f}x")
    
    # 計算 DoLP
    I_sum = I_parallel + I_cross
    valid_mask = I_sum > 1e-6
    dolp = np.zeros_like(I_parallel)
    dolp[valid_mask] = np.abs(I_parallel - I_cross)[valid_mask] / I_sum[valid_mask]
    
    # 分區域
    high_dolp_mask = (dolp > 0.1) & valid_mask  # 高偏振（玻璃）
    low_dolp_mask = (dolp <= 0.1) & valid_mask  # 低偏振（背景）
    
    # 背景區域（低偏振）
    print("\n🏠 背景區域 (DoLP ≤ 0.1):")
    if np.any(low_dolp_mask):
        bg_parallel = I_parallel[low_dolp_mask]
        bg_cross = I_cross[low_dolp_mask]
        bg_ratio = bg_parallel.mean() / (bg_cross.mean() + 1e-10)
        
        print(f"  像素數: {np.sum(low_dolp_mask)} ({np.sum(low_dolp_mask)/valid_mask.sum()*100:.1f}%)")
        print(f"  I∥ mean: {bg_parallel.mean():.4f}")
        print(f"  I⊥ mean: {bg_cross.mean():.4f}")
        print(f"  I∥/I⊥ 比值: {bg_ratio:.2f}x")
        
        if 0.5 <= bg_ratio <= 2.0:
            print(f"  狀態: ✅ 平衡 (0.5~2.0)")
        else:
            print(f"  狀態: ❌ 不平衡！RAFT 無法正常對齊")
    else:
        print("  ⚠️ 沒有找到低偏振區域")
    
    # 玻璃區域（高偏振）
    print("\n🪟 玻璃區域 (DoLP > 0.1):")
    if np.any(high_dolp_mask):
        glass_parallel = I_parallel[high_dolp_mask]
        glass_cross = I_cross[high_dolp_mask]
        glass_ratio = glass_parallel.mean() / (glass_cross.mean() + 1e-10)
        
        print(f"  像素數: {np.sum(high_dolp_mask)} ({np.sum(high_dolp_mask)/valid_mask.sum()*100:.1f}%)")
        print(f"  I∥ mean: {glass_parallel.mean():.4f}")
        print(f"  I⊥ mean: {glass_cross.mean():.4f}")
        print(f"  I∥/I⊥ 比值: {glass_ratio:.2f}x")
        print(f"  DoLP mean: {dolp[high_dolp_mask].mean():.4f}")
    else:
        print("  ⚠️ 沒有找到高偏振區域")
    
    # 採樣特定位置
    print("\n📍 特定位置採樣:")
    h, w = I_parallel.shape
    
    # 左上角（通常是背景）
    sample_regions = [
        ("左上角(背景)", slice(10, 50), slice(10, 50)),
        ("中央上方(背景)", slice(10, 50), slice(w//2-20, w//2+20)),
        ("左下角(地板)", slice(h-50, h-10), slice(10, 50)),
        ("中央(可能是玻璃)", slice(h//2-20, h//2+20), slice(w//2-20, w//2+20)),
    ]
    
    for name, row_slice, col_slice in sample_regions:
        region_parallel = I_parallel[row_slice, col_slice]
        region_cross = I_cross[row_slice, col_slice]
        ratio = region_parallel.mean() / (region_cross.mean() + 1e-10)
        print(f"  {name}:")
        print(f"    I∥={region_parallel.mean():.4f}, I⊥={region_cross.mean():.4f}, 比值={ratio:.2f}x")
    
    # 診斷結論
    print("\n" + "="*60)
    print("💡 診斷結論:")
    print("="*60)
    
    if np.any(low_dolp_mask):
        bg_ratio = I_parallel[low_dolp_mask].mean() / (I_cross[low_dolp_mask].mean() + 1e-10)
        
        if bg_ratio > 2.0:
            print("""
❌ 問題：背景區域 I∥ >> I⊥

原因分析：
1. 偏振光源太強，即使漫反射也保留了偏振
2. Mitsuba 的漫反射材質可能沒有完全 depolarize
3. 或者非偏振背景光沒有正確設置

建議解決方案：
1. 大幅增加非偏振背景光強度
2. 或者使用完全不同的光源設置策略
3. 考慮使用後處理來平衡背景亮度
""")
        elif bg_ratio < 0.5:
            print("""
❌ 問題：背景區域 I∥ << I⊥

這種情況比較罕見，可能是偏振角度設置問題。
""")
        else:
            print("""
✅ 背景區域平衡！

背景 I∥/I⊥ 接近 1，這是理想的狀態。
RAFT-Stereo 應該可以正常做雙目對齊。
""")


def main():
    if len(sys.argv) < 3:
        print("用法: python diagnose_exr.py <left_parallel.exr> <right_cross.exr>")
        print("\n範例:")
        print("  python diagnose_exr.py scene_0002_left_parallel.exr scene_0002_right_cross.exr")
        return 1
    
    parallel_path = sys.argv[1]
    cross_path = sys.argv[2]
    
    if not os.path.exists(parallel_path):
        print(f"錯誤: 找不到 {parallel_path}")
        return 1
    if not os.path.exists(cross_path):
        print(f"錯誤: 找不到 {cross_path}")
        return 1
    
    print(f"載入 {parallel_path}...")
    I_parallel = load_exr(parallel_path)
    
    print(f"載入 {cross_path}...")
    I_cross = load_exr(cross_path)
    
    analyze_regions(I_parallel, I_cross)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

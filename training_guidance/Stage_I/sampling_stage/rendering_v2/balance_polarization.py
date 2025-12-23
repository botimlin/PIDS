#!/usr/bin/env python3
"""
PIDS 強度平衡後處理腳本
=========================

問題：即使使用非偏振背景光，由於 Mitsuba 的偏振追蹤特性，
      漫反射表面仍然會有 I∥ > I⊥ 的現象。

解決方案：對渲染結果進行後處理，平衡背景區域的亮度。

原理：
1. 識別低偏振區域（背景）
2. 計算背景區域的 I∥/I⊥ 比值
3. 對 I⊥ 乘以這個比值來平衡
4. 玻璃區域的偏振差異得以保留

用法：
    python balance_polarization.py --input_dir ./output_stage1
    python balance_polarization.py -i left.exr -c right.exr -o ./balanced/
"""

import numpy as np
import os
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


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
        
        channels = list(header['channels'].keys())
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        
        if 'Y' in channels:
            y_str = exr.channel('Y', pt)
            return np.frombuffer(y_str, dtype=np.float32).reshape(height, width)
        elif 'R' in channels:
            r_str = exr.channel('R', pt)
            return np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
        else:
            first_ch = channels[0]
            ch_str = exr.channel(first_ch, pt)
            return np.frombuffer(ch_str, dtype=np.float32).reshape(height, width)
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
    except Exception as e:
        raise ImportError(f"需要 OpenEXR 或 OpenCV: {e}")


def save_exr(path, image, original_path=None):
    """保存 EXR 文件"""
    try:
        import OpenEXR
        import Imath
        
        height, width = image.shape
        header = OpenEXR.Header(width, height)
        header['channels'] = {'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        
        out = OpenEXR.OutputFile(path, header)
        out.writePixels({'Y': image.astype(np.float32).tobytes()})
        out.close()
        return True
    except ImportError:
        pass
    
    try:
        import cv2
        cv2.imwrite(path, image.astype(np.float32))
        return True
    except:
        pass
    
    return False


def balance_polarization(I_parallel, I_cross, method='background_ratio'):
    """
    平衡偏振圖像的背景亮度
    
    Args:
        I_parallel: 平行偏振圖像 (I∥)
        I_cross: 交叉偏振圖像 (I⊥)
        method: 平衡方法
            - 'background_ratio': 基於背景區域的比值
            - 'histogram': 直方圖匹配
            - 'mean': 平均值匹配
    
    Returns:
        balanced_parallel, balanced_cross: 平衡後的圖像
    """
    
    # 計算 DoLP 來識別背景區域
    I_sum = I_parallel + I_cross
    valid_mask = I_sum > 1e-6
    
    dolp = np.zeros_like(I_parallel)
    dolp[valid_mask] = np.abs(I_parallel - I_cross)[valid_mask] / I_sum[valid_mask]
    
    # 低偏振區域 = 背景（漫反射）
    background_mask = (dolp < 0.1) & valid_mask
    
    if method == 'background_ratio':
        # 方法 1：基於背景區域的比值
        if np.any(background_mask):
            bg_parallel = I_parallel[background_mask]
            bg_cross = I_cross[background_mask]
            
            # 計算背景區域的平均比值
            ratio = np.mean(bg_parallel) / (np.mean(bg_cross) + 1e-10)
            
            print(f"    背景 I∥/I⊥ 比值: {ratio:.3f}")
            
            if ratio > 1.1:
                # I∥ 比 I⊥ 亮，需要提升 I⊥
                # 但我們不想改變玻璃區域的對比度
                # 所以只在背景區域調整
                
                # 策略：對整個 I⊥ 乘以 ratio，然後在玻璃區域保留原始對比度
                balanced_cross = I_cross * ratio
                
                # 在高偏振區域（玻璃），混合原始值和調整值
                # 這樣可以保留玻璃的偏振特性
                high_dolp_mask = dolp > 0.2
                if np.any(high_dolp_mask):
                    # 使用 DoLP 作為混合權重
                    # DoLP 越高，越保留原始 I⊥
                    blend_weight = np.clip((dolp - 0.1) / 0.3, 0, 1)
                    balanced_cross = (1 - blend_weight) * balanced_cross + blend_weight * I_cross
                
                balanced_parallel = I_parallel.copy()
                
            elif ratio < 0.9:
                # I⊥ 比 I∥ 亮（少見）
                balanced_parallel = I_parallel * (1 / ratio)
                balanced_cross = I_cross.copy()
            else:
                # 已經平衡
                balanced_parallel = I_parallel.copy()
                balanced_cross = I_cross.copy()
        else:
            print("    ⚠️ 沒有找到低偏振背景區域，無法平衡")
            balanced_parallel = I_parallel.copy()
            balanced_cross = I_cross.copy()
    
    elif method == 'histogram':
        # 方法 2：直方圖匹配（更激進）
        # 讓背景區域的直方圖相似
        from scipy import ndimage
        
        balanced_parallel = I_parallel.copy()
        
        if np.any(background_mask):
            # 計算背景區域的累積分佈函數
            bg_parallel_sorted = np.sort(I_parallel[background_mask])
            bg_cross_sorted = np.sort(I_cross[background_mask])
            
            # 簡單的線性映射
            ratio = np.mean(bg_parallel_sorted) / (np.mean(bg_cross_sorted) + 1e-10)
            balanced_cross = I_cross * ratio
        else:
            balanced_cross = I_cross.copy()
    
    elif method == 'mean':
        # 方法 3：全局平均值匹配
        mean_parallel = np.mean(I_parallel[valid_mask])
        mean_cross = np.mean(I_cross[valid_mask])
        
        ratio = mean_parallel / (mean_cross + 1e-10)
        balanced_cross = I_cross * ratio
        balanced_parallel = I_parallel.copy()
    
    else:
        raise ValueError(f"未知方法: {method}")
    
    return balanced_parallel, balanced_cross


def process_scene(parallel_path, cross_path, output_dir, method='background_ratio'):
    """處理單個場景"""
    
    scene_name = Path(parallel_path).stem.replace('_left_parallel', '')
    print(f"\n處理場景: {scene_name}")
    
    # 載入圖像
    I_parallel = load_exr(parallel_path)
    I_cross = load_exr(cross_path)
    
    # 平衡
    balanced_parallel, balanced_cross = balance_polarization(I_parallel, I_cross, method)
    
    # 驗證
    I_sum = balanced_parallel + balanced_cross
    valid_mask = I_sum > 1e-6
    dolp = np.zeros_like(balanced_parallel)
    dolp[valid_mask] = np.abs(balanced_parallel - balanced_cross)[valid_mask] / I_sum[valid_mask]
    
    background_mask = (dolp < 0.1) & valid_mask
    if np.any(background_mask):
        new_ratio = np.mean(balanced_parallel[background_mask]) / (np.mean(balanced_cross[background_mask]) + 1e-10)
        print(f"    平衡後背景比值: {new_ratio:.3f}")
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    
    out_parallel = os.path.join(output_dir, f"{scene_name}_left_parallel.exr")
    out_cross = os.path.join(output_dir, f"{scene_name}_right_cross.exr")
    
    save_exr(out_parallel, balanced_parallel)
    save_exr(out_cross, balanced_cross)
    
    print(f"    保存: {out_parallel}")
    print(f"    保存: {out_cross}")
    
    return True


def process_directory(input_dir, output_dir, method='background_ratio'):
    """批量處理目錄"""
    
    input_path = Path(input_dir)
    parallel_files = sorted(input_path.glob("*_left_parallel.exr"))
    
    if not parallel_files:
        print(f"在 {input_dir} 中沒有找到 *_left_parallel.exr 文件")
        return
    
    print(f"找到 {len(parallel_files)} 個場景")
    
    success = 0
    for parallel_path in parallel_files:
        cross_path = str(parallel_path).replace('_left_parallel', '_right_cross')
        
        if not os.path.exists(cross_path):
            print(f"⚠️ 找不到對應的 cross 文件: {cross_path}")
            continue
        
        try:
            process_scene(str(parallel_path), cross_path, output_dir, method)
            success += 1
        except Exception as e:
            print(f"❌ 處理失敗: {e}")
    
    print(f"\n完成！成功處理 {success}/{len(parallel_files)} 個場景")
    print(f"輸出目錄: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='PIDS 偏振強度平衡後處理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 處理單個場景
  python balance_polarization.py -p left_parallel.exr -c right_cross.exr -o ./balanced/
  
  # 批量處理
  python balance_polarization.py --input_dir ./output_stage1 -o ./balanced/
  
  # 使用不同的平衡方法
  python balance_polarization.py --input_dir ./output_stage1 -o ./balanced/ --method mean
        """
    )
    
    # 輸入選項
    parser.add_argument('-p', '--parallel', type=str, help='平行偏振圖像 (left_parallel.exr)')
    parser.add_argument('-c', '--cross', type=str, help='交叉偏振圖像 (right_cross.exr)')
    parser.add_argument('--input_dir', '-i', type=str, help='輸入目錄（批量處理）')
    
    # 輸出選項
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='輸出目錄')
    
    # 平衡方法
    parser.add_argument('--method', type=str, default='background_ratio',
                        choices=['background_ratio', 'histogram', 'mean'],
                        help='平衡方法 (預設: background_ratio)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PIDS 偏振強度平衡後處理")
    print("="*60)
    
    if args.input_dir:
        # 批量處理
        process_directory(args.input_dir, args.output_dir, args.method)
    elif args.parallel and args.cross:
        # 單個場景
        process_scene(args.parallel, args.cross, args.output_dir, args.method)
    else:
        print("錯誤: 請指定 --input_dir 或 (-p 和 -c)")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

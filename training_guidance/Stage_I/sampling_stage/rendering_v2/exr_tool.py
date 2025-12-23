#!/usr/bin/env python3
"""
EXR 讀取與轉換工具
==================

功能：
1. 顯示 EXR 文件信息（通道、尺寸、數值範圍）
2. 轉換 EXR 為 PNG（可選擇 tone mapping 方式）
3. 比較兩個 EXR 文件
4. 批次轉換

使用方法：
    # 查看信息
    python exr_tool.py info image.exr
    
    # 轉換為 PNG
    python exr_tool.py convert image.exr -o output.png
    
    # 統一範圍轉換（比較用）
    python exr_tool.py convert left.exr right.exr --unified -o comparison/
    
    # 比較兩張圖
    python exr_tool.py compare left.exr right.exr

作者: PIDS Project
"""

import numpy as np
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 導入依賴
try:
    import mitsuba as mi
    HAS_MITSUBA = True
except ImportError:
    HAS_MITSUBA = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def load_exr(path: str) -> Tuple[Optional[np.ndarray], Dict]:
    """
    載入 EXR 文件，返回圖像數據和元信息
    """
    info = {
        'path': path,
        'exists': os.path.exists(path),
        'channels': None,
        'shape': None,
        'dtype': None,
    }
    
    if not info['exists']:
        return None, info
    
    if HAS_MITSUBA:
        try:
            bitmap = mi.Bitmap(path)
            image = np.array(bitmap)
            
            info['shape'] = image.shape
            info['dtype'] = str(image.dtype)
            info['channels'] = image.shape[2] if image.ndim == 3 else 1
            info['loader'] = 'mitsuba'
            
            return image.astype(np.float32), info
        except Exception as e:
            info['error'] = str(e)
    
    if HAS_CV2:
        try:
            os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
            image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if image is not None:
                # BGR to RGB
                if image.ndim == 3 and image.shape[2] >= 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                info['shape'] = image.shape
                info['dtype'] = str(image.dtype)
                info['channels'] = image.shape[2] if image.ndim == 3 else 1
                info['loader'] = 'opencv'
                
                return image.astype(np.float32), info
        except Exception as e:
            info['error'] = str(e)
    
    info['error'] = 'No loader available (need mitsuba or opencv with OpenEXR)'
    return None, info


def get_exr_info(path: str) -> Dict:
    """
    獲取 EXR 文件詳細信息
    """
    image, info = load_exr(path)
    
    if image is None:
        return info
    
    # 計算統計信息
    info['min'] = float(np.min(image))
    info['max'] = float(np.max(image))
    info['mean'] = float(np.mean(image))
    info['std'] = float(np.std(image))
    
    # 每個通道的統計
    if image.ndim == 3:
        info['per_channel'] = []
        channel_names = ['R', 'G', 'B', 'A'] if image.shape[2] <= 4 else [f'Ch{i}' for i in range(image.shape[2])]
        for i in range(image.shape[2]):
            ch = image[:, :, i]
            info['per_channel'].append({
                'name': channel_names[i] if i < len(channel_names) else f'Ch{i}',
                'min': float(np.min(ch)),
                'max': float(np.max(ch)),
                'mean': float(np.mean(ch)),
            })
    
    # 有效像素（非零、非 NaN、非 Inf）
    valid_mask = np.isfinite(image) & (image > 0)
    info['valid_pixels'] = int(np.sum(valid_mask))
    info['total_pixels'] = int(image.size)
    info['valid_ratio'] = float(np.sum(valid_mask) / image.size)
    
    return info


def print_exr_info(info: Dict):
    """
    打印 EXR 信息
    """
    print(f"\n{'='*50}")
    print(f"EXR 文件信息: {info['path']}")
    print(f"{'='*50}")
    
    if not info.get('exists', False):
        print("  [ERROR] 文件不存在")
        return
    
    if 'error' in info:
        print(f"  [ERROR] {info['error']}")
        return
    
    print(f"  尺寸:     {info['shape']}")
    print(f"  通道數:   {info['channels']}")
    print(f"  數據類型: {info['dtype']}")
    print(f"  載入器:   {info.get('loader', 'unknown')}")
    
    print(f"\n  數值範圍:")
    print(f"    min:  {info['min']:.6f}")
    print(f"    max:  {info['max']:.6f}")
    print(f"    mean: {info['mean']:.6f}")
    print(f"    std:  {info['std']:.6f}")
    
    if 'per_channel' in info:
        print(f"\n  各通道統計:")
        for ch in info['per_channel']:
            print(f"    {ch['name']}: min={ch['min']:.4f}, max={ch['max']:.4f}, mean={ch['mean']:.4f}")
    
    print(f"\n  有效像素: {info['valid_pixels']:,} / {info['total_pixels']:,} ({info['valid_ratio']*100:.1f}%)")


def tonemap(image: np.ndarray, method: str = 'normalize', 
            vmin: float = None, vmax: float = None,
            gamma: float = 2.2) -> np.ndarray:
    """
    將 HDR 圖像轉換為 LDR (0-255)
    
    方法:
    - normalize: 線性映射到 [0, 1]
    - gamma: gamma correction
    - log: 對數映射
    - reinhard: Reinhard tone mapping
    - percentile: 使用 percentile 避免極值影響
    """
    # 處理多通道
    if image.ndim == 3 and image.shape[2] > 3:
        # 多於 3 通道，只取前 3 個或轉灰階
        if image.shape[2] >= 3:
            image = image[:, :, :3]
        else:
            image = image[:, :, 0]
    
    # 確定範圍
    if vmin is None:
        vmin = np.min(image[np.isfinite(image)])
    if vmax is None:
        vmax = np.max(image[np.isfinite(image)])
    
    # 避免除零
    if vmax - vmin < 1e-10:
        vmax = vmin + 1
    
    if method == 'normalize':
        # 線性映射
        result = (image - vmin) / (vmax - vmin)
        
    elif method == 'gamma':
        # Gamma correction
        result = (image - vmin) / (vmax - vmin)
        result = np.clip(result, 0, 1)
        result = np.power(result, 1.0 / gamma)
        
    elif method == 'log':
        # 對數映射
        result = np.log1p(image - vmin) / np.log1p(vmax - vmin)
        
    elif method == 'reinhard':
        # Reinhard tone mapping
        result = (image - vmin) / (vmax - vmin)
        result = result / (1 + result)
        
    elif method == 'percentile':
        # 使用 percentile，避免極值
        valid = image[np.isfinite(image)]
        p_low = np.percentile(valid, 1)
        p_high = np.percentile(valid, 99)
        result = (image - p_low) / (p_high - p_low + 1e-10)
        
    else:
        result = (image - vmin) / (vmax - vmin)
    
    # Clip 並轉換為 uint8
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)
    
    return result


def save_as_png(image: np.ndarray, output_path: str, method: str = 'normalize',
                vmin: float = None, vmax: float = None):
    """
    將 EXR 圖像保存為 PNG
    """
    # Tone mapping
    ldr = tonemap(image, method=method, vmin=vmin, vmax=vmax)
    
    # 保存
    if HAS_CV2:
        if ldr.ndim == 3 and ldr.shape[2] >= 3:
            ldr = cv2.cvtColor(ldr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, ldr)
    else:
        # Fallback: 使用 PIL
        try:
            from PIL import Image
            if ldr.ndim == 2:
                img = Image.fromarray(ldr, mode='L')
            else:
                img = Image.fromarray(ldr, mode='RGB')
            img.save(output_path)
        except ImportError:
            print(f"  [ERROR] 需要 opencv 或 PIL 來保存 PNG")
            return False
    
    return True


def convert_exr_to_png(input_path: str, output_path: str = None,
                       method: str = 'normalize',
                       vmin: float = None, vmax: float = None):
    """
    轉換 EXR 為 PNG
    """
    image, info = load_exr(input_path)
    
    if image is None:
        print(f"  [ERROR] 無法載入: {input_path}")
        return False
    
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('.png'))
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    success = save_as_png(image, output_path, method=method, vmin=vmin, vmax=vmax)
    
    if success:
        print(f"  [OK] {input_path} -> {output_path}")
        print(f"       範圍: [{info['min']:.4f}, {info['max']:.4f}] -> [0, 255] ({method})")
    
    return success


def convert_unified(input_paths: List[str], output_dir: str,
                    method: str = 'normalize'):
    """
    使用統一範圍轉換多個 EXR（用於比較）
    """
    # 載入所有圖像，找出全局範圍
    images = []
    global_min = float('inf')
    global_max = float('-inf')
    
    for path in input_paths:
        image, info = load_exr(path)
        if image is not None:
            images.append((path, image))
            global_min = min(global_min, info['min'])
            global_max = max(global_max, info['max'])
    
    if not images:
        print("  [ERROR] 沒有成功載入任何圖像")
        return
    
    print(f"\n  統一範圍: [{global_min:.4f}, {global_max:.4f}]")
    
    # 轉換所有圖像
    os.makedirs(output_dir, exist_ok=True)
    
    for path, image in images:
        filename = Path(path).stem + '_unified.png'
        output_path = os.path.join(output_dir, filename)
        
        save_as_png(image, output_path, method=method, 
                    vmin=global_min, vmax=global_max)
        print(f"  [OK] {path} -> {output_path}")


def compare_exr(path1: str, path2: str, output_dir: str = None):
    """
    比較兩個 EXR 文件
    """
    image1, info1 = load_exr(path1)
    image2, info2 = load_exr(path2)
    
    if image1 is None or image2 is None:
        print("  [ERROR] 無法載入圖像")
        return
    
    print(f"\n{'='*60}")
    print(f"EXR 比較")
    print(f"{'='*60}")
    
    print(f"\n  圖像 1: {path1}")
    print(f"    範圍: [{info1['min']:.4f}, {info1['max']:.4f}]")
    print(f"    均值: {info1['mean']:.4f}")
    
    print(f"\n  圖像 2: {path2}")
    print(f"    範圍: [{info2['min']:.4f}, {info2['max']:.4f}]")
    print(f"    均值: {info2['mean']:.4f}")
    
    # 轉為相同形狀進行比較
    if image1.ndim == 3 and image1.shape[2] > 1:
        img1_gray = np.mean(image1, axis=2)
    else:
        img1_gray = image1.squeeze()
    
    if image2.ndim == 3 and image2.shape[2] > 1:
        img2_gray = np.mean(image2, axis=2)
    else:
        img2_gray = image2.squeeze()
    
    # 計算差異
    if img1_gray.shape == img2_gray.shape:
        diff = img1_gray - img2_gray
        abs_diff = np.abs(diff)
        
        print(f"\n  差異統計 (圖1 - 圖2):")
        print(f"    |diff| mean: {np.mean(abs_diff):.6f}")
        print(f"    |diff| max:  {np.max(abs_diff):.6f}")
        print(f"    signed mean: {np.mean(diff):.6f}")
        
        # 相對差異
        denom = (np.abs(img1_gray) + np.abs(img2_gray)) / 2 + 1e-10
        rel_diff = abs_diff / denom
        print(f"    relative mean: {np.mean(rel_diff)*100:.2f}%")
        
        # 比值
        valid_mask = img2_gray > 0.01
        if np.any(valid_mask):
            ratio = img1_gray[valid_mask] / img2_gray[valid_mask]
            print(f"\n  比值 (圖1 / 圖2):")
            print(f"    mean: {np.mean(ratio):.4f}x")
            print(f"    median: {np.median(ratio):.4f}x")
            print(f"    min: {np.min(ratio):.4f}x")
            print(f"    max: {np.max(ratio):.4f}x")
        
        # 保存比較圖
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 統一範圍
            global_min = min(info1['min'], info2['min'])
            global_max = max(info1['max'], info2['max'])
            
            save_as_png(image1, os.path.join(output_dir, 'image1_unified.png'),
                        vmin=global_min, vmax=global_max)
            save_as_png(image2, os.path.join(output_dir, 'image2_unified.png'),
                        vmin=global_min, vmax=global_max)
            
            # 差異圖
            diff_img = tonemap(abs_diff, method='percentile')
            if HAS_CV2:
                # 偽彩色
                diff_color = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(output_dir, 'difference.png'), diff_color)
            
            print(f"\n  比較圖已保存到: {output_dir}")
    else:
        print(f"\n  [WARNING] 圖像尺寸不同，無法計算差異")
        print(f"    圖像 1: {img1_gray.shape}")
        print(f"    圖像 2: {img2_gray.shape}")


def main():
    parser = argparse.ArgumentParser(
        description='EXR 讀取與轉換工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 查看 EXR 信息
  python exr_tool.py info image.exr
  
  # 轉換為 PNG
  python exr_tool.py convert image.exr
  python exr_tool.py convert image.exr -o output.png
  python exr_tool.py convert image.exr --method gamma
  
  # 統一範圍轉換（比較用）
  python exr_tool.py convert left.exr right.exr --unified -o ./comparison/
  
  # 比較兩張圖
  python exr_tool.py compare left.exr right.exr
  python exr_tool.py compare left.exr right.exr -o ./comparison/
  
  # 批次轉換
  python exr_tool.py convert *.exr -o ./png_output/

Tone mapping 方法:
  normalize   線性映射 [min, max] -> [0, 255]
  gamma       Gamma correction (γ=2.2)
  log         對數映射（適合 HDR）
  reinhard    Reinhard tone mapping
  percentile  使用 1-99 percentile（避免極值）
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # info 命令
    info_parser = subparsers.add_parser('info', help='顯示 EXR 信息')
    info_parser.add_argument('files', nargs='+', help='EXR 文件')
    
    # convert 命令
    convert_parser = subparsers.add_parser('convert', help='轉換 EXR 為 PNG')
    convert_parser.add_argument('files', nargs='+', help='EXR 文件')
    convert_parser.add_argument('-o', '--output', help='輸出路徑或目錄')
    convert_parser.add_argument('--method', default='normalize',
                                choices=['normalize', 'gamma', 'log', 'reinhard', 'percentile'],
                                help='Tone mapping 方法')
    convert_parser.add_argument('--unified', action='store_true',
                                help='使用統一範圍（用於比較多張圖）')
    
    # compare 命令
    compare_parser = subparsers.add_parser('compare', help='比較兩個 EXR')
    compare_parser.add_argument('file1', help='第一個 EXR')
    compare_parser.add_argument('file2', help='第二個 EXR')
    compare_parser.add_argument('-o', '--output', help='輸出目錄')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # 檢查依賴
    if not HAS_MITSUBA and not HAS_CV2:
        print("[ERROR] 需要 mitsuba 或 opencv 來讀取 EXR")
        print("        請確保已安裝: pip install opencv-python")
        return 1
    
    if args.command == 'info':
        for f in args.files:
            info = get_exr_info(f)
            print_exr_info(info)
    
    elif args.command == 'convert':
        if args.unified:
            # 統一範圍轉換
            output_dir = args.output or './png_unified'
            convert_unified(args.files, output_dir, method=args.method)
        else:
            # 單獨轉換
            for f in args.files:
                if args.output and len(args.files) == 1:
                    output = args.output
                elif args.output:
                    output = os.path.join(args.output, Path(f).stem + '.png')
                else:
                    output = None
                convert_exr_to_png(f, output, method=args.method)
    
    elif args.command == 'compare':
        compare_exr(args.file1, args.file2, args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

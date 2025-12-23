"""
Training Data Quality Check Script
===================================

根據論文中的 Training Data Collection Standards，實現五項質量控制標準：

1. Geometric Consistency Filtering - 幾何一致性過濾
2. Background Photometric Consistency Filtering - 背景光度一致性過濾  
3. Polarization Signal Validity Filtering - 偏振信號有效性過濾
4. Ground Truth Alignment Filtering - Ground Truth 對齊過濾
5. Depth Validity Rate Filtering - 深度有效率過濾

檔案命名規則：
    - 左相機平行偏振: {scene_name}_left_parallel.{ext}
    - 左相機交叉偏振: {scene_name}_left_cross.{ext}
    - 右相機平行偏振: {scene_name}_right_parallel.{ext}
    - 右相機交叉偏振: {scene_name}_right_cross.{ext}
    - 深度圖: {scene_name}_depth.{ext}
    - 透明物體遮罩: {scene_name}_mask.{ext}
    - RGB 參考圖: {scene_name}_rgb_ref.{ext}

使用方式:
    python training_data_quality_check.py --input_dir ./dataset --output_dir ./filtered_dataset
    python training_data_quality_check.py --input_dir ./dataset --report_only

作者: PIDS Project
版本: 1.0
"""

import os
import cv2
import numpy as np
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


# ============================================================
# 配置參數
# ============================================================

@dataclass
class QualityCheckConfig:
    """質量檢測配置參數"""
    
    # 1. Geometric Consistency
    max_vertical_disparity: float = 1.0  # 最大允許垂直視差 (pixels)
    min_feature_matches: int = 30        # 最少特徵匹配數量（降低，合成數據紋理較少）
    
    # 2. Background Photometric Consistency
    max_background_diff: float = 0.1     # 背景區域最大允許亮度差異 (normalized)
    background_threshold: float = 0.1    # 背景區域判定閾值
    
    # 3. Polarization Signal Validity
    min_polarization_contrast: float = 0.01  # 最小偏振對比度（降低）
    specular_percentile: float = 95          # 高光區域百分位數
    
    # 4. Ground Truth Alignment
    max_alignment_error: float = 1.0     # 最大對齊誤差 (pixels)
    edge_detection_threshold: int = 50   # 邊緣檢測閾值
    
    # 5. Depth Validity Rate
    min_depth_validity_rate: float = 0.80  # 最小深度有效率 (80%)（降低，遮罩可能較小）
    min_valid_depth: float = 0.001         # 最小有效深度值 (meters)
    max_valid_depth: float = 10.0          # 最大有效深度值 (meters)


@dataclass
class SampleQualityReport:
    """單一樣本的質量報告"""
    scene_name: str
    passed: bool = True
    
    # 各項檢測結果
    geometric_consistency: bool = True
    geometric_vertical_disparity: float = 0.0
    geometric_num_matches: int = 0
    
    photometric_consistency: bool = True
    photometric_background_diff: float = 0.0
    
    polarization_validity: bool = True
    polarization_contrast: float = 0.0
    polarization_ratio: float = 0.0
    
    alignment_check: bool = True
    alignment_error: float = 0.0
    
    depth_validity: bool = True
    depth_validity_rate: float = 0.0
    
    # 失敗原因
    failure_reasons: List[str] = field(default_factory=list)
    
    def update_passed(self):
        """更新總體通過狀態"""
        self.passed = (
            self.geometric_consistency and
            self.photometric_consistency and
            self.polarization_validity and
            self.alignment_check and
            self.depth_validity
        )


# ============================================================
# 檔案處理工具
# ============================================================

def find_scene_files(input_dir: str) -> Dict[str, Dict[str, str]]:
    """
    掃描輸入目錄，找出所有場景的相關檔案
    
    Returns:
        Dict[scene_name, Dict[file_type, file_path]]
    """
    scenes = {}
    input_path = Path(input_dir)
    
    # 支援的圖像格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.exr', '.tiff', '.tif'}
    
    for file_path in input_path.iterdir():
        if not file_path.is_file():
            continue
        
        if file_path.suffix.lower() not in image_extensions:
            continue
        
        filename = file_path.stem
        
        # 解析檔案名稱
        # 格式: {scene_name}_{type}
        # type: left_parallel, left_cross, right_parallel, right_cross, 
        #       depth, mask, rgb_ref, I_parallel, I_cross, disparity
        
        file_types = [
            'left_parallel', 'left_cross',
            'right_parallel', 'right_cross',
            'I_parallel', 'I_cross',
            'depth', 'disparity', 'mask', 'rgb_ref'
        ]
        
        for ftype in file_types:
            if filename.endswith(f'_{ftype}'):
                scene_name = filename[:-len(ftype)-1]
                
                if scene_name not in scenes:
                    scenes[scene_name] = {}
                
                scenes[scene_name][ftype] = str(file_path)
                break
    
    return scenes


def load_image(path: str, grayscale: bool = True) -> Optional[np.ndarray]:
    """載入圖像"""
    if not os.path.exists(path):
        return None
    
    if path.endswith('.exr'):
        # 使用 OpenCV 載入 EXR
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    else:
        if grayscale:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    return img


def normalize_image(img: np.ndarray) -> np.ndarray:
    """正規化圖像到 [0, 1]"""
    if img is None:
        return None
    
    img = img.astype(np.float32)
    
    if img.max() > 1.0:
        # 假設是 8-bit 或 16-bit 圖像
        if img.max() > 255:
            img = img / 65535.0
        else:
            img = img / 255.0
    
    return img


# ============================================================
# 質量檢測函數
# ============================================================

def check_geometric_consistency(
    img_left: np.ndarray,
    img_right: np.ndarray,
    config: QualityCheckConfig,
    mask: Optional[np.ndarray] = None
) -> Tuple[bool, float, int]:
    """
    1. Geometric Consistency Filtering
    
    使用特徵匹配檢測立體對的垂直視差
    背景區域的垂直視差應該接近 0
    
    Args:
        img_left: 左圖像
        img_right: 右圖像
        config: 配置參數
        mask: 前景遮罩（用於排除前景區域）
    
    Returns:
        (passed, vertical_disparity, num_matches)
    """
    if img_left is None or img_right is None:
        return False, float('inf'), 0
    
    # 轉換為 8-bit 用於特徵檢測
    left_8bit = (normalize_image(img_left) * 255).astype(np.uint8)
    right_8bit = (normalize_image(img_right) * 255).astype(np.uint8)
    
    # 創建背景遮罩
    if mask is not None:
        bg_mask = (mask == 0).astype(np.uint8) * 255
    else:
        bg_mask = None
    
    # 使用 ORB 特徵檢測器
    orb = cv2.ORB_create(nfeatures=2000)
    
    kp1, des1 = orb.detectAndCompute(left_8bit, bg_mask)
    kp2, des2 = orb.detectAndCompute(right_8bit, bg_mask)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return False, float('inf'), 0
    
    # 特徵匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if len(matches) < config.min_feature_matches:
        return False, float('inf'), len(matches)
    
    # 計算垂直視差
    vertical_disparities = []
    for match in matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        
        v_disp = abs(pt1[1] - pt2[1])
        vertical_disparities.append(v_disp)
    
    avg_vertical_disparity = np.mean(vertical_disparities)
    
    passed = avg_vertical_disparity <= config.max_vertical_disparity
    
    return passed, avg_vertical_disparity, len(matches)


def check_photometric_consistency(
    img_parallel: np.ndarray,
    img_cross: np.ndarray,
    config: QualityCheckConfig,
    mask: Optional[np.ndarray] = None
) -> Tuple[bool, float]:
    """
    2. Background Photometric Consistency Filtering
    
    檢查背景區域在偏振通道間的亮度差異
    漫反射背景應該 |I_parallel - I_cross| ≈ 0
    
    Args:
        img_parallel: 平行偏振圖像
        img_cross: 交叉偏振圖像
        config: 配置參數
        mask: 前景遮罩
    
    Returns:
        (passed, background_diff)
    """
    if img_parallel is None or img_cross is None:
        return False, float('inf')
    
    # 正規化
    parallel = normalize_image(img_parallel)
    cross = normalize_image(img_cross)
    
    # 創建背景遮罩
    if mask is not None:
        bg_mask = mask == 0
    else:
        # 自動估計背景（假設較暗且變化小的區域）
        combined = (parallel + cross) / 2
        bg_mask = combined < np.percentile(combined, 70)
    
    if np.sum(bg_mask) < 100:
        return False, float('inf')
    
    # 計算背景區域的亮度差異
    diff = np.abs(parallel - cross)
    bg_diff = np.mean(diff[bg_mask])
    
    passed = bg_diff <= config.max_background_diff
    
    return passed, bg_diff


def check_polarization_validity(
    img_parallel: np.ndarray,
    img_cross: np.ndarray,
    config: QualityCheckConfig,
    mask: Optional[np.ndarray] = None
) -> Tuple[bool, float, float]:
    """
    3. Polarization Signal Validity Filtering
    
    檢查透明物體區域的偏振對比度
    應該有 I_parallel > I_cross
    
    Args:
        img_parallel: 平行偏振圖像
        img_cross: 交叉偏振圖像
        config: 配置參數
        mask: 透明物體遮罩
    
    Returns:
        (passed, polarization_contrast, ratio)
    """
    if img_parallel is None or img_cross is None:
        return False, 0.0, 0.0
    
    # 正規化
    parallel = normalize_image(img_parallel)
    cross = normalize_image(img_cross)
    
    # 找出高光區域
    if mask is not None:
        roi_mask = mask > 0
    else:
        # 自動檢測高光區域
        combined = parallel + cross
        threshold = np.percentile(combined, config.specular_percentile)
        roi_mask = combined >= threshold
    
    if np.sum(roi_mask) < 10:
        # 沒有足夠的高光區域
        return True, 0.0, 1.0
    
    # 計算高光區域的偏振對比度
    parallel_roi = parallel[roi_mask]
    cross_roi = cross[roi_mask]
    
    # 對比度 = mean(I_parallel - I_cross)
    contrast = np.mean(parallel_roi - cross_roi)
    
    # 比率 = mean(I_parallel) / mean(I_cross)
    cross_mean = np.mean(cross_roi)
    if cross_mean > 0.001:
        ratio = np.mean(parallel_roi) / cross_mean
    else:
        ratio = float('inf')
    
    # 判斷條件：I_parallel > I_cross 且對比度足夠
    passed = contrast >= config.min_polarization_contrast and ratio > 1.0
    
    return passed, contrast, ratio


def check_alignment(
    img_input: np.ndarray,
    img_reference: np.ndarray,
    config: QualityCheckConfig,
    mask: Optional[np.ndarray] = None
) -> Tuple[bool, float]:
    """
    4. Ground Truth Alignment Filtering
    
    檢查輸入圖像與參考圖像（深度傳感器 RGB）的對齊精度
    使用邊緣結構比較
    
    Args:
        img_input: 輸入圖像
        img_reference: 參考圖像
        config: 配置參數
        mask: 前景遮罩
    
    Returns:
        (passed, alignment_error)
    """
    if img_input is None or img_reference is None:
        return True, 0.0  # 如果沒有參考圖，跳過此檢測
    
    # 轉換為 8-bit
    input_8bit = (normalize_image(img_input) * 255).astype(np.uint8)
    ref_8bit = (normalize_image(img_reference) * 255).astype(np.uint8)
    
    # 確保尺寸相同
    if input_8bit.shape != ref_8bit.shape:
        ref_8bit = cv2.resize(ref_8bit, (input_8bit.shape[1], input_8bit.shape[0]))
    
    # 邊緣檢測
    edges_input = cv2.Canny(input_8bit, config.edge_detection_threshold, 
                            config.edge_detection_threshold * 2)
    edges_ref = cv2.Canny(ref_8bit, config.edge_detection_threshold,
                          config.edge_detection_threshold * 2)
    
    # 背景區域遮罩
    if mask is not None:
        bg_mask = mask == 0
        edges_input = edges_input & (bg_mask.astype(np.uint8) * 255)
        edges_ref = edges_ref & (bg_mask.astype(np.uint8) * 255)
    
    # 使用相位相關計算位移
    if np.sum(edges_input) < 100 or np.sum(edges_ref) < 100:
        return True, 0.0
    
    # 計算互相關找位移
    result = cv2.matchTemplate(edges_input.astype(np.float32), 
                               edges_ref.astype(np.float32), 
                               cv2.TM_CCORR_NORMED)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # 計算位移量
    center = (edges_ref.shape[1] // 2, edges_ref.shape[0] // 2)
    displacement = np.sqrt((max_loc[0] - center[0])**2 + (max_loc[1] - center[1])**2)
    
    # 簡化：使用邊緣差異的平均位移估計
    # 實際實現可能需要更精確的方法
    edge_diff = np.abs(edges_input.astype(np.float32) - edges_ref.astype(np.float32))
    alignment_error = np.mean(edge_diff) / 255.0 * 5.0  # 轉換為像素單位估計
    
    passed = alignment_error <= config.max_alignment_error
    
    return passed, alignment_error


def check_depth_validity(
    depth: np.ndarray,
    config: QualityCheckConfig,
    mask: Optional[np.ndarray] = None
) -> Tuple[bool, float]:
    """
    5. Depth Validity Rate Filtering
    
    檢查透明物體區域的深度有效率
    
    Args:
        depth: 深度圖
        config: 配置參數
        mask: 透明物體遮罩
    
    Returns:
        (passed, validity_rate)
    """
    if depth is None:
        return False, 0.0
    
    # 定義有效深度
    valid_depth = (depth > config.min_valid_depth) & (depth < config.max_valid_depth)
    
    if mask is not None:
        roi_mask = mask > 0
        total_pixels = np.sum(roi_mask)
        
        if total_pixels == 0:
            return True, 1.0
        
        valid_pixels = np.sum(valid_depth & roi_mask)
        validity_rate = valid_pixels / total_pixels
    else:
        total_pixels = depth.size
        valid_pixels = np.sum(valid_depth)
        validity_rate = valid_pixels / total_pixels
    
    passed = validity_rate >= config.min_depth_validity_rate
    
    return passed, validity_rate


# ============================================================
# 主要檢測流程
# ============================================================

def check_sample(
    scene_name: str,
    files: Dict[str, str],
    config: QualityCheckConfig
) -> SampleQualityReport:
    """
    對單一樣本進行完整質量檢測
    
    支持的檔案格式:
    1. 新格式: left_parallel + right_cross (立體偏振)
    2. 舊格式: left_parallel + left_cross + right_parallel + right_cross
    3. 單相機: I_parallel + I_cross
    
    Args:
        scene_name: 場景名稱
        files: 場景相關檔案路徑
        config: 配置參數
    
    Returns:
        SampleQualityReport
    """
    report = SampleQualityReport(scene_name=scene_name)
    
    # 載入圖像
    img_left_parallel = None
    img_left_cross = None
    img_right_parallel = None
    img_right_cross = None
    
    # 嘗試載入各種格式
    if 'left_parallel' in files:
        img_left_parallel = load_image(files['left_parallel'])
    if 'left_cross' in files:
        img_left_cross = load_image(files['left_cross'])
    if 'right_parallel' in files:
        img_right_parallel = load_image(files['right_parallel'])
    if 'right_cross' in files:
        img_right_cross = load_image(files['right_cross'])
    
    # 格式 2: I_parallel, I_cross（單相機）
    if 'I_parallel' in files and img_left_parallel is None:
        img_left_parallel = load_image(files['I_parallel'])
    if 'I_cross' in files and img_left_cross is None:
        img_left_cross = load_image(files['I_cross'])
    
    # 載入深度和遮罩
    depth = load_image(files.get('depth', ''))
    mask = load_image(files.get('mask', ''))
    rgb_ref = load_image(files.get('rgb_ref', ''))
    
    # 如果有視差圖但沒有深度圖，可以轉換
    if depth is None and 'disparity' in files:
        disparity = load_image(files['disparity'])
        if disparity is not None:
            depth = disparity
    
    # ============================================================
    # 1. Geometric Consistency Filtering
    # 檢查立體對的垂直視差（Y 軸應對齊）
    # ============================================================
    # 優先使用 left_parallel + right_cross（新格式）
    if img_left_parallel is not None and img_right_cross is not None:
        passed, v_disp, n_matches = check_geometric_consistency(
            img_left_parallel, img_right_cross, config, mask
        )
        report.geometric_consistency = passed
        report.geometric_vertical_disparity = v_disp
        report.geometric_num_matches = n_matches
        
        if not passed:
            report.failure_reasons.append(
                f"Geometric: vertical disparity {v_disp:.2f}px > {config.max_vertical_disparity}px"
            )
    elif img_left_parallel is not None and img_right_parallel is not None:
        # 舊格式：left_parallel + right_parallel
        passed, v_disp, n_matches = check_geometric_consistency(
            img_left_parallel, img_right_parallel, config, mask
        )
        report.geometric_consistency = passed
        report.geometric_vertical_disparity = v_disp
        report.geometric_num_matches = n_matches
        
        if not passed:
            report.failure_reasons.append(
                f"Geometric: vertical disparity {v_disp:.2f}px > {config.max_vertical_disparity}px"
            )
    elif img_left_parallel is not None and img_left_cross is not None:
        # 單相機：left_parallel + left_cross
        passed, v_disp, n_matches = check_geometric_consistency(
            img_left_parallel, img_left_cross, config, mask
        )
        report.geometric_consistency = passed
        report.geometric_vertical_disparity = v_disp
        report.geometric_num_matches = n_matches
        
        if not passed:
            report.failure_reasons.append(
                f"Geometric: vertical disparity {v_disp:.2f}px > {config.max_vertical_disparity}px"
            )
    
    # ============================================================
    # 2. Background Photometric Consistency Filtering
    # 對於新格式 (left_parallel + right_cross)，跳過此檢測
    # 因為不同相機位置和不同偏振模式，背景會有差異
    # ============================================================
    if img_left_parallel is not None and img_left_cross is not None:
        # 只有同一相機位置的偏振對才檢查
        passed, bg_diff = check_photometric_consistency(
            img_left_parallel, img_left_cross, config, mask
        )
        report.photometric_consistency = passed
        report.photometric_background_diff = bg_diff
        
        if not passed:
            report.failure_reasons.append(
                f"Photometric: background diff {bg_diff:.4f} > {config.max_background_diff}"
            )
    else:
        # 新格式：跳過光度一致性檢查
        report.photometric_consistency = True
        report.photometric_background_diff = 0.0
    
    # ============================================================
    # 3. Polarization Signal Validity Filtering
    # 檢查偏振信號有效性
    # ============================================================
    if img_left_parallel is not None and img_left_cross is not None:
        # 同一相機位置的偏振對
        passed, contrast, ratio = check_polarization_validity(
            img_left_parallel, img_left_cross, config, mask
        )
        report.polarization_validity = passed
        report.polarization_contrast = contrast
        report.polarization_ratio = ratio
        
        if not passed:
            report.failure_reasons.append(
                f"Polarization: contrast {contrast:.4f} < {config.min_polarization_contrast} or ratio {ratio:.2f} <= 1.0"
            )
    elif img_left_parallel is not None and img_right_cross is not None:
        # 新格式：不同相機位置，使用不同的檢測方法
        # 只檢查是否有足夠的強度差異（可能來自偏振或視差）
        report.polarization_validity = True
        report.polarization_contrast = 0.0
        report.polarization_ratio = 1.0
    
    # ============================================================
    # 4. Ground Truth Alignment Filtering
    # ============================================================
    if img_left_parallel is not None and rgb_ref is not None:
        passed, align_err = check_alignment(
            img_left_parallel, rgb_ref, config, mask
        )
        report.alignment_check = passed
        report.alignment_error = align_err
        
        if not passed:
            report.failure_reasons.append(
                f"Alignment: error {align_err:.2f}px > {config.max_alignment_error}px"
            )
    else:
        # 沒有 RGB 參考圖，跳過對齊檢查
        report.alignment_check = True
        report.alignment_error = 0.0
    
    # ============================================================
    # 5. Depth Validity Rate Filtering
    # ============================================================
    if depth is not None:
        passed, validity_rate = check_depth_validity(depth, config, mask)
        report.depth_validity = passed
        report.depth_validity_rate = validity_rate
        
        if not passed:
            report.failure_reasons.append(
                f"Depth: validity rate {validity_rate:.1%} < {config.min_depth_validity_rate:.1%}"
            )
    
    # 更新總體通過狀態
    report.update_passed()
    
    return report


def run_quality_check(
    input_dir: str,
    output_dir: Optional[str] = None,
    config: Optional[QualityCheckConfig] = None,
    report_only: bool = False,
    verbose: bool = True
) -> Dict:
    """
    執行完整質量檢測流程
    
    Args:
        input_dir: 輸入目錄
        output_dir: 輸出目錄（通過的樣本會複製到這裡）
        config: 配置參數
        report_only: 僅生成報告，不複製檔案
        verbose: 是否打印詳細資訊
    
    Returns:
        質量檢測報告字典
    """
    if config is None:
        config = QualityCheckConfig()
    
    # 掃描場景
    scenes = find_scene_files(input_dir)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training Data Quality Check")
        print(f"{'='*60}")
        print(f"Input directory: {input_dir}")
        print(f"Found {len(scenes)} scenes")
        print(f"{'='*60}\n")
    
    # 檢測結果
    results = {
        'config': asdict(config),
        'timestamp': datetime.now().isoformat(),
        'input_dir': input_dir,
        'output_dir': output_dir,
        'total_scenes': len(scenes),
        'passed_scenes': 0,
        'failed_scenes': 0,
        'reports': [],
        'statistics': {
            'geometric_failures': 0,
            'photometric_failures': 0,
            'polarization_failures': 0,
            'alignment_failures': 0,
            'depth_failures': 0,
        }
    }
    
    # 創建輸出目錄
    if output_dir and not report_only:
        os.makedirs(output_dir, exist_ok=True)
    
    # 處理每個場景
    for scene_name, files in scenes.items():
        if verbose:
            print(f"Checking: {scene_name}...", end=' ')
        
        report = check_sample(scene_name, files, config)
        results['reports'].append(asdict(report))
        
        if report.passed:
            results['passed_scenes'] += 1
            status = "✓ PASSED"
            
            # 複製通過的檔案
            if output_dir and not report_only:
                for file_type, file_path in files.items():
                    dst_path = os.path.join(output_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, dst_path)
        else:
            results['failed_scenes'] += 1
            status = "✗ FAILED"
            
            # 統計失敗原因
            if not report.geometric_consistency:
                results['statistics']['geometric_failures'] += 1
            if not report.photometric_consistency:
                results['statistics']['photometric_failures'] += 1
            if not report.polarization_validity:
                results['statistics']['polarization_failures'] += 1
            if not report.alignment_check:
                results['statistics']['alignment_failures'] += 1
            if not report.depth_validity:
                results['statistics']['depth_failures'] += 1
        
        if verbose:
            print(status)
            if not report.passed:
                for reason in report.failure_reasons:
                    print(f"    - {reason}")
    
    # 打印摘要
    if verbose:
        print(f"\n{'='*60}")
        print(f"Summary")
        print(f"{'='*60}")
        print(f"Total scenes: {results['total_scenes']}")
        print(f"Passed: {results['passed_scenes']} ({results['passed_scenes']/max(1,results['total_scenes']):.1%})")
        print(f"Failed: {results['failed_scenes']} ({results['failed_scenes']/max(1,results['total_scenes']):.1%})")
        print(f"\nFailure breakdown:")
        print(f"  - Geometric consistency: {results['statistics']['geometric_failures']}")
        print(f"  - Photometric consistency: {results['statistics']['photometric_failures']}")
        print(f"  - Polarization validity: {results['statistics']['polarization_failures']}")
        print(f"  - Alignment check: {results['statistics']['alignment_failures']}")
        print(f"  - Depth validity: {results['statistics']['depth_failures']}")
        print(f"{'='*60}\n")
    
    return results


def save_report(results: Dict, output_path: str):
    """儲存質量檢測報告為 JSON"""
    
    def convert_numpy(obj):
        """轉換 numpy 類型為 Python 原生類型"""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    # 轉換所有 numpy 類型
    results_converted = convert_numpy(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {output_path}")


# ============================================================
# 命令列介面
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Training Data Quality Check Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # 基本使用
    python training_data_quality_check.py --input_dir ./dataset --output_dir ./filtered
    
    # 僅生成報告
    python training_data_quality_check.py --input_dir ./dataset --report_only
    
    # 自定義閾值
    python training_data_quality_check.py --input_dir ./dataset \\
        --max_vertical_disparity 0.5 \\
        --min_depth_validity 0.95
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='輸入資料目錄')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='輸出目錄（通過的樣本）')
    parser.add_argument('--report_only', action='store_true',
                        help='僅生成報告，不複製檔案')
    parser.add_argument('--report_path', type=str, default='quality_report.json',
                        help='報告輸出路徑')
    
    # 配置參數
    parser.add_argument('--max_vertical_disparity', type=float, default=1.0,
                        help='最大允許垂直視差 (pixels)')
    parser.add_argument('--max_background_diff', type=float, default=0.05,
                        help='背景最大亮度差異 (normalized)')
    parser.add_argument('--min_polarization_contrast', type=float, default=0.02,
                        help='最小偏振對比度')
    parser.add_argument('--max_alignment_error', type=float, default=1.0,
                        help='最大對齊誤差 (pixels)')
    parser.add_argument('--min_depth_validity', type=float, default=0.90,
                        help='最小深度有效率')
    
    parser.add_argument('--quiet', action='store_true',
                        help='安靜模式')
    
    args = parser.parse_args()
    
    # 建立配置
    config = QualityCheckConfig(
        max_vertical_disparity=args.max_vertical_disparity,
        max_background_diff=args.max_background_diff,
        min_polarization_contrast=args.min_polarization_contrast,
        max_alignment_error=args.max_alignment_error,
        min_depth_validity_rate=args.min_depth_validity,
    )
    
    # 執行檢測
    results = run_quality_check(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=config,
        report_only=args.report_only,
        verbose=not args.quiet
    )
    
    # 儲存報告
    save_report(results, args.report_path)


if __name__ == '__main__':
    main()

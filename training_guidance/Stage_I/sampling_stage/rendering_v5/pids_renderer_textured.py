"""
PIDS Stage 1 Renderer v5.1.8 (Strict Glass Mask)
=================================================

Fork from v4.0.0，修復偏振信號問題並整合 QA。

v5.1.8 更新:
-----------
1. 新增 glass_mask_strict（嚴格 mask，左右相機交集）
   - glass_mask（聯集）: 訓練用，覆蓋完整邊緣
   - glass_mask_strict（交集）: 評估用，只有確定是玻璃的像素
2. 新增 --rerender-strict-mask 模式
   - 用法: --rerender-strict-mask <params_dir> --obj-dir <obj_dir> --output <out> [--scene-list <file>]
   - 輸出: *_glass_mask_strict.exr

v5.1.7 更新:
-----------
1. _is_glass_name() 改用精確匹配 'Glass_Clear'
   - 舊邏輯: 'glass' in name -> 誤判 'glass_table', 'glass_shelf'
   - 新邏輯: name == 'Glass_Clear' (Blender 導出標準名稱)
2. 新增 --rerender-mask 模式
   - 使用已有 params.json 重新渲染 glass mask（聯集）
   - 不需重新渲染完整場景
   - 支持 --scene-list 篩選（如 train_scenes.txt）
   - 用法: --rerender-mask <params_dir> --obj-dir <obj_dir> --output <out> [--scene-list <file>]

v5.0.0 更新:
-----------
1. 天花板光強度降低 (100-250 → 10-25)，減少非偏振光干擾
2. DoLP 改用 Stokes 參數計算 (避免 65mm 基線誤差)
3. Glass Mask 改為左右相機聯集 (完整覆蓋邊緣)
4. DoLP 計算使用對齊的 mask (各相機用自己視角)
5. SNR 改用 DoLP-based 計算 (避免亮度差異影響)
6. 整合 QA 驗證 (渲染完成自動執行)

v4.0.0 更新:
-----------
- 新增 MTLParser.parse() 支持 map_Kd 紋理路徑解析
- 新增 MaterialFactory.diffuse_textured() 方法
- 修改 _create_meshes() 支持紋理材質
- 保持向後相容：無紋理時 fallback 到純色

紋理支持說明:
-----------
Mitsuba 3 spectral_polarized variant 只支持 Diffuse 紋理。
- 支持: map_Kd (diffuse/albedo)
- 不支持: map_Bump, map_Ks, map_Ns 等（會被忽略）

座標系統 (重要！):
-----------------
Blender/OBJ 座標系 → Mitsuba 座標系 (經過 -90° X 軸旋轉)
    Blender X → Mitsuba X (左右)
    Blender Y → Mitsuba Z (深度/前後)
    Blender Z → Mitsuba Y (高度)

Chamber 配置 (Blender/OBJ 座標，單位 mm):
-----------------------------------------
    前牆:   Y = 350mm
    相機:   Y = 360mm (在 chamber 內！)
    玻璃區: Y = 528-695mm
    傢俱區: Y = 700-800mm
    後牆:   Y = 900mm
    寬度:   X = -300 ~ +300mm (共 600mm)
    高度:   Z = 0 ~ 300mm

相機配置 (平行光軸):
------------------
    感測器: Sony IMX296LQR-C (5.023 × 3.754 mm)
    焦距: 6mm
    基線: 65mm
    FOV: 45.4° (水平)
    左相機: X = -32.5mm, Y = 200mm, 0° 偏振片 (I∥)
    右相機: X = +32.5mm, Y = 200mm, 90° 偏振片 (I⊥)
    光軸方向: (0, 1, 0) - 兩相機完全平行，無會聚

渲染原理:
--------
1. 使用 Mitsuba 3 spectral_polarized variant
2. Stokes integrator 輸出 [S0, S1, S2, S3]
3. 偏振片透射: I(θ) = 0.5 * (S0 + S1*cos(2θ) + S2*sin(2θ))
4. I∥ (θ=0°) = 0.5 * (S0 + S1)
5. I⊥ (θ=90°) = 0.5 * (S0 - S1)

作者: PIDS Project
版本: 5.1.6
日期: 2026-01-06

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

from __future__ import annotations

import numpy as np
import cv2
import os
import json
import argparse
import gc
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import hashlib
import sys

# QA 模組（延遲載入）
QualityValidator = None

def lazy_import_qa():
    """延遲載入 QA 模組"""
    global QualityValidator
    if QualityValidator is None:
        try:
            # 方法1: 直接 import（如果已在 PYTHONPATH）
            from Quality_Assurance.quality_validator import QualityValidator as QV
            QualityValidator = QV
        except ImportError:
            try:
                # 方法2: 將腳本所在目錄加入 sys.path
                script_dir = Path(__file__).parent.resolve()
                if str(script_dir) not in sys.path:
                    sys.path.insert(0, str(script_dir))
                from Quality_Assurance.quality_validator import QualityValidator as QV
                QualityValidator = QV
            except ImportError:
                # 方法3: 如果是符號連結或複製到其他位置，返回 None
                print("[QA] 無法載入 Quality_Assurance 模組，跳過 QA 驗證")
                return None
    return QualityValidator

# 延遲 import mitsuba，避免主進程初始化 CUDA
mi = None
dr = None

def lazy_import_mitsuba():
    """延遲載入 mitsuba，只在需要時才初始化"""
    global mi, dr
    if mi is None:
        import mitsuba as _mi
        import drjit as _dr
        mi = _mi
        dr = _dr


def deterministic_hash(s: str) -> int:
    """確定性 hash，跨 Python 會話一致"""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)


def compute_world_aligned_theta(origin: Tuple[float, float, float],
                                  target: Tuple[float, float, float],
                                  world_theta: float = 0.0,
                                  up: Tuple[float, float, float] = (0, 1, 0)) -> float:
    """
    計算使偏振角度在世界坐標系中保持一致的 theta 值。

    問題：Mitsuba 的 polarizer BSDF 中的 theta 是相對於 look_at 創建的
    局部坐標系。當 origin/target 變化時，局部坐標系旋轉，導致
    同樣的 theta 值在世界坐標中代表不同的偏振角度。

    解決方案：計算補償角度，使得無論 look_at 方向如何變化，
    世界坐標中的偏振角度都保持 world_theta。

    Args:
        origin: 偏振片位置（Mitsuba 坐標，單位 m）
        target: 偏振片朝向的目標點（Mitsuba 坐標，單位 m）
        world_theta: 期望的世界坐標偏振角度（度），0=水平，90=垂直
        up: 世界向上方向（默認 Mitsuba Y 軸）

    Returns:
        theta: 需要在 polarizer BSDF 中使用的角度（度）

    物理說明：
        - 世界坐標系：X=水平右, Y=垂直上, Z=深度
        - world_theta=0° 表示偏振方向沿世界 X 軸（水平）
        - world_theta=90° 表示偏振方向沿世界 Y 軸（垂直）
    """
    origin = np.array(origin)
    target = np.array(target)
    up = np.array(up)

    # 計算 look_at 的局部坐標系
    forward = target - origin
    forward_len = np.linalg.norm(forward)
    if forward_len < 1e-9:
        return world_theta  # origin == target，無法計算，返回原值
    forward = forward / forward_len

    # 計算 right 軸 (局部 X)
    right = np.cross(up, forward)
    right_len = np.linalg.norm(right)
    if right_len < 1e-9:
        # forward 與 up 平行，使用世界 X 軸作為 right
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_len

    # 計算 local up 軸 (局部 Y)
    local_up = np.cross(forward, right)
    local_up = local_up / np.linalg.norm(local_up)

    # 世界坐標中的偏振方向
    # world_theta=0° → 水平方向 (世界 X 軸)
    # world_theta=90° → 垂直方向 (世界 Y 軸)
    world_theta_rad = np.radians(world_theta)
    world_pol_dir = np.array([np.cos(world_theta_rad), np.sin(world_theta_rad), 0.0])

    # 將世界偏振方向投影到垂直於 forward 的平面上
    # (偏振方向必須垂直於光傳播方向)
    proj = world_pol_dir - np.dot(world_pol_dir, forward) * forward
    proj_len = np.linalg.norm(proj)
    if proj_len < 1e-9:
        # 世界偏振方向與 forward 平行，無法投影
        # 這種情況下使用局部水平
        return 0.0
    proj = proj / proj_len

    # 計算投影在局部坐標系中的角度
    # 局部坐標系：right = 局部 X，local_up = 局部 Y
    cos_theta = np.dot(proj, right)
    sin_theta = np.dot(proj, local_up)
    local_theta = np.arctan2(sin_theta, cos_theta)

    return np.degrees(local_theta)


# ============================================================
# 配置
# ============================================================

class Config:
    """渲染配置 - 所有單位為毫米 (mm)，除非特別說明"""

    # 渲染設定
    WIDTH = 640
    HEIGHT = 480
    SPP = 1024            # 每像素樣本數 (1K，批渲染用)
    SPP_PER_BATCH = 512   # 分批渲染，避免 GPU OOM (2 批次)
    MAX_DEPTH = 12        # 光線反彈次數

    # QA 指標 floor 值（避免除以零產生無意義的超大比值）
    DOLP_FLOOR = 0.01     # 背景 DoLP 下限 (1%)，用於計算比值

    # Chamber 尺寸 (Blender/OBJ 座標系，單位 mm)
    # 這些值來自 blender_furniture_randomizer_v17.py
    CHAMBER_WIDTH = 600.0           # X 方向: -300 ~ +300
    CHAMBER_HEIGHT = 300.0          # Z 方向: 0 ~ 300
    CHAMBER_Y_FRONT = 350.0         # 前牆 Y 位置
    CHAMBER_Y_BACK = 900.0          # 後牆 Y 位置

    # 立體相機位置 (在 chamber 內)
    CAMERA_X = -50.0                # 相機 X 偏移 (向左)
    CAMERA_Y = 400.0                # 相機 Y 位置 (往前移，深入 chamber)
    CAMERA_Z = 80.0                 # 相機高度 (較低)

    # 深度相機位置 (與立體相機同水平，光軸平行)
    DEPTH_CAMERA_OFFSET_Z = 0.0     # 深度相機與立體相機同高度

    # 立體相機
    BASELINE = 65.0                 # 基線距離 (mm)

    # 感測器: Sony IMX296LQR-C
    SENSOR_WIDTH = 5.023            # 感測器寬度 (mm) = 1456 × 3.45μm
    SENSOR_HEIGHT = 3.754           # 感測器高度 (mm) = 1088 × 3.45μm
    FOCAL_LENGTH = 6.0              # 鏡頭焦距 (mm)

    # FOV = 2 × arctan(sensor_width / (2 × focal_length)) ≈ 45.4°
    FOV = 45.4                      # 水平視場角 (度)

    # 目標區域 (玻璃物體範圍)
    GLASS_Y_MIN = 528.0
    GLASS_Y_MAX = 695.0

    # 光源配置 - 整面偏振光源
    # 原理：大面積均勻偏振照明
    # - 玻璃鏡面反射 → 保持偏振 → I∥ ≠ I⊥
    # - 背景漫反射 → 去偏振 → I∥ ≈ I⊥
    # 位置：相機同側（前牆位置），面向場景
    LED_INTENSITY = 10000.0         # LED 強度
    LED_SIZE = (500.0, 250.0)       # LED 尺寸：覆蓋整個場景 (寬500mm x 高250mm)
    LED_POSITION_X = 0.0            # LED X 位置（中心，均勻照明）
    LED_POSITION_Y = 150.0          # LED 高度 (場景中間)
    LED_POSITION_Z = 380.0          # LED 深度 (相機前方，前牆附近)
    LED_TARGET = (0.0, 600.0, 150.0)  # LED 朝向：場景中心

    # 環境光
    AMBIENT_INTENSITY = 0.0 # 已停用，由天花板發光體取代

    # 天花板燈陣列（非偏振光源，稀釋背景殘餘偏振）
    # 原理：背景漫反射需要多次 bounce 才能完全去偏振
    # 加入非偏振頂光，讓背景有更多非偏振光，稀釋殘餘偏振
    CEILING_LIGHTS_ENABLED = True
    CEILING_EMITTER_INTENSITY = 800.0   # 非偏振頂光強度

    # 四個燈的位置 (OBJ 座標)
    CEILING_LIGHT_POSITIONS = [
        (-150.0, 550.0, 295.0),  # 左前
        (150.0, 550.0, 295.0),   # 右前
        (-150.0, 750.0, 295.0),  # 左後
        (150.0, 750.0, 295.0),   # 右後
    ]

    # 材質
    GLASS_IOR = 1.65                # 玻璃折射率（固定，正入射 R≈6.2%）
    GLASS_ROUGHNESS = 0.02          # 玻璃粗糙度

    # 輸出
    SAVE_PREVIEW = True
    SAVE_DOLP = True

    @classmethod
    def target_point(cls) -> Tuple[float, float, float]:
        """相機目標點 (玻璃區域中心) - 用於計算視線方向"""
        target_y = (cls.GLASS_Y_MIN + cls.GLASS_Y_MAX) / 2
        return (0.0, target_y, cls.CAMERA_Z)

    @classmethod
    def forward_direction(cls) -> Tuple[float, float, float]:
        """
        相機光軸方向（歸一化）

        從相機中心指向目標區域的方向向量。
        所有相機都應該使用相同的視線方向，實現平行光軸。
        """
        # 中心相機位置
        center_pos = np.array([0.0, cls.CAMERA_Y, cls.CAMERA_Z])
        target = np.array(cls.target_point())
        direction = target - center_pos
        direction = direction / np.linalg.norm(direction)
        return tuple(direction)

    @classmethod
    def left_camera_position(cls) -> Tuple[float, float, float]:
        """左相機位置"""
        return (cls.CAMERA_X - cls.BASELINE / 2, cls.CAMERA_Y, cls.CAMERA_Z)

    @classmethod
    def right_camera_position(cls) -> Tuple[float, float, float]:
        """右相機位置"""
        return (cls.CAMERA_X + cls.BASELINE / 2, cls.CAMERA_Y, cls.CAMERA_Z)

    @classmethod
    def depth_camera_position(cls) -> Tuple[float, float, float]:
        """深度相機位置 (與左相機相同，確保視差計算正確)"""
        return cls.left_camera_position()

    @classmethod
    def camera_target_for_position(cls, camera_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        計算相機的個別目標點（確保平行光軸）

        每個相機的目標點是：相機位置 + 視線方向 * 固定距離
        這樣所有相機都有相同的視線方向，實現平行光軸。

        Args:
            camera_pos: 相機位置 (OBJ 座標)

        Returns:
            相機的目標點，確保光軸與其他相機平行
        """
        # 使用統一的視線方向
        direction = np.array(cls.forward_direction())

        # 計算到目標區域的距離
        center_pos = np.array([0.0, cls.CAMERA_Y, cls.CAMERA_Z])
        target = np.array(cls.target_point())
        distance = np.linalg.norm(target - center_pos)

        # 各相機的個別目標點 = 相機位置 + 方向 * 距離
        camera_pos_arr = np.array(camera_pos)
        camera_target = camera_pos_arr + direction * distance

        return tuple(camera_target)

    @classmethod
    def randomize_for_augmentation(cls, seed: int = None):
        """
        隨機化渲染參數（數據增強用）

        保持不變：
        - BASELINE (65mm) - 立體基線
        - FOV (45.4°) - 視場角
        - 感測器尺寸 - 硬體規格
        - 偏振片角度 (0°/90°) - 物理原理

        隨機化：
        - LED_INTENSITY: 偏振光強度 [20000, 30000]
        - CEILING_EMITTER_INTENSITY: 環境光強度 [1, 3]
        - CAMERA_X: 整體水平位置 [-120, 20]
          (左右相機、深度相機、光源一起移動)

        Args:
            seed: 隨機種子，用於可重現性
        """
        import random
        if seed is not None:
            random.seed(seed)

        # 偏振光強度 [8000, 12000]
        cls.LED_INTENSITY = random.uniform(8000, 12000)

        # 非偏振頂光強度 [600, 1000]
        cls.CEILING_EMITTER_INTENSITY = random.uniform(600, 1000)

        # 整體水平位置 [-120, 20]（所有相機和光源一起移動）
        cls.CAMERA_X = random.uniform(-120, 20)

        print(f"  [Augment] LED={cls.LED_INTENSITY:.0f}, "
              f"Ceiling={cls.CEILING_EMITTER_INTENSITY:.2f}, "
              f"CamX={cls.CAMERA_X:.1f}mm")


def mm_to_m(mm: float) -> float:
    """毫米轉米"""
    return mm / 1000.0


# ============================================================
# 數據集整理功能
# ============================================================

def organize_dataset(
    input_dir: Path,
    output_dir: Path,
    train_size: int = None,
    seed: int = 42,
    copy_mode: bool = True,
) -> None:
    """
    整理渲染輸出為訓練數據集格式

    只保留訓練必要的 EXR 檔案：
    - stereo_pairs/: left_parallel.exr, right_cross.exr
    - ground_truth/: disparity.exr
    - masks/: glass_mask.exr

    Args:
        input_dir: 渲染輸出目錄
        output_dir: 整理後的數據集目錄
        train_size: 訓練集大小（剩餘為測試集），None 表示全部作為訓練集
        seed: 隨機種子
        copy_mode: True=複製, False=移動（預設 True）
    """
    import re
    import random
    from collections import defaultdict

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"[Organize] 錯誤: 輸入目錄不存在: {input_dir}")
        return

    def get_scene_name(filename: str) -> str:
        """從檔名提取場景名稱"""
        match = re.match(r'(scene_\d+)', filename)
        return match.group(1) if match else None

    def is_training_file(filename: str) -> tuple:
        """判斷是否為訓練必要檔案"""
        if filename.endswith('.exr'):
            if '_left_parallel.exr' in filename or '_right_cross.exr' in filename:
                return ('stereo_pairs', True)
            if filename.endswith('_left.exr') or filename.endswith('_right.exr'):
                return ('stereo_pairs', True)
            if '_disparity.exr' in filename:
                return ('disparity', True)
            if '_glass_mask.exr' in filename:
                return ('masks', True)
        return (None, False)

    # 解析 quality_report.md 獲取失敗場景
    failed_scenes = set()
    report_path = input_dir / 'quality_report.md'
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 尋找表格中標記為 ✗ 的場景
        pattern = r'\|\s*(scene_\d+)\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*✗\s*\|'
        matches = re.findall(pattern, content)
        failed_scenes.update(matches)
        # 也檢查詳細報告區塊
        pattern2 = r'###\s+(scene_\d+)\s+✗'
        matches2 = re.findall(pattern2, content)
        failed_scenes.update(matches2)
        print(f"[Organize] 從 quality_report.md 排除 {len(failed_scenes)} 個不合格場景")

    # 收集所有合格場景
    scenes = set()
    for filepath in input_dir.iterdir():
        if not filepath.is_file():
            continue
        scene_name = get_scene_name(filepath.name)
        if scene_name and scene_name not in failed_scenes:
            _, keep = is_training_file(filepath.name)
            if keep:
                scenes.add(scene_name)

    all_scenes = sorted(scenes)
    total_scenes = len(all_scenes)
    print(f"[Organize] 合格場景數: {total_scenes}")

    # 分割訓練/測試集
    if train_size and train_size < total_scenes:
        random.seed(seed)
        random.shuffle(all_scenes)
        train_scenes = set(all_scenes[:train_size])
        test_scenes = set(all_scenes[train_size:])
        print(f"[Organize] 訓練/測試分割: {len(train_scenes)} / {len(test_scenes)} (seed={seed})")
    else:
        train_scenes = set(all_scenes)
        test_scenes = set()

    # 創建輸出目錄結構
    if test_scenes:
        train_subdirs = {
            'stereo_pairs': output_dir / 'train' / 'stereo_pairs',
            'disparity': output_dir / 'train' / 'ground_truth',
            'masks': output_dir / 'train' / 'masks',
        }
        test_subdirs = {
            'stereo_pairs': output_dir / 'test' / 'stereo_pairs',
            'disparity': output_dir / 'test' / 'ground_truth',
            'masks': output_dir / 'test' / 'masks',
        }
    else:
        train_subdirs = {
            'stereo_pairs': output_dir / 'stereo_pairs',
            'disparity': output_dir / 'ground_truth',
            'masks': output_dir / 'masks',
        }
        test_subdirs = {}

    for subdir in train_subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    for subdir in test_subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)

    # 統計
    train_stats = defaultdict(int)
    test_stats = defaultdict(int)

    print(f"\n[Organize] 開始整理...")
    print(f"  模式: {'複製' if copy_mode else '移動'}")

    # 處理每個檔案
    for filepath in sorted(input_dir.iterdir()):
        if not filepath.is_file():
            continue

        filename = filepath.name
        scene_name = get_scene_name(filename)
        category, keep = is_training_file(filename)

        if not keep:
            continue

        if scene_name in failed_scenes:
            continue

        # 決定是訓練還是測試
        if scene_name in train_scenes:
            target_dir = train_subdirs[category]
            stats = train_stats
        elif scene_name in test_scenes:
            target_dir = test_subdirs[category]
            stats = test_stats
        else:
            continue

        target_path = target_dir / filename

        if copy_mode:
            shutil.copy2(filepath, target_path)
        else:
            shutil.move(filepath, target_path)

        stats[category] += 1

    # 顯示統計
    train_total = sum(train_stats.values())
    train_scene_count = train_stats['stereo_pairs'] // 2

    print(f"\n[Organize] 完成!")
    print(f"  訓練集: {train_scene_count} 場景, {train_total} 檔案")

    if test_scenes:
        test_total = sum(test_stats.values())
        test_scene_count = test_stats['stereo_pairs'] // 2
        print(f"  測試集: {test_scene_count} 場景, {test_total} 檔案")

    # 複製 quality_report.md
    if report_path.exists():
        shutil.copy2(report_path, output_dir / 'quality_report.md')

    # 輸出場景列表
    if test_scenes:
        with open(output_dir / 'train_scenes.txt', 'w') as f:
            for scene in sorted(train_scenes):
                f.write(f"{scene}\n")
        with open(output_dir / 'test_scenes.txt', 'w') as f:
            for scene in sorted(test_scenes):
                f.write(f"{scene}\n")

    print(f"  輸出目錄: {output_dir}")


# ============================================================
# Mitsuba 設定
# ============================================================

def setup_mitsuba() -> str:
    """
    設定 Mitsuba variant

    必須使用 spectral_polarized variant！
    mono_polarized 在 luminance film 下會丟棄 S1/S2/S3
    """
    lazy_import_mitsuba()
    available = mi.variants()
    print(f"[Mitsuba] 可用 variants: {available}")

    # 優先順序：cuda > llvm > scalar
    preferred = [
        'cuda_ad_spectral_polarized',
        'cuda_spectral_polarized',
        'llvm_ad_spectral_polarized',
        'llvm_spectral_polarized',
        'scalar_spectral_polarized',
    ]

    for variant in preferred:
        if variant in available:
            try:
                # 初始化 DrJit CUDA backend
                if 'cuda' in variant:
                    dr.set_flag(dr.JitFlag.Debug, False)
                mi.set_variant(variant)
                print(f"[Mitsuba] 使用 variant: {variant}")
                return variant
            except ImportError as e:
                print(f"[Mitsuba] {variant} 初始化失敗: {e}")
                continue

    raise RuntimeError(
        f"找不到 spectral_polarized variant!\n"
        f"可用: {available}\n"
        f"請安裝包含 polarized 支援的 Mitsuba 3"
    )


# ============================================================
# 材質工廠
# ============================================================

class MaterialFactory:
    """材質創建工廠"""

    @staticmethod
    def glass(ior: float = Config.GLASS_IOR) -> Dict:
        """
        創建玻璃材質

        使用 dielectric 產生實心玻璃（有折射效果）
        """
        return {
            'type': 'dielectric',
            'int_ior': ior,
        }

    @staticmethod
    def diffuse(reflectance: float) -> Dict:
        """
        創建漫反射材質

        對於 spectral variant，使用單一反射率值
        """
        return {
            'type': 'diffuse',
            'reflectance': {
                'type': 'spectrum',
                'value': reflectance,
            },
        }

    @staticmethod
    def diffuse_rgb(color: Tuple[float, float, float]) -> Dict:
        """從 RGB 顏色創建漫反射材質"""
        # 計算亮度作為反射率
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        return MaterialFactory.diffuse(luminance)

    @staticmethod
    def diffuse_textured(texture_path: str) -> Dict:
        """
        創建帶紋理貼圖的漫反射材質

        使用 bitmap 紋理作為 diffuse reflectance。
        適用於 Mitsuba 3 spectral_polarized variant。

        Args:
            texture_path: 紋理圖像的絕對路徑

        Returns:
            Mitsuba BSDF dict
        """
        return {
            'type': 'diffuse',
            'reflectance': {
                'type': 'bitmap',
                'filename': texture_path,
                'filter_type': 'bilinear',
                'wrap_mode': 'repeat',
            },
        }


# ============================================================
# MTL 解析器
# ============================================================

class MTLParser:
    """OBJ MTL 材質檔案解析器（支持紋理）"""

    # 非玻璃材質關鍵字（優先排除，避免誤判）
    NON_GLASS_KEYWORDS = [
        'background', 'wall', 'floor', 'ground', 'ceiling',
        'diffuse', 'opaque', 'solid', 'wood', 'metal', 'fabric',
        'concrete', 'brick', 'stone', 'plastic', 'rubber',
    ]

    # 玻璃材質精確名稱（僅匹配 Blender 導出的標準名稱）
    # Blender furniture randomizer 使用 "Glass_Clear" 作為玻璃材質名稱
    GLASS_EXACT_NAMES = ['glass_clear']

    @classmethod
    def parse(cls, mtl_path: str) -> Dict[str, Dict]:
        """
        解析 MTL 檔案

        支持的 MTL 命令:
        - newmtl: 材質名稱
        - Kd: diffuse 顏色
        - d: 透明度
        - illum: 光照模型
        - map_Kd: diffuse 紋理路徑 (新增)

        Returns:
            Dict[mat_name, {
                'is_glass': bool,
                'color': Tuple[float, float, float],
                'textures': {
                    'diffuse': Optional[str]  # 紋理絕對路徑
                }
            }]
        """
        materials = {}
        current = None
        mtl_dir = os.path.dirname(os.path.abspath(mtl_path))

        if not os.path.exists(mtl_path):
            print(f"  [MTL] 找不到: {mtl_path}")
            return materials

        with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                cmd = parts[0].lower()

                if cmd == 'newmtl' and len(parts) > 1:
                    current = parts[1]
                    materials[current] = {
                        'is_glass': cls._is_glass_name(current),
                        'color': (0.5, 0.5, 0.5),
                        'textures': {
                            'diffuse': None,
                        },
                    }
                elif current:
                    if cmd == 'kd' and len(parts) >= 4:
                        materials[current]['color'] = (
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                        )
                    elif cmd == 'd' and len(parts) >= 2:
                        if float(parts[1]) < 0.95:
                            materials[current]['is_glass'] = True
                    elif cmd == 'illum' and len(parts) >= 2:
                        if int(parts[1]) in [4, 6, 7, 9]:
                            materials[current]['is_glass'] = True
                    elif cmd == 'map_kd' and len(parts) >= 2:
                        # 解析 diffuse 紋理路徑
                        # 支援帶空格的路徑（整行剩餘部分）
                        tex_path_raw = ' '.join(parts[1:])
                        tex_abs_path = cls._resolve_texture_path(tex_path_raw, mtl_dir)
                        if tex_abs_path:
                            materials[current]['textures']['diffuse'] = tex_abs_path
                            print(f"  [MTL] 材質 '{current}' 紋理: {tex_abs_path}")

        return materials

    @classmethod
    def _resolve_texture_path(cls, tex_path: str, mtl_dir: str) -> Optional[str]:
        """
        解析紋理路徑，轉換為絕對路徑

        支持:
        - 絕對路徑
        - 相對於 MTL 檔案的相對路徑
        - 只有檔名（在 MTL 同目錄搜尋）

        Args:
            tex_path: MTL 中的紋理路徑
            mtl_dir: MTL 檔案所在目錄

        Returns:
            紋理的絕對路徑，如果找不到則返回 None
        """
        # 移除可能的引號
        tex_path = tex_path.strip('"\'')

        # 處理 Windows/Unix 路徑分隔符
        tex_path = tex_path.replace('\\', '/')

        # 1. 已經是絕對路徑
        if os.path.isabs(tex_path):
            if os.path.exists(tex_path):
                return os.path.abspath(tex_path)
            else:
                print(f"    [Warning] 紋理不存在: {tex_path}")
                return None

        # 2. 相對路徑（相對於 MTL 目錄）
        abs_path = os.path.normpath(os.path.join(mtl_dir, tex_path))
        if os.path.exists(abs_path):
            return abs_path

        # 3. 只有檔名，在 MTL 同目錄搜尋
        basename = os.path.basename(tex_path)
        same_dir_path = os.path.join(mtl_dir, basename)
        if os.path.exists(same_dir_path):
            return os.path.abspath(same_dir_path)

        # 4. 搜尋常見紋理子目錄
        common_dirs = ['textures', 'tex', 'maps', 'images']
        for subdir in common_dirs:
            subdir_path = os.path.join(mtl_dir, subdir, basename)
            if os.path.exists(subdir_path):
                return os.path.abspath(subdir_path)

        print(f"    [Warning] 找不到紋理: {tex_path} (搜尋於 {mtl_dir})")
        return None

    @classmethod
    def _is_glass_name(cls, name: str) -> bool:
        """
        根據名稱判斷是否為玻璃材質

        v5.1.7 修正: 使用精確匹配而非關鍵字匹配
        - 舊邏輯: 'glass' in name -> 會誤判 'glass_table', 'glass_shelf' 等
        - 新邏輯: name == 'Glass_Clear' (Blender 導出的標準玻璃材質名稱)

        MTL 屬性 (d < 0.95, illum in [4,6,7,9]) 仍作為後備判斷
        """
        name_lower = name.lower()
        return name_lower in cls.GLASS_EXACT_NAMES


class OBJSplitter:
    """OBJ 檔案分離器 - 按材質分離玻璃、天花板和其他幾何"""

    # 天花板材質關鍵字
    CEILING_KEYWORDS = ['ceiling', 'roof', 'top', 'sky', 'plafond', 'techo']

    def __init__(self, obj_path: str, materials: Dict[str, Dict]):
        self.obj_path = obj_path
        self.materials = materials
        self.vertices = []      # v
        self.normals = []       # vn
        self.texcoords = []     # vt
        self.glass_faces = []   # 玻璃材質的面
        self.ceiling_faces = [] # 天花板材質的面
        self.other_faces = []   # 其他材質的面

    @classmethod
    def _is_ceiling(cls, mat_name: str) -> bool:
        """判斷是否為天花板材質"""
        name_lower = mat_name.lower()
        return any(kw in name_lower for kw in cls.CEILING_KEYWORDS)

    def parse_and_split(self) -> Tuple[str, str, str]:
        """
        解析 OBJ 並分離為三個檔案

        Returns:
            (glass_obj_path, ceiling_obj_path, other_obj_path)
        """
        current_material = None
        current_type = 'other'  # 'glass', 'ceiling', 'other'

        with open(self.obj_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                cmd = parts[0]

                if cmd == 'v' and len(parts) >= 4:
                    self.vertices.append(line)
                elif cmd == 'vn' and len(parts) >= 4:
                    self.normals.append(line)
                elif cmd == 'vt' and len(parts) >= 3:
                    self.texcoords.append(line)
                elif cmd == 'usemtl' and len(parts) >= 2:
                    current_material = parts[1]
                    mat_info = self.materials.get(current_material, {})

                    # 判斷材質類型（優先順序：玻璃 > 天花板 > 其他）
                    if mat_info.get('is_glass', False):
                        current_type = 'glass'
                        print(f"  [OBJ分離] 玻璃材質: {current_material}")
                    elif self._is_ceiling(current_material):
                        current_type = 'ceiling'
                        print(f"  [OBJ分離] 天花板材質: {current_material}")
                    else:
                        current_type = 'other'
                elif cmd == 'f':
                    if current_type == 'glass':
                        self.glass_faces.append(line)
                    elif current_type == 'ceiling':
                        self.ceiling_faces.append(line)
                    else:
                        self.other_faces.append(line)

        # 寫入分離的 OBJ 檔案到臨時目錄（避免污染場景目錄）
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix='pids_split_')
        base_name = os.path.splitext(os.path.basename(self.obj_path))[0]

        glass_path = os.path.join(temp_dir, f"{base_name}_glass.obj")
        ceiling_path = os.path.join(temp_dir, f"{base_name}_ceiling.obj")
        other_path = os.path.join(temp_dir, f"{base_name}_other.obj")

        self.temp_dir = temp_dir  # 保存臨時目錄路徑，供後續清理

        self._write_obj(glass_path, self.glass_faces)
        self._write_obj(ceiling_path, self.ceiling_faces)
        self._write_obj(other_path, self.other_faces)

        print(f"  [OBJ分離] 玻璃面數: {len(self.glass_faces)}")
        print(f"  [OBJ分離] 天花板面數: {len(self.ceiling_faces)}")
        print(f"  [OBJ分離] 其他面數: {len(self.other_faces)}")

        return glass_path, ceiling_path, other_path

    def _write_obj(self, path: str, faces: List[str]):
        """寫入 OBJ 檔案"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Split OBJ file\n")
            # 寫入所有頂點（保持索引一致）
            for v in self.vertices:
                f.write(v + '\n')
            for vn in self.normals:
                f.write(vn + '\n')
            for vt in self.texcoords:
                f.write(vt + '\n')
            # 寫入面
            for face in faces:
                f.write(face + '\n')


# ============================================================
# 場景建構器
# ============================================================

class SceneBuilder:
    """Mitsuba 場景建構器（支持紋理）"""

    def __init__(self, obj_path: str):
        # 使用絕對路徑，確保 Mitsuba 能找到檔案
        self.obj_path = str(Path(obj_path).resolve())
        self.mtl_path = self.obj_path.replace('.obj', '.mtl')

        # 驗證檔案存在
        if not os.path.exists(self.obj_path):
            raise FileNotFoundError(f"OBJ 檔案不存在: {self.obj_path}")

        print(f"[SceneBuilder] OBJ 路徑: {self.obj_path}")
        print(f"[SceneBuilder] MTL 路徑: {self.mtl_path}")

        self.materials = MTLParser.parse(self.mtl_path)

        # 統計紋理材質
        textured_count = sum(1 for m in self.materials.values()
                            if m.get('textures', {}).get('diffuse'))
        print(f"[SceneBuilder] 材質總數: {len(self.materials)}, 帶紋理: {textured_count}")

        # 分離玻璃、天花板和其他幾何
        splitter = OBJSplitter(self.obj_path, self.materials)
        self.glass_obj_path, self.ceiling_obj_path, self.other_obj_path = splitter.parse_and_split()
        self.has_glass = len(splitter.glass_faces) > 0
        self.has_ceiling = len(splitter.ceiling_faces) > 0
        self.temp_dir = splitter.temp_dir  # 保存臨時目錄路徑

    def cleanup(self):
        """清理臨時分離的 OBJ 檔案"""
        import shutil
        if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"  [Cleanup] 已刪除臨時目錄: {self.temp_dir}")

    def build(self,
              camera_position: Tuple[float, float, float],
              camera_target: Tuple[float, float, float],
              spp: int,
              camera_polarizer_angle: Optional[float] = None) -> Dict:
        """
        建構完整場景

        座標轉換：OBJ (mm) → Mitsuba (m)，並旋轉 -90° X 軸

        Args:
            camera_position: 相機位置 (OBJ 座標)
            camera_target: 相機目標點 (OBJ 座標)
            spp: 每像素樣本數
            camera_polarizer_angle: 相機前偏振片角度 (度)，None 表示不加偏振片
        """
        scene = {
            'type': 'scene',
            'integrator': self._create_integrator(),
            'sensor': self._create_sensor(camera_position, camera_target, spp),
        }

        # 添加光源（新架構：非偏振光源 + 獨立偏振片）
        led_emitter, led_polarizer = self._create_led_light(camera_position)
        scene['led_emitter'] = led_emitter
        scene['led_polarizer'] = led_polarizer

        # 添加相機前偏振片（如果指定）
        if camera_polarizer_angle is not None:
            scene['camera_polarizer'] = self._create_camera_polarizer(
                camera_position, camera_target, camera_polarizer_angle
            )

        # 添加分離的 OBJ 網格
        for name, mesh_dict in self._create_meshes():
            scene[name] = mesh_dict

        return scene

    def _create_camera_polarizer(self,
                                  camera_position: Tuple[float, float, float],
                                  camera_target: Tuple[float, float, float],
                                  world_theta: float) -> Dict:
        """
        創建相機前方的偏振片

        Args:
            camera_position: 相機位置 (OBJ 座標)
            camera_target: 相機目標點 (OBJ 座標)
            world_theta: 世界坐標偏振片角度 (度)，0°=世界水平，90°=世界垂直
        """
        # 轉換座標
        pos_m = self._transform_point(camera_position)
        tgt_m = self._transform_point(camera_target)

        # 計算相機朝向的方向向量
        direction = np.array(tgt_m) - np.array(pos_m)
        direction = direction / np.linalg.norm(direction)

        # 偏振片位置：在相機前方 10mm 處
        polarizer_offset = 0.010  # 10mm in meters
        polarizer_pos = np.array(pos_m) + direction * polarizer_offset

        # 偏振片尺寸（要足夠大以覆蓋整個視野）
        # 根據 FOV 和距離計算
        fov_rad = np.radians(Config.FOV)
        half_width = polarizer_offset * np.tan(fov_rad / 2) * 1.5  # 1.5x 安全邊距

        # 偏振片變換矩陣
        transform = mi.ScalarTransform4f.look_at(
            origin=polarizer_pos.tolist(),
            target=tgt_m,
            up=[0, 1, 0],
        ) @ mi.ScalarTransform4f.scale([half_width, half_width, 1])

        # 計算世界坐標對齊的 theta
        local_theta = compute_world_aligned_theta(
            polarizer_pos.tolist(), tgt_m, world_theta, up=(0, 1, 0)
        )

        return {
            'type': 'rectangle',
            'to_world': transform,
            'bsdf': {
                'type': 'polarizer',
                'theta': local_theta,  # 世界坐標對齊的角度
            },
        }

    def _create_integrator(self) -> Dict:
        """創建 Stokes integrator"""
        return {
            'type': 'stokes',
            'integrator': {
                'type': 'path',
                'max_depth': Config.MAX_DEPTH,
            },
        }

    def _create_sensor(self,
                       position: Tuple[float, float, float],
                       target: Tuple[float, float, float],
                       spp: int) -> Dict:
        """創建相機"""
        # 座標轉換：OBJ (X,Y,Z) → Mitsuba (X,Z,Y) 且 mm → m
        # 因為 OBJ 匯出時 forward=-Y, up=Z，Mitsuba 需要 -90° X 軸旋轉
        pos_m = self._transform_point(position)
        tgt_m = self._transform_point(target)

        return {
            'type': 'perspective',
            'fov': Config.FOV,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=pos_m,
                target=tgt_m,
                up=[0, 1, 0],  # Mitsuba Y 軸朝上
            ),
            'film': {
                'type': 'hdrfilm',
                'width': Config.WIDTH,
                'height': Config.HEIGHT,
                'pixel_format': 'luminance',  # 灰階格式
                'component_format': 'float32',
                'rfilter': {'type': 'gaussian'},
            },
            'sampler': {
                'type': 'independent',
                'sample_count': spp,
            },
        }

    def _create_led_light(self, camera_pos: Tuple[float, float, float]) -> Tuple[Dict, Dict]:
        """
        創建偏振 LED 光源（新架構）

        使用獨立的非偏振光源 + 偏振片幾何體，確保光線正確經過偏振處理。

        Returns:
            Tuple of (emitter_dict, polarizer_dict)
        """
        # 光源位置：Brewster 角配置
        # LED 在相機右側，以 56° 入射角照射玻璃
        light_pos = (
            Config.LED_POSITION_X,  # X: 右側位置（產生 Brewster 角）
            Config.LED_POSITION_Z,  # Y: 深度 (OBJ 座標)
            Config.LED_POSITION_Y,  # Z: 高度
        )

        # 光源朝向：玻璃中心（而非相機目標）
        target = Config.LED_TARGET

        # 轉換座標
        pos_m = self._transform_point(light_pos)
        tgt_m = self._transform_point(target)

        # 計算變換矩陣
        size_x = mm_to_m(Config.LED_SIZE[0])
        size_y = mm_to_m(Config.LED_SIZE[1])

        # 計算光源朝向的方向向量（用於偏移偏振片）
        direction = np.array(tgt_m) - np.array(pos_m)
        direction = direction / np.linalg.norm(direction)

        # 偏振片位置：在光源前方 5mm 處
        polarizer_offset = 0.005  # 5mm in meters
        polarizer_pos = np.array(pos_m) + direction * polarizer_offset

        # 光源變換矩陣
        emitter_transform = mi.ScalarTransform4f.look_at(
            origin=pos_m,
            target=tgt_m,
            up=[0, 1, 0],
        ) @ mi.ScalarTransform4f.scale([size_x/2, size_y/2, 1])

        # 偏振片變換矩陣（稍微大一點以確保覆蓋所有光線）
        polarizer_transform = mi.ScalarTransform4f.look_at(
            origin=polarizer_pos.tolist(),
            target=tgt_m,
            up=[0, 1, 0],
        ) @ mi.ScalarTransform4f.scale([size_x/2 * 1.1, size_y/2 * 1.1, 1])

        # 計算世界坐標對齊的偏振角度
        # world_theta=90° 表示世界坐標中的垂直偏振
        led_world_theta = 90.0  # 期望的世界坐標偏振角度
        led_local_theta = compute_world_aligned_theta(
            polarizer_pos.tolist(), tgt_m, led_world_theta, up=(0, 1, 0)
        )
        print(f"  [LED 偏振片] 世界角度={led_world_theta}° → 局部角度={led_local_theta:.1f}°")

        # 非偏振光源
        emitter_dict = {
            'type': 'rectangle',
            'to_world': emitter_transform,
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'spectrum',
                    'value': Config.LED_INTENSITY,
                },
            },
        }

        # 獨立偏振片（無發光，只過濾穿過的光）
        # 使用世界坐標對齊的 theta，確保不同場景偏振角度一致
        polarizer_dict = {
            'type': 'rectangle',
            'to_world': polarizer_transform,
            'bsdf': {
                'type': 'polarizer',
                'theta': led_local_theta,  # 世界坐標對齊的角度
            },
        }

        return emitter_dict, polarizer_dict

    def _create_meshes(self) -> List[Tuple[str, Dict]]:
        """
        創建分離的 OBJ 網格（支持紋理）

        使用 OBJSplitter 分離玻璃、天花板和其他幾何，
        分別載入並指定不同的 BSDF/emitter。

        紋理邏輯：
        - 如果材質有 diffuse 紋理 → 使用 MaterialFactory.diffuse_textured()
        - 否則 → 使用原有的 MaterialFactory.diffuse() 純色

        Returns:
            List of (name, mesh_dict) tuples
        """
        meshes = []

        # 基本變換矩陣: mm → m 且 OBJ 座標 → Mitsuba 座標
        transform = mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]) @ \
                    mi.ScalarTransform4f.rotate([1, 0, 0], -90)

        # 選擇非玻璃材質的 BSDF
        # 優先使用帶紋理的材質，fallback 到純色
        other_bsdf = self._select_other_bsdf()

        # 載入非玻璃幾何
        if os.path.exists(self.other_obj_path):
            print(f"[_create_meshes] 載入非玻璃: {self.other_obj_path}")
            meshes.append(('mesh_other', {
                'type': 'obj',
                'filename': self.other_obj_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': other_bsdf,
            }))

        # 選擇天花板材質的 BSDF
        ceiling_bsdf = self._select_ceiling_bsdf()

        # 載入天花板（普通漫反射 + 微弱發光體）
        if self.has_ceiling and os.path.exists(self.ceiling_obj_path):
            print(f"[_create_meshes] 載入天花板: {self.ceiling_obj_path}")
            meshes.append(('mesh_ceiling', {
                'type': 'obj',
                'filename': self.ceiling_obj_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': ceiling_bsdf,
                'emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'spectrum',
                        'value': Config.CEILING_EMITTER_INTENSITY,
                    },
                },
            }))

        # 載入玻璃幾何（使用 dielectric）
        if self.has_glass and os.path.exists(self.glass_obj_path):
            print(f"[_create_meshes] 載入玻璃: {self.glass_obj_path}")
            meshes.append(('mesh_glass', {
                'type': 'obj',
                'filename': self.glass_obj_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': MaterialFactory.glass(),
            }))

        return meshes

    def _select_other_bsdf(self) -> Dict:
        """
        選擇非玻璃、非天花板材質的 BSDF

        優先使用帶紋理的材質，fallback 到純色。

        注意: 當前實現對所有 'other' 面使用同一材質。
        未來可擴展為每個 face group 使用各自材質。

        Returns:
            Mitsuba BSDF dict
        """
        # 找到第一個帶紋理的非玻璃、非天花板材質
        for mat_name, mat_info in self.materials.items():
            # 跳過玻璃
            if mat_info.get('is_glass', False):
                continue
            # 跳過天花板
            if OBJSplitter._is_ceiling(mat_name):
                continue
            # 檢查是否有紋理
            tex_path = mat_info.get('textures', {}).get('diffuse')
            if tex_path and os.path.exists(tex_path):
                print(f"  [BSDF] 使用紋理材質: {mat_name} -> {tex_path}")
                return MaterialFactory.diffuse_textured(tex_path)

        # Fallback: 使用預設灰色
        print(f"  [BSDF] 使用預設灰色材質")
        return MaterialFactory.diffuse(0.5)

    def _select_ceiling_bsdf(self) -> Dict:
        """
        選擇天花板材質的 BSDF

        優先使用帶紋理的材質，fallback 到純色。

        Returns:
            Mitsuba BSDF dict
        """
        # 找到天花板材質
        for mat_name, mat_info in self.materials.items():
            if not OBJSplitter._is_ceiling(mat_name):
                continue
            # 檢查是否有紋理
            tex_path = mat_info.get('textures', {}).get('diffuse')
            if tex_path and os.path.exists(tex_path):
                print(f"  [BSDF] 天花板使用紋理: {mat_name} -> {tex_path}")
                return MaterialFactory.diffuse_textured(tex_path)

        # Fallback: 高反射率白色
        print(f"  [BSDF] 天花板使用預設白色")
        return MaterialFactory.diffuse(0.85)

    def _transform_point(self, point: Tuple[float, float, float]) -> List[float]:
        """
        座標轉換：OBJ (mm) → Mitsuba (m)

        OBJ 座標: X(右), Y(深度/前), Z(上)
        經過 -90° X 軸旋轉後：
        Mitsuba: X(右), Y(上), Z(深度)
        """
        x, y, z = point
        # OBJ (X, Y, Z) → Mitsuba (X, Z, Y)，並 mm → m
        return [mm_to_m(x), mm_to_m(z), mm_to_m(y)]


# ============================================================
# Stokes 處理器
# ============================================================

class StokesProcessor:
    """Stokes Vector 處理器"""

    @staticmethod
    def extract_stokes(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        從渲染結果提取 Stokes 參數

        Returns:
            (S0, S1, S2) - 各為 2D array
        """
        if image.ndim != 3:
            raise ValueError(f"預期 3D 圖像，得到 {image.ndim}D")

        h, w = image.shape[:2]
        n_channels = image.shape[2]
        print(f"  [Stokes] 圖像形狀: {image.shape}, 通道數: {n_channels}")

        if n_channels >= 12:
            # Spectral polarized: 假設 n_spectral 波長 × 3 Stokes
            if n_channels == 15:
                n_spectral = 5
            elif n_channels == 16:
                n_spectral = 4
            elif n_channels == 13:
                n_spectral = 4
            else:
                n_spectral = n_channels // 3

            print(f"  [Stokes] 推測 {n_spectral} 波長")
            S0 = np.mean(image[:, :, 0:n_spectral], axis=2)
            S1 = np.mean(image[:, :, n_spectral:2*n_spectral], axis=2)
            S2_end = min(3*n_spectral, n_channels)
            S2 = np.mean(image[:, :, 2*n_spectral:S2_end], axis=2)

        elif n_channels == 4:
            S0 = image[:, :, 0]
            S1 = image[:, :, 1]
            S2 = image[:, :, 2]
        elif n_channels == 3:
            print("  [警告] RGB 輸出，無偏振信息")
            S0 = 0.2126 * image[:,:,0] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,2]
            S1 = np.zeros_like(S0)
            S2 = np.zeros_like(S0)
        else:
            raise ValueError(f"不支援的通道數: {n_channels}")

        # 輸出統計
        print(f"  [Stokes] S0 範圍: [{S0.min():.4f}, {S0.max():.4f}]")
        print(f"  [Stokes] S1 範圍: [{S1.min():.4f}, {S1.max():.4f}]")
        print(f"  [Stokes] S2 範圍: [{S2.min():.4f}, {S2.max():.4f}]")

        return S0.astype(np.float32), S1.astype(np.float32), S2.astype(np.float32)

    @staticmethod
    def compute_polarization_images(S0: np.ndarray,
                                     S1: np.ndarray,
                                     S2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        計算偏振片透射強度

        根據 Malus 定律：
        I(θ) = 0.5 * (S0 + S1*cos(2θ) + S2*sin(2θ))

        I_parallel (θ=0°) = 0.5 * (S0 + S1)
        I_cross (θ=90°) = 0.5 * (S0 - S1)
        """
        I_parallel = 0.5 * (S0 + S1)
        I_cross = 0.5 * (S0 - S1)

        # 確保非負
        I_parallel = np.maximum(I_parallel, 0)
        I_cross = np.maximum(I_cross, 0)

        # 輸出統計
        diff = np.abs(I_parallel - I_cross)
        print(f"  [偏振] I∥ 範圍: [{I_parallel.min():.4f}, {I_parallel.max():.4f}]")
        print(f"  [偏振] I⊥ 範圍: [{I_cross.min():.4f}, {I_cross.max():.4f}]")
        print(f"  [偏振] |I∥-I⊥| 平均: {diff.mean():.4f}, 最大: {diff.max():.4f}")

        return I_parallel.astype(np.float32), I_cross.astype(np.float32)

    @staticmethod
    def compute_dolp(S0: np.ndarray, S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
        """計算偏振度 DoLP"""
        dolp = np.sqrt(S1**2 + S2**2) / (S0 + 1e-10)
        dolp = np.clip(dolp, 0, 1)
        return dolp.astype(np.float32)

    @staticmethod
    def compute_dolp_from_intensities(I_parallel: np.ndarray, I_cross: np.ndarray) -> np.ndarray:
        """從 I∥ 和 I⊥ 計算偏振度 DoLP"""
        I_sum = I_parallel + I_cross
        dolp = np.zeros_like(I_parallel)
        valid_mask = I_sum > 1e-6
        dolp[valid_mask] = np.abs(I_parallel[valid_mask] - I_cross[valid_mask]) / I_sum[valid_mask]
        dolp = np.clip(dolp, 0, 1)
        return dolp.astype(np.float32)


# ============================================================
# 場景報告生成器
# ============================================================

def warp_right_to_left(right_img: np.ndarray, disparity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用視差將右圖 warp 到左視角

    Args:
        right_img: 右相機圖像
        disparity: 視差圖 (左視角)

    Returns:
        (warped_img, valid_mask): warped 圖像和有效區域 mask
    """
    h, w = right_img.shape[:2]

    # 建立座標網格
    x_coords = np.arange(w, dtype=np.float32)
    y_coords = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # 右圖對應位置 = 左圖位置 - disparity
    # 因為：左相機在左邊 (X小)，右相機在右邊 (X大)
    # 同一 3D 點：在左圖 x 較大，在右圖 x 較小
    # 所以 disparity = x_left - x_right > 0
    # 要從右圖找來源：x_right = x_left - disparity
    xx_src = (xx - disparity).astype(np.float32)
    yy_src = yy.astype(np.float32)

    # 使用 OpenCV remap 進行 warp
    warped = cv2.remap(right_img.astype(np.float32), xx_src, yy_src,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 有效區域：視差有效且 warp 後在圖像範圍內
    valid_mask = (disparity > 0) & (xx_src >= 0) & (xx_src < w)

    return warped, valid_mask


def generate_scene_report(scene_name: str,
                          I_parallel: np.ndarray,
                          I_cross: np.ndarray,
                          depth: np.ndarray,
                          disparity: np.ndarray,
                          glass_mask: np.ndarray,
                          glass_mask_left: np.ndarray,
                          dolp: np.ndarray,
                          dolp_stats: dict) -> dict:
    """
    生成場景品質報告

    報告內容：
    1. DoLP 統計（全局、玻璃區域、背景區域）- 使用預先計算的對齊 DoLP
    2. 強度比值 I∥/I⊥ (使用 warp 對齊後的正確比較)
    3. 噪點水平和 SNR
    4. 深度圖統計
    5. 強度平衡檢測
    6. 綜合品質評分
    7. 玻璃區域深度有效率 (Criterion 5)

    Args:
        dolp: 左相機 DoLP 圖像（用於全局統計和保存）
        dolp_stats: 預先計算的對齊 DoLP 統計 {'glass_left', 'glass_right', 'background'}
    """
    from datetime import datetime

    report = {
        'scene_name': scene_name,
        'timestamp': datetime.now().isoformat(),
        'render_config': {
            'width': Config.WIDTH,
            'height': Config.HEIGHT,
            'spp': Config.SPP,
            'max_depth': Config.MAX_DEPTH,
        },
        'warnings': [],
    }

    # ============================================================
    # 1. 數值範圍
    # ============================================================
    report['value_range'] = {
        'I_parallel': {
            'min': float(np.min(I_parallel)),
            'max': float(np.max(I_parallel)),
            'mean': float(np.mean(I_parallel)),
        },
        'I_cross': {
            'min': float(np.min(I_cross)),
            'max': float(np.max(I_cross)),
            'mean': float(np.mean(I_cross)),
        },
    }

    # ============================================================
    # 2. 偏振分析（分區域）- 使用預先計算的對齊 DoLP
    # ============================================================
    valid_mask = dolp > 0

    # Warp I_cross 到左視角（確保比較同一 3D 點）
    I_cross_warped, warp_valid = warp_right_to_left(I_cross, disparity)

    # 偏振差異（用於 SNR 計算）- 使用 warp 後的對齊圖像
    diff = np.abs(I_parallel - I_cross_warped)

    # 強度比值（使用 warp 後的 I_cross，比較同一 3D 點）
    ratio_mask = (I_cross_warped > 0.01) & warp_valid
    intensity_ratio = np.zeros_like(I_parallel)
    if np.any(ratio_mask):
        intensity_ratio[ratio_mask] = I_parallel[ratio_mask] / I_cross_warped[ratio_mask]

    # 玻璃區域統計（使用預先計算的對齊 DoLP）
    glass_pixel_count = int(np.sum(glass_mask > 0.5)) if glass_mask is not None else 0

    # 計算玻璃區域的 warp 對齊強度比值
    # 注意：使用 glass_mask_left（左視角 mask），因為 I_cross_warped 已 warp 到左視角
    glass_ratio_mask = (glass_mask_left > 0.5) & ratio_mask if glass_mask_left is not None else ratio_mask
    if np.any(glass_ratio_mask):
        glass_intensity_ratio_mean = float(np.mean(intensity_ratio[glass_ratio_mask]))
        glass_intensity_ratio_max = float(np.max(intensity_ratio[glass_ratio_mask]))
    else:
        glass_intensity_ratio_mean = 1.0
        glass_intensity_ratio_max = 1.0

    glass_stats = {
        'pixel_count': glass_pixel_count,
        'pixel_ratio': float(glass_pixel_count / (Config.WIDTH * Config.HEIGHT)),
        'dolp_left': dolp_stats['glass_left'],    # 左眼玻璃 DoLP（對齊）
        'dolp_right': dolp_stats['glass_right'],  # 右眼玻璃 DoLP（對齊）
        'dolp_mean': dolp_stats['glass_left'],    # 只用左眼（更準確）
        'intensity_ratio_mean': glass_intensity_ratio_mean,  # 玻璃區域 I∥/I⊥（warp 對齊）
        'intensity_ratio_max': glass_intensity_ratio_max,
        'stokes_ratio': dolp_stats.get('true_glass_ratio', 1.0),  # I(90°)/I(0°) 從 Stokes 計算
    }

    # 背景區域統計
    background_mask = (glass_mask < 0.5) if glass_mask is not None else np.ones_like(dolp, dtype=bool)
    bg_pixel_count = int(np.sum(background_mask))
    bg_stats = {
        'pixel_count': bg_pixel_count,
        'pixel_ratio': float(bg_pixel_count / (Config.WIDTH * Config.HEIGHT)),
        'dolp_mean': dolp_stats['background'],
        'stokes_ratio': dolp_stats.get('true_bg_ratio', 1.0),  # I(90°)/I(0°) 從 Stokes 計算
    }

    # 玻璃/背景 DoLP 比值（使用 floor 避免除以零）
    dolp_ratio = glass_stats['dolp_mean'] / max(bg_stats['dolp_mean'], Config.DOLP_FLOOR)

    # 為了兼容舊代碼，創建分區域 mask（基於 DoLP 閾值）
    high_dolp_mask = (dolp > 0.1) & valid_mask
    low_dolp_mask = (dolp <= 0.1) & valid_mask
    if not np.any(low_dolp_mask):
        bg_stats = {
            'pixel_count': 0,
            'pixel_ratio': 0.0,
            'dolp_mean': 0.0,
            'dolp_median': 0.0,
        }

    # 全局統計
    dolp_valid = dolp[valid_mask]

    report['polarization'] = {
        'global': {
            'dolp_mean': float(np.mean(dolp_valid)) if len(dolp_valid) > 0 else 0,
            'dolp_std': float(np.std(dolp_valid)) if len(dolp_valid) > 0 else 0,
            'dolp_p95': float(np.percentile(dolp_valid, 95)) if len(dolp_valid) > 0 else 0,
            'dolp_max': float(np.max(dolp_valid)) if len(dolp_valid) > 0 else 0,
        },
        'glass_region': glass_stats,
        'background_region': bg_stats,
        'intensity_ratio': {
            'mean': float(np.mean(intensity_ratio[ratio_mask])) if np.any(ratio_mask) else 0,
            'median': float(np.median(intensity_ratio[ratio_mask])) if np.any(ratio_mask) else 0,
            'max': float(np.max(intensity_ratio[ratio_mask])) if np.any(ratio_mask) else 0,
        },
        'diff': {
            'mean': float(np.mean(diff)),
            'max': float(np.max(diff)),
        },
        'dolp_ratio': float(dolp_ratio),  # 玻璃/背景 DoLP 比值
    }

    # ============================================================
    # 3. 噪點分析（基於 DoLP，避免亮度差異影響）
    # ============================================================
    def estimate_noise(image):
        """使用 MAD 估計圖像噪點"""
        diff_h = np.abs(image[:, 1:] - image[:, :-1])
        diff_v = np.abs(image[1:, :] - image[:-1, :])
        mad = (np.median(diff_h) + np.median(diff_v)) / 2
        return float(mad / (np.sqrt(2) * 0.6745))

    noise_parallel = estimate_noise(I_parallel)
    noise_cross = estimate_noise(I_cross)
    noise_std = (noise_parallel + noise_cross) / 2

    # DoLP-based SNR（使用歸一化的偏振度，避免亮度影響）
    # Signal: 玻璃區域平均 DoLP
    # Noise: 背景區域 DoLP 標準差（理想背景 DoLP≈0，變異為噪點）
    glass_dolp_mean = (dolp_stats['glass_left'] + dolp_stats['glass_right']) / 2

    # 計算背景 DoLP 的變異作為噪點估計
    if glass_mask is not None:
        bg_mask = glass_mask < 0.5
        if np.any(bg_mask):
            bg_dolp_std = float(np.std(dolp[bg_mask]))
        else:
            bg_dolp_std = 0.01
    else:
        bg_dolp_std = float(np.std(dolp))

    # SNR = 玻璃 DoLP / 背景 DoLP 噪點（使用 floor 避免除以零）
    snr_polarization = glass_dolp_mean / max(bg_dolp_std, Config.DOLP_FLOOR)

    report['noise'] = {
        'noise_std': noise_std,
        'noise_parallel': noise_parallel,
        'noise_cross': noise_cross,
        'bg_dolp_std': bg_dolp_std,
        'snr_polarization': snr_polarization,
    }

    # ============================================================
    # 4. 深度圖分析
    # ============================================================
    if depth is not None:
        depth_valid = depth[depth > 0]
        report['depth'] = {
            'valid_ratio': float(np.sum(depth > 0) / depth.size),
            'min': float(np.min(depth_valid)) if len(depth_valid) > 0 else 0,
            'max': float(np.max(depth_valid)) if len(depth_valid) > 0 else 0,
            'mean': float(np.mean(depth_valid)) if len(depth_valid) > 0 else 0,
        }

    # ============================================================
    # 4.5 玻璃區域深度有效率 (PIDS Criterion 5)
    # ============================================================
    if glass_mask is not None and depth is not None:
        glass_pixels = glass_mask > 0.5
        glass_pixel_count = int(np.sum(glass_pixels))

        if glass_pixel_count > 0:
            glass_depth = depth[glass_pixels]
            valid_depth_in_glass = glass_depth > 0
            valid_count = int(np.sum(valid_depth_in_glass))
            validity_rate = valid_count / glass_pixel_count

            report['glass_depth_validity'] = {
                'glass_pixel_count': glass_pixel_count,
                'valid_depth_count': valid_count,
                'validity_rate': float(validity_rate),
                'pass': validity_rate >= 0.9,  # Criterion 5: > 90%
            }
        else:
            report['glass_depth_validity'] = {
                'glass_pixel_count': 0,
                'valid_depth_count': 0,
                'validity_rate': 0.0,
                'pass': False,
            }

    # ============================================================
    # 5. 強度平衡檢測（背景區域的 I∥/I⊥ 比值，使用 warp 對齊）
    # ============================================================
    # 使用 warp 後的 I_cross，確保比較同一 3D 點
    bg_balance_mask = low_dolp_mask & warp_valid & (I_cross_warped > 0.01)
    if np.any(bg_balance_mask):
        bg_intensity_ratio = I_parallel[bg_balance_mask] / I_cross_warped[bg_balance_mask]
        bg_ratio_mean = float(np.mean(bg_intensity_ratio))
        bg_ratio_std = float(np.std(bg_intensity_ratio))
        bg_mean_parallel = float(np.mean(I_parallel[bg_balance_mask]))
        bg_mean_cross = float(np.mean(I_cross_warped[bg_balance_mask]))
    else:
        bg_ratio_mean = 1.0
        bg_ratio_std = 0.0
        bg_mean_parallel = 0.0
        bg_mean_cross = 0.0

    report['intensity_balance'] = {
        'background_ratio_mean': bg_ratio_mean,
        'background_ratio_std': bg_ratio_std,
        'background_mean_parallel': bg_mean_parallel,
        'background_mean_cross': bg_mean_cross,
        'is_balanced': 0.8 <= bg_ratio_mean <= 1.25,  # 嚴格閾值
    }

    # ============================================================
    # 6. 綜合品質評估
    # ============================================================
    glass_ratio = glass_stats['pixel_ratio']
    glass_dolp_mean = glass_stats['dolp_mean']
    bg_dolp_mean = bg_stats['dolp_mean']

    scores = []

    # 偏振分數
    if glass_ratio > 0.05 and glass_dolp_mean > 0.3:
        pol_score = 100
    elif glass_ratio > 0.01 and glass_dolp_mean > 0.1:
        pol_score = 70
    elif report['polarization']['global']['dolp_p95'] > 0.1:
        pol_score = 50
    else:
        pol_score = 30
    scores.append(pol_score)

    # 噪點分數
    if snr_polarization >= 10:
        noise_score = 100
    elif snr_polarization >= 2:
        noise_score = 80
    elif snr_polarization >= 1:
        noise_score = 60
    else:
        noise_score = 40
    scores.append(noise_score)

    # 區域對比分數
    if glass_dolp_mean > bg_dolp_mean * 3:
        contrast_score = 100
    elif glass_dolp_mean > bg_dolp_mean * 2:
        contrast_score = 80
    elif glass_dolp_mean > bg_dolp_mean:
        contrast_score = 60
    else:
        contrast_score = 40
    scores.append(contrast_score)

    # 強度平衡分數（配合嚴格閾值 0.8~1.25）
    if report['intensity_balance']['is_balanced']:
        balance_score = 100
    elif 0.6 <= bg_ratio_mean <= 1.5:
        balance_score = 70
    elif 0.5 <= bg_ratio_mean <= 2.0:
        balance_score = 40
    else:
        balance_score = 20
    scores.append(balance_score)

    overall_score = int(np.mean(scores))

    if overall_score >= 80:
        level = 'excellent'
    elif overall_score >= 60:
        level = 'good'
    elif overall_score >= 40:
        level = 'acceptable'
    else:
        level = 'poor'

    report['quality'] = {
        'score': overall_score,
        'level': level,
        'polarization_score': pol_score,
        'noise_score': noise_score,
        'contrast_score': contrast_score,
        'balance_score': balance_score,
        'valid': overall_score >= 40,
    }

    # ============================================================
    # 7. 警告
    # ============================================================
    if glass_ratio > 0.8:
        report['warnings'].append('高偏振區域佔比過大（>80%），檢查光源/場景設置')
    if glass_ratio < 0.01:
        report['warnings'].append('高偏振區域過小（<1%），玻璃物體可能太小或偏振效果弱')
    if bg_dolp_mean > 0.2:
        report['warnings'].append('背景 DoLP 偏高，檢查漫反射材質設置')
    if snr_polarization < 1:
        report['warnings'].append(f'SNR 較低（{snr_polarization:.2f}），建議增加 SPP')
    glass_intensity_ratio = report['polarization']['glass_region'].get('intensity_ratio_mean', 1.0)
    if glass_intensity_ratio < 1.5:
        report['warnings'].append(f'I∥/I⊥ 玻璃區域比值較低 ({glass_intensity_ratio:.2f}x)，偏振效果可能不明顯')
    if 'glass_depth_validity' in report and not report['glass_depth_validity']['pass']:
        validity_rate = report['glass_depth_validity']['validity_rate'] * 100
        report['warnings'].append(f'玻璃區域深度有效率不足（{validity_rate:.1f}% < 90%），違反 Criterion 5')
    if not report['intensity_balance']['is_balanced']:
        report['warnings'].append(
            f'背景區域強度不平衡（I∥/I⊥={bg_ratio_mean:.2f}），'
            f'RAFT雙目對齊可能失敗'
        )

    return report


# ============================================================
# 渲染器
# ============================================================

class PIDSRenderer:
    """PIDS 偏振立體渲染器（支持紋理）"""

    def __init__(self):
        self.variant = setup_mitsuba()

    def render_scene(self, obj_path: str, output_dir: str, scene_name: str = None):
        """
        渲染單一場景

        輸出:
            - {scene}_left_parallel.exr  (左相機 I∥ 灰階)
            - {scene}_right_cross.exr    (右相機 I⊥ 灰階)
            - {scene}_depth.exr          (深度圖)
            - {scene}_disparity.exr      (視差圖)
            - {scene}_glass_mask.exr     (玻璃區域 mask)
            - {scene}_params.json        (參數)
            - {scene}_report.json        (品質報告)
        """
        if scene_name is None:
            scene_name = Path(obj_path).stem

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"渲染場景: {scene_name}")
        print(f"{'='*60}")

        # 相機位置和目標（使用平行光軸配置）
        left_pos = Config.left_camera_position()
        right_pos = Config.right_camera_position()
        depth_pos = Config.depth_camera_position()

        # 各相機的個別目標點，確保平行光軸
        left_target = Config.camera_target_for_position(left_pos)
        right_target = Config.camera_target_for_position(right_pos)
        depth_target = Config.camera_target_for_position(depth_pos)

        print(f"[配置] 左相機位置: {left_pos}")
        print(f"[配置] 左相機目標: {left_target}")
        print(f"[配置] 右相機位置: {right_pos}")
        print(f"[配置] 右相機目標: {right_target}")
        print(f"[配置] 深度相機位置: {depth_pos}")
        print(f"[配置] 深度相機目標: {depth_target}")
        print(f"[配置] 視線方向: {Config.forward_direction()} (平行光軸)")

        # 建構場景
        builder = SceneBuilder(obj_path)

        # 渲染左相機 (I∥) - 使用 90° 偏振片（交換角度修正比值方向）
        print(f"\n[1/6] 渲染左相機 (I∥, θ=90°)...")
        left_image = self._render_camera(builder, left_pos, left_target, polarizer_angle=90.0)

        # 處理 Stokes → I∥
        S0_left, S1_left, S2_left = StokesProcessor.extract_stokes(left_image)
        I_parallel, _ = StokesProcessor.compute_polarization_images(S0_left, S1_left, S2_left)

        del left_image
        gc.collect()

        # 渲染左相機 (I⊥) - 同視角 0° 偏振片（用於計算真實 I∥/I⊥ 比值）
        print(f"\n[2/6] 渲染左相機 (I⊥, θ=0°) - 同視角偏振測量...")
        left_cross_image = self._render_camera(builder, left_pos, left_target, polarizer_angle=0.0)

        # 處理 Stokes → I⊥_left（同視角）
        S0_left_cross, S1_left_cross, S2_left_cross = StokesProcessor.extract_stokes(left_cross_image)
        _, I_cross_left = StokesProcessor.compute_polarization_images(S0_left_cross, S1_left_cross, S2_left_cross)

        del left_cross_image
        gc.collect()

        # 渲染右相機 (I⊥) - 使用 0° 偏振片（用於 stereo pair）
        print(f"\n[3/6] 渲染右相機 (I⊥, θ=0°)...")
        right_image = self._render_camera(builder, right_pos, right_target, polarizer_angle=0.0)

        # 處理 Stokes → I⊥
        S0_right, S1_right, S2_right = StokesProcessor.extract_stokes(right_image)
        _, I_cross = StokesProcessor.compute_polarization_images(S0_right, S1_right, S2_right)

        del right_image
        gc.collect()

        # 渲染深度圖（使用獨立深度相機，在立體相機中央下方）
        print(f"\n[4/6] 渲染深度圖 (深度相機)...")
        depth = self._render_depth(builder, depth_pos, depth_target)

        # 渲染玻璃 mask（從左右相機視角取聯集，確保完整覆蓋 stereo pair 中的玻璃像素）
        # 同時計算對齊的 DoLP 統計（各相機用自己的 mask，更精確）
        print(f"\n[5/6] 渲染玻璃 mask + DoLP 統計...")

        # 左相機 glass mask + DoLP
        glass_mask_left = self._render_glass_mask(builder, left_pos, left_target, "左")
        dolp_left = StokesProcessor.compute_dolp(S0_left, S1_left, S2_left)
        glass_dolp_left = float(dolp_left[glass_mask_left > 0.5].mean()) if np.any(glass_mask_left > 0.5) else 0.0
        print(f"  [DoLP 左眼玻璃] {glass_dolp_left:.4f}")

        # 右相機 glass mask + DoLP
        glass_mask_right = self._render_glass_mask(builder, right_pos, right_target, "右")
        dolp_right = StokesProcessor.compute_dolp(S0_right, S1_right, S2_right)
        glass_dolp_right = float(dolp_right[glass_mask_right > 0.5].mean()) if np.any(glass_mask_right > 0.5) else 0.0
        print(f"  [DoLP 右眼玻璃] {glass_dolp_right:.4f}")

        # 聯集 mask (用於訓練 - 確保覆蓋所有玻璃)
        glass_mask = ((glass_mask_left > 0.5) | (glass_mask_right > 0.5)).astype(np.float32)
        print(f"  [Glass Mask (聯集)] 玻璃像素: {int(np.sum(glass_mask))} ({np.sum(glass_mask)/(Config.WIDTH*Config.HEIGHT)*100:.1f}%)")

        # 交集 mask (用於評估 - 更嚴格，只保留確定是玻璃的像素)
        glass_mask_strict = ((glass_mask_left > 0.5) & (glass_mask_right > 0.5)).astype(np.float32)
        print(f"  [Glass Mask (交集)] 玻璃像素: {int(np.sum(glass_mask_strict))} ({np.sum(glass_mask_strict)/(Config.WIDTH*Config.HEIGHT)*100:.1f}%)")

        # 背景 DoLP（用左眼，排除玻璃區域）
        background_mask = glass_mask < 0.5
        background_dolp = float(dolp_left[background_mask].mean()) if np.any(background_mask) else 0.0
        print(f"  [DoLP 背景] {background_dolp:.4f}")
        print(f"  [DoLP 玻璃/背景比] {(glass_dolp_left + glass_dolp_right) / 2 / max(background_dolp, Config.DOLP_FLOOR):.2f}x")

        # 計算同視角偏振比值（用左相機的 Stokes 參數，無需第二次渲染）
        # 根據 Malus 定律：I(θ) = 0.5 * (S0 + S1*cos(2θ) + S2*sin(2θ))
        # I(0°) = 0.5 * (S0 + S1), I(90°) = 0.5 * (S0 - S1)
        I_0deg = 0.5 * (S0_left + S1_left)  # 0° 偏振片強度
        I_90deg = 0.5 * (S0_left - S1_left)  # 90° 偏振片強度
        I_0deg = np.maximum(I_0deg, 0)
        I_90deg = np.maximum(I_90deg, 0)

        # 計算同視角偏振比值

        valid_pol_mask = (I_0deg > 0.01) & (I_90deg > 0.01)
        true_pol_ratio = np.ones_like(I_0deg)
        if np.any(valid_pol_mask):
            true_pol_ratio[valid_pol_mask] = I_90deg[valid_pol_mask] / I_0deg[valid_pol_mask]

        # 玻璃區域真實偏振比值
        glass_pol_mask = (glass_mask_left > 0.5) & valid_pol_mask
        if np.any(glass_pol_mask):
            true_glass_ratio = float(np.mean(true_pol_ratio[glass_pol_mask]))
        else:
            true_glass_ratio = 1.0

        # 背景區域真實偏振比值
        bg_pol_mask = (glass_mask_left < 0.5) & valid_pol_mask
        if np.any(bg_pol_mask):
            true_bg_ratio = float(np.mean(true_pol_ratio[bg_pol_mask]))
        else:
            true_bg_ratio = 1.0


        # 打包 DoLP 統計供報告使用
        dolp_stats = {
            'glass_left': glass_dolp_left,
            'glass_right': glass_dolp_right,
            'background': background_dolp,
            'true_glass_ratio': true_glass_ratio,
            'true_bg_ratio': true_bg_ratio,
        }

        # 計算視差
        print(f"\n[6/6] 計算視差...")
        disparity = self._compute_disparity(depth)

        # 保存結果
        print(f"\n[保存] 輸出檔案...")
        self._save_outputs(
            output_dir, scene_name,
            I_parallel, I_cross, depth, disparity,
            dolp_left, glass_mask, glass_mask_left, dolp_stats,
            glass_mask_strict
        )

        # 清理臨時檔案
        builder.cleanup()

        print(f"\n{'='*60}")
        print(f"場景 {scene_name} 渲染完成!")
        print(f"{'='*60}\n")

    def _render_camera(self,
                       builder: SceneBuilder,
                       position: Tuple[float, float, float],
                       target: Tuple[float, float, float],
                       polarizer_angle: Optional[float] = None) -> np.ndarray:
        """
        渲染單一相機視角 (分批渲染)

        Args:
            builder: 場景建構器
            position: 相機位置
            target: 相機目標點
            polarizer_angle: 相機前偏振片角度 (度)，None 表示不加偏振片
        """
        scene_dict = builder.build(position, target, Config.SPP_PER_BATCH, polarizer_angle)
        scene = mi.load_dict(scene_dict)

        total_spp = Config.SPP
        batch_spp = Config.SPP_PER_BATCH
        n_batches = max(1, total_spp // batch_spp)

        print(f"  [渲染] SPP: {total_spp}, 批次: {n_batches} x {batch_spp}")

        accumulated = None

        for i in range(n_batches):
            image = mi.render(scene, spp=batch_spp)
            img_np = np.array(image)

            if accumulated is None:
                accumulated = img_np.astype(np.float64)
            else:
                accumulated += img_np.astype(np.float64)

            print(f"    批次 {i+1}/{n_batches} 完成")

            del image
            gc.collect()

        result = (accumulated / n_batches).astype(np.float32)
        print(f"  [渲染] 完成，形狀: {result.shape}")

        return result

    def _render_depth(self,
                      builder: SceneBuilder,
                      position: Tuple[float, float, float],
                      target: Tuple[float, float, float]) -> np.ndarray:
        """渲染深度圖"""
        scene_dict = builder.build(position, target, 256)

        # 修改 integrator 為 depth AOV
        scene_dict['integrator'] = {
            'type': 'aov',
            'aovs': 'dd.y:depth',
            'integrator': {
                'type': 'path',
                'max_depth': 2,
            },
        }

        scene = mi.load_dict(scene_dict)
        image = mi.render(scene, spp=256)
        img_np = np.array(image)

        # 提取深度通道
        if img_np.ndim == 3 and img_np.shape[2] >= 2:
            depth = img_np[:, :, 1]
        else:
            depth = img_np[:, :, 0] if img_np.ndim == 3 else img_np

        print(f"  [深度] 範圍: [{depth.min():.4f}, {depth.max():.4f}] m")

        return depth.astype(np.float32)

    def _compute_disparity(self, ray_depth: np.ndarray) -> np.ndarray:
        """
        計算視差圖

        重要：Mitsuba 的 depth AOV 輸出的是 ray depth (歐幾里得距離)，
        但視差公式需要 Z depth (垂直距離)。必須先轉換：
        Z = ray_depth * cos(angle)
        """
        # 焦距 (pixels)
        fov_rad = np.radians(Config.FOV)
        focal_px = (Config.WIDTH / 2) / np.tan(fov_rad / 2)

        # 基線 (m)
        baseline_m = mm_to_m(Config.BASELINE)

        # ============ Ray depth → Z depth 轉換 ============
        # 計算每個像素的 cos(angle)
        # cos(angle) = focal / sqrt(focal^2 + dx^2 + dy^2)
        # 其中 dx = x - cx, dy = y - cy
        cx, cy = Config.WIDTH / 2, Config.HEIGHT / 2
        y_coords, x_coords = np.meshgrid(
            np.arange(Config.HEIGHT),
            np.arange(Config.WIDTH),
            indexing='ij'
        )
        dx = x_coords - cx
        dy = y_coords - cy

        # cos(angle) for each pixel
        cos_angle = focal_px / np.sqrt(focal_px**2 + dx**2 + dy**2)

        # 轉換為 Z depth (垂直距離)
        z_depth = ray_depth * cos_angle

        # 計算中央和邊角的 cos 差異 (debug info)
        cos_center = cos_angle[Config.HEIGHT//2, Config.WIDTH//2]
        cos_corner = cos_angle[0, 0]
        print(f"  [深度轉換] cos(center): {cos_center:.4f}, cos(corner): {cos_corner:.4f}")
        print(f"  [深度轉換] ray_depth 範圍: [{ray_depth[ray_depth>0].min():.4f}, {ray_depth[ray_depth>0].max():.4f}] m")
        print(f"  [深度轉換] z_depth 範圍: [{z_depth[z_depth>0].min():.4f}, {z_depth[z_depth>0].max():.4f}] m")

        # ============ 計算視差 ============
        # 視差 = baseline * focal / Z
        disparity = np.zeros_like(z_depth)
        valid = z_depth > 0
        disparity[valid] = (baseline_m * focal_px) / z_depth[valid]

        print(f"  [視差] 焦距: {focal_px:.1f} px")
        print(f"  [視差] 範圍: [{disparity[valid].min():.1f}, {disparity[valid].max():.1f}] px")

        return disparity.astype(np.float32)

    def _render_glass_mask(self,
                           builder: SceneBuilder,
                           position: Tuple[float, float, float],
                           target: Tuple[float, float, float],
                           camera_name: str = "") -> np.ndarray:
        """
        渲染玻璃區域 mask

        只渲染玻璃網格，使用純白材質，得到二值化 mask。
        用於計算 PIDS Criterion 5：玻璃區域深度有效率。

        Args:
            camera_name: 相機名稱，用於輸出顯示

        Returns:
            glass_mask: 2D array, 玻璃區域為 1，其他為 0
        """
        if not builder.has_glass:
            print(f"  [Glass Mask] 場景無玻璃，返回空 mask")
            return np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.float32)

        # 座標轉換
        pos_m = builder._transform_point(position)
        tgt_m = builder._transform_point(target)

        # 基本變換矩陣
        transform = mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]) @ \
                    mi.ScalarTransform4f.rotate([1, 0, 0], -90)

        # 只渲染玻璃 OBJ 的深度，有深度值 = 玻璃區域
        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'aov',
                'aovs': 'dd.y:depth',
                'integrator': {
                    'type': 'path',
                    'max_depth': 2,
                },
            },
            'sensor': {
                'type': 'perspective',
                'fov': Config.FOV,
                'fov_axis': 'x',
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=pos_m,
                    target=tgt_m,
                    up=[0, 1, 0],
                ),
                'film': {
                    'type': 'hdrfilm',
                    'width': Config.WIDTH,
                    'height': Config.HEIGHT,
                    'pixel_format': 'luminance',
                    'component_format': 'float32',
                },
                'sampler': {
                    'type': 'independent',
                    'sample_count': 4,  # 深度只需極少 SPP
                },
            },
            # 只載入玻璃 OBJ
            'glass_mesh': {
                'type': 'obj',
                'filename': builder.glass_obj_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': MaterialFactory.diffuse(0.5),
            },
        }

        scene = mi.load_dict(scene_dict)
        image = mi.render(scene, spp=4)
        img_np = np.array(image)

        # 提取深度通道 (AOV 輸出格式: [radiance, depth])
        if img_np.ndim == 3 and img_np.shape[2] >= 2:
            depth_channel = img_np[:, :, 1]
        elif img_np.ndim == 3:
            depth_channel = img_np[:, :, 0]
        else:
            depth_channel = img_np

        # 二值化（有深度值的地方就是玻璃）
        glass_mask = (depth_channel > 0).astype(np.float32)

        pixel_count = int(np.sum(glass_mask))
        pixel_ratio = pixel_count / (Config.WIDTH * Config.HEIGHT)
        cam_label = f" ({camera_name})" if camera_name else ""
        print(f"  [Glass Mask{cam_label}] 玻璃像素: {pixel_count} ({pixel_ratio*100:.1f}%)")

        return glass_mask

    def _save_outputs(self,
                      output_dir: str,
                      scene_name: str,
                      I_parallel: np.ndarray,
                      I_cross: np.ndarray,
                      depth: np.ndarray,
                      disparity: np.ndarray,
                      dolp: np.ndarray,
                      glass_mask: np.ndarray,
                      glass_mask_left: np.ndarray,
                      dolp_stats: dict,
                      glass_mask_strict: np.ndarray = None):
        """保存所有輸出（不含 right_parallel）"""
        # EXR 檔案
        self._save_exr(I_parallel, f"{output_dir}/{scene_name}_left_parallel.exr")
        self._save_exr(I_cross, f"{output_dir}/{scene_name}_right_cross.exr")
        self._save_exr(depth, f"{output_dir}/{scene_name}_depth.exr")
        self._save_exr(disparity, f"{output_dir}/{scene_name}_disparity.exr")

        # 預覽 PNG
        if Config.SAVE_PREVIEW:
            # 使用 percentile 統一範圍，避免極端值影響
            combined = np.concatenate([I_parallel.flatten(), I_cross.flatten()])
            vmax = np.percentile(combined, 99.5)  # 99.5 percentile 避免極端亮點
            vmin = 0
            self._save_png_fixed(I_parallel, f"{output_dir}/{scene_name}_left_parallel.png", vmin, vmax)
            self._save_png_fixed(I_cross, f"{output_dir}/{scene_name}_right_cross.png", vmin, vmax)

            # 偏振差異圖
            diff = np.abs(I_parallel - I_cross)
            self._save_png(diff, f"{output_dir}/{scene_name}_polarization_diff.png")

            # 深度圖 (colormap)
            self._save_depth_png(depth, f"{output_dir}/{scene_name}_depth.png")

        # DoLP 圖像（已預先計算，直接保存）
        if Config.SAVE_DOLP:
            self._save_png(dolp, f"{output_dir}/{scene_name}_DoLP.png")

            # DoLP 彩色熱力圖
            dolp_uint8 = (dolp * 255).astype(np.uint8)
            dolp_colored = cv2.applyColorMap(dolp_uint8, cv2.COLORMAP_JET)
            cv2.imwrite(f"{output_dir}/{scene_name}_DoLP_color.png", dolp_colored)
            print(f"  [DoLP] 平均: {dolp.mean():.4f}, 最大: {dolp.max():.4f}")

        # 保存玻璃 mask (聯集 - 用於訓練)
        if glass_mask is not None:
            self._save_exr(glass_mask, f"{output_dir}/{scene_name}_glass_mask.exr")
            self._save_png(glass_mask, f"{output_dir}/{scene_name}_glass_mask.png")

        # 保存嚴格玻璃 mask (交集 - 用於評估)
        if glass_mask_strict is not None:
            self._save_exr(glass_mask_strict, f"{output_dir}/{scene_name}_glass_mask_strict.exr")
            self._save_png(glass_mask_strict, f"{output_dir}/{scene_name}_glass_mask_strict.png")

        # 生成品質報告（使用預先計算的 DoLP 統計 + warp 對齊的強度比值）
        print(f"\n[Report] 生成品質報告...")
        report = generate_scene_report(scene_name, I_parallel, I_cross, depth, disparity, glass_mask, glass_mask_left, dolp, dolp_stats)

        report_path = f"{output_dir}/{scene_name}_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 印出關鍵指標（精簡版）
        true_glass_ratio = dolp_stats.get('true_glass_ratio', 1.0)
        true_bg_ratio = dolp_stats.get('true_bg_ratio', 1.0)
        bg_balance = report['intensity_balance']['background_ratio_mean']
        snr = report['noise']['snr_polarization']

        print(f"    ┌─────────────────────────────────────────┐")
        print(f"    │ 玻璃 I(90°)/I(0°)  : {true_glass_ratio:6.3f}x            │")
        print(f"    │ 背景 I(90°)/I(0°)  : {true_bg_ratio:6.3f}x            │")
        print(f"    │ 背景平衡 (warp)    : {bg_balance:6.2f}x " + ("✓" if report['intensity_balance']['is_balanced'] else "⚠") + "           │")
        print(f"    │ SNR               : {snr:6.2f}              │")

        # 玻璃深度有效率
        if 'glass_depth_validity' in report:
            gdv = report['glass_depth_validity']
            validity_pct = gdv['validity_rate'] * 100
            pass_mark = "✓" if gdv['pass'] else "✗"
            print(f"    │ 玻璃深度有效率    : {validity_pct:5.1f}% {pass_mark}           │")

        print(f"    │ 品質分數          : {report['quality']['score']:3d} ({report['quality']['level']})       │")
        print(f"    └─────────────────────────────────────────┘")

        # 參數 JSON
        params = {
            'scene_name': scene_name,
            'config': {
                'width': Config.WIDTH,
                'height': Config.HEIGHT,
                'spp': Config.SPP,
                'baseline_mm': Config.BASELINE,
                'fov_deg': Config.FOV,
            },
            'camera': {
                'left_position': list(Config.left_camera_position()),
                'right_position': list(Config.right_camera_position()),
                'target': list(Config.target_point()),
            },
            'lighting': {
                'led_intensity': Config.LED_INTENSITY,
                'ceiling_emitter_intensity': Config.CEILING_EMITTER_INTENSITY,
            },
            'stats': {
                'I_parallel_range': [float(I_parallel.min()), float(I_parallel.max())],
                'I_cross_range': [float(I_cross.min()), float(I_cross.max())],
                'depth_range_m': [float(depth[depth > 0].min()) if (depth > 0).any() else 0,
                                  float(depth[depth > 0].max()) if (depth > 0).any() else 0],
            },
        }

        with open(f"{output_dir}/{scene_name}_params.json", 'w') as f:
            json.dump(params, f, indent=2)

    def _save_exr(self, image: np.ndarray, path: str):
        """保存 EXR"""
        if image.ndim == 3:
            image = image[:, :, 0]
        bitmap = mi.Bitmap(image.astype(np.float32))
        bitmap.write(path)
        print(f"    -> {path}")

    def _save_png(self, image: np.ndarray, path: str, gamma: float = 2.2):
        """保存 PNG (自動範圍 + gamma correction)"""
        if image.ndim == 3:
            image = image[:, :, 0]
        # 使用 percentile 避免極端值影響
        vmin = np.percentile(image, 1)
        vmax = np.percentile(image, 99.5)
        if vmax > vmin:
            normalized = np.clip((image - vmin) / (vmax - vmin), 0, 1)
            # Gamma correction (線性 → sRGB)
            normalized = np.power(normalized, 1.0 / gamma)
            normalized = (normalized * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image, dtype=np.uint8)
        cv2.imwrite(path, normalized)

    def _save_png_fixed(self, image: np.ndarray, path: str, vmin: float, vmax: float, gamma: float = 2.2):
        """保存 PNG (固定範圍 + gamma correction)"""
        if image.ndim == 3:
            image = image[:, :, 0]
        clipped = np.clip(image, vmin, vmax)
        if vmax > vmin:
            normalized = (clipped - vmin) / (vmax - vmin)
            # Gamma correction (線性 → sRGB)
            normalized = np.power(normalized, 1.0 / gamma)
            normalized = (normalized * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image, dtype=np.uint8)
        cv2.imwrite(path, normalized)

    def _save_depth_png(self, depth: np.ndarray, path: str):
        """保存深度圖 PNG (colormap)"""
        valid = depth > 0
        if not valid.any():
            cv2.imwrite(path, np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.uint8))
            return

        vmin, vmax = depth[valid].min(), depth[valid].max()
        normalized = np.zeros_like(depth)
        normalized[valid] = (depth[valid] - vmin) / (vmax - vmin + 1e-6)
        normalized = (normalized * 255).astype(np.uint8)

        # 反轉 (近=亮, 遠=暗)
        normalized = 255 - normalized
        normalized[~valid] = 0

        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        cv2.imwrite(path, colored)


# ============================================================
# 命令行介面
# ============================================================

def run_single_gpu_wrapper(args_tuple):
    """包裝函數，用於 multiprocessing 啟動前設置環境變量"""
    gpu_id, scenes, output_dir, spp, save_preview = args_tuple

    import os
    import sys

    # 必須在 import mitsuba 之前設置
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 強制重新載入相關模組
    for mod_name in list(sys.modules.keys()):
        if 'mitsuba' in mod_name or 'drjit' in mod_name:
            del sys.modules[mod_name]

    # 現在 import mitsuba
    import mitsuba as mi
    import drjit as dr

    # 動態導入本模組的類和函數
    from pids_renderer_textured import Config, PIDSRenderer

    Config.SPP = spp
    Config.SAVE_PREVIEW = save_preview

    print(f"[GPU {gpu_id}] 初始化完成，CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    renderer = PIDSRenderer()

    for i, scene_path in enumerate(scenes):
        print(f"[GPU {gpu_id}] 進度 {i+1}/{len(scenes)}: {scene_path.name}")
        Config.randomize_for_augmentation(seed=deterministic_hash(scene_path.name))
        try:
            renderer.render_scene(str(scene_path), output_dir)
        except Exception as e:
            print(f"[GPU {gpu_id}] 錯誤 {scene_path}: {e}")
            import traceback
            traceback.print_exc()


def rerender_glass_masks(params_dir: str, obj_dir: str, output_dir: str,
                         max_scenes: Optional[int] = None,
                         scene_list_path: Optional[str] = None,
                         skip: int = 0):
    """
    重新渲染 glass mask（使用修正後的玻璃檢測邏輯）

    v5.1.7 新增功能：
    - 讀取已渲染場景的 params.json 獲取相機位置
    - 使用修正後的 _is_glass_name()（精確匹配 Glass_Clear）
    - 重新渲染 glass mask 並覆蓋原檔案
    - 支持場景列表文件篩選

    Args:
        params_dir: 包含 *_params.json 的目錄
        obj_dir: OBJ 場景目錄
        output_dir: 輸出目錄（通常與 params_dir 相同）
        max_scenes: 最大處理場景數
        scene_list_path: 場景列表文件路徑（每行一個場景名）
        skip: 跳過前 N 個場景（用於多 GPU 並行）
    """
    import json

    # 初始化 Mitsuba
    global mi
    import mitsuba as mi_module
    mi_module.set_variant('cuda_ad_rgb')
    mi = mi_module

    print("=" * 70)
    print("[PIDS Renderer v5.1.7 - Glass Mask Re-render Mode]")
    print("=" * 70)
    print(f"  修正: _is_glass_name() 現使用精確匹配 'Glass_Clear'")
    print(f"  Params 目錄: {params_dir}")
    print(f"  OBJ 目錄: {obj_dir}")
    print(f"  輸出目錄: {output_dir}")
    if scene_list_path:
        print(f"  場景列表: {scene_list_path}")
    print("=" * 70)

    # 讀取場景列表（如果提供）
    allowed_scenes = None
    if scene_list_path:
        if not os.path.exists(scene_list_path):
            print(f"[錯誤] 找不到場景列表文件: {scene_list_path}")
            return
        with open(scene_list_path, 'r', encoding='utf-8') as f:
            # 每行一個場景名，忽略空行和註釋
            allowed_scenes = set()
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 支持 "scene_0001" 或 "scene_0001.obj" 格式
                    scene_name = line.replace('.obj', '').replace('_params.json', '')
                    allowed_scenes.add(scene_name)
        print(f"  場景列表包含 {len(allowed_scenes)} 個場景")

    # 找到所有 params.json
    params_files = sorted(Path(params_dir).glob('*_params.json'))

    # 根據場景列表篩選
    if allowed_scenes:
        params_files = [p for p in params_files
                       if p.stem.replace('_params', '') in allowed_scenes]
        print(f"  篩選後: {len(params_files)} 個場景")

    # 跳過前 N 個場景（用於多 GPU 並行）
    if skip > 0:
        params_files = params_files[skip:]
        print(f"  跳過前 {skip} 個，剩餘: {len(params_files)} 個場景")

    if max_scenes:
        params_files = params_files[:max_scenes]

    print(f"\n待處理: {len(params_files)} 個場景")

    os.makedirs(output_dir, exist_ok=True)
    success_count = 0
    fail_count = 0

    for i, params_path in enumerate(params_files):
        print(f"\n[進度] {i+1}/{len(params_files)}: {params_path.name}")

        try:
            # 讀取 params.json
            with open(params_path, 'r') as f:
                params = json.load(f)

            scene_name = params['scene_name']
            camera = params['camera']
            left_pos = tuple(camera['left_position'])
            right_pos = tuple(camera['right_position'])
            target = tuple(camera['target'])

            # 找到對應的 OBJ
            obj_path = Path(obj_dir) / f"{scene_name}.obj"
            if not obj_path.exists():
                print(f"  [跳過] 找不到 OBJ: {obj_path}")
                fail_count += 1
                continue

            # 建立 SceneBuilder（使用修正後的玻璃檢測）
            builder = SceneBuilder(str(obj_path))

            if not builder.has_glass:
                print(f"  [跳過] 場景無玻璃材質")
                # 仍然保存空 mask
                empty_mask = np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.float32)
                _save_mask(empty_mask, output_dir, scene_name)
                builder.cleanup()
                success_count += 1
                continue

            # 渲染左右視角的 glass mask
            glass_mask_left = _render_glass_mask_standalone(builder, left_pos, target, "左")
            glass_mask_right = _render_glass_mask_standalone(builder, right_pos, target, "右")

            # 聯集 (用於訓練)
            glass_mask = ((glass_mask_left > 0.5) | (glass_mask_right > 0.5)).astype(np.float32)
            pixel_count = int(np.sum(glass_mask))
            print(f"  [Glass Mask (聯集)] 玻璃像素: {pixel_count} ({pixel_count/(Config.WIDTH*Config.HEIGHT)*100:.1f}%)")

            # 保存
            _save_mask(glass_mask, output_dir, scene_name)
            builder.cleanup()
            success_count += 1

        except Exception as e:
            print(f"  [錯誤] {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    print("\n" + "=" * 70)
    print(f"[完成] 成功: {success_count}, 失敗: {fail_count}")
    print("=" * 70)


def rerender_strict_masks(params_dir: str, obj_dir: str, output_dir: str,
                          max_scenes: Optional[int] = None,
                          scene_list_path: Optional[str] = None,
                          skip: int = 0):
    """
    重新渲染嚴格 glass mask（交集）

    與 rerender_glass_masks 類似，但輸出交集 mask (glass_mask_strict.exr)
    用於評估時更精確的 Glass EPE 計算

    Args:
        params_dir: 包含 *_params.json 的目錄
        obj_dir: OBJ 場景目錄
        output_dir: 輸出目錄
        max_scenes: 最大處理場景數
        scene_list_path: 場景列表文件路徑（每行一個場景名）
        skip: 跳過前 N 個場景
    """
    import json

    # 初始化 Mitsuba
    global mi
    import mitsuba as mi_module
    mi_module.set_variant('cuda_ad_rgb')
    mi = mi_module

    print("=" * 70)
    print("[PIDS Renderer - Strict Glass Mask Re-render Mode]")
    print("=" * 70)
    print(f"  輸出: glass_mask_strict.exr (左右視角交集)")
    print(f"  Params 目錄: {params_dir}")
    print(f"  OBJ 目錄: {obj_dir}")
    print(f"  輸出目錄: {output_dir}")
    if scene_list_path:
        print(f"  場景列表: {scene_list_path}")
    print("=" * 70)

    # 讀取場景列表（如果提供）
    allowed_scenes = None
    if scene_list_path:
        if not os.path.exists(scene_list_path):
            print(f"[錯誤] 找不到場景列表文件: {scene_list_path}")
            return
        with open(scene_list_path, 'r', encoding='utf-8') as f:
            allowed_scenes = set()
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    scene_name = line.replace('.obj', '').replace('_params.json', '')
                    allowed_scenes.add(scene_name)
        print(f"  場景列表包含 {len(allowed_scenes)} 個場景")

    # 找到所有 params.json
    params_files = sorted(Path(params_dir).glob('*_params.json'))

    # 根據場景列表篩選
    if allowed_scenes:
        params_files = [p for p in params_files
                       if p.stem.replace('_params', '') in allowed_scenes]
        print(f"  篩選後: {len(params_files)} 個場景")

    if skip > 0:
        params_files = params_files[skip:]
        print(f"  跳過前 {skip} 個，剩餘: {len(params_files)} 個場景")

    if max_scenes:
        params_files = params_files[:max_scenes]

    print(f"\n待處理: {len(params_files)} 個場景")

    os.makedirs(output_dir, exist_ok=True)
    success_count = 0
    fail_count = 0

    for i, params_path in enumerate(params_files):
        scene_name = params_path.stem.replace('_params', '')
        print(f"\n[{i+1}/{len(params_files)}] {scene_name}")

        try:
            # 讀取 params.json
            with open(params_path, 'r', encoding='utf-8') as f:
                params = json.load(f)

            # 獲取相機位置
            left_pos = tuple(params['camera']['left_position'])
            right_pos = tuple(params['camera']['right_position'])
            target = tuple(params['camera']['target'])

            # 找到對應的 OBJ
            obj_path = Path(obj_dir) / f"{scene_name}.obj"
            if not obj_path.exists():
                print(f"  [跳過] 找不到 OBJ: {obj_path}")
                fail_count += 1
                continue

            # 建立 SceneBuilder
            builder = SceneBuilder(str(obj_path))

            if not builder.has_glass:
                print(f"  [跳過] 場景無玻璃材質")
                empty_mask = np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.float32)
                _save_mask(empty_mask, output_dir, scene_name, suffix="_strict")
                builder.cleanup()
                success_count += 1
                continue

            # 渲染左右視角的 glass mask
            glass_mask_left = _render_glass_mask_standalone(builder, left_pos, target, "左")
            glass_mask_right = _render_glass_mask_standalone(builder, right_pos, target, "右")

            # 交集 (嚴格 mask)
            glass_mask_strict = ((glass_mask_left > 0.5) & (glass_mask_right > 0.5)).astype(np.float32)
            pixel_count = int(np.sum(glass_mask_strict))
            print(f"  [Glass Mask (交集)] 玻璃像素: {pixel_count} ({pixel_count/(Config.WIDTH*Config.HEIGHT)*100:.1f}%)")

            # 保存
            _save_mask(glass_mask_strict, output_dir, scene_name, suffix="_strict")
            builder.cleanup()
            success_count += 1

        except Exception as e:
            print(f"  [錯誤] {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    print("\n" + "=" * 70)
    print(f"[完成] 成功: {success_count}, 失敗: {fail_count}")
    print("=" * 70)


def _render_glass_mask_standalone(builder: 'SceneBuilder',
                                   position: Tuple[float, float, float],
                                   target: Tuple[float, float, float],
                                   camera_name: str = "") -> np.ndarray:
    """
    獨立渲染 glass mask 函數（用於 rerender 模式）

    與 PIDSRenderer._render_glass_mask 相同邏輯
    """
    if not builder.has_glass:
        return np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.float32)

    # 座標轉換
    pos_m = builder._transform_point(position)
    tgt_m = builder._transform_point(target)

    # 基本變換矩陣
    transform = mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]) @ \
                mi.ScalarTransform4f.rotate([1, 0, 0], -90)

    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'aov',
            'aovs': 'dd.y:depth',
            'integrator': {
                'type': 'path',
                'max_depth': 2,
            },
        },
        'sensor': {
            'type': 'perspective',
            'fov': Config.FOV,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=pos_m,
                target=tgt_m,
                up=[0, 1, 0],
            ),
            'film': {
                'type': 'hdrfilm',
                'width': Config.WIDTH,
                'height': Config.HEIGHT,
                'pixel_format': 'luminance',
                'component_format': 'float32',
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 4,
            },
        },
        'glass_mesh': {
            'type': 'obj',
            'filename': builder.glass_obj_path,
            'face_normals': False,
            'to_world': transform,
            'bsdf': MaterialFactory.diffuse(0.5),
        },
    }

    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, spp=4)
    img_np = np.array(image)

    # 提取深度通道
    if img_np.ndim == 3 and img_np.shape[2] >= 2:
        depth_channel = img_np[:, :, 1]
    elif img_np.ndim == 3:
        depth_channel = img_np[:, :, 0]
    else:
        depth_channel = img_np

    # 二值化
    glass_mask = (depth_channel > 0).astype(np.float32)

    pixel_count = int(np.sum(glass_mask))
    pixel_ratio = pixel_count / (Config.WIDTH * Config.HEIGHT)
    cam_label = f" ({camera_name})" if camera_name else ""
    print(f"  [Glass Mask{cam_label}] 玻璃像素: {pixel_count} ({pixel_ratio*100:.1f}%)")

    return glass_mask


def _save_mask(glass_mask: np.ndarray, output_dir: str, scene_name: str, suffix: str = ""):
    """保存 glass mask（EXR + PNG）

    Args:
        glass_mask: mask 陣列
        output_dir: 輸出目錄
        scene_name: 場景名稱
        suffix: 檔名後綴，例如 "_strict" 會變成 glass_mask_strict.exr
    """
    # EXR
    exr_path = f"{output_dir}/{scene_name}_glass_mask{suffix}.exr"
    bitmap = mi.Bitmap(glass_mask.astype(np.float32))
    bitmap.write(exr_path)
    print(f"    -> {exr_path}")

    # PNG
    png_path = f"{output_dir}/{scene_name}_glass_mask{suffix}.png"
    img_uint8 = (glass_mask * 255).astype(np.uint8)
    cv2.imwrite(png_path, img_uint8)
    print(f"    -> {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description='PIDS Stage 1 Renderer v5.1.8 (Strict Glass Mask)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 單一場景
  python pids_renderer_textured.py --scene scene_0001.obj --output ./output

  # 批次渲染 + QA（模擬場景跳過 C1）
  python pids_renderer_textured.py --input_dir ./scenes --output ./output --max_scenes 100 --skip-c1

  # 多 GPU 並行
  python pids_renderer_textured.py --input_dir ./scenes --output ./output --num_gpus 4 --skip-c1

  # 重新渲染 glass mask（聯集，使用已有的 params.json）
  python pids_renderer_textured.py --rerender-mask ./output --obj-dir ./scenes --output ./output

  # 只重新渲染篩選過的場景（使用場景列表文件）
  python pids_renderer_textured.py --rerender-mask ./output --obj-dir ./scenes --output ./output --scene-list train_scenes.txt

  # 重新渲染嚴格 glass mask（交集，用於評估）
  python pids_renderer_textured.py --rerender-strict-mask ./output --obj-dir ./scenes --output ./output --scene-list train_scenes.txt
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--scene', type=str, help='單一 OBJ 場景')
    input_group.add_argument('--input_dir', type=str, help='OBJ 場景目錄')
    input_group.add_argument('--rerender-mask', type=str, dest='rerender_mask',
                            help='重新渲染 glass mask（聯集）：指定包含 *_params.json 的目錄')
    input_group.add_argument('--rerender-strict-mask', type=str, dest='rerender_strict_mask',
                            help='重新渲染嚴格 glass mask（交集）：指定包含 *_params.json 的目錄')

    parser.add_argument('--obj-dir', type=str, dest='obj_dir',
                        help='OBJ 場景目錄（與 --rerender-mask/--rerender-strict-mask 搭配使用）')
    parser.add_argument('--scene-list', type=str, dest='scene_list',
                        help='場景列表文件（每行一個場景名，如 train_scenes.txt）')
    parser.add_argument('--output', type=str, required=True, help='輸出目錄')
    parser.add_argument('--spp', type=int, default=Config.SPP, help=f'SPP (預設: {Config.SPP})')
    parser.add_argument('--max_scenes', type=int, default=None, help='最大場景數')
    parser.add_argument('--skip', type=int, default=0, help='跳過前 N 個場景（用於多 GPU 並行）')
    parser.add_argument('--no_preview', action='store_true', help='不保存預覽 PNG')
    parser.add_argument('--num_gpus', type=int, default=1, help='使用的 GPU 數量（預設: 1）')
    parser.add_argument('--skip-qa', action='store_true', help='跳過 QA 驗證')
    parser.add_argument('--skip-c1', action='store_true', help='QA 時跳過 C1 (Geometric Consistency) - 模擬場景建議使用')

    # 數據集整理參數
    parser.add_argument('--organize', type=str, default=None,
                        help='整理後的數據集輸出目錄（啟用數據集整理）')
    parser.add_argument('--train_size', type=int, default=None,
                        help='訓練集場景數（剩餘為測試集）')
    parser.add_argument('--organize_seed', type=int, default=42,
                        help='訓練/測試分割隨機種子（預設: 42）')
    parser.add_argument('--organize_move', action='store_true',
                        help='使用移動模式（預設: 複製）')

    args = parser.parse_args()

    # ============================================================
    # 特殊模式：重新渲染 glass mask
    # ============================================================
    if args.rerender_mask:
        if not args.obj_dir:
            print("[錯誤] --rerender-mask 需要搭配 --obj-dir 指定 OBJ 場景目錄")
            return
        rerender_glass_masks(args.rerender_mask, args.obj_dir, args.output,
                            args.max_scenes, args.scene_list, args.skip)
        return

    # ============================================================
    # 特殊模式：重新渲染嚴格 glass mask（交集）
    # ============================================================
    if args.rerender_strict_mask:
        if not args.obj_dir:
            print("[錯誤] --rerender-strict-mask 需要搭配 --obj-dir 指定 OBJ 場景目錄")
            return
        rerender_strict_masks(args.rerender_strict_mask, args.obj_dir, args.output,
                             args.max_scenes, args.scene_list, args.skip)
        return

    # 更新配置
    Config.SPP = args.spp
    Config.SAVE_PREVIEW = not args.no_preview

    # 收集場景
    if args.scene:
        scenes = [Path(args.scene)]  # 轉換為 Path 物件
    else:
        # 排除渲染過程中產生的分離 OBJ 文件 (_glass, _ceiling, _other)
        all_objs = sorted(Path(args.input_dir).glob('*.obj'))
        scenes = [f for f in all_objs
                  if not any(x in f.stem for x in ['_glass', '_ceiling', '_other'])]
        # 跳過前 N 個場景（用於多 GPU 並行）
        if args.skip > 0:
            scenes = scenes[args.skip:]
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]

    print(f"[PIDS Renderer v5.1.8 - Strict Glass Mask]")
    print(f"  偏振修正: 天花板光降低, DoLP用Stokes, GlassMask聯集+交集")
    print(f"  QA 整合: {'跳過' if args.skip_qa else '啟用'} (C1: {'跳過' if args.skip_c1 else '啟用'})")
    print(f"  GPU 數量: {args.num_gpus}")
    if args.skip > 0:
        print(f"  跳過前 {args.skip} 個場景")
    print(f"待渲染: {len(scenes)} 個場景")

    start_time = time.time()

    if args.num_gpus > 1:
        # 多 GPU 並行模式 - 使用 subprocess 確保環境變量正確
        import subprocess

        # 均勻分配場景到各 GPU（餘數分散給前幾個 GPU，而非全給最後一個）
        total_scenes = len(scenes)
        base_count = total_scenes // args.num_gpus
        remainder = total_scenes % args.num_gpus

        print(f"\n場景分配 (總計 {total_scenes} 個，均勻分配):")
        processes = []

        current_idx = 0
        for gpu_id in range(args.num_gpus):
            # 前 remainder 個 GPU 各多分 1 個場景
            num_scenes = base_count + (1 if gpu_id < remainder else 0)
            start_idx = current_idx

            print(f"  GPU {gpu_id}: {num_scenes} 個場景 (skip={args.skip + start_idx})")
            current_idx += num_scenes

            # 構建子進程命令
            cmd = [
                'python', __file__,
                '--input_dir', args.input_dir,
                '--output', args.output,
                '--spp', str(args.spp),
                '--skip', str(args.skip + start_idx),
                '--max_scenes', str(num_scenes),
                '--skip-qa',  # 子進程跳過 QA，由主進程統一執行
            ]
            if args.no_preview:
                cmd.append('--no_preview')
            if args.skip_c1:
                cmd.append('--skip-c1')

            # 設置環境變量並啟動子進程
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            log_file = open(f'gpu{gpu_id}.log', 'w')
            p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((p, log_file))
            print(f"[啟動] GPU {gpu_id} 進程 PID: {p.pid}")

        # 等待所有進程完成
        print(f"\n等待所有 GPU 完成...")
        for p, log_file in processes:
            p.wait()
            log_file.close()

        print(f"所有 GPU 渲染完成！")

    else:
        # 單 GPU 模式
        renderer = PIDSRenderer()
        for i, scene_path in enumerate(scenes):
            print(f"\n[進度] {i+1}/{len(scenes)}")
            Config.randomize_for_augmentation(seed=deterministic_hash(scene_path.name))
            try:
                renderer.render_scene(str(scene_path), args.output)
            except Exception as e:
                print(f"[錯誤] {scene_path}: {e}")
                import traceback
                traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n[完成] 渲染耗時: {elapsed/60:.1f} 分鐘")

    # ============================================================
    # 自動執行 QA 驗證
    # ============================================================
    if not args.skip_qa:
        print(f"\n{'='*60}")
        print(f"[QA] 開始品質驗證...")
        print(f"{'='*60}")

        try:
            QV = lazy_import_qa()
            if QV is None:
                print(f"[QA] 模組不可用，跳過驗證")
            else:
                validator = QV(args.output, skip_c1=args.skip_c1)
                count = validator.load_reports()

                if count > 0:
                    results = validator.validate_all(check_exr=True)

                    # 生成 Markdown 報告
                    md_report = validator.generate_markdown_report(results)
                    report_path = Path(args.output) / 'quality_report.md'
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(md_report)

                    # 印出摘要
                    summary = results['summary']
                    pass_rate = summary['passed'] / summary['total'] * 100 if summary['total'] > 0 else 0

                    print(f"\n[QA 結果]")
                    print(f"  總場景: {summary['total']}")
                    print(f"  通過: {summary['passed']}")
                    print(f"  未通過: {summary['failed']}")
                    print(f"  通過率: {pass_rate:.1f}%")
                    print(f"  報告: {report_path}")

                    # 輸出未通過場景列表
                    failed_scenes = [item['scene_name'] for item in results['results'] if not item['passed']]
                    if failed_scenes:
                        failed_path = Path(args.output) / 'failed_scenes.txt'
                        with open(failed_path, 'w', encoding='utf-8') as f:
                            for scene in failed_scenes:
                                f.write(f"{scene}\n")
                        print(f"  未通過列表: {failed_path}")
                else:
                    print(f"[QA] 找不到報告檔案，跳過驗證")

        except Exception as e:
            print(f"[QA 錯誤] {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[QA] 已跳過（使用 --skip-qa）")

    # ============================================================
    # 自動整理數據集
    # ============================================================
    if args.organize:
        print(f"\n{'='*60}")
        print(f"[Organize] 開始整理數據集...")
        print(f"{'='*60}")

        organize_dataset(
            input_dir=Path(args.output),
            output_dir=Path(args.organize),
            train_size=args.train_size,
            seed=args.organize_seed,
            copy_mode=not args.organize_move,
        )

    total_elapsed = time.time() - start_time
    print(f"\n[完成] 總耗時: {total_elapsed/60:.1f} 分鐘")


if __name__ == '__main__':
    # 快速檢查是否為多 GPU 模式，如果是則直接啟動子進程，不初始化任何 CUDA
    import sys
    if '--num_gpus' in sys.argv:
        idx = sys.argv.index('--num_gpus')
        if idx + 1 < len(sys.argv):
            num_gpus = int(sys.argv[idx + 1])
            if num_gpus > 1:
                # 多 GPU 模式：只做調度，不 import mitsuba
                import subprocess

                # 解析必要參數
                parser = argparse.ArgumentParser()
                parser.add_argument('--input_dir', type=str, required=True)
                parser.add_argument('--output', type=str, required=True)
                parser.add_argument('--spp', type=int, default=1024)
                parser.add_argument('--max_scenes', type=int, default=None)
                parser.add_argument('--skip', type=int, default=0)
                parser.add_argument('--no_preview', action='store_true')
                parser.add_argument('--num_gpus', type=int, default=1)
                parser.add_argument('--skip-qa', action='store_true')
                parser.add_argument('--skip-c1', action='store_true')
                parser.add_argument('--organize', type=str, default=None)
                parser.add_argument('--train_size', type=int, default=None)
                parser.add_argument('--organize_seed', type=int, default=42)
                parser.add_argument('--organize_move', action='store_true')
                args = parser.parse_args()

                # 收集場景
                all_objs = sorted(Path(args.input_dir).glob('*.obj'))
                scenes = [f for f in all_objs
                          if not any(x in f.stem for x in ['_glass', '_ceiling', '_other'])]
                if args.skip > 0:
                    scenes = scenes[args.skip:]
                if args.max_scenes:
                    scenes = scenes[:args.max_scenes]

                print(f"[PIDS Renderer v5.1.8 - Multi-GPU Launcher]")
                print(f"  GPU 數量: {num_gpus}")
                print(f"  待渲染: {len(scenes)} 個場景")
                print(f"  QA: {'跳過' if '--skip-qa' in sys.argv else '啟用'}")

                # 均勻分配場景到各 GPU（餘數分散給前幾個 GPU）
                total_scenes = len(scenes)
                base_count = total_scenes // num_gpus
                remainder = total_scenes % num_gpus

                print(f"\n場景分配 (均勻分配):")
                processes = []

                current_idx = 0
                for gpu_id in range(num_gpus):
                    # 前 remainder 個 GPU 各多分 1 個場景
                    num_scenes_gpu = base_count + (1 if gpu_id < remainder else 0)
                    start_idx = current_idx

                    print(f"  GPU {gpu_id}: {num_scenes_gpu} 個場景 (skip={args.skip + start_idx})")
                    current_idx += num_scenes_gpu

                    # 單 GPU 子進程命令
                    cmd = [
                        sys.executable, __file__,
                        '--input_dir', args.input_dir,
                        '--output', args.output,
                        '--spp', str(args.spp),
                        '--skip', str(args.skip + start_idx),
                        '--max_scenes', str(num_scenes_gpu),
                    ]
                    if args.no_preview:
                        cmd.append('--no_preview')
                    if args.skip_qa:
                        cmd.append('--skip-qa')
                    if args.skip_c1:
                        cmd.append('--skip-c1')

                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

                    log_file = open(f'gpu{gpu_id}.log', 'w')
                    p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
                    processes.append((p, log_file))
                    print(f"[啟動] GPU {gpu_id} PID: {p.pid}")

                print(f"\n等待所有 GPU 完成...")
                print(f"監控: tail -f gpu0.log gpu1.log ...")

                for p, log_file in processes:
                    p.wait()
                    log_file.close()

                print(f"\n所有 GPU 渲染完成！")

                # 整理數據集（只在主進程執行一次）
                if args.organize:
                    print(f"\n{'='*60}")
                    print(f"[Organize] 開始整理數據集...")
                    print(f"{'='*60}")
                    organize_dataset(
                        input_dir=Path(args.output),
                        output_dir=Path(args.organize),
                        train_size=args.train_size,
                        seed=args.organize_seed,
                        copy_mode=not args.organize_move,
                    )

                sys.exit(0)

    # 單 GPU 模式或子進程
    main()

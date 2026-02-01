"""
PIDS Synthetic Dataset Loader
用於載入 Mitsuba 渲染的合成偏振立體數據

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import OpenEXR
import Imath


class EXRReader:
    """EXR 檔案讀取器"""

    @staticmethod
    def read_exr(filepath: str, channels: List[str] = ['R', 'G', 'B']) -> np.ndarray:
        """
        讀取 EXR 檔案

        Args:
            filepath: EXR 檔案路徑
            channels: 要讀取的通道列表

        Returns:
            numpy array of shape (H, W) or (H, W, C)
        """
        exr_file = OpenEXR.InputFile(filepath)
        header = exr_file.header()

        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # 獲取可用通道
        available_channels = list(header['channels'].keys())

        # 決定要讀取的通道
        if len(available_channels) == 1:
            # 單通道 (深度/視差)
            channel = available_channels[0]
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            data = exr_file.channel(channel, pt)
            arr = np.frombuffer(data, dtype=np.float32).reshape(height, width)
            return arr
        else:
            # 多通道 (RGB)
            arrays = []
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            for ch in channels:
                if ch in available_channels:
                    data = exr_file.channel(ch, pt)
                    arr = np.frombuffer(data, dtype=np.float32).reshape(height, width)
                    arrays.append(arr)

            if len(arrays) == 1:
                return arrays[0]
            return np.stack(arrays, axis=-1)

    @staticmethod
    def read_rgb(filepath: str) -> np.ndarray:
        """讀取 RGB EXR，返回 (H, W, 3)"""
        return EXRReader.read_exr(filepath, ['R', 'G', 'B'])

    @staticmethod
    def read_depth(filepath: str) -> np.ndarray:
        """讀取深度 EXR，返回 (H, W)"""
        return EXRReader.read_exr(filepath)


class PIDSSyntheticDataset(Dataset):
    """
    PIDS 合成數據集

    支援兩種目錄結構:
    1. 整理後結構 (organized):
       dataset/
       ├── stereo_pairs/    (left_parallel.exr, right_cross.exr)
       ├── ground_truth/    (disparity.exr, depth.exr)
       └── masks/           (glass_mask.exr)

    2. 原始結構 (all in one folder):
       output/
       ├── scene_XXXX_left_parallel.exr
       ├── scene_XXXX_right_cross.exr
       ├── scene_XXXX_disparity.exr
       └── scene_XXXX_glass_mask.exr
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform = None,
        max_disparity: float = 576.0,
        exclude_failed: bool = True,
        failed_scenes: Optional[List[str]] = None,
        augment: bool = True,
        val_split: float = 0.2,
    ):
        """
        Args:
            data_dir: 數據目錄路徑 (可以是 dataset/ 或 stereo_pairs/)
            split: 'train' 或 'val'
            transform: 額外的數據變換
            max_disparity: 最大視差值（用於正規化）
            exclude_failed: 是否排除品質檢測未通過的場景
            failed_scenes: 未通過場景列表
            augment: 是否進行數據增強 (亮度/對比度/翻轉，不含 crop)
            val_split: 驗證集比例 (預設 0.2 = 80/20 分割)
        """
        self.val_split = val_split
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.max_disparity = max_disparity
        self.augment = augment and (split == 'train')
        self._use_external_split = False  # 是否使用外部 train/val 分割

        # 自動偵測目錄結構
        self._detect_directory_structure()

        # 預設的未通過場景列表
        default_failed = [
            'scene_0057', 'scene_0062', 'scene_0082', 'scene_0117',
            'scene_0140', 'scene_0163', 'scene_0171', 'scene_0232',
            'scene_0331', 'scene_0333', 'scene_0391', 'scene_0452'
        ]
        self.failed_scenes = set(failed_scenes or default_failed) if exclude_failed else set()

        # 掃描場景
        self.scenes = self._scan_scenes()

        # 分割訓練/驗證集
        if self._use_external_split:
            # 使用外部分割 (data/train, data/val 目錄結構)
            # 不做內部分割，直接使用掃描到的所有場景
            print(f"[PIDSDataset] {split} split: {len(self.scenes)} scenes (external split)")
        else:
            # 內部分割 (固定種子確保可重現)
            np.random.seed(42)
            indices = np.random.permutation(len(self.scenes))
            split_idx = int(len(indices) * (1 - self.val_split))

            if split == 'train':
                self.scenes = [self.scenes[i] for i in indices[:split_idx]]
            else:
                self.scenes = [self.scenes[i] for i in indices[split_idx:]]

            train_ratio = int((1 - self.val_split) * 100)
            val_ratio = int(self.val_split * 100)
            print(f"[PIDSDataset] {split} split: {len(self.scenes)} scenes ({train_ratio}/{val_ratio} split)")
        print(f"[PIDSDataset] Directory structure: {self.dir_structure}")

        # 統計命名格式
        pol_count = sum(1 for s in self.scenes if self.scene_naming.get(s) == 'pol')
        nopol_count = sum(1 for s in self.scenes if self.scene_naming.get(s) == 'nopol')
        if pol_count > 0 and nopol_count > 0:
            print(f"[PIDSDataset] Mixed naming: {pol_count} pol + {nopol_count} nopol")
        elif nopol_count > 0:
            print(f"[PIDSDataset] Naming format: nopol (_left.exr, _right.exr)")

    def _detect_directory_structure(self):
        """自動偵測目錄結構"""
        # 檢查是否為整理後的結構
        if self.data_dir.name == 'stereo_pairs':
            # 傳入的是 stereo_pairs 目錄
            self.stereo_dir = self.data_dir
            self.depth_dir = self.data_dir.parent / 'ground_truth'
            self.mask_dir = self.data_dir.parent / 'masks'
            self.dir_structure = 'organized'
        elif (self.data_dir / 'train' / 'stereo_pairs').exists():
            # 新結構：dataset/train/stereo_pairs (只讀取訓練集)
            self.stereo_dir = self.data_dir / 'train' / 'stereo_pairs'
            self.depth_dir = self.data_dir / 'train' / 'ground_truth'
            self.mask_dir = self.data_dir / 'train' / 'masks'
            self.dir_structure = 'train_split'
            print(f"[PIDSDataset] Using train/ subdirectory (ignoring test/)")
        elif (self.data_dir / 'train').exists() and (self.data_dir / 'val').exists():
            # 新結構：dataset/train 和 dataset/val 分開 (flat 格式)
            # 根據 split 選擇目錄
            if self.split == 'train':
                target_dir = self.data_dir / 'train'
            else:
                target_dir = self.data_dir / 'val'
            self.stereo_dir = target_dir
            self.depth_dir = target_dir
            self.mask_dir = target_dir
            self.dir_structure = 'split_flat'
            self._use_external_split = True  # 標記使用外部分割，不做內部分割
            print(f"[PIDSDataset] Using {self.split}/ subdirectory (external split)")
        elif (self.data_dir / 'stereo_pairs').exists():
            # 傳入的是 dataset 根目錄
            self.stereo_dir = self.data_dir / 'stereo_pairs'
            self.depth_dir = self.data_dir / 'ground_truth'
            self.mask_dir = self.data_dir / 'masks'
            self.dir_structure = 'organized'
        else:
            # 原始結構 (all in one)
            self.stereo_dir = self.data_dir
            self.depth_dir = self.data_dir
            self.mask_dir = self.data_dir
            self.dir_structure = 'flat'

    def _scan_scenes(self) -> List[str]:
        """掃描數據目錄中的有效場景"""
        scenes = []
        self.scene_naming = {}  # 記錄每個場景的命名格式: 'pol' or 'nopol'

        # 查找 polarized 格式: *_left_parallel.exr
        for f in self.stereo_dir.glob("*_left_parallel.exr"):
            scene_name = f.stem.replace('_left_parallel', '')

            # 跳過 glass/other 子場景，只用主場景
            if '_glass' in scene_name or '_other' in scene_name:
                continue

            # 跳過未通過品質檢測的場景
            if scene_name in self.failed_scenes:
                continue

            # 確認必要檔案存在
            stereo_files = [
                self.stereo_dir / f"{scene_name}_left_parallel.exr",
                self.stereo_dir / f"{scene_name}_right_cross.exr",
            ]
            depth_files = [
                self.depth_dir / f"{scene_name}_disparity.exr",
            ]

            if all(f.exists() for f in stereo_files + depth_files):
                scenes.append(scene_name)
                self.scene_naming[scene_name] = 'pol'

        # 查找 non-polarized 格式: *_left.exr (排除已找到的 pol 場景)
        for f in self.stereo_dir.glob("*_left.exr"):
            # 排除 _left_parallel.exr
            if '_left_parallel' in f.stem:
                continue

            scene_name = f.stem.replace('_left', '')

            # 跳過 glass/other 子場景
            if '_glass' in scene_name or '_other' in scene_name:
                continue

            # 跳過未通過品質檢測的場景
            if scene_name in self.failed_scenes:
                continue

            # 跳過已經找到的場景
            if scene_name in self.scene_naming:
                continue

            # 確認必要檔案存在
            stereo_files = [
                self.stereo_dir / f"{scene_name}_left.exr",
                self.stereo_dir / f"{scene_name}_right.exr",
            ]
            depth_files = [
                self.depth_dir / f"{scene_name}_disparity.exr",
            ]

            if all(f.exists() for f in stereo_files + depth_files):
                scenes.append(scene_name)
                self.scene_naming[scene_name] = 'nopol'

        scenes.sort()
        return scenes

    def __len__(self) -> int:
        return len(self.scenes)

    def _load_scene(self, scene_name: str) -> Dict[str, np.ndarray]:
        """載入單一場景的所有數據"""
        # 根據場景命名格式選擇正確的檔案名
        naming = self.scene_naming.get(scene_name, 'pol')
        if naming == 'pol':
            left_path = self.stereo_dir / f"{scene_name}_left_parallel.exr"
            right_path = self.stereo_dir / f"{scene_name}_right_cross.exr"
        else:  # nopol
            left_path = self.stereo_dir / f"{scene_name}_left.exr"
            right_path = self.stereo_dir / f"{scene_name}_right.exr"

        # 載入視差圖 (從 depth_dir)
        disp_path = self.depth_dir / f"{scene_name}_disparity.exr"

        # 載入遮罩 (從 mask_dir)
        mask_path = self.mask_dir / f"{scene_name}_glass_mask.exr"
        mask_strict_path = self.mask_dir / f"{scene_name}_glass_mask_strict.exr"

        left = EXRReader.read_rgb(str(left_path))
        right = EXRReader.read_rgb(str(right_path))
        disparity = EXRReader.read_depth(str(disp_path))

        # 處理單通道圖像：擴展為 3 通道
        if left.ndim == 2:
            left = np.stack([left, left, left], axis=-1)
        if right.ndim == 2:
            right = np.stack([right, right, right], axis=-1)

        # 載入玻璃遮罩（如果存在）
        if mask_path.exists():
            mask = EXRReader.read_depth(str(mask_path))
        else:
            mask = np.zeros_like(disparity)

        # 載入嚴格玻璃遮罩（交集，用於評估）
        if mask_strict_path.exists():
            mask_strict = EXRReader.read_depth(str(mask_strict_path))
        else:
            # 如果沒有 strict mask，使用普通 mask
            mask_strict = mask.copy()

        return {
            'left': left,
            'right': right,
            'disparity': disparity,
            'glass_mask': mask,
            'glass_mask_strict': mask_strict,
        }

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """正規化圖像到 [0, 1] 範圍"""
        # 處理 HDR 範圍
        img = np.clip(img, 0, None)

        # 使用 percentile 來處理極端值
        p99 = np.percentile(img, 99)
        if p99 > 0:
            img = img / p99

        img = np.clip(img, 0, 1)
        return img


    def _augment(
        self,
        left: np.ndarray,
        right: np.ndarray,
        disparity: np.ndarray,
        mask: np.ndarray,
        mask_strict: np.ndarray,
        scene_name: str = ''
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        數據增強 (含 Sensor Realism Augmentation)

        增強分為兩類:
        1. 場景級增強: 亮度、對比度、翻轉
        2. 感測器真實性增強: 消光比洩漏、shot noise、read noise

        Mitsuba 使用理想偏振器 (消光比=∞) 和無噪聲感測器，
        感測器真實性增強縮小 synthetic-to-real domain gap。
        """
        naming = self.scene_naming.get(scene_name, 'pol')

        # ========== Sensor Realism Augmentation ==========

        # 1. 模擬有限消光比 (僅對偏振數據，光學層面效應)
        #    真實偏振片消光比約 100:1 ~ 10000:1
        #    左相機 (0° pol): I_left_real  = I_parallel + I_cross / ER
        #    右相機 (90° pol): I_right_real = I_cross + I_parallel / ER
        #    主要效應在右相機 (I_parallel >> I_cross 在玻璃上)
        if naming == 'pol' and np.random.rand() < 0.5:
            log_er = np.random.uniform(np.log(100), np.log(10000))
            extinction_ratio = np.exp(log_er)
            left_leakage = right / extinction_ratio
            right_leakage = left / extinction_ratio
            left = left + left_leakage
            right = right + right_leakage

        # ========== 場景級增強 ==========

        # 隨機亮度調整
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            left = left * brightness
            right = right * brightness

        # 隨機對比度調整
        if np.random.rand() < 0.5:
            contrast = np.random.uniform(0.8, 1.2)
            mean_left = np.mean(left)
            mean_right = np.mean(right)
            left = (left - mean_left) * contrast + mean_left
            right = (right - mean_right) * contrast + mean_right

        # ========== 感測器噪聲 (在曝光/增益之後) ==========

        # 2. 模擬感測器噪聲
        if np.random.rand() < 0.5:
            # Shot noise (signal-dependent, Poisson 高斯近似)
            # std = sqrt(signal * gain), gain 控制噪聲強度
            shot_gain = np.random.uniform(0.0005, 0.005)
            left_std = np.sqrt(np.maximum(np.abs(left) * shot_gain, 1e-10))
            right_std = np.sqrt(np.maximum(np.abs(right) * shot_gain, 1e-10))
            left = left + np.random.normal(0, left_std)
            right = right + np.random.normal(0, right_std)

            # Read noise (signal-independent, 讀出電路熱噪聲)
            read_std = np.random.uniform(0.002, 0.01)
            left = left + np.random.normal(0, read_std, left.shape).astype(np.float32)
            right = right + np.random.normal(0, read_std, right.shape).astype(np.float32)

        # ========== 幾何增強 ==========

        # 隨機垂直翻轉
        if np.random.rand() < 0.5:
            left = np.flip(left, axis=0).copy()
            right = np.flip(right, axis=0).copy()
            disparity = np.flip(disparity, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
            mask_strict = np.flip(mask_strict, axis=0).copy()

        return left, right, disparity, mask, mask_strict

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene_name = self.scenes[idx]
        data = self._load_scene(scene_name)

        left = data['left']
        right = data['right']
        disparity = data['disparity']
        mask = data['glass_mask']
        mask_strict = data['glass_mask_strict']

        # 正規化圖像
        left = self._normalize_image(left)
        right = self._normalize_image(right)

        # 數據增強 (場景級 + Sensor Realism，不做 crop 以保持全局上下文)
        if self.augment:
            left, right, disparity, mask, mask_strict = self._augment(
                left, right, disparity, mask, mask_strict, scene_name=scene_name
            )

        # 確保範圍正確
        left = np.clip(left, 0, 1)
        right = np.clip(right, 0, 1)

        # 轉換為 tensor (C, H, W) - 使用 .copy() 確保 array 可寫
        left_tensor = torch.from_numpy(left.copy()).permute(2, 0, 1).float()
        right_tensor = torch.from_numpy(right.copy()).permute(2, 0, 1).float()
        disparity_tensor = torch.from_numpy(disparity.copy()).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask.copy()).float().unsqueeze(0)
        mask_strict_tensor = torch.from_numpy(mask_strict.copy()).float().unsqueeze(0)

        # 創建有效深度遮罩
        valid_mask = (disparity_tensor > 0) & (disparity_tensor < self.max_disparity)

        # 獲取數據類型 (pol/nopol)
        data_type = self.scene_naming.get(scene_name, 'pol')

        return {
            'left': left_tensor,
            'right': right_tensor,
            'disparity': disparity_tensor,
            'glass_mask': mask_tensor,              # 聯集 mask (用於訓練)
            'glass_mask_strict': mask_strict_tensor, # 交集 mask (用於評估)
            'valid_mask': valid_mask.float(),
            'scene_name': scene_name,
            'data_type': data_type,                 # 'pol' 或 'nopol'
        }


def create_data_loaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    val_split: float = 0.2,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    創建訓練和驗證 DataLoader

    注意：不使用 crop，保持全圖輸入以維持全局幾何上下文 (符合 PIDS 論文設計)

    Args:
        data_dir: 數據目錄
        batch_size: Batch 大小
        num_workers: 數據載入 worker 數量
        val_split: 驗證集比例 (預設 0.2 = 80/20 分割)

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = PIDSSyntheticDataset(
        data_dir=data_dir,
        split='train',
        augment=True,
        val_split=val_split,
    )

    val_dataset = PIDSSyntheticDataset(
        data_dir=data_dir,
        split='val',
        augment=False,
        val_split=val_split,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # 測試數據集
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    dataset = PIDSSyntheticDataset(args.data_dir, split='train')
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Left shape: {sample['left'].shape}")
    print(f"Right shape: {sample['right'].shape}")
    print(f"Disparity shape: {sample['disparity'].shape}")
    print(f"Disparity range: [{sample['disparity'].min():.2f}, {sample['disparity'].max():.2f}]")

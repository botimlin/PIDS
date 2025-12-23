"""
PIDS Synthetic Dataset Loader
用於載入 Mitsuba 渲染的合成偏振立體數據
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
        max_disparity: float = 192.0,
        exclude_failed: bool = True,
        failed_scenes: Optional[List[str]] = None,
        augment: bool = True,
        crop_size: Tuple[int, int] = (320, 480),  # (H, W)
    ):
        """
        Args:
            data_dir: 數據目錄路徑 (可以是 dataset/ 或 stereo_pairs/)
            split: 'train' 或 'val'
            transform: 額外的數據變換
            max_disparity: 最大視差值（用於正規化）
            exclude_failed: 是否排除品質檢測未通過的場景
            failed_scenes: 未通過場景列表
            augment: 是否進行數據增強
            crop_size: 隨機裁切大小 (H, W)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.max_disparity = max_disparity
        self.augment = augment and (split == 'train')
        self.crop_size = crop_size

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

        # 分割訓練/驗證集 (90/10)
        np.random.seed(42)
        indices = np.random.permutation(len(self.scenes))
        split_idx = int(len(indices) * 0.9)

        if split == 'train':
            self.scenes = [self.scenes[i] for i in indices[:split_idx]]
        else:
            self.scenes = [self.scenes[i] for i in indices[split_idx:]]

        print(f"[PIDSDataset] {split} split: {len(self.scenes)} scenes")
        print(f"[PIDSDataset] Directory structure: {self.dir_structure}")

    def _detect_directory_structure(self):
        """自動偵測目錄結構"""
        # 檢查是否為整理後的結構
        if self.data_dir.name == 'stereo_pairs':
            # 傳入的是 stereo_pairs 目錄
            self.stereo_dir = self.data_dir
            self.depth_dir = self.data_dir.parent / 'ground_truth'
            self.mask_dir = self.data_dir.parent / 'masks'
            self.dir_structure = 'organized'
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

        # 查找所有 left_parallel.exr 檔案
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

        scenes.sort()
        return scenes

    def __len__(self) -> int:
        return len(self.scenes)

    def _load_scene(self, scene_name: str) -> Dict[str, np.ndarray]:
        """載入單一場景的所有數據"""
        # 載入左右圖像 (從 stereo_dir)
        left_path = self.stereo_dir / f"{scene_name}_left_parallel.exr"
        right_path = self.stereo_dir / f"{scene_name}_right_cross.exr"

        # 載入視差圖 (從 depth_dir)
        disp_path = self.depth_dir / f"{scene_name}_disparity.exr"

        # 載入遮罩 (從 mask_dir)
        mask_path = self.mask_dir / f"{scene_name}_glass_mask.exr"

        left = EXRReader.read_rgb(str(left_path))
        right = EXRReader.read_rgb(str(right_path))
        disparity = EXRReader.read_depth(str(disp_path))

        # 載入玻璃遮罩（如果存在）
        if mask_path.exists():
            mask = EXRReader.read_depth(str(mask_path))
        else:
            mask = np.zeros_like(disparity)

        return {
            'left': left,
            'right': right,
            'disparity': disparity,
            'glass_mask': mask,
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

    def _random_crop(
        self,
        left: np.ndarray,
        right: np.ndarray,
        disparity: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """隨機裁切"""
        h, w = left.shape[:2]
        crop_h, crop_w = self.crop_size

        if h < crop_h or w < crop_w:
            # 如果圖像太小，直接 resize
            return left, right, disparity, mask

        # 隨機選擇裁切位置
        y = np.random.randint(0, h - crop_h + 1)
        x = np.random.randint(0, w - crop_w + 1)

        left = left[y:y+crop_h, x:x+crop_w]
        right = right[y:y+crop_h, x:x+crop_w]
        disparity = disparity[y:y+crop_h, x:x+crop_w]
        mask = mask[y:y+crop_h, x:x+crop_w]

        return left, right, disparity, mask

    def _augment(
        self,
        left: np.ndarray,
        right: np.ndarray,
        disparity: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """數據增強"""
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

        # 隨機垂直翻轉
        if np.random.rand() < 0.5:
            left = np.flip(left, axis=0).copy()
            right = np.flip(right, axis=0).copy()
            disparity = np.flip(disparity, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        return left, right, disparity, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene_name = self.scenes[idx]
        data = self._load_scene(scene_name)

        left = data['left']
        right = data['right']
        disparity = data['disparity']
        mask = data['glass_mask']

        # 正規化圖像
        left = self._normalize_image(left)
        right = self._normalize_image(right)

        # 數據增強
        if self.augment:
            left, right, disparity, mask = self._random_crop(left, right, disparity, mask)
            left, right, disparity, mask = self._augment(left, right, disparity, mask)

        # 確保範圍正確
        left = np.clip(left, 0, 1)
        right = np.clip(right, 0, 1)

        # 轉換為 tensor (C, H, W)
        left_tensor = torch.from_numpy(left).permute(2, 0, 1).float()
        right_tensor = torch.from_numpy(right).permute(2, 0, 1).float()
        disparity_tensor = torch.from_numpy(disparity).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        # 創建有效深度遮罩
        valid_mask = (disparity_tensor > 0) & (disparity_tensor < self.max_disparity)

        return {
            'left': left_tensor,
            'right': right_tensor,
            'disparity': disparity_tensor,
            'glass_mask': mask_tensor,
            'valid_mask': valid_mask.float(),
            'scene_name': scene_name,
        }


def create_data_loaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    crop_size: Tuple[int, int] = (320, 480),
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    創建訓練和驗證 DataLoader

    Args:
        data_dir: 數據目錄
        batch_size: Batch 大小
        num_workers: 數據載入 worker 數量
        crop_size: 裁切大小

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = PIDSSyntheticDataset(
        data_dir=data_dir,
        split='train',
        augment=True,
        crop_size=crop_size,
    )

    val_dataset = PIDSSyntheticDataset(
        data_dir=data_dir,
        split='val',
        augment=False,
        crop_size=crop_size,
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

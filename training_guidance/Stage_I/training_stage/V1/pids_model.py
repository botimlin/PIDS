"""
PIDS Stereo Model - Based on RAFT-Stereo Architecture
用於偏振立體匹配的深度學習模型

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class ResidualBlock(nn.Module):
    """殘差塊"""

    def __init__(self, in_planes: int, planes: int, norm_fn: str = 'batch', stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        if norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
        else:
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes) if norm_fn == 'batch' else nn.Sequential()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class FeatureEncoder(nn.Module):
    """
    特徵編碼器
    提取 1/4 解析度的特徵圖
    """

    def __init__(self, output_dim: int = 128, norm_fn: str = 'batch', input_dim: int = 3):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.BatchNorm2d(64) if norm_fn == 'batch' else nn.InstanceNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        # 殘差層
        self.layer1 = self._make_layer(64, 64, 2, norm_fn, stride=1)
        self.layer2 = self._make_layer(64, 96, 2, norm_fn, stride=2)
        self.layer3 = self._make_layer(96, 128, 2, norm_fn, stride=1)

        # 輸出投影
        self.conv_out = nn.Conv2d(128, output_dim, kernel_size=1)

    def _make_layer(self, in_planes: int, planes: int, num_blocks: int,
                    norm_fn: str, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_planes, planes, norm_fn, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(planes, planes, norm_fn))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv_out(x)
        return x


class ContextEncoder(nn.Module):
    """
    上下文編碼器
    提取上下文特徵和隱藏狀態初始化
    """

    def __init__(self, output_dim: int = 128, hidden_dim: int = 128):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 96, 2, stride=2)
        self.layer3 = self._make_layer(96, 128, 2, stride=1)

        # 輸出上下文和隱藏狀態
        self.conv_ctx = nn.Conv2d(128, output_dim, kernel_size=1)
        self.conv_hidden = nn.Conv2d(128, hidden_dim, kernel_size=1)

    def _make_layer(self, in_planes: int, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_planes, planes, 'batch', stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(planes, planes, 'batch'))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        ctx = self.conv_ctx(x)
        hidden = torch.tanh(self.conv_hidden(x))

        return ctx, hidden


class PolCorrBlock:
    """
    Polarization Volume - 偏振差異體積計算和查詢

    與 CorrBlock 類似，但計算偏振差異而非特徵相關性：
    - CorrBlock:    corr[x, d] = dot(fmap_left[x], fmap_right[x-d])
    - PolCorrBlock: pol[x, d]  = img_left[x] - img_right[x-d]

    優勢：
    - 預先計算所有 disparity 的 pol_diff
    - 不依賴預測的 disparity 來構建 volume
    - Oracle 和 Real 完全相同，沒有 gap！
    """

    def __init__(self, img_left: torch.Tensor, img_right: torch.Tensor,
                 num_levels: int = 4, radius: int = 4):
        """
        Args:
            img_left: (B, C, H, W) 左圖偏振影像 (已降到 1/4 解析度)
            img_right: (B, C, H, W) 右圖偏振影像 (已降到 1/4 解析度)
            num_levels: 金字塔層數
            radius: 查詢半徑
        """
        self.num_levels = num_levels
        self.radius = radius

        # 計算偏振差異金字塔
        self.pol_pyramid = []

        # 計算全分辨率偏振差異體積
        pol_volume = self._compute_pol_volume(img_left, img_right)
        self.pol_pyramid.append(pol_volume)

        # 構建金字塔
        for _ in range(num_levels - 1):
            pol_volume = F.avg_pool2d(pol_volume, kernel_size=2, stride=2)
            self.pol_pyramid.append(pol_volume)

    def _compute_pol_volume(self, img_left: torch.Tensor, img_right: torch.Tensor) -> torch.Tensor:
        """
        計算偏振差異體積

        pol_volume[b, h, x_left, x_right] = img_left[b, :, h, x_left] - img_right[b, :, h, x_right]

        對於 stereo matching: disparity d = x_left - x_right
        所以 pol_diff at disparity d = pol_volume[b, h, x, x-d]
        """
        batch, channels, h, w = img_left.shape

        # 平均通道 (如果是 RGB，取平均；如果是灰度，不變)
        left_gray = img_left.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        right_gray = img_right.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # 擴展維度計算所有組合
        # left: (B, 1, H, W, 1)
        # right: (B, 1, H, 1, W)
        left_exp = left_gray.unsqueeze(-1)
        right_exp = right_gray.unsqueeze(-2)

        # pol_volume[b, 1, h, x_left, x_right] = left[x_left] - right[x_right]
        pol_volume = left_exp - right_exp  # (B, 1, H, W, W)

        # 重塑為 (B*H, 1, W, W) 與 CorrBlock 格式一致
        pol_volume = pol_volume.squeeze(1)  # (B, H, W, W)
        pol_volume = pol_volume.reshape(batch * h, 1, w, w)

        return pol_volume

    def __call__(self, disp: torch.Tensor) -> torch.Tensor:
        """
        從偏振金字塔中查詢

        Args:
            disp: (B, 1, H, W) 視差估計 (在 1/4 解析度)

        Returns:
            (B, C, H, W) 偏振特徵，C = num_levels * (2*radius+1)
        """
        batch_h, _, w, _ = self.pol_pyramid[0].shape
        batch = disp.shape[0]
        h = batch_h // batch

        disp = disp.squeeze(1).view(batch * h, w)

        out_pyramid = []
        for i, pol_vol in enumerate(self.pol_pyramid):
            _, _, w_pol, _ = pol_vol.shape

            scale = 1 / (2 ** i)
            disp_scaled = scale * disp

            dx = torch.linspace(-self.radius, self.radius, 2 * self.radius + 1, device=disp.device)

            x_left = torch.arange(w, device=disp.device, dtype=disp.dtype).view(1, w, 1).expand(batch * h, -1, -1)
            x_right = x_left - disp_scaled.unsqueeze(-1) + dx.view(1, 1, -1)

            x_norm = 2 * x_right / (w_pol - 1) - 1
            y_norm = 2 * x_left / (w_pol - 1) - 1
            y_norm = y_norm.expand(-1, -1, 2 * self.radius + 1)

            grid = torch.stack([x_norm, y_norm], dim=-1)

            pol_sampled = F.grid_sample(pol_vol, grid, align_corners=True, mode='bilinear', padding_mode='zeros')

            out_pyramid.append(pol_sampled.view(batch, h, w, -1))

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2)  # (B, C, H, W)


class CorrBlock:
    """
    相關性體積計算和查詢
    """

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor,
                 num_levels: int = 4, radius: int = 4):
        self.num_levels = num_levels
        self.radius = radius

        # 計算相關性金字塔
        self.corr_pyramid = []

        # 計算全分辨率相關性 (只計算水平方向)
        corr = self._compute_correlation(fmap1, fmap2)
        self.corr_pyramid.append(corr)

        # 構建金字塔
        for _ in range(num_levels - 1):
            corr = F.avg_pool2d(corr, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr)

    def _compute_correlation(self, fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        """計算立體相關性 (只沿水平方向)"""
        batch, dim, h, w = fmap1.shape

        fmap1 = fmap1.view(batch, dim, h, w)
        fmap2 = fmap2.view(batch, dim, h, w)

        # 正規化
        fmap1 = fmap1 / (torch.norm(fmap1, dim=1, keepdim=True) + 1e-6)
        fmap2 = fmap2 / (torch.norm(fmap2, dim=1, keepdim=True) + 1e-6)

        # 計算相關性 (batch, h, w, w) - 每個左圖像素與右圖同一行所有像素的相關性
        corr = torch.einsum('bchw,bchx->bhwx', fmap1, fmap2)

        return corr.reshape(batch * h, 1, w, w)

    def __call__(self, disp: torch.Tensor) -> torch.Tensor:
        """從相關性金字塔中查詢

        Args:
            disp: (B, 1, H, W) 視差估計 (在 1/4 解析度)

        Returns:
            (B, C, H, W) 相關性特徵，C = num_levels * (2*radius+1)
        """
        batch_h, _, w, _ = self.corr_pyramid[0].shape
        batch = disp.shape[0]
        h = batch_h // batch

        # disp: (B, 1, H, W) -> (B*H, W)
        disp = disp.squeeze(1).view(batch * h, w)

        out_pyramid = []
        for i, corr in enumerate(self.corr_pyramid):
            # corr shape: (B*H, 1, W_left, W_right)
            _, _, w_corr, _ = corr.shape

            # 縮放視差到當前金字塔層級
            scale = 1 / (2 ** i)
            disp_scaled = scale * disp  # (B*H, W)

            # 採樣偏移量
            dx = torch.linspace(-self.radius, self.radius, 2 * self.radius + 1, device=disp.device)

            # 計算採樣位置
            # x_left: 左圖像素位置 (B*H, W, 1)
            x_left = torch.arange(w, device=disp.device, dtype=disp.dtype).view(1, w, 1).expand(batch * h, -1, -1)

            # x_right: 右圖對應位置 = x_left - disp + dx
            # (B*H, W, 2*r+1)
            x_right = x_left - disp_scaled.unsqueeze(-1) + dx.view(1, 1, -1)

            # 正規化到 [-1, 1]
            # grid_sample: grid[..., 0] 是 x (沿 dim 3)，grid[..., 1] 是 y (沿 dim 2)
            x_norm = 2 * x_right / (w_corr - 1) - 1  # 採樣 W_right 維度
            y_norm = 2 * x_left / (w_corr - 1) - 1   # 採樣 W_left 維度
            y_norm = y_norm.expand(-1, -1, 2 * self.radius + 1)

            # grid: (B*H, W, 2*r+1, 2)
            grid = torch.stack([x_norm, y_norm], dim=-1)

            # 雙線性採樣
            corr_sampled = F.grid_sample(corr, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
            # corr_sampled: (B*H, 1, W, 2*r+1)

            out_pyramid.append(corr_sampled.view(batch, h, w, -1))

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2)  # (B, C, H, W)


class ConvGRU(nn.Module):
    """卷積 GRU 單元"""

    def __init__(self, hidden_dim: int, input_dim: int):
        super().__init__()

        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class DispHead(nn.Module):
    """視差預測頭"""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x)))


class UpdateBlock(nn.Module):
    """迭代更新塊"""

    def __init__(self, hidden_dim: int = 128, context_dim: int = 128, corr_dim: int = 196):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(corr_dim + 1, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.gru = ConvGRU(hidden_dim, 128 + context_dim)
        self.disp_head = DispHead(hidden_dim)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor,
                corr: torch.Tensor, disp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 編碼相關性和當前視差
        motion_features = self.encoder(torch.cat([corr, disp], dim=1))

        # GRU 更新
        inp = torch.cat([motion_features, context], dim=1)
        hidden = self.gru(hidden, inp)

        # 預測視差更新量
        delta_disp = self.disp_head(hidden)

        return hidden, delta_disp


class PIDSStereo(nn.Module):
    """
    PIDS 立體匹配網絡

    基於 RAFT-Stereo 架構，針對偏振立體對進行優化
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        feature_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        iters: int = 12,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.iters = iters
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius

        # 相關性維度
        self.corr_dim = corr_levels * (2 * corr_radius + 1)

        # 編碼器
        self.fnet = FeatureEncoder(output_dim=feature_dim)
        self.cnet = ContextEncoder(output_dim=context_dim, hidden_dim=hidden_dim)

        # 更新塊
        self.update_block = UpdateBlock(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_dim=self.corr_dim
        )

    def freeze_bn(self):
        """凍結 BatchNorm 層"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters: Optional[int] = None,
        test_mode: bool = False
    ) -> List[torch.Tensor]:
        """
        Args:
            left: 左圖像 (B, 3, H, W)
            right: 右圖像 (B, 3, H, W)
            iters: 迭代次數（覆蓋預設值）
            test_mode: 是否為測試模式（只返回最終預測）

        Returns:
            視差預測列表 (每次迭代的預測)
        """
        if iters is None:
            iters = self.iters

        # 特徵提取 (1/4 解析度)
        fmap1 = self.fnet(left)
        fmap2 = self.fnet(right)

        # 上下文提取
        context, hidden = self.cnet(left)

        # 構建相關性體積
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # 初始化視差 (從零開始)
        b, _, h, w = fmap1.shape
        disp = torch.zeros(b, 1, h, w, device=left.device)

        disp_predictions = []

        for _ in range(iters):
            disp = disp.detach()

            # 查詢相關性
            corr = corr_fn(disp)

            # 更新
            hidden, delta_disp = self.update_block(hidden, context, corr, disp)

            # 更新視差
            disp = disp + delta_disp

            # 上採樣到原始解析度
            disp_up = 4 * F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)

            disp_predictions.append(disp_up)

        if test_mode:
            return disp_predictions[-1]

        return disp_predictions


# =============================================================================
# Dual-Stream Polarization Architecture
# =============================================================================

class SpatialAttention(nn.Module):
    """空間注意力模組 - 學習哪些區域的偏振信號更重要"""

    def __init__(self, in_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 attention-weighted 特徵"""
        attn = self.conv(x)  # (B, 1, H, W)
        return x * attn, attn


class ResidualBlock2D(nn.Module):
    """殘差塊 - 更好的梯度流動"""

    def __init__(self, in_dim: int, out_dim: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection
        self.skip = nn.Identity()
        if stride != 1 or in_dim != out_dim:
            self.skip = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


def warp_with_disparity(
    img: torch.Tensor,
    disparity: torch.Tensor,
    return_valid_mask: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    使用視差圖將右圖 warp 到左圖視角

    基於已知相機參數:
    - Baseline: 65mm
    - Focal: 765px (計算值)
    - 視差公式: disparity = baseline * focal / depth

    Args:
        img: (B, C, H, W) - 右圖
        disparity: (B, 1, H, W) - GT 視差圖 (像素單位)
        return_valid_mask: 是否返回有效區域 mask

    Returns:
        warped: (B, C, H, W) - warp 後的右圖 (對齊到左圖視角)
        valid_mask: (B, 1, H, W) - 有效區域 mask (超出邊界為 0)
    """
    B, C, H, W = img.shape

    # 建立座標網格
    y_coords = torch.arange(H, device=img.device, dtype=torch.float32)
    x_coords = torch.arange(W, device=img.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # 擴展到 batch
    xx = xx.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    yy = yy.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)

    # 視差偏移: 右圖中對應點在 x - disparity 位置
    # disparity shape: (B, 1, H, W) -> (B, H, W)
    disp = disparity.squeeze(1)
    xx_warped = xx - disp  # 右圖的 x 座標

    # 計算有效區域 mask (採樣座標在圖像範圍內)
    valid_mask = None
    if return_valid_mask:
        valid_mask = (xx_warped >= 0) & (xx_warped <= W - 1)
        valid_mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)

    # 正規化到 [-1, 1] (grid_sample 需要)
    xx_norm = 2.0 * xx_warped / (W - 1) - 1.0
    yy_norm = 2.0 * yy / (H - 1) - 1.0

    # 組合成 grid (B, H, W, 2)
    grid = torch.stack([xx_norm, yy_norm], dim=-1)

    # Warp
    warped = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return warped, valid_mask


class PolarizationEncoder(nn.Module):
    """
    強化版偏振特徵提取器 (支援視差對齊)

    改進:
    1. 殘差連接 - 更好的梯度流動
    2. 更大維度 - 更豐富的特徵表達
    3. 空間注意力 - 學習哪裡重要
    4. 視差對齊 - 使用 GT disparity warp 右圖後再計算 pol_diff
    5. 輸出 attention map - 可視化 & 用於 loss
    """

    def __init__(self, out_dim: int = 64, threshold: float = 0.05, sharpness: float = 20.0):
        """
        Args:
            out_dim: 輸出特徵維度 (建議 64)
            threshold: 偏振差異門檻 (0-1)，降低到 0.05 捕捉更弱信號
            sharpness: Soft threshold 銳利度
        """
        super().__init__()

        self.threshold = threshold
        self.sharpness = sharpness
        self.out_dim = out_dim

        # Stage 1: 初始特徵提取 (H, W) → (H/2, W/2)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 2: 殘差塊 (H/2, W/2) → (H/4, W/4)
        self.res1 = ResidualBlock2D(32, 64, stride=2)
        self.res2 = ResidualBlock2D(64, 128, stride=1)

        # Stage 3: 空間注意力
        self.spatial_attn = SpatialAttention(128)

        # Stage 4: 輸出投影
        self.out_conv = nn.Sequential(
            nn.Conv2d(128, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        # 保存 attention map 供 loss 使用
        self.last_attn_map = None
        self.last_pol_diff = None
        self.last_valid_mask = None

    def compute_pol_diff(self, left: torch.Tensor, right: torch.Tensor,
                         disparity: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        計算偏振差異圖 (Soft Threshold)

        重要: 使用 GT disparity 對齊後再計算，確保比較同一 3D 點

        相機參數:
        - Baseline: 65mm
        - Focal: ~765px
        - 視差範圍: 55-94px (對應深度 900-528mm)

        Args:
            left: (B, 3, H, W) - I∥ (平行偏振，左相機)
            right: (B, 3, H, W) - I⊥ (交叉偏振，右相機)
            disparity: (B, 1, H, W) - GT 視差圖 (可選，用於對齊)

        Returns:
            pol_diff: (B, 1, H, W) - 過濾後的偏振差異圖
            valid_mask: (B, 1, H, W) - 有效區域 mask (warp 後超出邊界為 0)
        """
        valid_mask = None

        # 如果有 GT disparity，先 warp 右圖對齊到左圖視角
        if disparity is not None:
            right_aligned, valid_mask = warp_with_disparity(right, disparity, return_valid_mask=True)
        else:
            # 無 disparity 時退化為原始方式 (inference 時)
            right_aligned = right

        # 計算 RGB 平均差異 (現在是比較同一 3D 點的偏振差異)
        pol_diff_raw = torch.abs(left - right_aligned).mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # 將無效區域設為 0（避免邊界假信號）
        if valid_mask is not None:
            pol_diff_raw = pol_diff_raw * valid_mask

        # 正規化到 0-1（只考慮有效區域的最大值）
        B = pol_diff_raw.shape[0]
        if valid_mask is not None:
            # 有效區域的最大值
            pol_max = (pol_diff_raw * valid_mask).view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        else:
            pol_max = pol_diff_raw.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        pol_diff = pol_diff_raw / (pol_max + 1e-6)

        # Soft threshold: sigmoid 平滑過渡
        weight = torch.sigmoid(self.sharpness * (pol_diff - self.threshold))
        pol_diff = pol_diff * weight

        # 再次 mask 確保無效區域為 0
        if valid_mask is not None:
            pol_diff = pol_diff * valid_mask

        # 保存供 loss 使用
        self.last_pol_diff = pol_diff.detach()
        self.last_valid_mask = valid_mask.detach() if valid_mask is not None else None

        return pol_diff, valid_mask

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                disparity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            left: (B, 3, H, W) - I∥
            right: (B, 3, H, W) - I⊥
            disparity: (B, 1, H, W) - GT 視差 (訓練時提供，推論時為 None)

        Returns:
            pol_feat: (B, out_dim, H/4, W/4) - 偏振特徵
        """
        pol_diff, valid_mask = self.compute_pol_diff(left, right, disparity)

        # 編碼
        x = self.stem(pol_diff)      # (B, 32, H/2, W/2)
        x = self.res1(x)             # (B, 64, H/4, W/4)
        x = self.res2(x)             # (B, 128, H/4, W/4)

        # 空間注意力
        x, attn = self.spatial_attn(x)  # (B, 128, H/4, W/4), (B, 1, H/4, W/4)
        self.last_attn_map = attn.detach()

        # 輸出投影
        out = self.out_conv(x)       # (B, out_dim, H/4, W/4)

        return out

    def get_attention_map(self) -> Optional[torch.Tensor]:
        """獲取最後一次 forward 的 attention map (用於可視化或 loss)"""
        return self.last_attn_map

    def get_pol_diff(self) -> Optional[torch.Tensor]:
        """獲取最後一次 forward 的偏振差異圖"""
        return self.last_pol_diff

    def get_valid_mask(self) -> Optional[torch.Tensor]:
        """獲取最後一次 forward 的有效區域 mask (warp 邊界外為 0)"""
        return self.last_valid_mask


class FeatureFusion(nn.Module):
    """
    融合 stereo features 和 polarization features
    """

    def __init__(self, stereo_dim: int = 128, pol_dim: int = 32, out_dim: int = 128):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(stereo_dim + pol_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, stereo_feat: torch.Tensor, pol_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stereo_feat: (B, stereo_dim, H/4, W/4)
            pol_feat: (B, pol_dim, H/4, W/4)

        Returns:
            fused: (B, out_dim, H/4, W/4)
        """
        combined = torch.cat([stereo_feat, pol_feat], dim=1)
        return self.fusion(combined)


class PIDSStereoDualStream(nn.Module):
    """
    PIDS Dual-Stream 立體匹配網絡 (自包含版本)

    獨立實現，不依賴 RAFT-Stereo 的內部 API。
    加入專門的 Polarization Encoder 來明確利用偏振信息。

    架構:
        left ──→ [Feature Encoder] ──→ fmap1 ─┐
                                               ├──→ [Fusion] ──→ [Corr + GRU] ──→ disparity
        right ─→ [Feature Encoder] ──→ fmap2 ─┤
                                               │
        |left-right| ─→ [Pol Encoder] ────────┘

    特點:
    - 自包含實現，可獨立運行
    - pol_encoder 和 fusion 是新層，其他層結構類似 RAFT-Stereo
    - 可部分載入 RAFT-Stereo 預訓練權重到 fnet/cnet/update_block
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        feature_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        iters: int = 12,
        pol_dim: int = 64,           # 提高: 32 → 64
        pol_threshold: float = 0.05,  # 降低: 0.1 → 0.05 (捕捉更弱信號)
        pol_sharpness: float = 20.0,
        mixed_precision: bool = False,
        # Training-Inference Consistency 參數
        disparity_noise_std: float = 2.0,  # 雜訊標準差 (像素)
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.iters = iters
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.mixed_precision = mixed_precision
        self.disparity_noise_std = disparity_noise_std

        # 相關性維度
        self.corr_dim = corr_levels * (2 * corr_radius + 1)

        # Stereo 編碼器 (類似 RAFT-Stereo 的 fnet)
        self.fnet = FeatureEncoder(output_dim=feature_dim)

        # Context 編碼器 (類似 RAFT-Stereo 的 cnet)
        self.cnet = ContextEncoder(output_dim=context_dim, hidden_dim=hidden_dim)

        # 新增: 強化版偏振編碼器 (殘差連接 + 空間注意力)
        self.pol_encoder = PolarizationEncoder(
            out_dim=pol_dim,
            threshold=pol_threshold,
            sharpness=pol_sharpness,
        )

        # 新增: 融合層 (將偏振特徵融入 stereo 特徵)
        self.fusion = FeatureFusion(
            stereo_dim=feature_dim,
            pol_dim=pol_dim,
            out_dim=feature_dim,  # 保持維度不變
        )

        # 更新塊 (類似 RAFT-Stereo 的 update_block)
        self.update_block = UpdateBlock(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_dim=self.corr_dim
        )

    def freeze_bn(self):
        """凍結 BatchNorm 層"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_param_groups(self, base_lr: float = 1e-5, new_lr: float = 1e-4) -> List[dict]:
        """
        返回 optimizer 用的 param groups

        原始層用小 lr (保護預訓練權重)
        新增層 (pol_encoder, fusion) 用正常 lr
        """
        base_params = []
        new_params = []

        for name, param in self.named_parameters():
            if 'pol_encoder' in name or 'fusion' in name:
                new_params.append(param)
            else:
                base_params.append(param)

        return [
            {'params': base_params, 'lr': base_lr},
            {'params': new_params, 'lr': new_lr},
        ]

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters: Optional[int] = None,
        test_mode: bool = False,
        disparity_gt: Optional[torch.Tensor] = None,
        noise_ratio: float = 0.0,
    ) -> List[torch.Tensor]:
        """
        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            iters: GRU 迭代次數 (覆蓋預設值)
            test_mode: 是否為測試模式
            disparity_gt: GT 視差 (B, 1, H, W) - 訓練時用於對齊 pol_diff 計算
            noise_ratio: 加入雜訊的機率 (0.0-1.0)，用於 Training-Inference Consistency
                         Curriculum Learning: 訓練初期=0，後期逐漸增加到1

        Returns:
            flow_predictions: List[Tensor] - 每次迭代的 flow 預測 (H-方向 flow，用於 stereo)
        """
        if iters is None:
            iters = self.iters

        # ============ Training-Inference Consistency ============
        # 訓練時以 noise_ratio 機率對 disparity_gt 加入雜訊
        # 模擬推論時使用 predicted disparity 的誤差
        disparity_for_pol = disparity_gt
        if self.training and disparity_gt is not None and noise_ratio > 0:
            if torch.rand(1).item() < noise_ratio:
                # 加入高斯雜訊，標準差 = disparity_noise_std 像素
                noise = torch.randn_like(disparity_gt) * self.disparity_noise_std
                disparity_for_pol = disparity_gt + noise

        # 1. 提取偏振特徵 (使用可能加了雜訊的 disparity 對齊)
        pol_feat = self.pol_encoder(left, right, disparity_for_pol)  # (B, pol_dim, H/4, W/4)

        # 2. 提取 stereo 特徵 (1/4 解析度)
        fmap1 = self.fnet(left)   # (B, feature_dim, H/4, W/4)
        fmap2 = self.fnet(right)  # (B, feature_dim, H/4, W/4)

        # 3. 融合偏振特徵到 stereo 特徵
        fmap1 = self.fusion(fmap1, pol_feat)
        fmap2 = self.fusion(fmap2, pol_feat)

        # 4. 提取 context 和 hidden state
        context, hidden = self.cnet(left)

        # 5. 構建相關性體積
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # 6. 初始化視差 (從零開始，在 1/4 解析度)
        b, _, h, w = fmap1.shape
        disp = torch.zeros(b, 1, h, w, device=left.device)

        flow_predictions = []

        for _ in range(iters):
            disp = disp.detach()

            # 查詢相關性
            corr = corr_fn(disp)

            # 更新
            hidden, delta_disp = self.update_block(hidden, context, corr, disp)

            # 更新視差
            disp = disp + delta_disp

            # 上採樣到原始解析度 (4x)
            disp_up = 4 * F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)

            # RAFT-Stereo 格式: 返回 flow (負值表示向左移動 = 正視差)
            # flow shape: (B, 2, H, W)，但 stereo 只用 x-direction
            flow_up = torch.cat([disp_up, torch.zeros_like(disp_up)], dim=1)  # (B, 2, H, W)
            flow_predictions.append(flow_up)

        if test_mode:
            return flow_predictions[-1]

        return flow_predictions

    def get_pol_diff(self) -> Optional[torch.Tensor]:
        """獲取最後一次 forward 的偏振差異圖 (用於 Polarization-aware Loss)"""
        return self.pol_encoder.get_pol_diff()

    def get_attention_map(self) -> Optional[torch.Tensor]:
        """獲取最後一次 forward 的 attention map (用於可視化)"""
        return self.pol_encoder.get_attention_map()

    def forward_inference(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters: Optional[int] = None,
        pol_update_iters: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        兩階段推論方法 (解決 65mm baseline 對齊問題)

        問題:
        - 訓練時使用 GT disparity 做 warp，計算正確的偏振差異
        - 推論時沒有 GT，直接計算 |I∥ - I⊥| 會因為 65mm baseline 導致比較不同區域

        解決方案:
        - 先用未對齊的偏振特徵獲得初始 disparity 估計
        - 使用估計的 disparity 做 warp，重新計算對齊的偏振特徵
        - 繼續 GRU 迭代，用更準確的偏振特徵優化 disparity

        設計考量:
        - 只更新 1-2 次偏振特徵 (平衡準確度和效率)
        - fnet 只計算一次 (不重複計算 stereo features)
        - GRU hidden state 保持連續 (不重啟)

        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            iters: GRU 迭代次數
            pol_update_iters: 在哪些迭代後更新偏振特徵
                              預設 [iters//2] (中間更新一次)
                              例如 [4, 8] 表示在第4和第8次迭代後更新

        Returns:
            final_flow: (B, 2, H, W) 最終 disparity (x-component)
        """
        if iters is None:
            iters = self.iters

        # 預設: 在中間更新一次
        if pol_update_iters is None:
            pol_update_iters = [iters // 2]

        # ============ Stage 0: 初始化 ============
        # Stereo features (只計算一次)
        fmap1_base = self.fnet(left)   # (B, feature_dim, H/4, W/4)
        fmap2_base = self.fnet(right)  # (B, feature_dim, H/4, W/4)

        # 初始偏振特徵 (未對齊)
        pol_feat = self.pol_encoder(left, right, disparity=None)

        # 融合
        fmap1 = self.fusion(fmap1_base, pol_feat)
        fmap2 = self.fusion(fmap2_base, pol_feat)

        # Context 和 hidden state
        context, hidden = self.cnet(left)

        # Correlation volume
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # 初始化 disparity
        b, _, h, w = fmap1.shape
        disp = torch.zeros(b, 1, h, w, device=left.device)

        # ============ GRU 迭代 (帶偏振更新) ============
        for i in range(iters):
            disp = disp.detach()

            # 查詢相關性
            corr = corr_fn(disp)

            # GRU 更新
            hidden, delta_disp = self.update_block(hidden, context, corr, disp)

            # 更新 disparity
            disp = disp + delta_disp

            # ============ 偏振特徵更新點 ============
            if (i + 1) in pol_update_iters:
                # 上採樣當前 disparity 到原始解析度
                disp_up = 4 * F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)

                # 重新計算偏振特徵 (使用當前 disparity 估計做 warp)
                pol_feat = self.pol_encoder(left, right, disparity=disp_up)

                # 重新融合 (fmap1_base, fmap2_base 不變)
                fmap1 = self.fusion(fmap1_base, pol_feat)
                fmap2 = self.fusion(fmap2_base, pol_feat)

                # 重建 correlation volume
                corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # ============ 輸出 ============
        disp_up = 4 * F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)
        flow_up = torch.cat([disp_up, torch.zeros_like(disp_up)], dim=1)  # (B, 2, H, W)

        return flow_up

    def forward_two_pass(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters_pass1: int = 6,
        iters_pass2: int = 6,
    ) -> torch.Tensor:
        """
        完整兩階段推論 (更高精度，更高計算成本)

        Stage 1: 用未對齊 pol_diff → disparity_v1 (粗略)
        Stage 2: 用 disparity_v1 對齊 pol_diff → disparity_v2 (精確)

        注意: Pass 1 仍使用未對齊的 pol_diff，因為模型沒有訓練過 pol_feat=0 的情況。
        雖然 Pass 1 的結果不完美，但可以提供足夠好的初始對齊給 Pass 2。

        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            iters_pass1: 第一階段迭代次數
            iters_pass2: 第二階段迭代次數

        Returns:
            final_flow: (B, 2, H, W) 最終 disparity
        """
        # ============ Pass 1: 無偏振對齊 (粗略估計) ============
        flow_v1 = self.forward(left, right, iters=iters_pass1, test_mode=True, disparity_gt=None)
        disp_v1 = flow_v1[:, :1, :, :]  # 取 x-component

        # ============ Pass 2: 用估計的 disparity 對齊偏振 ============
        flow_v2 = self.forward(left, right, iters=iters_pass2, test_mode=True, disparity_gt=disp_v1)

        return flow_v2


# ============================================================================
# NEW ARCHITECTURE: Polarization Volume (No Oracle/Real Gap)
# ============================================================================

class UpdateBlockWithPol(nn.Module):
    """
    迭代更新塊 - 支援 Polarization Volume

    輸入: correlation features + polarization features + disparity
    """

    def __init__(self, hidden_dim: int = 128, context_dim: int = 128,
                 corr_dim: int = 36, pol_dim: int = 36):
        super().__init__()

        # 總輸入維度: corr + pol + disp
        input_dim = corr_dim + pol_dim + 1

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.gru = ConvGRU(hidden_dim, 128 + context_dim)
        self.disp_head = DispHead(hidden_dim)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor,
                corr: torch.Tensor, pol_corr: torch.Tensor,
                disp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: GRU hidden state
            context: Context features
            corr: Correlation lookup (B, corr_dim, H, W)
            pol_corr: Polarization lookup (B, pol_dim, H, W)
            disp: Current disparity estimate (B, 1, H, W)
        """
        # 編碼相關性 + 偏振 + 視差
        motion_features = self.encoder(torch.cat([corr, pol_corr, disp], dim=1))

        # GRU 更新
        inp = torch.cat([motion_features, context], dim=1)
        hidden = self.gru(hidden, inp)

        # 預測視差更新量
        delta_disp = self.disp_head(hidden)

        return hidden, delta_disp


# ============================================================================
# V2 ARCHITECTURE: Polarization Attention + Gated Fusion
# ============================================================================

class PolarizationAttention(nn.Module):
    """
    從偏振特徵生成空間注意力圖

    高偏振差異區域（玻璃）獲得更高權重
    讓模型知道「哪裡該信任偏振資訊」
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 8)

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, pol_corr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pol_corr: (B, C, H, W) 偏振特徵
        Returns:
            (B, 1, H, W) 空間注意力圖，值域 [0, 1]
        """
        return self.attention(pol_corr)


class GatedFusion(nn.Module):
    """
    學習 stereo 與 polarization 特徵的融合權重

    gate = 0 → 完全信任 pol
    gate = 1 → 完全信任 stereo

    在玻璃區域（pol_attn 高）自動降低 stereo 權重，提高 pol 權重
    """

    def __init__(self, corr_dim: int, pol_dim: int, out_dim: int = 128):
        super().__init__()

        # Gate network: 決定信任 stereo 還是 pol
        self.gate_net = nn.Sequential(
            nn.Conv2d(corr_dim + pol_dim + 1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 特徵增強: 將 corr 和 pol 投影到相同維度
        self.corr_enhance = nn.Sequential(
            nn.Conv2d(corr_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pol_enhance = nn.Sequential(
            nn.Conv2d(pol_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, corr: torch.Tensor, pol_corr: torch.Tensor,
                pol_attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            corr: (B, corr_dim, H, W) 相關性特徵
            pol_corr: (B, pol_dim, H, W) 偏振特徵
            pol_attn: (B, 1, H, W) 偏振注意力圖

        Returns:
            fused: (B, out_dim, H, W) 融合特徵
            gate: (B, 1, H, W) 融合權重 (用於視覺化/分析)
        """
        # 計算 gate
        gate_input = torch.cat([corr, pol_corr, pol_attn], dim=1)
        gate = self.gate_net(gate_input)

        # 增強特徵
        corr_feat = self.corr_enhance(corr)
        pol_feat = self.pol_enhance(pol_corr)

        # Gated fusion: 在高 pol_attn 區域（玻璃），gate 趨向 0，更信任 pol
        # 這裡我們希望 pol_attn 高時 gate 低，所以可以用 pol_attn 來調節
        # 但 gate_net 已經學習這個關係，所以直接用 gate 即可
        fused = gate * corr_feat + (1 - gate) * pol_feat

        return fused, gate


class UpdateBlockV2(nn.Module):
    """
    迭代更新塊 V2 - 支援 Polarization Attention + Gated Fusion

    改進：
    1. PolarizationAttention: 識別玻璃區域
    2. GatedFusion: 學習 stereo vs pol 的最佳融合
    """

    def __init__(self, hidden_dim: int = 128, context_dim: int = 128,
                 corr_dim: int = 36, pol_dim: int = 36, fused_dim: int = 128):
        super().__init__()

        self.pol_attention = PolarizationAttention(pol_dim)
        self.gated_fusion = GatedFusion(corr_dim, pol_dim, fused_dim)

        # 編碼器: fused features + disp
        self.encoder = nn.Sequential(
            nn.Conv2d(fused_dim + 1, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.gru = ConvGRU(hidden_dim, 128 + context_dim)
        self.disp_head = DispHead(hidden_dim)

    def freeze_pol_modules(self):
        """凍結 V2 特有的偏振模組 (PolarizationAttention, GatedFusion)"""
        for param in self.pol_attention.parameters():
            param.requires_grad = False
        for param in self.gated_fusion.parameters():
            param.requires_grad = False

    def unfreeze_pol_modules(self):
        """解凍 V2 特有的偏振模組"""
        for param in self.pol_attention.parameters():
            param.requires_grad = True
        for param in self.gated_fusion.parameters():
            param.requires_grad = True

    def forward(self, hidden: torch.Tensor, context: torch.Tensor,
                corr: torch.Tensor, pol_corr: torch.Tensor,
                disp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: GRU hidden state
            context: Context features
            corr: Correlation lookup (B, corr_dim, H, W)
            pol_corr: Polarization lookup (B, pol_dim, H, W)
            disp: Current disparity estimate (B, 1, H, W)

        Returns:
            hidden: Updated hidden state
            delta_disp: Disparity update
            gate: Fusion gate (for visualization)
        """
        # Step 1: 計算偏振注意力
        pol_attn = self.pol_attention(pol_corr)

        # Step 2: Gated fusion
        fused, gate = self.gated_fusion(corr, pol_corr, pol_attn)

        # Step 3: 編碼並更新
        motion_features = self.encoder(torch.cat([fused, disp], dim=1))

        # GRU 更新
        inp = torch.cat([motion_features, context], dim=1)
        hidden = self.gru(hidden, inp)

        # 預測視差更新量
        delta_disp = self.disp_head(hidden)

        return hidden, delta_disp, gate


class PIDSStereoPolVolumeV2(nn.Module):
    """
    PIDS Stereo with Polarization Volume V2

    相比 V1 的改進：
    1. PolarizationAttention: 從偏振特徵識別玻璃區域
    2. GatedFusion: 學習在何時信任 stereo vs pol

    架構:
        left ──→ [Feature Encoder] ──→ fmap1 ─┐
                                               ├──→ [CorrBlock] ──────────┐
        right ─→ [Feature Encoder] ──→ fmap2 ─┘                           │
                                                                          │
        left ──→ [Downsample] ────────────────┐                           │
                                               ├──→ [PolCorrBlock] ───────┤
        right ─→ [Downsample] ────────────────┘                           │
                                                                          ↓
                                                              [PolarizationAttention]
                                                                          │
                                                              [GatedFusion] ──→ fused
                                                                          │
                                                              [GRU] ──→ disparity
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        feature_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        pol_levels: int = 4,
        pol_radius: int = 4,
        fused_dim: int = 128,
        iters: int = 12,
        mixed_precision: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.iters = iters
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.pol_levels = pol_levels
        self.pol_radius = pol_radius
        self.fused_dim = fused_dim
        self.mixed_precision = mixed_precision

        # Correlation volume dimensions
        self.corr_dim = corr_levels * (2 * corr_radius + 1)
        self.pol_corr_dim = pol_levels * (2 * pol_radius + 1)

        # Feature encoder (shared weights for left/right)
        self.fnet = FeatureEncoder(output_dim=feature_dim)

        # Context encoder (for left image only)
        self.cnet = ContextEncoder(output_dim=context_dim, hidden_dim=hidden_dim)

        # Update block V2 with attention and gated fusion
        self.update_block = UpdateBlockV2(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_dim=self.corr_dim,
            pol_dim=self.pol_corr_dim,
            fused_dim=fused_dim,
        )

    def freeze_backbone(self):
        """凍結 backbone (fnet, cnet)"""
        for param in self.fnet.parameters():
            param.requires_grad = False
        for param in self.cnet.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """解凍 backbone"""
        for param in self.fnet.parameters():
            param.requires_grad = True
        for param in self.cnet.parameters():
            param.requires_grad = True

    def freeze_pol_modules(self):
        """
        凍結 V2 特有的偏振模組 (PolarizationAttention, GatedFusion)
        用於 nopol batch，避免 V2 模組從無偏振數據學習
        """
        self.update_block.freeze_pol_modules()

    def unfreeze_pol_modules(self):
        """
        解凍 V2 特有的偏振模組
        用於 pol batch，讓 V2 模組從有偏振數據學習
        """
        self.update_block.unfreeze_pol_modules()

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                iters: Optional[int] = None, test_mode: bool = False,
                disparity_gt: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Forward pass

        Args:
            left: (B, 3, H, W) 左圖 (I_parallel)
            right: (B, 3, H, W) 右圖 (I_cross)
            iters: 迭代次數 (覆蓋預設值)
            test_mode: 測試模式 (只返回最終預測)
            disparity_gt: 未使用 (為了與舊 API 兼容，V2 不需要 GT)

        Returns:
            List of flow predictions [(B, 2, H, W), ...] (disparity 為負的 x-flow)
        """
        if iters is None:
            iters = self.iters

        # Normalize images
        left = 2 * (left / 255.0) - 1.0 if left.max() > 1 else 2 * left - 1.0
        right = 2 * (right / 255.0) - 1.0 if right.max() > 1 else 2 * right - 1.0

        # Feature extraction
        fmap1 = self.fnet(left)
        fmap2 = self.fnet(right)

        # Context extraction (returns tuple: context, hidden)
        context, hidden = self.cnet(left)
        context = torch.tanh(context)
        # Note: hidden is already tanh'd in ContextEncoder.forward()

        # Build correlation volume
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # Build polarization volume (downsample images to 1/4)
        left_ds = F.avg_pool2d(left, 4)
        right_ds = F.avg_pool2d(right, 4)
        pol_fn = PolCorrBlock(left_ds, right_ds, num_levels=self.pol_levels, radius=self.pol_radius)

        # Initialize disparity
        b, _, h, w = fmap1.shape
        disp = torch.zeros(b, 1, h, w, device=left.device, dtype=left.dtype)

        flow_predictions = []
        gate_maps = []

        for _ in range(iters):
            disp = disp.detach()

            # Correlation lookup
            corr = corr_fn(disp)

            # Polarization lookup
            pol_corr = pol_fn(disp)

            # Update with V2 block (returns gate for visualization)
            hidden, delta_disp, gate = self.update_block(hidden, context, corr, pol_corr, disp)

            # Update disparity
            disp = disp + delta_disp

            # Store flow (convert disparity to flow format)
            flow = torch.cat([disp, torch.zeros_like(disp)], dim=1)

            if test_mode:
                flow_up = 4 * F.interpolate(flow, scale_factor=4, mode='bilinear', align_corners=True)
            else:
                flow_up = 4 * F.interpolate(flow, scale_factor=4, mode='bilinear', align_corners=True)

            flow_predictions.append(-flow_up)  # Negative because disparity convention
            gate_maps.append(gate)

        if test_mode:
            return flow_predictions[-1]

        return flow_predictions


class PIDSStereoPolVolume(nn.Module):
    """
    PIDS Stereo with Polarization Volume

    核心改進：使用 PolCorrBlock 取代 PolEncoder
    - 不需要 disparity 來計算 pol_diff
    - Oracle 和 Real 完全相同！
    - 訓練/推論完全一致

    架構:
        left ──→ [Feature Encoder] ──→ fmap1 ─┐
                                               ├──→ [CorrBlock] ──────────┐
        right ─→ [Feature Encoder] ──→ fmap2 ─┘                           ├──→ [GRU] ──→ disparity
                                                                          │
        left ──→ [Downsample] ────────────────┐                           │
                                               ├──→ [PolCorrBlock] ───────┘
        right ─→ [Downsample] ────────────────┘

    GRU 每次迭代:
        1. 用當前 disp 查詢 CorrBlock → corr features
        2. 用當前 disp 查詢 PolCorrBlock → pol features
        3. concat(corr, pol, disp) → UpdateBlock → new disp
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        feature_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        pol_levels: int = 4,
        pol_radius: int = 4,
        iters: int = 12,
        mixed_precision: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.iters = iters
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.pol_levels = pol_levels
        self.pol_radius = pol_radius
        self.mixed_precision = mixed_precision

        # 維度計算
        self.corr_dim = corr_levels * (2 * corr_radius + 1)
        self.pol_dim = pol_levels * (2 * pol_radius + 1)

        # Stereo 編碼器
        self.fnet = FeatureEncoder(output_dim=feature_dim)

        # Context 編碼器
        self.cnet = ContextEncoder(output_dim=context_dim, hidden_dim=hidden_dim)

        # 更新塊 (支援 polarization volume)
        self.update_block = UpdateBlockWithPol(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_dim=self.corr_dim,
            pol_dim=self.pol_dim,
        )

    def freeze_bn(self):
        """凍結 BatchNorm 層"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _downsample_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        將影像降採樣到 1/4 解析度（與 feature map 相同）

        Args:
            img: (B, C, H, W)

        Returns:
            (B, C, H/4, W/4)
        """
        return F.avg_pool2d(img, kernel_size=4, stride=4)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters: Optional[int] = None,
        test_mode: bool = False,
        **kwargs  # 忽略 disparity_gt 等不需要的參數
    ) -> List[torch.Tensor]:
        """
        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            iters: GRU 迭代次數
            test_mode: 是否為測試模式

        Returns:
            flow_predictions: List[Tensor] - 每次迭代的 flow 預測
        """
        if iters is None:
            iters = self.iters

        # 1. 提取 stereo 特徵
        fmap1 = self.fnet(left)   # (B, feature_dim, H/4, W/4)
        fmap2 = self.fnet(right)  # (B, feature_dim, H/4, W/4)

        # 2. 提取 context 和 hidden state
        context, hidden = self.cnet(left)

        # 3. 構建相關性體積
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # 4. 構建偏振差異體積 (新架構核心！)
        # 降採樣影像到 1/4 解析度
        left_ds = self._downsample_image(left)
        right_ds = self._downsample_image(right)
        pol_fn = PolCorrBlock(left_ds, right_ds, num_levels=self.pol_levels, radius=self.pol_radius)

        # 5. 初始化視差
        b, _, h, w = fmap1.shape
        disp = torch.zeros(b, 1, h, w, device=left.device)

        flow_predictions = []

        for _ in range(iters):
            disp = disp.detach()

            # 查詢相關性
            corr = corr_fn(disp)

            # 查詢偏振差異 (和 corr 用相同的 disp！)
            pol_corr = pol_fn(disp)

            # 更新
            hidden, delta_disp = self.update_block(hidden, context, corr, pol_corr, disp)

            # 更新視差
            disp = disp + delta_disp

            # 上採樣到原始解析度
            disp_up = 4 * F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)

            # RAFT-Stereo 格式
            flow_up = torch.cat([disp_up, torch.zeros_like(disp_up)], dim=1)
            flow_predictions.append(flow_up)

        if test_mode:
            return flow_predictions[-1]

        return flow_predictions

    def get_pol_diff(self) -> Optional[torch.Tensor]:
        """為了兼容性，返回 None (新架構不單獨計算 pol_diff)"""
        return None


# =============================================================================
# V2-A: Pol-Conditioned Corr Residual (Disparity-aware modulation)
# =============================================================================

class PolCorrResidual(nn.Module):
    """
    Polarization-Conditioned Correlation Residual

    核心思想：將 pol_corr 轉換為 corr 的 additive bias
    corr_enhanced = corr + α * PolCorrResidual(pol_corr)

    優點：
    1. 在 disparity space 工作（因為 pol_corr 是用 disp lookup 的）
    2. Additive bias 保留 RAFT 的 inductive bias
    3. 極其簡單穩定

    Args:
        pol_dim: pol_corr 的維度 = pol_levels * (2 * pol_radius + 1)
        corr_dim: corr 的維度 = corr_levels * (2 * corr_radius + 1)
        hidden_dim: 中間層維度
        init_scale: 初始 scale（小值確保穩定開始）
    """

    def __init__(
        self,
        pol_dim: int,
        corr_dim: int,
        hidden_dim: int = 64,
        init_scale: float = 0.1,
    ):
        super().__init__()

        self.pol_dim = pol_dim
        self.corr_dim = corr_dim

        # 簡單的 conv 網絡：pol_dim → hidden → corr_dim
        self.net = nn.Sequential(
            nn.Conv2d(pol_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, corr_dim, kernel_size=1),  # 1x1 投影到 corr_dim
        )

        # 可學習的 scale，初始化為小值
        self.scale = nn.Parameter(torch.tensor(init_scale))

        # 初始化最後一層為接近 0，確保初始行為接近 V1
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pol_corr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pol_corr: (B, pol_dim, H, W) - 偏振相關性特徵

        Returns:
            residual: (B, corr_dim, H, W) - 要加到 corr 上的 residual
        """
        residual = self.net(pol_corr)
        return self.scale * residual


class PIDSStereoPolVolumeV2A(nn.Module):
    """
    PIDS Stereo with Polarization Volume V2-A

    核心改進：Pol-Conditioned Corr Residual
    corr_enhanced = corr + PolCorrResidual(pol_corr)

    vs V1: pol_corr 直接 concat 到 update_block
    vs V2 (Attention+Gate): spatial multiplicative control

    V2-A 優點：
    1. 在 disparity space 工作
    2. Additive bias 不破壞 RAFT inductive bias
    3. 簡單穩定，mixed training 極穩
    4. 可以視為 V1.5，但預期效果更好

    架構:
        corr = CorrBlock(disp)
        pol_corr = PolCorrBlock(disp)
        corr_enhanced = corr + PolCorrResidual(pol_corr)  # ← 核心改進
        delta = UpdateBlock(hidden, context, corr_enhanced, disp)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        feature_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        pol_levels: int = 4,
        pol_radius: int = 4,
        residual_hidden_dim: int = 64,
        residual_init_scale: float = 0.1,
        iters: int = 12,
        mixed_precision: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.iters = iters
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.pol_levels = pol_levels
        self.pol_radius = pol_radius
        self.mixed_precision = mixed_precision

        # 維度計算
        self.corr_dim = corr_levels * (2 * corr_radius + 1)
        self.pol_dim = pol_levels * (2 * pol_radius + 1)

        # Stereo 編碼器
        self.fnet = FeatureEncoder(output_dim=feature_dim)

        # Context 編碼器
        self.cnet = ContextEncoder(output_dim=context_dim, hidden_dim=hidden_dim)

        # V2-A 核心：Pol-Conditioned Corr Residual
        self.pol_residual = PolCorrResidual(
            pol_dim=self.pol_dim,
            corr_dim=self.corr_dim,
            hidden_dim=residual_hidden_dim,
            init_scale=residual_init_scale,
        )

        # 更新塊 - 使用原始 UpdateBlock（不需要 pol_corr！）
        self.update_block = UpdateBlock(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_dim=self.corr_dim,
        )

    def freeze_bn(self):
        """凍結 BatchNorm 層"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _downsample_image(self, img: torch.Tensor) -> torch.Tensor:
        """將影像降採樣到 1/4 解析度"""
        return F.avg_pool2d(img, kernel_size=4, stride=4)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters: Optional[int] = None,
        test_mode: bool = False,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            iters: GRU 迭代次數
            test_mode: 是否為測試模式

        Returns:
            flow_predictions: List[Tensor] - 每次迭代的 flow 預測
        """
        if iters is None:
            iters = self.iters

        # 1. 提取 stereo 特徵
        fmap1 = self.fnet(left)
        fmap2 = self.fnet(right)

        # 2. 提取 context 和 hidden state
        context, hidden = self.cnet(left)

        # 3. 構建相關性體積
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # 4. 構建偏振差異體積
        left_ds = self._downsample_image(left)
        right_ds = self._downsample_image(right)
        pol_fn = PolCorrBlock(left_ds, right_ds, num_levels=self.pol_levels, radius=self.pol_radius)

        # 5. 初始化視差
        b, _, h, w = fmap1.shape
        disp = torch.zeros(b, 1, h, w, device=left.device)

        flow_predictions = []

        for _ in range(iters):
            disp = disp.detach()

            # 查詢相關性
            corr = corr_fn(disp)

            # 查詢偏振差異
            pol_corr = pol_fn(disp)

            # V2-A 核心：Pol-Conditioned Corr Residual
            # corr_enhanced = corr + α * f(pol_corr)
            corr_enhanced = corr + self.pol_residual(pol_corr)

            # 使用增強後的 corr 更新（不需要單獨的 pol_corr！）
            hidden, delta_disp = self.update_block(hidden, context, corr_enhanced, disp)

            # 更新視差
            disp = disp + delta_disp

            # 上採樣到原始解析度
            disp_up = 4 * F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)

            # RAFT-Stereo 格式
            flow_up = torch.cat([disp_up, torch.zeros_like(disp_up)], dim=1)
            flow_predictions.append(flow_up)

        if test_mode:
            return flow_predictions[-1]

        return flow_predictions

    def get_pol_diff(self) -> Optional[torch.Tensor]:
        """為了兼容性，返回 None"""
        return None

    def get_residual_scale(self) -> float:
        """返回當前的 residual scale（用於監控訓練）"""
        return self.pol_residual.scale.item()


class PIDSStereoPolVolumeV2B(nn.Module):
    """
    PIDS Stereo with Polarization Volume V2-B (Scheduled Residual)

    核心改進：Iteration-Scheduled Residual
    corr_enhanced = corr + α(iter) * PolCorrResidual(pol_corr)

    其中 α(iter) = iter / (iters - 1)，從 0 → 1

    vs V2-A: static residual（每次 iteration 相同權重）
    vs V2-B: scheduled residual（前期弱、後期強）

    V2-B 優點：
    1. 符合 RAFT 逐步修正的精神
    2. 前期讓 stereo 先穩定，後期 pol 強化
    3. 低風險改動，只需一行修改
    4. 預期可進入 3.x px 區間

    架構:
        for i in range(iters):
            corr = CorrBlock(disp)
            pol_corr = PolCorrBlock(disp)
            alpha = i / (iters - 1)  # 0 → 1
            corr_enhanced = corr + alpha * PolCorrResidual(pol_corr)  # ← 核心改進
            delta = UpdateBlock(hidden, context, corr_enhanced, disp)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        feature_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        pol_levels: int = 4,
        pol_radius: int = 4,
        residual_hidden_dim: int = 64,
        residual_init_scale: float = 0.1,
        iters: int = 12,
        mixed_precision: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.iters = iters
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.pol_levels = pol_levels
        self.pol_radius = pol_radius
        self.mixed_precision = mixed_precision

        # 維度計算
        self.corr_dim = corr_levels * (2 * corr_radius + 1)
        self.pol_dim = pol_levels * (2 * pol_radius + 1)

        # Stereo 編碼器
        self.fnet = FeatureEncoder(output_dim=feature_dim)

        # Context 編碼器
        self.cnet = ContextEncoder(output_dim=context_dim, hidden_dim=hidden_dim)

        # V2-B 核心：Pol-Conditioned Corr Residual（同 V2-A）
        self.pol_residual = PolCorrResidual(
            pol_dim=self.pol_dim,
            corr_dim=self.corr_dim,
            hidden_dim=residual_hidden_dim,
            init_scale=residual_init_scale,
        )

        # 更新塊 - 使用原始 UpdateBlock
        self.update_block = UpdateBlock(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_dim=self.corr_dim,
        )

    def freeze_bn(self):
        """凍結 BatchNorm 層"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _downsample_image(self, img: torch.Tensor) -> torch.Tensor:
        """將影像降採樣到 1/4 解析度"""
        return F.avg_pool2d(img, kernel_size=4, stride=4)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters: Optional[int] = None,
        test_mode: bool = False,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            iters: GRU 迭代次數
            test_mode: 是否為測試模式

        Returns:
            flow_predictions: List[Tensor] - 每次迭代的 flow 預測
        """
        if iters is None:
            iters = self.iters

        # 1. 提取 stereo 特徵
        fmap1 = self.fnet(left)
        fmap2 = self.fnet(right)

        # 2. 提取 context 和 hidden state
        context, hidden = self.cnet(left)

        # 3. 構建相關性體積
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # 4. 構建偏振差異體積
        left_ds = self._downsample_image(left)
        right_ds = self._downsample_image(right)
        pol_fn = PolCorrBlock(left_ds, right_ds, num_levels=self.pol_levels, radius=self.pol_radius)

        # 5. 初始化視差
        b, _, h, w = fmap1.shape
        disp = torch.zeros(b, 1, h, w, device=left.device)

        flow_predictions = []

        for i in range(iters):
            disp = disp.detach()

            # 查詢相關性
            corr = corr_fn(disp)

            # 查詢偏振差異
            pol_corr = pol_fn(disp)

            # V2-B 核心：Iteration-Scheduled Residual
            # α(iter) = iter / (iters - 1)，從 0 → 1
            # 前期弱（讓 stereo 先穩定）、後期強（pol 強化）
            alpha = i / max(iters - 1, 1)
            corr_enhanced = corr + alpha * self.pol_residual(pol_corr)

            # 使用增強後的 corr 更新
            hidden, delta_disp = self.update_block(hidden, context, corr_enhanced, disp)

            # 更新視差
            disp = disp + delta_disp

            # 上採樣到原始解析度
            disp_up = 4 * F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)

            # RAFT-Stereo 格式
            flow_up = torch.cat([disp_up, torch.zeros_like(disp_up)], dim=1)
            flow_predictions.append(flow_up)

        if test_mode:
            return flow_predictions[-1]

        return flow_predictions

    def get_pol_diff(self) -> Optional[torch.Tensor]:
        """為了兼容性，返回 None"""
        return None

    def get_residual_scale(self) -> float:
        """返回當前的 residual scale（用於監控訓練）"""
        return self.pol_residual.scale.item()


class GradientGating(nn.Module):
    """
    極簡 Gradient Gating Module for V2-C

    設計原則：
    1. 極小網絡（2 層 conv，無 BN，無 attention）
    2. 避免 reviewer 質疑 "performance 來自 gating net capacity"
    3. 目標是 mechanism proof，不是 black box
    """

    def __init__(self, pol_dim: int, hidden_dim: int = 32):
        super().__init__()

        # 輸入: pol_corr (pol_dim) + disp_grad (1) → gate (1)
        self.net = nn.Sequential(
            nn.Conv2d(pol_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),  # gate ∈ [0, 1]
        )

        # 初始化為中性（gate ≈ 0.5）
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, pol_corr: torch.Tensor, disp_grad: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pol_corr: (B, pol_dim, H, W)
            disp_grad: (B, 1, H, W) - MUST be detached!

        Returns:
            gate: (B, 1, H, W) ∈ [0, 1]
        """
        x = torch.cat([pol_corr, disp_grad], dim=1)
        return self.net(x)


class PIDSStereoPolVolumeV2C(nn.Module):
    """
    PIDS Stereo with Polarization Volume V2-C (Gradient Gating)

    核心改進：用 disparity gradient 作為 uncertainty proxy
    - high gradient / unstable region → pol 有話語權
    - flat / confident region → pol 安靜

    公式：
        disp_grad = |∇disp|.detach()  # 結構信息，stop-grad 避免 feedback loop
        gate = GatingNetwork(pol_corr, disp_grad)  # gate ∈ [0, 1]
        alpha = i / (iters - 1)  # 保留 V2-B 的 schedule
        corr_enhanced = corr + alpha * gate * pol_residual(pol_corr)

    vs V2-A: static residual
    vs V2-B: α schedule only
    vs V2-C: α schedule + gradient gating (where & when)

    設計要點：
    1. disp_grad.detach() - 避免 feedback loop
    2. GatingNetwork 極小 - mechanism proof, not black box
    3. α * gate - 解決 early iteration disp_grad 是噪音的問題
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        feature_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        pol_levels: int = 4,
        pol_radius: int = 4,
        residual_hidden_dim: int = 64,
        residual_init_scale: float = 0.1,
        gating_hidden_dim: int = 32,
        iters: int = 12,
        mixed_precision: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.iters = iters
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.pol_levels = pol_levels
        self.pol_radius = pol_radius
        self.mixed_precision = mixed_precision

        # 維度計算
        self.corr_dim = corr_levels * (2 * corr_radius + 1)
        self.pol_dim = pol_levels * (2 * pol_radius + 1)

        # Stereo 編碼器
        self.fnet = FeatureEncoder(output_dim=feature_dim)

        # Context 編碼器
        self.cnet = ContextEncoder(output_dim=context_dim, hidden_dim=hidden_dim)

        # Pol-Conditioned Corr Residual（同 V2-A/V2-B）
        self.pol_residual = PolCorrResidual(
            pol_dim=self.pol_dim,
            corr_dim=self.corr_dim,
            hidden_dim=residual_hidden_dim,
            init_scale=residual_init_scale,
        )

        # V2-C 核心：Gradient Gating（極小網絡）
        self.gradient_gating = GradientGating(
            pol_dim=self.pol_dim,
            hidden_dim=gating_hidden_dim,
        )

        # 更新塊 - 使用原始 UpdateBlock
        self.update_block = UpdateBlock(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_dim=self.corr_dim,
        )

    def freeze_bn(self):
        """凍結 BatchNorm 層"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _downsample_image(self, img: torch.Tensor) -> torch.Tensor:
        """將影像降採樣到 1/4 解析度"""
        return F.avg_pool2d(img, kernel_size=4, stride=4)

    def _compute_disp_gradient(self, disp: torch.Tensor) -> torch.Tensor:
        """
        計算 disparity gradient 作為 uncertainty proxy

        Args:
            disp: (B, 1, H, W)

        Returns:
            grad_mag: (B, 1, H, W) - gradient magnitude, DETACHED
        """
        # 簡單的 Sobel-like gradient（水平方向為主，因為 stereo 是水平的）
        # 使用 padding='same' 保持尺寸
        grad_x = disp[:, :, :, 1:] - disp[:, :, :, :-1]  # (B, 1, H, W-1)
        grad_y = disp[:, :, 1:, :] - disp[:, :, :-1, :]  # (B, 1, H-1, W)

        # Padding 回原始尺寸
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')  # (B, 1, H, W)
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')  # (B, 1, H, W)

        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # 正規化到 [0, 1] 範圍（避免數值問題）
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)

        # CRITICAL: detach 避免 feedback loop
        return grad_mag.detach()

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters: Optional[int] = None,
        test_mode: bool = False,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            iters: GRU 迭代次數
            test_mode: 是否為測試模式

        Returns:
            flow_predictions: List[Tensor] - 每次迭代的 flow 預測
        """
        if iters is None:
            iters = self.iters

        # 1. 提取 stereo 特徵
        fmap1 = self.fnet(left)
        fmap2 = self.fnet(right)

        # 2. 提取 context 和 hidden state
        context, hidden = self.cnet(left)

        # 3. 構建相關性體積
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # 4. 構建偏振差異體積
        left_ds = self._downsample_image(left)
        right_ds = self._downsample_image(right)
        pol_fn = PolCorrBlock(left_ds, right_ds, num_levels=self.pol_levels, radius=self.pol_radius)

        # 5. 初始化視差
        b, _, h, w = fmap1.shape
        disp = torch.zeros(b, 1, h, w, device=left.device)

        flow_predictions = []

        for i in range(iters):
            disp = disp.detach()

            # 查詢相關性
            corr = corr_fn(disp)

            # 查詢偏振差異
            pol_corr = pol_fn(disp)

            # V2-C 核心：Gradient Gating
            # 1. 計算 disp gradient（detached！）
            disp_grad = self._compute_disp_gradient(disp)

            # 2. Gating: 用 pol_corr + disp_grad 決定 where
            gate = self.gradient_gating(pol_corr, disp_grad)  # (B, 1, H, W)

            # 3. α schedule: 決定 when（early iteration 抑制 gate）
            alpha = i / max(iters - 1, 1)

            # 4. 組合：alpha * gate * residual
            # gate 會 broadcast 到 corr_dim
            pol_contribution = alpha * gate * self.pol_residual(pol_corr)
            corr_enhanced = corr + pol_contribution

            # 使用增強後的 corr 更新
            hidden, delta_disp = self.update_block(hidden, context, corr_enhanced, disp)

            # 更新視差
            disp = disp + delta_disp

            # 上採樣到原始解析度
            disp_up = 4 * F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)

            # RAFT-Stereo 格式
            flow_up = torch.cat([disp_up, torch.zeros_like(disp_up)], dim=1)
            flow_predictions.append(flow_up)

        if test_mode:
            return flow_predictions[-1]

        return flow_predictions

    def get_pol_diff(self) -> Optional[torch.Tensor]:
        """為了兼容性，返回 None"""
        return None

    def get_residual_scale(self) -> float:
        """返回當前的 residual scale（用於監控訓練）"""
        return self.pol_residual.scale.item()


class DisparityAwarePolVolume:
    """
    Disparity-Aware Polarization Volume (V2-D 核心)

    核心思想：
    - 對於每個 disparity d，計算 pol_diff_d = left - right_at_d
    - 這告訴我們「在 disparity d 的假設下，左右是否來自同一物理點」

    物理意義：
    - 當 d = d_gt（正確 disparity）:
        - 非玻璃: pol_diff_d ≈ 0（同一點，無偏振差異）
        - 玻璃: pol_diff_d ≠ 0（同一點，但偏振不同 = material cue）
    - 當 d ≠ d_gt（錯誤 disparity / 假匹配）:
        - pol_diff_d 的 spatial pattern 會是錯位、不連續的
        - 網路可以學習區分這種「假匹配」的 signature

    與 CorrBlock 的關係：
    - 使用相同的 disparity sampling grid
    - 在 iteration 中根據當前 disp 查詢附近的 pol_diff
    """

    def __init__(self, left: torch.Tensor, right: torch.Tensor, radius: int = 4):
        """
        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            radius: disparity 查詢半徑
        """
        self.left = left
        self.right = right
        self.radius = radius
        self.batch, self.c, self.h, self.w = left.shape

    def __call__(self, disp: torch.Tensor) -> torch.Tensor:
        """
        查詢當前 disparity 附近的 pol_diff

        Args:
            disp: 當前估計的 disparity (B, 1, H, W) 在 1/4 解析度

        Returns:
            pol_volume: (B, 3, 2*radius+1, H, W) 或 (B, 3*(2*radius+1), H, W)
                        每個 disparity 候選的 pol_diff
        """
        B, _, h, w = disp.shape  # 1/4 解析度
        device = disp.device

        # 將原圖下採樣到 1/4 解析度（與 disp 匹配）
        left_ds = F.interpolate(self.left, size=(h, w), mode='bilinear', align_corners=True)
        right_ds = F.interpolate(self.right, size=(h, w), mode='bilinear', align_corners=True)

        # 構建採樣網格
        # x 座標：[0, 1, ..., w-1]
        coords_x = torch.arange(w, device=device, dtype=disp.dtype)
        coords_x = coords_x.view(1, 1, 1, w).expand(B, 1, h, w)

        # y 座標：[0, 1, ..., h-1]
        coords_y = torch.arange(h, device=device, dtype=disp.dtype)
        coords_y = coords_y.view(1, 1, h, 1).expand(B, 1, h, w)

        pol_diffs = []

        for dr in range(-self.radius, self.radius + 1):
            # disparity 候選 = 當前估計 + 偏移
            d = disp + dr  # (B, 1, H, W)

            # 計算右圖採樣位置：x_right = x - d
            x_right = coords_x - d  # (B, 1, H, W)

            # 正規化到 [-1, 1] for grid_sample
            x_norm = 2 * x_right / (w - 1) - 1
            y_norm = 2 * coords_y / (h - 1) - 1

            # grid: (B, H, W, 2)
            grid = torch.cat([x_norm, y_norm], dim=1).permute(0, 2, 3, 1)

            # 採樣右圖在 disparity d 處的值
            right_at_d = F.grid_sample(
                right_ds, grid, mode='bilinear', padding_mode='zeros', align_corners=True
            )  # (B, 3, H, W)

            # 計算 disparity-aware pol_diff
            pol_diff_d = left_ds - right_at_d  # (B, 3, H, W)
            pol_diffs.append(pol_diff_d)

        # 堆疊成 volume: (B, 3, 2*radius+1, H, W)
        pol_volume = torch.stack(pol_diffs, dim=2)

        return pol_volume


class PolGate3D(nn.Module):
    """
    Polarization Gate Network (V2-D)

    輸入：pol_volume (B, C, D, H, W) - disparity-aware pol differences
    輸出：gate (B, D, H, W) - per-disparity gate

    設計原則：
    - 極輕量（3D Conv）
    - 不要 over-design，先驗證概念
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, pol_volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pol_volume: (B, C, D, H, W)

        Returns:
            gate: (B, D, H, W)
        """
        gate = self.net(pol_volume)  # (B, 1, D, H, W)
        return gate.squeeze(1)  # (B, D, H, W)


class PIDSStereoPolVolumeV2D(nn.Module):
    """
    PIDS Stereo V2-D: Disparity-Aware Polarization Modulation

    這是 V2 系列的關鍵進化：

    V2-C (之前):
    - pol_corr → spatial gate [H,W]
    - gate 對所有 disparity 一視同仁
    - 無法區分「哪個 disparity 是假匹配」

    V2-D (現在):
    - pol_volume [H,W,D] → per-disparity gate [H,W,D]
    - 每個 disparity 候選有獨立的 gate
    - pol 可以說「這個 d 是反射造成的假匹配」

    核心創新：
    > pol 不是告訴模型「這裡重要」
    > 而是告訴模型「這個 disparity 不合理」

    這是第一個讓 polarization 參與「disparity 判別」的設計。

    物理一致性：
    - pol_diff_d = left - right_at_d
    - 當 d 正確時，pol_diff 反映 material 特性
    - 當 d 錯誤時，pol_diff 反映幾何錯位
    - 網路學習區分這兩種 signature
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        feature_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        iters: int = 12,
        mixed_precision: bool = False,
        # V2-D 專用參數
        pol_gate_hidden: int = 8,
        pol_alpha: float = 0.2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.iters = iters
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.mixed_precision = mixed_precision

        # 維度計算
        self.corr_dim = corr_levels * (2 * corr_radius + 1)

        # 標準 RAFT-Stereo 組件
        self.fnet = FeatureEncoder(output_dim=feature_dim)
        self.cnet = ContextEncoder(output_dim=context_dim, hidden_dim=hidden_dim)
        self.update_block = UpdateBlock(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_dim=self.corr_dim,
        )

        # V2-D 核心：Disparity-Aware Pol Gate
        self.pol_gate = PolGate3D(in_channels=3, hidden_channels=pol_gate_hidden)

        # Learnable modulation strength
        self.pol_alpha = nn.Parameter(torch.tensor(pol_alpha))

    def freeze_bn(self):
        """凍結 BatchNorm 層"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters: Optional[int] = None,
        test_mode: bool = False,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            iters: GRU 迭代次數
            test_mode: 是否為測試模式

        Returns:
            flow_predictions: List[Tensor] - 每次迭代的 flow 預測
        """
        if iters is None:
            iters = self.iters

        # 1. 提取 stereo 特徵（標準 RAFT-Stereo）
        fmap1 = self.fnet(left)
        fmap2 = self.fnet(right)

        # 2. 提取 context 和 hidden state
        context, hidden = self.cnet(left)

        # 3. 構建相關性體積
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # 4. 構建 Disparity-Aware Pol Volume（V2-D 核心）
        pol_volume_fn = DisparityAwarePolVolume(left, right, radius=self.corr_radius)

        # 5. 初始化視差
        b, _, h, w = fmap1.shape
        disp = torch.zeros(b, 1, h, w, device=left.device)

        flow_predictions = []

        # 6. 迭代更新
        for i in range(iters):
            disp = disp.detach()

            # 6a. 標準 corr lookup
            corr = corr_fn(disp)  # (B, corr_dim, H, W)

            # 6b. V2-D: Disparity-Aware Pol Modulation
            pol_volume = pol_volume_fn(disp)  # (B, 3, 2*radius+1, H, W)
            pol_gate = self.pol_gate(pol_volume)  # (B, 2*radius+1, H, W)

            # 將 pol_gate 擴展到與 corr 相同的維度
            # corr: (B, num_levels * (2*radius+1), H, W)
            # pol_gate: (B, 2*radius+1, H, W)
            # 策略：在每個 pyramid level 重複使用 pol_gate
            pol_gate_expanded = pol_gate.unsqueeze(1).repeat(1, self.corr_levels, 1, 1, 1)
            pol_gate_expanded = pol_gate_expanded.view(b, self.corr_dim, h, w)

            # 6c. Residual modulation（不破壞 RGB stereo baseline）
            # corr_mod = corr * (1 + alpha * (gate - 0.5) * 2)
            # 讓 gate=0.5 為中性，<0.5 抑制，>0.5 增強
            corr_mod = corr * (1.0 + self.pol_alpha * (pol_gate_expanded - 0.5) * 2)

            # 6d. 標準 update
            hidden, delta_disp = self.update_block(hidden, context, corr_mod, disp)

            # 更新視差
            disp = disp + delta_disp

            # 上採樣到原始解析度
            disp_up = 4 * F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)

            # RAFT-Stereo 格式
            flow_up = torch.cat([disp_up, torch.zeros_like(disp_up)], dim=1)
            flow_predictions.append(flow_up)

        if test_mode:
            return flow_predictions[-1]

        return flow_predictions

    def get_pol_diff(self) -> Optional[torch.Tensor]:
        """為了兼容性，返回 None"""
        return None

    def get_pol_alpha(self) -> float:
        """返回當前的 pol_alpha（用於監控訓練）"""
        return self.pol_alpha.item()


class PolWeightNet(nn.Module):
    """
    輕量網路：將 pol_diff 轉換為 weight

    輸入：pol_diff (B, 3, H, W)
    輸出：weight (B, 1, H, W) in [0, 1]
    """

    def __init__(self, hidden_channels: int = 8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, pol_diff: torch.Tensor) -> torch.Tensor:
        return self.net(pol_diff)


class PolWeightedCorrBlock:
    """
    Pre-Corr Pol Weighting: pol 在 correlation volume 構建時就介入

    與 V2-D 的關鍵區別：
    - V2-D: 在每次 iteration 的 lookup 時計算 pol_gate
    - V2-E: 在 forward 開始時一次性將 pol_weight 應用到 correlation volume

    這是真正的「源頭介入」：
    - pol 影響 correlation 的形成，而非事後修正
    - 後續 iteration 使用的是「已被 pol 調制過的 correlation」
    """

    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        pol_weight_net: nn.Module,
        num_levels: int = 4,
        radius: int = 4,
    ):
        self.num_levels = num_levels
        self.radius = radius

        batch, dim, h, w = fmap1.shape

        # 1. 計算原始 correlation [B*H, 1, W_left, W_right]
        corr = self._compute_correlation(fmap1, fmap2)

        # 2. 計算 pol_weight volume [B*H, 1, W_left, W_right]
        pol_weight = self._compute_pol_weight_volume(left, right, pol_weight_net, h, w)

        # 3. 在構建時就加權（Pre-Corr 核心）
        corr_weighted = corr * pol_weight

        # 4. 構建金字塔
        self.corr_pyramid = [corr_weighted]
        for _ in range(num_levels - 1):
            corr_weighted = F.avg_pool2d(corr_weighted, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr_weighted)

    def _compute_correlation(self, fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        """計算立體相關性（標準 RAFT-Stereo）"""
        batch, dim, h, w = fmap1.shape

        # 正規化
        fmap1 = fmap1 / (torch.norm(fmap1, dim=1, keepdim=True) + 1e-6)
        fmap2 = fmap2 / (torch.norm(fmap2, dim=1, keepdim=True) + 1e-6)

        # 計算相關性 [B, H, W_left, W_right]
        corr = torch.einsum('bchw,bchx->bhwx', fmap1, fmap2)

        return corr.reshape(batch * h, 1, w, w)

    def _compute_pol_weight_volume(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        pol_weight_net: nn.Module,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        計算 pol_weight volume [B*H, 1, W_left, W_right]

        對於每個 disparity d = x_left - x_right：
        pol_diff_d = left[:, :, y, x_left] - right[:, :, y, x_right]
        weight_d = pol_weight_net(pol_diff_d)
        """
        B, C, H_orig, W_orig = left.shape
        device = left.device

        # 下採樣到 feature 解析度 (1/4)
        left_ds = F.interpolate(left, size=(h, w), mode='bilinear', align_corners=True)
        right_ds = F.interpolate(right, size=(h, w), mode='bilinear', align_corners=True)

        # 初始化 weight volume [B, H, W, W]
        pol_weight_volume = torch.ones(B, h, w, w, device=device)

        # 對於每個 disparity d（從右圖的角度）
        # 實際上我們需要 weight[x_left, x_right] = f(left[x_left] - right[x_right])
        # 這等價於 weight[x_left, d] = f(left[x_left] - right[x_left - d])

        # 為了效率，我們用滑動窗口計算
        # 只計算合理的 disparity 範圍（0 到 W）
        for d in range(w):
            if d == 0:
                pol_diff_d = left_ds - right_ds  # [B, 3, H, W]
            else:
                # right 向右 shift d（相當於 disparity = d）
                right_shifted = F.pad(right_ds[:, :, :, d:], (0, d, 0, 0), mode='constant', value=0)
                pol_diff_d = left_ds - right_shifted  # [B, 3, H, W]

            # 計算 weight [B, 1, H, W]
            weight_d = pol_weight_net(pol_diff_d)  # [B, 1, H, W]

            # 填入 volume：weight[x_left, x_right] where x_right = x_left - d
            # 即 weight[:, :, x_left, x_left - d] = weight_d[:, :, :, x_left]
            for x_left in range(d, w):
                x_right = x_left - d
                pol_weight_volume[:, :, x_left, x_right] = weight_d[:, 0, :, x_left]

        # reshape to [B*H, 1, W, W]
        return pol_weight_volume.view(B * h, 1, w, w)

    def __call__(self, disp: torch.Tensor) -> torch.Tensor:
        """從加權後的相關性金字塔中查詢（與 CorrBlock 相同）"""
        batch_h, _, w, _ = self.corr_pyramid[0].shape
        batch = disp.shape[0]
        h = batch_h // batch

        disp = disp.squeeze(1).view(batch * h, w)

        out_pyramid = []
        for i, corr in enumerate(self.corr_pyramid):
            _, _, w_corr, _ = corr.shape

            scale = 1 / (2 ** i)
            disp_scaled = scale * disp

            dx = torch.linspace(-self.radius, self.radius, 2 * self.radius + 1, device=disp.device)

            x_left = torch.arange(w, device=disp.device, dtype=disp.dtype).view(1, w, 1).expand(batch * h, -1, -1)
            x_right = x_left - disp_scaled.unsqueeze(-1) + dx.view(1, 1, -1)

            x_norm = 2 * x_right / (w_corr - 1) - 1
            y_norm = 2 * x_left / (w_corr - 1) - 1
            y_norm = y_norm.expand(-1, -1, 2 * self.radius + 1)

            grid = torch.stack([x_norm, y_norm], dim=-1)

            corr_sampled = F.grid_sample(corr, grid, align_corners=True, mode='bilinear', padding_mode='zeros')

            out_pyramid.append(corr_sampled.view(batch, h, w, -1))

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2)


class PIDSStereoPolVolumeV2E(nn.Module):
    """
    PIDS Stereo V2-E: Pre-Corr Pol Weighting

    這是 V2 系列的最後嘗試：讓 pol 在 correlation volume 構建時就介入。

    與之前版本的關鍵區別：

    | 版本 | 介入時機 | 作用方式 |
    |------|----------|----------|
    | V2-A~D | Post-Corr | 修正/調制已有的 correlation |
    | V2-E | **Pre-Corr** | **影響 correlation 的形成** |

    設計原理：
    - 對於每個 disparity d，計算 pol_diff_d = left - shift(right, d)
    - 用 PolWeightNet 生成 pol_weight_d ∈ [0, 1]
    - 在 correlation volume 構建時：corr_weighted = corr * pol_weight

    物理意義：
    - pol_weight 高：這個 disparity 的 pol_diff 符合「同一物理點」的特徵
    - pol_weight 低：這個 disparity 可能是假匹配
    - **在 correlation 形成時就抑制假匹配，而非事後修正**
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        feature_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        iters: int = 12,
        mixed_precision: bool = False,
        # V2-E 專用參數
        pol_weight_hidden: int = 8,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.iters = iters
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.mixed_precision = mixed_precision

        # 維度計算
        self.corr_dim = corr_levels * (2 * corr_radius + 1)

        # 標準 RAFT-Stereo 組件
        self.fnet = FeatureEncoder(output_dim=feature_dim)
        self.cnet = ContextEncoder(output_dim=context_dim, hidden_dim=hidden_dim)
        self.update_block = UpdateBlock(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_dim=self.corr_dim,
        )

        # V2-E 核心：Pol Weight Network
        self.pol_weight_net = PolWeightNet(hidden_channels=pol_weight_hidden)

    def freeze_bn(self):
        """凍結 BatchNorm 層"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters: Optional[int] = None,
        test_mode: bool = False,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            iters: GRU 迭代次數
            test_mode: 是否為測試模式

        Returns:
            flow_predictions: List[Tensor] - 每次迭代的 flow 預測
        """
        if iters is None:
            iters = self.iters

        # 1. 提取 stereo 特徵（標準 RAFT-Stereo）
        fmap1 = self.fnet(left)
        fmap2 = self.fnet(right)

        # 2. 提取 context 和 hidden state
        context, hidden = self.cnet(left)

        # 3. V2-E 核心：構建 Pol-Weighted Correlation Volume
        # pol_weight 在此時就應用到 correlation volume（Pre-Corr）
        corr_fn = PolWeightedCorrBlock(
            fmap1, fmap2, left, right,
            self.pol_weight_net,
            num_levels=self.corr_levels,
            radius=self.corr_radius
        )

        # 4. 初始化視差
        b, _, h, w = fmap1.shape
        disp = torch.zeros(b, 1, h, w, device=left.device)

        flow_predictions = []

        # 5. 標準 RAFT-Stereo 迭代（correlation 已經被 pol 加權）
        for _ in range(iters):
            disp = disp.detach()

            # 查詢已加權的 correlation
            corr = corr_fn(disp)

            # 標準 update
            hidden, delta_disp = self.update_block(hidden, context, corr, disp)

            # 更新視差
            disp = disp + delta_disp

            # 上採樣到原始解析度
            disp_up = 4 * F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)

            # RAFT-Stereo 格式
            flow_up = torch.cat([disp_up, torch.zeros_like(disp_up)], dim=1)
            flow_predictions.append(flow_up)

        if test_mode:
            return flow_predictions[-1]

        return flow_predictions

    def get_pol_diff(self) -> Optional[torch.Tensor]:
        """為了兼容性，返回 None"""
        return None


class PIDSStereoPolVolumeV3(nn.Module):
    """
    PIDS Stereo V3: Pol-in-Feature（最小可驗證版本）

    核心改動：讓 polarization 進入 feature encoder
    fmap1 = fnet(concat(left, pol_diff))
    fmap2 = fnet(concat(right, pol_diff))

    設計原則：
    - 不加新 branch
    - 不加 attention
    - 不加 residual
    - 不加 gating
    - 只問一件事：pol 是否能影響 feature matching

    vs V1: pol_corr concat 到 update_block（too late）
    vs V2-A/B/C: corr residual/gating（still too late）
    vs V3: pol 在 feature extraction 階段就融入（early fusion）

    這是最簡單的驗證：
    如果 V3 > V1，證明 early fusion 有效
    如果 V3 <= V1，說明 pol 信息本身不足以改善 feature matching
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        feature_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        iters: int = 12,
        mixed_precision: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.iters = iters
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.mixed_precision = mixed_precision

        # 維度計算
        self.corr_dim = corr_levels * (2 * corr_radius + 1)

        # V3 核心：Feature Encoder 接受 6 channels (RGB + pol_diff)
        self.fnet = FeatureEncoder(output_dim=feature_dim, input_dim=6)

        # Context Encoder 仍然只用 left image (3 channels)
        self.cnet = ContextEncoder(output_dim=context_dim, hidden_dim=hidden_dim)

        # 更新塊 - 標準 UpdateBlock，無 pol 相關模組
        self.update_block = UpdateBlock(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_dim=self.corr_dim,
        )

    def freeze_bn(self):
        """凍結 BatchNorm 層"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters: Optional[int] = None,
        test_mode: bool = False,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            iters: GRU 迭代次數
            test_mode: 是否為測試模式

        Returns:
            flow_predictions: List[Tensor] - 每次迭代的 flow 預測
        """
        if iters is None:
            iters = self.iters

        # V3 核心：計算 pol_diff 並 concat 到 feature encoder 輸入
        pol_diff = left - right  # (B, 3, H, W)

        left_pol = torch.cat([left, pol_diff], dim=1)   # (B, 6, H, W)
        right_pol = torch.cat([right, pol_diff], dim=1)  # (B, 6, H, W)

        # 1. 提取 stereo 特徵（融入 pol 信息）
        fmap1 = self.fnet(left_pol)
        fmap2 = self.fnet(right_pol)

        # 2. 提取 context 和 hidden state（只用 left，不含 pol）
        context, hidden = self.cnet(left)

        # 3. 構建相關性體積（標準 RAFT-Stereo）
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # 4. 初始化視差
        b, _, h, w = fmap1.shape
        disp = torch.zeros(b, 1, h, w, device=left.device)

        flow_predictions = []

        # 5. 標準 RAFT-Stereo 迭代（無 pol 後處理）
        for _ in range(iters):
            disp = disp.detach()

            # 查詢相關性
            corr = corr_fn(disp)

            # 標準 update（無 pol residual/gating）
            hidden, delta_disp = self.update_block(hidden, context, corr, disp)

            # 更新視差
            disp = disp + delta_disp

            # 上採樣到原始解析度
            disp_up = 4 * F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)

            # RAFT-Stereo 格式
            flow_up = torch.cat([disp_up, torch.zeros_like(disp_up)], dim=1)
            flow_predictions.append(flow_up)

        if test_mode:
            return flow_predictions[-1]

        return flow_predictions

    def get_pol_diff(self) -> Optional[torch.Tensor]:
        """為了兼容性，返回 None"""
        return None


class FeatureEncoderWithPolFusion(nn.Module):
    """
    特徵編碼器 with Additive Pol Fusion (V3-B)

    設計原則：
    - 保持原始 3ch input（完整保留 pretrained weights）
    - 用獨立 side branch 處理 pol_diff
    - 在 conv1 之後用 additive fusion（soft, 不破壞 pretrained）

    vs V3: V3 用 6ch concat 破壞了 conv1 的 pretrained weights
    vs V3-B: 保持 conv1 3ch，用加法融合 pol 信息
    """

    def __init__(self, output_dim: int = 128, norm_fn: str = 'batch', pol_scale: float = 0.1):
        super().__init__()

        self.pol_scale = pol_scale

        # 原始 encoder（保持 3ch input，pretrained weights 完整）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.BatchNorm2d(64) if norm_fn == 'batch' else nn.InstanceNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        # Pol side branch（同結構，獨立學習，random init）
        self.pol_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pol_norm1 = nn.BatchNorm2d(64) if norm_fn == 'batch' else nn.InstanceNorm2d(64)
        self.pol_relu1 = nn.ReLU(inplace=True)

        # 殘差層（共享，pretrained weights）
        self.layer1 = self._make_layer(64, 64, 2, norm_fn, stride=1)
        self.layer2 = self._make_layer(64, 96, 2, norm_fn, stride=2)
        self.layer3 = self._make_layer(96, 128, 2, norm_fn, stride=1)

        # 輸出投影
        self.conv_out = nn.Conv2d(128, output_dim, kernel_size=1)

    def _make_layer(self, in_planes: int, planes: int, num_blocks: int,
                    norm_fn: str, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_planes, planes, norm_fn, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(planes, planes, norm_fn))
        return nn.Sequential(*layers)

    def forward(self, img: torch.Tensor, pol_diff: torch.Tensor) -> torch.Tensor:
        # 主分支（保留 pretrained weights）
        x = self.relu1(self.norm1(self.conv1(img)))

        # Pol side branch（獨立學習）
        pol_feat = self.pol_relu1(self.pol_norm1(self.pol_conv1(pol_diff)))

        # Additive fusion（early, soft）
        # pol_scale 控制 pol 的影響程度，初始小值讓 pretrained 主導
        x = x + self.pol_scale * pol_feat

        # 共享後續層
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv_out(x)
        return x


class PIDSStereoPolVolumeV3B(nn.Module):
    """
    PIDS Stereo V3-B: Additive Pol Fusion

    修復 V3 的問題：
    - V3 問題：6ch concat 破壞 conv1 pretrained weights → D1/D3 居高不下
    - V3-B 解決：保持 3ch input，用 side branch + additive fusion

    設計原則：
    - 完整保留 pretrained FeatureEncoder weights
    - pol_diff 通過獨立 branch 學習 → 不干擾主幹
    - soft additive fusion (scale=0.1) → 讓 pretrained 主導，pol 輔助

    預期效果：
    - D1/D3 應該恢復正常（pretrained 幾何能力保留）
    - Val 應該穩定（不再有 train-val gap）
    - Glass EPE 應該有提升（pol 信息在 feature 階段融入）
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        feature_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        iters: int = 12,
        mixed_precision: bool = False,
        pol_scale: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.iters = iters
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.mixed_precision = mixed_precision

        # 維度計算
        self.corr_dim = corr_levels * (2 * corr_radius + 1)

        # V3-B 核心：FeatureEncoder with Additive Pol Fusion
        self.fnet = FeatureEncoderWithPolFusion(
            output_dim=feature_dim,
            pol_scale=pol_scale
        )

        # Context Encoder 仍然只用 left image (3 channels)
        self.cnet = ContextEncoder(output_dim=context_dim, hidden_dim=hidden_dim)

        # 更新塊 - 標準 UpdateBlock，無 pol 相關模組
        self.update_block = UpdateBlock(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_dim=self.corr_dim,
        )

    def freeze_bn(self):
        """凍結 BatchNorm 層"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters: Optional[int] = None,
        test_mode: bool = False,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            iters: GRU 迭代次數
            test_mode: 是否為測試模式

        Returns:
            flow_predictions: List[Tensor] - 每次迭代的 flow 預測
        """
        if iters is None:
            iters = self.iters

        # V3-B: 計算 pol_diff
        pol_diff = left - right  # (B, 3, H, W)

        # 1. 提取 stereo 特徵（additive fusion）
        fmap1 = self.fnet(left, pol_diff)
        fmap2 = self.fnet(right, pol_diff)

        # 2. 提取 context 和 hidden state（只用 left，不含 pol）
        context, hidden = self.cnet(left)

        # 3. 構建相關性體積（標準 RAFT-Stereo）
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # 4. 初始化視差
        b, _, h, w = fmap1.shape
        disp = torch.zeros(b, 1, h, w, device=left.device)

        flow_predictions = []

        # 5. 標準 RAFT-Stereo 迭代（無 pol 後處理）
        for _ in range(iters):
            disp = disp.detach()

            # 查詢相關性
            corr = corr_fn(disp)

            # 標準 update
            hidden, delta_disp = self.update_block(hidden, context, corr, disp)

            # 更新視差
            disp = disp + delta_disp

            # 上採樣到原始解析度
            disp_up = 4 * F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)

            # RAFT-Stereo 格式
            flow_up = torch.cat([disp_up, torch.zeros_like(disp_up)], dim=1)
            flow_predictions.append(flow_up)

        if test_mode:
            return flow_predictions[-1]

        return flow_predictions

    def get_pol_diff(self) -> Optional[torch.Tensor]:
        """為了兼容性，返回 None"""
        return None


class PIDSStereoLoss(nn.Module):
    """
    PIDS 損失函數 (強化版)

    結合:
    - L1 損失 (基礎)
    - 多尺度損失 (迭代預測)
    - 玻璃區域加權 (強調透明物體)
    - Polarization-aware 加權 (偏振差異大的區域更重要)
    """

    def __init__(
        self,
        gamma: float = 0.9,
        max_disp: float = 576.0,
        glass_weight: float = 5.0,  # 提高預設值
        pol_weight: float = 2.0,    # 新增: 偏振區域額外權重
        strict_glass_weight: float = 0.5,  # 嚴格 mask 額外權重 (交集區域，保守預設)
    ):
        super().__init__()

        self.gamma = gamma
        self.max_disp = max_disp
        self.glass_weight = glass_weight
        self.pol_weight = pol_weight
        self.strict_glass_weight = strict_glass_weight

    def forward(
        self,
        disp_predictions: List[torch.Tensor],
        disp_gt: torch.Tensor,
        valid_mask: torch.Tensor,
        glass_mask: Optional[torch.Tensor] = None,
        pol_diff: Optional[torch.Tensor] = None,
        glass_mask_strict: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            disp_predictions: 視差預測列表
            disp_gt: Ground truth 視差 (B, 1, H, W)
            valid_mask: 有效像素遮罩 (B, 1, H, W)
            glass_mask: 玻璃區域遮罩 (B, 1, H, W) - 聯集，用於訓練
            pol_diff: 偏振差異圖 (B, 1, H, W)，用於 polarization-aware weighting
            glass_mask_strict: 嚴格玻璃遮罩 (B, 1, H, W) - 交集，額外加權

        Returns:
            (total_loss, metrics_dict)
        """
        n_predictions = len(disp_predictions)
        total_loss = 0.0

        # 計算權重遮罩 (顯式拆分 union-only 和 strict 區域)
        #
        # 權重分布:
        #   - 背景 (Background):     w = 1.0
        #   - 邊緣 (Union-only):     w = glass_weight
        #   - 核心 (Strict):         w = glass_weight + strict_glass_weight
        #
        weight_mask = torch.ones_like(valid_mask)

        if glass_mask is not None and glass_mask_strict is not None:
            # 顯式計算 union-only 區域 (邊緣)
            union_only = glass_mask * (1.0 - glass_mask_strict)  # U & ~S
            strict = glass_mask_strict

            # 邊緣區域: +glass_weight (從 1 變成 glass_weight)
            weight_mask = weight_mask + (self.glass_weight - 1.0) * union_only
            # 核心區域: +glass_weight + strict_glass_weight
            weight_mask = weight_mask + (self.glass_weight - 1.0 + self.strict_glass_weight) * strict

        elif glass_mask is not None:
            # 只有 union mask，沒有 strict (fallback)
            weight_mask = weight_mask + (self.glass_weight - 1.0) * glass_mask

        # Polarization-aware 加權: 偏振差異大的區域更重要
        if pol_diff is not None:
            # 確保 pol_diff 和 disp_gt 尺寸匹配
            if pol_diff.shape[-2:] != disp_gt.shape[-2:]:
                pol_diff = F.interpolate(pol_diff, size=disp_gt.shape[-2:],
                                         mode='bilinear', align_corners=True)
            # 正規化 pol_diff 到 [0, 1]
            pol_diff_norm = pol_diff / (pol_diff.max() + 1e-6)
            # 額外權重: 1 + pol_weight * pol_diff
            weight_mask = weight_mask + self.pol_weight * pol_diff_norm

        weight_mask = weight_mask * valid_mask

        # 序列損失 (越後面的預測權重越高)
        for i, disp_pred in enumerate(disp_predictions):
            # 指數增長的權重
            weight = self.gamma ** (n_predictions - i - 1)

            # L1 損失
            loss = torch.abs(disp_pred - disp_gt)
            loss = (loss * weight_mask).sum() / (weight_mask.sum() + 1e-6)

            total_loss += weight * loss

        # 計算指標
        with torch.no_grad():
            final_pred = disp_predictions[-1]
            epe = torch.abs(final_pred - disp_gt)
            epe = (epe * valid_mask).sum() / (valid_mask.sum() + 1e-6)

            # 閾值誤差率 (D1, D3, D5, D10)
            abs_error = torch.abs(final_pred - disp_gt)
            thresh_1 = ((abs_error > 1.0) * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            thresh_3 = ((abs_error > 3.0) * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            thresh_5 = ((abs_error > 5.0) * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            thresh_10 = ((abs_error > 10.0) * valid_mask).sum() / (valid_mask.sum() + 1e-6)

            # 玻璃區域誤差 (union mask — 包含邊緣)
            if glass_mask is not None and glass_mask.sum() > 0:
                glass_valid = glass_mask * valid_mask
                glass_epe = (abs_error * glass_valid).sum() / (glass_valid.sum() + 1e-6)
            else:
                glass_epe = torch.tensor(0.0)

            # 玻璃區域誤差 (strict mask — 只有核心區域)
            if glass_mask_strict is not None and glass_mask_strict.sum() > 0:
                glass_strict_valid = glass_mask_strict * valid_mask
                glass_epe_strict = (abs_error * glass_strict_valid).sum() / (glass_strict_valid.sum() + 1e-6)
            else:
                glass_epe_strict = glass_epe  # fallback: 沒有 strict mask 時用 union

        metrics = {
            'loss': total_loss.item(),
            'epe': epe.item(),
            'd1': thresh_1.item() * 100,  # 百分比
            'd3': thresh_3.item() * 100,
            'd5': thresh_5.item() * 100,
            'd10': thresh_10.item() * 100,
            'glass_epe': glass_epe.item(),
            'glass_epe_strict': glass_epe_strict.item(),
        }

        return total_loss, metrics


def build_model(cfg: dict = None, dual_stream: bool = False) -> nn.Module:
    """
    建立模型

    Args:
        cfg: 配置字典
        dual_stream: 是否使用 Dual-Stream 架構 (含偏振編碼器)

    Returns:
        PIDSStereo 或 PIDSStereoDualStream model
    """
    if cfg is None:
        cfg = {}

    if dual_stream:
        model = PIDSStereoDualStream(
            hidden_dim=cfg.get('hidden_dim', 128),
            context_dim=cfg.get('context_dim', 128),
            feature_dim=cfg.get('feature_dim', 128),
            corr_levels=cfg.get('corr_levels', 4),
            corr_radius=cfg.get('corr_radius', 4),
            iters=cfg.get('iters', 12),
            pol_dim=cfg.get('pol_dim', 32),
            pol_threshold=cfg.get('pol_threshold', 0.1),
            pol_sharpness=cfg.get('pol_sharpness', 20.0),
        )
    else:
        model = PIDSStereo(
            hidden_dim=cfg.get('hidden_dim', 128),
            context_dim=cfg.get('context_dim', 128),
            feature_dim=cfg.get('feature_dim', 128),
            corr_levels=cfg.get('corr_levels', 4),
            corr_radius=cfg.get('corr_radius', 4),
            iters=cfg.get('iters', 12),
        )

    return model


def build_model_dual_stream(cfg: dict = None) -> PIDSStereoDualStream:
    """
    建立 Dual-Stream 模型 (快捷方式)

    Args:
        cfg: 配置字典

    Returns:
        PIDSStereoDualStream model
    """
    return build_model(cfg, dual_stream=True)


# ============================================================================
# NEW ARCHITECTURE: Learnable Polarization Volume (Exp #25)
# ============================================================================

class PolarizationInputTransform(nn.Module):
    """
    將原始左右圖像轉換為偏振語義輸入

    輸入: I_∥ (left), I_⊥ (right)
    輸出: [I_sum, I_diff, DoLP] - 3 通道偏振特徵

    設計理念:
    - I_sum: 亮度基底，保留紋理信息
    - I_diff: 偏振差異，玻璃區域高
    - DoLP: 正規化偏振度，不受亮度影響

    這樣的輸入降低 PolEncoder 學習紋理的動機，聚焦偏振信號
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, img_left: torch.Tensor, img_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_left: (B, C, H, W) I_∥ 左圖
            img_right: (B, C, H, W) I_⊥ 右圖

        Returns:
            pol_input_left: (B, 3, H, W) [I_sum, I_diff, DoLP] for left
            pol_input_right: (B, 3, H, W) [I_sum, I_diff, DoLP] for right

        Note: 由於左右有視差，這裡分別計算各自的特徵
              實際的偏振比較在 LearnablePolCorrBlock 中進行
        """
        # 轉為灰度 (如果是 RGB)
        if img_left.shape[1] == 3:
            left_gray = img_left.mean(dim=1, keepdim=True)
            right_gray = img_right.mean(dim=1, keepdim=True)
        else:
            left_gray = img_left
            right_gray = img_right

        # 對於單張圖像，我們用自身作為參考計算特徵
        # I_sum: 自身強度 (用於紋理)
        # I_self: 自身強度 (用於 DoLP 計算時的分母)

        # 左圖特徵
        pol_left = torch.cat([
            left_gray,                                    # 強度
            left_gray,                                    # 預留給 diff (在 corr 計算時用)
            left_gray / (left_gray.abs() + self.eps),    # 正規化強度
        ], dim=1)

        # 右圖特徵
        pol_right = torch.cat([
            right_gray,
            right_gray,
            right_gray / (right_gray.abs() + self.eps),
        ], dim=1)

        return pol_left, pol_right


class LightweightPolHead(nn.Module):
    """
    輕量級偏振特徵頭 (~18K 參數)

    從 FeatureEncoder 的輸出提取偏振專用特徵

    設計理念:
    - 用 1x1 conv 保持輕量
    - 有一層非線性增加表達能力
    - 學習「哪些特徵組合對偏振重要」

    結構: feature_dim -> 64 -> pol_dim
    """

    def __init__(self, in_dim: int = 256, pol_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, pol_dim, 1),
        )
        # 參數量: in_dim*64 + 64 + 64*pol_dim + pol_dim ≈ 18K (for in_dim=256, pol_dim=32)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fmap: (B, in_dim, H, W) from FeatureEncoder
        Returns:
            pol_feat: (B, pol_dim, H, W)
        """
        return self.net(fmap)


class LearnablePolCorrBlock:
    """
    可學習版偏振相關體積 (向量化計算)

    與 PolCorrBlock 的差異:
    - 輸入是學習到的特徵，不是原始像素
    - 用 L1 差異而非簡單差值
    - 向量化計算，無 for-loop

    核心: pol_corr[x, d] = |pol_feat_left[x] - pol_feat_right[x-d]|

    語意: 差異大 = 偏振信號強 (可能是玻璃)
    """

    def __init__(self, pol_feat_left: torch.Tensor, pol_feat_right: torch.Tensor,
                 num_levels: int = 4, radius: int = 4):
        """
        Args:
            pol_feat_left: (B, C, H, W) 左圖偏振特徵
            pol_feat_right: (B, C, H, W) 右圖偏振特徵
            num_levels: 金字塔層數
            radius: 查詢半徑
        """
        self.num_levels = num_levels
        self.radius = radius

        # 計算偏振體積 (向量化)
        pol_volume = self._compute_pol_volume_vectorized(pol_feat_left, pol_feat_right)

        # 構建金字塔
        self.pol_pyramid = [pol_volume]
        for _ in range(num_levels - 1):
            pol_volume = F.avg_pool2d(pol_volume, kernel_size=2, stride=2)
            self.pol_pyramid.append(pol_volume)

    def _compute_pol_volume_vectorized(self, feat_left: torch.Tensor,
                                        feat_right: torch.Tensor) -> torch.Tensor:
        """
        向量化計算偏振差異體積

        使用 einsum 和矩陣運算，避免 Python for-loop

        pol_volume[b, h, x_left, x_right] = mean(|feat_left[b,:,h,x_left] - feat_right[b,:,h,x_right]|)
        """
        batch, channels, h, w = feat_left.shape

        # 擴展維度進行廣播計算
        # feat_left: (B, C, H, W) -> (B, C, H, W, 1)
        # feat_right: (B, C, H, W) -> (B, C, H, 1, W)
        feat_left_exp = feat_left.unsqueeze(-1)      # (B, C, H, W, 1)
        feat_right_exp = feat_right.unsqueeze(-2)    # (B, C, H, 1, W)

        # 計算 L1 差異並平均 channels
        # diff: (B, C, H, W_left, W_right)
        diff = torch.abs(feat_left_exp - feat_right_exp)

        # 平均 channel 維度
        pol_volume = diff.mean(dim=1)  # (B, H, W, W)

        # 重塑為 (B*H, 1, W, W) 與 CorrBlock 格式一致
        pol_volume = pol_volume.reshape(batch * h, 1, w, w)

        return pol_volume

    def __call__(self, disp: torch.Tensor) -> torch.Tensor:
        """
        從偏振金字塔中查詢 (與 CorrBlock 接口一致)

        Args:
            disp: (B, 1, H, W) 視差估計

        Returns:
            (B, C, H, W) 偏振特徵，C = num_levels * (2*radius+1)
        """
        batch_h, _, w, _ = self.pol_pyramid[0].shape
        batch = disp.shape[0]
        h = batch_h // batch

        disp = disp.squeeze(1).view(batch * h, w)

        out_pyramid = []
        for i, pol_vol in enumerate(self.pol_pyramid):
            _, _, w_pol, _ = pol_vol.shape

            scale = 1 / (2 ** i)
            disp_scaled = scale * disp

            dx = torch.linspace(-self.radius, self.radius, 2 * self.radius + 1, device=disp.device)

            x_left = torch.arange(w, device=disp.device, dtype=disp.dtype).view(1, w, 1).expand(batch * h, -1, -1)
            x_right = x_left - disp_scaled.unsqueeze(-1) + dx.view(1, 1, -1)

            x_norm = 2 * x_right / (w_pol - 1) - 1
            y_norm = 2 * x_left / (w_pol - 1) - 1
            y_norm = y_norm.expand(-1, -1, 2 * self.radius + 1)

            grid = torch.stack([x_norm, y_norm], dim=-1)

            pol_sampled = F.grid_sample(pol_vol, grid, align_corners=True, mode='bilinear', padding_mode='zeros')

            out_pyramid.append(pol_sampled.view(batch, h, w, -1))

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2)  # (B, C, H, W)


class GlassAwareLoss(nn.Module):
    """
    Glass-aware Auxiliary Loss

    讓 PolEncoder 學習「玻璃偏振特徵」而不是「紋理」

    設計理念:
    - 玻璃區域的 pol_volume 應該高 (偏振差異大)
    - 背景區域的 pol_volume 應該低 (無偏振差異)

    Loss = -mean(pol_volume[glass]) + λ * mean(pol_volume[non_glass])
    或 margin ranking: pol_glass > pol_bg + margin
    """

    def __init__(self, margin: float = 0.1, lambda_bg: float = 1.0):
        super().__init__()
        self.margin = margin
        self.lambda_bg = lambda_bg

    def forward(self, pol_volume: torch.Tensor, glass_mask: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pol_volume: (B, D, H, W) 偏振體積 (或取某個 d 的 slice)
            glass_mask: (B, 1, H, W) 玻璃區域 mask
            valid_mask: (B, 1, H, W) 有效區域 mask

        Returns:
            loss: scalar
        """
        # 確保維度匹配
        if pol_volume.dim() == 4 and pol_volume.shape[1] > 1:
            # 如果是完整 volume，取中間的 disparity slice 作為代表
            # 或者取 max over disparity
            pol_signal = pol_volume.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        else:
            pol_signal = pol_volume

        # 調整 mask 尺寸
        if pol_signal.shape[-2:] != glass_mask.shape[-2:]:
            glass_mask = F.interpolate(glass_mask, size=pol_signal.shape[-2:], mode='nearest')
        if valid_mask is not None and pol_signal.shape[-2:] != valid_mask.shape[-2:]:
            valid_mask = F.interpolate(valid_mask, size=pol_signal.shape[-2:], mode='nearest')

        # 計算有效區域
        if valid_mask is not None:
            glass_valid = glass_mask * valid_mask
            bg_valid = (1 - glass_mask) * valid_mask
        else:
            glass_valid = glass_mask
            bg_valid = 1 - glass_mask

        # 計算玻璃和背景的平均偏振信號
        glass_pol = (pol_signal * glass_valid).sum() / (glass_valid.sum() + 1e-6)
        bg_pol = (pol_signal * bg_valid).sum() / (bg_valid.sum() + 1e-6)

        # Margin ranking loss: glass_pol > bg_pol + margin
        # Loss = max(0, margin - (glass_pol - bg_pol))
        #      = max(0, margin + bg_pol - glass_pol)
        loss = F.relu(self.margin + bg_pol - glass_pol)

        return loss


class PIDSStereoLearnablePol(nn.Module):
    """
    PIDS Stereo with Learnable Polarization Volume (Exp #25)

    相比 PIDSStereoPolVolume 的改進:
    1. PolHead 學習提取偏振特徵 (不是固定公式)
    2. 可以學習「什麼是穩健的偏振信號」
    3. 支持 Glass-aware auxiliary loss

    架構:
        left ──→ [FeatureEncoder] ──→ fmap1 ─┬──→ [CorrBlock] ──────────┐
                       │                     │                          │
                       └──→ [PolHead] ───────┼──→ pol_feat1             │
                                             │                          │
        right ─→ [FeatureEncoder] ──→ fmap2 ─┼──→ [CorrBlock] ──────────┤
                       │                     │                          │
                       └──→ [PolHead] ───────┼──→ pol_feat2             │
                                             │                          │
                              [LearnablePolCorrBlock] ──────────────────┤
                                             │                          │
                                             ↓                          ↓
                                      pol_volume              corr_volume
                                             │                          │
                                             └──────────┬───────────────┘
                                                        ↓
                                                [UpdateBlockWithPol]
                                                        │
                                                        ↓
                                                   disparity

    新增參數量: ~18K (PolHead)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        feature_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        pol_dim: int = 32,
        pol_levels: int = 4,
        pol_radius: int = 4,
        iters: int = 12,
        mixed_precision: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.iters = iters
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.pol_dim = pol_dim
        self.pol_levels = pol_levels
        self.pol_radius = pol_radius
        self.mixed_precision = mixed_precision

        # 維度計算
        self.corr_dim = corr_levels * (2 * corr_radius + 1)
        self.pol_corr_dim = pol_levels * (2 * pol_radius + 1)

        # Stereo 編碼器
        self.fnet = FeatureEncoder(output_dim=feature_dim)

        # Context 編碼器
        self.cnet = ContextEncoder(output_dim=context_dim, hidden_dim=hidden_dim)

        # 新增: 輕量級偏振頭 (~18K 參數)
        self.pol_head = LightweightPolHead(in_dim=feature_dim, pol_dim=pol_dim)

        # 更新塊 (支援 learnable polarization volume)
        self.update_block = UpdateBlockWithPol(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_dim=self.corr_dim,
            pol_dim=self.pol_corr_dim,
        )

        # Glass-aware loss 模組
        self.glass_loss = GlassAwareLoss(margin=0.1, lambda_bg=1.0)

        # 保存最後的 pol_volume 供 loss 使用
        self._last_pol_volume = None

    def freeze_bn(self):
        """凍結 BatchNorm 層"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_param_groups(self, base_lr: float = 1e-5, new_lr: float = 1e-4) -> List[dict]:
        """
        返回 optimizer 用的 param groups

        原始層用小 lr (保護預訓練權重)
        新增層 (pol_head) 用正常 lr
        """
        base_params = []
        new_params = []

        for name, param in self.named_parameters():
            if 'pol_head' in name:
                new_params.append(param)
            else:
                base_params.append(param)

        return [
            {'params': base_params, 'lr': base_lr},
            {'params': new_params, 'lr': new_lr},
        ]

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters: Optional[int] = None,
        test_mode: bool = False,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            iters: GRU 迭代次數
            test_mode: 是否為測試模式

        Returns:
            flow_predictions: List[Tensor] - 每次迭代的 flow 預測
        """
        if iters is None:
            iters = self.iters

        # 1. 提取 stereo 特徵
        fmap1 = self.fnet(left)   # (B, feature_dim, H/4, W/4)
        fmap2 = self.fnet(right)  # (B, feature_dim, H/4, W/4)

        # 2. 提取 context 和 hidden state
        context, hidden = self.cnet(left)

        # 3. 構建幾何相關性體積
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # 4. 提取偏振特徵 (新增: learnable)
        pol_feat1 = self.pol_head(fmap1)  # (B, pol_dim, H/4, W/4)
        pol_feat2 = self.pol_head(fmap2)  # (B, pol_dim, H/4, W/4)

        # 5. 構建可學習偏振體積
        pol_fn = LearnablePolCorrBlock(
            pol_feat1, pol_feat2,
            num_levels=self.pol_levels,
            radius=self.pol_radius
        )

        # 保存 pol_volume 供 glass-aware loss 使用
        self._last_pol_volume = pol_fn.pol_pyramid[0]  # 最高解析度

        # 6. 初始化視差
        b, _, h, w = fmap1.shape
        disp = torch.zeros(b, 1, h, w, device=left.device)

        flow_predictions = []

        for _ in range(iters):
            disp = disp.detach()

            # 查詢幾何相關性
            corr = corr_fn(disp)

            # 查詢偏振相關性
            pol_corr = pol_fn(disp)

            # 更新
            hidden, delta_disp = self.update_block(hidden, context, corr, pol_corr, disp)

            # 更新視差
            disp = disp + delta_disp

            # 上採樣到原始解析度
            disp_up = 4 * F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)

            # RAFT-Stereo 格式
            flow_up = torch.cat([disp_up, torch.zeros_like(disp_up)], dim=1)
            flow_predictions.append(flow_up)

        if test_mode:
            return flow_predictions[-1]

        return flow_predictions

    def compute_glass_aware_loss(self, glass_mask: torch.Tensor,
                                  valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        計算 Glass-aware auxiliary loss

        在 forward() 之後調用

        Args:
            glass_mask: (B, 1, H, W) 玻璃區域 mask
            valid_mask: (B, 1, H, W) 有效區域 mask

        Returns:
            loss: scalar
        """
        if self._last_pol_volume is None:
            return torch.tensor(0.0, device=glass_mask.device)

        # 重塑 pol_volume: (B*H, 1, W, W) -> (B, H, W, W) -> 取對角線近似
        batch_h, _, w, _ = self._last_pol_volume.shape
        batch = glass_mask.shape[0]
        h = batch_h // batch

        # 簡化: 取 d=0 附近的平均作為偏振信號強度
        pol_volume = self._last_pol_volume.view(batch, h, w, w)

        # 取對角線附近 (d ≈ 0 到 d ≈ max_expected_disp)
        # 簡單做法: 取每個位置的 max
        pol_signal = pol_volume.max(dim=-1)[0].unsqueeze(1)  # (B, 1, H, W)

        return self.glass_loss(pol_signal, glass_mask, valid_mask)

    def get_pol_volume(self) -> Optional[torch.Tensor]:
        """獲取最後一次 forward 的偏振體積 (用於可視化)"""
        return self._last_pol_volume


if __name__ == '__main__':
    # 測試原始模型
    print("=" * 60)
    print("Testing PIDSStereo (Original)")
    print("=" * 60)
    model = build_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    left = torch.randn(2, 3, 480, 640)
    right = torch.randn(2, 3, 480, 640)

    with torch.no_grad():
        predictions = model(left, right, iters=6)

    print(f"Number of predictions: {len(predictions)}")
    print(f"Final prediction shape: {predictions[-1].shape}")

    # 測試 Dual-Stream 模型
    print("\n" + "=" * 60)
    print("Testing PIDSStereoDualStream (Dual-Stream)")
    print("=" * 60)
    model_ds = build_model_dual_stream({
        'pol_dim': 32,
        'pol_threshold': 0.1,
        'pol_sharpness': 20.0,
    })
    print(f"Model parameters: {sum(p.numel() for p in model_ds.parameters()):,}")

    # 計算新增參數量
    pol_params = sum(p.numel() for n, p in model_ds.named_parameters()
                     if 'pol_encoder' in n or 'fusion' in n)
    print(f"  - Polarization Encoder + Fusion: {pol_params:,} params")

    with torch.no_grad():
        predictions_ds = model_ds(left, right, iters=6)

    print(f"Number of predictions: {len(predictions_ds)}")
    print(f"Final prediction shape: {predictions_ds[-1].shape}")

    # 測試 param groups
    print("\n" + "=" * 60)
    print("Testing param groups for optimizer")
    print("=" * 60)
    param_groups = model_ds.get_param_groups(base_lr=1e-5, new_lr=1e-4)
    print(f"Base params (lr=1e-5): {sum(p.numel() for p in param_groups[0]['params']):,}")
    print(f"New params (lr=1e-4): {sum(p.numel() for p in param_groups[1]['params']):,}")

    # 測試推論方法
    print("\n" + "=" * 60)
    print("Testing Inference Methods (Two-Stage Polarization Update)")
    print("=" * 60)

    import time

    # 方法 1: forward_inference (高效版，只更新 1-2 次偏振)
    print("\n[Method 1] forward_inference (efficient, 1-2 pol updates):")
    with torch.no_grad():
        t0 = time.time()
        result_infer = model_ds.forward_inference(left, right, iters=12, pol_update_iters=[6])
        t1 = time.time()
    print(f"  Output shape: {result_infer.shape}")
    print(f"  Time: {(t1-t0)*1000:.1f} ms")
    print(f"  Disparity range: [{result_infer[:, 0].min():.2f}, {result_infer[:, 0].max():.2f}]")

    # 方法 2: forward_two_pass (完整兩階段)
    print("\n[Method 2] forward_two_pass (full two-stage, 2x computation):")
    with torch.no_grad():
        t0 = time.time()
        result_two = model_ds.forward_two_pass(left, right, iters_pass1=6, iters_pass2=6)
        t1 = time.time()
    print(f"  Output shape: {result_two.shape}")
    print(f"  Time: {(t1-t0)*1000:.1f} ms")
    print(f"  Disparity range: [{result_two[:, 0].min():.2f}, {result_two[:, 0].max():.2f}]")

    # 方法 3: 訓練時用 GT disparity
    print("\n[Method 3] Training forward (with GT disparity):")
    fake_gt_disp = torch.ones(2, 1, 480, 640) * 60  # 假設平均視差 60px
    with torch.no_grad():
        t0 = time.time()
        predictions_gt = model_ds(left, right, iters=12, test_mode=True, disparity_gt=fake_gt_disp)
        t1 = time.time()
    print(f"  Output shape: {predictions_gt.shape}")
    print(f"  Time: {(t1-t0)*1000:.1f} ms")
    print(f"  Disparity range: [{predictions_gt[:, 0].min():.2f}, {predictions_gt[:, 0].max():.2f}]")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

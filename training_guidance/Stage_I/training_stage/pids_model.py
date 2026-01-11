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

    def __init__(self, output_dim: int = 128, norm_fn: str = 'batch'):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
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
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.iters = iters
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.mixed_precision = mixed_precision

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
    ) -> List[torch.Tensor]:
        """
        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            iters: GRU 迭代次數 (覆蓋預設值)
            test_mode: 是否為測試模式
            disparity_gt: GT 視差 (B, 1, H, W) - 訓練時用於對齊 pol_diff 計算

        Returns:
            flow_predictions: List[Tensor] - 每次迭代的 flow 預測 (H-方向 flow，用於 stereo)
        """
        if iters is None:
            iters = self.iters

        # 1. 提取偏振特徵 (使用 GT disparity 對齊，確保比較同一 3D 點)
        pol_feat = self.pol_encoder(left, right, disparity_gt)  # (B, pol_dim, H/4, W/4)

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

        Stage 1: 完全無偏振對齊 → disparity_v1
        Stage 2: 用 disparity_v1 做 warp → 正確偏振特徵 → disparity_v2

        這個方法比 forward_inference() 更精確但計算量是 2x

        Args:
            left: 左圖像 I∥ (B, 3, H, W)
            right: 右圖像 I⊥ (B, 3, H, W)
            iters_pass1: 第一階段迭代次數
            iters_pass2: 第二階段迭代次數

        Returns:
            final_flow: (B, 2, H, W) 最終 disparity
        """
        # ============ Pass 1: 無偏振對齊 ============
        flow_v1 = self.forward(left, right, iters=iters_pass1, test_mode=True, disparity_gt=None)
        disp_v1 = flow_v1[:, :1, :, :]  # 取 x-component

        # ============ Pass 2: 用估計的 disparity 對齊偏振 ============
        flow_v2 = self.forward(left, right, iters=iters_pass2, test_mode=True, disparity_gt=disp_v1)

        return flow_v2


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

            # 玻璃區域誤差
            if glass_mask is not None and glass_mask.sum() > 0:
                glass_valid = glass_mask * valid_mask
                glass_epe = (abs_error * glass_valid).sum() / (glass_valid.sum() + 1e-6)
            else:
                glass_epe = torch.tensor(0.0)

        metrics = {
            'loss': total_loss.item(),
            'epe': epe.item(),
            'd1': thresh_1.item() * 100,  # 百分比
            'd3': thresh_3.item() * 100,
            'd5': thresh_5.item() * 100,
            'd10': thresh_10.item() * 100,
            'glass_epe': glass_epe.item(),
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

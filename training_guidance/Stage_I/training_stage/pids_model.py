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
        """從相關性金字塔中查詢"""
        batch_h, _, w, _ = self.corr_pyramid[0].shape
        batch = disp.shape[0]
        h = batch_h // batch

        disp = disp.view(batch * h, 1, w, 1)

        out_pyramid = []
        for i, corr in enumerate(self.corr_pyramid):
            dx = torch.linspace(-self.radius, self.radius, 2 * self.radius + 1, device=disp.device)
            dx = dx.view(1, 1, 1, -1)

            # 縮放視差
            scale = 1 / (2 ** i)
            x0 = scale * disp

            # 採樣位置
            x = x0 + dx
            x = 2 * x / (w - 1) - 1

            # 固定 y 座標 (因為是立體匹配)
            y = torch.zeros_like(x)

            grid = torch.cat([x, y], dim=-1)

            # 雙線性採樣
            corr_sampled = F.grid_sample(corr, grid, align_corners=True, mode='bilinear')
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


class PIDSStereoLoss(nn.Module):
    """
    PIDS 損失函數

    結合:
    - L1 損失 (基礎)
    - 多尺度損失 (迭代預測)
    - 玻璃區域加權 (強調透明物體)
    """

    def __init__(
        self,
        gamma: float = 0.9,
        max_disp: float = 576.0,
        glass_weight: float = 2.0,
    ):
        super().__init__()

        self.gamma = gamma
        self.max_disp = max_disp
        self.glass_weight = glass_weight

    def forward(
        self,
        disp_predictions: List[torch.Tensor],
        disp_gt: torch.Tensor,
        valid_mask: torch.Tensor,
        glass_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            disp_predictions: 視差預測列表
            disp_gt: Ground truth 視差 (B, 1, H, W)
            valid_mask: 有效像素遮罩 (B, 1, H, W)
            glass_mask: 玻璃區域遮罩 (B, 1, H, W)

        Returns:
            (total_loss, metrics_dict)
        """
        n_predictions = len(disp_predictions)
        total_loss = 0.0

        # 計算權重遮罩
        if glass_mask is not None:
            weight_mask = 1.0 + (self.glass_weight - 1.0) * glass_mask
        else:
            weight_mask = torch.ones_like(valid_mask)

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


def build_model(cfg: dict = None) -> PIDSStereo:
    """
    建立模型

    Args:
        cfg: 配置字典

    Returns:
        PIDSStereo model
    """
    if cfg is None:
        cfg = {}

    model = PIDSStereo(
        hidden_dim=cfg.get('hidden_dim', 128),
        context_dim=cfg.get('context_dim', 128),
        feature_dim=cfg.get('feature_dim', 128),
        corr_levels=cfg.get('corr_levels', 4),
        corr_radius=cfg.get('corr_radius', 4),
        iters=cfg.get('iters', 12),
    )

    return model


if __name__ == '__main__':
    # 測試模型
    model = build_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 測試前向傳播
    left = torch.randn(2, 3, 480, 640)
    right = torch.randn(2, 3, 480, 640)

    with torch.no_grad():
        predictions = model(left, right, iters=6)

    print(f"Number of predictions: {len(predictions)}")
    print(f"Final prediction shape: {predictions[-1].shape}")

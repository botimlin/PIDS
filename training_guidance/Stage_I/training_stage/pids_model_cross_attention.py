"""
PIDS Cross-Attention Model (Fork from pids_model.py)
=====================================================

Fork 版本，不修改原始 pids_model.py。
新增 Cross-Attention Fusion 機制。

基於 Exp #16 的 Dual-Stream 架構，加入:
1. SafeCrossAttentionFusion - 安全的跨注意力融合
2. Pooled Attention - 記憶體高效的注意力計算
3. Alpha Cap Warmup - 漸進式啟用 cross-attention

使用方式:
    from pids_model_cross_attention import PIDSStereoDualStreamCrossAttention
    model = PIDSStereoDualStreamCrossAttention(...)

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

# 從原始 pids_model.py 導入基礎組件
from pids_model import (
    FeatureEncoder,
    ContextEncoder,
    CorrBlock,
    UpdateBlock,
    PolarizationEncoder,
    warp_with_disparity,
    SpatialAttention,
    ResidualBlock2D,
)


class PooledCrossAttention(nn.Module):
    """
    Memory-Efficient Pooled Cross-Attention

    將 K, V 池化到較小的空間尺寸，大幅減少記憶體使用。
    原始 attention: O(H*W * H*W) = O(N^2)
    Pooled attention: O(H*W * pool_h*pool_w) = O(N * M), M << N

    Memory 減少約 64 倍 (pool_size=8 時)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        pool_size: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.pool_size = pool_size
        self.scale = self.head_dim ** -0.5

        # Q, K, V 投影
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)

        # 輸出投影
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, C, H, W) - Query features
            key: (B, C, H, W) - Key features (will be pooled)
            value: (B, C, H, W) - Value features (will be pooled)
            mask: (B, 1, H, W) - Optional attention mask

        Returns:
            out: (B, C, H, W) - Attended features
        """
        B, C, H, W = query.shape

        # 計算池化後的尺寸
        pool_h = max(1, H // self.pool_size)
        pool_w = max(1, W // self.pool_size)

        # Q 投影 (保持原始解析度)
        q = self.q_proj(query)  # (B, C, H, W)

        # K, V 投影後池化
        k = self.k_proj(key)
        v = self.v_proj(value)
        k_pooled = F.adaptive_avg_pool2d(k, (pool_h, pool_w))  # (B, C, pool_h, pool_w)
        v_pooled = F.adaptive_avg_pool2d(v, (pool_h, pool_w))  # (B, C, pool_h, pool_w)

        # Reshape for multi-head attention
        # Q: (B, heads, H*W, head_dim)
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        # K, V: (B, heads, pool_h*pool_w, head_dim)
        k_pooled = k_pooled.view(B, self.num_heads, self.head_dim, pool_h * pool_w).permute(0, 1, 3, 2)
        v_pooled = v_pooled.view(B, self.num_heads, self.head_dim, pool_h * pool_w).permute(0, 1, 3, 2)

        # Attention: (B, heads, H*W, pool_h*pool_w)
        attn = torch.matmul(q, k_pooled.transpose(-2, -1)) * self.scale

        # 可選的 mask (池化 mask 到 K 的尺寸)
        if mask is not None:
            mask_pooled = F.adaptive_avg_pool2d(mask.float(), (pool_h, pool_w))
            mask_pooled = mask_pooled.view(B, 1, 1, pool_h * pool_w)
            # 只在有足夠有效值時應用 mask，避免全 -inf 導致 NaN
            valid_count = (mask_pooled >= 0.5).sum(dim=-1, keepdim=True)
            if valid_count.min() > 0:
                attn = attn.masked_fill(mask_pooled < 0.5, float('-inf'))

        # 數值穩定的 softmax (減去 max 防止 overflow)
        attn_max = attn.max(dim=-1, keepdim=True)[0]
        attn = attn - attn_max
        attn = F.softmax(attn, dim=-1)

        # 處理 NaN (如果仍然發生)
        attn = torch.nan_to_num(attn, nan=0.0)

        attn = self.dropout(attn)

        # 輸出: (B, heads, H*W, head_dim)
        out = torch.matmul(attn, v_pooled)

        # Reshape back: (B, C, H, W)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.out_proj(out)

        return out


class SafeCrossAttentionFusion(nn.Module):
    """
    安全版 Cross-Attention Fusion

    設計原則:
    1. 非對稱雙向: Pol->Stereo 強，Stereo->Pol 弱或關閉
    2. Residual + Learnable Scalar Gate: alpha 初始化接近 0
    3. Alpha Cap: 限制 attention 影響力的上限
    4. Pooled Attention: Memory-efficient
    5. Masked Attention: 只在可信偏振區域應用

    架構:
        stereo_feat ─┬─ [Q] ←─ pol_feat [K,V] ─→ attended_stereo
                     │
                     └─→ stereo_out = stereo_feat + alpha_ps * attended_stereo
    """

    def __init__(
        self,
        stereo_dim: int = 128,
        pol_dim: int = 128,
        out_dim: int = 128,
        num_heads: int = 4,
        pool_size: int = 8,
        dropout: float = 0.0,
        enable_stereo_to_pol: bool = False,  # 預設關閉 Stereo->Pol
        stereo_to_pol_scale: float = 0.1,    # Stereo->Pol 的額外縮放
    ):
        super().__init__()

        self.stereo_dim = stereo_dim
        self.pol_dim = pol_dim
        self.out_dim = out_dim
        self.enable_stereo_to_pol = enable_stereo_to_pol
        self.stereo_to_pol_scale = stereo_to_pol_scale

        # 維度對齊
        self.stereo_proj = nn.Conv2d(stereo_dim, out_dim, kernel_size=1) if stereo_dim != out_dim else nn.Identity()
        self.pol_proj = nn.Conv2d(pol_dim, out_dim, kernel_size=1) if pol_dim != out_dim else nn.Identity()

        # Pol -> Stereo Cross-Attention (主要作用)
        self.cross_attn_ps = PooledCrossAttention(
            dim=out_dim,
            num_heads=num_heads,
            pool_size=pool_size,
            dropout=dropout,
        )

        # Learnable scalar gate for Pol->Stereo
        # 初始化為 -5，sigmoid(-5) ≈ 0.007，接近 0
        self.alpha_ps_logit = nn.Parameter(torch.tensor(-5.0))

        # Pol->Stereo 的輸出投影 (初始化為 0)
        self.proj_ps = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        nn.init.zeros_(self.proj_ps.weight)
        nn.init.zeros_(self.proj_ps.bias)

        # 可選: Stereo -> Pol Cross-Attention (預設關閉)
        if enable_stereo_to_pol:
            self.cross_attn_sp = PooledCrossAttention(
                dim=out_dim,
                num_heads=num_heads,
                pool_size=pool_size,
                dropout=dropout,
            )
            self.alpha_sp_logit = nn.Parameter(torch.tensor(-5.0))
            self.proj_sp = nn.Conv2d(out_dim, out_dim, kernel_size=1)
            nn.init.zeros_(self.proj_sp.weight)
            nn.init.zeros_(self.proj_sp.bias)

        # 最終融合 (concat 後投影)
        self.final_fusion = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        # 保存 alpha 值供監控
        self.last_alpha_ps = None
        self.last_alpha_sp = None

    def forward(
        self,
        stereo_feat: torch.Tensor,
        pol_feat: torch.Tensor,
        pol_conf_mask: Optional[torch.Tensor] = None,
        alpha_cap: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            stereo_feat: (B, stereo_dim, H, W)
            pol_feat: (B, pol_dim, H, W)
            pol_conf_mask: (B, 1, H, W) - 偏振置信度 mask (0-1)
            alpha_cap: float - Alpha 上限 (用於 warmup)

        Returns:
            fused: (B, out_dim, H, W)
        """
        # 維度對齊
        stereo = self.stereo_proj(stereo_feat)  # (B, out_dim, H, W)
        pol = self.pol_proj(pol_feat)            # (B, out_dim, H, W)

        # Pol -> Stereo Cross-Attention
        # Query: stereo, Key/Value: pol
        attended_stereo = self.cross_attn_ps(
            query=stereo,
            key=pol,
            value=pol,
            mask=pol_conf_mask,
        )
        attended_stereo = self.proj_ps(attended_stereo)

        # 計算 alpha (with cap)
        alpha_ps = torch.sigmoid(self.alpha_ps_logit)
        alpha_ps = alpha_ps * alpha_cap  # 應用 warmup cap
        self.last_alpha_ps = alpha_ps.item()

        # Residual connection
        stereo_out = stereo + alpha_ps * attended_stereo

        # 可選: Stereo -> Pol
        if self.enable_stereo_to_pol:
            attended_pol = self.cross_attn_sp(
                query=pol,
                key=stereo,
                value=stereo,
                mask=None,  # Stereo->Pol 不需要 mask
            )
            attended_pol = self.proj_sp(attended_pol)

            alpha_sp = torch.sigmoid(self.alpha_sp_logit) * self.stereo_to_pol_scale
            alpha_sp = alpha_sp * alpha_cap
            self.last_alpha_sp = alpha_sp.item()

            pol_out = pol + alpha_sp * attended_pol
        else:
            pol_out = pol

        # 最終融合 (concat + projection)
        fused = torch.cat([stereo_out, pol_out], dim=1)
        fused = self.final_fusion(fused)

        return fused

    def get_alpha_values(self) -> Dict[str, float]:
        """獲取當前 alpha 值 (用於 logging)"""
        values = {'alpha_ps': self.last_alpha_ps}
        if self.enable_stereo_to_pol:
            values['alpha_sp'] = self.last_alpha_sp
        return values


class PIDSStereoDualStreamCrossAttention(nn.Module):
    """
    PIDS Dual-Stream + Cross-Attention 立體匹配網絡

    Fork from PIDSStereoDualStream，使用 Cross-Attention 替代 Concatenation 融合。

    架構:
        left ──→ [FeatureEncoder] ──→ fmap1 ─┐
                                              ├──→ [CrossAttnFusion] ──→ [Corr + GRU] ──→ disparity
        right ─→ [FeatureEncoder] ──→ fmap2 ─┤
                                              │
        |left-right| ─→ [PolarizationEncoder]─┘
                  (warp-aligned pol_diff)

    新增功能:
    - SafeCrossAttentionFusion 替代 FeatureFusion
    - Alpha Cap Warmup 機制
    - Pooled Attention (memory-efficient)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        feature_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        iters: int = 12,
        pol_dim: int = 64,
        pol_threshold: float = 0.05,
        pol_sharpness: float = 20.0,
        mixed_precision: bool = False,
        # Cross-Attention 參數
        num_heads: int = 4,
        pool_size: int = 8,
        enable_stereo_to_pol: bool = False,
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

        # Stereo 編碼器
        self.fnet = FeatureEncoder(output_dim=feature_dim)

        # Context 編碼器
        self.cnet = ContextEncoder(output_dim=context_dim, hidden_dim=hidden_dim)

        # 偏振編碼器
        self.pol_encoder = PolarizationEncoder(
            out_dim=pol_dim,
            threshold=pol_threshold,
            sharpness=pol_sharpness,
        )

        # Cross-Attention Fusion (替代原本的 FeatureFusion)
        self.fusion = SafeCrossAttentionFusion(
            stereo_dim=feature_dim,
            pol_dim=pol_dim,
            out_dim=feature_dim,
            num_heads=num_heads,
            pool_size=pool_size,
            enable_stereo_to_pol=enable_stereo_to_pol,
        )

        # 更新塊
        self.update_block = UpdateBlock(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_dim=self.corr_dim
        )

        # Alpha cap (用於 warmup)
        self.current_alpha_cap = 1.0

    def set_alpha_cap(self, cap: float):
        """設置當前 alpha cap (由 trainer 調用)"""
        self.current_alpha_cap = cap

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
            iters: GRU 迭代次數
            test_mode: 是否為測試模式
            disparity_gt: GT 視差 (B, 1, H, W) - 訓練時用於對齊 pol_diff 計算

        Returns:
            flow_predictions: List[Tensor] - 每次迭代的 flow 預測
        """
        if iters is None:
            iters = self.iters

        # 1. 提取偏振特徵 (使用 GT disparity 對齊)
        pol_feat = self.pol_encoder(left, right, disparity_gt)  # (B, pol_dim, H/4, W/4)

        # 獲取偏振置信度 mask (用於 masked attention)
        pol_conf_mask = self.pol_encoder.get_pol_diff()  # (B, 1, H, W)
        if pol_conf_mask is not None:
            # 下採樣到 feature 尺寸
            pol_conf_mask = F.interpolate(
                pol_conf_mask,
                size=pol_feat.shape[-2:],
                mode='bilinear',
                align_corners=True
            )

        # 2. 提取 stereo 特徵
        fmap1 = self.fnet(left)   # (B, feature_dim, H/4, W/4)
        fmap2 = self.fnet(right)  # (B, feature_dim, H/4, W/4)

        # 3. Cross-Attention Fusion (替代 concat)
        fmap1 = self.fusion(fmap1, pol_feat, pol_conf_mask, self.current_alpha_cap)
        fmap2 = self.fusion(fmap2, pol_feat, pol_conf_mask, self.current_alpha_cap)

        # 4. 提取 context 和 hidden state
        context, hidden = self.cnet(left)

        # 5. 構建相關性體積
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # 6. 初始化視差
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

            # 上採樣到原始解析度
            disp_up = 4 * F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True)

            # RAFT-Stereo 格式: flow (B, 2, H, W)
            flow_up = torch.cat([disp_up, torch.zeros_like(disp_up)], dim=1)
            flow_predictions.append(flow_up)

        if test_mode:
            return flow_predictions[-1]

        return flow_predictions

    def get_pol_diff(self) -> Optional[torch.Tensor]:
        """獲取偏振差異圖"""
        return self.pol_encoder.get_pol_diff()

    def get_attention_map(self) -> Optional[torch.Tensor]:
        """獲取 attention map"""
        return self.pol_encoder.get_attention_map()

    def get_alpha_values(self) -> Dict[str, float]:
        """獲取 cross-attention 的 alpha 值"""
        return self.fusion.get_alpha_values()


def build_model_cross_attention(cfg: dict = None) -> PIDSStereoDualStreamCrossAttention:
    """
    建立 Cross-Attention 模型

    Args:
        cfg: 配置字典

    Returns:
        PIDSStereoDualStreamCrossAttention model
    """
    if cfg is None:
        cfg = {}

    model = PIDSStereoDualStreamCrossAttention(
        hidden_dim=cfg.get('hidden_dim', 128),
        context_dim=cfg.get('context_dim', 128),
        feature_dim=cfg.get('feature_dim', 128),
        corr_levels=cfg.get('corr_levels', 4),
        corr_radius=cfg.get('corr_radius', 4),
        iters=cfg.get('iters', 12),
        pol_dim=cfg.get('pol_dim', 64),
        pol_threshold=cfg.get('pol_threshold', 0.05),
        pol_sharpness=cfg.get('pol_sharpness', 20.0),
        num_heads=cfg.get('num_heads', 4),
        pool_size=cfg.get('pool_size', 8),
        enable_stereo_to_pol=cfg.get('enable_stereo_to_pol', False),
    )

    return model


if __name__ == '__main__':
    # 測試 Cross-Attention 模型
    print("=" * 60)
    print("Testing PIDSStereoDualStreamCrossAttention")
    print("=" * 60)

    model = build_model_cross_attention({
        'pol_dim': 128,
        'num_heads': 4,
        'pool_size': 8,
    })

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 計算新增參數量
    new_params = sum(
        p.numel() for n, p in model.named_parameters()
        if 'pol_encoder' in n or 'fusion' in n
    )
    print(f"  - New params (pol_encoder + fusion): {new_params:,}")

    # 測試 forward
    left = torch.randn(2, 3, 480, 640)
    right = torch.randn(2, 3, 480, 640)
    disp_gt = torch.rand(2, 1, 480, 640) * 100

    with torch.no_grad():
        predictions = model(left, right, iters=6, disparity_gt=disp_gt)

    print(f"Number of predictions: {len(predictions)}")
    print(f"Final prediction shape: {predictions[-1].shape}")

    # 顯示 alpha 值
    alpha_values = model.get_alpha_values()
    print(f"Alpha values: {alpha_values}")

    print("\nTest passed!")

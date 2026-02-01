# PIDS 訓練方法論

> 從 DEVELOPMENT_LOG.md 整理，記錄架構設計決策與核心方法論

---

## 一、架構演進歷程

### 1.1 Dual-Stream 架構（Exp #15-18）

**設計動機**：
- 偏振資訊 (I∥, I⊥) 需要專門的編碼器處理
- 偏振特徵需要與幾何特徵融合

**核心設計**：
```
RGB Stream:  left, right → fnet → fmap1, fmap2 → correlation
Pol Stream:  I∥, I⊥ → PolEncoder → pol_feat → fusion with corr
```

**關鍵發現**：
- Dual-Stream + Concat 融合：Glass EPE 21.82 px（-48.3% vs Baseline）
- Cross-Attention 融合效果類似但參數更多

---

### 1.2 Oracle vs Real 模式（關鍵轉折）

**問題發現**：
Dual-Stream 在訓練時使用 GT disparity 計算 pol_diff，但推論時沒有 GT。

```
┌─────────────────────────────────────────────────────────────────┐
│              Oracle vs Real 模式對比                             │
├─────────────────────────────────────────────────────────────────┤
│  Oracle 模式 (用 GT disparity):                                  │
│    Pol:   5.87 px  |  Nopol: 11.27 px  |  改進: 47.9%           │
│                                                                 │
│  Real 模式 (用 predicted disparity):                             │
│    Pol:  11.92 px  |  Nopol: 89.45 px  |  改進: 86.7%           │
└─────────────────────────────────────────────────────────────────┘
```

**核心發現**：
- Oracle-Real Gap 揭示了 Dual-Stream 的本質問題
- Pol 在 Real 模式下的優勢更大（86.7% vs 47.9%）
- 偏振訊號具有幾何魯棒性：I∥ >> I⊥ 的強對比能「穿透」幾何誤差

---

### 1.3 Polarization Volume 架構（Exp #24+）

**設計動機**：
消除 Oracle-Real Gap — 讓 pol_diff 計算不依賴 disparity。

**核心創新**：
```python
# Dual-Stream (需要 disparity)
pol_diff = left - warp(right, disparity)  # 依賴 GT 或 predicted disparity

# Polarization Volume (不需要 disparity)
for d in disparity_range:
    pol_volume[d] = left - shift(right, d)  # 預計算所有候選
```

**結果**：Oracle = Real，無 Gap！

---

## 二、混合 Pol/Nopol 訓練策略（Exp #28）

### 2.1 動機

**問題觀察**：
- 純強偏振訓練 (Strict QA)：模型過度依賴偏振特徵
- 弱偏振區域預測失效
- Val EPE 波動大

**物理洞察 - 偏振反射的空間不均勻性**：
```
玻璃表面偏振分佈:
┌─────────────────────────────────────┐
│  ████  強偏振 (接近 Brewster 角)     │
│  ▓▓▓▓  中等偏振                      │
│  ░░░░  弱/無偏振 (正入射或邊緣角度)   │
└─────────────────────────────────────┘

物理原因:
- 偏振反射強度取決於入射角（Fresnel 方程）
- 只有接近 Brewster 角（~56° for glass）時才有強烈偏振
- 玻璃表面不同區域入射角不同 → 偏振強度不均勻
```

### 2.2 數據配置

```
混合比例: Pol 60% + Nopol 40%

Pol 數據:
├── 來源: 嚴格 QA 通過場景
├── Train: 3000 場景
└── Val: 300 場景 (只有 pol!)

Nopol 數據:
├── 來源: 全新 OBJ (無重疊)
├── Train: 2000 場景 (不進 val!)
└── 用途: 強迫學習純幾何特徵

關鍵規則:
├── 場景不重疊: Pol 和 Nopol 使用完全不同的 OBJ
├── Val = Pol only: 驗證集只有偏振數據
└── Nopol = Train only: 無偏振數據只用於訓練
```

### 2.3 Curriculum Learning 策略

**三階段訓練**：
```
總步數: 60000 steps

Phase 1 (前 20%): 幾何 + 偏振基礎
├── Steps 0 ~ 12000
├── 數據: 50% nopol + 50% pol
├── Nopol batch: 凍結 Pol Module (防止學到垃圾特徵)
├── Pol batch: 正常訓練
└── 目標: Backbone 學幾何，Pol Module 學基礎偏振

Phase 2 (中 40%): 混合強化
├── Steps 12000 ~ 36000
├── 數據: 40% nopol + 60% pol
├── 全部解凍，混合訓練
└── 目標: 整合幾何與偏振特徵

Phase 3 (後 40%): 偏振精修
├── Steps 36000 ~ 60000
├── 數據: 30% nopol + 70% pol
├── 全部解凍，混合訓練
└── 目標: 強化偏振特徵，保持幾何能力
```

### 2.4 關鍵發現：RAFT 雙目一致性被強偏振破壞

**RAFT-Stereo 的假設**：
```
理想情況: I_left(x, y) ≈ I_right(x - d, y)
         → 高 Correlation → 正確匹配
```

**強偏振破壞這個假設**：
```
強偏振 (Strict QA 保留):
├── 玻璃: I∥ >> I⊥ (例如 0.8 vs 0.2, 4倍差異)
├── RAFT 看到: 左邊亮斑 vs 右邊暗區
└── 結果: Correlation 低 → 匹配困難

弱偏振 (Raw data 包含):
├── 玻璃: I∥ ≈ I⊥ (例如 0.5 vs 0.4, 1.25倍差異)
├── RAFT 看到: 左右相似
└── 結果: Correlation 高 → 匹配容易
```

**PIDS 訓練的兩難**：
| 偏振強度 | 對 Pol Module | 對 RAFT Backbone |
|---------|--------------|------------------|
| 強偏振 | ✅ 明確信號 | ❌ 難匹配 |
| 弱偏振 | ⚠️ 信號弱 | ✅ 容易匹配 |

**最佳策略**: 混合數據讓模型同時學習「容易」和「困難」案例

### 2.5 實驗結論

| 實驗 | 數據策略 | Glass EPE |
|------|----------|-----------|
| Exp #24 | Raw data (含弱偏振) | **2.44 px** 🏆 |
| Exp #28 | Mixed pol + nopol | 4.88 px |
| Exp #27 | Strict QA (純強偏振) | 5.03 px |

**核心發現**：
1. **弱偏振 ≠ Nopol**: 弱偏振仍有微弱線索，模型可學習「軟過渡」
2. **Raw Data 是「軟標籤」**: 自然包含強→中→弱的連續過渡
3. **人工二分法有害**: pol/nopol 的硬分類反而干擾學習

---

## 三、凍結策略演進

### 3.1 Freeze Backbone First（Exp #29）

**理論基礎**：
1. Backbone 已預訓練，有良好的幾何特徵提取能力
2. Pol Module 需從零學習，需要穩定的 Backbone 作為基礎
3. 凍結可防止 Backbone 學到被強偏振破壞的錯誤匹配模式

**策略**：
```
Phase 1 (0-60%):  freeze Backbone, 只訓練 Pol Module + update_block
                  └── 70% pol + 30% nopol
Phase 2 (60-100%): 全部解凍, 協同微調
                   └── 70% pol + 30% nopol
```

**重要修正**：
```python
# 錯誤：凍結 update_block 會導致沒有參數可學習
BACKBONE_KEYWORDS = ['fnet', 'cnet', 'update_block']  # ❌

# 正確：保留 update_block 可訓練
BACKBONE_KEYWORDS = ['fnet', 'cnet']  # ✅
```

**結果**：Glass EPE 8.38 px（不如預期）
- Phase 1 空轉問題：16000 steps 低效訓練
- 需要更平滑的解凍策略

### 3.2 Gradual Unfreezing（Exp #30）

**改進**：
- 避免 Phase 1 plateau
- 平滑 backbone 解凍過程
- 使用連續 LR 調整而非硬切換

### 3.3 V2 系列的 Pol Module 凍結

**在 nopol data 上凍結 pol modules**：
```python
class PolModuleFreezer:
    """凍結 pol 相關參數"""
    POL_KEYWORDS = ['pol', 'polarization']

    def freeze(self):
        for name, param in self.model.named_parameters():
            if any(k in name.lower() for k in self.POL_KEYWORDS):
                param.requires_grad = False
```

**特殊處理 V2-A/B/C/D**：
- `pol_residual` 等新模組需要加入凍結列表
- 防止在 nopol 數據上學到錯誤模式

---

## 四、三條鐵律（V3 失敗後確立）

經過 V3/V3-B 的失敗（early fusion 破壞預訓練權重），確立了三條鐵律：

| # | 鐵律 | 原因 |
|---|------|------|
| 1 | ❌ pol 不進 fnet | 會破壞 RAFT-Stereo 預訓練的 feature extraction |
| 2 | ❌ pol 不只做 spatial [H,W] gating | 不知道「是哪個 disparity」 |
| 3 | ✅ pol 只能在 disparity-aware 空間作用 | cost volume / disparity index |

---

## 五、Post-Corr vs Pre-Corr 介入

### 5.1 Post-Corr 系列（V2-A ~ V2-D）

**共同特點**：
```python
# 先計算 correlation
corr = compute_correlation(fmap1, fmap2)

# 再用 pol 修正
corr_mod = corr * f(pol)  # 或 corr + f(pol)
```

| 版本 | 策略 | 特點 |
|------|------|------|
| V2-A | Static Residual | `corr + α * residual(pol)` |
| V2-B | Scheduled Residual | `corr + α(iter) * residual(pol)` |
| V2-C | Gradient Gating | `corr + gate(pol, ∇disp) * residual` |
| V2-D | Disparity-Aware Gate | `corr * (1 + α * (gate3D - 0.5))` |

**觀察**：V2 系列後期曲線重疊，天花板相似

### 5.2 Pre-Corr 介入（V2-E）

**核心轉變**：
```python
# V2 系列：Post-Corr（事後修正）
corr = dot(fmap1, fmap2)
corr_mod = corr * gate

# V2-E：Pre-Corr（源頭介入）
for d in disparity_range:
    pol_diff_d = left - shift(right, d)
    pol_weight_d = PolWeightNet(pol_diff_d)
    corr[d] = dot(fmap1, fmap2_at_d) * pol_weight_d
```

| | Post-Corr (V2-A~D) | Pre-Corr (V2-E) |
|--|---------------------|-----------------|
| 時機 | corr 計算後 | corr 計算時 |
| 作用 | 修正已有的 corr | 影響 corr 的形成 |
| 類比 | 事後補救 | 源頭介入 |
| 速度 | 快 | 慢（D 次 forward） |
| 穩定性 | 一般 | 更穩定（觀察中） |

---

## 六、V2-D 的質變：從 importance 到 validity

```
V2-C: pol 告訴模型「這裡重要」→ spatial gate [H,W]
V2-D: pol 告訴模型「這個 disparity 不合理」→ per-disparity gate [H,W,D]
```

| | V2-C | V2-D |
|--|------|------|
| gate 維度 | [H,W] | [H,W,D] |
| 能否區分 disparity | ❌ | ✅ |
| pol 角色 | importance mask | validity judge |

---

## 七、設計陷阱警告

### 7.1 Disparity 作為 Conditioning

**❌ 危險做法**：
```python
# 錯誤！early iteration 的 disp 是錯的
residual = pol_residual(pol_corr, disp)
```

**問題**：
- Early iteration 的 disparity 是 noise
- 會產生 feedback loop
- 訓練不穩，後期可能退化

**✅ 安全做法**：
- Option A: 使用 iteration index + fixed schedule (V2-B)
- Option B: 使用 disparity gradient/variance，不用值本身
- 關鍵: pol_residual 只學「where & how much」，不是「what disparity」

### 7.2 凍結 update_block

**❌ 危險做法**：
```python
# PIDSStereoPolVolume 中 PolCorrBlock 是非參數的
# 凍結 update_block 會導致沒有參數可學習！
BACKBONE_KEYWORDS = ['fnet', 'cnet', 'update_block']
```

**✅ 正確做法**：
```python
BACKBONE_KEYWORDS = ['fnet', 'cnet']  # 保留 update_block 可訓練
```

---

## 八、Loss 設計

### 8.1 Glass 區域加權

```python
loss = L1(pred, gt) * weight_mask

# weight_mask 分層
background:     w = 1.0
glass_edge:     w = glass_weight (5.0)
glass_strict:   w = glass_weight + strict_glass_weight (5.5)
```

### 8.2 Composite Score

```python
composite = glass_epe + d1_weight * d1_error
```
- 同時考慮平均誤差 (EPE) 和離群值 (D1)
- 防止模型只優化平均而忽略 outliers

---

## 九、實驗指標定義

### 9.1 基本指標

| 指標 | 說明 | 理想值 |
|------|------|--------|
| Glass EPE | 玻璃區域平均誤差（核心） | < 4.0 px |
| EPE | 全圖平均誤差 | < 2.0 px |
| D1 | 誤差 >1px 比例 | < 10% |
| D3 | 誤差 >3px 比例 | < 5% |
| Composite | glass_epe + 0.1×d1 | 越低越好 |

### 9.2 Robust Statistics（Exp #40 新增）

**動機**：Exp #39 發現 Mean vs Median 差距 +116%（5.039 vs 2.333 px）

| 指標 | 說明 | 用途 |
|------|------|------|
| Mean | 傳統平均值 | 易受 outlier 影響 |
| **Median** | 中位數 | **Primary metric** (穩定) |
| Trimmed Mean | 移除 top 10% 後平均 | 論文報告用 |
| P90 | 90th percentile | **Safety gate** |
| P95 | 95th percentile | 觀察用 |
| Outlier Rate | > 20px 的比例 | 資料品質指標 |

### 9.3 Checkpoint 選擇邏輯（Exp #40）

```python
# Primary: Median 更低
is_better_median = current_median < best_median

# Safety Gate: P90 不能太差 (不能超過 best_p90 的 1.5 倍)
p90_safe = current_p90 <= best_p90 * 1.5

# 同時滿足才是 best
is_best = is_better_median and p90_safe
```

**設計理由**：
- Median 不受 outlier 影響，更能反映「典型」表現
- P90 確保尾端不會太差（避免 median 好但 worst case 很爛）

---

## 十、架構演進總結圖

```
Baseline RAFT-Stereo
    │
    ├─→ Dual-Stream (Exp #16)
    │       │ Oracle-Real Gap 問題
    │       ↓
    └─→ Polarization Volume V1 (Exp #24)
            │ 消除 Gap，但效果有限
            ↓
        ┌── V2 系列 (Post-Corr) ──┐
        │   V2-A: Static Residual │
        │   V2-B: Scheduled       │
        │   V2-C: Gradient Gate   │
        │   V2-D: Disparity Gate  │ ← 天花板相似
        └─────────────────────────┘
            │
            ↓
        V3 系列 (Early Fusion) → ❌ 失敗（破壞預訓練）
            │
            ↓
        V2-E (Pre-Corr) ← PIDS 1.x 最佳
            │
            │  ══ PIDS 2.0 (RGB 時代) ══
            │
            ├─→ 候選 A: Two-Pass (V2-B + Pol-Context)
            │     Pass 1: V2-B correlation 修正
            │     Pass 2: Pol-aware Context 精修
            │
            └─→ 候選 B: Coarse-Scale Pol Context
                  單 pass: 低解析度 pol → cnet 注入
```

---

## 十一、Directional Impulse Descent (DID)

### 11.1 設計動機

V2-E 訓練觀察到：
- 曲線非常穩定，但中間會有暫時的 plateau
- OneCycle 是**時間驅動**的，無法感知 plateau
- 需要**狀態驅動**的機制來逃離 local minimum

### 11.2 核心理念

```
OneCycle = 全局退火曲線 (reference trajectory)
DID      = 狀態驅動的局部脈衝 (event-driven impulse disturbance)
```

**一句話總結**：OneCycle 管整體探索–收斂節律，DID 負責在錯誤 basin 內的瞬時能量注入。

### 11.3 物理類比

球在 loss landscape 的 saddle point 附近：
- 不是「沒有下降」，而是「一直想下降但下不去」
- `sign(∂L/∂t)` 一致 = 有方向意圖
- `|∂L/∂t|` 很小 = 動能不足

### 11.4 狀態機設計

```
NORMAL → (trigger) → IMPULSE → (T_impulse) → COOLDOWN → (T_cool) → PROTECTED → (P cycles) → NORMAL
```

### 11.5 觸發條件（同時滿足）

**必要條件：**
1. **Train slope plateau**: `median(s_recent) < ε`，其中 `ε = percentile(s_history, 10%)`
2. **Moving-best stale** [改進 D]: val best 連續 4 個 cycles 沒更新
3. **Direction consistent**: slope EMA 方向穩定（用 EMA + sign，不用 all()）
4. **Not in late phase**: `progress < 0.75`
5. **Under trigger limit**: `trigger_count < 3`
6. **Not in protection**: 距離上次觸發已過 3 個 validation cycles

**輔助證據（有助判斷，非必要）：**
- Val oscillating: val Glass EPE 在 ±0.5 px 內震盪（改進 D: 不再是必要條件）

### 11.6 LR 控制邏輯

**改進 B: 漸進式 Impulse** | **改進 C: 參數群分離**

| 狀態 | Pol/Update LR | Backbone LR | 說明 |
|------|---------------|-------------|------|
| NORMAL | `lr_base` | `lr_base` | 正常跟隨 OneCycle |
| IMPULSE (1st) | `lr_base × 2.0` | `lr_base × 1.2` | 第一次較保守 |
| IMPULSE (2nd+) | `lr_base × 3.0` | `lr_base × 1.2` | 後續用完整脈衝 |
| COOLDOWN | `lr_base × 0.1` | `lr_base × 1.0` | 快速阻尼，backbone 直接回 ref |
| PROTECTED | `lr_base` | `lr_base` | 回歸正常，觀察效果 |

**設計理由**：
- pol module 是需要擾動的目標（pol-only 驗證集顯示這是瓶頸）
- backbone 已收斂到好的 ImageNet 特徵空間，不應大幅擾動

### 11.7 關鍵設計決策

| 決策點 | 選擇 | 理由 |
|--------|------|------|
| ε 定義 | percentile(s_history, 10%) | 自適應、scale-free |
| Direction check | EMA + sign | all() 對 noise 太敏感 |
| γ_down 基準 | relative_base | 不偏離 OneCycle 軌跡 |
| 保護期 | 3 val cycles | 等待脈衝效果顯現 |
| 最大觸發 | 3 次 | 超過代表架構問題 |
| 第一次 γ_up | 2.0 (非 3.0) | [改進 B] 避免 bf16 grad overflow |
| backbone γ | 1.2 (非 full) | [改進 C] 保護 ImageNet 特徵 |
| 主觸發條件 | best stale | [改進 D] pol-only val 較安靜 |

### 11.8 改進 E: EPE-Adaptive Impulse

EPE 越低代表模型越精細，脈衝應該自動縮小，避免過度擾動：

```
γ_effective = 1.0 + (γ_raw - 1.0) × clamp(epe_smooth / epe_ref, 0, 1)

epe_ref = 8.0:
  epe_smooth = 10 px → γ = 3.0  (full)
  epe_smooth =  5 px → γ = 2.25
  epe_smooth =  3 px → γ = 1.75
  epe_smooth =  1 px → γ = 1.25 (gentle)
```

**保險絲 1: Smoothed EPE**
- `epe_smooth = EMA(val_glass_epe, α=0.4)` — 避免單次 val 抖動導致 γ 跟著抖

**保險絲 2: γ 變化率限制**
- `|Δγ| ≤ Δγ_max (0.3)` — 防止 EPE 突變導致 γ 突降，打斷訓練節奏

### 11.9 與其他方法的比較

| 面向 | OneCycle | ReduceLROnPlateau | DID |
|------|----------|-------------------|-----|
| 驅動方式 | 時間 | 狀態 | **狀態** |
| 作用方向 | 單向下降 | 只降 | **先升後降** |
| 控制論 | 開迴路 | 閉迴路 | **脈衝閉迴路** |
| 參數群控制 | 無 | 無 | **分群 (改進 C)** |

### 11.10 Exp #41 前期結果與凍結 🧊

| 指標 | 走勢 | 判定 |
|------|------|------|
| EPE Smooth | 51→41.5 (5k steps) → **42.12 (8k-19.5k 停滯)** | ❌ |
| Epsilon | 0.035 → 0.079 (持續上升) | ❌ |
| Gamma_eff | 1.0 (恆定不變) | ❌ |

**診斷**：base model 在 DID 額外優化複雜度下無法收斂基礎立體匹配（EPE 42 px）。gamma_eff 退化、epsilon 惡性上升。DID 需要 base model 先收斂到合理水平才能發揮作用。

**決定**：凍結 DID 調適工作，主線轉向 PIDS 2.0 架構。待架構穩定後再評估是否重啟。

---

## 十二、Pol Volume Crosstalk 與灰階瓶頸

### 12.1 問題觀察

即使使用 Median 作為 checkpoint 指標，val Glass EPE 仍有 spike。根因是 pol volume 的**結構性弱點**。

### 12.2 Disparity-Pol Crosstalk

Pol volume 假設：**左右影像的亮度差 = 偏振效應**。但這只在 disparity 正確時成立。

```
正確 disparity:
  Left I_∥(x,y) vs Right I_⊥(x, y-d)  →  同一物理點  →  真 pol 信號  ✓

錯誤 disparity:
  Left I_∥(x,y) vs Right I_⊥(x, y-d') →  不同物件  →  假 pol 信號  ✗
```

**正反饋迴路**：
```
disparity 錯位 → 假 pol 信號 → 引導 disparity 往錯誤方向 → 更大錯位 → ...
```

### 12.3 灰階的雙重弱化

灰階輸入 (I_∥, I_⊥ 各 1 channel) 從兩方面加劇問題：

1. **Backbone 特徵區分力不足**：RGB 有 3 通道建立特徵，灰階只有 1 個純量 — 不同物件可能灰度相似但顏色完全不同
2. **無法辨別假信號**：RGB 的 color 差異可以暗示「這不是 pol 效應」；灰階沒有這個能力

**根本矛盾 (Chicken-and-Egg)**：
```
標準 stereo:  RGB × 2 cameras  →  6 channels，特徵豐富但看不到玻璃
偏振 stereo:  Gray × 2 cameras →  2 channels，能看到玻璃但特徵貧乏
```

Pol volume 需要準確的 disparity 才能正常工作，但準確的 disparity 正是灰階最缺的。

### 12.4 緩解方案

| 方案 | 複雜度 | 原理 | 優先級 |
|------|--------|------|--------|
| **Iteration-level Pol Warmup** | 極低 | 前幾輪迭代不啟用 pol，讓 correlation 先穩住 disparity | 先做 |
| **Correlation Confidence Gating** | 中 | stereo match 信心低時抑制 pol（pixel-level 軟門控） | 第二步 |
| **Iteration Consistency Check** | 中高 | 跨迭代驗證 pol 信號穩定性，真 pol 應收斂穩定 | 可選 |
| **RGB 第三方裁判** | 高 | 引入無偏振 RGB 解耦 stereo matching 和 pol detection | **已完成 (V6 渲染器)** |

**Iteration Warmup 概念**：
```
Iteration:  1  2  3  4  5  6 ... 18 19 20 21 22 23 24
Pol weight: 0  0  0  0  .2 .4 ... 1  1  1  1  1  1  1
            ├─ correlation 先穩住 ─┤├─ pol 精修 ──────┤
```

**RGB 第三方裁判** — Mitsuba 3 研究結果：
- `cuda_ad_spectral_polarized` 已 pip 預裝，可輸出 15 channels (RGB + S0/S1/S2/S3 各 3ch)
- `I_∥(RGB) = (S0 + S1) / 2`，`I_⊥(RGB) = (S0 - S1) / 2`
- 無需編譯、spectral→RGB 由 Mitsuba 內部處理
- ~~限制：需升級現有渲染器從 `cuda_ad_rgb` 到 `cuda_ad_spectral_polarized` + Stokes integrator~~ → **已解決：V6 渲染器完成升級**

**V6 渲染器 (2026-01-31 完成)**：
- Film `pixel_format` 從 `luminance` 改為 `rgb`，輸出 15ch RGB Stokes
- `I_∥`, `I_⊥` 現為 (H,W,3) RGB 陣列，EXR 以 3ch 儲存
- DoLP / QA / report 等內部計算全部轉為 luminance（ITU-R BT.709）後處理
- 支援 resume（以 `_report.json` 存在為完成標記），可安全中斷/重啟 18000 場景渲染
- 物理觀察：Fresnel 反射僅覆蓋玻璃 10-50%（Brewster 角附近），RGB+pol vs RGB-only 需消融實驗驗證 pol 邊際貢獻

---

## 十三、實驗進度總表

| Exp | 架構 | 特點 | Glass EPE (Mean) | Glass EPE (Median) | 狀態 |
|-----|------|------|------------------|-------------------|------|
| #22 | Baseline | RAFT-Stereo | 6.542 px | 3.489 px | ✅ 完成 |
| #39 | V2-E | Pre-Corr Pol Weighting | 5.039 px | 2.333 px | ✅ 完成 |
| #40 | V2-E | + Robust Checkpoint | 5.402 px | 2.501 px | ✅ 完成 |
| #41 | V2-E | + DID (40k steps) | - (EPE 42px) | - | 🧊 凍結 |

**Exp #39 vs Baseline 改善**：
- Glass D1: 75% → 35% (改善 54%)
- Glass EPE Median: 3.489 → 2.333 px (改善 33%)

**PIDS 2.0 實驗規劃**（灰階已證明無 pol module 的隱式偏振無效，所有實驗均以 V2-B 為基礎）：
| Exp | 架構 | 資料 | 目的 |
|-----|------|------|------|
| 2.0-A | V2-B | RGB mixed | RGB + pol module 基準 |
| 2.0-A' | V2-B | RGB all-pol | 驗證是否需要 nopol 數據 |
| 2.0-C | Two-Pass (V2-B + Pol-Context) | RGB | 候選架構 A |
| 2.0-D | Coarse-Scale Pol Context | RGB | 候選架構 B |

對照組：Exp #22 (灰階 Baseline)、Exp #39 (灰階 V2-E PIDS 1.x 最佳)

---

## 十四、數值穩定性分析

### 14.1 問題：跨平台結果不一致

原始觀察：H200 上 Baseline Median 3.489 px vs RTX 4090 5.264 px (+51%)。

### 14.2 根因確認：TF32 在 Hopper 架構上的行為

在另一台 H200 上跑 C1-C5 穩定性測試，與 RTX 4090 對照：

```
                    C1       C2       C3       C4       C5        σ    CV(%)
                   TF32✓    TF32✗    TF32✓    TF32✗    TF32✓
────────────────────────────────────────────────────────────────────────────
4090 (Ada)        5.264    5.251    5.258    5.251    5.262    0.006   0.12%
H200 (Hopper)     3.504    5.250    3.501    5.250    3.501    0.958  22.79%
────────────────────────────────────────────────────────────────────────────
```

**根因**：TF32 (`torch.backends.cuda.matmul.allow_tf32`) 在 Hopper Tensor Core 上的精度截斷行為與 Ada Lovelace 完全不同：
- **Hopper**：TF32 ON → 3.50 px, OFF → 5.25 px（差 50%）
- **Ada**：TF32 ON → 5.26 px, OFF → 5.25 px（差 0.2%，幾乎無影響）
- **Benchmark 無影響**（C1≈C3, C2≈C4）
- **FP32 (TF32=OFF) 跨平台完全一致**：H200 5.250 ≈ 4090 5.251

原始假說（cuDNN 算法選擇差異）被推翻。實際是 TF32 matmul 精度截斷 → GRU 32 次迭代放大。
Hopper TF32 結果反而更好（3.50 < 5.25），可能是精度截斷的隱式正則化壓制了數值發散。

### 14.3 解法

在 evaluation 程式碼中預設關閉 TF32：
```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```
→ 所有平台得到一致的 FP32 結果（Baseline Median ≈ 5.25 px）。

### 14.4 實務意義

- **關閉 TF32** 是保證跨平台 reproducibility 的最簡單方案
- ~~所有比較實驗必須在同一 GPU 上進行~~ → FP32 模式下跨平台一致，不再需要
- **論文仍應標明 GPU 型號**，但 TF32=OFF 時結果可復現
- **RTX 5090 (Blackwell) 已驗證**：行為與 Hopper 一致（CV 22.73%），Ada Lovelace 是唯一不受影響的架構
- 三代架構 FP32 完全一致（5.250 ± 0.001），TF32=OFF 是跨平台 reproducibility 的保證

---

## 十五、核心結論

1. **Raw Data 最佳**：不做 QA 過濾，讓模型自然學習各種偏振強度
2. **Oracle-Real Gap 是關鍵**：Polarization Volume 消除了這個問題
3. **三條鐵律必須遵守**：pol 不進 fnet、不只做 spatial gate、只在 disparity-aware 空間作用
4. **Post-Corr 有天花板**：V2-A~D 後期表現類似
5. **Pre-Corr 更穩定**：V2-E 訓練曲線波動更小
6. **偏振的價值在 Real 模式**：Real 模式下偏振優勢更明顯（86.7% vs 47.9%）
7. **DID 需要穩定的 base model**：Exp #41 顯示 DID 在 base model 未收斂時反而增加優化難度（EPE 42px 停滯、gamma_eff 退化）。狀態驅動的 LR 控制理論上合理，但前提是 base model 已有基礎匹配能力。**狀態：凍結**
8. **Median 比 Mean 更可靠**：outlier 會嚴重拉高 Mean，Median 更能反映真實表現
9. **TF32 在 Hopper/Blackwell 上行為異常**：Baseline 在 H200/5090 上 TF32 ON/OFF 差異 50%，4090 上僅 0.2%。三代架構 FP32 完全一致（5.250±0.001）。**所有 evaluation/training 必須關閉 TF32**
10. **Pol Volume 存在 Disparity-Pol Crosstalk**：disparity 錯位時不同物件亮度差被誤判為偏振信號，灰階加劇此問題（特徵區分力弱 + 無 color 辨別假信號）
11. **灰階是結構性瓶頸**：偏振 stereo 犧牲 color 換取 pol 資訊，但 pol module 自己又依賴準確 disparity — 引入 RGB 可打破此 chicken-and-egg
12. **V6 渲染器解決 RGB 瓶頸**：Mitsuba `spectral_polarized` + `rgb` film 輸出 15ch RGB Stokes，`I_∥(RGB)` / `I_⊥(RGB)` 各 3ch — 打破灰階 chicken-and-egg。待消融實驗 (RGB+pol vs RGB-only) 量化 pol 的邊際價值
13. **Context Encoder 是被忽略的介入點**：RAFT-Stereo 的 cnet 只看 left image，PIDS 1.x 從未改動。context 是 region-level 注意力引導，是 PIDS 2.0 注入 pol 語義資訊的核心位置
14. **PIDS 2.0 兩個候選架構**：(A) Two-Pass — V2-B 產出 disp₁ 對齊後注入 pol-aware context，避免 Oracle-Real Gap；(B) Coarse-Scale — 低解析度 pol 特徵直接注入 cnet，單 pass 極輕量。兩者獨立實驗比較
15. **偏振片近似 wavelength-independent**：交叉偏振改變的是 intensity 不是 spectral shape，chromaticity 近似不變 → RGB 裁判在正常 SNR 下有效。Mitsuba 的理想偏振比現實更極端（完全消光 vs 部分消光），需 Sensor Realism Augmentation（消光比洩漏 + 噪聲 + clipping）縮小 synthetic-to-real gap
16. **Oracle 2.44 px 是偏振天花板（論文素材）**：Oracle mode（GT warp）的 2.44 px 是 **mean**（含 outlier），證明偏振在理想條件下的極限。V2-E median 2.333 px 已打贏 Oracle mean → 典型表現已達極限，瓶頸是 outlier。Dual-Stream 有 Oracle-Real Gap (+103%)，Pol Volume 消除了 Gap，Two-Pass 目標是在 Gap-free 下逼近 Oracle 的 outlier 控制能力

---

## 十六、PIDS 2.0 架構規劃（構想階段）

### 16.1 版本定義

```
PIDS 1.x (灰階時代):
  輸入: 灰階 I∥, I⊥
  架構演進: Dual-Stream → Pol Volume V1 → V2-A~D → V2-E
  改動範圍: 只動 correlation volume
  Context Encoder: 原封不動

PIDS 2.0 (RGB 時代):
  輸入: RGB I∥(3ch), I⊥(3ch)
  架構: 兩個候選方案獨立實驗
  改動範圍: correlation + context encoder
  Pol 角色: 從「唯一線索」→「RGB 的輔助裁判」
```

### 16.2 動機

V6 渲染器完成後，面臨的核心問題：**直接把 RGB 丟進現有架構不會自動變好**。回顧 PIDS 歷史，pol 也是經歷多代架構演進才發揮作用。RGB 升級需要對應的架構創新。

回讀 RAFT-Stereo 論文後的關鍵發現：PIDS 1.x 只改了 correlation volume，**Context Encoder 從未被動過**。

### 16.3 Context Encoder 的角色

```
cnet(left) → context (B, 128, H/4, W/4) + hidden (B, 128, H/4, W/4)

context 的兩個用途:
  1. 初始化 GRU hidden state → 決定迭代起點
  2. 每個 GRU iteration 都注入 → 引導「往哪裡看」

性質: 一次性計算、region-level、注意力引導
      不需要 pixel-perfect，是注入 pol 語義的理想位置
```

### 16.4 65mm 基線限制與 Dual-Stream 陷阱

不能直接 concat 右圖進 cnet — 回到 Oracle-Real Gap：

| 方案 | 對齊方式 | 問題 |
|------|----------|------|
| concat(left, right) 進 cnet | 無對齊 | 每個 (x,y) 的左右 ch 不是同一物理點 |
| concat(left, warp(right, GT_d)) | 用 GT | Oracle-Real Gap |
| concat(left, warp(right, pred_d)) | 用預測 d | **如果 pred_d 來自自己 → 可行** |

兩個候選架構用不同方式解決此問題。

### 16.5 候選架構 A: Two-Pass (V2-B + Pol-Context)

**核心思想**：Pass 1 做幾何搜尋，對齊後萃取 pol，Pass 2 做 pol-aware 精修。

```
Pass 1 — 幾何搜尋 (N=12 iter)
  fnet(L), fnet(R) → fmaps (只算一次)
  cnet(L) → context₁, hidden₁
  corr_pyramid + V2-B pol_corr
  GRU₁ × 12 → disp₁,  L₁ = 0.3 × L(disp₁, GT)

Between — pol 萃取
  right_warped = warp(R, disp₁.detach())
  pol_diff = (L - right_warped) / (L + right_warped + ε)  # normalized contrast
  pol_feat = PolEncoder(pol_diff)            # 3→Cp(16~32), 小 RF, 不下採樣

Pass 2 — pol-aware 精修 (M=4~6 iter)
  context₂ = cnet(L) + Wc(pol_feat)        # additive
  hidden₂  = cnet_h(L) + Wh(pol_feat)      # hidden 也注入
  corr_pyramid 重用 (vanilla, 無 pol_corr)
  GRU₂ × M, 從 disp₁.detach() 出發 → disp₂  [GRU₂ ≠ GRU₁]
  L₂ = 1.0 × L(disp₂, GT)

Total = L₁ + L₂ | 訓練: end-to-end + L₂ 注入係數漸進
```

**設計決策摘要**：
1. **normalized contrast + PolEncoder** — `(I∥-I⊥)/(I∥+I⊥+ε)` 為 scale-invariant 偏振度（近似 DoLP），PolEncoder 只做空間平滑 + channel projection。本質是 glass indicator map
2. **context + hidden 都注入** — hidden 是 GRU 初始記憶，只改 context = pol 看得到但改不動
3. **從 disp₁ 出發** — Pass 2 是 refine 不是 re-search
4. **GRU₁/₂ 權重獨立** — 搜尋 vs 修正是不同 dynamics
5. **disp₁ 必須有 loss** — detach 阻斷梯度，Pass 1 靠自己的 L₁
6. **Pass 2 無 pol_corr** — pol_corr 是搜尋用的，精修階段不再需要
7. **M < N** — refinement ≈ search 的 1/3~1/2，避免 over-correction
8. **軟凍結** — L₂ pol 注入係數漸進 + Pass 2 LR 輕微壓低，不 hard 凍結

- 計算成本：~1.5x baseline（Pass 2 只 4~6 iter）
- 避免 Oracle-Real Gap：訓練和推論都用自己的 disp₁

### 16.6 候選架構 B: Coarse-Scale Pol Context

**核心思想**：低解析度下計算 pol 特徵（不需精確對齊），注入 cnet，單 pass。

```
coarse_pol = CoarsePolEncoder(downsample(left), downsample(right))
cnet(left, upsample(coarse_pol)) → context, hidden
CorrBlock → GRU × N → disp
```

- 對齊問題的誠實評估：1/16 解析度下 disparity 仍佔圖寬 25%，不能忽略
- 但提供 region-level 統計差異資訊，CoarsePolEncoder 的 receptive field 可能補償
- 計算成本：~1.15x baseline

### 16.7 兩架構比較

| | Two-Pass (A) | Coarse-Scale (B) |
|--|-------------|------------------|
| Pol 精度 | pixel-aligned | region-level |
| 對齊保證 | 有（顯式 warp） | 無（靠 receptive field） |
| 計算成本 | ~1.5x | ~1.15x |
| 新增參數 | PolEncoder+Wc/Wh+GRU₂ | CoarsePolEncoder ~0.1M |
| 風險 | disp₁ 品質影響 Pass 2 | 資訊太粗可能無效 |
| 實作複雜度 | 高（雙 GRU、雙 loss、軟凍結） | 低（多一個小 encoder） |

### 16.8 三條鐵律檢查

兩個架構都符合：
1. ❌ pol 不進 fnet — 未違反
2. ❌ pol 不只做 spatial gate — 未違反（context 注入 GRU，GRU 在 disparity-aware 空間操作）
3. ✅ pol 只能在 disparity-aware 空間作用 — 符合

### 16.9 PIDS 2.0 實驗矩陣

灰階時代已證明 vanilla RAFT-Stereo 不會主動利用 pol 訊號（Exp #13/#14: 隱式偏振僅改善 7%），無 pol module 的實驗不再重複。所有 2.0 實驗均以 V2-B (pol module) 為基礎。

```
Exp 2.0-A:  V2-B (mixed)       — RGB + pol module 基準
Exp 2.0-A': V2-B (all-pol)     — 驗證是否需要 nopol 數據
Exp 2.0-C:  Two-Pass           — 候選架構 A (V2-B + Pol-Context)
Exp 2.0-D:  Coarse-Scale       — 候選架構 B (Coarse Pol Context)

對照組:
  Exp #22: Baseline（灰階 RAFT-Stereo）
  Exp #39: PIDS 1.x 最佳（灰階 V2-E）

比較:
  A vs #22:  RGB+pol vs 灰階（整體升級效果）
  A vs A':   mixed vs all-pol（決定數據策略）
  A vs C:    Two-Pass 額外價值
  A vs D:    Coarse-Scale 額外價值
  C vs D:    兩架構直接比較
  全部 vs #39: PIDS 2.0 vs 1.x
```

**數據策略決策**：A vs A' 結果決定後續實驗使用 mixed 或 all-pol。
若差距 < 0.5 px → 全線改 all-pol，Two-Pass 設計大幅簡化（無需 nopol 防護邏輯）。

### 16.10 研究路線

```
階段 0 (目前): 資料準備
  ✅ V6 渲染器完成
  → 大規模渲染 18000 場景
  → QA 篩選

階段 1: 數據策略驗證 + RGB 基準
  Exp 2.0-A (mixed) + Exp 2.0-A' (all-pol)
  → 同時建立 RGB 基準數字 + 決定數據組成

階段 2a: Two-Pass (Exp 2.0-C) ← 獨立實驗
階段 2b: Coarse-Scale (Exp 2.0-D) ← 獨立實驗
  → 兩者獨立測試，不合併
  → 比較各自對 Exp 2.0-A 的增量

階段 3: 結論
  → 選擇最佳架構作為 PIDS 2.0 正式版
```

---

## 十七、偏振渲染真實性與 Sensor Realism Augmentation

### 17.1 偏振片的物理行為

偏振片主要影響 intensity，近似 wavelength-independent。交叉偏振下：
- chromaticity（顏色比例）近似不變
- 變化的是亮度/對比，不是「顏色變成另一個物體」
- 「RGB 裁判」在正常 SNR 下可以工作

現實中交叉偏振不是 100% 消光：extinction ratio 通常 1:100 ~ 1:10,000。鏡面反射產生部分偏振（非完美線偏振），多次散射會去偏振。結果：I⊥ 不會完全黑掉。

### 17.2 Mitsuba vs 現實

| 面向 | Mitsuba | 現實 | 影響 |
|------|---------|------|------|
| 偏振光學 | ✅ Fresnel, Mueller, 部分偏振 | 同 | 無 gap |
| 偏振片 | 理想（extinction = ∞） | 有限 1:100~1:10,000 | Mitsuba **更極端** |
| 感測器噪聲 | 無 | shot + read noise | 低光區 chromaticity 不可靠 |
| 動態範圍 | 無限 (float32) | 8-12bit | 高光 clip |

**結論**：Mitsuba 的偏振比現實更極端 — 模型訓練時看到的是「最難情況」。

### 17.3 Sensor Realism Augmentation

在 training pipeline 加入三項 augmentation 縮小 synthetic-to-real gap：

| Augmentation | 模擬 | 公式 | 參數 |
|-------------|------|------|------|
| 消光比洩漏 | 非理想偏振片 | `I⊥' = I⊥ + I∥ / ε` | ε ∈ [100, 10000] |
| 感測器噪聲 | Shot + Read noise | `I' = I + poisson(I×g) + N(0,σ)` | g, σ 隨場景隨機 |
| Clipping | 有限動態範圍 | `I' = clip(I, 0, sat)` | sat ∈ [0.8, 1.0] |

這些 augmentation 對 Stage I→II 的泛化至關重要。

### 17.4 對架構設計的影響

- Specular-Invariance Auxiliary Loss **可能不需要**：chromaticity 本來就近似不變
- RGB 裁判的精確定位：提供高維外觀特徵，在正常 SNR 下區分同物體/異物體；低照度下失效（可管理）
- Two-Pass / Coarse-Scale 設計不需因此改變

---

*最後更新: 2026-01-31*

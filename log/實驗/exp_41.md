## Exp #41: Directional Impulse Descent (DID) (2026-01-29)

### 背景

V2-E 訓練過程中觀察到：
1. 曲線**非常穩定**，但中間會有暫時的 plateau（例如 Step 40500~41500 的 8.7 px 跳升）
2. OneCycle 是**時間驅動**的，無法感知 plateau
3. 需要**狀態驅動**的機制來逃離 local minimum

### DID 的核心理念

```
OneCycle = 全局退火曲線 (reference trajectory)
DID      = 狀態驅動的局部脈衝 (event-driven impulse disturbance)
```

**一句話總結**：OneCycle 管整體探索–收斂節律，DID 負責在錯誤 basin 內的瞬時能量注入。

### 物理類比

球在 loss landscape 的 saddle point 附近：

```
┌────────────────────────────────────────┐
│         ╱╲                             │
│        ╱  ╲  ← ridge (脊線)            │
│       ╱    ╲                           │
│   ○ ←───────→ ○  球來回滾              │
│      同一坡面，方向一致                 │
│      但動能不足以翻過脊線              │
└────────────────────────────────────────┘
```

**關鍵洞察**：
- 不是「沒有下降」，而是「一直想下降但下不去」
- `sign(∂L/∂t)` 一致 = 有方向意圖
- `|∂L/∂t|` 很小 = 動能不足

### 狀態機設計

```
┌─────────┐    trigger    ┌─────────┐    T_impulse    ┌──────────┐
│ NORMAL  │──────────────►│ IMPULSE │───────────────►│ COOLDOWN │
└─────────┘               └─────────┘                └──────────┘
     ▲                                                    │
     │                     ┌───────────┐                  │ T_cool
     │         P cycles    │ PROTECTED │◄─────────────────┘
     └─────────────────────┴───────────┘
```

### 觸發條件（同時滿足）

**必要條件：**
1. **Train slope plateau**: `median(s_recent) < ε`，其中 `ε = percentile(s_history, 10%)`
2. **Moving-best stale** (改進 D): val best 連續 4 個 validation cycles 沒更新
3. **Direction consistent**: slope EMA 方向穩定（用 EMA + sign，不用 all()）
4. **Not in late phase**: `progress < 0.75`
5. **Under trigger limit**: `trigger_count < 3`
6. **Not in protection**: 距離上次觸發已過 3 個 validation cycles

**輔助證據（有助判斷，非必要）：**
- Val oscillating: val Glass EPE 在 ±0.5 px 內震盪（改進 D: 不再是必要條件）

### LR 控制

**改進 B: 漸進式 Impulse**
| 狀態 | Pol/Update LR | Backbone LR | 說明 |
|------|---------------|-------------|------|
| NORMAL | `lr_base` | `lr_base` | 正常跟隨 OneCycle |
| IMPULSE (1st) | `lr_base × 2.0` | `lr_base × 1.2` | 第一次較保守 |
| IMPULSE (2nd+) | `lr_base × 3.0` | `lr_base × 1.2` | 後續用完整脈衝 |
| COOLDOWN | `lr_base × 0.1` | `lr_base × 1.0` | 快速阻尼，backbone 直接回 reference |
| PROTECTED | `lr_base` | `lr_base` | 回歸正常，觀察效果 |

**改進 C: 參數群分離**
- **Pol modules** (`pol`): 完整脈衝 (×2.0/×3.0)，這是真正需要擾動的部分
- **Update block** (`update`): 完整脈衝，與 pol 相關的 decoder
- **Backbone** (`fnet`, `cnet`): 只輕微擾動 (×1.2)，保護 ImageNet 特徵

理由：pol-only 驗證集表示 pol module 是瓶頸；backbone 已經收斂到好的特徵空間，不應大幅擾動。

**改進 E: EPE-Adaptive Impulse**

EPE 越低代表模型越精細，脈衝應該自動縮小，避免過度擾動。

```
γ_effective = 1.0 + (γ_raw - 1.0) × clamp(epe_smooth / epe_ref, 0, 1)

例：γ_raw = 3.0, epe_ref = 8.0

epe_smooth = 10 px → γ = 3.0  (full)
epe_smooth =  8 px → γ = 3.0  (full)
epe_smooth =  5 px → γ = 2.25
epe_smooth =  3 px → γ = 1.75
epe_smooth =  1 px → γ = 1.25 (gentle)
```

**保險絲 1: Smoothed EPE**
- 用 `epe_smooth = EMA(val_glass_epe, α=0.4)` 而非即時值
- 避免單次 val 抖動 (±0.2 px) 導致 γ 跟著抖
- 讓 DID 的行為「有智慧」而非「神經質」

**保險絲 2: γ 變化率限制**
- `γ_eff = clamp(γ_eff, γ_prev - Δγ_max, γ_prev + Δγ_max)`
- `Δγ_max = 0.3`
- 防止 EPE 因一次 val 偏低導致 γ 突降，打斷訓練節奏

### CLI 參數

```bash
--did                          # 啟用 DID
--did_epsilon_pct 0.10         # train slope threshold percentile
--did_train_window 300         # train slope EMA window
--did_val_m 5                  # val oscillation window (輔助證據)
--did_delta 0.5                # val EPE tolerance (px)
--did_gamma_up 3.0             # impulse multiplier (2nd+ trigger)
--did_gamma_up_first 2.0       # impulse multiplier (1st trigger) [改進 B]
--did_gamma_backbone 1.2       # backbone impulse multiplier [改進 C]
--did_gamma_down 0.1           # cooldown multiplier
--did_t_impulse 200            # impulse duration (steps)
--did_t_cool 100               # cooldown duration (steps)
--did_cooldown_mode relative_base  # or 'relative_impulse'
--did_max_triggers 3           # max triggers per training
--did_protection 3             # protection period (val cycles)
--did_late_lock 0.75           # disable after 75% progress
--did_best_stale 4             # cycles since best before trigger [改進 D]
--did_epe_ref 8.0              # EPE reference for adaptive γ [改進 E]
--did_epe_smooth_alpha 0.4     # smoothed EPE α [保險絲 1]
--did_delta_gamma_max 0.3      # max γ change per impulse [保險絲 2]
```

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/mixed_dataset_exp28/data \
    --output_dir ./checkpoints_exp41_v2e_did \
    --pol_volume_v2e \
    --pol_weight_hidden 8 \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 40000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    --did \
    --did_epsilon_pct 0.10 \
    --did_train_window 300 \
    --did_val_m 5 \
    --did_best_stale 4 \
    --did_gamma_up 3.0 \
    --did_gamma_up_first 2.0 \
    --did_gamma_backbone 1.2 \
    --did_t_impulse 200 \
    --did_t_cool 100 \
    --did_max_triggers 3 \
    --did_protection 3 \
    --did_late_lock 0.75 \
    --did_epe_ref 8.0 \
    --did_epe_smooth_alpha 0.4 \
    --did_delta_gamma_max 0.3 \
    > train_exp41.log 2>&1 &
```

### TensorBoard 記錄

```
did/state              # 0=NORMAL, 1=IMPULSE, 2=COOLDOWN, 3=PROTECTED
did/trigger_count      # 已觸發次數
did/train_slope_ema    # train slope 的 EMA
did/val_slope_ema      # val slope 的 EMA
did/epsilon            # 當前 threshold
did/lr_base            # OneCycle 的 LR
did/lr_effective       # 應用 DID 後的 LR
did/lr_multiplier      # 當前乘數
did/val_baseline       # val baseline (用於 spike filtering)
did/epe_smooth         # smoothed EPE (EMA) [改進 E / 保險絲 1]
did/gamma_eff          # 實際使用的 γ (EPE-adaptive 後) [改進 E]
```

### 設計決策

| 決策點 | 選擇 | 理由 |
|--------|------|------|
| ε 定義 | percentile(s_history, 10%) | 自適應、scale-free |
| Direction check | EMA + sign | all() 對 noise 太敏感 |
| γ_down 基準 | relative_base (可切換) | 不偏離 OneCycle 軌跡 |
| 保護期 | 3 val cycles | 等待脈衝效果顯現 |
| 最大觸發 | 3 次 | 超過代表架構問題 |
| 第一次 γ_up | 2.0 (非 3.0) | [改進 B] 避免 bf16 grad overflow |
| backbone γ | 1.2 (非 full impulse) | [改進 C] 保護 ImageNet 特徵 |
| 主觸發條件 | best stale (非 oscillation) | [改進 D] pol-only val 較安靜，oscillation 太難滿足 |
| EPE-adaptive γ | 連續縮放 (非離散閾值) | [改進 E] EPE 低時自動減小脈衝 |
| smoothed EPE | EMA α=0.4 | [保險絲 1] 避免 val 抖動導致 γ 抖 |
| γ 變化率限制 | Δγ_max=0.3 | [保險絲 2] 防止 γ 突變打斷節奏 |

### 與 Exp #40 的關係

Exp #41 = Exp #40 (Robust Checkpoint) + DID

| | Exp #39 | Exp #40 | Exp #41 |
|--|---------|---------|---------|
| Checkpoint | Mean | Median + P90 | Median + P90 |
| DID | ✗ | ✗ | **✓** |

### 訓練進度

初次 60k 運行已砍掉（早期數據與 Exp #40 無異），改以 40k steps 重新啟動。

**變更**：總訓練 steps 從 60000 縮減至 **40000**。原因：
1. OneCycle LR 策略相對激進 (lr=0.0003)，60k 下 peak LR 要到 step ~18000 才到達，浪費前期時間
2. Exp #40 數據顯示 val median 在 step 35k-40k 已進入穩態，後 20k steps 邊際收益極低
3. 40k steps 下 peak LR 提前至 ~step 12000，DID 窗口 (0~30000) 更有效

### 前期結果 (Step 19500 / 40000)

| 指標 | 數值 | 走勢 | 判定 |
|------|------|------|------|
| EPE Smooth | 42.12 px | 51→41.5 (5k) → **42.12 (8k-19.5k 停滯)** | ❌ |
| Epsilon | 0.079 | 0.035 → 0.079 (持續上升) | ❌ |
| Gamma_eff | 1.0 | 恆定不變，從未偏離 | ❌ |

**診斷**：
1. **EPE 42 px = 未學到立體匹配**（正常 <5 px），Step 8000 後 11500 steps 零進展
2. **gamma_eff = 1.0 恆定** → adaptive gamma 退化，對所有 GRU iteration 等權
3. **epsilon 持續上升** → 惡性循環：模型不準 → DID 擴大探索 → 無效 → 繼續擴大
4. **結論**：DID 在 base model 未收斂的情況下增加了不必要的優化複雜度

### 狀態：🧊 凍結 (2026-02-01)

前期結果顯示 DID 需要更多調適工作，但主線已轉向 PIDS 2.0 架構規劃（Two-Pass / Coarse-Scale Pol Context）。為避免影響主線進度，DID 相關工作暫時凍結。

**後續可能方向**（待 PIDS 2.0 穩定後）：
- 先讓 base model 收斂到合理 EPE（<10 px），再引入 DID
- 檢查 DID 與 Curriculum 的交互作用是否產生衝突
- 考慮簡化 DID 設計（減少同時學習的超參數數量）

---

*建立日期: 2026-01-29*
*凍結日期: 2026-02-01*

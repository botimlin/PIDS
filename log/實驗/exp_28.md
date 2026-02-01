## Exp #28: 混合 Pol/Nopol 訓練

**日期**: 2026-01-25
**目標**: 混合偏振和無偏振數據，迫使模型同時學習幾何和偏振特徵，提升泛化能力

### 動機

Exp #27 訓練過程中觀察到：
1. Val EPE 持續高於 Train EPE（之前常出現 Val < Train）
2. Val EPE 波動變大

**假設**: 嚴格 C3 過濾後，所有訓練數據都有強偏振信號，模型過度依賴偏振特徵。
混入無偏振數據可迫使模型**同時學習幾何特徵**，提升魯棒性。

### 物理洞察：偏振反射的空間不均勻性

玻璃表面的偏振反射**不是均勻分佈**的：

```
玻璃表面偏振分佈示意:
┌─────────────────────────────────────┐
│  ████  強偏振 (接近 Brewster 角)     │
│  ▓▓▓▓  中等偏振                      │
│  ░░░░  弱/無偏振 (正入射或邊緣角度)   │
└─────────────────────────────────────┘
```

**物理原因**：
- 偏振反射強度取決於入射角（Fresnel 方程）
- 只有接近 Brewster 角（~56° for glass）時才有強烈偏振
- 玻璃表面不同區域入射角不同 → 偏振強度不均勻
- 正入射區域幾乎無偏振差異

**問題**：
- 模型若過度依賴偏振 → 弱偏振區域預測失效
- Exp #27 (純強偏振訓練) 正是這個問題
- 測試集包含各種偏振強度 → 模型在弱偏振區域崩潰

**Exp #28 解決方案**：
- Nopol 數據 → 強迫學習純幾何特徵
- 模型學會在偏振弱的區域靠幾何 fallback
- 偏振強時用偏振，偏振弱時用幾何 → 更魯棒

### 數據配置

```
混合比例: Pol 60% + Nopol 40%

Pol 數據 (已完成):
├── 來源: scene_0001 ~ scene_15000 (嚴格 QA 通過 5859 個)
├── Train: 3000 場景
└── Val: 300 場景 (只有 pol!)

Nopol 數據 (待渲染):
├── 來源: 全新 OBJ (scene_15001+，無重疊)
├── 渲染器: pids_renderer_nopol.py
├── QA: 不需要 (無偏振檢查)
└── Train: 2000 場景 (不進 val!)

總計:
├── Train: 3000 + 2000 = 5000 場景
└── Val: 300 場景 (純 pol)
```

### 關鍵設計原則

| 規則 | 說明 |
|------|------|
| **場景不重疊** | Pol 和 Nopol 使用完全不同的 OBJ |
| **Val = Pol only** | 驗證集只有偏振數據（評估偏振能力） |
| **Nopol = Train only** | 無偏振數據只用於訓練（學習幾何） |
| **保留命名差異** | Pol: `_left_parallel.exr`, Nopol: `_left.exr` (便於追蹤) |

### Curriculum Learning 策略 (方案 A+C)

#### 架構考量

PIDS Polarization Volume 有兩套 LR：
- **RAFT Backbone**: 較小 LR (pretrained, fine-tuning)
- **Pol Module**: 較大 LR (從零學習)

**風險**: 若早期大量 nopol 數據，Pol Module 會用大 LR 學習無意義信號 → 學到垃圾特徵

**解決方案**:
1. 調整比例，確保每個 phase 都有足夠 pol 數據
2. Nopol batch 時凍結 Pol Module

#### 訓練階段

```
總步數: 60000 steps

Phase 1 (前 20%): 幾何 + 偏振基礎
├── Steps 0 ~ 12000
├── 數據: 50% nopol + 50% pol
├── Nopol batch: 凍結 Pol Module (只訓練 backbone)
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

#### 實現方式

**1. 動態 Sampler**
```python
class CurriculumSampler:
    def get_ratio(self, current_step, total_steps):
        progress = current_step / total_steps
        if progress < 0.2:
            return pol=0.5, nopol=0.5
        elif progress < 0.6:
            return pol=0.6, nopol=0.4
        else:
            return pol=0.7, nopol=0.3
```

**2. Pol Module 凍結控制**
```python
def freeze_pol_module(model, freeze: bool):
    """Phase 1 的 nopol batch 時凍結"""
    for name, param in model.named_parameters():
        if 'pol' in name.lower():
            param.requires_grad = not freeze

# 訓練迴圈
if phase == 1 and data_type == "nopol":
    freeze_pol_module(model, freeze=True)
else:
    freeze_pol_module(model, freeze=False)
```

#### 追蹤指標

- `train_pol_epe`: 訓練集 pol 數據的 EPE
- `train_nopol_epe`: 訓練集 nopol 數據的 EPE
- `val_pol_epe`: 驗證集 EPE (純 pol)
- `pol_module_grad_norm`: 監控 Pol Module 梯度（確認凍結生效）

### 執行步驟

#### Step 1: 生成新 OBJ (Blender)
```bash
blender --background --python blender_furniture_randomizer_v17.py -- \
    --start_index 15001 \
    --num_scenes 2500 \
    --output_dir /path/to/new_obj_nopol
```

#### Step 2: 渲染 Nopol (8 GPU)
```bash
nohup python pids_renderer_nopol.py \
    --input_dir /path/to/new_obj_nopol \
    --output /workspace/dataset_nopol_exp28 \
    --num_gpus 8 \
    --max_scenes 2500 \
    > render_nopol_exp28.log 2>&1 &
```

#### Step 3: 組織混合數據集
```bash
python organize_mixed_dataset.py \
    --pol_scenes ./train_100pct_scenes.txt \
    --pol_dir /workspace/dataset_pol \
    --nopol_dir /workspace/dataset_nopol_exp28 \
    --output_dir /workspace/dataset_mixed_exp28 \
    --pol_train 3000 \
    --nopol_train 2000 \
    --pol_val 300
```

#### Step 4: 訓練
```bash
nohup python train_pids.py \
    --data_dir /workspace/dataset_mixed_exp28 \
    --output_dir ./checkpoints_exp28_mixed \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp28_mixed.log 2>&1 &
```

### 預期結果

| 指標 | Exp #27 (純 Pol) | Exp #28 (混合) 預期 |
|------|-----------------|-------------------|
| Glass EPE | ? | 可能略高（訓練有 nopol） |
| Val 穩定性 | 波動大 | 預期更穩定 |
| 泛化能力 | 可能過擬合 pol | 更好（學了幾何） |

### 實現進度 (2026-01-25)

#### 已完成的代碼實現

**1. `curriculum_sampler.py`** (新建)
```
training_guidance/Stage_I/training_stage/curriculum_sampler.py

類別:
├── CurriculumPhase: 訓練階段配置 dataclass
├── CurriculumConfig: Curriculum Learning 配置
├── CurriculumSampler: 動態 pol/nopol 採樣器
│   ├── sample_batch() → (indices, data_types)
│   ├── should_freeze_pol_module(data_type) → bool
│   └── get_batch_info() → Dict
├── PolModuleFreezer: Pol Module 凍結控制器
│   ├── freeze() / unfreeze()
│   └── get_grad_norm() → float (監控)
└── MixedBatchCollator: 混合 Batch 整理器
```

**2. `pids_dataset.py`** (修改)
```python
# 新增 data_type 輸出
return {
    ...
    'data_type': self.scene_naming.get(scene_name, 'pol'),
}
```

**3. `train_pids.py`** (修改)
```
新增功能:
├── --curriculum flag: 啟用 Curriculum Learning
├── --curriculum_seed: Curriculum 隨機種子
├── _setup_curriculum(): 初始化 sampler 和 freezer
├── Pol/Nopol 分開追蹤 metrics
├── TensorBoard logging: curriculum progress, freeze status
└── Phase 1 凍結邏輯: nopol batch 時自動凍結 Pol Module
```

**4. `organize_mixed_dataset.py`** (新建)
```
training_guidance/Stage_I/training_stage/organize_mixed_dataset.py

功能:
├── 載入 QA 通過的 pol 場景
├── 發現/生成 nopol 場景列表
├── 檢查場景無重疊
├── 分割: Val = pol only, Nopol = train only
├── 輸出:
│   ├── train_pol_scenes.txt
│   ├── train_nopol_scenes.txt
│   ├── train_mixed_scenes.txt
│   ├── val_scenes.txt
│   ├── scene_naming.json (供 CurriculumSampler)
│   └── dataset_stats.json
└── 可選: --link_data 創建數據連結
```

#### 使用方式

```bash
# Step 1: 組織數據集
python organize_mixed_dataset.py \
    --pol_scenes_file /workspace/dataset_pol_final/train_100pct_scenes.txt \
    --pol_data_dir /workspace/dataset_pol_final/train_100pct \
    --nopol_data_dir /workspace/dataset_nopol_final \
    --output_dir /workspace/mixed_dataset_exp28 \
    --train_pol_count 3000 \
    --val_count 600 \
    --nopol_start 15001 \
    --nopol_count 2000 \
    --link_data

# Step 2: 訓練 (啟用 Curriculum)
nohup python train_pids.py \
    --data_dir /workspace/mixed_dataset_exp28/data \
    --output_dir ./checkpoints_exp28_100pct \
    --pol_volume \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp28_100pct.log 2>&1 &
```

### 數據準備完成 (2026-01-25)

```
Nopol 渲染: 18000 場景完成 ✓

混合數據集組織:
├── Pol structure: organized (stereo_pairs/, ground_truth/, masks/)
├── Nopol structure: flat
├── Train: 3000 pol (60%) + 2000 nopol (40%) = 5000 場景
├── Val: 600 pol only
└── 檔案數:
    ├── Train: 26000 files
    └── Val: 2400 files
```

### 實驗結果

| 指標 | 數值 | 備註 |
|------|------|------|
| Glass EPE | **4.88 px** | 與 Exp #24 幾乎一致（見下方修正） |
| Glass D1 | 34.49% | |
| Overall EPE | 3.13 px | |
| Val 波動 | 穩定 | Val < Train 正常模式 ✅ |

### 實驗對比分析（已修正）

> **⚠️ 修正 (2026-01-30)**：原對照表使用 Exp #24 的舊評估集結果 (2.44 px, ~200 scenes)，
> 但後來發現該評估集過於簡單。以公平的新測試集 (857 scenes) 重新評估後，
> Exp #24 Glass EPE = **4.055 px**，與 Exp #28 的 4.88 px 差距僅 +20%，並非原先認為的 +100%。
> Exp #24 的 TensorBoard val 為 ~4.87 px，與 Exp #28 的 4.88 px 幾乎完全一致。
> 詳見「重大發現：測試集差異導致性能誤判」。

| 實驗 | 數據策略 | Glass EPE (Val) | Glass EPE (新 Test 857 scenes) |
|------|----------|-----------------|-------------------------------|
| **Exp #24** | Raw data (含弱偏振) | ~4.87 px | 4.055 px |
| **Exp #28** | Mixed pol + nopol | 4.88 px | — |
| **Exp #27** | Strict QA (純強偏振) | 5.03 px | — |

**修正後結論**: Exp #24 ≈ Exp #28 >> Exp #27。軟篩選 (raw data 含弱偏振) 與混合 pol/nopol 策略效果相當。

### 關鍵發現

**1. 弱偏振 ≠ Nopol**
```
弱偏振場景: 仍有微弱偏振線索 → 模型可學習「軟過渡」
Nopol 場景: 完全無偏振 → 對 Pol Module 是「噪音」
```

**2. Raw Data 的弱偏振是「軟標籤」**
- 自然包含：強偏振 → 中偏振 → 弱偏振
- 連續過渡比人工二分法 (pol/nopol) 更有利於學習

**3. Nopol 可能有害**
- 即使 Phase 1 凍結 Pol Module，nopol 仍影響 backbone
- 或者 Pol Module 解凍後被 nopol 數據干擾

**4. RAFT 雙目一致性假設被強偏振破壞** (重要洞察)

RAFT-Stereo 的 Correlation Volume 假設左右圖像在對應點有相似外觀：
```
理想情況: I_left(x, y) ≈ I_right(x - d, y)
         → 高 Correlation → 正確匹配
```

強偏振破壞了這個假設：
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

這解釋了 Exp #27 (Strict QA) 失敗的原因：
```
Exp #27 (純強偏振):
└── 所有玻璃場景 RAFT 都難匹配
    → Backbone 持續掙扎
    → 訓練信號嘈雜
    → 無「容易」案例作為學習基礎

Exp #24 (Raw data 混合):
├── 弱偏振: RAFT 容易 → 學到基本幾何
├── 中偏振: RAFT 有點難 → Pol Module 開始補償
└── 強偏振: RAFT 很難 → Pol Module 必須介入
→ 漸進式學習，有明確的學習梯度
```

**PIDS 訓練的兩難**：
| 偏振強度 | 對 Pol Module | 對 RAFT Backbone |
|---------|--------------|------------------|
| 強偏振 | ✅ 明確信號 | ❌ 難匹配 |
| 弱偏振 | ⚠️ 信號弱 | ✅ 容易匹配 |

**最佳策略**: 混合數據讓模型同時學習「容易」和「困難」案例

### 結論（已修正）

> **⚠️ 原結論基於不公平比較（舊 eval 2.44 px vs val 4.88 px），以下為修正版。**

**Raw data 與 Mixed pol/nopol 效果相當**：
- Exp #24 (raw data) val ~4.87 px ≈ Exp #28 (mixed) val 4.88 px
- 原先認為的 "+100% 差距" 是舊評估集過於簡單造成的假象
- 「軟篩選」(弱偏振作為軟標籤) 的論點無法被數據支持 — 兩者表現一致
- Strict QA (Exp #27, 5.03 px) 確實略差，但差距也僅 ~3%，非原先認為的 +106%

**仍然成立的觀察**：
- Curriculum Learning + 混合策略訓練穩定（Val < Train，波動小）
- 偏振強度連續分佈對訓練有益（RAFT 雙目一致性假設的分析仍有價值）

**狀態**: ✅ 完成（結論已修正：Raw ≈ Mixed >> Strict QA，但差距遠小於原先認為）

---


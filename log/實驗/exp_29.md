## Exp #29: Freeze Backbone First 策略 (提議)

**日期**: 2026-01-26
**目標**: 驗證「先凍結 Backbone 訓練 Pol Module」是否優於當前 Curriculum 策略

### 動機與假設

**Exp #28 訓練觀察**：
```
Phase 1 (0-20%, freeze pol on nopol): Val < Train ✅ 正常
Phase 2/3 (20-100%, all unfrozen):    Val > Train ❌ Plateau + 退化
```

**關鍵發現**: 解凍 Pol Module 後訓練開始退化

**假設 1 (Domain Gap)**: Nopol 數據會讓 Backbone (RAFT-Stereo) 適應「無偏振特徵」場景，這種適應性與 Pol 數據產生 domain gap。Backbone 的變化導致 Pol Module 追著不斷變化的特徵空間學習。

**假設 2 (雙目一致性破壞)**: RAFT 的 Correlation Volume 假設左右圖像相似，但強偏振數據 (I∥ >> I⊥) 嚴重破壞這個假設。如果 Backbone 在強偏振數據上持續訓練而沒有 Pol Module 的補償，會學到錯誤的匹配模式。

```
強偏振對 RAFT 的影響:
├── 玻璃區域: 左亮右暗 → Correlation 低 → Backbone 掙扎
├── 如果 Backbone 解凍: 會試圖「適應」這種不一致
└── 這種適應可能破壞原本良好的幾何特徵提取能力
```

### 新策略：Freeze Backbone First

```
原始策略 (Exp #28):
├── Phase 1 (0-20%):   freeze Pol Module on nopol
├── Phase 2 (20-60%):  全部解凍
└── Phase 3 (60-100%): 全部解凍
→ 結果: 4.88 px (比 raw data 差)

新策略 (Exp #29):
├── Phase 1 (0-60%):   freeze Backbone, 只訓練 Pol Module
│                       └── 70% pol + 30% nopol (獲得幾何多樣性)
└── Phase 2 (60-100%): 全部解凍, 一起微調
                        └── 70% pol + 30% nopol
```

### 理論基礎

1. **Backbone 已預訓練**：RAFT-Stereo 在 SceneFlow 等數據上已有良好的幾何特徵提取能力

2. **Pol Module 需從零學習**：偏振特徵提取是全新任務，需要穩定的 Backbone 作為基礎

3. **避免特徵空間漂移**：如果 Backbone 持續變化，Pol Module 學到的特徵可能與 Backbone 當前表示不匹配

4. **協同微調更穩定**：當 Pol Module 成熟後，再讓 Backbone 適應偏振特徵，兩者可協同改進

5. **保護 Backbone 免受雙目不一致干擾**：強偏振數據的 I∥ >> I⊥ 會破壞 RAFT 的 Correlation 假設。凍結 Backbone 可防止它學到錯誤的匹配模式，讓 Pol Module 先學會補償這種不一致

6. **70/30 混合比例的優勢**：使用 nopol 數據提供幾何多樣性，但 Backbone 凍結確保 nopol 不會破壞預訓練的匹配能力。Pol Module 仍能從 pol 數據學習偏振特徵

### 實現細節

**curriculum_sampler.py 更新**:

```python
# 新增 CurriculumPhase 欄位
@dataclass
class CurriculumPhase:
    freeze_backbone: bool = False  # 新增

# 新策略配置
FREEZE_BACKBONE_PHASES = [
    CurriculumPhase(
        name="Phase 1: Train Pol Module Only",
        start_progress=0.0,
        end_progress=0.6,
        pol_ratio=0.7,          # 70% pol + 30% nopol
        nopol_ratio=0.3,        # 獲得幾何多樣性
        freeze_pol_on_nopol=False,
        freeze_backbone=True,   # 關鍵：凍結 Backbone，保護不受 nopol 干擾
    ),
    CurriculumPhase(
        name="Phase 2: Joint Fine-tuning",
        start_progress=0.6,
        end_progress=1.0,
        pol_ratio=0.7,
        nopol_ratio=0.3,
        freeze_pol_on_nopol=False,
        freeze_backbone=False,  # 解凍
    ),
]
```

**BackboneFreezer 類別**:
- 凍結: fnet, cnet (特徵提取器)
- **保留 update_block 可訓練**：因為 `pol_volume` 架構的 `PolCorrBlock` 是非參數的，只有 `update_block` (GRU) 能學習如何使用 pol correlation features
- 監控: 參數數量統計、gradient norm

**重要修正** (2026-01-26):
```python
# 原本 (會導致所有參數被凍結)
BACKBONE_KEYWORDS = ['fnet', 'cnet', 'update_block', 'corr_fn']

# 修正後 (保留 update_block 可訓練)
BACKBONE_KEYWORDS = ['fnet', 'cnet']
```

**原因**: `PIDSStereoPolVolume` 架構中，`PolCorrBlock` 只是非參數的相關性計算，沒有可學習參數。如果凍結 `update_block`，就沒有任何參數可以學習如何使用偏振特徵。

### 數據集配置

```
Training Set:
  Pol:   3500 (70.0%)   ← 來自 Exp #27 的 QA passed scenes
  Nopol: 1500 (30.0%)   ← scene_15001 ~ scene_16500
  Total: 5000

Validation Set:
  Pol only: 600         ← 只用 pol 評估偏振檢測能力
```

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir ./PIDS_dataset_exp29_mixed/data \
    --output_dir ./checkpoints_exp29_freeze_backbone \
    --pol_volume \
    --curriculum \
    --curriculum_strategy freeze_backbone \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp29_freeze_backbone.log 2>&1 &
```

### 使用方法

```bash
# 使用新策略
python curriculum_sampler.py --strategy freeze_backbone

# 在訓練中
config = CurriculumConfig(strategy='freeze_backbone')
sampler = CurriculumSampler(dataset, total_steps, batch_size, config=config)

# 訓練迴圈
backbone_freezer = BackboneFreezer(model)
for step in range(total_steps):
    if sampler.should_freeze_backbone():
        backbone_freezer.freeze()
    else:
        backbone_freezer.unfreeze()
```

### 預期結果

| 指標 | Exp #24 (Raw) | Exp #28 (Mixed) | Exp #29 (Freeze BB) |
|------|---------------|-----------------|---------------------|
| Glass EPE | **2.44 px** | 4.88 px | ? |
| 訓練穩定性 | 高 | Phase 2/3 plateau | ? |

**假設**：如果 domain gap 假設正確，Exp #29 應優於 Exp #28

**狀態**: ✅ 完成 (2026-01-26)

### 訓練中期觀察 (Step 25500, 42.5%)

**Validation Metrics**:
```
Step   500:  57.38 px
Step 10000:  33.39 px
Step 20000:  23.84 px
Step 25500:  21.37 px
```
- Val Glass EPE 持續下降，**↓ 63%** 改善
- 訓練穩定，無發散

**Phase 1 瓶頸觀察**:
```
Step 25100-26000: Glass EPE 在 21.9 ~ 23.1 px 震盪
```
- Step ~20000 後進入 mini-plateau
- 原因：Backbone 凍結，update_block 能學的已趨飽和
- 還需等到 Step 36000 才進 Phase 2
- **約 16000 steps 的「低效訓練」**

**改進方向**: 見 Exp #30 設計

### 最終結果 (Step 60000, 100%)

**Final Training Output**:
```
Step 60000 | Train Loss: 6.0078 | Val Loss: 6.5178
Glass EPE - Train: 6.85 px | Val: 8.83 px
NON Glass EPE - Train: 2.03 px | Val: 2.36 px
All EPE - Train: 4.23 px | Val: 5.36 px
Phase: Phase 2: Joint Fine-tuning | Pol Ratio: 0.70 | Freeze Pol: False
Best checkpoint updated at step 59000 with val_glass_epe: 8.381
```

**最佳結果**: **Glass EPE = 8.381 px** @ Step 59000

**完整訓練曲線**:
```
Phase 1 (0-60%, Backbone Frozen):
Step   500:  57.38 px
Step 10000:  33.39 px  ← 開始減速
Step 17000:  22.58 px  ← Phase 1 最低
Step 35000:  21.37 px  ← Phase 1 結束前

Phase 2 (60-100%, Full Fine-tuning):
Step 36000:  解凍開始
Step 45000:  11.33 px  ← 快速下降
Step 55000:  10.14 px
Step 59000:   8.38 px  ← Best
Step 60000:   8.83 px
```

**關鍵觀察**:
1. **Phase 1 plateau 確認**: Step 17000~36000 約 16000 steps 空轉
2. **Phase 2 有效**: 解凍後從 21 px → 8.4 px，**↓ 60%**
3. **Val > Train 現象**: 整體 train_glass_epe < val_glass_epe，可能有輕微過擬合
4. **尾端仍有改善空間**: 最後幾千步仍有改善，支持低 LR 精修策略

**與預期比較**:
| 指標 | Exp #24 (Raw) | Exp #28 (Mixed) | Exp #29 (Freeze BB) |
|------|---------------|-----------------|---------------------|
| Glass EPE | **2.44 px** | 4.88 px | 8.38 px |
| 訓練穩定性 | 高 | Phase 2/3 plateau | 穩定但 Phase 1 空轉 |

**結論**: Exp #29 未達預期（比 Exp #28 還差）
- Freeze Backbone 策略本身並未帶來改善
- Raw data + 弱 QA 仍是最佳配置 (Exp #24)
- Phase 1 空轉問題明顯，需改進 → 見 Exp #30

**待執行**: 評估腳本 (evaluate_pids.py)

---


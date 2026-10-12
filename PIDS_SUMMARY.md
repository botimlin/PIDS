# PIDS 開發總結

**專案**：Physics-Informed Deep Stereo (PIDS)
**作者**：Po-Ting Lin
**期間**：2025 年底 ~ 2026-02-26（約 3 個月）
**實驗次數**：45+ 次訓練實驗、60+ 章節開發日誌
**結果**：證明路徑不可行，結案並轉向後續專案 PULSE

---

## 專案目標

使用主動非對稱偏振（左 I∥ / 右 I⊥）的立體相機系統，偵測透明障礙物（玻璃門、壓克力板）的深度。

---

## 渲染器演進（Stage I Sampling）

| 版本 | 關鍵修正 |
|------|----------|
| v1.x | 初版偏振渲染器，I∥/I⊥ 背景嚴重失衡 |
| v2.x | 嘗試 Constant Emitter、放大因子調整，皆失敗 |
| **v3.1.0** | 實施**物理偏振片幾何架構**：LED 發非偏振光 → 穿過獨立偏振片（θ=0°） |
| v3.4.2 | 穩定版本，I∥/I⊥ 背景平衡達 1.0003 |
| **v4.0.0** | 加入紋理支持（Blender v18 + Mitsuba v4） |
| v5.0.0 | 使用單相機 Stokes 參數計算 DoLP（修正不同 3D 點比對錯誤） |
| v5.0.1 | 玻璃比值計算對齊 3D 點 |
| v5.1.7 | 玻璃 mask 精確匹配 + GT 修正 |
| v6 | 相機與光源配置微調 |
| **v7.x** | 座標系統大翻新、自動 QA、相機隨機化、多 GPU 批次渲染 |

渲染解析度 640×480、基線 65mm、FOV 65°、Mitsuba `cuda_ad_rgb` 變體、SPP 65536。

---

## 訓練架構演進（完整歷史）

### PIDS 1.x（RAFT-Stereo 基底）

| 版本 | 核心想法 | 結果 |
|------|----------|------|
| V1 | RAFT-Stereo + 偏振輸入（6 channel） | 模型學不到偏振，退化為 RGB baseline |
| V2 (Dual-Stream) | 左右各一 encoder，concat 融合（Exp #16） | Glass EPE 21.82 px，相對 baseline -48.3% |
| V2 + Cross-Attention (Exp #18) | PooledCrossAttention + Alpha warmup | 19.26 px，但同架構 nopol 也差不多 → 偏振貢獻可疑 |
| V3 (Polarization Volume) | 顯式 4D pol cost volume，不依賴 disparity | 改善有限 |
| V3 Learnable | Exp #25 學習式 pol volume | 與手刻版差距不大 |
| **V2 系列 (A/B/C/D/E)** | Post-Corr gate/residual、scheduled、pre-corr | V2-E 訓到 3 px 級別但 plateau |
| V3 Pol-in-Feature | Pol 直接進入 feature extraction | 無明顯突破 |
| **V4 (True Dual-Stream)** | α per-pixel fusion（Heavy/Mid/Lite 三種） | α 完全無法分化，glass/non-glass 都 ≈0.06 |
| V5 (Cost Concatenation) | 放棄 α，改用 concat | 特徵無法有效融合 |
| V6 (Glass-Aware Side Info) | 注入 glass probability | 些微改善但 overfit test 顯示容量上限 |
| V6.1 / V6.2 | 淺層 learnable encoder / 最簡化 | 都失敗 |

### PIDS 2.0（Two-Pass RGB）

- Pass 1：V2-C 做初步 disparity
- Pass 2：用 pol-aware context 做 refinement（獨立 GRU₂）
- **Exp #43** synthetic val 1.131 px median，但 sim-to-real gap 巨大

### PIDS 3.0（S2M2 轉向）

- 更換 backbone 到 **S2M2**（Scalable Stereo Matching Model）
- 原因：RAFT-Stereo 的三個結構性瓶頸（固定 dot-product correlation、BN 洗掉偏振、GRU 單流容量不足）
- 關鍵優勢：S2M2 用 **LayerNorm** 而非 BatchNorm
- 設計 Polarization-Aware Refinement + Correction module
- **三階段訓練**：Domain Adaptation → Pol-only Learning → Joint Finetune
- 結論：Correction 仍無法從 pol_diff 提取有效結構

### 最後嘗試

- **ComplexCNNEncoder**：使用 Complex-valued encoder 處理偏振 phase → 理論可行但未驗證
- **DID (Directional Impulse Descent)**（Exp #41）：優化器層面嘗試
- **Polarization-Guided Segmented Matching**：分段對齊系統

---

## 趣聞與經典踩坑 🐛

### 1. BatchNorm 把偏振信號「洗澡洗掉」（第二十一章）

```
輸入: I∥=0.8, I⊥=0.2  (4x 差異)
  ↓ Conv + BatchNorm
輸出: feature_∥ ≈ feature_⊥  (差異歸零)
```

BN 對 channel 做 mean/variance 標準化，**第一層就把宏觀強度差異消滅**。所有 V1~V3 系列的 α 學不會，都是因為 BN。移除後 MC noise 又冒出來→「BN 三角困境」。

### 2. CorrBlock 的 L2 Normalization 殺手（第二十九章）

```python
fmap1 = fmap1 / torch.norm(fmap1, dim=1, keepdim=True)  # ← 元凶
```

```
L2 norm 前: I∥=[0.8, 0.6, 0.4], I⊥=[0.4, 0.3, 0.2]
L2 norm 後: 兩者都變成 [0.74, 0.56, 0.37]  ← 完全一樣！
```

Correlation 變平坦 1.0，GRU 無資訊可學。對 learned features (256ch) 沒事，但對 raw pixels (3ch) 致命。

### 3. 獨立正規化毀了所有實驗（第三十四章）

Dataset 預處理：左右圖**各自**用 p99 正規化：

```
left / p99_left  → [0.67, 1.0, 0.83]
right / p99_right → [0.67, 1.0, 0.83]  ← 偏振關係消失
```

**V1~V5 所有版本都中招**，模型從一開始就看不到偏振差異。改為聯合 p99 後才保留 I∥>I⊥。

### 4. TF32 在不同 GPU 架構上的 50% 差異（Exp #42）

原本 H200 Baseline Median 3.489 px，RTX 4090 卻 5.264 px，差 +51%！排查後發現是 TF32：

```
              TF32 ON   TF32 OFF
4090 (Ada)    5.264    5.251   ← 幾乎無差異
H200 (Hopper) 3.504    5.250   ← 差 50%！
5090 (Black.) 3.498    5.250   ← 差 50%！
```

**Ada Lovelace 是唯一 TF32 不影響結果的架構**。Hopper/Blackwell 的 TF32 精度截斷反而成了隱式正則化。解法：

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

### 5. FP16 → BF16 → FP32 + 梯度累積（Exp #8~#10）

- 最初 FP16：Validation Loss 偶爾飆升到 1360.5（NaN/Infinity）
- 改 FP32：數值穩定但 batch_size 被迫降到 8（OOM）
- 改 BF16：H100 最佳，範圍同 FP32
- 紋理版訓練再次不穩 → 最終方案：**BS=8 × accumulation=2 = effective BS=16，FP32**

### 6. max_disparity=192 的 NaN 慘案（Exp #10）

紋理資料 disparity 範圍 90~524，但 dataset 寫死 `max_disparity=192`：

```python
valid_mask = (disparity > 0) & (disparity < max_disparity)
```

→ 超過 192 的像素全部無效 → **沒有像素參與訓練** → Loss/EPE 全部 NaN。改成 576 才恢復。

### 7. Tanh 完全飽和的「放棄學習」（第五十三章）

`correction = tanh(raw) × max_correction=30px`，但 base model 誤差 ~150px：

```
網路發現：全部輸出 +30（常數）能減少誤差到 120
tanh(15) 的梯度 ≈ 0，網路停止學習
```

**任務本身不可能完成**：correction 上限 < base error。這啟發了三階段訓練（先做 domain adaptation 降到 < 30px，再訓 correction）。

### 8. Dual-Stream 的維度 Bug（Exp #15）

```python
# flow 輸出 (B, 2, H, W)，disp_gt 是 (B, 1, H, W)
loss = |flow - disp_gt|  # broadcasting
# Channel 0: |disp - disp_gt| 正確
# Channel 1: |0 - disp_gt| = disp_gt  ← GT 本身被加進 loss！
```

修正維度後（Exp #16）立刻從 baseline 跳到 Glass EPE 21.82 px。

### 9. AvgPool 把偏振峰值洗掉（第三十章）

V4 Heavy 的 3D CNN + AvgPool 在 disparity 維度平均 → 「正確 disparity 的尖峰」被周圍抹平，α predictor 看不到特徵。

### 10. MC Noise 的「假紋理」遮罩（第四十五章後）

Mitsuba 合成玻璃的 Baseline EPE 反而比真實玻璃**低**，原因：

```
真實玻璃:  Left=[清晰 specular A], Right=[完全不同 A']  → correlation 失敗
合成玻璃:  Left=[noisy blob],      Right=[noisy blob]   → MC noise 互相「像」
```

**SPP 越高 transfer 越差**——模型學的其實是 MC noise pattern，不是物理偏振。

### 11. LayerNorm vs BatchNorm 的血淚領悟

整個 RAFT-Stereo 系列失敗後才發現：**S2M2 用 LayerNorm 不會洗掉 batch 內差異**。花了 6 個版本才學到這一課。

### 12. 渲染器裡 `Back_Panel` 被誤判為玻璃

`is_glass_material()` 只看關鍵字：`door`, `partition`, `panel`, `acrylic`... → 家具的 `Back_Panel` 被當玻璃 → 變成巨大偏振產生器 → 鬼影遍地。加入 `non_glass_keywords` 黑名單才解決。

---

## 關鍵技術發現

### 偏振信號的測量困境

| 指標 | 數值 | 問題 |
|------|------|------|
| Peak-to-Mean Ratio | **0.963** | < 1.0，GT 處比平均還低 |
| Peak Sharpness | **-19,295,587,477** | 負數，完全反向 |
| I∥/I⊥ Ratio | 1.168 | 有差異，但太小 |

→ **偏振信號在 correlation space 沒有 sharp peak 可匹配**。

### 物理天花板（核心洞察）

> Stereo matching 在玻璃上匹配的是**背景深度**，不是玻璃表面深度。

**問題不是沒有紋理，而是左右不一致**（photometric consistency 違反）：
- 鏡面反射 view-dependent
- 複合光路（折射+內反射+折射）對角度極度敏感
- 左右相機看到完全不同的 pattern
- 偏振只能改變反射比例，**無法創造可匹配的特徵點**

### 被偏振逼進「表面重建」，而表面重建也無解（更深一層的死路）

**偏振造成的左右不一致，逼得我們只能走表面重建**——這不是主動「放棄」stereo 的選擇，而是被迫的結果：主動非對稱偏振刻意製造左右差異，等於**破壞 photometric consistency**，stereo matching（靠左右對齊找同一點）因此根本不可行，唯一剩下的路就只有「表面重建」而非「靠對齊測深度」。

但被逼進表面重建之後，PIDS 一樣有兩個無法繞過的問題：

**1. 反射表面覆蓋率不足**

偏振 / 主動探測依賴鏡面反射，而玻璃上的鏡面反射只在**特定入射角區域**出現 → 無法覆蓋整片玻璃。就算改用「探測」思路，也只能拿到零星反射區，**做不到全玻璃面的深度量測**。

**2. 承擔與 ClearGrasp 相同的理論風險**

- 表面重建必須依賴**表面法向量**，而我們的方法依然解決不了「**由神經網路推論法向量**」這件事。
- 等於最後又回到 **ClearGrasp** 等方法的同一個未解難題，承擔相同的理論風險——偏振並沒有提供繞過它的捷徑。

---

## 經驗教訓（150+ 條中精選）

1. **Normalization 是偏振殺手**：BatchNorm、LayerNorm、L2 norm、獨立 p99 都會洗掉量值差異
2. **TF32 跨平台一致性不可假設**：Hopper/Blackwell ≠ Ada Lovelace
3. **物理天花板無法用演算法突破**：stereo 在無紋理透明表面的失敗是光學原理
4. **問對問題比找對答案重要**：「偏振 + stereo matching」本身就是錯的問題
5. **先驗證假設再大規模開發**：應用 overfit test 先證明「可學」再投入
6. **偏振適合 segmentation，不適合 matching**
7. **任務可行性分析**：`max_correction >= base_error` 才訓得起來
8. **飽和是危險信號**：tanh/sigmoid 飽和時梯度消失，網路停止學習
9. **MC noise 是雙面刃**：提供合成數據的 photometric consistency，卻是 sim-to-real gap 的根源
10. **知道何時止損**：45 次實驗後果斷 refresh 到 S2M2，證明路不通也是價值
11. **失敗的專案仍有價值**：硬體、校正工具、渲染腳本、踩坑清單都是後續專案的基礎

---

## 資產轉移至 PULSE

**PULSE**（Polarization-Ultrasonic Layered Sensing for depth Estimation）
- 超音波**直接測量**玻璃表面深度（繞過 photometric consistency）
- 偏振相機負責**玻璃區域分割**
- 融合兩者得到完整深度圖

### 可重用資產

- **硬體**：雙相機系統、偏振片、校正治具、同步電路
- **軟體**：校正程式（含 dark/flat field）、偏振影像處理函式庫、Mitsuba 3 渲染腳本、Benchmark 系統
- **知識**：150+ 條經驗教訓、偏振光學實務、NN 正規化對物理信號的影響、跨 GPU 架構一致性檢查清單

---

**PIDS 專案結案 — 2026-02-26**

> PIDS 作為研究專案，成功證明了一條路走不通。這本身就是有價值的科學貢獻——我們現在知道：偏振立體視覺**無法**測量透明物體的表面深度，因為 stereo matching 的前提是「左右 photometric consistency」，而透明物體本質就是 view-dependent 不一致。

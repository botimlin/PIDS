# PIDS 偏振信號 Sanity Check 使用指南

## 核心觀點

> **神經網路不是人，它可以穩定利用人眼幾乎感覺不到的物理偏差**

不要用肉眼判斷偏振信號是否「明顯」，用數值指標！

---

## 快速使用

```bash
# 分析單一場景
python pids_sanity_check.py --scene scene_0002 --input_dir ./output

# 分析所有場景
python pids_sanity_check.py --input_dir ./output

# 保存 JSON 報告
python pids_sanity_check.py --input_dir ./output --save_report
```

---

## 判斷標準

### ✅ 指標 1：DoLP 統計分佈

| 指標 | 閾值 | 意義 |
|------|------|------|
| `mean DoLP` | > 0.02 | 有效偏振信號 |
| `std DoLP` | > 0.01 | 有變化（不是噪點） |
| `p95 DoLP` | > 0.05 | 局部區域有較強偏振 |

**注意**：即使 mean DoLP = 0.005，只要 > 0，就代表有信息！

### ✅ 指標 2：與深度的 Correlation

| 指標 | 意義 |
|------|------|
| `correlation ≠ 0` | 偏振包含幾何信息 |
| `gradient_correlation` | 與深度梯度（表面法向）的相關性 |

**物理原理**：
- 偏振強度 ∝ Fresnel term ∝ 入射角 ∝ surface normal
- 所以偏振和深度/法向「應該」有 correlation

### ✅ 指標 3：邊界聚集程度

| 指標 | 閾值 | 意義 |
|------|------|------|
| `concentration_ratio` | > 1.2 | 偏振信號在物體邊界聚集 |

**物理原理**：透明物體邊界處，折射率變化導致偏振梯度最大

---

## 輸出範例

```
分析場景: scene_0002
============================================================
  [1] DoLP 統計:
      mean:   0.008234  ✗ (閾值: > 0.02)
      std:    0.012456  ✓ (閾值: > 0.01)
      p95:    0.045678
      p99:    0.089012
      max:    0.234567

  [2] 偏振差異 (I∥ - I⊥) 統計:
      |diff| mean: 0.001234
      relative p95:  2.3456%

  [3] 與深度的相關性:
      correlation:      0.034567  ✓
      gradient corr:    0.056789  (與深度梯度)

  [4] 邊界聚集程度:
      聚集比率:           1.456  ✓ (> 1.2 表示聚集)

  ==================================================
  總結判斷:
    △ DoLP 微弱但非零（mean > 0.005）
    ✓ DoLP 有變化（std > 0.01）
    ✓ 與深度有相關性（correlation ≠ 0）
    ✓ 偏振信號在邊界聚集

  🟢 結論: 偏振信號有效，可以用於訓練
     （神經網路可以學習微弱但一致的統計規律）
```

---

## 為什麼「微弱」是好事？

| 如果偏振「非常誇張」 | 如果偏振「微弱但一致」 |
|---------------------|----------------------|
| Network 學「亮=有物體」 | Network 被迫整合多 pixel |
| Shortcut learning | 學統計規律 |
| Stage 2 會崩 | 學幾何而不是亮度 |

**這正是你想要的 representation！**

---

## Network 在學什麼？

1. **法向–偏振關係**：哪裡的表面傾斜比較大
2. **邊界訊號**：哪裡有折射界面
3. **Specular vs Diffuse**：哪些地方是 specular dominated
4. **跨視角不變性**：與視角有關、但與材質弱相關的特徵

這些都是 pixel-wise 很弱，但 **整張圖統計上非常穩定** 的信號。

---

## 下一步：最小訓練實驗

如果 sanity check 通過，可以做 1-2 epoch 的 ablation：

1. 用 `I∥ / I⊥` 訓練
2. 用 unpolarized 訓練（兩張圖一樣）

只要 performance 有差 → 偏振是有資訊的。

## 實驗 #15: Dual-Stream 首次嘗試 (BUG)

**日期**: 2026-01-04
**狀態**: 失敗 (發現 bug)

### 配置

```
GPU: H200
訓練: 50K steps, batch=8, iters=24
參數: pol_dim=128, pol_threshold=0.05, pol_weight=2.0, glass_weight=5.0
```

### 結果 (有 BUG)

| 指標 | 數值 | 問題 |
|------|------|------|
| Val Loss | 2154.68 | 比 baseline (~248) 高 9x |
| Glass EPE | 288.71 px | 比 baseline (~42) 高 7x |

### 發現的 BUG

```python
# train_pids.py 原始代碼
disp_preds = [-f for f in flow_preds]  # f shape: (B, 2, H, W)

# 問題: flow 輸出是 (B, 2, H, W)，但 disp_gt 是 (B, 1, H, W)
# 導致 loss 計算時 broadcasting 錯誤：
# - Channel 0: |disp - disp_gt| 正確
# - Channel 1: |0 - disp_gt| = disp_gt  GT 本身被加進 loss！

# 修正後
disp_preds = [-f[:, :1] for f in flow_preds]  # 只取第一個 channel
```

**結論:** 實驗 #15 數據無效，需重新訓練

---


## 實驗 #8：數值穩定性優化

**日期**: 2025-12-29
**狀態**: 實驗成功

### 問題描述

我們觀察到嚴重的 Validation Loss 震盪現象，偶發性 Loss 瞬間飆升至天文數字 (如 1360.5)。

### 根本原因分析

| 問題 | 原設定 | 後果 |
|------|--------|------|
| **過度訓練** | 1900 組數據跑 1,000,000 步 | 模型死記硬背 |
| **數值不穩定 (Epsilon)** | PyTorch 預設 eps=1e-8 | 梯度爆炸 |
| **混合精度溢出** | 開啟 FP16 | NaN/Infinity |
| **顯存限制** | 切回 FP32 後 OOM | 無法訓練 |

### 我們的優化歷程

| 階段 | 修改項目 | 具體數值 |
|------|----------|----------|
| Phase 1 | 縮減步數 | 1,000,000 → 20,000 |
| Phase 2 | 對齊論文參數 | eps=1e-6 |
| Phase 3 | 強力梯度裁剪 | clip_grad 1.0 → 0.1 |
| Phase 4 | 棄用混合精度 | 移除 --mixed_precision |
| Phase 5 | 調整 Batch Size | 16 → 8 |

### 最終配置 (Golden Command)

```bash
nohup python train_pids.py \
    --data_dir ./dataset \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --lr 0.00002 \
    --clip_grad 0.1 \
    --weight_decay 0.0001 \
    --scheduler cosine \
    --num_steps 20000 \
    --val_freq 100 \
    --val_split 0.2 \
    --d1_weight 0.1 \
    > train_fp32_final.log 2>&1 &
```

### 結果

| 指標 | 優化前 | 優化後 |
|------|--------|--------|
| Loss 穩定性 | 極不穩定 | 穩定下降 |
| Loss 峰值 | 1360.50 | 44.31 |
| 收斂趨勢 | 發散 | 收斂 |

### 關鍵發現

1. **FP32 是必要的** - 高反光區域的極端數值需要全精度處理
2. **eps=1e-6 是必要的** - 論文參數不可省略
3. **強梯度裁剪 (0.1) 有效** - 抑制極端樣本的梯度衝擊
4. **步數要合理** - 小數據集不需要百萬步

---


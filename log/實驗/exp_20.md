## 實驗 #20：Nopol V5/V6 消融實驗（消融實驗 - 控制組）

**日期**: 2026-01-10

> **消融實驗設計**：Exp #20 (Nopol) 與 Exp #21 (Pol) 構成一組消融實驗，用於驗證偏振信息對玻璃深度估計的貢獻。Nopol 為**控制組**（無偏振差異），Pol 為**實驗組**（有偏振差異）。兩者使用**完全相同的架構與超參數**，僅改變輸入數據。

### 目的

驗證偏振信息對深度估計的貢獻。通過保持**完全相同的架構**，只改變輸入數據（有偏振 vs 無偏振），進行公平的消融實驗。

### 實驗設計

| 變數 | Pol 版 (控制組) | Nopol 版 (實驗組) |
|------|-----------------|-------------------|
| 架構 | Dual-Stream ✓ | Dual-Stream ✓ |
| 預訓練 | raftstereo-sceneflow.pth | raftstereo-sceneflow.pth |
| batch_size | 8 | 8 |
| num_steps | 60000 | 60000 |
| lr | 0.0003 | 0.0003 |
| iters | 24 | 24 |
| pol_dim | 128 | 128 |
| pol_threshold | 0.05 | 0.05 |
| pol_weight | 2.0 | 2.0 |
| glass_weight | 5.0 | 5.0 |
| strict_glass_weight | 0.5 | 0.5 |
| **輸入數據** | **I∥, I⊥ (偏振差明顯)** | **Left, Right (無偏振差)** |

### Nopol 訓練命令

```bash
nohup python train_pids.py \
    --data_dir ./PIDS_dataset_nopol_V5 \
    --output_dir ./checkpoints_nopol_v5 \
    --dual_stream \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --num_steps 60000 \
    --lr 0.0003 \
    --iters 24 \
    --pol_dim 128 \
    --pol_threshold 0.05 \
    --pol_sharpness 20.0 \
    --pol_lr_mult 5.0 \
    --pol_weight 2.0 \
    --glass_weight 5.0 \
    --strict_glass_weight 0.5 \
    --val_freq 500 \
    --num_workers 4 \
    > train_nopol_v5.log 2>&1 &
```

### 數據集準備

| 項目 | 數量 | 說明 |
|------|------|------|
| V5/V6 Pol 渲染 | ~4700 場景 | `pids_renderer_textured.py` |
| V5/V6 Nopol 渲染 | ~4700 場景 | `pids_renderer_nopol.py --from-params` |
| Strict Glass Mask | ✓ | 左右視角交集 |
| 訓練/測試分割 | 相同 | 使用 `train_scenes.txt` / `test_scenes.txt` |

### 實驗結果

**狀態**: 已完成 ✓

#### Glass EPE 結果

| Step | Glass EPE (px) | 備註 |
|------|----------------|------|
| 500 | 74.87 | 初始高誤差 |
| 3000 | **93.40** | 最差值 |
| 17500 | 23.19 | 局部最佳 |
| 23000 | 16.56 | 持續改善 |
| 29000 | 13.67 | 局部最佳 |
| 44000 | **10.99** | 全局最佳 |
| 60000 | 52.51 | 最終值（反彈） |

#### Val Loss 結果

| Step | Val Loss | 備註 |
|------|----------|------|
| 500 | 634.23 | 初始 |
| 1500 | **641.67** | 最差值 |
| 29000 | 114.27 | 局部最佳 |
| 44000 | **95.23** | 全局最佳 |
| 60000 | 321.83 | 最終值（反彈） |

#### 關鍵觀察

1. **極端震盪**: Glass EPE 在 10.99 ~ 93.40 px 之間劇烈波動
2. **無收斂趨勢**: 訓練後期 (50k-60k) 誤差反彈至 ~350，無法穩定
3. **最佳 vs 最終**: 最佳 10.99 px @ Step 44000，但最終 52.51 px @ Step 60000
4. **Val Loss 同步震盪**: 與 Glass EPE 呈現相同的不穩定模式

#### Pol vs Nopol 對比

| 指標 | Pol 版 | Nopol 版 | 偏振優勢 |
|------|--------|----------|----------|
| Best Glass EPE | **3.99 px** | 10.99 px | **2.75x** |
| Final Glass EPE | ~4 px | 52.51 px | **13x** |
| 收斂穩定性 | 穩定下降 | 極端震盪 | ∞ |
| 訓練可靠性 | 高 | 低 | - |

### 分析與假設

#### 核心發現

**當 I∥ - I⊥ 在 nopol 數據中不再提供任何有意義的差異時，pol_diff 計算結果變成了一種干擾信號而非有效特徵。**

#### 理論解釋

在 pol 數據中：
```
pol_diff = I∥ - I⊥
        = 玻璃區域強反射 - 玻璃區域弱反射
        = 明確的正向信號 (高對比度)
```

在 nopol 數據中：
```
pol_diff = Left - Right
        = 幾乎相同的亮度 - 幾乎相同的亮度
        = 接近零的噪聲信號 + 視差引起的錯位偽影
```

#### Dual-Stream 架構在 nopol 下的問題

1. **PolEncoder 接收無效輸入**:
   - 設計用於處理偏振差異
   - 在 nopol 中只看到噪聲

2. **pol_diff 引導失效**:
   - `pol_threshold=0.05` 無法區分玻璃與背景
   - 所有區域的 pol_diff 都接近閾值

3. **訓練信號矛盾**:
   - glass_weight 懲罰玻璃區域誤差
   - 但模型無法從 pol_diff 獲得玻璃位置線索
   - 導致梯度方向不穩定

#### 為何最佳性能發生在中期 (Step 44000)

推測：
- 早期：模型嘗試學習 pol_diff，但信號是噪聲
- 中期：模型開始忽略 pol_diff，依靠純視差匹配
- 後期：過擬合噪聲，性能崩潰

這解釋了為何 nopol 最佳 (10.99 px) 約為 pol 最佳 (3.99 px) 的 2.75 倍：
- 2.75x 差距 = 失去偏振特徵的代價
- 模型仍能從傳統立體匹配獲得一定性能，但失去了關鍵優勢

### 結論

消融實驗證實：

1. **偏振信息是 PIDS 成功的關鍵**: 2.75 倍的 Glass EPE 改善
2. **Dual-Stream 架構需要有效的偏振輸入**: 在無偏振數據上表現不穩定
3. **pol_diff 在 nopol 上成為干擾**: 不僅沒幫助，還導致訓練不穩定

**建議**: 未來如需處理無偏振場景，應使用傳統 RAFT-Stereo 架構而非 Dual-Stream。

---


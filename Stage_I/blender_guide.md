# PIDS Blender 場景建立完整指南

本指南說明如何在 Blender 中建立符合 PIDS 系統配置的場景。

---

## 目錄

1. [系統配置總覽](#1-系統配置總覽)
2. [Blender 初始設定](#2-blender-初始設定)
3. [場景幾何配置](#3-場景幾何配置)
4. [物體尺寸參考](#4-物體尺寸參考)
5. [材質設定](#5-材質設定)
6. [命名規則](#6-命名規則)
7. [匯出設定](#7-匯出設定)
8. [檢查清單](#8-檢查清單)

---

## 1. 系統配置總覽

### 核心參數

| 參數 | 數值 | 說明 |
|------|------|------|
| **縮尺比例** | 1:10 | 模型:真實 |
| **光圈** | F1.2 | 固定，景深有限 |
| **對焦距離** | 600 mm | 最清晰位置 |
| **景深範圍** | 528 - 695 mm | ⚠️ 物體必須在此範圍 |
| **基線** | 30 mm | 雙相機間距 |
| **光源入射角** | 55° | 接近 Brewster angle |

### 對應真實世界

| 模型 | 真實 |
|------|------|
| 528 - 695 mm | 5.3 - 6.9 m |
| 250 × 200 mm 場景 | 2.5 × 2.0 m |

---

## 2. Blender 初始設定

### 2.1 開啟 Blender 後，設定單位

1. 右側 **Properties** 面板 → **Scene Properties** (場景圖示)
2. 展開 **Units**：
   - Unit System: **Metric**
   - Length: **Millimeters**
   - Unit Scale: **0.001**

```
設定後，輸入數值直接對應 mm：
  輸入 600 = 600mm
  輸入 0.6 = 0.6mm (注意！不是 0.6m)
```

### 2.2 調整視圖裁剪範圍

因為場景在較遠距離，需要調整視圖：

1. 按 **N** 開啟側邊欄
2. **View** 標籤：
   - Clip Start: **1 mm**
   - Clip End: **10000 mm**

### 2.3 建議的視圖設定

- 按 **Numpad 7** → 俯視圖
- 按 **Numpad 1** → 前視圖
- 按 **Numpad 3** → 側視圖

---

## 3. 場景幾何配置

### 3.1 座標系統

```
Blender 座標系 (右手系)：

        Z (上)
        │
        │
        │
        └──────── X (右)
       ╱
      ╱
     Y (前，相機看向的方向)
     
注意：Blender 預設 Z 朝上，Y 朝前
```

### 3.2 相機位置

**不需要在 Blender 中建立相機，渲染腳本會自動處理！**

腳本設定：
- 相機位置：原點 (0, 0, 0) 附近
- 左相機：(-15, 0, 0) mm
- 右相機：(+15, 0, 0) mm
- 看向：+Y 方向

### 3.3 ⭐ 最重要：景深範圍

**因為 F1.2 光圈，景深只有 167mm！所有透明物體必須放在 Y = 528~695mm 範圍內。**

```
俯視圖 (從 Z 軸往下看)：

                        Y 軸 (深度方向)
                        │
    X ──────────────────┼──────────────────────────────────►
                        │
                  相機 ●│ Y=0
                        │
                        │
                        │ Y=528mm ─────── 景深起點
                        │    ┌─────────────────────┐
                        │    │                     │
                        │    │   透明物體放這裡     │
                        │    │   (玻璃杯、玻璃門)   │
                        │    │                     │
                        │ Y=600mm ── 對焦點（最清晰）
                        │    │                     │
                        │    │                     │
                        │    └─────────────────────┘
                        │ Y=695mm ─────── 景深終點
                        │
                        │ Y=750mm ─────── 背景牆
                        │
                        ▼
```

### 3.4 場景元素 Y 座標對照表

| 元素 | Y 座標 (mm) | 說明 |
|------|-------------|------|
| 相機 | 0 | 原點（腳本處理） |
| **透明物體** | **528 - 695** | ⚠️ 必須在此範圍！ |
| 對焦平面 | 600 | 最清晰 |
| 背景家具 | 700 - 750 | 書架等（可稍模糊） |
| 背景牆 | 750 | 最遠端 |

### 3.5 建立場景底座（地面）

1. **Add → Mesh → Plane**
2. 設定：
   - Location X: **0**
   - Location Y: **610** (場景中心)
   - Location Z: **0** (地面)
   - Scale X: **125** (250mm 寬)
   - Scale Y: **100** (200mm 深)
3. 命名：`ground_floor`
4. **Ctrl+A → Apply Scale**

### 3.6 建立背景牆

1. **Add → Mesh → Plane**
2. 設定：
   - Location X: **0**
   - Location Y: **750**
   - Location Z: **100** (牆中心)
   - Rotation X: **90°**
   - Scale X: **125** (250mm)
   - Scale Z: **100** (200mm 高)
3. 命名：`bg_wall_back`
4. **Ctrl+A → Apply All Transforms**

---

## 4. 物體尺寸參考

### 4.1 透明物體 (1:10 縮尺)

| 物體 | 模型尺寸 | 真實尺寸 | Y 座標 |
|------|----------|----------|--------|
| 玻璃杯 | 7mm 直徑, 12mm 高 | 70×120mm | 528-695 |
| 玻璃瓶 | 8mm 直徑, 25mm 高 | 80×250mm | 528-695 |
| 玻璃花瓶 | 15mm 直徑, 30mm 高 | 150×300mm | 528-695 |
| 玻璃板 | 50×40×0.5mm | 500×400×5mm | 528-695 |
| 玻璃桌面 | 80×80×0.5mm | 800×800×5mm | 528-695 |
| 玻璃門 | 90×150×1mm | 900×1500×10mm | 528-695 |
| 玻璃窗 | 100×80×0.5mm | 1000×800×5mm | 528-695 |

### 4.2 背景家具 (1:10 縮尺)

| 物體 | 模型尺寸 | 真實尺寸 | Y 座標 |
|------|----------|----------|--------|
| 書架 | 80×30×180mm | 0.8×0.3×1.8m | 720-750 |
| 茶几 | 60×40×45mm | 0.6×0.4×0.45m | 550-650 |
| 椅子 | 45×45×90mm | 0.45×0.45×0.9m | 550-700 |
| 沙發 | 150×70×80mm | 1.5×0.7×0.8m | 700-750 |
| 電視櫃 | 120×40×50mm | 1.2×0.4×0.5m | 720-750 |

### 4.3 建立物體步驟範例

**玻璃杯：**
1. Add → Mesh → Cylinder
2. Radius: **3.5** (7mm 直徑)
3. Depth: **12**
4. Location: (20, **600**, 6)  ← Y=600 在景深範圍內
5. 命名：`glass_cup_01`
6. 指定材質：`Glass_Clear`

**玻璃板：**
1. Add → Mesh → Cube
2. Location: (-30, **580**, 40)  ← Y=580 在景深範圍內
3. Scale: (25, 0.25, 20) → 50×0.5×40mm
4. 命名：`glass_panel_01`
5. **Ctrl+A → Apply Scale**
6. 指定材質：`Glass_Clear`

---

## 5. 材質設定

### 5.1 材質命名規則（重要！）

渲染腳本根據材質名稱自動指定物理材質：

| Blender 材質名稱 | Mitsuba 材質 | IOR |
|------------------|--------------|-----|
| `Glass_*` | dielectric | 1.5 |
| `Acrylic_*` | dielectric | 1.49 |
| `Diffuse_*` | diffuse | - |
| `Mirror_*` | conductor | - |

### 5.2 建立玻璃材質

1. 選擇物體 → **Material Properties** (球體圖示)
2. 點擊 **New**
3. 重新命名為 `Glass_Clear`
4. Surface: **Principled BSDF**
5. 設定：
   - Base Color: **(1, 1, 1)** 白色
   - Roughness: **0.0**
   - Transmission: **1.0**
   - IOR: **1.5**

### 5.3 建立漫反射材質

1. **New** → 命名為 `Diffuse_Wood`
2. Surface: **Principled BSDF**
3. 設定：
   - Base Color: 依物體 (木頭約 0.6, 0.4, 0.2)
   - Roughness: **1.0**
   - Specular: **0.0**

---

## 6. 命名規則

### 6.1 物體命名前綴

| 前綴 | 類型 | 範例 |
|------|------|------|
| `glass_` | 玻璃 | `glass_cup_01`, `glass_door` |
| `acrylic_` | 壓克力 | `acrylic_panel` |
| `diffuse_` | 漫反射 | `diffuse_table`, `diffuse_chair` |
| `ground_` | 地面 | `ground_floor` |
| `bg_` | 背景 | `bg_wall_back`, `bg_bookshelf` |

### 6.2 建議的 Collection 結構

```
Scene Collection
├── Environment
│   ├── ground_floor
│   ├── bg_wall_back
│   ├── bg_wall_left
│   └── bg_wall_right
├── Furniture  
│   ├── diffuse_table_01
│   ├── diffuse_chair_01
│   └── bg_bookshelf_01
└── Glass_Objects
    ├── glass_cup_01
    ├── glass_bottle_01
    ├── glass_panel_01
    └── glass_door_01
```

---

## 7. 匯出設定

### 7.1 匯出前準備

1. 選擇所有物體：**A**
2. Apply Transforms：**Ctrl+A → All Transforms**

### 7.2 OBJ 匯出步驟

1. **File → Export → Wavefront (.obj)**
2. 右側設定面板：
   
   **Include:**
   - ✅ Objects as OBJ Groups
   - ✅ Material Groups
   
   **Transform:**
   - Forward: **-Y Forward**
   - Up: **Z Up**
   - Scale: **0.001**
   
   **Geometry:**
   - ✅ Apply Modifiers
   - ✅ Triangulate Faces

3. 選擇儲存位置，命名如 `scene_001.obj`

### 7.3 批次匯出

使用 `blender_batch_export.py` 自動產生多個場景變體。

---

## 8. 檢查清單

### ✅ 建模前設定

- [ ] Unit System = Metric
- [ ] Length = Millimeters  
- [ ] Unit Scale = 0.001
- [ ] View Clip End = 10000mm

### ✅ 場景幾何

- [ ] 地面 Y ≈ 610mm
- [ ] 背景牆 Y = 750mm
- [ ] **所有透明物體 Y 在 528-695mm**

### ✅ 物體

- [ ] 尺寸符合 1:10 縮尺
- [ ] 命名符合規則 (glass_, diffuse_, bg_...)
- [ ] 已指定正確材質

### ✅ 匯出前

- [ ] Apply All Transforms (Ctrl+A)
- [ ] Forward = -Y, Up = Z
- [ ] Scale = 0.001

---

## 快速開始腳本

在 Blender 的 **Scripting** 標籤中執行：

```python
import bpy
import math

# 清除預設物體
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# 設定單位
bpy.context.scene.unit_settings.system = 'METRIC'
bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'
bpy.context.scene.unit_settings.scale_length = 0.001

# ===== 建立地面 =====
bpy.ops.mesh.primitive_plane_add(size=1)
ground = bpy.context.active_object
ground.name = 'ground_floor'
ground.location = (0, 0.610, 0)
ground.scale = (0.125, 0.100, 1)
bpy.ops.object.transform_apply(scale=True)

# ===== 建立背景牆 =====
bpy.ops.mesh.primitive_plane_add(size=1)
wall = bpy.context.active_object
wall.name = 'bg_wall_back'
wall.location = (0, 0.750, 0.100)
wall.rotation_euler = (math.radians(90), 0, 0)
wall.scale = (0.125, 0.100, 1)
bpy.ops.object.transform_apply(rotation=True, scale=True)

# ===== 建立玻璃杯 =====
bpy.ops.mesh.primitive_cylinder_add(
    radius=0.0035,  # 7mm 直徑
    depth=0.012,    # 12mm 高
    location=(0.020, 0.600, 0.006)
)
cup = bpy.context.active_object
cup.name = 'glass_cup_01'

# ===== 建立玻璃板 =====
bpy.ops.mesh.primitive_cube_add(size=1)
panel = bpy.context.active_object
panel.name = 'glass_panel_01'
panel.location = (-0.030, 0.580, 0.040)
panel.scale = (0.025, 0.00025, 0.020)  # 50 x 0.5 x 40mm
bpy.ops.object.transform_apply(scale=True)

# ===== 建立玻璃材質 =====
glass_mat = bpy.data.materials.new(name="Glass_Clear")
glass_mat.use_nodes = True
bsdf = glass_mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Transmission"].default_value = 1.0
bsdf.inputs["Roughness"].default_value = 0.0
bsdf.inputs["IOR"].default_value = 1.5

# 指定材質給玻璃物體
for obj in [cup, panel]:
    obj.data.materials.append(glass_mat)

# ===== 建立漫反射材質 =====
diffuse_mat = bpy.data.materials.new(name="Diffuse_Gray")
diffuse_mat.use_nodes = True
bsdf = diffuse_mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1)
bsdf.inputs["Roughness"].default_value = 1.0

# 指定材質給地面和牆壁
ground.data.materials.append(diffuse_mat)
wall.data.materials.append(diffuse_mat)

print("=" * 50)
print("場景建立完成！")
print("=" * 50)
print(f"地面位置: Y = {ground.location.y * 1000:.0f}mm")
print(f"背景牆位置: Y = {wall.location.y * 1000:.0f}mm")
print(f"玻璃杯位置: Y = {cup.location.y * 1000:.0f}mm")
print(f"玻璃板位置: Y = {panel.location.y * 1000:.0f}mm")
print()
print("下一步:")
print("1. 添加更多物體")
print("2. 確認所有透明物體 Y 在 528-695mm")
print("3. File → Export → Wavefront (.obj)")
```

---

## 常見問題

**Q: 渲染時物體看不到？**
A: 檢查 Y 座標是否在 528-695mm 範圍內。

**Q: 玻璃沒有透明效果？**
A: 確認材質名稱以 `Glass_` 開頭。

**Q: 尺寸錯誤？**
A: 確認：
- Unit Scale = 0.001
- 匯出 Scale = 0.001
- 已 Apply Transform

**Q: 物體位置錯誤？**
A: 匯出前執行 Ctrl+A → All Transforms

---

## 下一步

1. 建立場景
2. 檢查所有設定
3. 匯出 OBJ
4. 執行渲染：
```bash
python pids_renderer_blender.py --scene scene_001.obj --output ./output
```

---

## 9. 場景擺放原則

### 9.1 透明物體多樣性

| 原則 | 說明 |
|------|------|
| **不同深度** | 物體分散在 528-695mm，不要都在同一平面 |
| **不同角度** | 玻璃板要有正面、斜面、側面 |
| **不同大小** | 混合大小物體（7mm 玻璃杯 ~ 90mm 玻璃門） |
| **部分遮擋** | 有些物體被其他物體部分擋住 |

```
好 ✅                           不好 ❌

   ┌─┐                          ┌─┐ ┌─┐ ┌─┐
   │ │    ○                     │ │ │ │ │ │
   └─┘       ┌──┐               └─┘ └─┘ └─┘
        ○    │  │               (全部同深度、同角度)
             └──┘
   (不同深度、角度、遮擋)
```

### 9.2 背景複雜度

| 原則 | 說明 |
|------|------|
| **有紋理** | 背景要有書架、海報、家具等細節 |
| **有顏色變化** | 不要純白或純灰背景 |
| **有邊緣** | 背景中有直線、邊角 |

**原因**：立體匹配需要紋理來找對應點，純色背景會失敗。

### 9.3 光源考量

| 原則 | 說明 |
|------|------|
| **確保玻璃有反光** | 玻璃表面要能接收到光（55° 入射） |
| **避免全黑區域** | 太暗的區域偏振信號弱 |
| **背景適度照亮** | 背景不要太暗 |

### 9.4 避免的情況

| ❌ 避免 | 原因 |
|--------|------|
| 玻璃完全平行相機 | 反射信號弱 |
| 玻璃完全垂直（90°）| 幾乎沒有反射 |
| 多層玻璃重疊 | 過於複雜，真實場景少見 |
| 純色背景 | 立體匹配會失敗 |
| 物體全在同一深度 | 缺乏深度多樣性 |

### 9.5 典型場景組合範例

**場景 A：客廳一角**
```
Y=750mm  ┃ 書架 ┃ 牆壁 ┃
Y=700mm      沙發
Y=650mm           ╔══╗ 玻璃門(斜)
Y=600mm     茶几 ║  ║
Y=550mm  ○杯    ╚══╝    ○瓶
         └─────────────────┘ 地面
```

**場景 B：餐廳**
```
Y=750mm  ┃ 櫃子 ┃ 窗戶 ┃
Y=680mm      ┌────┐ 玻璃隔板
Y=620mm      │桌子│
Y=580mm   ○  └────┘  ○
         杯          瓶
```

### 9.6 數據多樣性建議

每個場景變體應該變化：

| 變化項目 | 範圍 |
|----------|------|
| 玻璃物體數量 | 1-5 個 |
| 玻璃位置 X | -80 ~ +80 mm |
| 玻璃位置 Y | 528 ~ 695 mm |
| 玻璃旋轉 | -30° ~ +30° |
| 背景顏色 | 多種 |
| 家具擺設 | 多種組合 |

### 9.7 場景擺放檢查清單

- [ ] 透明物體有不同深度（Y 座標分散）
- [ ] 透明物體有不同角度
- [ ] 有大有小的透明物體
- [ ] 背景有紋理和顏色變化
- [ ] 有部分遮擋情況
- [ ] 沒有純色大面積區域
- [ ] 玻璃不是完全平行或垂直於相機

---

## 10. 渲染輸出說明

執行渲染腳本後，會產生以下檔案：

### 10.1 每個場景的輸出

```
output/
├── scene_001_I_parallel.exr    # 左相機影像 (I∥)，0° 偏振
├── scene_001_I_cross.exr       # 右相機影像 (I⊥)，90° 偏振
├── scene_001_I_parallel.png    # 預覽圖 (8-bit)
├── scene_001_I_cross.png       # 預覽圖 (8-bit)
├── scene_001_polarization_diff.png  # 偏振差異圖 (視覺化)
├── scene_001_disparity.exr     # 視差圖 (Ground Truth)
└── scene_001_params.json       # 場景參數
```

### 10.2 檔案格式說明

| 檔案 | 格式 | 用途 |
|------|------|------|
| `*_I_parallel.exr` | 32-bit HDR | 訓練輸入 (左圖) |
| `*_I_cross.exr` | 32-bit HDR | 訓練輸入 (右圖) |
| `*_disparity.exr` | 32-bit float | Ground Truth |
| `*.png` | 8-bit | 預覽用，不用於訓練 |
| `*_params.json` | JSON | 場景元資料 |

### 10.3 訓練數據使用方式

```python
# PyTorch DataLoader 範例
import cv2
import torch

# 讀取 EXR
left = cv2.imread('scene_001_I_parallel.exr', cv2.IMREAD_UNCHANGED)
right = cv2.imread('scene_001_I_cross.exr', cv2.IMREAD_UNCHANGED)
disparity = cv2.imread('scene_001_disparity.exr', cv2.IMREAD_UNCHANGED)

# 轉換為 Tensor
left_tensor = torch.from_numpy(left).permute(2, 0, 1)  # [C, H, W]
right_tensor = torch.from_numpy(right).permute(2, 0, 1)
disp_tensor = torch.from_numpy(disparity)
```

### 10.4 偏振影像的物理意義

| 影像 | 符號 | 偏振片角度 | 內容 |
|------|------|------------|------|
| I_parallel | I∥ | 0° | 保留鏡面反射 |
| I_cross | I⊥ | 90° | 抑制鏡面反射 |
| 差異 | I∥ - I⊥ | - | 透明物體明顯 |

透明物體在 I∥ 較亮（有反射），在 I⊥ 較暗（反射被抑制），
這個差異就是網路學習偵測透明物體的關鍵特徵。

---

## 11. 環境光與訓練策略

### 11.1 兩階段訓練策略

| 階段 | 環境光 | 目的 |
|------|--------|------|
| **Phase 1** | ❌ 關閉 | 驗證系統可行性，偏振信號最純淨 |
| **Phase 2** | ✅ 開啟 | 提升泛化能力 (Domain Randomization) |

### 11.2 為什麼需要兩階段？

**Phase 1 (無環境光)：**
- 只有 LED 偏振光源
- 偏振信號最純淨、最強
- Ground Truth 最準確
- 先確認系統原理可行

**Phase 2 (有環境光)：**
- LED 偏振光 + 隨機環境光
- 模擬真實部署環境（可能有其他光源干擾）
- 提升模型對光線變化的魯棒性

### 11.3 環境光配置

在 `pids_config.py` 中：

```python
'ambient_light': {
    'enabled': False,  # Phase 1: 關閉, Phase 2: 改為 True
    'intensity_range': (0.0, 0.1),  # LED 強度的 0-10%
    'color_temp_range_k': (4000, 6500),  # 色溫範圍
}
```

### 11.4 環境光的物理意義

| 光源 | 偏振狀態 | 對玻璃的影響 |
|------|----------|--------------|
| LED (通過偏振片) | 線性偏振 | 產生偏振反射 |
| 環境光 | 非偏振 | 不產生偏振差異 |

環境光會：
- 稀釋偏振信號（降低 I∥ 和 I⊥ 的差異）
- 增加整體亮度
- 模擬真實世界的干擾

### 11.5 建議的訓練流程

```
1. Phase 1: 無環境光
   - 生成 3000-5000 張訓練數據
   - 訓練模型至收斂
   - 驗證偏振原理是否有效

2. Phase 2: 有環境光 (如果 Phase 1 成功)
   - 修改 config: enabled = True
   - 生成額外 2000-3000 張數據
   - 微調模型或混合訓練
   - 測試泛化能力
```

### 11.6 現實採集的對應

| 模擬環境 | 現實採集 |
|----------|----------|
| Phase 1 (無環境光) | 暗室 + 只有 LED 光源 |
| Phase 2 (有環境光) | 一般室內環境 |

**建議**：現實採集時也在可控光源環境下進行，與 Phase 1 模擬一致。

---

## 12. 訓練策略：大型透明物體優先

### 12.1 策略概述

| 階段 | 物體類型 | 縮尺 | 目的 |
|------|----------|------|------|
| **模擬訓練** | 玻璃牆、門、隔板 | 1:10 | 大物體容易製作，訓練基礎能力 |
| **真實部署** | 各種尺寸（含杯、瓶） | 1:1 | 測試泛化能力 |

### 12.2 為什麼這樣做？

**優點：**
1. **微縮模型更容易製作** — 大型平面玻璃只需裁切壓克力板
2. **偏振信號更強** — 大面積玻璃有更明顯的反射
3. **訓練更穩定** — 大物體佔更多像素，特徵更明顯
4. **測試泛化能力** — 看模型能否從「大玻璃」泛化到「小玻璃」

### 12.3 模擬訓練的物體清單

**✅ 建議使用（大型平面玻璃）：**

| 物體 | 模型尺寸 | 真實尺寸 | 製作方式 |
|------|----------|----------|----------|
| 玻璃門 | 90 × 150 mm | 0.9 × 1.5 m | 壓克力板裁切 |
| 玻璃窗 | 100 × 80 mm | 1.0 × 0.8 m | 壓克力板裁切 |
| 玻璃隔板 | 50 × 40 mm | 0.5 × 0.4 m | 壓克力板裁切 |
| 玻璃桌面 | 80 × 80 mm | 0.8 × 0.8 m | 壓克力板裁切 |
| 玻璃櫃門 | 40 × 60 mm | 0.4 × 0.6 m | 壓克力板裁切 |
| 玻璃牆 | 120 × 100 mm | 1.2 × 1.0 m | 壓克力板裁切 |

**❌ 模擬階段不需要（小型物體）：**

| 物體 | 模型尺寸 | 原因 |
|------|----------|------|
| 玻璃杯 | 7 mm | 太小，難製作，留待真實部署測試 |
| 玻璃瓶 | 8 mm | 太小，難製作，留待真實部署測試 |
| 玻璃花瓶 | 15 mm | 較小，可選擇性加入 |

### 12.4 材料準備

**壓克力板規格建議：**
- 厚度：1-2 mm（模型尺寸，對應真實 10-20mm）
- 材質：透明壓克力 (PMMA, IOR ≈ 1.49)
- 購買：文具店、壓克力加工店
- 裁切：美工刀 + 直尺，或請店家代切

### 12.5 Blender 建模調整

只建立大型透明物體：

```python
# 建議的物體
glass_objects = [
    {'name': 'glass_door_01', 'size': (90, 150, 1)},      # 玻璃門
    {'name': 'glass_window_01', 'size': (100, 80, 1)},    # 玻璃窗
    {'name': 'glass_partition_01', 'size': (50, 40, 1)},  # 玻璃隔板
    {'name': 'glass_table_01', 'size': (80, 80, 1)},      # 玻璃桌面
    {'name': 'glass_cabinet_01', 'size': (40, 60, 1)},    # 玻璃櫃門
]

# 不需要建立
# glass_cup (7mm) - 留待真實部署
# glass_bottle (8mm) - 留待真實部署
```

### 12.6 真實部署測試計畫

訓練完成後，用 1:1 真實環境測試：

| 測試項目 | 目的 |
|----------|------|
| 大型玻璃（門、窗） | 驗證訓練效果 |
| 中型玻璃（隔板、桌面） | 驗證尺寸泛化 |
| **小型玻璃（杯、瓶）** | **測試泛化極限** |
| 不同角度 | 測試角度泛化 |
| 不同光線 | 測試光線泛化 |

### 12.7 預期結果

| 物體類型 | 預期表現 |
|----------|----------|
| 大型玻璃 | ✅ 應該表現良好（訓練過） |
| 中型玻璃 | ✅ 應該可以（尺寸相近） |
| 小型玻璃 | ⚠️ 可能需要額外訓練數據 |

如果小型物體表現不佳，可以：
1. 補充小型物體的訓練數據
2. 使用真實採集的小型物體數據微調

---

## 13. 傢俱隨機擺放腳本

### 13.1 腳本功能

`blender_furniture_randomizer.py` 可自動化場景生成：

| 功能 | 說明 |
|------|------|
| 自動建立基礎場景 | 地面 + 背景牆 |
| 隨機放置透明物體 | 1-3 個大型玻璃 |
| 隨機放置傢俱 | 2-4 個背景傢俱 |
| 碰撞檢測 | 避免物體重疊 |
| 批次匯出 | 自動匯出 OBJ |

### 13.2 傢俱種類（5 種）

| 傢俱 | 模型尺寸 (mm) | 真實尺寸 | 顏色 |
|------|---------------|----------|------|
| 書架 (bookshelf) | 80 × 30 × 180 | 0.8 × 0.3 × 1.8 m | 木頭色 |
| 櫃子 (cabinet) | 100 × 40 × 90 | 1.0 × 0.4 × 0.9 m | 深木色 |
| 桌子 (table) | 80 × 50 × 45 | 0.8 × 0.5 × 0.45 m | 淺木色 |
| 椅子 (chair) | 45 × 45 × 85 | 0.45 × 0.45 × 0.85 m | 木頭色 |
| 沙發 (sofa) | 120 × 60 × 70 | 1.2 × 0.6 × 0.7 m | 灰色 |

### 13.3 透明物體種類

| 物體 | 模型尺寸 (mm) | 真實尺寸 |
|------|---------------|----------|
| 玻璃門 (glass_door) | 90 × 150 × 1 | 0.9 × 1.5 m |
| 玻璃窗 (glass_window) | 100 × 80 × 1 | 1.0 × 0.8 m |
| 玻璃隔板 (glass_partition) | 50 × 40 × 1 | 0.5 × 0.4 m |
| 玻璃桌面 (glass_table) | 80 × 80 × 1 | 0.8 × 0.8 m |
| 玻璃櫃門 (glass_cabinet) | 40 × 60 × 1 | 0.4 × 0.6 m |

### 13.4 使用方式

1. 開啟 Blender
2. 切換到 **Scripting** 標籤
3. 點擊 **Open** 開啟 `blender_furniture_randomizer.py`
4. 修改配置（可選）：
   ```python
   CONFIG = {
       'num_scenes': 10,  # 生成場景數量
       
       'glass_objects': {
           'count_range': (1, 3),  # 每場景透明物體數量
       },
       
       'furniture': {
           'count_range': (2, 4),  # 每場景傢俱數量
       },
   }
   ```
5. 點擊 **Run Script** 或按 **Alt+P** 執行

### 13.5 輸出結果

腳本執行後會在 `./scenes/` 目錄產生：

```
scenes/
├── scene_0001.obj
├── scene_0001.mtl
├── scene_0002.obj
├── scene_0002.mtl
├── scene_0003.obj
├── scene_0003.mtl
...
```

### 13.6 配置參數說明

```python
CONFIG = {
    # 生成場景數量
    'num_scenes': 10,
    
    # 輸出目錄
    'output_dir': './scenes',
    
    # 場景範圍 (mm)
    'scene': {
        'width': 250,       # 場景寬度
        'y_min': 528,       # 景深起點 (透明物體放這裡)
        'y_max': 695,       # 景深終點
        'y_bg': 720,        # 背景傢俱位置
    },
    
    # 透明物體
    'glass_objects': {
        'count_range': (1, 3),  # 每場景數量範圍
    },
    
    # 背景傢俱
    'furniture': {
        'count_range': (2, 4),  # 每場景數量範圍
    },
    
    # 隨機旋轉範圍
    'randomization': {
        'glass_rotation_y': (-30, 30),    # 玻璃水平旋轉
        'glass_rotation_x': (-15, 15),    # 玻璃俯仰
        'furniture_rotation_z': (-15, 15), # 傢俱旋轉
    },
}
```

### 13.7 自訂傢俱

如需新增傢俱類型，修改 `CONFIG['furniture']['types']`：

```python
'furniture': {
    'types': [
        {
            'name': 'my_furniture',      # 名稱
            'size': (100, 50, 80),       # 寬 × 深 × 高 (mm)
            'color': (0.5, 0.4, 0.3),    # RGB 顏色
            'weight': 2,                  # 出現權重
        },
        # ... 其他傢俱
    ],
}
```

### 13.8 工作流程

```
1. 執行 blender_furniture_randomizer.py
   ↓
2. 自動生成 N 個隨機場景
   ↓
3. 匯出為 OBJ 檔案
   ↓
4. 使用 pids_renderer_blender.py 渲染
   ↓
5. 產生訓練數據 (I∥, I⊥, disparity)
```

### 13.9 注意事項

- 腳本會自動建立地面和背景牆（如果不存在）
- 每次執行會清除舊的傢俱和玻璃物體
- 透明物體自動放置在景深範圍內 (Y = 528-695mm)
- 背景傢俱自動放置在背景區域 (Y ≈ 720mm)

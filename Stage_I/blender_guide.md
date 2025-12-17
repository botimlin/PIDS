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

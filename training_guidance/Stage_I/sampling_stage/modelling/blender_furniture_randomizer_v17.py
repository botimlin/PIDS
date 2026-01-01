"""
PIDS Blender 傢俱隨機擺放腳本 (v17.2)
=====================================

更新內容:
v17.2:
- 新增多種物件類型支援 (clock, lamp, plant, tv, frame, vase, books)
- 動態檢測可用的 source_* 物件，只使用存在的類型
- 改進啟動訊息，顯示可用傢俱類型列表
- 確保缺少的物件不會導致錯誤，只會被跳過

v17.1:
- 新增隨機種子管理 (random.seed(scene_id))
- 確保每個場景有獨特且可重現的隨機序列
- 避免重複執行產生相同場景

v17.0:
- 新增完整封閉 chamber（前牆、後牆、左牆、右牆、地板、天花板）
- 移除自動生成傢俱功能，只使用已有素材

功能:
1. 生成隨機玻璃和傢俱場景
2. 生成完整封閉房間背景（6 面）
3. 匯出為 OBJ 檔案
4. 同時匯出場景描述 JSON

座標系統:
- Blender: X(右), Y(前/深度), Z(上)
- 相機在原點，朝向 +Y 方向
- 玻璃在 Y=528~695mm (工作距離)
- 傢俱在 Y=700~800mm (背景)
- 封閉 chamber 包圍整個場景

Chamber 結構 (俯視圖):
    
         後牆 (back)
    ┌─────────────────┐
    │                 │
 左 │    傢俱區域     │ 右
 牆 │                 │ 牆
    │    玻璃區域     │
    │                 │
    │     相機        │
    └─────────────────┘
         前牆 (front)
    
    相機 → +Y 方向

使用方法:
1. 在 Blender 中準備 source_* 傢俱物件
2. 執行此腳本
3. OBJ 檔案輸出到指定目錄
"""

import bpy
import math
import random
import os
import re
import json
from mathutils import Vector

# ============================================================
# 配置參數
# ============================================================

CONFIG = {
    # 生成設定
    'num_scenes': 500,
    'output_dir': 'C:\\Users\\tim\\Documents\\PIDS_3\\PIDS\\training_guidance\\Stage_I\\sampling_stage\\modelling\\scenes_output_V2',    
    
    # 場景範圍 (mm)
    'scene': {
        'width': 250,           # X 軸範圍 (-125 ~ +125)
        'glass_y_min': 528,     # 玻璃最近距離
        'glass_y_max': 695,     # 玻璃最遠距離
        'furniture_y': 750,     # 傢俱基準深度
        'furniture_y_jitter': 50,  # 傢俱深度隨機範圍
    },
    
    # 背景設定 (牆壁、地板、天花板) - 封閉 chamber
    'background': {
        'enabled': True,        # 是否生成背景
        
        # 後牆 (面向相機)
        'back_wall': {
            'enabled': True,
            'y_offset': 100,    # 牆壁在傢俱後方的距離 (mm)
            'width': 600,       # 牆壁寬度 (mm) - 增加以完全覆蓋 FOV
            'height': 300,      # 牆壁高度 (mm)
            'thickness': 5,     # 牆壁厚度 (mm)
            'color': (0.75, 0.70, 0.65),  # 米色
        },
        
        # 前牆 (相機後方)
        'front_wall': {
            'enabled': True,
            'y_position': 350,  # 前牆 Y 位置 (mm)，相機前方一點
            'thickness': 5,
            'color': (0.75, 0.70, 0.65),
        },
        
        # 左牆
        'left_wall': {
            'enabled': True,
            'thickness': 5,
            'color': (0.72, 0.68, 0.63),  # 稍微不同的米色
        },
        
        # 右牆
        'right_wall': {
            'enabled': True,
            'thickness': 5,
            'color': (0.72, 0.68, 0.63),
        },
        
        # 地板
        'ground': {
            'enabled': True,
            'thickness': 2,     # 地板厚度 (mm)
            'color': (0.45, 0.40, 0.35),  # 木地板色
        },
        
        # 天花板
        'ceiling': {
            'enabled': True,
            'thickness': 5,
            'color': (0.85, 0.83, 0.80),  # 淺色天花板
        },
    },
    
    # 玻璃設定 (寬, 高, 厚) mm
    'glass': {
        'count_range': (1, 3),
        'types': [
            {'name': 'door',      'size': (90, 150, 2)},
            {'name': 'window',    'size': (100, 80, 2)},
            {'name': 'partition', 'size': (50, 40, 2)},
            {'name': 'panel',     'size': (60, 60, 2)},
        ],
        'rotation_z': (-30, 30),  # 度
        'tilt_x': (-5, 5),        # 度
    },
    
    # 傢俱設定 (寬, 深, 高) mm
    # 注意：只有在 Blender 中存在 source_<name> 物件時才會使用
    'furniture': {
        'count_range': (3, 4),  # 每場景至少 3 個傢俱
        'types': [
            # 原有傢俱
            {'name': 'shelf',   'size': (80, 30, 180)},
            {'name': 'cabinet', 'size': (100, 40, 90)},
            {'name': 'table',   'size': (80, 50, 45)},
            {'name': 'chair',   'size': (45, 45, 85)},
            {'name': 'sofa',    'size': (120, 60, 70)},
            # 新增物件 (需在 Blender 中建立 source_* 物件)
            {'name': 'clock',   'size': (25, 5, 25)},    # 掛鐘/桌鐘
            {'name': 'lamp',    'size': (20, 20, 40)},   # 檯燈
            {'name': 'plant',   'size': (25, 25, 50)},   # 盆栽
            {'name': 'tv',      'size': (80, 10, 50)},   # 電視
            {'name': 'frame',   'size': (40, 3, 30)},    # 畫框
            {'name': 'vase',    'size': (15, 15, 30)},   # 花瓶
            {'name': 'books',   'size': (30, 25, 20)},   # 書本堆
        ],
        'rotation_z': (-15, 15),
        'scale_range': (0.85, 1.1),  # 保守縮放範圍
    },
    
    # 碰撞檢測
    'collision_margin': 30,  # mm (增加間距避免碰撞)
}


# ============================================================
# 工具函數
# ============================================================

def get_rotated_aabb(width, depth, angle_rad):
    """計算旋轉後的 AABB 尺寸"""
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    new_w = width * cos_a + depth * sin_a
    new_d = width * sin_a + depth * cos_a
    return new_w, new_d


def check_collision(pos1, size1, pos2, size2, margin):
    """檢查兩個物件是否碰撞（AABB）"""
    x1, y1 = pos1[0], pos1[1]
    w1, d1 = size1[0], size1[1]
    
    x2, y2 = pos2[0], pos2[1]
    w2, d2 = size2[0], size2[1]
    
    overlap_x = abs(x1 - x2) < (w1 + w2) / 2 + margin
    overlap_y = abs(y1 - y2) < (d1 + d2) / 2 + margin
    
    return overlap_x and overlap_y


def snap_to_ground(obj):
    """將物件貼地（最低點 Z=0）"""
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    min_z = min(c.z for c in corners)
    obj.location.z -= min_z


def ensure_dir(path):
    """確保目錄存在"""
    if not os.path.exists(path):
        os.makedirs(path)


def get_next_scene_index(output_dir):
    """取得下一個場景編號"""
    ensure_dir(output_dir)
    pattern = re.compile(r'scene_(\d+)')
    max_idx = 0
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match:
            idx = int(match.group(1))
            max_idx = max(max_idx, idx)
    return max_idx + 1


# ============================================================
# 材質建立
# ============================================================

def create_glass_material():
    """建立玻璃材質"""
    name = "Glass_Clear"
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        # Blender 4.0+
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 1.0
        elif "Transmission" in bsdf.inputs:
            bsdf.inputs["Transmission"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = 0.0
        bsdf.inputs["IOR"].default_value = 1.5
        mat.blend_method = 'BLEND'
    return mat


def create_diffuse_material(name, color):
    """建立漫反射材質"""
    mat_name = f"Diffuse_{name}"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
    return mat


# 傢俱顏色對照表
FURNITURE_COLORS = {
    # 原有傢俱
    'shelf':   (0.35, 0.25, 0.15),  # 深木色
    'cabinet': (0.50, 0.40, 0.30),  # 淺木色
    'table':   (0.45, 0.35, 0.25),  # 木色
    'chair':   (0.40, 0.30, 0.20),  # 木色
    'sofa':    (0.30, 0.30, 0.35),  # 灰色
    # 新增物件
    'clock':   (0.20, 0.20, 0.22),  # 深灰/黑色
    'lamp':    (0.85, 0.82, 0.75),  # 米白色
    'plant':   (0.25, 0.45, 0.20),  # 綠色
    'tv':      (0.10, 0.10, 0.12),  # 黑色
    'frame':   (0.55, 0.45, 0.35),  # 木框色
    'vase':    (0.70, 0.65, 0.60),  # 陶瓷色
    'books':   (0.45, 0.35, 0.30),  # 書本棕色
}


# ============================================================
# 背景物件生成 (牆壁、地板、天花板)
# ============================================================

def get_room_dimensions():
    """
    計算房間尺寸（根據配置自動計算）
    
    Returns: dict with 'width', 'depth', 'height', 'y_start', 'y_end'
    """
    scene_cfg = CONFIG['scene']
    bg_cfg = CONFIG['background']
    back_wall_cfg = bg_cfg['back_wall']
    front_wall_cfg = bg_cfg['front_wall']
    
    # 房間寬度 = 後牆寬度
    width = back_wall_cfg['width']
    
    # 房間高度 = 後牆高度
    height = back_wall_cfg['height']
    
    # Y 起始位置 = 前牆位置
    y_start = front_wall_cfg['y_position']
    
    # Y 結束位置 = 後牆位置
    y_end = scene_cfg['furniture_y'] + scene_cfg['furniture_y_jitter'] + back_wall_cfg['y_offset']
    
    # 房間深度
    depth = y_end - y_start
    
    return {
        'width': width,
        'height': height,
        'depth': depth,
        'y_start': y_start,
        'y_end': y_end,
    }


def create_back_wall():
    """
    生成後牆（面向相機）
    """
    if not CONFIG['background']['enabled'] or not CONFIG['background']['back_wall']['enabled']:
        return None
    
    wall_cfg = CONFIG['background']['back_wall']
    room = get_room_dimensions()
    
    width = wall_cfg['width']
    height = wall_cfg['height']
    thickness = wall_cfg['thickness']
    
    # 建立牆壁
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    obj.name = "background_wall_back"
    
    # 尺寸: 寬 x 厚 x 高
    obj.scale = (width, thickness, height)
    
    # 位置: 中央, 後方, 高度中心
    obj.location = (0, room['y_end'], height / 2)
    
    # 應用變換
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)
    
    # 材質
    mat = create_diffuse_material('wall_back', wall_cfg['color'])
    obj.data.materials.append(mat)
    
    print(f"  + {obj.name} (後牆, Y={room['y_end']:.0f}mm)")
    return obj


def create_front_wall():
    """
    生成前牆（相機後方）
    """
    if not CONFIG['background']['enabled'] or not CONFIG['background']['front_wall']['enabled']:
        return None
    
    wall_cfg = CONFIG['background']['front_wall']
    back_cfg = CONFIG['background']['back_wall']
    room = get_room_dimensions()
    
    width = room['width']
    height = room['height']
    thickness = wall_cfg['thickness']
    
    # 建立牆壁
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    obj.name = "background_wall_front"
    
    # 尺寸: 寬 x 厚 x 高
    obj.scale = (width, thickness, height)
    
    # 位置: 中央, 前方, 高度中心
    obj.location = (0, room['y_start'], height / 2)
    
    # 應用變換
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)
    
    # 材質
    mat = create_diffuse_material('wall_front', wall_cfg['color'])
    obj.data.materials.append(mat)
    
    print(f"  + {obj.name} (前牆, Y={room['y_start']:.0f}mm)")
    return obj


def create_left_wall():
    """
    生成左牆
    """
    if not CONFIG['background']['enabled'] or not CONFIG['background']['left_wall']['enabled']:
        return None
    
    wall_cfg = CONFIG['background']['left_wall']
    back_cfg = CONFIG['background']['back_wall']
    room = get_room_dimensions()
    
    thickness = wall_cfg['thickness']
    height = room['height']
    depth = room['depth'] + back_cfg['thickness']  # 延伸到後牆後方
    
    # 建立牆壁
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    obj.name = "background_wall_left"
    
    # 尺寸: 厚 x 深 x 高
    obj.scale = (thickness, depth, height)
    
    # 位置: 左邊緣, 深度中心, 高度中心
    x_pos = -room['width'] / 2
    y_center = room['y_start'] + depth / 2
    obj.location = (x_pos, y_center, height / 2)
    
    # 應用變換
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)
    
    # 材質
    mat = create_diffuse_material('wall_left', wall_cfg['color'])
    obj.data.materials.append(mat)
    
    print(f"  + {obj.name} (左牆, X={x_pos:.0f}mm)")
    return obj


def create_right_wall():
    """
    生成右牆
    """
    if not CONFIG['background']['enabled'] or not CONFIG['background']['right_wall']['enabled']:
        return None
    
    wall_cfg = CONFIG['background']['right_wall']
    back_cfg = CONFIG['background']['back_wall']
    room = get_room_dimensions()
    
    thickness = wall_cfg['thickness']
    height = room['height']
    depth = room['depth'] + back_cfg['thickness']
    
    # 建立牆壁
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    obj.name = "background_wall_right"
    
    # 尺寸: 厚 x 深 x 高
    obj.scale = (thickness, depth, height)
    
    # 位置: 右邊緣, 深度中心, 高度中心
    x_pos = room['width'] / 2
    y_center = room['y_start'] + depth / 2
    obj.location = (x_pos, y_center, height / 2)
    
    # 應用變換
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)
    
    # 材質
    mat = create_diffuse_material('wall_right', wall_cfg['color'])
    obj.data.materials.append(mat)
    
    print(f"  + {obj.name} (右牆, X={x_pos:.0f}mm)")
    return obj


def create_ground():
    """
    生成地板
    """
    if not CONFIG['background']['enabled'] or not CONFIG['background']['ground']['enabled']:
        return None
    
    ground_cfg = CONFIG['background']['ground']
    back_cfg = CONFIG['background']['back_wall']
    room = get_room_dimensions()
    
    width = room['width']
    depth = room['depth'] + back_cfg['thickness']
    thickness = ground_cfg['thickness']
    
    # 建立地板
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    obj.name = "background_ground"
    
    # 尺寸: 寬 x 深 x 厚
    obj.scale = (width, depth, thickness)
    
    # 位置: 中央, 深度中心, 地面以下
    y_center = room['y_start'] + depth / 2
    obj.location = (0, y_center, -thickness / 2)
    
    # 應用變換
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)
    
    # 材質
    mat = create_diffuse_material('ground', ground_cfg['color'])
    obj.data.materials.append(mat)
    
    print(f"  + {obj.name} (地板)")
    return obj


def create_ceiling():
    """
    生成天花板
    """
    if not CONFIG['background']['enabled'] or not CONFIG['background']['ceiling']['enabled']:
        return None
    
    ceiling_cfg = CONFIG['background']['ceiling']
    back_cfg = CONFIG['background']['back_wall']
    room = get_room_dimensions()
    
    width = room['width']
    depth = room['depth'] + back_cfg['thickness']
    thickness = ceiling_cfg['thickness']
    height = room['height']
    
    # 建立天花板
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    obj.name = "background_ceiling"
    
    # 尺寸: 寬 x 深 x 厚
    obj.scale = (width, depth, thickness)
    
    # 位置: 中央, 深度中心, 天花板高度
    y_center = room['y_start'] + depth / 2
    obj.location = (0, y_center, height + thickness / 2)
    
    # 應用變換
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)
    
    # 材質
    mat = create_diffuse_material('ceiling', ceiling_cfg['color'])
    obj.data.materials.append(mat)
    
    print(f"  + {obj.name} (天花板, Z={height:.0f}mm)")
    return obj


def create_all_backgrounds():
    """
    生成所有背景物件（牆壁、地板、天花板）- 封閉 chamber
    
    Returns: list of (object, type, material) tuples
    """
    objects_info = []
    
    # 後牆
    obj = create_back_wall()
    if obj:
        objects_info.append((obj, 'background', 'Diffuse_wall_back'))
    
    # 前牆
    obj = create_front_wall()
    if obj:
        objects_info.append((obj, 'background', 'Diffuse_wall_front'))
    
    # 左牆
    obj = create_left_wall()
    if obj:
        objects_info.append((obj, 'background', 'Diffuse_wall_left'))
    
    # 右牆
    obj = create_right_wall()
    if obj:
        objects_info.append((obj, 'background', 'Diffuse_wall_right'))
    
    # 地板
    obj = create_ground()
    if obj:
        objects_info.append((obj, 'background', 'Diffuse_ground'))
    
    # 天花板
    obj = create_ceiling()
    if obj:
        objects_info.append((obj, 'background', 'Diffuse_ceiling'))
    
    return objects_info


# ============================================================
# 物件生成
# ============================================================

def create_glass(glass_type, index, occupied):
    """
    生成玻璃物件
    
    Returns: (object, position, aabb_size) or (None, None, None)
    """
    width, height, thickness = glass_type['size']
    name = f"glass_{glass_type['name']}_{index:02d}"
    
    # 嘗試找到有效位置
    for _ in range(100):
        rot_z = math.radians(random.uniform(*CONFIG['glass']['rotation_z']))
        aabb_w, aabb_d = get_rotated_aabb(width, thickness, rot_z)
        
        # X 位置
        x_range = (CONFIG['scene']['width'] - aabb_w) / 2
        x_range = max(0, x_range)
        x = random.uniform(-x_range, x_range)
        
        # Y 位置（深度）
        y = random.uniform(CONFIG['scene']['glass_y_min'], CONFIG['scene']['glass_y_max'])
        
        # 碰撞檢測
        collides = False
        for occ_pos, occ_size in occupied:
            if check_collision((x, y), (aabb_w, aabb_d), occ_pos, occ_size, CONFIG['collision_margin']):
                collides = True
                break
        
        if not collides:
            # 建立玻璃
            bpy.ops.mesh.primitive_cube_add(size=1)
            obj = bpy.context.active_object
            obj.name = name
            obj.scale = (width, thickness, height)
            obj.location = (x, y, height / 2)
            
            # 旋轉
            tilt_x = math.radians(random.uniform(*CONFIG['glass']['tilt_x']))
            obj.rotation_euler = (tilt_x, 0, rot_z)
            
            # 應用變換
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(scale=True)
            
            # 材質
            mat = create_glass_material()
            obj.data.materials.append(mat)
            
            return obj, (x, y), (aabb_w, aabb_d)
    
    return None, None, None


def create_furniture(furniture_type, index, occupied):
    """
    生成傢俱物件（從 source 複製）
    
    Returns: (object, position, aabb_size) or (None, None, None)
    """
    name = furniture_type['name']
    source_name = f"source_{name}"
    source_obj = bpy.data.objects.get(source_name)
    
    if source_obj is None:
        print(f"  [!] 找不到 {source_name}")
        return None, None, None
    
    base_w, base_d, base_h = furniture_type['size']
    
    # 嘗試不同縮放找到有效位置
    scale_min, scale_max = CONFIG['furniture']['scale_range']
    
    for _ in range(100):
        scale = random.uniform(scale_min, scale_max)
        rot_z = math.radians(random.uniform(*CONFIG['furniture']['rotation_z']))
        
        scaled_w = base_w * scale
        scaled_d = base_d * scale
        aabb_w, aabb_d = get_rotated_aabb(scaled_w, scaled_d, rot_z)
        
        # X 位置
        x_range = (CONFIG['scene']['width'] - aabb_w) / 2
        x_range = max(0, x_range)
        x = random.uniform(-x_range, x_range)
        
        # Y 位置
        y_base = CONFIG['scene']['furniture_y']
        y_jitter = CONFIG['scene']['furniture_y_jitter']
        y = random.uniform(y_base - y_jitter, y_base + y_jitter)
        
        # 碰撞檢測
        collides = False
        for occ_pos, occ_size in occupied:
            if check_collision((x, y), (aabb_w, aabb_d), occ_pos, occ_size, CONFIG['collision_margin']):
                collides = True
                break
        
        if not collides:
            # 複製物件
            obj = source_obj.copy()
            if source_obj.data:
                obj.data = source_obj.data.copy()
            bpy.context.collection.objects.link(obj)
            
            obj.name = f"diffuse_{name}_{index:02d}"
            obj.hide_render = False
            obj.hide_viewport = False
            
            # 設定變換
            obj.location = (x, y, 0)
            obj.rotation_euler = (0, 0, rot_z)
            
            # 縮放到目標尺寸
            current_h = source_obj.dimensions.z
            source_scale = source_obj.scale[:]
            print(f"    [DEBUG] {source_name}: dim.z={current_h:.4f}, source_scale={source_scale}, base_h={base_h}")

            if current_h > 0.001:
                target_h = base_h * scale
                s = target_h / current_h
                print(f"    [DEBUG]   -> target_h={target_h:.2f}, final_scale={s:.4f}")
                obj.scale = (s, s, s)
            
            # 應用變換
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(scale=True)
            
            # 貼地
            snap_to_ground(obj)
            
            # 材質
            color = FURNITURE_COLORS.get(name, (0.5, 0.5, 0.5))
            mat = create_diffuse_material(name, color)
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)
            
            return obj, (x, y), (aabb_w, aabb_d)
    
    return None, None, None


# ============================================================
# 匯出函數
# ============================================================

def export_scene(scene_id, output_dir, objects_info):
    """
    匯出場景
    
    為每個物件匯出獨立的 OBJ 檔案，並建立場景描述 JSON
    """
    scene_dir = os.path.join(output_dir, f"scene_{scene_id:04d}")
    ensure_dir(scene_dir)
    
    scene_info = {
        'scene_id': scene_id,
        'objects': [],
    }
    
    for obj, obj_type, material_type in objects_info:
        obj_filename = f"{obj.name}.obj"
        obj_filepath = os.path.join(scene_dir, obj_filename)
        
        # 選擇物件
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        # 匯出
        try:
            bpy.ops.wm.obj_export(
                filepath=obj_filepath,
                export_selected_objects=True,
                forward_axis='NEGATIVE_Y',
                up_axis='Z',
                apply_modifiers=True,
                export_materials=True,
            )
        except AttributeError:
            bpy.ops.export_scene.obj(
                filepath=obj_filepath,
                use_selection=True,
                axis_forward='-Y',
                axis_up='Z',
                use_materials=True,
            )
        
        # 記錄物件資訊
        scene_info['objects'].append({
            'name': obj.name,
            'file': obj_filename,
            'type': obj_type,  # 'glass', 'furniture', 'background'
            'material': material_type,
            'location': list(obj.location),
            'rotation': list(obj.rotation_euler),
            'dimensions': list(obj.dimensions),
        })
    
    # 儲存場景描述
    json_path = os.path.join(scene_dir, 'scene.json')
    with open(json_path, 'w') as f:
        json.dump(scene_info, f, indent=2)
    
    print(f"  -> 匯出完成: {scene_dir}")
    return scene_dir


def export_scene_combined(scene_id, output_dir):
    """
    匯出整個場景為單一 OBJ 檔案
    """
    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, f"scene_{scene_id:04d}.obj")
    
    # 選擇所有生成的物件（包含背景）
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.name.startswith(('glass_', 'diffuse_', 'background_')) and obj.type == 'MESH':
            obj.select_set(True)
    
    if not bpy.context.selected_objects:
        print(f"  [!] 沒有物件可匯出")
        return None
    
    # 匯出
    try:
        bpy.ops.wm.obj_export(
            filepath=filepath,
            export_selected_objects=True,
            forward_axis='NEGATIVE_Y',
            up_axis='Z',
            apply_modifiers=True,
            export_materials=True,
        )
    except AttributeError:
        bpy.ops.export_scene.obj(
            filepath=filepath,
            use_selection=True,
            axis_forward='-Y',
            axis_up='Z',
            use_materials=True,
        )
    
    print(f"  -> 匯出: {filepath}")
    return filepath


# ============================================================
# 清理函數
# ============================================================

def clear_generated():
    """清除生成的物件（保留 source_ 物件）"""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        # 刪除 glass_, diffuse_, background_ 開頭的物件
        # 確保不會刪除 source_ 物件
        if obj.name.startswith(('glass_', 'diffuse_', 'background_')) and obj.type == 'MESH':
            if not obj.name.startswith('source_'):  # 雙重保險
                obj.select_set(True)
    bpy.ops.object.delete()


def setup_blender():
    """設定 Blender 單位"""
    unit = bpy.context.scene.unit_settings
    unit.system = 'METRIC'
    unit.length_unit = 'MILLIMETERS'
    unit.scale_length = 0.001


def check_source_objects(verbose=True):
    """
    檢查來源物件是否存在
    只檢查並報告，不自動建立

    Returns: (available_types, missing_names)
        available_types: 可用的傢俱類型列表 (從 CONFIG 中篩選)
        missing_names: 缺少的來源物件名稱列表
    """
    if verbose:
        print("檢查來源物件...")

    # 從 CONFIG 動態取得所有傢俱類型
    all_types = CONFIG['furniture']['types']

    missing = []
    found = []
    available_types = []

    for f_type in all_types:
        source_name = f"source_{f_type['name']}"
        if bpy.data.objects.get(source_name) is None:
            missing.append(source_name)
        else:
            found.append(source_name)
            available_types.append(f_type)

    if verbose:
        for name in found:
            print(f"  ✓ {name}")
        for name in missing:
            print(f"  ✗ {name} (缺少，將跳過)")

    if missing:
        print(f"  [資訊] 缺少 {len(missing)} 個來源物件，將只使用 {len(found)} 個可用類型")

    if not available_types:
        print("  [警告] 沒有任何可用的傢俱來源物件！")

    return available_types, missing


# ============================================================
# 主程式
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("PIDS 場景生成器 v17.2 (支援多種物件類型)")
    print("=" * 60)

    setup_blender()

    # 檢查來源物件，取得可用的傢俱類型
    available_furniture_types, missing = check_source_objects()

    if not available_furniture_types:
        print("\n[錯誤] 沒有可用的傢俱來源物件，無法生成場景！")
        print("請在 Blender 中建立 source_* 物件（如 source_shelf, source_table 等）")
        return

    output_dir = CONFIG['output_dir']
    start_idx = get_next_scene_index(output_dir)
    num_scenes = CONFIG['num_scenes']

    print(f"\n輸出目錄: {output_dir}")
    print(f"場景範圍: {start_idx} ~ {start_idx + num_scenes - 1}")
    print(f"可用傢俱類型: {len(available_furniture_types)} 種")
    print(f"  {[t['name'] for t in available_furniture_types]}")
    print(f"背景設定 (封閉 chamber):")
    print(f"  - 後牆: {'啟用' if CONFIG['background']['back_wall']['enabled'] else '停用'}")
    print(f"  - 前牆: {'啟用' if CONFIG['background']['front_wall']['enabled'] else '停用'}")
    print(f"  - 左牆: {'啟用' if CONFIG['background']['left_wall']['enabled'] else '停用'}")
    print(f"  - 右牆: {'啟用' if CONFIG['background']['right_wall']['enabled'] else '停用'}")
    print(f"  - 地板: {'啟用' if CONFIG['background']['ground']['enabled'] else '停用'}")
    print(f"  - 天花板: {'啟用' if CONFIG['background']['ceiling']['enabled'] else '停用'}")
    print("=" * 60 + "\n")

    success = 0

    for i in range(num_scenes):
        scene_id = start_idx + i

        # 用場景 ID 設定隨機種子，確保：
        # 1. 每個場景有獨特的隨機序列
        # 2. 相同 scene_id 可以重現相同結果
        # 3. 不同 scene_id 永遠不會產生完全相同的場景
        random.seed(scene_id)

        print(f"[場景 {scene_id}] (seed={scene_id})")

        clear_generated()

        occupied = []
        objects_info = []

        # === 生成背景 (牆壁、地板、天花板) ===
        bg_objects = create_all_backgrounds()
        objects_info.extend(bg_objects)

        # === 生成玻璃 ===
        glass_count = random.randint(*CONFIG['glass']['count_range'])
        for g_idx in range(glass_count):
            g_type = random.choice(CONFIG['glass']['types'])
            obj, pos, aabb = create_glass(g_type, g_idx + 1, occupied)
            if obj:
                occupied.append((pos, aabb))
                objects_info.append((obj, 'glass', 'Glass_Clear'))
                print(f"  + {obj.name}")

        # === 生成傢俱 (只使用可用的類型，確保至少放置 min_count 個) ===
        min_count, max_count = CONFIG['furniture']['count_range']
        target_count = random.randint(min_count, max_count)

        # 確保有足夠的可用類型
        if len(available_furniture_types) < min_count:
            print(f"  [警告] 可用傢俱類型 ({len(available_furniture_types)}) 少於最低需求 ({min_count})")

        # 隨機打亂可用類型順序
        shuffled_types = available_furniture_types.copy()
        random.shuffle(shuffled_types)

        # 大物件優先排序
        shuffled_types.sort(key=lambda x: x['size'][0] * x['size'][1], reverse=True)

        placed_count = 0
        f_idx = 0
        type_idx = 0

        # 持續嘗試直到放置足夠數量或用盡所有類型
        while placed_count < target_count and type_idx < len(shuffled_types):
            f_type = shuffled_types[type_idx]
            obj, pos, aabb = create_furniture(f_type, f_idx + 1, occupied)
            if obj:
                occupied.append((pos, aabb))
                mat_name = f"Diffuse_{f_type['name']}"
                objects_info.append((obj, 'furniture', mat_name))
                print(f"  + {obj.name}")
                placed_count += 1
                f_idx += 1
            type_idx += 1

        # 檢查是否達到最低需求
        if placed_count < min_count:
            print(f"  [警告] 只放置了 {placed_count} 個傢俱 (需求: {min_count})")

        # === 匯出（合併為單一 OBJ）===
        if objects_info:
            export_scene_combined(scene_id, output_dir)
            success += 1
        else:
            print(f"  [!] 生成失敗")

    clear_generated()

    print("\n" + "=" * 60)
    print(f"完成！成功: {success}/{num_scenes}")
    print("=" * 60)


if __name__ == "__main__":
    main()

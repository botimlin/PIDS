"""
PIDS Blender 傢俱隨機擺放腳本 (v17.0)
=====================================

更新內容:
- 新增背景牆壁生成
- 新增地板生成
- 牆壁和地板使用漫反射材質

功能:
1. 生成隨機玻璃和傢俱場景
2. 生成背景牆壁和地板
3. 匯出為 OBJ 檔案
4. 同時匯出場景描述 JSON

座標系統:
- Blender: X(右), Y(前/深度), Z(上)
- 相機在原點，朝向 +Y 方向
- 玻璃在 Y=528~695mm (工作距離)
- 傢俱在 Y=700~800mm (背景)
- 牆壁在傢俱後方
- 地板在 Z=0

使用方法:
1. 在 Blender 中準備 source_* 物件
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
    'num_scenes': 1000,
    'output_dir': 'C:\\Users\\tim\\Documents\\PIDS_2\\PIDS\\training_guidance\\Stage_I\\sampling_stage\\modelling\\scenes_output',    
    
    # 場景範圍 (mm)
    'scene': {
        'width': 250,           # X 軸範圍 (-125 ~ +125)
        'glass_y_min': 528,     # 玻璃最近距離
        'glass_y_max': 695,     # 玻璃最遠距離
        'furniture_y': 750,     # 傢俱基準深度
        'furniture_y_jitter': 50,  # 傢俱深度隨機範圍
    },
    
    # 背景設定 (牆壁和地板)
    'background': {
        'enabled': True,        # 是否生成背景
        'wall': {
            'enabled': True,
            'y_offset': 100,    # 牆壁在傢俱後方的距離 (mm)
            'width': 400,       # 牆壁寬度 (mm)
            'height': 300,      # 牆壁高度 (mm)
            'thickness': 5,     # 牆壁厚度 (mm)
            'color': (0.75, 0.70, 0.65),  # 米色
        },
        'ground': {
            'enabled': True,
            'width': 400,       # 地板寬度 (mm)
            'depth': 500,       # 地板深度 (mm) - 從相機前方延伸到牆壁
            'thickness': 2,     # 地板厚度 (mm)
            'y_start': 400,     # 地板起始 Y 位置 (mm)
            'color': (0.45, 0.40, 0.35),  # 木地板色
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
    'furniture': {
        'count_range': (2, 4),
        'types': [
            {'name': 'shelf',   'size': (80, 30, 180)},
            {'name': 'cabinet', 'size': (100, 40, 90)},
            {'name': 'table',   'size': (80, 50, 45)},
            {'name': 'chair',   'size': (45, 45, 85)},
            {'name': 'sofa',    'size': (120, 60, 70)},
        ],
        'rotation_z': (-15, 15),
        'scale_range': (0.6, 1.05),
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
    'shelf':   (0.35, 0.25, 0.15),
    'cabinet': (0.50, 0.40, 0.30),
    'table':   (0.45, 0.35, 0.25),
    'chair':   (0.40, 0.30, 0.20),
    'sofa':    (0.30, 0.30, 0.35),
}


# ============================================================
# 背景物件生成 (牆壁和地板)
# ============================================================

def create_background_wall():
    """
    生成背景牆壁
    
    牆壁位於傢俱後方，面向相機
    """
    if not CONFIG['background']['enabled'] or not CONFIG['background']['wall']['enabled']:
        return None
    
    wall_cfg = CONFIG['background']['wall']
    scene_cfg = CONFIG['scene']
    
    width = wall_cfg['width']
    height = wall_cfg['height']
    thickness = wall_cfg['thickness']
    
    # 計算牆壁 Y 位置（傢俱最遠處 + 偏移）
    wall_y = scene_cfg['furniture_y'] + scene_cfg['furniture_y_jitter'] + wall_cfg['y_offset']
    
    # 建立牆壁 (薄板)
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    obj.name = "background_wall"
    
    # 設定尺寸: 寬 x 厚 x 高
    obj.scale = (width, thickness, height)
    
    # 位置: 中央, 牆壁深度, 高度中心
    obj.location = (0, wall_y, height / 2)
    
    # 應用變換
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)
    
    # 材質
    mat = create_diffuse_material('wall', wall_cfg['color'])
    obj.data.materials.append(mat)
    
    print(f"  + {obj.name} (Y={wall_y:.0f}mm)")
    return obj


def create_background_ground():
    """
    生成地板
    
    地板在 Z=0，從相機前方延伸到牆壁
    """
    if not CONFIG['background']['enabled'] or not CONFIG['background']['ground']['enabled']:
        return None
    
    ground_cfg = CONFIG['background']['ground']
    scene_cfg = CONFIG['scene']
    wall_cfg = CONFIG['background']['wall']
    
    width = ground_cfg['width']
    depth = ground_cfg['depth']
    thickness = ground_cfg['thickness']
    y_start = ground_cfg['y_start']
    
    # 如果牆壁啟用，地板深度延伸到牆壁位置
    if CONFIG['background']['wall']['enabled']:
        wall_y = scene_cfg['furniture_y'] + scene_cfg['furniture_y_jitter'] + wall_cfg['y_offset']
        depth = wall_y - y_start + wall_cfg['thickness']
    
    # 建立地板 (薄板)
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    obj.name = "background_ground"
    
    # 設定尺寸: 寬 x 深 x 厚
    obj.scale = (width, depth, thickness)
    
    # 位置: 中央, 深度中心, 地面以下（厚度的一半）
    y_center = y_start + depth / 2
    obj.location = (0, y_center, -thickness / 2)
    
    # 應用變換
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)
    
    # 材質
    mat = create_diffuse_material('ground', ground_cfg['color'])
    obj.data.materials.append(mat)
    
    print(f"  + {obj.name} (Y={y_start:.0f}~{y_start + depth:.0f}mm)")
    return obj


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
            if current_h > 0.001:
                target_h = base_h * scale
                s = target_h / current_h
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
    """
    if verbose:
        print("檢查來源物件...")
    
    # 傢俱來源物件名稱
    source_names = [
        'source_shelf',
        'source_cabinet',
        'source_table',
        'source_chair',
        'source_sofa',
    ]
    
    missing = []
    found = []
    
    for source_name in source_names:
        if bpy.data.objects.get(source_name) is None:
            missing.append(source_name)
        else:
            found.append(source_name)
    
    if verbose:
        for name in found:
            print(f"  ✓ {name}")
        for name in missing:
            print(f"  ✗ {name} (缺少)")
    
    if missing:
        print(f"  [警告] 缺少 {len(missing)} 個來源物件，這些傢俱類型將無法生成")
    
    return missing


# ============================================================
# 主程式
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("PIDS 場景生成器 v17.0 (含背景牆壁/地板)")
    print("=" * 60)
    
    setup_blender()
    check_source_objects()  # 檢查來源物件是否存在
    
    output_dir = CONFIG['output_dir']
    start_idx = get_next_scene_index(output_dir)
    num_scenes = CONFIG['num_scenes']
    
    print(f"輸出目錄: {output_dir}")
    print(f"場景範圍: {start_idx} ~ {start_idx + num_scenes - 1}")
    print(f"背景牆壁: {'啟用' if CONFIG['background']['wall']['enabled'] else '停用'}")
    print(f"背景地板: {'啟用' if CONFIG['background']['ground']['enabled'] else '停用'}")
    print("=" * 60 + "\n")
    
    success = 0
    
    for i in range(num_scenes):
        scene_id = start_idx + i
        print(f"[場景 {scene_id}]")
        
        clear_generated()
        
        occupied = []
        objects_info = []
        
        # === 生成背景 (牆壁和地板) ===
        wall_obj = create_background_wall()
        if wall_obj:
            objects_info.append((wall_obj, 'background', 'Diffuse_wall'))
        
        ground_obj = create_background_ground()
        if ground_obj:
            objects_info.append((ground_obj, 'background', 'Diffuse_ground'))
        
        # === 生成玻璃 ===
        glass_count = random.randint(*CONFIG['glass']['count_range'])
        for g_idx in range(glass_count):
            g_type = random.choice(CONFIG['glass']['types'])
            obj, pos, aabb = create_glass(g_type, g_idx + 1, occupied)
            if obj:
                occupied.append((pos, aabb))
                objects_info.append((obj, 'glass', 'Glass_Clear'))
                print(f"  + {obj.name}")
        
        # === 生成傢俱 ===
        furniture_count = random.randint(*CONFIG['furniture']['count_range'])
        furniture_types = random.sample(
            CONFIG['furniture']['types'],
            min(furniture_count, len(CONFIG['furniture']['types']))
        )
        # 大物件優先
        furniture_types.sort(key=lambda x: x['size'][0] * x['size'][1], reverse=True)
        
        for f_idx, f_type in enumerate(furniture_types):
            obj, pos, aabb = create_furniture(f_type, f_idx + 1, occupied)
            if obj:
                occupied.append((pos, aabb))
                mat_name = f"Diffuse_{f_type['name']}"
                objects_info.append((obj, 'furniture', mat_name))
                print(f"  + {obj.name}")
        
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

"""
PIDS Blender 傢俱隨機擺放腳本 (v15.0 - 自動落地版)
===================================================
功能升級:
1. [自動落地] 新增 snap_to_ground 函式，計算物件最低點並強制貼地 (Z=0)。
2. [修正] 移除原本依賴中心點高度的放置邏輯，改為物理邊界偵測。
3. [繼承] 保留防穿模 (AABB)、量產輸出、座標保留等所有功能。
"""

import bpy
import math
import random
import os
import re
from mathutils import Vector # 需要引入 Vector 進行頂點運算

# ============================================================
# 1. 配置參數
# ============================================================

CONFIG = {
    'num_scenes_per_run': 10, 
    'output_dir': 'C:/Temp/PIDS_Dataset_Output',
    
    'scene': {
        'width': 250,      
        'y_min': 528,      
        'y_max': 695,      
        'y_bg_base': 730,  
        'y_bg_jitter': (-10, 10), 
    },
    
    'glass_objects': {
        'count_range': (1, 3),
        'types': [
            {'name': 'glass_door',      'size': (90, 150, 2),  'weight': 2}, 
            {'name': 'glass_window',    'size': (100, 80, 2),  'weight': 2},
            {'name': 'glass_partition', 'size': (50, 40, 2),   'weight': 3},
            {'name': 'glass_table',     'size': (80, 80, 2),   'weight': 2},
        ],
    },
    
    'furniture': {
        'count_range': (2, 4), 
        'types': [
            {'name': 'shelf',   'size': (80, 30, 180), 'weight': 1},
            {'name': 'cabinet', 'size': (100, 40, 90), 'weight': 1},
            {'name': 'table',   'size': (80, 50, 45),  'weight': 1},
            {'name': 'chair',   'size': (45, 45, 85),  'weight': 1},
            {'name': 'sofa',    'size': (120, 60, 70), 'weight': 1},
        ],
    },
    
    'randomization': {
        'initial_max_scale': 1.05,     
        'min_scale_limit': 0.6,        
        'glass_rotation_y': (-30, 30), 
        'glass_rotation_x': (-5, 5),   
        'furniture_rotation_z': (-15, 15), 
    },
}

# ============================================================
# 2. 核心數學與物理
# ============================================================

def get_rotated_size(w, d, angle_rad):
    """計算旋轉後的 2D 佔用尺寸 (防穿模用)"""
    abs_cos = abs(math.cos(angle_rad))
    abs_sin = abs(math.sin(angle_rad))
    new_w = w * abs_cos + d * abs_sin
    new_d = w * abs_sin + d * abs_cos
    return new_w, new_d

def snap_to_ground(obj):
    """
    [關鍵新功能] 強制將物件貼地
    計算物件世界座標中 bounding box 的最低 Z 值，然後向下移動該距離。
    """
    # 強制更新視圖層，確保矩陣運算正確
    bpy.context.view_layer.update()
    
    # 取得物件的 8 個角落的世界座標
    world_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # 找出最低的 Z 值
    min_z = min([c.z for c in world_corners])
    
    # 將物件向下(或向上)移動，使最低點剛好在 Z=0
    obj.location.z -= min_z

# ============================================================
# 3. 檔案管理
# ============================================================

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_next_start_index(directory):
    ensure_directory(directory)
    max_idx = 0
    pattern = re.compile(r"scene_(\d+)\.obj")
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            idx = int(match.group(1))
            if idx > max_idx: max_idx = idx
    return max_idx + 1

# ============================================================
# 4. Blender 操作工具
# ============================================================

def validate_and_setup_sources():
    print("\n[系統] 正在驗證 Source 物件...")
    unit = bpy.context.scene.unit_settings
    unit.system = 'METRIC'
    unit.length_unit = 'MILLIMETERS'
    unit.scale_length = 0.001
    
    valid_furniture_types = []
    for f_type in CONFIG['furniture']['types']:
        target_name = f_type['name']
        source_name = f"source_{target_name}"
        obj = bpy.data.objects.get(source_name)
        if obj:
            obj.hide_render = True
            obj.hide_viewport = False
            
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            
            # 應用旋轉與縮放，但不應用位置 (以免原點跑掉，雖然現在有 snap 機制比較沒差了)
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
            
            current_h = obj.dimensions.z
            target_h = f_type['size'][2]
            if current_h > 0.001:
                scale_factor = target_h / current_h
                obj.scale = (scale_factor, scale_factor, scale_factor)
                bpy.ops.object.transform_apply(scale=True)
                
                # 重新設定原點至幾何中心 (方便後續旋轉)
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                
                f_type['real_size'] = (obj.dimensions.x, obj.dimensions.y, obj.dimensions.z)
                valid_furniture_types.append(f_type)
                print(f"  [O] {source_name} 準備就緒")
        else:
            print(f"  [X] 警告: 找不到 {source_name}")
    return valid_furniture_types

def clear_generated_objects():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if not obj.name.startswith(('ground_', 'bg_', 'source_')) and obj.type == 'MESH':
            obj.select_set(True)
    if bpy.context.selected_objects:
        bpy.ops.object.delete()

def create_material(name, is_glass=False):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        if is_glass:
            if "Transmission" in bsdf.inputs: bsdf.inputs["Transmission"].default_value = 1.0
            elif "Transmission Weight" in bsdf.inputs: bsdf.inputs["Transmission Weight"].default_value = 1.0
            bsdf.inputs["Roughness"].default_value = 0.0
            bsdf.inputs["IOR"].default_value = 1.5
            mat.blend_method = 'BLEND'
            mat.shadow_method = 'NONE'
    return mat

def check_overlap(new_pos, new_size_rotated, existing_objects, margin=5):
    nx, ny, _ = new_pos
    nw_eff, nd_eff, _ = new_size_rotated
    for obj_pos, obj_size_rotated in existing_objects:
        ox, oy, _ = obj_pos
        ow_eff, od_eff, _ = obj_size_rotated
        
        x_overlap = abs(nx - ox) < (nw_eff + ow_eff)/2 + margin
        y_overlap = abs(ny - oy) < (nd_eff + od_eff)/2 + margin
        
        if x_overlap and y_overlap: return True
    return False

# ============================================================
# 5. 生成邏輯
# ============================================================

def create_glass_object_safe(glass_type, index, existing_objects):
    name = f"{glass_type['name']}_{index:02d}"
    w, h, d = glass_type['size']
    
    valid_pos = None
    final_rot_z = 0
    rotated_w, rotated_d = 0, 0

    for _ in range(100):
        rot_z = math.radians(random.uniform(*CONFIG['randomization']['glass_rotation_y']))
        rotated_w, rotated_d = get_rotated_size(w, d, rot_z)
        
        x_range = (CONFIG['scene']['width'] - rotated_w) / 2
        if x_range < 0: x_range = 0
        rand_x = random.uniform(-x_range, x_range)
        y_pos = random.uniform(CONFIG['scene']['y_min'], CONFIG['scene']['y_max'])
        
        # 碰撞檢查 (Z軸忽略，只要XY不重疊)
        if not check_overlap((rand_x, y_pos, 0), (rotated_w, rotated_d, h), existing_objects, margin=5):
            valid_pos = (rand_x, y_pos, 0) # 暫時設 Z=0
            final_rot_z = rot_z
            break
            
    if valid_pos is None: return None, None, None

    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    obj.name = name
    mat = create_material("Glass_Clear", is_glass=True)
    obj.data.materials.append(mat)
    
    obj.scale = (w, d, h)
    obj.location = valid_pos
    
    # 應用旋轉
    rot_x = math.radians(random.uniform(*CONFIG['randomization']['glass_rotation_x']))
    obj.rotation_euler = (rot_x, 0, final_rot_z)
    
    # [修正] 不要在這裡應用 Scale，因為 snap_to_ground 需要正確的邊界盒
    # 先做 snap
    snap_to_ground(obj)
    
    # 最後再 Apply Scale (習慣動作，確保匯出時 scale 為 1)
    bpy.ops.object.transform_apply(scale=True)
    
    return obj, valid_pos, (rotated_w, rotated_d, h)

def create_furniture_adaptive_safe(furniture_type, index, existing_objects, current_scene_y_center):
    name = furniture_type['name']
    base_w, base_d, base_h = furniture_type['real_size']
    source_obj = bpy.data.objects.get(f"source_{name}")
    if not source_obj: return None, None, None

    valid_pos = None
    final_rot_z = 0
    final_scale = 0
    final_rotated_w, final_rotated_d = 0, 0
    
    current_scale = CONFIG['randomization']['initial_max_scale']
    min_scale = CONFIG['randomization']['min_scale_limit']
    step = 0.05
    
    while current_scale >= min_scale:
        current_w_raw = base_w * current_scale
        current_d_raw = base_d * current_scale
        current_h = base_h * current_scale
        
        found = False
        for _ in range(50):
            rot_z = math.radians(random.uniform(*CONFIG['randomization']['furniture_rotation_z']))
            rotated_w, rotated_d = get_rotated_size(current_w_raw, current_d_raw, rot_z)
            
            x_range = (CONFIG['scene']['width'] - rotated_w) / 2
            if x_range < 0: x_range = 0 
            rand_x = random.uniform(-x_range, x_range)
            rand_y = random.uniform(current_scene_y_center - 100, current_scene_y_center + 50) 
            
            if not check_overlap((rand_x, rand_y, 0), (rotated_w, rotated_d, current_h), existing_objects, margin=5):
                valid_pos = (rand_x, rand_y, 0)
                final_rot_z = rot_z
                final_scale = current_scale
                final_rotated_w, final_rotated_d = rotated_w, rotated_d
                found = True
                break
        
        if found: break
        current_scale -= step
    
    if valid_pos is None: return None, None, None

    new_obj = source_obj.copy()
    if source_obj.data: new_obj.data = source_obj.data.copy()
    bpy.context.collection.objects.link(new_obj)
    
    new_obj.name = f"diffuse_{name}_{index:02d}"
    new_obj.scale = (final_scale, final_scale, final_scale)
    new_obj.location = (valid_pos[0], valid_pos[1], 0) # 先放在 Z=0
    new_obj.rotation_euler = (0, 0, final_rot_z)
    
    # [修正] 使用自動落地函式
    snap_to_ground(new_obj)
    
    return new_obj, valid_pos, (final_rotated_w, final_rotated_d, current_h)

# ============================================================
# 6. 主控流程
# ============================================================

def export_current_selection(scene_id, base_dir):
    filename = f"scene_{scene_id:04d}.obj"
    filepath = os.path.join(base_dir, filename)
    print(f"  -> 正在匯出: {filename} ...")
    try:
        bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True, forward_axis='NEGATIVE_Y', up_axis='Z', apply_modifiers=True)
    except AttributeError:
        bpy.ops.export_scene.obj(filepath=filepath, use_selection=True, axis_forward='-Y', axis_up='Z')

def main():
    output_dir = CONFIG['output_dir']
    valid_types = validate_and_setup_sources()
    start_idx = get_next_start_index(output_dir)
    end_idx = start_idx + CONFIG['num_scenes_per_run']
    
    print(f"\n=== PIDS 批次生成開始 (v15.0 自動落地版) ===")
    
    for i in range(start_idx, end_idx):
        clear_generated_objects()
        occupied = []
        jitter_min, jitter_max = CONFIG['scene']['y_bg_jitter']
        scene_y_center = CONFIG['scene']['y_bg_base'] + random.uniform(jitter_min, jitter_max)
        
        # 生成物件
        for g_idx in range(random.randint(*CONFIG['glass_objects']['count_range'])):
            g_type = random.choice(CONFIG['glass_objects']['types'])
            obj, pos, size = create_glass_object_safe(g_type, g_idx+1, occupied)
            if obj: occupied.append((pos, size))
            
        if valid_types:
            target_num = random.randint(*CONFIG['furniture']['count_range'])
            actual_num = min(target_num, len(valid_types))
            selected_types = random.sample(valid_types, actual_num)
            selected_types.sort(key=lambda x: x['real_size'][0] * x['real_size'][1], reverse=True)
            for f_idx, f_type in enumerate(selected_types):
                obj, pos, size = create_furniture_adaptive_safe(f_type, f_idx+1, occupied, scene_y_center)
                if obj: occupied.append((pos, size))
        
        # 匯出
        bpy.ops.object.select_all(action='DESELECT')
        has_selection = False
        for obj in bpy.data.objects:
            if obj.name.startswith(('diffuse_', 'glass_')) and obj.type == 'MESH':
                obj.select_set(True)
                has_selection = True
        
        if has_selection: export_current_selection(i, output_dir)
        else: print(f"  [警示] 場景 {i} 為空。")

    clear_generated_objects()
    print(f"\n=== 全部完成 ===")

if __name__ == "__main__":
    main()
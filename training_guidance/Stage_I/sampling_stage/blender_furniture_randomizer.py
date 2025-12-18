"""
PIDS Blender 傢俱隨機擺放腳本 (v14.0 - 批次量產/獨立輸出版)
===================================================
功能:
1. 讀取 Source 物件。
2. 根據設定數量 (num_scenes_per_run) 進行迴圈。
3. 生成場景 -> 匯出 OBJ (保留座標) -> 刪除生成物 -> 下一個。
4. 自動遞增檔名 (scene_0001, scene_0002...) 且不覆蓋舊檔。
"""

import bpy
import math
import random
import os
import re

# ============================================================
# 1. 配置參數
# ============================================================

CONFIG = {
    # [設定] 這次按下執行要產生幾個場景？
    'num_scenes_per_run': 10, 
    
    # [設定] 檔案要存在哪裡？(建議用絕對路徑，或保持預設)
    # 如果路徑不存在，腳本會自動建立
    'output_dir': 'C:\\Users\\tim\\Documents\\PIDS\\PIDS\\training_guidance\\Stage_I\\Models\\scenes',
    
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
# 2. 核心數學 (防穿模)
# ============================================================

def get_rotated_size(w, d, angle_rad):
    abs_cos = abs(math.cos(angle_rad))
    abs_sin = abs(math.sin(angle_rad))
    new_w = w * abs_cos + d * abs_sin
    new_d = w * abs_sin + d * abs_cos
    return new_w, new_d

# ============================================================
# 3. 檔案與目錄管理
# ============================================================

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[系統] 建立輸出目錄: {directory}")

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
            # 確保 Source 物件不被選取、不被渲染，但存在於場景中
            obj.hide_render = True
            obj.hide_viewport = False # 方便除錯，可視情況改 True
            
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
            
            # 標準化尺寸計算
            current_h = obj.dimensions.z
            target_h = f_type['size'][2]
            if current_h > 0.001:
                scale_factor = target_h / current_h
                obj.scale = (scale_factor, scale_factor, scale_factor)
                bpy.ops.object.transform_apply(scale=True)
                
                f_type['real_size'] = (obj.dimensions.x, obj.dimensions.y, obj.dimensions.z)
                valid_furniture_types.append(f_type)
                print(f"  [O] {source_name} 準備就緒")
        else:
            print(f"  [X] 警告: 找不到 {source_name}")
    return valid_furniture_types

def clear_generated_objects():
    """刪除所有自動生成的物件，保留 Source 和環境"""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        # 邏輯：只要名字不是 source 開頭，也不是背景(ground/bg)，且是 MESH，就刪除
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
# 5. 生成邏輯 (Generate)
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
        
        if not check_overlap((rand_x, y_pos, 0), (rotated_w, rotated_d, h), existing_objects, margin=5):
            valid_pos = (rand_x, y_pos, h/2)
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
    rot_x = math.radians(random.uniform(*CONFIG['randomization']['glass_rotation_x']))
    obj.rotation_euler = (rot_x, 0, final_rot_z)
    
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
    new_obj.location = (valid_pos[0], valid_pos[1], current_h/2)
    new_obj.rotation_euler = (0, 0, final_rot_z)
    
    return new_obj, valid_pos, (final_rotated_w, final_rotated_d, current_h)

# ============================================================
# 6. 主控流程 (Main Loop)
# ============================================================

def export_current_selection(scene_id, base_dir):
    """
    將當前選取的物件匯出為 OBJ
    """
    filename = f"scene_{scene_id:04d}.obj"
    filepath = os.path.join(base_dir, filename)
    
    # 這裡的設定確保座標保留
    # use_selection=True: 只匯出我們選中的生成的傢俱
    # apply_modifiers=True: 確保幾何體正確
    # forward/up: 配合一般 3D 軟體習慣 (Y向前, Z向上)，這保留了 Blender 的座標系視覺感
    
    print(f"  -> 正在匯出: {filename} ...")
    
    try:
        # Blender 3.6+ 新版 OBJ 匯出器
        bpy.ops.wm.obj_export(
            filepath=filepath,
            export_selected_objects=True,
            forward_axis='NEGATIVE_Y', 
            up_axis='Z',
            apply_modifiers=True
        )
    except AttributeError:
        # 舊版 Blender 備用
        bpy.ops.export_scene.obj(
            filepath=filepath, 
            use_selection=True, 
            axis_forward='-Y', 
            axis_up='Z'
        )

def main():
    # 1. 初始化路徑
    output_dir = CONFIG['output_dir']
    
    # 2. 準備 Source
    valid_types = validate_and_setup_sources()
    
    # 3. 計算本次任務的起始與結束 ID
    start_idx = get_next_start_index(output_dir)
    end_idx = start_idx + CONFIG['num_scenes_per_run']
    
    print(f"\n=== PIDS 批次生成開始 ===")
    print(f"目標: 生成 {CONFIG['num_scenes_per_run']} 個場景")
    print(f"編號範圍: {start_idx:04d} ~ {end_idx-1:04d}")
    print(f"輸出目錄: {output_dir}\n")
    
    # 4. 批次迴圈
    for i in range(start_idx, end_idx):
        # A. 清除上一輪的物件
        clear_generated_objects()
        
        # B. 生成新場景
        occupied = []
        jitter_min, jitter_max = CONFIG['scene']['y_bg_jitter']
        scene_y_center = CONFIG['scene']['y_bg_base'] + random.uniform(jitter_min, jitter_max)
        
        # 生成玻璃
        for g_idx in range(random.randint(*CONFIG['glass_objects']['count_range'])):
            g_type = random.choice(CONFIG['glass_objects']['types'])
            obj, pos, size = create_glass_object_safe(g_type, g_idx+1, occupied)
            if obj: occupied.append((pos, size))
            
        # 生成傢俱
        if valid_types:
            target_num = random.randint(*CONFIG['furniture']['count_range'])
            actual_num = min(target_num, len(valid_types))
            selected_types = random.sample(valid_types, actual_num)
            selected_types.sort(key=lambda x: x['real_size'][0] * x['real_size'][1], reverse=True)
            
            for f_idx, f_type in enumerate(selected_types):
                obj, pos, size = create_furniture_adaptive_safe(f_type, f_idx+1, occupied, scene_y_center)
                if obj: occupied.append((pos, size))
        
        # C. 選取生成的物件 (準備匯出)
        bpy.ops.object.select_all(action='DESELECT')
        has_selection = False
        for obj in bpy.data.objects:
            # 只選取新生成的 (diffuse_開頭 或 glass_開頭)
            if obj.name.startswith(('diffuse_', 'glass_')) and obj.type == 'MESH':
                obj.select_set(True)
                has_selection = True
        
        # D. 執行匯出
        if has_selection:
            export_current_selection(i, output_dir)
        else:
            print(f"  [警示] 場景 {i} 生成失敗或為空，跳過匯出。")

    # 5. 收尾：再清除一次最後的場景，保持乾淨 (可選)
    clear_generated_objects()
    print(f"\n=== 全部完成 ===")

if __name__ == "__main__":
    main()
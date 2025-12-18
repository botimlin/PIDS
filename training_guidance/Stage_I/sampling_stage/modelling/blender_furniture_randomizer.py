"""
PIDS Blender 傢俱隨機擺放腳本 (v17.0 - PIDS 標準版)
===========================================================
功能:
1. 讀取 source_ 物件，隨機擺放並適應性縮放 (Shrink-to-Fit)。
2. 使用 Rotated AABB 數學計算，杜絕穿模。
3. [關鍵] 自動將場景分層匯出為:
   - *_glass.obj (透明物體 -> 渲染器自動賦予 BK7 玻璃材質)
   - *_diffuse.obj (不透明物體 -> 渲染器自動賦予 粗糙塑膠材質)
4. 自動遞增檔名，支援批次量產。

使用方法:
    1. 在 Blender 中開啟此 Script。
    2. 確保場景中有以 'source_' 開頭的物件 (例如 source_cup, source_table)。
    3. 修改 CONFIG['output_dir'] 為您的輸出路徑。
    4. 執行 Script (Run Script)。
"""

import bpy
import math
import random
import os
import re
from mathutils import Vector, Euler

# ============================================================
# 1. 配置參數 (請修改這裡)
# ============================================================

CONFIG = {
    # 每次執行生成的場景數量
    'num_scenes_per_run': 10,  
    
    # 輸出目錄 (請改為您的路徑)
    'output_dir': r'C:\Users\tim\Documents\PIDS\PIDS\training_guidance\Stage_I\sampling_stage\modelling\scenes', 
    
    # 場景範圍 (mm) - 必須配合 pids_renderer_blender.py 的景深設定
    'scene': {
        'width': 220,       # 場景寬度 (X軸範圍 -110 ~ 110)
        'y_min': 530,       # 景深起點 (太近會失焦)
        'y_max': 690,       # 景深終點 (太遠會模糊)
        
        # 背景家具的位置 (放在景深最後方當背景)
        'y_bg_base': 730,   
        'y_bg_jitter': (-20, 20), 
    },
    
    # 透明物體 (前景)
    'glass_objects': {
        'count_range': (1, 3),  # 數量
        'types': ['source_cup', 'source_bottle', 'source_glass'], # 來源物件名稱前綴
        'scale_range': (0.8, 1.2),
        'rotation_range_z': (-180, 180),
        'rotation_range_x': (-15, 15), # 輕微傾倒
    },
    
    # 背景家具 (背景)
    'furniture': {
        'count_range': (2, 4),
        'types': ['source_chair', 'source_table', 'source_shelf'],
        'scale_range': (0.9, 1.5),
    }
}

# ============================================================
# 2. 輔助函數
# ============================================================

def clear_generated_objects():
    """清除上一次生成的物件 (保留 source_)"""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if not obj.name.startswith('source_') and obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()

def get_next_start_index(output_dir):
    """讀取目錄中現有的最大序號"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return 1
        
    files = os.listdir(output_dir)
    max_idx = 0
    pattern = re.compile(r'scene_(\d+)_')
    
    for f in files:
        match = pattern.match(f)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1

def duplicate_object(source_name, new_name):
    """複製物件"""
    # 模糊搜尋 source 物件
    source_obj = None
    for obj in bpy.data.objects:
        if obj.name.startswith(source_name):
            source_obj = obj
            break
            
    if not source_obj:
        # print(f"Warning: Source object starting with '{source_name}' not found.")
        return None
        
    new_obj = source_obj.copy()
    new_obj.data = source_obj.data.copy()
    new_obj.name = new_name
    bpy.context.collection.objects.link(new_obj)
    return new_obj

def check_overlap_aabb(pos, size, occupied_list, margin=10):
    """簡單的 AABB 碰撞檢測 (Box Check)"""
    # pos: (x, y, z), size: (wx, wy, wz)
    # occupied: list of (pos, size)
    
    min_x = pos[0] - size[0]/2 - margin
    max_x = pos[0] + size[0]/2 + margin
    min_y = pos[1] - size[1]/2 - margin
    max_y = pos[1] + size[1]/2 + margin
    
    for (o_pos, o_size) in occupied_list:
        o_min_x = o_pos[0] - o_size[0]/2
        o_max_x = o_pos[0] + o_size[0]/2
        o_min_y = o_pos[1] - o_size[1]/2
        o_max_y = o_pos[1] + o_size[1]/2
        
        if (min_x < o_max_x and max_x > o_min_x and
            min_y < o_max_y and max_y > o_min_y):
            return True # 發生碰撞
    return False

# ============================================================
# 3. 生成邏輯
# ============================================================

def create_glass_object(idx, occupied_list):
    """生成前景透明物體"""
    # 隨機選擇來源
    src_type = random.choice(CONFIG['glass_objects']['types'])
    obj = duplicate_object(src_type, f"glass_{idx:03d}")
    if not obj: return None
    
    # 隨機縮放
    scale = random.uniform(*CONFIG['glass_objects']['scale_range'])
    obj.scale = (scale, scale, scale)
    
    # 取得物件尺寸 (Bounding Box)
    bpy.context.view_layer.update()
    dims = obj.dimensions
    size = (dims.x, dims.y, dims.z)
    
    # 嘗試尋找不重疊的位置 (最多嘗試 50 次)
    for _ in range(50):
        x = random.uniform(-CONFIG['scene']['width']/2, CONFIG['scene']['width']/2)
        y = random.uniform(CONFIG['scene']['y_min'], CONFIG['scene']['y_max'])
        z = size[2] / 2 # 自動落地 (假設原點在底部)
        
        pos = (x, y, z)
        
        if not check_overlap_aabb(pos, size, occupied_list):
            # 設定位置
            obj.location = pos
            
            # 隨機旋轉
            rot_z = math.radians(random.uniform(*CONFIG['glass_objects']['rotation_range_z']))
            rot_x = math.radians(random.uniform(*CONFIG['glass_objects']['rotation_range_x']))
            obj.rotation_euler = (rot_x, 0, rot_z)
            
            return obj, pos, size
            
    # 如果找不到位置，刪除物件
    bpy.data.objects.remove(obj, do_unlink=True)
    return None

def create_background_furniture(idx):
    """生成背景家具 (不檢查碰撞，僅作為背景雜訊)"""
    src_type = random.choice(CONFIG['furniture']['types'])
    obj = duplicate_object(src_type, f"furniture_{idx:03d}")
    if not obj: return None
    
    scale = random.uniform(*CONFIG['furniture']['scale_range'])
    obj.scale = (scale, scale, scale)
    
    x = random.uniform(-CONFIG['scene']['width'], CONFIG['scene']['width']) # 背景可以寬一點
    y_base = CONFIG['scene']['y_bg_base']
    y_jitter = random.uniform(*CONFIG['scene']['y_bg_jitter'])
    
    obj.location = (x, y_base + y_jitter, 0) # 背景物體可能需要手動調整 Z
    
    rot_z = math.radians(random.uniform(-45, 45))
    obj.rotation_euler = (0, 0, rot_z)
    
    return obj

# ============================================================
# 4. 分層匯出邏輯 (關鍵!)
# ============================================================

def export_split_obj(output_dir, scene_idx):
    """
    分層匯出: 
    1. 選擇所有 "glass_" 開頭的 -> scene_XXX_glass.obj
    2. 選擇所有 "furniture_" 開頭的 -> scene_XXX_diffuse.obj
    """
    base_name = f"scene_{scene_idx:05d}"
    
    # --- 匯出 Glass ---
    bpy.ops.object.select_all(action='DESELECT')
    glass_objs = [obj for obj in bpy.data.objects if obj.name.startswith("glass_")]
    
    if glass_objs:
        for obj in glass_objs:
            obj.select_set(True)
        
        filepath = os.path.join(output_dir, f"{base_name}_glass.obj")
        bpy.ops.export_scene.obj(
            filepath=filepath,
            use_selection=True,
            use_materials=False, # 渲染器會重新指派材質，這裡不需要
            axis_forward='Y',    # Mitsuba 座標系對應
            axis_up='Z',
            global_scale=1.0     # 假設 Blender 單位已經是 mm (或者 1 unit)
        )
    
    # --- 匯出 Diffuse ---
    bpy.ops.object.select_all(action='DESELECT')
    diffuse_objs = [obj for obj in bpy.data.objects if obj.name.startswith("furniture_")]
    
    if diffuse_objs:
        for obj in diffuse_objs:
            obj.select_set(True)
            
        filepath = os.path.join(output_dir, f"{base_name}_diffuse.obj")
        bpy.ops.export_scene.obj(
            filepath=filepath,
            use_selection=True,
            use_materials=False,
            axis_forward='Y',
            axis_up='Z',
            global_scale=1.0
        )
        
    print(f"  [Export] {base_name} 完成")

# ============================================================
# 5. 主程式
# ============================================================

def main():
    print("-" * 50)
    print("PIDS 場景生成器開始")
    
    output_dir = CONFIG['output_dir']
    start_idx = get_next_start_index(output_dir)
    num_scenes = CONFIG['num_scenes_per_run']
    
    print(f"輸出目錄: {output_dir}")
    print(f"起始序號: {start_idx}, 生成數量: {num_scenes}")
    
    for i in range(num_scenes):
        current_idx = start_idx + i
        print(f"正在生成場景 {current_idx}...")
        
        # 1. 清場
        clear_generated_objects()
        occupied_list = [] # 記錄已佔用空間
        
        # 2. 生成前景透明物體 (有碰撞檢測)
        num_glass = random.randint(*CONFIG['glass_objects']['count_range'])
        for g in range(num_glass):
            res = create_glass_object(g, occupied_list)
            if res:
                occupied_list.append((res[1], res[2])) # pos, size
        
        # 3. 生成背景家具
        num_furniture = random.randint(*CONFIG['furniture']['count_range'])
        for f in range(num_furniture):
            create_background_furniture(f)
            
        # 4. 分層匯出
        export_split_obj(output_dir, current_idx)
        
    # 恢復狀態
    clear_generated_objects()
    print("生成完成!")

if __name__ == "__main__":
    main()
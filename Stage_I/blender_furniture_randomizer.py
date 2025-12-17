"""
PIDS Blender 傢俱隨機擺放腳本
==============================

此腳本用於在 Blender 中隨機擺放背景傢俱和透明物體。
在 Blender 的 Scripting 標籤中執行。

使用方式:
1. 在 Blender 中開啟基礎場景（已有地面和牆壁）
2. 切換到 Scripting 標籤
3. 貼上此腳本並執行
4. 腳本會自動建立並隨機擺放傢俱

配置:
- NUM_SCENES: 要生成的場景數量
- 傢俱類型、數量、位置範圍都可調整
"""

import bpy
import math
import random
import os

# ============================================================
# 配置參數
# ============================================================

CONFIG = {
    'num_scenes': 5,  # 測試時建議先設少一點
    'output_dir': './scenes',
    
    # 場景範圍 (mm)
    'scene': {
        'width': 250,
        'depth': 200,
        'height': 200,
        'y_min': 528,
        'y_max': 695,
        'y_bg': 720,
    },
    
    # 透明物體配置
    'glass_objects': {
        'count_range': (1, 3),
        'types': [
            {'name': 'glass_door',      'size': (90, 150, 5),  'weight': 2}, # 厚度改為 5mm 比較合理
            {'name': 'glass_window',    'size': (100, 80, 5),  'weight': 2},
            {'name': 'glass_partition', 'size': (50, 40, 5),   'weight': 3},
            {'name': 'glass_table',     'size': (80, 80, 5),   'weight': 2},
        ],
    },
    
    # 背景傢俱配置
    'furniture': {
        'count_range': (2, 4),
        'types': [
            {'name': 'bookshelf', 'size': (80, 30, 180), 'color': (0.4, 0.25, 0.15), 'weight': 2},
            {'name': 'cabinet',   'size': (100, 40, 90), 'color': (0.35, 0.22, 0.12), 'weight': 2},
            {'name': 'table',     'size': (80, 50, 45),  'color': (0.45, 0.3, 0.18),  'weight': 3},
            {'name': 'chair',     'size': (45, 45, 85),  'color': (0.5, 0.35, 0.2),   'weight': 2},
            {'name': 'sofa',      'size': (120, 60, 70), 'color': (0.3, 0.3, 0.35),   'weight': 1},
        ],
    },
    
    'randomization': {
        'glass_rotation_y': (-30, 30),
        'glass_rotation_x': (-5, 5),
        'furniture_rotation_z': (-15, 15),
    },
}

# ============================================================
# 工具函數
# ============================================================

def clear_objects(keep_basics=True):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        # 保留相機、燈光、地面、牆壁
        if keep_basics and (obj.name.startswith('ground_') or 
                            obj.name.startswith('bg_wall_') or 
                            obj.type in ['CAMERA', 'LIGHT']):
            continue
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()

def create_material(name, color, is_glass=False):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        
        if is_glass:
            bsdf.inputs["Transmission"].default_value = 1.0
            bsdf.inputs["Roughness"].default_value = 0.05
            bsdf.inputs["IOR"].default_value = 1.45
            bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
            # Eevee 设置 (如果是 Eevee 渲染器需要開啟)
            mat.blend_method = 'BLEND'
            mat.shadow_method = 'NONE'
        else:
            bsdf.inputs["Base Color"].default_value = (*color, 1)
            bsdf.inputs["Roughness"].default_value = 0.8
    return mat

def weighted_choice(items):
    total = sum(item['weight'] for item in items)
    r = random.uniform(0, total)
    cumulative = 0
    for item in items:
        cumulative += item['weight']
        if r <= cumulative:
            return item
    return items[-1]

def check_overlap(new_pos, new_size, existing_objects, margin=10):
    """檢查是否重疊 (簡單 AABB)"""
    nx, ny, _ = new_pos  # 忽略 Z 軸
    nw, nd, _ = new_size # 忽略高度
    
    for obj_pos, obj_size in existing_objects:
        ox, oy, _ = obj_pos
        ow, od, _ = obj_size
        
        # 計算兩物體中心點距離是否小於 兩者半寬之和
        x_overlap = abs(nx - ox) < (nw + ow) / 2 + margin
        y_overlap = abs(ny - oy) < (nd + od) / 2 + margin
        
        if x_overlap and y_overlap:
            return True
    return False

# ============================================================
# 物體建立函數
# ============================================================

def create_glass_object(glass_type, index, y_pos):
    """建立透明玻璃物體"""
    name = f"{glass_type['name']}_{index:02d}"
    size = glass_type['size']
    
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    obj.name = name
    
    # 尺寸設定 (mm -> m)
    obj.scale = (size[0]/1000, size[1]/1000, size[2]/1000) # Blender Cube 預設 X,Y,Z
    
    # 計算隨機 X 位置
    x_range = (CONFIG['scene']['width'] - size[0]) / 2
    x_pos = random.uniform(-x_range, x_range)
    z_pos = size[2] / 2 # 底部貼地 (假設 Z 是高度)
    
    # 這裡假設配置檔 size 是 (W, D, H) 或者是 (W, H, Thickness)? 
    # 原代碼 glass_type['size'] 似乎是 (W, H, D)。這裡修正為直立放置。
    # 重設 Scale 對應：X=寬, Y=厚, Z=高
    # 注意：原代碼配置是 (W, H, D) -> (90, 150, 1)。通常 D=1 是厚度。
    # 為了直立，我們要讓 Z 軸是 150。
    w, h, d = size
    obj.scale = (w/1000, d/1000, h/1000) # X=寬, Y=厚, Z=高
    
    z_pos = h / 2
    obj.location = (x_pos / 1000, y_pos / 1000, z_pos / 1000)
    
    # 隨機旋轉
    rot_y = math.radians(random.uniform(*CONFIG['randomization']['glass_rotation_y']))
    rot_x = math.radians(random.uniform(*CONFIG['randomization']['glass_rotation_x']))
    obj.rotation_euler = (rot_x, 0, rot_y) # 繞 Z 軸旋轉才是在地面上轉，原代碼 rot_y 可能是想繞 Z?
    # 修正：通常傢俱旋轉是繞 Z 軸 (垂直軸)。原代碼寫 rot_y 可能是想做傾倒效果？
    # 這裡依照原邏輯保留，但建議確認是否要是 Z 軸。
    
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    mat = create_material("Glass_Clear", (1, 1, 1), is_glass=True)
    obj.data.materials.append(mat)
    
    return obj, (x_pos, y_pos, 0), (w, d, h)

def create_furniture_primitive(furniture_type, index, pos, rot_z):
    """當找不到 Asset 時，建立替代用的方塊"""
    name = f"diffuse_{furniture_type['name']}_{index:02d}"
    w, d, h = furniture_type['size']
    
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    obj.name = name
    
    obj.scale = (w/1000, d/1000, h/1000)
    obj.location = (pos[0]/1000, pos[1]/1000, pos[2]/1000)
    obj.rotation_euler = (0, 0, rot_z)
    
    bpy.ops.object.transform_apply(scale=True)
    
    color = furniture_type.get('color', (0.5, 0.5, 0.5))
    mat_name = f"Mat_{furniture_type['name']}"
    mat = create_material(mat_name, color)
    obj.data.materials.append(mat)
    
    return obj

def create_furniture_smart(furniture_type, index, existing_objects):
    """
    智能建立傢俱：嘗試尋找來源，找不到則建立方塊，並包含位置計算
    """
    w, d, h = furniture_type['size']
    
    # 嘗試尋找合適的空位 (嘗試 50 次)
    valid_pos = None
    final_rot_z = 0
    
    for _ in range(50):
        # 隨機 X
        x_range = (CONFIG['scene']['width'] - w) / 2
        rand_x = random.uniform(-x_range, x_range)
        
        # Y 位置固定在背景牆附近，有些許前後隨機
        y_center = CONFIG['scene']['y_bg']
        rand_y = random.uniform(y_center - 20, y_center + 20)
        
        rand_z = h / 2
        
        candidate_pos = (rand_x, rand_y, 0)
        candidate_size = (w, d, h)
        
        if not check_overlap(candidate_pos, candidate_size, existing_objects, margin=20):
            valid_pos = (rand_x, rand_y, rand_z)
            final_rot_z = math.radians(random.uniform(*CONFIG['randomization']['furniture_rotation_z']))
            break
    
    if valid_pos is None:
        print(f"  警告: 無法為 {furniture_type['name']} 找到空位，跳過。")
        return None, None, None

    # 嘗試從 Asset 複製
    source_name = f"source_{furniture_type['name']}"
    source_obj = bpy.data.objects.get(source_name)
    
    obj = None
    if source_obj:
        obj = source_obj.copy()
        if source_obj.data:
            obj.data = source_obj.data.copy()
        bpy.context.collection.objects.link(obj)
        obj.name = f"diffuse_{furniture_type['name']}_{index:02d}"
        obj.location = (valid_pos[0]/1000, valid_pos[1]/1000, valid_pos[2]/1000)
        obj.rotation_euler = (0, 0, final_rot_z)
    else:
        # Fallback: 建立方塊
        obj = create_furniture_primitive(furniture_type, index, valid_pos, final_rot_z)
        
    return obj, valid_pos, (w, d, h)

# ============================================================
# 場景生成
# ============================================================

def setup_base_scene():
    """建立基礎場景"""
    # 檢查是否已有基礎物體
    if not bpy.data.objects.get('ground_floor'):
        bpy.ops.mesh.primitive_plane_add(size=1)
        ground = bpy.context.active_object
        ground.name = 'ground_floor'
        # 調整為符合 250x200 左右的比例
        ground.location = (0, 0.600, 0) 
        ground.scale = (0.5, 0.5, 1) 
        bpy.ops.object.transform_apply(scale=True)
        mat = create_material("Diffuse_Floor", (0.35, 0.30, 0.25))
        ground.data.materials.append(mat)

    if not bpy.data.objects.get('bg_wall_back'):
        bpy.ops.mesh.primitive_plane_add(size=1)
        wall = bpy.context.active_object
        wall.name = 'bg_wall_back'
        wall.location = (0, 0.800, 0.25)
        wall.rotation_euler = (math.radians(90), 0, 0)
        wall.scale = (0.5, 0.5, 1)
        bpy.ops.object.transform_apply(rotation=True, scale=True)
        mat = create_material("Diffuse_Wall", (0.7, 0.68, 0.65))
        wall.data.materials.append(mat)

def generate_scene(scene_index):
    print(f"\n生成場景 {scene_index + 1}...")
    clear_objects(keep_basics=True)
    
    # 記錄佔用空間 [(pos, size), ...]
    # 先放入玻璃的位置，避免傢俱穿插玻璃
    occupied_spaces = [] 
    
    # 1. 放置透明玻璃物體
    glass_count = random.randint(*CONFIG['glass_objects']['count_range'])
    print(f"  放置 {glass_count} 個透明物體...")
    
    for i in range(glass_count):
        glass_type = weighted_choice(CONFIG['glass_objects']['types'])
        y_pos = random.uniform(CONFIG['scene']['y_min'], CONFIG['scene']['y_max'])
        
        # 修正：呼叫 create_glass_object
        obj, pos, size = create_glass_object(glass_type, i + 1, y_pos)
        occupied_spaces.append((pos, size))
        print(f"    - {obj.name}")

    # 2. 放置背景傢俱
    furniture_count = random.randint(*CONFIG['furniture']['count_range'])
    print(f"  放置 {furniture_count} 個傢俱...")
    
    for i in range(furniture_count):
        furniture_type = weighted_choice(CONFIG['furniture']['types'])
        
        # 修正：呼叫 smart 函數，並傳入 occupied_spaces
        obj, pos, size = create_furniture_smart(furniture_type, i + 1, occupied_spaces)
        
        if obj:
            occupied_spaces.append((pos, size))
            print(f"    - {obj.name}")
            
    return True

def export_scene(scene_index, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
            
    filepath = os.path.join(output_dir, f"scene_{scene_index + 1:04d}.obj")
    
    # Blender 3.6+ 使用 wm.obj_export，舊版可能使用 export_scene.obj
    if hasattr(bpy.ops.wm, 'obj_export'):
        bpy.ops.wm.obj_export(
            filepath=filepath,
            export_selected_objects=True,
            forward_axis='NEGATIVE_Y',
            up_axis='Z',
            export_materials=True,
            export_triangulated_mesh=True,
            apply_modifiers=True,
        )
    else:
        # 舊版 Blender 兼容
        bpy.ops.export_scene.obj(
            filepath=filepath,
            use_selection=True,
            axis_forward='-Y',
            axis_up='Z',
            use_materials=True,
            use_triangles=True,
        )
    
    print(f"  匯出: {filepath}")

# ============================================================
# 主程式
# ============================================================

def main():
    print("=" * 60)
    print("PIDS 傢俱隨機擺放器 (修正版)")
    print("=" * 60)
    
    # 建立基礎場景
    setup_base_scene()
    
    # 路徑處理
    blend_path = bpy.data.filepath
    if not blend_path:
        print("警告: 尚未儲存 .blend 檔案，將使用腳本當前目錄作為輸出基準。")
        blend_dir = os.getcwd()
    else:
        blend_dir = os.path.dirname(blend_path)
        
    output_dir = os.path.join(blend_dir, CONFIG['output_dir'])
    
    num_scenes = CONFIG['num_scenes']
    
    for i in range(num_scenes):
        generate_scene(i)
        export_scene(i, output_dir)
    
    print("\n" + "=" * 60)
    print("全部完成！")

if __name__ == "__main__":
    main()
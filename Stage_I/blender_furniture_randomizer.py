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
    # 生成場景數量
    'num_scenes': 10,
    
    # 輸出目錄 (相對於 .blend 檔案位置)
    'output_dir': './scenes',
    
    # 場景範圍 (mm) - 符合 PIDS 配置
    'scene': {
        'width': 250,       # X 方向
        'depth': 200,       # Y 方向 (528-695mm 範圍 ≈ 170mm)
        'height': 200,      # Z 方向
        'y_min': 528,       # 景深起點
        'y_max': 695,       # 景深終點
        'y_bg': 720,        # 背景傢俱 Y 位置
    },
    
    # 透明物體配置 (大型平面玻璃優先)
    'glass_objects': {
        'count_range': (1, 3),  # 每場景 1-3 個透明物體
        'types': [
            {'name': 'glass_door',      'size': (90, 150, 1),  'weight': 2},
            {'name': 'glass_window',    'size': (100, 80, 1),  'weight': 2},
            {'name': 'glass_partition', 'size': (50, 40, 1),   'weight': 3},
            {'name': 'glass_table',     'size': (80, 80, 1),   'weight': 2},
            {'name': 'glass_cabinet',   'size': (40, 60, 1),   'weight': 2},
        ],
    },
    
    # 背景傢俱配置 (5 種)
    'furniture': {
        'count_range': (2, 4),  # 每場景 2-4 個傢俱
        'types': [
            {
                'name': 'bookshelf',
                'size': (80, 30, 180),  # W x D x H (mm)
                'color': (0.4, 0.25, 0.15),  # 木頭色
                'weight': 2,
            },
            {
                'name': 'cabinet',
                'size': (100, 40, 90),
                'color': (0.35, 0.22, 0.12),
                'weight': 2,
            },
            {
                'name': 'table',
                'size': (80, 50, 45),
                'color': (0.45, 0.3, 0.18),
                'weight': 3,
            },
            {
                'name': 'chair',
                'size': (45, 45, 85),
                'color': (0.5, 0.35, 0.2),
                'weight': 2,
            },
            {
                'name': 'sofa',
                'size': (120, 60, 70),
                'color': (0.3, 0.3, 0.35),  # 灰色
                'weight': 1,
            },
        ],
    },
    
    # 隨機化範圍
    'randomization': {
        'glass_rotation_y': (-30, 30),   # 玻璃 Y 軸旋轉 (度)
        'glass_rotation_x': (-15, 15),   # 玻璃 X 軸旋轉 (度)
        'furniture_rotation_z': (-15, 15),  # 傢俱 Z 軸旋轉 (度)
    },
}


# ============================================================
# 工具函數
# ============================================================

def clear_objects(keep_basics=True):
    """
    清除場景中的物體
    
    keep_basics: 保留 ground_ 和 bg_wall_ 開頭的物體
    """
    bpy.ops.object.select_all(action='DESELECT')
    
    for obj in bpy.data.objects:
        if keep_basics and (obj.name.startswith('ground_') or obj.name.startswith('bg_wall_')):
            continue
        if obj.type == 'MESH':
            obj.select_set(True)
    
    bpy.ops.object.delete()


def create_material(name, color, is_glass=False):
    """建立材質"""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        
        if is_glass:
            bsdf.inputs["Transmission"].default_value = 1.0
            bsdf.inputs["Roughness"].default_value = 0.0
            bsdf.inputs["IOR"].default_value = 1.5
            bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
        else:
            bsdf.inputs["Base Color"].default_value = (*color, 1)
            bsdf.inputs["Roughness"].default_value = 0.8
    
    return mat


def weighted_choice(items):
    """根據權重隨機選擇"""
    total = sum(item['weight'] for item in items)
    r = random.uniform(0, total)
    cumulative = 0
    for item in items:
        cumulative += item['weight']
        if r <= cumulative:
            return item
    return items[-1]


def check_overlap(new_pos, new_size, existing_objects, margin=10):
    """
    檢查新物體是否與現有物體重疊
    
    margin: 物體間最小間距 (mm)
    """
    nx, ny, nz = new_pos
    nw, nd, nh = new_size
    
    for obj_pos, obj_size in existing_objects:
        ox, oy, oz = obj_pos
        ow, od, oh = obj_size
        
        # 簡單的 AABB 碰撞檢測
        if (abs(nx - ox) < (nw + ow) / 2 + margin and
            abs(ny - oy) < (nd + od) / 2 + margin):
            return True
    
    return False


# ============================================================
# 物體建立函數
# ============================================================

def create_glass_object(glass_type, index, y_pos):
    """
    建立透明玻璃物體
    """
    name = f"{glass_type['name']}_{index:02d}"
    size = glass_type['size']  # (W, H, D) in mm
    
    # 轉換為 Blender 單位 (m)
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.active_object
    obj.name = name
    
    # 設定尺寸 (mm -> m)
    obj.scale = (size[0] / 2000, size[2] / 2000, size[1] / 2000)  # Blender: X, Z, Y
    
    # 隨機位置
    x_range = CONFIG['scene']['width'] / 2 - size[0] / 2
    x_pos = random.uniform(-x_range, x_range)
    z_pos = size[1] / 2  # 物體底部在地面上
    
    obj.location = (x_pos / 1000, y_pos / 1000, z_pos / 1000)
    
    # 隨機旋轉
    rot_y = math.radians(random.uniform(*CONFIG['randomization']['glass_rotation_y']))
    rot_x = math.radians(random.uniform(*CONFIG['randomization']['glass_rotation_x']))
    obj.rotation_euler = (rot_x, 0, rot_y)
    
    # Apply transforms
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # 指定玻璃材質
    mat = create_material("Glass_Clear", (1, 1, 1), is_glass=True)
    obj.data.materials.append(mat)
    
    return obj, (x_pos, y_pos, z_pos), size


def create_furniture_from_asset(furniture_type, index, existing_objects):
    """
    從現有資產複製傢俱，而不是建立方塊
    需確保場景中有對應名稱的來源物體
    """
    # 假設你的來源物體命名為 "source_chair", "source_table" 等
    source_name = f"source_{furniture_type['name']}" 
    source_obj = bpy.data.objects.get(source_name)

    if not source_obj:
        print(f"警告: 找不到來源物體 {source_name}，改為生成方塊。")
        # 回退到建立方塊的邏輯...
        return create_furniture_from_asset(furniture_type, index, existing_objects)

    # 1. 複製物體
    new_obj = source_obj.copy()
    if source_obj.data:
        new_obj.data = source_obj.data.copy()
    
    # 2. 連結到當前場景
    bpy.context.collection.objects.link(new_obj)
    
    # 3. 重新命名 (符合 PIDS 格式)
    new_obj.name = f"diffuse_{furniture_type['name']}_{index:02d}"
    
    # 4. 取得來源物體的尺寸 (用於計算不重疊位置)
    # 注意：這裡假設來源物體已經應用了旋轉與縮放
    size = (new_obj.dimensions.x * 1000, new_obj.dimensions.y * 1000, new_obj.dimensions.z * 1000)
    
    # ... (接下來的位置計算邏輯與原腳本相同) ...
    
    # 計算位置 (這裡簡化，直接沿用原本的隨機邏輯)
    x_range = CONFIG['scene']['width'] / 2 - size[0] / 2
    x_pos = random.uniform(-x_range, x_range)
    y_pos = CONFIG['scene']['y_bg']
    z_pos = size[2] / 2 
    
    new_obj.location = (x_pos / 1000, y_pos / 1000, z_pos / 1000)
    
    # 隨機旋轉
    rot_z = math.radians(random.uniform(*CONFIG['randomization']['furniture_rotation_z']))
    new_obj.rotation_euler = (0, 0, rot_z)
    
    return new_obj, (x_pos, y_pos, z_pos), size

# ============================================================
# 場景生成
# ============================================================

def setup_base_scene():
    """
    建立基礎場景（地面和牆壁）
    """
    # 檢查是否已有基礎物體
    has_ground = any(obj.name.startswith('ground_') for obj in bpy.data.objects)
    has_wall = any(obj.name.startswith('bg_wall_') for obj in bpy.data.objects)
    
    if has_ground and has_wall:
        return
    
    # 建立地面
    if not has_ground:
        bpy.ops.mesh.primitive_plane_add(size=1)
        ground = bpy.context.active_object
        ground.name = 'ground_floor'
        ground.location = (0, 0.610, 0)
        ground.scale = (0.125, 0.150, 1)
        bpy.ops.object.transform_apply(scale=True)
        
        mat = create_material("Diffuse_Floor", (0.35, 0.30, 0.25))
        ground.data.materials.append(mat)
    
    # 建立背景牆
    if not has_wall:
        bpy.ops.mesh.primitive_plane_add(size=1)
        wall = bpy.context.active_object
        wall.name = 'bg_wall_back'
        wall.location = (0, 0.750, 0.100)
        wall.rotation_euler = (math.radians(90), 0, 0)
        wall.scale = (0.150, 0.120, 1)
        bpy.ops.object.transform_apply(rotation=True, scale=True)
        
        mat = create_material("Diffuse_Wall", (0.7, 0.68, 0.65))
        wall.data.materials.append(mat)


def generate_scene(scene_index):
    """
    生成單一場景
    """
    print(f"\n生成場景 {scene_index + 1}...")
    
    # 清除舊物體（保留基礎）
    clear_objects(keep_basics=True)
    
    # 記錄已放置的物體
    placed_objects = []
    
    # 1. 放置透明玻璃物體
    glass_count = random.randint(*CONFIG['glass_objects']['count_range'])
    print(f"  放置 {glass_count} 個透明物體...")
    
    for i in range(glass_count):
        glass_type = weighted_choice(CONFIG['glass_objects']['types'])
        
        # 隨機 Y 位置（在景深範圍內）
        y_pos = random.uniform(CONFIG['scene']['y_min'], CONFIG['scene']['y_max'])
        
        obj, pos, size = create_glass_object(glass_type, i + 1, y_pos)
        placed_objects.append((pos, size))
        print(f"    - {obj.name} at Y={y_pos:.0f}mm")
    
    # 2. 放置背景傢俱
    furniture_count = random.randint(*CONFIG['furniture']['count_range'])
    print(f"  放置 {furniture_count} 個傢俱...")
    
    furniture_placed = []
    for i in range(furniture_count):
        furniture_type = weighted_choice(CONFIG['furniture']['types'])
        obj, pos, size = create_furniture_from_asset(furniture_type, i + 1, furniture_placed)
        furniture_placed.append((pos, size))
        print(f"    - {obj.name}")
    
    return True


def export_scene(scene_index, output_dir):
    """
    匯出場景為 OBJ
    """
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 選擇所有 mesh 物體
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    
    # 匯出
    filepath = os.path.join(output_dir, f"scene_{scene_index + 1:04d}.obj")
    bpy.ops.wm.obj_export(
        filepath=filepath,
        export_selected_objects=True,
        forward_axis='NEGATIVE_Y',
        up_axis='Z',
        export_materials=True,
        export_triangulated_mesh=True,
        apply_modifiers=True,
    )
    
    print(f"  匯出: {filepath}")
    return filepath


# ============================================================
# 主程式
# ============================================================

def main():
    """主程式"""
    print("=" * 60)
    print("PIDS 傢俱隨機擺放器")
    print("=" * 60)
    
    # 設定單位
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'
    bpy.context.scene.unit_settings.scale_length = 0.001
    
    # 建立基礎場景
    setup_base_scene()
    
    # 取得輸出目錄
    blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
    output_dir = os.path.join(blend_dir, CONFIG['output_dir'])
    
    # 生成場景
    num_scenes = CONFIG['num_scenes']
    print(f"\n將生成 {num_scenes} 個場景...")
    
    for i in range(num_scenes):
        generate_scene(i)
        export_scene(i, output_dir)
    
    print("\n" + "=" * 60)
    print(f"完成！共生成 {num_scenes} 個場景")
    print(f"輸出目錄: {output_dir}")
    print("=" * 60)


# 執行
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PIDS 渲染器 v8 - 雙次渲染法
===========================

方法：不依賴 Stokes integrator，直接渲染兩次
- 第一次：光源 + 相機都用 0° 偏振片 → I∥
- 第二次：光源 0°，相機 90° 偏振片 → I⊥

這樣可以確保偏振效果正確。
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import sys
import argparse
import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    print("請安裝: pip install opencv-python")
    sys.exit(1)


# ============================================================
# 配置
# ============================================================
CONFIG = {
    'camera': {
        'resolution': (640, 480),
        'fov': 55.0,
        'baseline_mm': 65.0,
        'height_mm': 20.0,
        'position_y': -400.0,
    },
    'scene': {
        'focus_y': -750.0,
        'focus_z': 20.0,
    },
    'light': {
        'height_mm': 250.0,
        'size_mm': 150.0,
        'intensity': 80000.0,
    },
    'render': {
        'samples': 4096,
        'max_depth': 8,
    },
}


# ============================================================
# OBJ 解析器
# ============================================================
def parse_obj_file(filepath):
    """解析 OBJ 檔案"""
    objects = {}
    current_name = 'default'
    all_verts = []
    
    print(f"  解析 OBJ: {filepath}")
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            cmd = parts[0]
            
            if cmd == 'o' and len(parts) > 1:
                current_name = parts[1]
                if current_name not in objects:
                    objects[current_name] = {
                        'verts': [],
                        'faces': [],
                        'vert_start': len(all_verts)
                    }
                print(f"    物件: {current_name}")
                
            elif cmd == 'v' and len(parts) >= 4:
                v = [float(parts[1]), float(parts[2]), float(parts[3])]
                all_verts.append(v)
                if current_name in objects:
                    objects[current_name]['verts'].append(v)
                    
            elif cmd == 'f':
                if current_name not in objects:
                    objects[current_name] = {'verts': [], 'faces': [], 'vert_start': 0}
                
                indices = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0])
                    local_idx = idx - 1 - objects[current_name]['vert_start']
                    indices.append(local_idx)
                
                if len(indices) >= 3:
                    objects[current_name]['faces'].append(indices[:3])
                if len(indices) == 4:
                    objects[current_name]['faces'].append([indices[0], indices[2], indices[3]])
    
    return objects


def write_ply_binary(filepath, vertices, faces):
    """寫入二進位 PLY"""
    import struct
    
    with open(filepath, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""
        f.write(header.encode('ascii'))
        
        for v in vertices:
            f.write(struct.pack('<fff', float(v[0]), float(v[1]), float(v[2])))
        
        for face in faces:
            f.write(struct.pack('<B', 3))
            f.write(struct.pack('<iii', int(face[0]), int(face[1]), int(face[2])))


# ============================================================
# Mitsuba
# ============================================================
def init_mitsuba():
    """初始化 Mitsuba 偏振模式"""
    import mitsuba as mi
    
    try:
        mi.set_variant('cuda_ad_spectral_polarized')
        print(f"  Mitsuba: {mi.variant()} (GPU)")
    except:
        try:
            mi.set_variant('scalar_spectral_polarized')
            print(f"  Mitsuba: {mi.variant()} (CPU)")
        except Exception as e:
            print(f"  錯誤: {e}")
            raise
    
    return mi


def init_mitsuba_rgb():
    """RGB 模式"""
    import mitsuba as mi
    try:
        mi.set_variant('cuda_ad_rgb')
    except:
        mi.set_variant('scalar_rgb')
    return mi


# ============================================================
# 場景建立
# ============================================================
def build_scene(mi, obj_path, temp_dir, analyzer_angle):
    """
    建立場景
    
    analyzer_angle: 相機偏振片角度
    - 0° = I∥ (與光源平行)
    - 90° = I⊥ (與光源垂直)
    """
    
    objects = parse_obj_file(obj_path)
    
    cam_y = CONFIG['camera']['position_y']
    cam_z = CONFIG['camera']['height_mm']
    focus_y = CONFIG['scene']['focus_y']
    focus_z = CONFIG['scene']['focus_z']
    half_baseline = CONFIG['camera']['baseline_mm'] / 2.0
    
    light_y = cam_y
    light_z = CONFIG['light']['height_mm']
    light_size = CONFIG['light']['size_mm']
    
    # 使用 path integrator (不是 stokes)
    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': CONFIG['render']['max_depth'],
        },
    }
    
    # --------------------------------------------------
    # 載入物件
    # --------------------------------------------------
    for name, data in objects.items():
        if not data['verts'] or not data['faces']:
            continue
        
        verts = data['verts']
        faces = data['faces']
        
        max_idx = max(max(f) for f in faces) if faces else 0
        if max_idx >= len(verts):
            continue
        
        safe_name = name.replace(' ', '_').replace('.', '_').replace('-', '_')
        ply_path = str(Path(temp_dir) / f"{safe_name}.ply")
        write_ply_binary(ply_path, verts, faces)
        
        name_lower = name.lower()
        if 'glass' in name_lower or 'window' in name_lower:
            bsdf = {
                'type': 'thindielectric',
                'int_ior': 1.5,
                'ext_ior': 1.0,
            }
            print(f"    玻璃: {name}")
        else:
            bsdf = {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.7, 0.7, 0.7]}
            }
        
        scene[f'mesh_{safe_name}'] = {
            'type': 'ply',
            'filename': ply_path,
            'bsdf': bsdf,
        }
    
    # --------------------------------------------------
    # 光源 (帶 0° 偏振片)
    # --------------------------------------------------
    scene['light_emitter'] = {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, light_y, light_z],
            target=[0, focus_y, focus_z],
            up=[0, 0, 1]
        ) @ mi.ScalarTransform4f.scale([light_size, light_size, 1]),
        'emitter': {
            'type': 'area',
            'radiance': {'type': 'spectrum', 'value': CONFIG['light']['intensity']}
        }
    }
    
    # 光源偏振片 (0° 水平偏振)
    scene['light_polarizer'] = {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, light_y - 5, light_z - 5],
            target=[0, focus_y, focus_z],
            up=[0, 0, 1]
        ) @ mi.ScalarTransform4f.scale([light_size * 1.2, light_size * 1.2, 1]),
        'bsdf': {
            'type': 'polarizer',
            'theta': 0.0,
        }
    }
    
    # --------------------------------------------------
    # 相機偏振片
    # --------------------------------------------------
    # 在相機前方放置偏振片
    analyzer_distance = 10  # 相機前 10mm
    
    # 計算偏振片位置 (沿相機視線方向)
    cam_origin = np.array([-half_baseline, cam_y, cam_z])
    cam_target = np.array([0, focus_y, focus_z])
    cam_dir = cam_target - cam_origin
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    
    analyzer_pos = cam_origin + cam_dir * analyzer_distance
    
    scene['camera_polarizer'] = {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=analyzer_pos.tolist(),
            target=cam_target.tolist(),
            up=[0, 0, 1]
        ) @ mi.ScalarTransform4f.scale([100, 100, 1]),  # 大到覆蓋視野
        'bsdf': {
            'type': 'polarizer',
            'theta': float(analyzer_angle),
        }
    }
    
    # 環境光
    scene['envlight'] = {
        'type': 'constant',
        'radiance': {'type': 'spectrum', 'value': 50.0}
    }
    
    # --------------------------------------------------
    # 相機
    # --------------------------------------------------
    scene['sensor'] = {
        'type': 'perspective',
        'fov': CONFIG['camera']['fov'],
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[-half_baseline, cam_y, cam_z],
            target=[0, focus_y, focus_z],
            up=[0, 0, 1]
        ),
        'film': {
            'type': 'hdrfilm',
            'width': CONFIG['camera']['resolution'][0],
            'height': CONFIG['camera']['resolution'][1],
            'pixel_format': 'luminance',
            'component_format': 'float32',
        },
        'sampler': {
            'type': 'independent',
            'sample_count': CONFIG['render']['samples'],
        },
    }
    
    return scene


def build_depth_scene(mi, obj_path, temp_dir):
    """深度場景"""
    
    objects = parse_obj_file(obj_path)
    
    cam_y = CONFIG['camera']['position_y']
    cam_z = CONFIG['camera']['height_mm']
    focus_y = CONFIG['scene']['focus_y']
    focus_z = CONFIG['scene']['focus_z']
    half_baseline = CONFIG['camera']['baseline_mm'] / 2.0
    
    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'aov',
            'aovs': 'dd:depth',
            'nested': {'type': 'path', 'max_depth': 2}
        },
        'sensor': {
            'type': 'perspective',
            'fov': CONFIG['camera']['fov'],
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[-half_baseline, cam_y, cam_z],
                target=[0, focus_y, focus_z],
                up=[0, 0, 1]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': CONFIG['camera']['resolution'][0],
                'height': CONFIG['camera']['resolution'][1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
            },
            'sampler': {'type': 'independent', 'sample_count': 64},
        },
        'light': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': [1, 1, 1]}
        },
    }
    
    exclude = ['wall', 'ceiling']
    
    for name, data in objects.items():
        if not data['verts'] or not data['faces']:
            continue
        
        name_lower = name.lower()
        if any(kw in name_lower for kw in exclude):
            continue
        
        verts = data['verts']
        faces = data['faces']
        
        max_idx = max(max(f) for f in faces) if faces else 0
        if max_idx >= len(verts):
            continue
        
        safe_name = name.replace(' ', '_').replace('.', '_').replace('-', '_')
        ply_path = str(Path(temp_dir) / f"depth_{safe_name}.ply")
        write_ply_binary(ply_path, verts, faces)
        
        scene[f'mesh_{safe_name}'] = {
            'type': 'ply',
            'filename': ply_path,
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}}
        }
    
    return scene


# ============================================================
# 主渲染
# ============================================================
def render_scene(obj_path, output_dir):
    """渲染場景"""
    
    obj_path = Path(obj_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_dir / '_temp'
    temp_dir.mkdir(exist_ok=True)
    
    name = obj_path.stem
    
    print(f"\n{'='*60}")
    print(f"渲染: {name}")
    print(f"{'='*60}")
    
    mi = init_mitsuba()
    
    # ==========================================
    # 渲染 I∥ (analyzer = 0°)
    # ==========================================
    print("\n[渲染 I∥]")
    scene_parallel = build_scene(mi, str(obj_path), str(temp_dir), analyzer_angle=0)
    mi_scene_par = mi.load_dict(scene_parallel)
    
    print("  渲染中...")
    img_parallel = mi.render(mi_scene_par)
    I_parallel = np.array(img_parallel, dtype=np.float32)
    
    if I_parallel.ndim == 3:
        I_parallel = I_parallel[:, :, 0]
    
    print(f"  I∥: [{I_parallel.min():.1f}, {I_parallel.max():.1f}]")
    
    # ==========================================
    # 渲染 I⊥ (analyzer = 90°)
    # ==========================================
    print("\n[渲染 I⊥]")
    scene_cross = build_scene(mi, str(obj_path), str(temp_dir), analyzer_angle=90)
    mi_scene_cross = mi.load_dict(scene_cross)
    
    print("  渲染中...")
    img_cross = mi.render(mi_scene_cross)
    I_cross = np.array(img_cross, dtype=np.float32)
    
    if I_cross.ndim == 3:
        I_cross = I_cross[:, :, 0]
    
    print(f"  I⊥: [{I_cross.min():.1f}, {I_cross.max():.1f}]")
    
    # ==========================================
    # 儲存
    # ==========================================
    cv2.imwrite(str(output_dir / f"{name}_I_parallel.exr"), I_parallel)
    cv2.imwrite(str(output_dir / f"{name}_I_cross.exr"), I_cross)
    
    # PNG 預覽
    all_vals = np.concatenate([I_parallel.flatten(), I_cross.flatten()])
    valid = all_vals[all_vals > 0]
    p99 = np.percentile(valid, 99) if len(valid) > 0 else 1.0
    
    I_par_norm = np.clip(I_parallel / max(p99, 1e-6), 0, 1)
    I_cross_norm = np.clip(I_cross / max(p99, 1e-6), 0, 1)
    
    cv2.imwrite(str(output_dir / f"{name}_I_parallel.png"), 
                (np.power(I_par_norm, 1/2.2) * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / f"{name}_I_cross.png"), 
                (np.power(I_cross_norm, 1/2.2) * 255).astype(np.uint8))
    
    print("\n已儲存偏振圖")
    
    # ==========================================
    # 深度
    # ==========================================
    print("\n[渲染深度]")
    mi_rgb = init_mitsuba_rgb()
    
    depth_scene = build_depth_scene(mi_rgb, str(obj_path), str(temp_dir))
    mi_depth = mi_rgb.load_dict(depth_scene)
    
    depth_img = mi_rgb.render(mi_depth)
    depth_data = np.array(depth_img)
    
    if depth_data.ndim == 3 and depth_data.shape[2] >= 4:
        depth = depth_data[:, :, 3]
    elif depth_data.ndim == 3:
        depth = depth_data[:, :, 0]
    else:
        depth = depth_data
    
    cv2.imwrite(str(output_dir / f"{name}_depth.exr"), depth.astype(np.float32))
    
    valid_d = depth[depth > 0]
    if len(valid_d) > 0:
        d_min, d_max = np.percentile(valid_d, [1, 99])
        depth_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-6), 0, 1)
        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(output_dir / f"{name}_depth.png"), depth_color)
    
    print("已儲存深度圖")
    print(f"\n完成: {output_dir}")


def batch_render(scene_dir, output_dir, max_scenes=None):
    """批次"""
    scene_dir = Path(scene_dir)
    obj_files = sorted(scene_dir.glob("*.obj"))
    
    if max_scenes:
        obj_files = obj_files[:max_scenes]
    
    print(f"找到 {len(obj_files)} 個場景")
    
    for i, obj_path in enumerate(obj_files):
        print(f"\n[{i+1}/{len(obj_files)}]")
        try:
            render_scene(obj_path, output_dir)
        except Exception as e:
            print(f"錯誤: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIDS Renderer v8")
    parser.add_argument("input", help="OBJ 或目錄")
    parser.add_argument("-o", "--output", default="./output")
    parser.add_argument("-n", "--max", type=int)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("-s", "--samples", type=int)
    
    args = parser.parse_args()
    
    if args.samples:
        CONFIG['render']['samples'] = args.samples
    
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        batch_render(input_path, args.output, args.max)
    elif input_path.is_file():
        render_scene(input_path, args.output)
    else:
        print(f"錯誤: {input_path} 不存在")
        sys.exit(1)

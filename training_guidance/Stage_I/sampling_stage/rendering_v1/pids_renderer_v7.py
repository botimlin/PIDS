#!/usr/bin/env python3
"""
PIDS 渲染器 v7 - 完全重寫
========================

基於 blender_furniture_randomizer_v17.py 座標系統：
- OBJ 座標：前牆 Y=-350, 傢俱 Y=-750, 後牆 Y=-900
- 相機在 Y=-400 (chamber 內), Z=20 (貼地)
- 光源在相機上方

輸出:
- I_parallel.exr : 灰階 I∥ (32-bit HDR)
- I_cross.exr    : 灰階 I⊥ (32-bit HDR)
- depth.exr      : 深度 GT (mm, 32-bit float)
- *.png          : 預覽圖

使用:
  python pids_renderer_v7.py scene.obj -o ./output
  python pids_renderer_v7.py ./scenes/ -o ./output --batch
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
        'height_mm': 20.0,      # 貼地
        'position_y': -400.0,   # chamber 內
    },
    'scene': {
        'focus_y': -750.0,      # 傢俱區
        'focus_z': 20.0,        # 與相機同高
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
    """
    解析 OBJ 檔案，按物件分組
    回傳: {物件名: {'verts': [[x,y,z]...], 'faces': [[i,j,k]...]}}
    """
    objects = {}
    current_name = 'default'
    all_verts = []  # 全域頂點列表
    
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
                    objects[current_name] = {
                        'verts': [],
                        'faces': [],
                        'vert_start': 0
                    }
                
                # 解析面索引
                indices = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0])
                    # 轉換為物件內的局部索引
                    local_idx = idx - 1 - objects[current_name]['vert_start']
                    indices.append(local_idx)
                
                # 三角化
                if len(indices) >= 3:
                    objects[current_name]['faces'].append(indices[:3])
                if len(indices) == 4:
                    objects[current_name]['faces'].append([indices[0], indices[2], indices[3]])
    
    # 統計
    for name, data in objects.items():
        nv, nf = len(data['verts']), len(data['faces'])
        if nv > 0:
            verts = np.array(data['verts'])
            print(f"      {name}: {nv} verts, {nf} faces, "
                  f"Y=[{verts[:,1].min():.0f}, {verts[:,1].max():.0f}]")
    
    return objects


def write_ply_binary(filepath, vertices, faces):
    """寫入二進位 PLY 檔案"""
    import struct
    
    with open(filepath, 'wb') as f:
        # Header
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
        
        # Vertices
        for v in vertices:
            f.write(struct.pack('<fff', float(v[0]), float(v[1]), float(v[2])))
        
        # Faces
        for face in faces:
            f.write(struct.pack('<B', 3))
            f.write(struct.pack('<iii', int(face[0]), int(face[1]), int(face[2])))


# ============================================================
# Mitsuba 初始化
# ============================================================
def init_mitsuba_polarized():
    """初始化 Mitsuba 偏振模式"""
    import mitsuba as mi
    
    # 嘗試 GPU 偏振模式
    try:
        mi.set_variant('cuda_ad_spectral_polarized')
        print(f"  Mitsuba: {mi.variant()} (GPU)")
    except:
        try:
            mi.set_variant('scalar_spectral_polarized')
            print(f"  Mitsuba: {mi.variant()} (CPU)")
        except Exception as e:
            print(f"  錯誤: 無法啟用偏振模式")
            print(f"  可用變體: {mi.variants()}")
            raise
    
    return mi


def init_mitsuba_rgb():
    """初始化 Mitsuba RGB 模式 (深度用)"""
    import mitsuba as mi
    
    try:
        mi.set_variant('cuda_ad_rgb')
    except:
        mi.set_variant('scalar_rgb')
    
    return mi


# ============================================================
# 場景建立
# ============================================================
def build_polarized_scene(mi, obj_path, temp_dir):
    """
    建立偏振渲染場景
    
    關鍵：使用 stokes integrator 才能輸出 Stokes 向量
    """
    
    objects = parse_obj_file(obj_path)
    
    # 相機與場景參數
    cam_y = CONFIG['camera']['position_y']
    cam_z = CONFIG['camera']['height_mm']
    focus_y = CONFIG['scene']['focus_y']
    focus_z = CONFIG['scene']['focus_z']
    half_baseline = CONFIG['camera']['baseline_mm'] / 2.0
    
    # 場景字典
    scene = {
        'type': 'scene',
        
        # 關鍵：使用 stokes integrator
        'integrator': {
            'type': 'stokes',
            'nested': {
                'type': 'path',
                'max_depth': CONFIG['render']['max_depth'],
            }
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
        
        # 檢查索引
        max_idx = max(max(f) for f in faces) if faces else 0
        if max_idx >= len(verts):
            print(f"    跳過 {name}: 索引超出範圍")
            continue
        
        # 寫入 PLY
        safe_name = name.replace(' ', '_').replace('.', '_').replace('-', '_')
        ply_path = str(Path(temp_dir) / f"pol_{safe_name}.ply")
        write_ply_binary(ply_path, verts, faces)
        
        # 材質
        name_lower = name.lower()
        if 'glass' in name_lower or 'window' in name_lower:
            # 玻璃：薄介電質
            bsdf = {
                'type': 'thindielectric',
                'int_ior': 1.5,
                'ext_ior': 1.0,
            }
            print(f"    玻璃: {name}")
        else:
            # 其他：漫反射
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
    # 光源 (帶偏振片)
    # --------------------------------------------------
    light_y = cam_y
    light_z = CONFIG['light']['height_mm']
    light_size = CONFIG['light']['size_mm']
    
    # 光源本體
    scene['light_source'] = {
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
    
    # 光源偏振片 (水平偏振 θ=0)
    scene['light_polarizer'] = {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, light_y - 5, light_z - 5],
            target=[0, focus_y, focus_z],
            up=[0, 0, 1]
        ) @ mi.ScalarTransform4f.scale([light_size * 1.1, light_size * 1.1, 1]),
        'bsdf': {
            'type': 'polarizer',
            'theta': 0.0,
        }
    }
    
    # 環境光 (填補暗區)
    scene['envlight'] = {
        'type': 'constant',
        'radiance': {'type': 'spectrum', 'value': 100.0}
    }
    
    # --------------------------------------------------
    # 相機
    # --------------------------------------------------
    film = {
        'type': 'hdrfilm',
        'width': CONFIG['camera']['resolution'][0],
        'height': CONFIG['camera']['resolution'][1],
        'pixel_format': 'luminance',
        'component_format': 'float32',
    }
    
    sampler = {
        'type': 'independent',
        'sample_count': CONFIG['render']['samples'],
    }
    
    # 左眼
    scene['sensor_left'] = {
        'type': 'perspective',
        'fov': CONFIG['camera']['fov'],
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[-half_baseline, cam_y, cam_z],
            target=[0, focus_y, focus_z],
            up=[0, 0, 1]
        ),
        'film': film.copy(),
        'sampler': sampler.copy(),
    }
    
    # 右眼
    scene['sensor_right'] = {
        'type': 'perspective',
        'fov': CONFIG['camera']['fov'],
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[half_baseline, cam_y, cam_z],
            target=[0, focus_y, focus_z],
            up=[0, 0, 1]
        ),
        'film': film.copy(),
        'sampler': sampler.copy(),
    }
    
    return scene


def build_depth_scene(mi, obj_path, temp_dir):
    """建立深度渲染場景"""
    
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
    
    # 排除牆壁和天花板
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
        
        # 深度用不透明材質
        scene[f'mesh_{safe_name}'] = {
            'type': 'ply',
            'filename': ply_path,
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}}
        }
        print(f"    深度物件: {name}")
    
    return scene


# ============================================================
# Stokes 處理
# ============================================================
def process_stokes(img, analyzer_angle):
    """
    處理 Stokes 向量，計算通過偏振片後的強度
    
    luminance + spectral_polarized 模式輸出 4 通道: S0, S1, S2, S3
    
    I(θ) = 0.5 * (S0 + S1*cos(2θ) + S2*sin(2θ))
    """
    data = np.array(img, dtype=np.float32)
    
    print(f"    數據 shape: {data.shape}")
    
    if data.ndim == 2:
        print("    警告: 2D 數據，無偏振")
        return data
    
    c = data.shape[2]
    
    # 打印通道資訊
    for i in range(min(c, 6)):
        ch = data[:, :, i]
        print(f"      ch{i}: [{ch.min():.1f}, {ch.max():.1f}], mean={ch.mean():.1f}")
    
    # 解析 Stokes
    if c >= 4:
        S0 = data[:, :, 0]
        S1 = data[:, :, 1]
        S2 = data[:, :, 2]
    else:
        print(f"    警告: 只有 {c} 通道")
        return data[:, :, 0]
    
    # 計算偏振後強度
    theta = np.radians(analyzer_angle)
    I_out = 0.5 * (S0 + S1 * np.cos(2*theta) + S2 * np.sin(2*theta))
    
    return np.maximum(I_out, 0)


# ============================================================
# 主渲染函數
# ============================================================
def render_scene(obj_path, output_dir):
    """渲染單一場景"""
    
    obj_path = Path(obj_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_dir / '_temp'
    temp_dir.mkdir(exist_ok=True)
    
    name = obj_path.stem
    
    print(f"\n{'='*60}")
    print(f"渲染: {name}")
    print(f"{'='*60}")
    
    # ==========================================
    # 1. 偏振渲染
    # ==========================================
    print("\n[偏振渲染]")
    mi = init_mitsuba_polarized()
    
    scene_dict = build_polarized_scene(mi, str(obj_path), str(temp_dir))
    scene = mi.load_dict(scene_dict)
    
    print("  渲染左眼...")
    img_left = mi.render(scene, sensor=0)
    
    print("  渲染右眼...")
    img_right = mi.render(scene, sensor=1)
    
    # 處理 Stokes
    print("  處理 Stokes (左眼)...")
    I_parallel = process_stokes(img_left, 0)   # 0° = I∥
    
    print("  處理 Stokes (右眼)...")
    I_cross = process_stokes(img_right, 90)    # 90° = I⊥
    
    print(f"  I∥: [{I_parallel.min():.1f}, {I_parallel.max():.1f}]")
    print(f"  I⊥: [{I_cross.min():.1f}, {I_cross.max():.1f}]")
    
    # 儲存 EXR
    cv2.imwrite(str(output_dir / f"{name}_I_parallel.exr"), I_parallel.astype(np.float32))
    cv2.imwrite(str(output_dir / f"{name}_I_cross.exr"), I_cross.astype(np.float32))
    
    # 儲存 PNG
    all_vals = np.concatenate([I_parallel.flatten(), I_cross.flatten()])
    valid = all_vals[all_vals > 0]
    p99 = np.percentile(valid, 99) if len(valid) > 0 else 1.0
    
    I_par_norm = np.clip(I_parallel / max(p99, 1e-6), 0, 1)
    I_cross_norm = np.clip(I_cross / max(p99, 1e-6), 0, 1)
    
    cv2.imwrite(str(output_dir / f"{name}_I_parallel.png"), 
                (np.power(I_par_norm, 1/2.2) * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / f"{name}_I_cross.png"), 
                (np.power(I_cross_norm, 1/2.2) * 255).astype(np.uint8))
    
    print("  已儲存偏振圖")
    
    # ==========================================
    # 2. 深度渲染
    # ==========================================
    print("\n[深度渲染]")
    mi_rgb = init_mitsuba_rgb()
    
    depth_scene = build_depth_scene(mi_rgb, str(obj_path), str(temp_dir))
    depth_mi = mi_rgb.load_dict(depth_scene)
    
    print("  渲染深度...")
    depth_img = mi_rgb.render(depth_mi)
    depth_data = np.array(depth_img)
    
    # 提取深度
    if depth_data.ndim == 3 and depth_data.shape[2] >= 4:
        depth = depth_data[:, :, 3]
    elif depth_data.ndim == 3:
        depth = depth_data[:, :, 0]
    else:
        depth = depth_data
    
    # 儲存深度
    cv2.imwrite(str(output_dir / f"{name}_depth.exr"), depth.astype(np.float32))
    
    # 深度可視化
    valid_d = depth[depth > 0]
    if len(valid_d) > 0:
        d_min, d_max = np.percentile(valid_d, [1, 99])
        depth_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-6), 0, 1)
        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(output_dir / f"{name}_depth.png"), depth_color)
    
    print("  已儲存深度圖")
    print(f"\n完成: {output_dir}")


def batch_render(scene_dir, output_dir, max_scenes=None):
    """批次渲染"""
    
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


# ============================================================
# 主程式
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIDS Renderer v7")
    parser.add_argument("input", help="OBJ 檔案或目錄")
    parser.add_argument("-o", "--output", default="./output", help="輸出目錄")
    parser.add_argument("-n", "--max", type=int, help="最大場景數")
    parser.add_argument("--batch", action="store_true", help="批次模式")
    parser.add_argument("-s", "--samples", type=int, help="採樣數")
    
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

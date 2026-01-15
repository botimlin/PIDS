#!/usr/bin/env python3
"""
PIDS 渲染器 v10 - 最終版
========================

根據論文和 Blender 腳本的精確配置
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
    print("pip install opencv-python")
    sys.exit(1)


# ============================================================
# 配置
# ============================================================
CONFIG = {
    'camera': {
        'resolution': (640, 480),
        'fov': 55.0,
        'baseline_mm': 65.0,
    },
    'render': {
        'samples': 4096,
        'max_depth': 8,
    },
}


# ============================================================
# OBJ 解析
# ============================================================
def parse_obj(filepath):
    """解析 OBJ"""
    objects = {}
    current = 'default'
    global_verts = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if not parts:
                continue
            
            if parts[0] == 'o' and len(parts) > 1:
                current = parts[1]
                if current not in objects:
                    objects[current] = {'verts': [], 'faces': [], 'offset': len(global_verts)}
                    
            elif parts[0] == 'v' and len(parts) >= 4:
                v = [float(parts[1]), float(parts[2]), float(parts[3])]
                global_verts.append(v)
                if current in objects:
                    objects[current]['verts'].append(v)
                    
            elif parts[0] == 'f':
                if current not in objects:
                    objects[current] = {'verts': [], 'faces': [], 'offset': 0}
                face = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0]) - 1
                    local = idx - objects[current]['offset']
                    face.append(local)
                if len(face) >= 3:
                    objects[current]['faces'].append(face[:3])
                if len(face) == 4:
                    objects[current]['faces'].append([face[0], face[2], face[3]])
    
    return objects


def write_ply(path, verts, faces):
    """寫入 PLY"""
    import struct
    with open(path, 'wb') as f:
        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
        f.write(header.encode())
        for v in verts:
            f.write(struct.pack('<fff', v[0], v[1], v[2]))
        for face in faces:
            f.write(struct.pack('<Biii', 3, face[0], face[1], face[2]))


# ============================================================
# Mitsuba
# ============================================================
def get_mitsuba():
    import mitsuba as mi
    try:
        mi.set_variant('cuda_ad_spectral_polarized')
        print(f"  Mitsuba: {mi.variant()} (GPU)")
    except:
        mi.set_variant('scalar_spectral_polarized')
        print(f"  Mitsuba: {mi.variant()} (CPU)")
    return mi


def get_mitsuba_rgb():
    import mitsuba as mi
    try:
        mi.set_variant('cuda_ad_rgb')
    except:
        mi.set_variant('scalar_rgb')
    return mi


# ============================================================
# 場景建立
# ============================================================
def create_scene(mi, obj_path, temp_dir, camera_side):
    """
    建立場景
    
    OBJ 座標系:
    - 相機在 Y=0 看向 -Y
    - 前牆 Y=-350
    - 玻璃 Y=-550 ~ -700
    - 傢俱 Y=-750
    - 後牆 Y=-900
    - 地板 Z=0, 天花板 Z=300
    """
    
    print(f"  解析: {obj_path}")
    objects = parse_obj(obj_path)
    
    # 列出物件
    for name, data in objects.items():
        if data['verts']:
            v = np.array(data['verts'])
            print(f"    {name}: Y=[{v[:,1].min():.0f}, {v[:,1].max():.0f}], Z=[{v[:,2].min():.0f}, {v[:,2].max():.0f}]")
    
    # 相機參數
    baseline = CONFIG['camera']['baseline_mm']
    cam_y = 0      # 相機在原點
    cam_z = 150    # 中間高度
    target_y = -625  # 看向場景中心
    
    if camera_side == 'left':
        cam_x = -baseline / 2.0
    else:
        cam_x = baseline / 2.0
    
    # 光源位置 (在 chamber 內，前牆和玻璃之間)
    light_y = -400   # 前牆(-350)和玻璃(-550)之間
    light_z = 280    # 接近天花板
    light_size = 200.0
    
    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'stokes',
            'nested': {
                'type': 'path',
                'max_depth': CONFIG['render']['max_depth'],
            }
        },
    }
    
    # 載入物件
    for name, data in objects.items():
        if not data['verts'] or not data['faces']:
            continue
        
        verts, faces = data['verts'], data['faces']
        if max(max(f) for f in faces) >= len(verts):
            continue
        
        safe = name.replace(' ', '_').replace('.', '_').replace('-', '_')
        ply = str(Path(temp_dir) / f"{safe}.ply")
        write_ply(ply, verts, faces)
        
        if 'glass' in name.lower():
            bsdf = {'type': 'thindielectric', 'int_ior': 1.5, 'ext_ior': 1.0}
        else:
            bsdf = {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.7, 0.7, 0.7]}}
        
        scene[f'm_{safe}'] = {'type': 'ply', 'filename': ply, 'bsdf': bsdf}
    
    # 光源 (使用偏振發光)
    # 在 Mitsuba 3 中，要產生偏振光，需要用特殊設定
    # 這裡先用普通光源，讓場景正常渲染
    scene['light'] = {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, light_y, light_z],
            target=[0, -625, 40],  # 照向玻璃區
            up=[0, 0, 1]
        ) @ mi.ScalarTransform4f.scale([light_size, light_size, 1]),
        'emitter': {
            'type': 'area',
            'radiance': {'type': 'spectrum', 'value': 80000.0}
        }
    }
    
    # 環境光
    scene['env'] = {
        'type': 'constant',
        'radiance': {'type': 'spectrum', 'value': 500.0}
    }
    
    # 相機
    # 相機在 Y=0，Z=150，要看到整個場景
    # target 應該在場景中心：Y=-625 (玻璃和傢俱中間), Z=75 (場景高度中間)
    scene['sensor'] = {
        'type': 'perspective',
        'fov': CONFIG['camera']['fov'],
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[cam_x, cam_y, cam_z],
            target=[0, -625, 75],  # 場景中心，稍微往下看
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


def create_depth_scene(mi, obj_path, temp_dir):
    """深度場景"""
    
    objects = parse_obj(obj_path)
    
    baseline = CONFIG['camera']['baseline_mm']
    cam_y = 0
    cam_z = 150
    target_y = -625
    
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
                origin=[-baseline/2, cam_y, cam_z],
                target=[0, -625, 75],  # 與偏振相機相同
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
        'light': {'type': 'constant', 'radiance': {'type': 'rgb', 'value': [1,1,1]}},
    }
    
    exclude = ['wall', 'ceiling']
    
    for name, data in objects.items():
        if not data['verts'] or not data['faces']:
            continue
        if any(k in name.lower() for k in exclude):
            continue
        
        verts, faces = data['verts'], data['faces']
        if max(max(f) for f in faces) >= len(verts):
            continue
        
        safe = name.replace(' ', '_').replace('.', '_').replace('-', '_')
        ply = str(Path(temp_dir) / f"d_{safe}.ply")
        write_ply(ply, verts, faces)
        
        scene[f'm_{safe}'] = {
            'type': 'ply',
            'filename': ply,
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.8,0.8,0.8]}}
        }
    
    return scene


# ============================================================
# 主渲染
# ============================================================
def render(obj_path, output_dir):
    """渲染"""
    
    obj_path = Path(obj_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp = output_dir / '_temp'
    temp.mkdir(exist_ok=True)
    
    name = obj_path.stem
    
    print(f"\n{'='*50}")
    print(f"PIDS: {name}")
    print(f"{'='*50}")
    
    mi = get_mitsuba()
    
    # 渲染左相機
    print("\n[左相機]")
    scene_left = create_scene(mi, str(obj_path), str(temp), 'left')
    s_left = mi.load_dict(scene_left)
    img_left = mi.render(s_left)
    data_left = np.array(img_left, dtype=np.float32)
    
    print(f"  shape: {data_left.shape}")
    if data_left.ndim == 3:
        for i in range(min(4, data_left.shape[2])):
            print(f"    ch{i}: [{data_left[:,:,i].min():.1f}, {data_left[:,:,i].max():.1f}]")
    
    # 渲染右相機
    print("\n[右相機]")
    scene_right = create_scene(mi, str(obj_path), str(temp), 'right')
    s_right = mi.load_dict(scene_right)
    img_right = mi.render(s_right)
    data_right = np.array(img_right, dtype=np.float32)
    
    # Stokes 處理
    def stokes_to_I(data, angle):
        if data.ndim == 2:
            return data
        if data.shape[2] >= 4:
            S0, S1, S2 = data[:,:,0], data[:,:,1], data[:,:,2]
        else:
            return data[:,:,0]
        theta = np.radians(angle)
        return np.maximum(0.5 * (S0 + S1*np.cos(2*theta) + S2*np.sin(2*theta)), 0)
    
    I_par = stokes_to_I(data_left, 0)
    I_cross = stokes_to_I(data_right, 90)
    
    print(f"\n  I∥: [{I_par.min():.1f}, {I_par.max():.1f}]")
    print(f"  I⊥: [{I_cross.min():.1f}, {I_cross.max():.1f}]")
    
    # 儲存 (用 np.save 避免 OpenCV EXR 問題)
    np.save(str(output_dir / f"{name}_I_parallel.npy"), I_par)
    np.save(str(output_dir / f"{name}_I_cross.npy"), I_cross)
    
    # PNG
    vals = np.concatenate([I_par.flatten(), I_cross.flatten()])
    valid = vals[vals > 0]
    p99 = np.percentile(valid, 99) if len(valid) > 0 else 1.0
    
    cv2.imwrite(str(output_dir / f"{name}_I_parallel.png"),
                (np.power(np.clip(I_par/p99, 0, 1), 1/2.2) * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / f"{name}_I_cross.png"),
                (np.power(np.clip(I_cross/p99, 0, 1), 1/2.2) * 255).astype(np.uint8))
    
    print("\n已儲存偏振圖")
    
    # 深度
    print("\n[深度]")
    mi_d = get_mitsuba_rgb()
    scene_d = create_depth_scene(mi_d, str(obj_path), str(temp))
    s_d = mi_d.load_dict(scene_d)
    img_d = mi_d.render(s_d)
    depth = np.array(img_d)
    if depth.ndim == 3:
        depth = depth[:,:,3] if depth.shape[2] > 3 else depth[:,:,0]
    
    np.save(str(output_dir / f"{name}_depth.npy"), depth)
    
    vd = depth[depth > 0]
    if len(vd) > 0:
        d1, d99 = np.percentile(vd, [1, 99])
        dn = np.clip((depth - d1) / (d99 - d1 + 1e-6), 0, 1)
        cv2.imwrite(str(output_dir / f"{name}_depth.png"),
                    cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    
    print("已儲存深度圖")
    print(f"\n完成: {output_dir}")


def batch(scene_dir, output_dir, n=None):
    files = sorted(Path(scene_dir).glob("*.obj"))
    if n:
        files = files[:n]
    print(f"找到 {len(files)} 個場景")
    for i, f in enumerate(files):
        print(f"\n[{i+1}/{len(files)}]")
        try:
            render(f, output_dir)
        except Exception as e:
            print(f"錯誤: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", default="./output")
    p.add_argument("-n", type=int)
    p.add_argument("--batch", action="store_true")
    p.add_argument("-s", "--samples", type=int)
    args = p.parse_args()
    
    if args.samples:
        CONFIG['render']['samples'] = args.samples
    
    inp = Path(args.input)
    if args.batch or inp.is_dir():
        batch(inp, args.output, args.n)
    else:
        render(inp, args.output)

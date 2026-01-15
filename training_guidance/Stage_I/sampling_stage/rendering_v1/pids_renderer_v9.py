#!/usr/bin/env python3
"""
PIDS 渲染器 - 從零重寫
======================

完全按照論文和 blender_furniture_randomizer_v17.py 座標系統

論文 PIDS 系統 (Figure 1):
- 光源有線性偏振片 (0°)
- Camera A (I∥): 偏振片 0° (平行，保留高光)
- Camera B (I⊥): 偏振片 90° (垂直，抑制高光)
- 光源傾斜約 35° (接近 Brewster 角 ~56°)

Blender 座標:
- 相機在原點 (0, 0, 0)，朝向 +Y
- 前牆 Y = 350mm
- 玻璃 Y = 528~695mm  
- 傢俱 Y = 750mm
- 後牆 Y ≈ 900mm
- Chamber 高度 = 300mm

OBJ 匯出 (forward_axis='NEGATIVE_Y'):
- OBJ Y = -Blender Y
- 所以相機在原點看向 -Y 方向
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
# 配置 (基於論文和 Blender 腳本)
# ============================================================
CONFIG = {
    # 相機參數
    'camera': {
        'resolution': (640, 480),
        'fov': 55.0,              # 度
        'baseline_mm': 65.0,      # 雙目基線
    },
    
    # 場景座標 (OBJ 座標)
    # 相機在原點看向 -Y
    # 前牆 Y=-350, 玻璃 Y=-550~-700, 傢俱 Y=-750, 後牆 Y=-900
    'scene': {
        'camera_y': 0,            # 相機在原點
        'camera_z': 150,          # 相機高度 (chamber 一半高度)
        'target_y': -625,         # 看向場景中心 (玻璃和傢俱之間)
    },
    
    # 光源參數 (論文: 傾斜 35° 接近 Brewster 角)
    'light': {
        'angle_deg': 35.0,        # 光源傾斜角度
        'size_mm': 150.0,         # 光源尺寸
        'intensity': 50000.0,     # 光源強度
        'polarizer_angle': 0.0,   # 光源偏振片角度
    },
    
    # 渲染參數
    'render': {
        'samples': 4096,
        'max_depth': 8,
    },
}


# ============================================================
# OBJ 解析
# ============================================================
def parse_obj(filepath):
    """
    解析 OBJ 檔案
    回傳: {物件名: {'verts': [], 'faces': []}}
    """
    objects = {}
    current = 'default'
    global_verts = []
    
    print(f"  解析: {filepath}")
    
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
                    objects[current] = {
                        'verts': [],
                        'faces': [],
                        'offset': len(global_verts)
                    }
                    
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
    
    # 列印物件資訊
    for name, data in objects.items():
        if data['verts']:
            v = np.array(data['verts'])
            print(f"    {name}: Y=[{v[:,1].min():.0f}, {v[:,1].max():.0f}]")
    
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
# Mitsuba 初始化
# ============================================================
def get_mitsuba():
    """初始化 Mitsuba 偏振模式"""
    import mitsuba as mi
    
    try:
        mi.set_variant('cuda_ad_spectral_polarized')
        print(f"  Mitsuba: {mi.variant()} (GPU)")
    except:
        mi.set_variant('scalar_spectral_polarized')
        print(f"  Mitsuba: {mi.variant()} (CPU)")
    
    return mi


def get_mitsuba_depth():
    """RGB 模式 (深度用)"""
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
    建立 Mitsuba 場景 (使用 stokes integrator)
    
    camera_side: 'left' 或 'right'
    """
    
    objects = parse_obj(obj_path)
    
    # 座標
    cam_y = CONFIG['scene']['camera_y']      # 0
    cam_z = CONFIG['scene']['camera_z']      # 150
    target_y = CONFIG['scene']['target_y']   # -625
    baseline = CONFIG['camera']['baseline_mm']  # 65mm
    
    # 相機 X 位置
    if camera_side == 'left':
        cam_x = -baseline / 2.0  # -32.5mm
    else:
        cam_x = baseline / 2.0   # +32.5mm
    
    # 光源位置
    light_z = 350
    light_y = -250
    
    scene = {
        'type': 'scene',
        # 使用 stokes integrator 來追蹤偏振狀態
        'integrator': {
            'type': 'stokes',
            'nested': {
                'type': 'path',
                'max_depth': CONFIG['render']['max_depth'],
            }
        },
    }
    
    # --------------------------------------------------
    # 物件
    # --------------------------------------------------
    for name, data in objects.items():
        if not data['verts'] or not data['faces']:
            continue
        
        verts, faces = data['verts'], data['faces']
        if max(max(f) for f in faces) >= len(verts):
            continue
        
        safe = name.replace(' ', '_').replace('.', '_').replace('-', '_')
        ply = str(Path(temp_dir) / f"{safe}.ply")
        write_ply(ply, verts, faces)
        
        # 材質
        if 'glass' in name.lower():
            bsdf = {'type': 'thindielectric', 'int_ior': 1.5, 'ext_ior': 1.0}
        else:
            bsdf = {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.7, 0.7, 0.7]}}
        
        scene[f'm_{safe}'] = {'type': 'ply', 'filename': ply, 'bsdf': bsdf}
    
    # --------------------------------------------------
    # 光源 + 偏振片 (0°)
    # --------------------------------------------------
    light_size = CONFIG['light']['size_mm']
    
    scene['light'] = {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, light_y, light_z],
            target=[0, target_y, cam_z],
            up=[0, 0, 1]
        ) @ mi.ScalarTransform4f.scale([light_size, light_size, 1]),
        'bsdf': {
            'type': 'polarizer',
            'theta': CONFIG['light']['polarizer_angle'],
        },
        'emitter': {
            'type': 'area',
            'radiance': {'type': 'spectrum', 'value': CONFIG['light']['intensity']}
        }
    }
    
    # 環境光
    scene['env'] = {
        'type': 'constant',
        'radiance': {'type': 'spectrum', 'value': 100.0}
    }
    
    # --------------------------------------------------
    # 相機 (不用實體偏振片，用 stokes integrator 後處理)
    # --------------------------------------------------
    scene['sensor'] = {
        'type': 'perspective',
        'fov': CONFIG['camera']['fov'],
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[cam_x, cam_y, cam_z],
            target=[0, target_y, cam_z],
            up=[0, 0, 1]
        ),
        'film': {
            'type': 'hdrfilm',
            'width': CONFIG['camera']['resolution'][0],
            'height': CONFIG['camera']['resolution'][1],
            'pixel_format': 'luminance',  # 灰階
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
    
    cam_y = CONFIG['scene']['camera_y']      # 0
    cam_z = CONFIG['scene']['camera_z']      # 150
    target_y = CONFIG['scene']['target_y']   # -625
    baseline = CONFIG['camera']['baseline_mm']
    
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
                target=[0, target_y, cam_z],
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
    """渲染場景"""
    
    obj_path = Path(obj_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp = output_dir / '_temp'
    temp.mkdir(exist_ok=True)
    
    name = obj_path.stem
    
    print(f"\n{'='*50}")
    print(f"PIDS 渲染: {name}")
    print(f"基線: {CONFIG['camera']['baseline_mm']}mm")
    print(f"{'='*50}")
    
    mi = get_mitsuba()
    
    # 渲染左相機 (I∥)
    print("\n[左相機] 渲染中...")
    scene_left = create_scene(mi, str(obj_path), str(temp), camera_side='left')
    s_left = mi.load_dict(scene_left)
    img_left = mi.render(s_left)
    data_left = np.array(img_left, dtype=np.float32)
    
    print(f"  數據 shape: {data_left.shape}")
    if data_left.ndim == 3:
        for i in range(min(4, data_left.shape[2])):
            ch = data_left[:,:,i]
            print(f"    ch{i}: [{ch.min():.1f}, {ch.max():.1f}]")
    
    # 渲染右相機 (I⊥)
    print("\n[右相機] 渲染中...")
    scene_right = create_scene(mi, str(obj_path), str(temp), camera_side='right')
    s_right = mi.load_dict(scene_right)
    img_right = mi.render(s_right)
    data_right = np.array(img_right, dtype=np.float32)
    
    # 處理 Stokes 向量
    # luminance + stokes 輸出 4 通道: S0, S1, S2, S3
    # I(θ) = 0.5 * (S0 + S1*cos(2θ) + S2*sin(2θ))
    
    def stokes_to_intensity(data, angle_deg):
        """從 Stokes 向量計算通過偏振片後的強度"""
        if data.ndim == 2:
            return data
        
        if data.shape[2] >= 4:
            S0 = data[:,:,0]
            S1 = data[:,:,1]
            S2 = data[:,:,2]
        else:
            S0 = data[:,:,0]
            S1 = np.zeros_like(S0)
            S2 = np.zeros_like(S0)
        
        theta = np.radians(angle_deg)
        I = 0.5 * (S0 + S1 * np.cos(2*theta) + S2 * np.sin(2*theta))
        return np.maximum(I, 0)
    
    # 左相機 + 0° 偏振片 = I∥
    I_par = stokes_to_intensity(data_left, 0)
    
    # 右相機 + 90° 偏振片 = I⊥
    I_cross = stokes_to_intensity(data_right, 90)
    
    print(f"\n  I∥: [{I_par.min():.1f}, {I_par.max():.1f}]")
    print(f"  I⊥: [{I_cross.min():.1f}, {I_cross.max():.1f}]")
    
    # 儲存 EXR (灰階 HDR)
    cv2.imwrite(str(output_dir / f"{name}_I_parallel.exr"), I_par)
    cv2.imwrite(str(output_dir / f"{name}_I_cross.exr"), I_cross)
    
    # PNG 預覽 (灰階)
    vals = np.concatenate([I_par.flatten(), I_cross.flatten()])
    valid = vals[vals > 0]
    p99 = np.percentile(valid, 99) if len(valid) > 0 else 1.0
    
    cv2.imwrite(str(output_dir / f"{name}_I_parallel.png"),
                (np.power(np.clip(I_par/p99, 0, 1), 1/2.2) * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / f"{name}_I_cross.png"),
                (np.power(np.clip(I_cross/p99, 0, 1), 1/2.2) * 255).astype(np.uint8))
    
    print("\n已儲存偏振圖 (灰階)")
    
    # 深度
    print("\n[深度] 渲染中...")
    mi_d = get_mitsuba_depth()
    scene_d = create_depth_scene(mi_d, str(obj_path), str(temp))
    s_d = mi_d.load_dict(scene_d)
    img_d = mi_d.render(s_d)
    depth = np.array(img_d)
    if depth.ndim == 3:
        depth = depth[:,:,3] if depth.shape[2] > 3 else depth[:,:,0]
    
    cv2.imwrite(str(output_dir / f"{name}_depth.exr"), depth.astype(np.float32))
    
    vd = depth[depth > 0]
    if len(vd) > 0:
        d1, d99 = np.percentile(vd, [1, 99])
        dn = np.clip((depth - d1) / (d99 - d1 + 1e-6), 0, 1)
        cv2.imwrite(str(output_dir / f"{name}_depth.png"),
                    cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    
    print("已儲存深度圖")
    print(f"\n完成: {output_dir}")


def batch(scene_dir, output_dir, n=None):
    """批次"""
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

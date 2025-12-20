"""
PIDS 渲染器 v5.2 - HDR EXR + 批次處理 + OIDN 降噪
=================================================

輸出 (訓練用):
- I_parallel.exr  : 灰階 I∥ (32-bit HDR，降噪後)
- I_cross.exr     : 灰階 I⊥ (32-bit HDR，降噪後)
- depth.exr       : 深度 GT (米，32-bit float)

輸出 (預覽用):
- I_parallel.png  : 灰階 I∥ (8-bit，正規化)
- I_cross.png     : 灰階 I⊥ (8-bit，正規化)
- depth.png       : 深度視覺化 (彩色)

使用:
  # 單一檔案
  python pids_renderer_v5.py -i scene.obj -o ./output --polarized

  # 批次處理資料夾
  python pids_renderer_v5.py -i ./scenes/ -o ./output --polarized --batch

  # 指定採樣數 (預設 16384)
  python pids_renderer_v5.py -i ./scenes/ -o ./output --polarized --batch -s 16384
  
  # 關閉降噪 (預設開啟)
  python pids_renderer_v5.py -i ./scenes/ -o ./output --polarized --batch --no-denoise

安裝 OIDN:
  pip install pyoidn
  # 或
  pip install oidn
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
# OIDN 降噪器
# ============================================================

_oidn_device = None
_oidn_available = False

def init_oidn():
    """初始化 OIDN 降噪器 (包含防呆機制)"""
    global _oidn_device, _oidn_available
    
    if _oidn_device is not None:
        return _oidn_available
    
    try:
        import oidn
        device = oidn.NewDevice() 
        
        # 【修正重點】檢查回傳是否為 int (導致崩潰的原因)
        if isinstance(device, int):
            print("  警告: OIDN 版本不相容 (回傳 handle)，已自動關閉降噪。")
            _oidn_available = False
            return False
            
        device.commit()
        _oidn_device = device
        _oidn_available = True
        print("  OIDN 降噪器已啟用")
        return True
    except ImportError:
        print("  警告: 未安裝 OIDN (pip install oidn)，跳過降噪。")
        _oidn_available = False
        return False
    except Exception as e:
        print(f"  警告: OIDN 初始化失敗 ({e})，已自動關閉降噪。")
        _oidn_available = False
        return False


def denoise_image(img, is_hdr=True):
    """
    使用 OIDN 對圖像進行降噪
    
    Args:
        img: 輸入圖像 (H, W) 或 (H, W, C)，float32
        is_hdr: 是否為 HDR 圖像
    
    Returns:
        降噪後的圖像
    """
    global _oidn_device, _oidn_available
    
    if not _oidn_available:
        return img
    
    try:
        import oidn
        
        # 確保是 float32
        img = img.astype(np.float32)
        
        # OIDN 需要 3 通道輸入
        if img.ndim == 2:
            img_3ch = np.stack([img, img, img], axis=2)
        else:
            img_3ch = img
        
        h, w = img_3ch.shape[:2]
        
        # 建立降噪 filter
        filter = _oidn_device.newFilter("RT")  # Ray Tracing filter
        
        # 設定輸入輸出
        filter.setImage("color", img_3ch, oidn.Format.FLOAT3, w, h)
        output = np.zeros_like(img_3ch)
        filter.setImage("output", output, oidn.Format.FLOAT3, w, h)
        
        # HDR 模式
        if is_hdr:
            filter.set("hdr", True)
        
        filter.commit()
        filter.execute()
        
        # 檢查錯誤
        error = _oidn_device.getError()
        if error[0] != oidn.Error.NONE:
            print(f"  OIDN 警告: {error[1]}")
        
        # 返回單通道或多通道
        if img.ndim == 2:
            return output[:, :, 0]
        else:
            return output
            
    except Exception as e:
        print(f"  降噪失敗: {e}，返回原圖")
        return img


# ============================================================
# 配置
# ============================================================

CONFIG = {
    'camera': {
        'resolution': (640, 480),  # 維持原分辨率
        'fov': 55.0,               # IMX296 + 6mm 鏡頭: 2*arctan(6.3/(2*6)) ≈ 55°
        'baseline_mm': 65.0,
    },
    'render': {
        'samples': 16384,   # 偏振模式需要較高採樣數以減少噪點
        'max_depth': 16,
        'denoise': True,   # 預設開啟 OIDN 降噪
    },
    'colors': {
        'shelf':      [0.35, 0.25, 0.15],
        'cabinet':    [0.50, 0.40, 0.30],
        'table':      [0.45, 0.35, 0.25],
        'chair':      [0.40, 0.30, 0.20],
        'sofa':       [0.35, 0.30, 0.30],
        'default':    [0.50, 0.45, 0.40],
        'background': [0.85, 0.82, 0.78],
        'ground':     [0.55, 0.50, 0.45],
    },
}


# ============================================================
# Mitsuba 初始化
# ============================================================

_mi = None

def get_mitsuba(polarized=False):
    global _mi
    import mitsuba as mi
    
    # 確保 Dr.Jit 有偵測到 CUDA 裝置
    try:
        if polarized:
            # 改用 cuda_spectral_polarized
            mi.set_variant('cuda_ad_spectral_polarized')
        else:
            # 改用 cuda_rgb
            mi.set_variant('cuda_ad_rgb')
    except Exception as e:
        print(f"警告: 無法啟用 CUDA 模式，回退至 CPU 模式。錯誤: {e}")
        # 如果沒有 GPU 或驅動程式問題，回退到 scalar
        if polarized:
            mi.set_variant('scalar_spectral_polarized')
        else:
            mi.set_variant('scalar_rgb')
    
    print(f"Mitsuba variant: {mi.variant()}")
    _mi = mi
    return mi


# ============================================================
# OBJ 解析
# ============================================================

def parse_obj(filepath):
    """解析 OBJ，回傳每個物件的頂點和面"""
    
    objects = {}
    current = None
    all_verts = []
    
    print(f"  解析 OBJ: {filepath}")
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('o '):
                name = line[2:].strip()
                current = name
                objects[name] = {
                    'verts': [],
                    'faces': [],
                    'offset': len(all_verts),
                }
                print(f"    物件: {name}")
            
            elif line.startswith('v ') and not line.startswith(('vt', 'vn')):
                parts = line.split()
                v = [float(parts[1]), float(parts[2]), float(parts[3])]
                all_verts.append(v)
                if current:
                    objects[current]['verts'].append(v)
            
            elif line.startswith('f ') and current:
                parts = line.split()[1:]
                face = []
                for p in parts:
                    idx = int(p.split('/')[0]) - 1
                    local = idx - objects[current]['offset']
                    face.append(local)
                
                # 三角化
                if len(face) >= 3:
                    for i in range(1, len(face) - 1):
                        objects[current]['faces'].append([face[0], face[i], face[i+1]])
    
    # 印出統計
    for name, data in objects.items():
        print(f"      {name}: {len(data['verts'])} verts, {len(data['faces'])} faces")
        if data['verts']:
            verts = np.array(data['verts'])
            print(f"        範圍: X[{verts[:,0].min():.1f}, {verts[:,0].max():.1f}] "
                  f"Y[{verts[:,1].min():.1f}, {verts[:,1].max():.1f}] "
                  f"Z[{verts[:,2].min():.1f}, {verts[:,2].max():.1f}]")
    
    return objects


def transform_blender_to_mitsuba(verts):
    """
    Blender OBJ (forward=-Y, up=Z) -> Mitsuba (Y-up, Z-forward)
    """
    result = []
    for v in verts:
        result.append([
            v[0] / 1000.0,
            v[2] / 1000.0,
            -v[1] / 1000.0,
        ])
    return result


def write_ply(path, verts, faces):
    """寫入 PLY"""
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in verts:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


# ============================================================
# 材質
# ============================================================

def get_color(name):
    """根據名稱取得顏色"""
    name_lower = name.lower()
    for key, color in CONFIG['colors'].items():
        if key in name_lower:
            return color
    return CONFIG['colors']['default']


def is_glass(name):
    return 'glass' in name.lower()


# ============================================================
# 場景建立 - RGB 模式
# ============================================================

def build_scene_rgb(mi, obj_path, temp_dir):
    """建立 RGB 場景 (非偏振)"""
    
    objects = parse_obj(obj_path)
    
    # 計算物件的 Z 範圍 (轉換後)
    all_z = []
    for data in objects.values():
        if data['verts']:
            for v in data['verts']:
                z_mitsuba = -v[1] / 1000.0
                all_z.append(z_mitsuba)
    
    if all_z:
        z_min, z_max = min(all_z), max(all_z)
        z_center = (z_min + z_max) / 2
        print(f"  物件 Z 範圍: [{z_min:.3f}, {z_max:.3f}] m, 中心: {z_center:.3f} m")
    else:
        z_center = 0.6
        z_max = 0.8
        print(f"  警告: 沒有頂點，使用預設 Z 中心: {z_center}")
    
    res = CONFIG['camera']['resolution']
    fov = CONFIG['camera']['fov']
    samples = CONFIG['render']['samples']
    
    scene = {
        'type': 'scene',
        
        'integrator': {
            'type': 'path',
            'max_depth': CONFIG['render']['max_depth'],
        },
        
        'sensor': {
            'type': 'perspective',
            'fov': fov,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 0.05, 0],
                target=[0, 0.05, z_center],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': res[0],
                'height': res[1],
                'pixel_format': 'rgb',
            },
            'sampler': {
                'type': 'independent',
                'sample_count': samples,
            },
        },
        
        # 主光源
        'main_light': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, 0.8, z_center]) @
                mi.ScalarTransform4f.rotate([1, 0, 0], -90) @
                mi.ScalarTransform4f.scale([0.5, 0.5, 1])
            ),
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [80, 80, 80]},
            },
        },
        
        # 環境光
        'env': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': [0.3, 0.3, 0.3]},
        },
        
        # 背景牆
        'wall': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, 0.25, z_max + 0.2]) @
                mi.ScalarTransform4f.scale([2.0, 0.6, 1])
            ),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': CONFIG['colors']['background']},
            },
        },
        
        # 地面
        'floor': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, 0, z_center]) @
                mi.ScalarTransform4f.rotate([1, 0, 0], -90) @
                mi.ScalarTransform4f.scale([1.5, 1.0, 1])
            ),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': CONFIG['colors']['ground']},
            },
        },
    }
    
    # 加入物件
    for name, data in objects.items():
        if not data['verts'] or not data['faces']:
            continue
        
        verts = transform_blender_to_mitsuba(data['verts'])
        faces = data['faces']
        
        max_idx = max(max(f) for f in faces) if faces else 0
        if max_idx >= len(verts):
            print(f"    警告: {name} 面索引超出範圍")
            continue
        
        safe_name = name.replace(' ', '_').replace('.', '_').replace('-', '_')
        ply_path = os.path.join(temp_dir, f"{safe_name}.ply")
        write_ply(ply_path, verts, faces)
        
        if is_glass(name):
            bsdf = {
                'type': 'dielectric',
                'int_ior': 1.5,
                'ext_ior': 1.0,
            }
        else:
            color = get_color(name)
            bsdf = {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': color},
            }
        
        scene[f'obj_{safe_name}'] = {
            'type': 'ply',
            'filename': ply_path,
            'bsdf': bsdf,
        }
    
    return scene


# ============================================================
# 場景建立 - 偏振模式 (彩色光源版本)
# ============================================================

def build_scene_polarized_colored(mi, obj_path, temp_dir, light_color):
    """建立偏振場景，使用指定顏色的光源"""
    
    objects = parse_obj(obj_path)
    
    all_z = []
    for data in objects.values():
        if data['verts']:
            for v in data['verts']:
                all_z.append(-v[1] / 1000.0)
    
    z_center = (min(all_z) + max(all_z)) / 2 if all_z else 0.6
    z_max = max(all_z) if all_z else 0.8
    z_min = min(all_z) if all_z else 0.5
    
    res = CONFIG['camera']['resolution']
    fov = CONFIG['camera']['fov']
    samples = CONFIG['render']['samples']
    half_baseline = CONFIG['camera']['baseline_mm'] / 2000.0
    
    # 光源顏色乘以強度
    light_radiance = [c * 120.0 for c in light_color]
    
    scene = {
        'type': 'scene',
        
        'integrator': {
            'type': 'stokes',
            'nested': {
                'type': 'path',
                'max_depth': CONFIG['render']['max_depth'],
            },
        },
        
        'sensor_left': {
            'type': 'perspective',
            'fov': fov,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[-half_baseline, 0.05, 0],
                target=[-half_baseline, 0.05, z_center],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': res[0],
                'height': res[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
            },
            'sampler': {'type': 'independent', 'sample_count': samples},
        },
        
        'sensor_right': {
            'type': 'perspective',
            'fov': fov,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[half_baseline, 0.05, 0],
                target=[half_baseline, 0.05, z_center],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': res[0],
                'height': res[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
            },
            'sampler': {'type': 'independent', 'sample_count': samples},
        },
        
        # 主光源 - 使用指定顏色，相機上方，與光軸夾角 35°
        'main_light': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, 0.4, 0.1]) @  # 相機上方
                mi.ScalarTransform4f.rotate([1, 0, 0], -35) @
                mi.ScalarTransform4f.scale([0.4, 0.3, 1])
            ),
            'bsdf': {'type': 'null'},
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': light_radiance},
            },
        },
        
        # 偏振片，與光源同角度
        'source_polarizer': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, 0.38, 0.12]) @  # 緊貼光源前方
                mi.ScalarTransform4f.rotate([1, 0, 0], -35) @
                mi.ScalarTransform4f.scale([0.42, 0.32, 1])
            ),
            'bsdf': {'type': 'polarizer', 'theta': 0.0},
        },
        
        # 背景牆
        'wall': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, 0.25, z_max + 0.2]) @
                mi.ScalarTransform4f.scale([2.0, 0.6, 1])
            ),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': CONFIG['colors']['background']},
            },
        },
        
        # 地面
        'floor': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, 0, z_center]) @
                mi.ScalarTransform4f.rotate([1, 0, 0], -90) @
                mi.ScalarTransform4f.scale([1.5, 1.0, 1])
            ),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': CONFIG['colors']['ground']},
            },
        },
    }
    
    # 加入物件
    for name, data in objects.items():
        if not data['verts'] or not data['faces']:
            continue
        
        verts = transform_blender_to_mitsuba(data['verts'])
        faces = data['faces']
        
        max_idx = max(max(f) for f in faces) if faces else 0
        if max_idx >= len(verts):
            continue
        
        safe_name = name.replace(' ', '_').replace('.', '_').replace('-', '_')
        ply_path = os.path.join(temp_dir, f"{safe_name}.ply")
        
        if not os.path.exists(ply_path):
            write_ply(ply_path, verts, faces)
        
        if is_glass(name):
            bsdf = {
                'type': 'dielectric',
                'int_ior': 1.5,
                'ext_ior': 1.0,
            }
        else:
            color = get_color(name)
            bsdf = {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': color},
            }
        
        scene[f'obj_{safe_name}'] = {
            'type': 'ply',
            'filename': ply_path,
            'bsdf': bsdf,
        }
    
    return scene


# ============================================================
# 場景建立 - 偏振模式 (原版白光)
# ============================================================

def build_scene_polarized(mi, obj_path, temp_dir):
    """
    建立偏振場景 - 符合 PIDS 論文的光學設置
    
    使用 scalar_rgb_polarized 模式，直接輸出 RGB，避免光譜轉換問題
    """
    
    objects = parse_obj(obj_path)
    
    # 計算物件範圍
    all_z = []
    for data in objects.values():
        if data['verts']:
            for v in data['verts']:
                all_z.append(-v[1] / 1000.0)
    
    z_center = (min(all_z) + max(all_z)) / 2 if all_z else 0.6
    z_max = max(all_z) if all_z else 0.8
    z_min = min(all_z) if all_z else 0.5
    
    print(f"  物件 Z 範圍: [{z_min:.3f}, {z_max:.3f}] m")
    
    res = CONFIG['camera']['resolution']
    fov = CONFIG['camera']['fov']
    samples = CONFIG['render']['samples']
    half_baseline = CONFIG['camera']['baseline_mm'] / 2000.0
    
    scene = {
        'type': 'scene',
        
        'integrator': {
            'type': 'stokes',
            'nested': {
                'type': 'path',
                'max_depth': CONFIG['render']['max_depth'],
            },
        },
        
        # 左相機
        'sensor_left': {
            'type': 'perspective',
            'fov': fov,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[-half_baseline, 0.05, 0],
                target=[-half_baseline, 0.05, z_center],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': res[0],
                'height': res[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
            },
            'sampler': {'type': 'independent', 'sample_count': samples},
        },
        
        # 右相機
        'sensor_right': {
            'type': 'perspective',
            'fov': fov,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[half_baseline, 0.05, 0],
                target=[half_baseline, 0.05, z_center],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': res[0],
                'height': res[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
            },
            'sampler': {'type': 'independent', 'sample_count': samples},
        },
        
        # ============================================================
        # 光源系統 - 偏振光 (使用 RGB 顏色)
        # ============================================================
        
        # 主光源 (相機上方，與光軸夾角 35°，入射角接近 Brewster's angle ~55°)
        # 光源在相機正上方，向前傾斜照射場景
        'main_light': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, 0.4, 0.1]) @  # 相機上方 400mm，稍微前移
                mi.ScalarTransform4f.rotate([1, 0, 0], -35) @     # 與光軸夾角 35°
                mi.ScalarTransform4f.scale([0.4, 0.3, 1])
            ),
            'bsdf': {'type': 'null'},
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [120.0, 120.0, 120.0]},
            },
        },
        
        # 光源前的偏振片 (0°)，與光源同角度
        'source_polarizer': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, 0.38, 0.12]) @  # 緊貼光源前方
                mi.ScalarTransform4f.rotate([1, 0, 0], -35) @
                mi.ScalarTransform4f.scale([0.42, 0.32, 1])
            ),
            'bsdf': {
                'type': 'polarizer',
                'theta': 0.0,
            },
        },
        
        # 補光
        'fill_light': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0.4, 0.4, z_center]) @
                mi.ScalarTransform4f.rotate([0, 1, 0], -45) @
                mi.ScalarTransform4f.rotate([1, 0, 0], -30) @
                mi.ScalarTransform4f.scale([0.2, 0.2, 1])
            ),
            'bsdf': {'type': 'null'},
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [30.0, 30.0, 30.0]},
            },
        },
        
        # 環境光
        'env': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]},
        },
        
        # ============================================================
        # 背景 - 使用 RGB 顏色
        # ============================================================
        'wall': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, 0.3, z_max + 0.3]) @
                mi.ScalarTransform4f.scale([1.5, 0.8, 1])
            ),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': CONFIG['colors']['background']},
            },
        },
        
        # 地面
        'floor': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, 0, z_center]) @
                mi.ScalarTransform4f.rotate([1, 0, 0], -90) @
                mi.ScalarTransform4f.scale([1.5, 1.0, 1])
            ),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': CONFIG['colors']['ground']},
            },
        },
    }
    
    # 加入物件
    for name, data in objects.items():
        if not data['verts'] or not data['faces']:
            continue
        
        verts = transform_blender_to_mitsuba(data['verts'])
        faces = data['faces']
        
        max_idx = max(max(f) for f in faces) if faces else 0
        if max_idx >= len(verts):
            continue
        
        safe_name = name.replace(' ', '_').replace('.', '_').replace('-', '_')
        ply_path = os.path.join(temp_dir, f"{safe_name}.ply")
        write_ply(ply_path, verts, faces)
        
        if is_glass(name):
            # 玻璃：使用 thindielectric (薄介電質)
            # 更適合薄玻璃片（窗戶、面板），正確模擬偏振 Fresnel 反射
            bsdf = {
                'type': 'thindielectric',
                'int_ior': 1.5,
                'ext_ior': 1.0,
            }
            print(f"    {name} -> 玻璃 (thindielectric)")
        else:
            # 非玻璃物件用 RGB 漫反射
            color = get_color(name)
            bsdf = {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': color}
            }
            print(f"    {name} -> 漫反射 RGB {color}")
        
        scene[f'obj_{safe_name}'] = {
            'type': 'ply',
            'filename': ply_path,
            'bsdf': bsdf,
        }
    
    return scene


# ============================================================
# 深度場景
# ============================================================

def build_scene_depth(mi, obj_path, temp_dir):
    """
    建立深度渲染場景
    
    注意：深度 GT 從左相機位置渲染，以對齊 I∥ (Parallel View)
    使用高精度設定確保深度圖品質
    """
    
    objects = parse_obj(obj_path)
    
    all_z = []
    for data in objects.values():
        if data['verts']:
            for v in data['verts']:
                all_z.append(-v[1] / 1000.0)
    
    z_center = (min(all_z) + max(all_z)) / 2 if all_z else 0.6
    z_max = max(all_z) if all_z else 0.8
    
    res = CONFIG['camera']['resolution']
    fov = CONFIG['camera']['fov']
    half_baseline = CONFIG['camera']['baseline_mm'] / 2000.0  # 轉換為米
    
    scene = {
        'type': 'scene',
        
        'integrator': {
            'type': 'aov',
            'aovs': 'dd:depth',
            'nested': {
                'type': 'path',
                'max_depth': 2,
            },
        },
        
        # 深度相機與左相機 (I∥) 對齊
        'sensor': {
            'type': 'perspective',
            'fov': fov,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[-half_baseline, 0.05, 0],    # 左相機位置
                target=[-half_baseline, 0.05, z_center],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': res[0],
                'height': res[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',  # 最高精度
                'rfilter': {'type': 'box'},     # 不模糊，保持銳利邊緣
            },
            'sampler': {
                'type': 'independent', 
                'sample_count': 64,  # 增加採樣數提高精度
            },
        },
        
        'light': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': [1, 1, 1]},
        },
        
        'wall': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, 0.3, z_max + 0.3]) @
                mi.ScalarTransform4f.scale([1.5, 0.8, 1])
            ),
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}},
        },
        
        'floor': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.translate([0, 0, z_center]) @
                mi.ScalarTransform4f.rotate([1, 0, 0], -90) @
                mi.ScalarTransform4f.scale([1.5, 1.0, 1])
            ),
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}},
        },
    }
    
    for name, data in objects.items():
        if not data['verts'] or not data['faces']:
            continue
        
        verts = transform_blender_to_mitsuba(data['verts'])
        faces = data['faces']
        
        max_idx = max(max(f) for f in faces) if faces else 0
        if max_idx >= len(verts):
            continue
        
        safe_name = name.replace(' ', '_').replace('.', '_').replace('-', '_')
        ply_path = os.path.join(temp_dir, f"{safe_name}.ply")
        write_ply(ply_path, verts, faces)
        
        # 深度渲染時用不透明材質
        bsdf = {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}}
        
        scene[f'obj_{safe_name}'] = {
            'type': 'ply',
            'filename': ply_path,
            'bsdf': bsdf,
        }
    
    return scene


# ============================================================
# Stokes 處理
# ============================================================

def process_stokes_rgb(img, analyzer_angle_deg):
    """
    處理 spectral_polarized 模式的 Stokes 輸出
    
    根據實際測試發現：
    - ch0-ch2 = S0, S1, S2 (第一組)
    - ch3-ch5 = S0, S1, S2 (重複，數值相同)
    - ch6-ch14 = 0
    
    所以直接用 ch0=S0, ch1=S1, ch2=S2
    """
    img = np.array(img, dtype=np.float64)
    theta = np.radians(analyzer_angle_deg)
    
    print(f"    Stokes shape: {img.shape}, range: [{img.min():.4f}, {img.max():.4f}]")
    
    if img.ndim == 2:
        return np.stack([img, img, img], axis=2)
    
    if img.shape[2] == 3:
        return img
    
    # 直接取前三個通道
    S0 = img[:, :, 0]  # 總強度
    S1 = img[:, :, 1]  # 線性偏振 (0°-90°)
    S2 = img[:, :, 2]  # 線性偏振 (45°-135°)
    
    print(f"    S0: [{S0.min():.4f}, {S0.max():.4f}], mean={S0.mean():.4f}")
    print(f"    S1: [{S1.min():.4f}, {S1.max():.4f}], mean={S1.mean():.4f}")
    
    # 計算通過 analyzer 的強度
    # I(θ) = (S0 + S1*cos(2θ) + S2*sin(2θ)) / 2
    cos2theta = np.cos(2 * theta)
    sin2theta = np.sin(2 * theta)
    
    I = (S0 + S1 * cos2theta + S2 * sin2theta) / 2
    I = np.maximum(I, 0)
    
    print(f"    I (θ={analyzer_angle_deg}°): [{I.min():.4f}, {I.max():.4f}], mean={I.mean():.4f}")
    
    # 輸出灰階 (因為 spectral 模式沒有保留顏色資訊)
    def linear_to_srgb(c):
        c = np.clip(c, 0, None)
        return np.where(c <= 0.0031308,
                       12.92 * c,
                       1.055 * np.power(np.maximum(c, 1e-10), 1/2.4) - 0.055)
    
    I_gamma = linear_to_srgb(I)
    
    # 返回灰階
    return np.stack([I_gamma, I_gamma, I_gamma], axis=2)


# ============================================================
# 渲染
# ============================================================

def render_rgb(obj_path, output_dir):
    """RGB 渲染 (測試用)"""
    mi = get_mitsuba(polarized=False)
    
    obj_path = Path(obj_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_dir / '_temp'
    temp_dir.mkdir(exist_ok=True)
    
    name = obj_path.stem
    
    print(f"\n{'='*50}")
    print(f"RGB 渲染: {name}")
    print('='*50)
    
    try:
        scene_dict = build_scene_rgb(mi, str(obj_path), str(temp_dir))
        scene = mi.load_dict(scene_dict)
        
        print("  渲染中...")
        img = mi.render(scene)
        img = np.array(img)
        
        print(f"  輸出 shape: {img.shape}")
        print(f"  範圍: [{img.min():.4f}, {img.max():.4f}]")
        
        # 正規化
        p99 = np.percentile(img, 99)
        if p99 > 0:
            img = np.clip(img / p99, 0, 1)
        
        # 儲存
        img_uint8 = (img * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{name}_rgb.png"), 
                   cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        
        print(f"  完成: {name}_rgb.png")
        return True
        
    except Exception as e:
        print(f"  錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def render_polarized(obj_path, output_dir):
    """偏振渲染 - (GPU 分批累積版)"""
    obj_path = Path(obj_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / '_temp'
    temp_dir.mkdir(exist_ok=True)
    name = obj_path.stem
    
    print(f"\n{'='*50}\n偏振渲染 (GPU 分批累積): {name}\n{'='*50}")
    
    use_denoise = CONFIG['render'].get('denoise', True)
    if use_denoise: init_oidn()
    
    try:
        mi = get_mitsuba(polarized=True)
        
        # ==========================================
        # 1. 計算分批策略 (避免 OOM)
        # ==========================================
        target_samples = CONFIG['render']['samples']
        batch_spp = 1024  # 每次只算 1024 (H100 很輕鬆)
        num_batches = (target_samples + batch_spp - 1) // batch_spp
        
        print(f"  目標採樣: {target_samples}, 分批大小: {batch_spp}, 總批次: {num_batches}")
        
        # 暫時修改 CONFIG 以建立正確的場景描述
        CONFIG['render']['samples'] = batch_spp
        scene_dict = build_scene_polarized(mi, str(obj_path), str(temp_dir))
        scene = mi.load_dict(scene_dict)
        
        # 準備累積器 (初始化為 0)
        acc_left = 0
        acc_right = 0
        
        # ==========================================
        # 2. 開始分批渲染迴圈
        # ==========================================
        import drjit as dr
        
        for i in range(num_batches):
            sys.stdout.write(f"\r  正在渲染批次 [{i+1}/{num_batches}] ... ")
            sys.stdout.flush()
            
            # 渲染左相機 (累積)
            # seed=i 確保每次雜訊分佈不同
            img_l = mi.render(scene, sensor=0, seed=i)
            acc_left += img_l
            
            # 渲染右相機 (累積)
            img_r = mi.render(scene, sensor=1, seed=i)
            acc_right += img_r
            
            # 釋放該批次記憶體
            dr.flush_malloc_cache()
            
        print("\n  渲染完成，正在處理數據...")
        
        # 取平均
        avg_left = acc_left / num_batches
        avg_right = acc_right / num_batches
        
        # ==========================================
        # 3. 後處理 (轉 Stokes -> 存檔)
        # ==========================================
        I_parallel = process_stokes_rgb(avg_left, 0.0)[:,:,0]
        I_cross = process_stokes_rgb(avg_right, 90.0)[:,:,0]
        
        if use_denoise and _oidn_available:
            print("  執行 OIDN 降噪...")
            I_parallel = denoise_image(I_parallel, is_hdr=True)
            I_cross = denoise_image(I_cross, is_hdr=True)
        
        cv2.imwrite(str(output_dir / f"{name}_I_parallel.exr"), I_parallel.astype(np.float32))
        cv2.imwrite(str(output_dir / f"{name}_I_cross.exr"), I_cross.astype(np.float32))
        
        # 產生預覽圖 (正規化)
        combined = np.maximum(I_parallel, I_cross)
        p99 = np.percentile(combined, 99)
        scale = 1.0 / p99 if p99 > 1e-6 else 1.0
        I_par_norm = np.clip(I_parallel * scale, 0, 1)
        I_cross_norm = np.clip(I_cross * scale, 0, 1)
        
        cv2.imwrite(str(output_dir / f"{name}_I_parallel.png"), (I_par_norm * 255).astype(np.uint8))
        cv2.imwrite(str(output_dir / f"{name}_I_cross.png"), (I_cross_norm * 255).astype(np.uint8))
        
        # ========== 渲染深度 GT (不需要分批，因為只有 64 spp) ==========
        print("  渲染深度 GT...")
        import mitsuba as mi_depth
        try:
            mi_depth.set_variant('cuda_ad_rgb')
        except:
            mi_depth.set_variant('scalar_rgb')
        
        # 深度圖用原本的採樣設定即可
        depth_scene_dict = build_scene_depth(mi_depth, str(obj_path), str(temp_dir))
        depth_scene = mi_depth.load_dict(depth_scene_dict)
        depth_img = mi_depth.render(depth_scene)
        
        if hasattr(depth_img, 'torch'):
            depth = depth_img.torch().cpu().numpy()
        else:
            depth = np.array(depth_img)
            
        if depth.ndim == 3: depth = depth[:, :, 0]
        
        cv2.imwrite(str(output_dir / f"{name}_depth.exr"), depth.astype(np.float32))
        
        # 深度圖可視化
        valid_mask = (depth > 0) & (depth < 10)
        if valid_mask.any():
            d_min = depth[valid_mask].min()
            d_max = depth[valid_mask].max()
            depth_norm = np.zeros_like(depth)
            depth_norm[valid_mask] = (depth[valid_mask] - d_min) / (d_max - d_min + 1e-6)
        else:
            depth_norm = np.zeros_like(depth)
        
        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(output_dir / f"{name}_depth.png"), depth_color)
        
        print(f"  完成！ 輸出於: {output_dir}")
        return True
        
    except Exception as e:
        print(f"  錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        import shutil
        if temp_dir.exists(): shutil.rmtree(temp_dir)


# ============================================================
# 主程式 - 批次處理版本
# ============================================================

def batch_render(input_dir, output_dir, samples, polarized=True):
    """
    批次渲染資料夾中的所有 OBJ 檔案
    
    Args:
        input_dir: 包含 OBJ 檔案的資料夾
        output_dir: 輸出根目錄
        samples: 採樣數
        polarized: 是否使用偏振模式
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 尋找所有 OBJ 檔案
    obj_files = list(input_path.glob('*.obj'))
    
    if not obj_files:
        # 也搜尋子資料夾
        obj_files = list(input_path.glob('**/*.obj'))
    
    if not obj_files:
        print(f"錯誤: 在 {input_dir} 中找不到 OBJ 檔案")
        return
    
    print(f"\n{'='*60}")
    print(f"PIDS 批次渲染器")
    print(f"{'='*60}")
    print(f"輸入目錄: {input_dir}")
    print(f"輸出目錄: {output_dir}")
    print(f"找到 {len(obj_files)} 個 OBJ 檔案")
    print(f"採樣數: {samples}")
    print(f"模式: {'偏振' if polarized else 'RGB'}")
    print(f"{'='*60}\n")
    
    CONFIG['render']['samples'] = samples
    
    success_count = 0
    fail_count = 0
    
    for i, obj_file in enumerate(obj_files):
        print(f"\n[{i+1}/{len(obj_files)}] 處理: {obj_file.name}")
        print("-" * 40)
        
        try:
            if polarized:
                result = render_polarized(str(obj_file), str(output_path))
            else:
                result = render_rgb(str(obj_file), str(output_path))
            
            if result:
                success_count += 1
            else:
                fail_count += 1
                
        except Exception as e:
            print(f"  錯誤: {e}")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"批次處理完成!")
    print(f"成功: {success_count}/{len(obj_files)}")
    print(f"失敗: {fail_count}/{len(obj_files)}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='PIDS 渲染器 v5.2 - 支援批次處理 + OIDN 降噪',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 單一檔案渲染
  python pids_renderer_v5.py -i scene.obj -o ./output --polarized
  
  # 批次渲染資料夾中所有 OBJ
  python pids_renderer_v5.py -i ./scenes/ -o ./output --polarized --batch
  
  # 指定採樣數 (預設 16384)
  python pids_renderer_v5.py -i ./scenes/ -o ./output --polarized --batch -s 16384
  
  # 關閉降噪
  python pids_renderer_v5.py -i ./scenes/ -o ./output --polarized --batch --no-denoise

安裝 OIDN:
  pip install oidn
        """
    )
    parser.add_argument('-i', '--input', required=True, 
                        help='OBJ 檔案或包含 OBJ 的資料夾')
    parser.add_argument('-o', '--output', required=True, 
                        help='輸出目錄')
    parser.add_argument('-s', '--samples', type=int, default=16384,
                        help='採樣數 (預設: 16384，偏振模式建議 16384+)')
    parser.add_argument('--polarized', action='store_true', 
                        help='偏振模式 (輸出 I∥ 和 I⊥)')
    parser.add_argument('--batch', action='store_true',
                        help='批次處理模式 (處理資料夾中所有 OBJ)')
    parser.add_argument('--no-denoise', action='store_true',
                        help='關閉 OIDN 降噪 (預設開啟)')
    
    args = parser.parse_args()
    
    CONFIG['render']['samples'] = args.samples
    CONFIG['render']['denoise'] = not args.no_denoise
    
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # 批次處理模式
        batch_render(args.input, args.output, args.samples, args.polarized)
    else:
        # 單一檔案模式
        if args.polarized:
            render_polarized(args.input, args.output)
        else:
            render_rgb(args.input, args.output)


if __name__ == "__main__":
    main()

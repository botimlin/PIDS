"""
PIDS 渲染器 v5.2 - HDR EXR + 批次處理 + OIDN 降噪 + GPU 支援
============================================================

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

  # 指定採樣數 (預設 4096)
  python pids_renderer_v5.py -i ./scenes/ -o ./output --polarized --batch -s 16384
  
  # 關閉降噪 (預設開啟)
  python pids_renderer_v5.py -i ./scenes/ -o ./output --polarized --batch --no-denoise

安裝 OIDN:
  pip install oidn
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # 啟用 OpenEXR 支援
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
        
        # 檢查回傳是否為 int (某些版本的 bug)
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
        print("  警告: OIDN 未安裝，跳過降噪 (pip install oidn)")
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
        'samples': 4096,   # 偏振模式需要較高採樣數以減少噪點
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
# Mitsuba 初始化 (GPU 優先)
# ============================================================

_mi = None

def get_mitsuba(polarized=False):
    """
    初始化 Mitsuba，優先使用 CUDA GPU
    如果 GPU 不可用，自動回退到 CPU
    """
    global _mi
    import mitsuba as mi
    
    target_variant = 'cuda_ad_spectral_polarized' if polarized else 'cuda_ad_rgb'
    fallback_variant = 'scalar_spectral_polarized' if polarized else 'scalar_rgb'
    
    print(f"  請求變體: {target_variant} (polarized={polarized})")
    
    # 嘗試 GPU 模式
    try:
        mi.set_variant(target_variant)
        actual = mi.variant()
        print(f"  實際變體: {actual} (GPU)")
        
        # 驗證偏振變體
        if polarized and 'polarized' not in actual:
            print(f"  ⚠️ 警告: 請求偏振模式但實際變體不含 'polarized'!")
            
    except Exception as e:
        print(f"  警告: 無法啟用 {target_variant}，回退至 CPU。錯誤: {e}")
        try:
            mi.set_variant(fallback_variant)
            actual = mi.variant()
            print(f"  實際變體: {actual} (CPU)")
        except Exception as e2:
            print(f"  錯誤: 無法設定任何變體: {e2}")
            raise
    
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
    
    # 背景物件 (牆壁、地板、天花板) 使用特定顏色
    if 'background_wall' in name_lower or 'wall' in name_lower:
        return CONFIG['colors']['background']
    if 'background_ground' in name_lower or 'floor' in name_lower:
        return CONFIG['colors']['ground']
    if 'background_ceiling' in name_lower or 'ceiling' in name_lower:
        return [0.85, 0.83, 0.80]  # 淺色天花板
    
    # 傢俱物件
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
                origin=[0, 0.02, 0.42],
                target=[0, 0.02, z_center],
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
                'type': 'pplastic', 'diffuse_reflectance': {'type': 'rgb', 'value': CONFIG['colors']['background']}, 'int_ior': 1.49, 'ext_ior': 1.0,
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
                'type': 'pplastic', 'diffuse_reflectance': {'type': 'rgb', 'value': CONFIG['colors']['ground']}, 'int_ior': 1.49, 'ext_ior': 1.0,
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
                'type': 'pplastic', 'diffuse_reflectance': {'type': 'rgb', 'value': color}, 'int_ior': 1.49, 'ext_ior': 1.0,
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

def build_scene_polarized(mi, obj_path, temp_dir, samples=4096):
    """
    基於 blender_furniture_randomizer_v17.py 座標系統
    
    Blender 座標 (相機在原點，朝 +Y):
    - 相機: Y = 0
    - 前牆: Y = 350mm
    - 傢俱: Y = 750mm
    - 後牆: Y = 900mm
    
    OBJ 匯出 (forward_axis='NEGATIVE_Y'):
    - OBJ Y = -Blender Y
    
    所以在 Mitsuba 中:
    - 相機: Y = 0
    - 前牆: Y = -350
    - 傢俱: Y = -750
    - 後牆: Y = -900
    """
    
    # -----------------------------------------------------------
    # 1. 解析 OBJ 並為不同材質設定 BSDF
    # -----------------------------------------------------------
    objects = parse_obj(obj_path)
    
    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'stokes', 'nested': {'type': 'path', 'max_depth': 12}}, 
    }
    
    # 為每個物件設定材質
    for name, data in objects.items():
        if not data['verts'] or not data['faces']:
            continue
        
        verts = data['verts']  # 直接用原始座標，不做轉換
        faces = data['faces']
        
        max_idx = max(max(f) for f in faces) if faces else 0
        if max_idx >= len(verts):
            continue
        
        safe_name = name.replace(' ', '_').replace('.', '_').replace('-', '_')
        ply_path = os.path.join(temp_dir, f"pol_{safe_name}.ply")
        write_ply(ply_path, verts, faces)
        
        # 根據名稱判斷材質
        name_lower = name.lower()
        if 'glass' in name_lower or 'window' in name_lower or '玻璃' in name_lower:
            # 玻璃：薄介電質 (支援偏振)
            bsdf = {
                'type': 'thindielectric',
                'int_ior': 1.5,  # 玻璃折射率
                'ext_ior': 1.0,  # 空氣
            }
            print(f"    玻璃物件: {name} (thindielectric)")
        else:
            # 其他：漫反射
            bsdf = {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}
            }
        
        scene_dict[f'obj_{safe_name}'] = {
            'type': 'ply',
            'filename': ply_path,
            'bsdf': bsdf,
        }

    # -----------------------------------------------------------
    # 2. 座標設定
    # -----------------------------------------------------------
    # 根據深度場景報告: 模型中心 = [0, -626.25, 151.5]
    # 這表示 chamber 大約在 Y = -350 到 -900 之間
    # 相機應該在 chamber 內部
    
    # 相機放在 chamber 前端 (Y=-400)，看向後方
    # 焦點與相機同高，避免仰角
    cam_y = -400
    cam_z = 20  # 貼地
    focus_point = [0, -750, cam_z]  # 焦點與相機同高
    baseline = 65.0
    
    # -----------------------------------------------------------
    # 3. 光源配置 (在相機正上方)
    # -----------------------------------------------------------
    light_z = 250  # 光源高度
    
    light_pos = [0, cam_y, light_z]  # 光源在相機正上方
    light_size = 150.0
    
    scene_dict['light_source'] = {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=light_pos,
            target=focus_point,
            up=[0, 0, 1]
        ).scale([light_size, light_size, 1.0]),
        
        'emitter': {
            'type': 'area',
            'radiance': {'type': 'spectrum', 'value': 100000.0}
        }
    }
    
    # 光源偏振片 (緊貼光源前方)
    scene_dict['light_filter'] = {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, cam_y - 5, light_z - 5],  # 光源前方一點
            target=focus_point,
            up=[0, 0, 1]
        ).scale([light_size * 1.1, light_size * 1.1, 1.0]),
        'bsdf': {
            'type': 'polarizer',
            'theta': 0 
        }
    }
    
    # 環境光 (非偏振，減少死黑區域)
    scene_dict['envmap'] = {
        'type': 'constant',
        'radiance': {'type': 'spectrum', 'value': 500.0}  # 弱環境光
    }

    # -----------------------------------------------------------
    # 4. 相機配置 (貼地)
    # -----------------------------------------------------------
    # cam_y, cam_z 已在上面定義
    
    half_base = baseline / 2.0
    target_fov = 55.0

    sensor_props = {
        'type': 'perspective',
        'fov': target_fov,
        'sampler': {'type': 'ldsampler', 'sample_count': samples},
        'film': {
            'type': 'hdrfilm',
            'width': 640, 'height': 480,
            'pixel_format': 'luminance',  # 灰階 Stokes 輸出 (4通道: S0,S1,S2,S3)
            'component_format': 'float32',
        }
    }

    # 左眼相機 (用於渲染 Stokes 向量)
    scene_dict['sensor_left'] = sensor_props.copy()
    scene_dict['sensor_left']['to_world'] = mi.ScalarTransform4f.look_at(
        origin=[-half_base, cam_y, cam_z],
        target=focus_point,
        up=[0, 0, 1]
    )
    
    # 右眼相機 (雙目立體)
    scene_dict['sensor_right'] = sensor_props.copy()
    scene_dict['sensor_right']['to_world'] = mi.ScalarTransform4f.look_at(
        origin=[half_base, cam_y, cam_z],
        target=focus_point,
        up=[0, 0, 1]
    )

    return scene_dict



# ============================================================
# 深度場景
# ============================================================

def build_scene_depth(mi, obj_path, temp_dir):
    """
    建立深度渲染場景
    - 相機與偏振左相機對齊
    - 只載入 chamber 內部物件（排除牆壁）
    """
    
    # 解析 OBJ 找出內部物件
    objects = parse_obj(obj_path)
    
    # 過濾：排除牆壁和天花板，但保留地板
    exclude_keywords = ['wall', 'ceiling']  # 不排除 floor/ground
    interior_objects = {}
    for name, data in objects.items():
        name_lower = name.lower()
        if not any(kw in name_lower for kw in exclude_keywords):
            interior_objects[name] = data
    
    print(f"  深度場景: 找到 {len(interior_objects)} 個內部物件 (排除牆壁)")
    for name in interior_objects.keys():
        print(f"    - {name}")
    
    # 相機參數（與偏振相機完全一致）
    res = CONFIG['camera']['resolution']
    fov = CONFIG['camera']['fov']
    half_baseline = CONFIG['camera']['baseline_mm'] / 2.0
    
    # 與 build_scene_polarized 相同的座標
    cam_y = -400
    cam_z = 20  # 貼地
    focus_point = [0, -750, cam_z]  # 焦點與相機同高
    
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
        
        # 深度相機與左相機對齊
        'sensor': {
            'type': 'perspective',
            'fov': fov,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[-half_baseline, cam_y, cam_z],
                target=focus_point,
                up=[0, 0, 1]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': res[0],
                'height': res[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
                'rfilter': {'type': 'box'},
            },
            'sampler': {
                'type': 'independent', 
                'sample_count': 64,
            },
        },
        
        # 環境光
        'light': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': [1, 1, 1]},
        },
    }
    
    # 加入內部物件
    for name, data in interior_objects.items():
        if not data['verts'] or not data['faces']:
            continue
        
        verts = data['verts']  # 直接用原始座標
        faces = data['faces']
        
        max_idx = max(max(f) for f in faces) if faces else 0
        if max_idx >= len(verts):
            continue
        
        safe_name = name.replace(' ', '_').replace('.', '_').replace('-', '_')
        ply_path = os.path.join(temp_dir, f"depth_{safe_name}.ply")
        write_ply(ply_path, verts, faces)
        
        # 深度渲染用不透明材質
        scene[f'obj_{safe_name}'] = {
            'type': 'ply',
            'filename': ply_path,
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}}
        }
    
    return scene


# ============================================================
# Stokes 處理
# ============================================================

def process_stokes_luminance(img, analyzer_angle_deg):
    """
    處理 Mitsuba 'luminance' 模式的輸出 (4通道: S0, S1, S2, S3)
    計算通過特定角度偏振片後的灰階強度。
    """
    # 轉為 Numpy array
    data = np.array(img, dtype=np.float32)
    
    # 檢查通道數是否正確
    h, w, c = data.shape
    if c != 4:
        print(f"錯誤：預期 4 個通道 (S0,S1,S2,S3)，但收到 {c} 個。請確認 pixel_format='luminance'")
        return np.zeros((h, w), dtype=np.float32)

    # 讀取 Stokes 分量
    S0 = data[:, :, 0]
    S1 = data[:, :, 1]
    S2 = data[:, :, 2]
    # S3 = data[:, :, 3] # PIDS 不需要圓偏振

    # 計算偏振角度 (轉弧度)
    theta = np.radians(analyzer_angle_deg)
    
    # Mueller Calculus (物理公式)
    # I = 0.5 * (S0 + S1 * cos(2θ) + S2 * sin(2θ))
    cos2 = np.cos(2 * theta)
    sin2 = np.sin(2 * theta)
    
    I_out = 0.5 * (S0 + S1 * cos2 + S2 * sin2)
    
    # 確保物理合理性 (光強 >= 0)
    I_out = np.maximum(I_out, 0)
    
    return I_out


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

def process_stokes_luminance(img, analyzer_angle_deg):
    """
    解析 Mitsuba 偏振模式輸出
    
    Mitsuba spectral_polarized 模式輸出格式:
    - 4 通道: S0, S1, S2, S3 (luminance 模式)
    - 13 通道: 多光譜 Stokes (每個光譜帶 4 個 Stokes 分量，但有重疊)
               通常是 ch0-3 = 第一組 S0,S1,S2,S3
    
    計算通過特定角度偏振片後的光強。
    """
    # 轉為 Numpy 並確保是 float32
    data = np.array(img, dtype=np.float32)
    
    h, w = data.shape[:2]
    c = data.shape[2] if data.ndim == 3 else 1
    
    print(f"    Stokes 數據 shape: {data.shape}")
    
    if data.ndim == 2:
        # 單通道，無偏振信息
        print(f"    警告: 單通道數據，無偏振信息")
        return data
    
    if c == 4:
        # 標準 4 通道 Stokes
        S0 = data[:, :, 0]
        S1 = data[:, :, 1]
        S2 = data[:, :, 2]
    elif c == 13:
        # Mitsuba spectral_polarized + luminance 輸出格式分析:
        # ch0-3: 光譜帶強度 (全正)
        # ch4-6: 更多光譜帶
        # ch7-9: S1 分量 (可正可負，這是線性偏振！)
        # ch10-12: S3 分量 (圓偏振，通常為 0)
        
        print(f"    所有通道範圍:")
        for i in range(min(c, 13)):
            ch = data[:, :, i]
            print(f"      ch{i}: [{ch.min():.2f}, {ch.max():.2f}], mean={ch.mean():.2f}")
        
        # 正確的 Stokes 對應:
        # S0 (總強度) = ch0 + ch4 的平均，或直接用 ch3 (看起來最合理)
        # 但根據數據，ch3 和 ch0 很接近
        # S1 (線性偏振 0/90) = ch7
        # S2 (線性偏振 45/135) = ch8 或 ch9
        
        # 嘗試用 ch3 作為 S0，ch7 作為 S1，ch8 作為 S2
        S0 = data[:, :, 3]  # 總強度
        S1 = data[:, :, 7]  # 線性偏振 (有正有負)
        S2 = data[:, :, 8]  # 線性偏振 (有正有負)
        
        print(f"    使用 ch3=S0, ch7=S1, ch8=S2")
        print(f"    S0 範圍: [{S0.min():.4f}, {S0.max():.4f}]")
        print(f"    S1 範圍: [{S1.min():.4f}, {S1.max():.4f}]")
        print(f"    S2 範圍: [{S2.min():.4f}, {S2.max():.4f}]")
        
        # 驗證物理合理性: |S1| <= S0
        if np.abs(S1).max() <= S0.max() * 1.1:
            print(f"    ✓ 物理合理: |S1| <= S0")
    elif c >= 4:
        # 其他多通道格式，取前 4 個
        S0 = data[:, :, 0]
        S1 = data[:, :, 1]
        S2 = data[:, :, 2]
        print(f"    使用前 4 通道作為 Stokes")
    else:
        print(f"    警告: 通道數 {c} 不足，僅回傳第一通道")
        return data[:, :, 0]

    # 物理計算: I = 0.5 * (S0 + S1 cos(2θ) + S2 sin(2θ))
    theta = np.radians(analyzer_angle_deg)
    cos2 = np.cos(2 * theta)
    sin2 = np.sin(2 * theta)
    
    I_out = 0.5 * (S0 + S1 * cos2 + S2 * sin2)
    return np.maximum(I_out, 0)  # 確保不為負

def render_polarized(obj_path, output_dir):
    """偏振渲染 - GPU 分批累積版 + OIDN 降噪 (修正版)"""
    
    obj_path = Path(obj_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_dir / '_temp'
    temp_dir.mkdir(exist_ok=True)
    
    name = obj_path.stem
    
    print(f"\n{'='*50}")
    print(f"偏振渲染 (GPU 分批累積): {name}")
    print('='*50)
    
    # 初始化降噪器
    use_denoise = CONFIG['render'].get('denoise', True)
    if use_denoise:
        init_oidn()
    
    try:
        # 確保開啟偏振模式
        mi = get_mitsuba(polarized=True)
        
        # ==========================================
        # 1. 計算分批策略
        # ==========================================
        target_samples = CONFIG['render']['samples']
        # RTX 5090 32GB VRAM + RT cores，可以一次跑更多
        batch_spp = 8192  
        num_batches = (target_samples + batch_spp - 1) // batch_spp
        
        print(f"  目標採樣: {target_samples}, 分批大小: {batch_spp}, 總批次: {num_batches}")
        
        # 暫時修改 CONFIG 以建立正確的場景
        original_samples = CONFIG['render']['samples']
        CONFIG['render']['samples'] = batch_spp
        
        # 建立場景 (這裡會呼叫我們修改過的 build_scene_polarized)
        scene_dict = build_scene_polarized(mi, str(obj_path), str(temp_dir), samples=batch_spp)
        scene = mi.load_dict(scene_dict)
        
        # 還原 CONFIG
        CONFIG['render']['samples'] = original_samples
        
        # 準備累積器
        acc_left = 0
        acc_right = 0
        
        # ==========================================
        # 2. 分批渲染迴圈
        # ==========================================
        try:
            import drjit as dr
            has_drjit = True
        except ImportError:
            has_drjit = False
        
        for i in range(num_batches):
            sys.stdout.write(f"\r  正在渲染批次 [{i+1}/{num_batches}] ... ")
            sys.stdout.flush()
            
            # 渲染左相機 (sensor=0)
            img_l = mi.render(scene, sensor=0, seed=i)
            acc_left = acc_left + img_l
            
            # 渲染右相機 (sensor=1)
            img_r = mi.render(scene, sensor=1, seed=i)
            acc_right = acc_right + img_r
            
            # 釋放 GPU 記憶體
            if has_drjit:
                dr.flush_malloc_cache()
        
        print("\n  渲染完成，正在處理數據...")
        
        # 取平均
        avg_left = acc_left / num_batches
        avg_right = acc_right / num_batches
        
        # ==========================================
        # 3. 後處理 (關鍵修改區)
        # ==========================================
        
        # 【修改點】使用 luminance 處理函數
        # 左眼 (Sensor 0) -> 視為平行視角 (0度)
        I_parallel = process_stokes_luminance(avg_left, 0.0)
        
        # 右眼 (Sensor 1) -> 視為垂直視角 (90度)
        I_cross = process_stokes_luminance(avg_right, 90.0)
        
        print(f"  I∥ 範圍: [{I_parallel.min():.4f}, {I_parallel.max():.4f}]")
        print(f"  I⊥ 範圍: [{I_cross.min():.4f}, {I_cross.max():.4f}]")
        
        # ========== OIDN 降噪 ==========
        if use_denoise and _oidn_available:
            print("  執行 OIDN 降噪...")
            # 注意: OIDN 預期 3 通道，如果我們是單通道灰階，先轉成 (H,W,1) 或 (H,W,3)
            # 這裡簡單處理：複製成 3 通道給 OIDN 吃，再轉回來
            def denoise_gray(img_gray):
                img_3c = np.stack([img_gray, img_gray, img_gray], axis=2)
                img_clean = denoise_image(img_3c, is_hdr=True)
                return img_clean[:,:,0] # 只取回單通道

            I_parallel = denoise_gray(I_parallel)
            I_cross = denoise_gray(I_cross)
            print("  降噪完成")
        
        # ========== 儲存 EXR (Linear HDR，訓練用) ==========
        cv2.imwrite(str(output_dir / f"{name}_I_parallel.exr"), I_parallel.astype(np.float32))
        cv2.imwrite(str(output_dir / f"{name}_I_cross.exr"), I_cross.astype(np.float32))
        print(f"  已儲存 EXR (HDR)")
        
        # ========== 儲存 PNG (Gamma 校正預覽，解決全黑問題) ==========
        
        # 1. 統一正規化 (使用 99th percentile)
        all_values = np.concatenate([I_parallel.flatten(), I_cross.flatten()])
        p99 = np.percentile(all_values, 99)
        print(f"  正規化係數 (p99): {p99:.4f}")
        
        if p99 > 1e-9:
            # Linear -> Normalized Linear
            I_par_norm = np.clip(I_parallel / p99, 0, 1)
            I_cross_norm = np.clip(I_cross / p99, 0, 1)
        else:
            I_par_norm = I_parallel
            I_cross_norm = I_cross

        # 2. 【關鍵】Gamma 校正 (Linear -> sRGB)
        # 讓暗部變亮，符合人眼與螢幕顯示
        I_par_disp = np.power(I_par_norm, 1.0/2.2)
        I_cross_disp = np.power(I_cross_norm, 1.0/2.2)
        
        cv2.imwrite(str(output_dir / f"{name}_I_parallel.png"), (I_par_disp * 255).astype(np.uint8))
        cv2.imwrite(str(output_dir / f"{name}_I_cross.png"), (I_cross_disp * 255).astype(np.uint8))
        print(f"  已儲存 PNG (含 Gamma 校正)")
        
        # ========== 渲染深度 GT (保持不變) ==========
        print("  渲染深度 GT...")
        
        depth_temp_dir = output_dir / '_temp_depth'
        depth_temp_dir.mkdir(exist_ok=True)
        
        import mitsuba as mi_depth
        try:
            mi_depth.set_variant('cuda_ad_rgb')
        except:
            mi_depth.set_variant('scalar_rgb')
        
        # 注意：這裡確保 build_scene_depth 使用與左眼相同的相機參數
        depth_scene_dict = build_scene_depth(mi_depth, str(obj_path), str(depth_temp_dir))
        depth_scene = mi_depth.load_dict(depth_scene_dict)
        depth_img = mi_depth.render(depth_scene)
        
        if hasattr(depth_img, 'torch'):
            depth = depth_img.torch().cpu().numpy()
        else:
            depth = np.array(depth_img)
        
        if depth.ndim == 3:
            depth = depth[:, :, 0]
            
        # 儲存深度
        cv2.imwrite(str(output_dir / f"{name}_depth.exr"), depth.astype(np.float32))
        
        # 深度視覺化
        valid_mask = (depth > 0.01) & (depth < 10.0)
        if valid_mask.any():
            d_min = depth[valid_mask].min()
            d_max = depth[valid_mask].max()
            depth_norm = np.zeros_like(depth)
            depth_norm[valid_mask] = (depth[valid_mask] - d_min) / (d_max - d_min + 1e-6)
            depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cv2.imwrite(str(output_dir / f"{name}_depth.png"), depth_color)
        
        # 清理
        import shutil
        if depth_temp_dir.exists():
            shutil.rmtree(depth_temp_dir)
        
        print(f"  完成！輸出於: {output_dir}")
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
  
  # 指定採樣數 (預設 4096)
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
    parser.add_argument('-s', '--samples', type=int, default=4096,
                        help='採樣數 (預設: 4096，偏振模式建議 4096+)')
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

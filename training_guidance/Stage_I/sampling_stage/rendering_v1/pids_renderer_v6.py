#!/usr/bin/env python3
"""
PIDS Renderer v6 - 重新設計
專注於正確的偏振渲染

關鍵修正：
1. 正確使用 Mitsuba 3 偏振模式
2. 正確解析 Stokes 向量
3. 深度圖玻璃用不透明材質
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys
import numpy as np
from pathlib import Path
import cv2

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
    },
    'light': {
        'intensity': 50000.0,
        'size': 150.0,
    }
}

# ============================================================
# OBJ 解析
# ============================================================
def parse_obj(filepath):
    """解析 OBJ 檔案，按物件名稱分組"""
    objects = {}
    current_obj = 'default'
    vertices = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            if parts[0] == 'o':
                current_obj = parts[1] if len(parts) > 1 else 'unnamed'
                if current_obj not in objects:
                    objects[current_obj] = {'verts': [], 'faces': [], 'vert_offset': len(vertices)}
                print(f"    物件: {current_obj}")
                
            elif parts[0] == 'v':
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
                if current_obj in objects:
                    objects[current_obj]['verts'].append([x, y, z])
                    
            elif parts[0] == 'f':
                if current_obj not in objects:
                    objects[current_obj] = {'verts': [], 'faces': [], 'vert_offset': 0}
                
                face = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0])
                    # 轉換為該物件的局部索引
                    local_idx = idx - 1 - objects[current_obj]['vert_offset']
                    face.append(local_idx)
                
                if len(face) == 3:
                    objects[current_obj]['faces'].append(face)
                elif len(face) == 4:
                    objects[current_obj]['faces'].append([face[0], face[1], face[2]])
                    objects[current_obj]['faces'].append([face[0], face[2], face[3]])
    
    # 打印每個物件的範圍
    for name, data in objects.items():
        if data['verts']:
            verts = np.array(data['verts'])
            print(f"      {name}: {len(data['verts'])} verts, {len(data['faces'])} faces")
            print(f"        範圍: X[{verts[:,0].min():.1f}, {verts[:,0].max():.1f}] "
                  f"Y[{verts[:,1].min():.1f}, {verts[:,1].max():.1f}] "
                  f"Z[{verts[:,2].min():.1f}, {verts[:,2].max():.1f}]")
    
    return objects


def write_ply(path, verts, faces):
    """寫入 PLY (二進位格式，更快)"""
    import struct
    
    with open(path, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(verts)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""
        f.write(header.encode('ascii'))
        
        # Vertices
        for v in verts:
            f.write(struct.pack('<fff', v[0], v[1], v[2]))
        
        # Faces
        for face in faces:
            f.write(struct.pack('<B', 3))  # 3 vertices per face
            f.write(struct.pack('<iii', face[0], face[1], face[2]))


# ============================================================
# Mitsuba 初始化
# ============================================================
def get_mitsuba_polarized():
    """初始化 Mitsuba 偏振模式"""
    import mitsuba as mi
    
    # 必須使用偏振變體
    try:
        mi.set_variant('cuda_ad_spectral_polarized')
        print(f"  Mitsuba 變體: {mi.variant()} (GPU 偏振)")
    except:
        try:
            mi.set_variant('scalar_spectral_polarized')
            print(f"  Mitsuba 變體: {mi.variant()} (CPU 偏振)")
        except Exception as e:
            print(f"  錯誤: 無法啟用偏振變體: {e}")
            print(f"  可用變體: {mi.variants()}")
            raise
    
    return mi


def get_mitsuba_rgb():
    """初始化 Mitsuba RGB 模式 (深度用)"""
    import mitsuba as mi
    
    try:
        mi.set_variant('cuda_ad_rgb')
    except:
        mi.set_variant('scalar_rgb')
    
    return mi


# ============================================================
# 偏振場景建立
# ============================================================
def build_polarized_scene(mi, obj_path, temp_dir):
    """
    建立偏振渲染場景
    
    重點：
    1. 使用 'stokes' integrator
    2. 光源前有線性偏振片
    3. 玻璃用 pplastic (會產生偏振反射)
    """
    
    print(f"  解析 OBJ: {obj_path}")
    objects = parse_obj(obj_path)
    
    # 相機參數
    cam_y = -400  # 在 chamber 內
    cam_z = 20    # 貼地
    focus_y = -750  # 傢俱區
    half_baseline = CONFIG['camera']['baseline_mm'] / 2.0
    
    # 光源參數
    light_z = 250
    light_size = CONFIG['light']['size']
    
    # 場景字典
    scene_dict = {
        'type': 'scene',
        
        # Stokes integrator - 關鍵！
        'integrator': {
            'type': 'stokes',
            'nested': {
                'type': 'path',
                'max_depth': 8,
            }
        },
    }
    
    # -----------------------------------------------------------
    # 加入物件
    # -----------------------------------------------------------
    for name, data in objects.items():
        if not data['verts'] or not data['faces']:
            continue
        
        verts = data['verts']
        faces = data['faces']
        
        # 驗證面索引
        max_idx = max(max(f) for f in faces) if faces else 0
        if max_idx >= len(verts):
            print(f"    跳過 {name}: 面索引超出範圍")
            continue
        
        safe_name = name.replace(' ', '_').replace('.', '_').replace('-', '_')
        ply_path = os.path.join(temp_dir, f"pol_{safe_name}.ply")
        write_ply(ply_path, verts, faces)
        
        # 材質設定
        name_lower = name.lower()
        if 'glass' in name_lower or 'window' in name_lower:
            # 玻璃：薄介電質
            bsdf = {
                'type': 'thindielectric',
                'int_ior': 1.5,
                'ext_ior': 1.0,
            }
            print(f"    玻璃: {name} (thindielectric)")
        else:
            # 其他物件：漫反射 (類似木頭)
            bsdf = {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.7, 0.7, 0.7]}
            }
        
        scene_dict[f'mesh_{safe_name}'] = {
            'type': 'ply',
            'filename': ply_path,
            'bsdf': bsdf,
        }
    
    # -----------------------------------------------------------
    # 光源 (帶偏振片)
    # -----------------------------------------------------------
    # 使用 spot 光源配合偏振片
    scene_dict['main_light'] = {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, cam_y, light_z],
            target=[0, focus_y, cam_z],
            up=[0, 0, 1]
        ) @ mi.ScalarTransform4f.scale([light_size, light_size, 1]),
        'bsdf': {
            'type': 'polarizer',  # 線性偏振片
            'theta': 0,  # 0 度 = 水平偏振
        },
        'emitter': {
            'type': 'area',
            'radiance': {'type': 'spectrum', 'value': CONFIG['light']['intensity']}
        }
    }
    
    # 環境光 (非偏振，填補暗區)
    scene_dict['envlight'] = {
        'type': 'constant',
        'radiance': {'type': 'spectrum', 'value': 200.0}
    }
    
    # -----------------------------------------------------------
    # 相機
    # -----------------------------------------------------------
    film_config = {
        'type': 'hdrfilm',
        'width': CONFIG['camera']['resolution'][0],
        'height': CONFIG['camera']['resolution'][1],
        'pixel_format': 'luminance',  # 灰階 Stokes
        'component_format': 'float32',
    }
    
    sampler_config = {
        'type': 'independent',
        'sample_count': CONFIG['render']['samples'],
    }
    
    # 左眼
    scene_dict['sensor_left'] = {
        'type': 'perspective',
        'fov': CONFIG['camera']['fov'],
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[-half_baseline, cam_y, cam_z],
            target=[0, focus_y, cam_z],
            up=[0, 0, 1]
        ),
        'film': film_config.copy(),
        'sampler': sampler_config.copy(),
    }
    
    # 右眼
    scene_dict['sensor_right'] = {
        'type': 'perspective',
        'fov': CONFIG['camera']['fov'],
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[half_baseline, cam_y, cam_z],
            target=[0, focus_y, cam_z],
            up=[0, 0, 1]
        ),
        'film': film_config.copy(),
        'sampler': sampler_config.copy(),
    }
    
    return scene_dict


# ============================================================
# 深度場景建立
# ============================================================
def build_depth_scene(mi, obj_path, temp_dir):
    """建立深度渲染場景"""
    
    print(f"  解析 OBJ: {obj_path}")
    objects = parse_obj(obj_path)
    
    # 相機參數 (與偏振相同)
    cam_y = -400
    cam_z = 20
    focus_y = -750
    half_baseline = CONFIG['camera']['baseline_mm'] / 2.0
    
    # 過濾：排除牆壁和天花板
    exclude = ['wall', 'ceiling']
    
    scene_dict = {
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
                target=[0, focus_y, cam_z],
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
    
    # 加入物件 (全部用不透明材質)
    for name, data in objects.items():
        if not data['verts'] or not data['faces']:
            continue
        
        # 過濾
        name_lower = name.lower()
        if any(kw in name_lower for kw in exclude):
            continue
        
        verts = data['verts']
        faces = data['faces']
        
        max_idx = max(max(f) for f in faces) if faces else 0
        if max_idx >= len(verts):
            continue
        
        safe_name = name.replace(' ', '_').replace('.', '_').replace('-', '_')
        ply_path = os.path.join(temp_dir, f"depth_{safe_name}.ply")
        write_ply(ply_path, verts, faces)
        
        # 深度用不透明材質
        scene_dict[f'mesh_{safe_name}'] = {
            'type': 'ply',
            'filename': ply_path,
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}}
        }
        print(f"    深度物件: {name}")
    
    return scene_dict


# ============================================================
# Stokes 處理
# ============================================================
def process_stokes_to_polarized(img_data, analyzer_angle_deg):
    """
    從 Stokes 向量計算通過偏振片後的強度
    
    Mitsuba spectral_polarized + luminance 輸出：
    - 4 通道: S0, S1, S2, S3 (灰階 Stokes)
    
    I(θ) = 0.5 * (S0 + S1*cos(2θ) + S2*sin(2θ))
    """
    data = np.array(img_data, dtype=np.float32)
    h, w = data.shape[:2]
    c = data.shape[2] if data.ndim == 3 else 1
    
    print(f"    Stokes 數據: shape={data.shape}")
    
    # 打印通道資訊
    for i in range(min(c, 8)):
        ch = data[:, :, i] if data.ndim == 3 else data
        print(f"      ch{i}: min={ch.min():.2f}, max={ch.max():.2f}, mean={ch.mean():.2f}")
    
    # 根據通道數解析 Stokes
    if c == 4:
        # 標準 luminance Stokes: S0, S1, S2, S3
        S0 = data[:, :, 0]
        S1 = data[:, :, 1]
        S2 = data[:, :, 2]
        print(f"    Stokes (4通道): S0=[{S0.min():.2f}, {S0.max():.2f}], S1=[{S1.min():.2f}, {S1.max():.2f}], S2=[{S2.min():.2f}, {S2.max():.2f}]")
        
    elif c == 1 or data.ndim == 2:
        # 只有單通道，沒有偏振
        print(f"    警告: 只有 1 通道，無偏振信息")
        S0 = data[:, :, 0] if data.ndim == 3 else data
        S1 = np.zeros_like(S0)
        S2 = np.zeros_like(S0)
        
    else:
        # 其他格式，嘗試用前 4 通道
        print(f"    使用前 4 通道作為 Stokes ({c} 通道)")
        S0 = data[:, :, 0]
        S1 = data[:, :, 1] if c > 1 else np.zeros_like(S0)
        S2 = data[:, :, 2] if c > 2 else np.zeros_like(S0)
        print(f"    Stokes: S0=[{S0.min():.2f}, {S0.max():.2f}], S1=[{S1.min():.2f}, {S1.max():.2f}], S2=[{S2.min():.2f}, {S2.max():.2f}]")
    
    # 計算通過偏振片後的強度
    theta = np.radians(analyzer_angle_deg)
    cos2 = np.cos(2 * theta)
    sin2 = np.sin(2 * theta)
    
    I_out = 0.5 * (S0 + S1 * cos2 + S2 * sin2)
    
    return np.maximum(I_out, 0)


# ============================================================
# 主渲染函數
# ============================================================
def render_pids(obj_path, output_dir):
    """渲染 PIDS 數據集"""
    
    obj_path = Path(obj_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_dir / '_temp'
    temp_dir.mkdir(exist_ok=True)
    
    name = obj_path.stem
    
    print(f"\n{'='*60}")
    print(f"PIDS 渲染: {name}")
    print(f"{'='*60}")
    
    # ========================================
    # 1. 偏振渲染
    # ========================================
    print("\n[1] 偏振渲染")
    mi = get_mitsuba_polarized()
    
    scene_dict = build_polarized_scene(mi, str(obj_path), str(temp_dir))
    scene = mi.load_dict(scene_dict)
    
    # 渲染左眼
    print("  渲染左眼...")
    img_left = mi.render(scene, sensor=0)
    
    # 渲染右眼
    print("  渲染右眼...")
    img_right = mi.render(scene, sensor=1)
    
    # 處理 Stokes
    print("  處理 Stokes...")
    I_parallel = process_stokes_to_polarized(img_left, 0)    # 0° = I∥
    I_cross = process_stokes_to_polarized(img_right, 90)     # 90° = I⊥
    
    print(f"  I∥ 範圍: [{I_parallel.min():.2f}, {I_parallel.max():.2f}]")
    print(f"  I⊥ 範圍: [{I_cross.min():.2f}, {I_cross.max():.2f}]")
    
    # 儲存 EXR (Linear)
    cv2.imwrite(str(output_dir / f"{name}_I_parallel.exr"), I_parallel.astype(np.float32))
    cv2.imwrite(str(output_dir / f"{name}_I_cross.exr"), I_cross.astype(np.float32))
    print("  已儲存 EXR")
    
    # 儲存 PNG (預覽)
    all_vals = np.concatenate([I_parallel.flatten(), I_cross.flatten()])
    p99 = np.percentile(all_vals[all_vals > 0], 99) if np.any(all_vals > 0) else 1.0
    
    I_par_norm = np.clip(I_parallel / p99, 0, 1)
    I_cross_norm = np.clip(I_cross / p99, 0, 1)
    
    # Gamma 校正
    I_par_disp = np.power(I_par_norm, 1/2.2)
    I_cross_disp = np.power(I_cross_norm, 1/2.2)
    
    cv2.imwrite(str(output_dir / f"{name}_I_parallel.png"), (I_par_disp * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / f"{name}_I_cross.png"), (I_cross_disp * 255).astype(np.uint8))
    print("  已儲存 PNG")
    
    # ========================================
    # 2. 深度渲染
    # ========================================
    print("\n[2] 深度渲染")
    mi_rgb = get_mitsuba_rgb()
    
    depth_scene_dict = build_depth_scene(mi_rgb, str(obj_path), str(temp_dir))
    depth_scene = mi_rgb.load_dict(depth_scene_dict)
    
    print("  渲染深度...")
    depth_img = mi_rgb.render(depth_scene)
    depth_data = np.array(depth_img)
    
    # 提取深度通道 (通常是 R 或第 4 通道)
    if depth_data.ndim == 3:
        if depth_data.shape[2] >= 4:
            depth = depth_data[:, :, 3]  # Alpha 通道可能是深度
        else:
            depth = depth_data[:, :, 0]  # R 通道
    else:
        depth = depth_data
    
    # 儲存深度 EXR
    cv2.imwrite(str(output_dir / f"{name}_depth.exr"), depth.astype(np.float32))
    
    # 深度可視化
    valid_depth = depth[depth > 0]
    if len(valid_depth) > 0:
        d_min, d_max = np.percentile(valid_depth, [1, 99])
        depth_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-6), 0, 1)
        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(output_dir / f"{name}_depth.png"), depth_color)
    
    print("  已儲存深度")
    
    print(f"\n完成！輸出於: {output_dir}")
    
    return {
        'I_parallel': I_parallel,
        'I_cross': I_cross,
        'depth': depth,
    }


# ============================================================
# 批次處理
# ============================================================
def process_batch(scene_dir, output_dir, max_scenes=None):
    """批次處理多個場景"""
    
    scene_dir = Path(scene_dir)
    output_dir = Path(output_dir)
    
    obj_files = sorted(scene_dir.glob("*.obj"))
    
    if max_scenes:
        obj_files = obj_files[:max_scenes]
    
    print(f"找到 {len(obj_files)} 個場景")
    
    for i, obj_path in enumerate(obj_files):
        print(f"\n[{i+1}/{len(obj_files)}] 處理 {obj_path.name}")
        try:
            render_pids(obj_path, output_dir)
        except Exception as e:
            print(f"  錯誤: {e}")
            import traceback
            traceback.print_exc()


# ============================================================
# 主程式
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PIDS Renderer v6")
    parser.add_argument("input", help="OBJ 檔案或目錄")
    parser.add_argument("-o", "--output", default="./output", help="輸出目錄")
    parser.add_argument("-n", "--max-scenes", type=int, help="最大場景數")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        render_pids(input_path, args.output)
    elif input_path.is_dir():
        process_batch(input_path, args.output, args.max_scenes)
    else:
        print(f"錯誤: {input_path} 不存在")
        sys.exit(1)

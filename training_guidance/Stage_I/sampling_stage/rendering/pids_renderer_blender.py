"""
PIDS Mitsuba 3 偏振渲染器 (修正版)
==========================================
配合 Blender v16.0 分層匯出腳本使用。
修正項目:
1. 自動縮放 0.001 以解決全黑問題。
2. 支援 _glass.obj 與 _diffuse.obj 分層載入，解決 shape_group 報錯。
"""

import argparse
import sys
from pathlib import Path
import mitsuba as mi

# 1. 初始化 Mitsuba (必須最先執行)
try:
    mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
except Exception as e:
    print(f"[警告] 無法設定 GPU 模式，嘗試使用 CPU: {e}")
    mi.set_variant('scalar_rgb')

# 2. 導入配置 (pids_config.py 需在同一目錄)
try:
    from pids_config import (
        CAMERA_CONFIG, 
        LIGHT_CONFIG, 
        RENDER_CONFIG
    )
except ImportError:
    print("錯誤: 找不到 pids_config.py，請確保它在同一目錄下。")
    sys.exit(1)

def create_pids_scene(glass_obj_path, diffuse_obj_path):
    """
    建立 Mitsuba 場景字典
    """
    
    # --- 基礎場景配置 ---
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': RENDER_CONFIG['max_depth'],
        },
        'sensor': {
            'type': 'perspective',
            # 使用 Config 中的 FOV
            'fov': CAMERA_CONFIG['fov_horizontal_deg'],
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 0, 0],    # 相機在原點
                target=[0, 1, 0],    # 看向 Y 軸正向
                up=[0, 0, 1]         # Z 軸向上
            ),
            'film': {
                'type': 'hdrfilm',
                'width': CAMERA_CONFIG['output_resolution'][0],
                'height': CAMERA_CONFIG['output_resolution'][1],
                'pixel_format': 'rgba',
                'rfilter': {'type': 'gaussian'},
            },
            'sampler': {
                'type': 'independent',
                'sample_count': RENDER_CONFIG['spp'],
            },
        },
        # --- 光源配置 ---
        'light_source': {
            'type': 'spot',
            'intensity': {
                'type': 'spectrum',
                'value': LIGHT_CONFIG['intensity_watts_sr'],
            },
            # 光源位置：左上方打向中心 (單位: 公尺)
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[-0.15, 0.0, 0.3], 
                target=[0.0, 0.6, 0.0],
                up=[0, 0, 1]
            ),
            'cutoff_angle': 60.0,
        }
    }

    # --- 載入玻璃物件 (如果存在) ---
    if glass_obj_path and glass_obj_path.exists():
        scene_dict['glass_objects'] = {
            'type': 'obj',
            'filename': str(glass_obj_path),
            # [關鍵修正] 縮放 0.001 (mm -> m)
            'to_world': mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]),
            # 強制指定為玻璃材質
            'bsdf': {
                'type': 'dielectric',
                'int_ior': 1.5,
                'ext_ior': 1.0,
            }
        }

    # --- 載入傢俱物件 (如果存在) ---
    if diffuse_obj_path and diffuse_obj_path.exists():
        scene_dict['diffuse_objects'] = {
            'type': 'obj',
            'filename': str(diffuse_obj_path),
            # [關鍵修正] 縮放 0.001 (mm -> m)
            'to_world': mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]),
            # 強制指定為漫反射材質 (木頭色)
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.5, 0.4, 0.3]
                }
            }
        }
        
    return scene_dict

def render_scene_files(scene_name, glass_path, diffuse_path, output_dir):
    print(f"正在渲染: {scene_name} ...")
    
    # 建立場景
    scene_dict = create_pids_scene(glass_path, diffuse_path)
    
    try:
        scene = mi.load_dict(scene_dict)
        # 執行渲染
        img = mi.render(scene)
        
        # 儲存 EXR (數據用)
        out_exr = output_dir / f"{scene_name}.exr"
        mi.util.write_bitmap(str(out_exr), img)
        
        # 儲存 PNG (預覽用) - 自動做 Tone mapping 轉成 sRGB
        out_png = output_dir / f"{scene_name}.png"
        mi.util.write_bitmap(str(out_png), img)
        
        print(f"  -> 完成: {out_png}")
        
    except Exception as e:
        print(f"  -> 渲染失敗: {e}")

def main():
    parser = argparse.ArgumentParser(description='PIDS Renderer (v16 Compatible)')
    parser.add_argument('--scene_dir', '-d', type=str, required=True, help='OBJ 檔案所在目錄')
    parser.add_argument('--output', '-o', type=str, default='./rendered_output', help='輸出目錄')
    args = parser.parse_args()
    
    input_dir = Path(args.scene_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 自動掃描並配對場景
    scenes = {}
    print(f"掃描目錄: {input_dir}")
    
    for f in input_dir.glob("scene_*.obj"):
        # 解析檔名: scene_0001_glass.obj -> scene_0001
        parts = f.stem.split('_')
        # 格式檢查: 至少要有 scene 和 編號
        if len(parts) >= 2 and parts[0] == 'scene' and parts[1].isdigit():
            scene_id = f"scene_{parts[1]}"
            if scene_id not in scenes:
                scenes[scene_id] = {'glass': None, 'diffuse': None}
            
            if 'glass' in f.name:
                scenes[scene_id]['glass'] = f
            elif 'diffuse' in f.name:
                scenes[scene_id]['diffuse'] = f
    
    if not scenes:
        print("找不到符合格式 (scene_XXXX_glass.obj / scene_XXXX_diffuse.obj) 的檔案。")
        print("請確認您是否執行了 v16.0 Blender 腳本？")
        return

    print(f"共發現 {len(scenes)} 組場景，開始批次渲染...")
    
    for scene_name, paths in sorted(scenes.items()):
        render_scene_files(scene_name, paths['glass'], paths['diffuse'], output_dir)
        
    print("\n所有渲染作業完成！")

if __name__ == "__main__":
    main()

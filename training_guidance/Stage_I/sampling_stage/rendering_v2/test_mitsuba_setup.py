"""
PIDS Mitsuba 3 安裝測試腳本
============================

功能:
1. 檢查 Mitsuba 3 安裝
2. 檢查 CUDA 支援
3. 執行簡單渲染測試
4. 生成測試場景驗證座標系統

使用方式:
    python test_mitsuba_setup.py
"""

import sys
import os


def check_mitsuba_installation():
    """檢查 Mitsuba 3 是否正確安裝"""
    print("\n" + "="*60)
    print("Mitsuba 3 安裝檢查")
    print("="*60)
    
    try:
        import mitsuba as mi
        print(f"✓ Mitsuba 已安裝")
        print(f"  版本: {mi.__version__}")
    except ImportError as e:
        print(f"✗ Mitsuba 未安裝或安裝失敗")
        print(f"  錯誤: {e}")
        print(f"\n  安裝方式: pip install mitsuba")
        return False
    
    try:
        import drjit as dr
        print(f"✓ DrJit 已安裝")
        print(f"  版本: {dr.__version__}")
    except ImportError:
        print(f"✗ DrJit 未安裝")
        return False
    
    return True


def check_variants():
    """檢查可用的 Mitsuba variants"""
    import mitsuba as mi
    
    print("\n" + "-"*40)
    print("可用的 Mitsuba Variants:")
    print("-"*40)
    
    variants = mi.variants()
    
    # 分類 variants
    cuda_variants = [v for v in variants if 'cuda' in v]
    llvm_variants = [v for v in variants if 'llvm' in v]
    scalar_variants = [v for v in variants if 'scalar' in v]
    
    print(f"\nCUDA variants ({len(cuda_variants)}):")
    for v in cuda_variants:
        print(f"  - {v}")
    
    print(f"\nLLVM variants ({len(llvm_variants)}):")
    for v in llvm_variants:
        print(f"  - {v}")
    
    print(f"\nScalar variants ({len(scalar_variants)}):")
    for v in scalar_variants:
        print(f"  - {v}")
    
    # 檢查 CUDA 支援
    has_cuda = len(cuda_variants) > 0
    
    if has_cuda:
        print(f"\n✓ CUDA 加速可用")
    else:
        print(f"\n✗ CUDA 加速不可用")
        print(f"  可能原因:")
        print(f"  - 未安裝 CUDA toolkit")
        print(f"  - 未安裝 CUDA 版本的 Mitsuba")
        print(f"  - GPU 驅動問題")
    
    return cuda_variants, llvm_variants, scalar_variants


def select_best_variant(cuda_variants, llvm_variants, scalar_variants):
    """選擇最佳 variant"""
    import mitsuba as mi
    
    # 優先順序: cuda_ad_rgb > cuda_rgb > llvm_ad_rgb > scalar_rgb
    preferred = ['cuda_ad_rgb', 'cuda_rgb', 'llvm_ad_rgb', 'llvm_rgb', 'scalar_rgb']
    
    all_variants = cuda_variants + llvm_variants + scalar_variants
    
    for v in preferred:
        if v in all_variants:
            mi.set_variant(v)
            print(f"\n選擇 variant: {v}")
            return v
    
    # 使用第一個可用的
    if all_variants:
        mi.set_variant(all_variants[0])
        print(f"\n選擇 variant: {all_variants[0]} (備選)")
        return all_variants[0]
    
    print("\n✗ 沒有可用的 variant!")
    return None


def test_simple_render():
    """執行簡單渲染測試"""
    import mitsuba as mi
    import numpy as np
    
    print("\n" + "="*60)
    print("渲染測試")
    print("="*60)
    
    # 建立簡單場景: 球體 + 地面 + 光源
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 4,
        },
        'sensor': {
            'type': 'perspective',
            'fov': 45,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 2, 5],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': 320,
                'height': 240,
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 16,
            },
        },
        # 球體
        'sphere': {
            'type': 'sphere',
            'center': [0, 0.5, 0],
            'radius': 0.5,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.8, 0.2, 0.2]},
            },
        },
        # 地面
        'ground': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 0, 0]) @ mi.ScalarTransform4f.scale([5, 5, 1]) @ mi.ScalarTransform4f.rotate([1, 0, 0], -90),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]},
            },
        },
        # 光源
        'light': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 3, 0]) @ mi.ScalarTransform4f.scale([2, 2, 1]) @ mi.ScalarTransform4f.rotate([1, 0, 0], 90),
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [10, 10, 10]},
            },
        },
    }
    
    print("\n建立測試場景...")
    
    try:
        scene = mi.load_dict(scene_dict)
        print("✓ 場景建立成功")
    except Exception as e:
        print(f"✗ 場景建立失敗: {e}")
        return False
    
    print("執行渲染...")
    
    try:
        import time
        start = time.time()
        image = mi.render(scene)
        elapsed = time.time() - start
        print(f"✓ 渲染成功 ({elapsed:.2f}s)")
    except Exception as e:
        print(f"✗ 渲染失敗: {e}")
        return False
    
    # 檢查輸出
    img_np = np.array(image)
    print(f"\n輸出資訊:")
    print(f"  形狀: {img_np.shape}")
    print(f"  範圍: [{img_np.min():.4f}, {img_np.max():.4f}]")
    print(f"  平均值: {img_np.mean():.4f}")
    
    # 儲存測試圖
    try:
        output_path = "test_render.png"
        
        # 色調映射
        img_np = np.clip(img_np, 0, None)
        img_np = img_np / (1 + img_np)
        img_np = np.power(img_np, 1/2.2)
        img_np = (img_np * 255).astype(np.uint8)
        
        from PIL import Image
        Image.fromarray(img_np).save(output_path)
        print(f"\n✓ 測試圖已儲存: {output_path}")
    except ImportError:
        print("\n[警告] PIL 未安裝，無法儲存測試圖")
        print("  安裝方式: pip install Pillow")
    except Exception as e:
        print(f"\n[警告] 無法儲存測試圖: {e}")
    
    return True


def test_glass_material():
    """測試玻璃材質渲染"""
    import mitsuba as mi
    import numpy as np
    
    print("\n" + "="*60)
    print("玻璃材質測試")
    print("="*60)
    
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 8,
        },
        'sensor': {
            'type': 'perspective',
            'fov': 45,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 1.5, 4],
                target=[0, 0.5, 0],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': 320,
                'height': 240,
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 32,
            },
        },
        # 玻璃球
        'glass_sphere': {
            'type': 'sphere',
            'center': [0, 0.5, 0],
            'radius': 0.5,
            'bsdf': {
                'type': 'dielectric',
                'int_ior': 1.5,
                'ext_ior': 1.0,
            },
        },
        # 背景牆
        'back_wall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 1, -2]) @ mi.ScalarTransform4f.scale([4, 2, 1]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'checkerboard',
                    'color0': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]},
                    'color1': {'type': 'rgb', 'value': [0.2, 0.2, 0.2]},
                    'to_uv': mi.ScalarTransform4f.scale([8, 4, 1]),
                },
            },
        },
        # 地面
        'ground': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.scale([4, 4, 1]) @ mi.ScalarTransform4f.rotate([1, 0, 0], -90),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]},
            },
        },
        # 光源
        'light': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([1, 3, 1]) @ mi.ScalarTransform4f.scale([1, 1, 1]) @ mi.ScalarTransform4f.rotate([1, 0, 0], 90),
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [15, 15, 15]},
            },
        },
    }
    
    print("\n建立玻璃測試場景...")
    
    try:
        scene = mi.load_dict(scene_dict)
        print("✓ 場景建立成功")
        
        import time
        start = time.time()
        image = mi.render(scene)
        elapsed = time.time() - start
        
        print(f"✓ 渲染成功 ({elapsed:.2f}s)")
        
        # 儲存
        img_np = np.array(image)
        img_np = np.clip(img_np, 0, None)
        img_np = img_np / (1 + img_np)
        img_np = np.power(img_np, 1/2.2)
        img_np = (img_np * 255).astype(np.uint8)
        
        try:
            from PIL import Image
            Image.fromarray(img_np).save("test_glass.png")
            print(f"✓ 玻璃測試圖已儲存: test_glass.png")
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"✗ 玻璃材質測試失敗: {e}")
        return False


def test_coordinate_system():
    """測試座標系統轉換"""
    import mitsuba as mi
    
    print("\n" + "="*60)
    print("座標系統說明")
    print("="*60)
    
    print("""
    Blender 座標系 (OBJ 匯出後):
    ============================
    - X: 右
    - Y: 前 (深度方向，相機看向的方向)
    - Z: 上
    
    匯出設定: forward=-Y, up=Z
    
    Mitsuba 座標系:
    ===============
    - X: 右
    - Y: 上
    - Z: 前 (深度方向)
    
    轉換關係:
    =========
    Blender (X, Y, Z) → Mitsuba (X, Z, Y)
    
    或者在 OBJ 載入時使用 to_world 變換:
    mi.ScalarTransform4f.scale([1, 1, 1])  # 單位轉換
    
    Chamber 尺寸對應:
    =================
    Blender:
    - 前牆 Y = 350mm → Mitsuba Z = 350mm
    - 後牆 Y = 900mm → Mitsuba Z = 900mm
    - 高度 Z = 300mm → Mitsuba Y = 300mm
    - 寬度 X = 600mm → Mitsuba X = 600mm
    
    相機位置:
    =========
    - Blender: (0, 0, 150) 看向 +Y
    - Mitsuba: (0, 150, 0) 看向 +Z (mm)
    - Mitsuba: (0, 0.15, 0) 看向 +Z (meters)
    """)
    
    return True


def test_obj_loading():
    """測試 OBJ 載入"""
    import mitsuba as mi
    
    print("\n" + "="*60)
    print("OBJ 載入測試")
    print("="*60)
    
    # 建立一個簡單的測試 OBJ
    test_obj_content = """# Test OBJ file
# Simple cube

v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5

f 1 2 3 4
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""
    
    test_obj_path = "test_cube.obj"
    
    try:
        with open(test_obj_path, 'w') as f:
            f.write(test_obj_content)
        print(f"✓ 測試 OBJ 已建立: {test_obj_path}")
        
        # 嘗試載入
        scene_dict = {
            'type': 'scene',
            'integrator': {'type': 'path'},
            'mesh': {
                'type': 'obj',
                'filename': test_obj_path,
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]},
                },
            },
            'sensor': {
                'type': 'perspective',
                'fov': 45,
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=[2, 2, 2],
                    target=[0, 0, 0],
                    up=[0, 1, 0]
                ),
                'film': {'type': 'hdrfilm', 'width': 64, 'height': 64},
                'sampler': {'type': 'independent', 'sample_count': 4},
            },
            'light': {
                'type': 'constant',
                'radiance': {'type': 'rgb', 'value': [1, 1, 1]},
            },
        }
        
        scene = mi.load_dict(scene_dict)
        print("✓ OBJ 載入成功")
        
        # 清理
        os.remove(test_obj_path)
        
        return True
        
    except Exception as e:
        print(f"✗ OBJ 載入失敗: {e}")
        if os.path.exists(test_obj_path):
            os.remove(test_obj_path)
        return False


def run_all_tests():
    """執行所有測試"""
    print("\n" + "="*60)
    print("PIDS Mitsuba 3 環境測試")
    print("="*60)
    
    results = {}
    
    # 1. 檢查安裝
    results['installation'] = check_mitsuba_installation()
    if not results['installation']:
        print("\n[ERROR] Mitsuba 未正確安裝，無法繼續測試")
        return results
    
    # 2. 檢查 variants
    cuda, llvm, scalar = check_variants()
    results['cuda_available'] = len(cuda) > 0
    
    # 3. 選擇 variant
    variant = select_best_variant(cuda, llvm, scalar)
    if not variant:
        print("\n[ERROR] 無可用的 variant，無法繼續測試")
        return results
    
    # 4. 簡單渲染測試
    results['simple_render'] = test_simple_render()
    
    # 5. 玻璃材質測試
    results['glass_material'] = test_glass_material()
    
    # 6. OBJ 載入測試
    results['obj_loading'] = test_obj_loading()
    
    # 7. 座標系統說明
    test_coordinate_system()
    
    # 總結
    print("\n" + "="*60)
    print("測試總結")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ 通過" if passed else "✗ 失敗"
        print(f"  {test_name}: {status}")
    
    if all_passed:
        print(f"\n✓ 所有測試通過！環境已準備就緒。")
    else:
        print(f"\n✗ 部分測試失敗，請檢查錯誤訊息。")
    
    print("\n" + "="*60)
    
    return results


if __name__ == '__main__':
    run_all_tests()

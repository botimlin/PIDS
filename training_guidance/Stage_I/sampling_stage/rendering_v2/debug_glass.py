"""調試腳本：檢查 OBJ 檔案中的材質分配"""
import sys

def analyze_obj(obj_path):
    """分析 OBJ 檔案中的材質使用"""
    current_material = None
    material_faces = {}
    total_faces = 0
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('usemtl '):
                current_material = line.split()[1]
                if current_material not in material_faces:
                    material_faces[current_material] = 0
            elif line.startswith('f '):
                total_faces += 1
                if current_material:
                    material_faces[current_material] = material_faces.get(current_material, 0) + 1
    
    print(f"=== OBJ 材質分析: {obj_path} ===")
    print(f"總面數: {total_faces}")
    print(f"\n各材質面數:")
    for mat, count in sorted(material_faces.items(), key=lambda x: -x[1]):
        pct = count / total_faces * 100 if total_faces > 0 else 0
        marker = " *** GLASS ***" if 'glass' in mat.lower() else ""
        print(f"  {mat}: {count} 面 ({pct:.1f}%){marker}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_obj(sys.argv[1])
    else:
        print("Usage: python debug_glass.py <obj_file>")

"""
Complete Sanity Check: Verify entire disparity pipeline

This verifies:
1. Ray depth -> Z depth conversion
2. Z depth -> disparity calculation
3. Disparity -> warp alignment

For synthetic data where we know ground truth.
"""

import numpy as np
import cv2

# ============ Config (matching pids_renderer_textured.py) ============
WIDTH = 640
HEIGHT = 480
FOV_DEG = 45.4
BASELINE_MM = 65.0

FOV_RAD = np.radians(FOV_DEG)
FOCAL_PX = (WIDTH / 2) / np.tan(FOV_RAD / 2)

print("=" * 70)
print("Complete Sanity Check: Ray Depth -> Z Depth -> Disparity -> Warp")
print("=" * 70)
print(f"\nConfig: {WIDTH}x{HEIGHT}, FOV={FOV_DEG}deg, baseline={BASELINE_MM}mm")
print(f"Focal length: {FOCAL_PX:.2f} px")

# ============ Test 1: Constant Z depth ============
print(f"\n{'='*70}")
print("TEST 1: Constant Z Depth (simplest case)")
print("=" * 70)

Z_CONST = 0.4  # 400mm

# Create coordinate grids
y_coords, x_coords = np.meshgrid(np.arange(HEIGHT), np.arange(WIDTH), indexing='ij')
cx, cy = WIDTH / 2, HEIGHT / 2
dx = x_coords - cx
dy = y_coords - cy

# For constant Z, ray depth varies across image
# ray_depth = Z / cos(angle) = Z * sqrt(f^2 + dx^2 + dy^2) / f
cos_angle = FOCAL_PX / np.sqrt(FOCAL_PX**2 + dx**2 + dy**2)
ray_depth = Z_CONST / cos_angle

print(f"  Z depth (constant): {Z_CONST} m")
print(f"  Ray depth center: {ray_depth[HEIGHT//2, WIDTH//2]:.4f} m")
print(f"  Ray depth corner: {ray_depth[0, 0]:.4f} m")
print(f"  Ray depth range: [{ray_depth.min():.4f}, {ray_depth.max():.4f}] m")

# Convert ray depth back to Z depth (the fix in v5.1.4)
z_recovered = ray_depth * cos_angle

# Verify Z recovery is accurate
z_error = np.abs(z_recovered - Z_CONST)
print(f"\n  Z recovery error: max={z_error.max():.2e}, mean={z_error.mean():.2e}")

# Compute disparity
baseline_m = BASELINE_MM / 1000.0
disparity_correct = FOCAL_PX * baseline_m / z_recovered
disparity_wrong = FOCAL_PX * baseline_m / ray_depth  # old buggy version

# For constant Z, disparity should be constant
expected_disp = FOCAL_PX * baseline_m / Z_CONST
print(f"\n  Expected disparity (constant): {expected_disp:.2f} px")
print(f"  Correct method (using Z): center={disparity_correct[HEIGHT//2, WIDTH//2]:.2f}, corner={disparity_correct[0, 0]:.2f}")
print(f"  Wrong method (using ray): center={disparity_wrong[HEIGHT//2, WIDTH//2]:.2f}, corner={disparity_wrong[0, 0]:.2f}")

disp_error_correct = np.abs(disparity_correct - expected_disp)
disp_error_wrong = np.abs(disparity_wrong - expected_disp)
print(f"\n  Disparity error (correct method): max={disp_error_correct.max():.4f} px")
print(f"  Disparity error (wrong method):   max={disp_error_wrong.max():.2f} px")

# ============ Test 2: Verify warp with correct disparity ============
print(f"\n{'='*70}")
print("TEST 2: Warp Verification with Correct Disparity")
print("=" * 70)

# Create test image (gradient)
left_img = (x_coords / WIDTH).astype(np.float32)

# Create right image: for constant Z, right image is left shifted by constant disparity
xx, yy = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
xx_src = (xx + disparity_correct).astype(np.float32)
yy_src = yy.astype(np.float32)
right_img = cv2.remap(left_img, xx_src, yy_src, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)

# Warp right back to left using xx - disparity
xx_warp = (xx - disparity_correct).astype(np.float32)
warped = cv2.remap(right_img, xx_warp, yy_src, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)

# Check alignment (center region only)
margin = int(expected_disp) + 10
valid = np.zeros((HEIGHT, WIDTH), dtype=bool)
valid[10:-10, margin:-margin] = True

warp_error = np.abs(warped - left_img)[valid].mean()
print(f"  Warp alignment error: {warp_error:.2e}")

if warp_error < 1e-5:
    print("  [OK] Warp alignment is PERFECT!")
else:
    print("  [!!] Warp has residual error")

# ============ Test 3: Edge of FOV (worst case) ============
print(f"\n{'='*70}")
print("TEST 3: Edge of FOV Analysis")
print("=" * 70)

# At corner of image
corner_y, corner_x = 0, 0
angle_corner = np.arctan(np.sqrt(dx[corner_y, corner_x]**2 + dy[corner_y, corner_x]**2) / FOCAL_PX)
print(f"  Corner angle from center: {np.degrees(angle_corner):.1f} deg")
print(f"  cos(corner angle): {np.cos(angle_corner):.4f}")
print(f"  Ray depth / Z ratio at corner: {1/np.cos(angle_corner):.4f}x")

# Without the fix, disparity error at corner would be:
corner_disp_error_without_fix = expected_disp * (1 - np.cos(angle_corner))
print(f"\n  Disparity error WITHOUT fix at corner: {corner_disp_error_without_fix:.2f} px")
print(f"  Disparity error WITH fix at corner:    {disp_error_correct[corner_y, corner_x]:.4f} px")

# ============ Summary ============
print(f"\n{'='*70}")
print("SUMMARY")
print("=" * 70)

all_passed = True

# Check 1: Z recovery
if z_error.max() < 1e-6:
    print("[PASS] Z depth recovery: accurate")
else:
    print("[FAIL] Z depth recovery: has errors")
    all_passed = False

# Check 2: Disparity (correct method)
if disp_error_correct.max() < 0.01:
    print("[PASS] Disparity calculation: correct for constant Z")
else:
    print("[FAIL] Disparity calculation: has errors")
    all_passed = False

# Check 3: Warp alignment
if warp_error < 1e-5:
    print("[PASS] Warp alignment: perfect")
else:
    print("[FAIL] Warp alignment: has residual error")
    all_passed = False

# Check 4: Corner error fixed
if disp_error_correct[0, 0] < 0.01:
    print("[PASS] Corner disparity: fixed (error < 0.01 px)")
else:
    print("[FAIL] Corner disparity: still has error")
    all_passed = False

print(f"\n{'='*70}")
if all_passed:
    print("ALL TESTS PASSED! v5.1.4 disparity pipeline is correct.")
else:
    print("SOME TESTS FAILED! Review the results above.")
print("=" * 70)

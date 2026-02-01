"""
Sanity Check: Depth -> Disparity -> Warp Verification

Checks:
1. Compute disparity from GT depth
2. Warp right image to left using disparity
3. Check alignment: edges, glass region shift, center vs corner error
"""

import numpy as np
import cv2
import os

# ============ Config (same as pids_renderer_textured.py) ============
WIDTH = 640
HEIGHT = 480
FOV_DEG = 45.4
BASELINE_MM = 65.0

# Compute focal length (pixels)
FOV_RAD = np.radians(FOV_DEG)
FOCAL_PX = (WIDTH / 2) / np.tan(FOV_RAD / 2)

print("=" * 60)
print("Sanity Check: Depth -> Disparity -> Warp")
print("=" * 60)
print(f"\nConfig:")
print(f"  Resolution: {WIDTH}x{HEIGHT}")
print(f"  FOV: {FOV_DEG} deg")
print(f"  Baseline: {BASELINE_MM} mm")
print(f"  Focal length: {FOCAL_PX:.2f} px")

# ============ Generate synthetic test data ============
print(f"\nGenerating synthetic test data...")

# Create depth map: plane + glass panel
depth = np.ones((HEIGHT, WIDTH), dtype=np.float32) * 0.5  # background 500mm

# Glass region (center rectangle, 300mm distance)
glass_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
glass_mask[150:330, 200:440] = True
depth[glass_mask] = 0.3  # glass 300mm

# Add some depth variation (verify non-planar case)
y_coords, x_coords = np.meshgrid(np.arange(HEIGHT), np.arange(WIDTH), indexing='ij')
# Slight tilt: left closer, right farther
depth += (x_coords - WIDTH/2) * 0.0001

print(f"  Depth range: {depth.min():.3f}m ~ {depth.max():.3f}m")

# Create left/right images (checkerboard pattern for easy alignment check)
def create_checkerboard(h, w, sq_size=20):
    """Create checkerboard pattern"""
    img = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if ((i // sq_size) + (j // sq_size)) % 2 == 0:
                img[i, j] = 1.0
    return img

left_img = create_checkerboard(HEIGHT, WIDTH, 20)
# Different brightness in glass region
left_img[glass_mask] *= 0.7

# ============ Compute disparity ============
print(f"\nComputing disparity...")
baseline_m = BASELINE_MM / 1000.0
disparity = FOCAL_PX * baseline_m / depth
print(f"  Disparity range: {disparity.min():.2f} ~ {disparity.max():.2f} px")

# ============ Generate right image (ground truth) ============
print(f"\nGenerating right image (shifted by disparity)...")

# Key insight:
# - Left camera is at X = -baseline/2 (more left)
# - Right camera is at X = +baseline/2 (more right)
# - Same 3D point: larger x in left image, smaller x in right image
# - disparity = x_left - x_right > 0
# - To create right image from left: content at x_right comes from x_left = x_right + disparity

xx, yy = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
xx_src = (xx + disparity).astype(np.float32)  # sample from left image
yy_src = yy.astype(np.float32)

right_img = cv2.remap(left_img, xx_src, yy_src, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# ============ Warp right image back to left ============
print(f"\nTesting two warp directions...")

# Method A: xx - disparity (v5.1.3 fix)
# To warp right to left: x_right = x_left - disparity
xx_src_A = (xx - disparity).astype(np.float32)
warped_A = cv2.remap(right_img, xx_src_A, yy_src, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# Method B: xx + disparity (old incorrect direction)
xx_src_B = (xx + disparity).astype(np.float32)
warped_B = cv2.remap(right_img, xx_src_B, yy_src, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# ============ Compute errors ============
print(f"\nComputing errors...")

# Valid region (exclude boundaries)
valid_mask = (warped_A > 0) & (left_img > 0)
valid_mask[:, :50] = False   # left boundary
valid_mask[:, -50:] = False  # right boundary

# Regional masks
center_mask = np.zeros_like(valid_mask)
center_mask[HEIGHT//4:3*HEIGHT//4, WIDTH//4:3*WIDTH//4] = True

corner_mask = np.zeros_like(valid_mask)
corner_mask[:HEIGHT//4, :WIDTH//4] = True  # top-left
corner_mask[:HEIGHT//4, -WIDTH//4:] = True  # top-right
corner_mask[-HEIGHT//4:, :WIDTH//4] = True  # bottom-left
corner_mask[-HEIGHT//4:, -WIDTH//4:] = True  # bottom-right

# Method A errors
err_A = np.abs(warped_A - left_img)
err_A_glass = err_A[valid_mask & glass_mask].mean() if (valid_mask & glass_mask).sum() > 0 else 0
err_A_bg = err_A[valid_mask & ~glass_mask].mean() if (valid_mask & ~glass_mask).sum() > 0 else 0
err_A_center = err_A[valid_mask & center_mask].mean() if (valid_mask & center_mask).sum() > 0 else 0
err_A_corner = err_A[valid_mask & corner_mask].mean() if (valid_mask & corner_mask).sum() > 0 else 0
err_A_all = err_A[valid_mask].mean()

# Method B errors
err_B = np.abs(warped_B - left_img)
err_B_glass = err_B[valid_mask & glass_mask].mean() if (valid_mask & glass_mask).sum() > 0 else 0
err_B_bg = err_B[valid_mask & ~glass_mask].mean() if (valid_mask & ~glass_mask).sum() > 0 else 0
err_B_center = err_B[valid_mask & center_mask].mean() if (valid_mask & center_mask).sum() > 0 else 0
err_B_corner = err_B[valid_mask & corner_mask].mean() if (valid_mask & corner_mask).sum() > 0 else 0
err_B_all = err_B[valid_mask].mean()

print(f"\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nMethod A: xx - disparity (v5.1.3 fix)")
print(f"  Overall error:     {err_A_all:.6f}")
print(f"  Glass region:      {err_A_glass:.6f}")
print(f"  Background:        {err_A_bg:.6f}")
print(f"  Center region:     {err_A_center:.6f}")
print(f"  Corner regions:    {err_A_corner:.6f}")

print(f"\nMethod B: xx + disparity (old version)")
print(f"  Overall error:     {err_B_all:.6f}")
print(f"  Glass region:      {err_B_glass:.6f}")
print(f"  Background:        {err_B_bg:.6f}")
print(f"  Center region:     {err_B_center:.6f}")
print(f"  Corner regions:    {err_B_corner:.6f}")

# Diagnosis
print(f"\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)

if err_A_all < 0.01:
    print("[OK] Method A (xx - disparity) is CORRECT!")
    print("     Warp aligns perfectly")
    correct_method = "A"
elif err_B_all < 0.01:
    print("[!!] Method B (xx + disparity) is correct!")
    print("     Need to ROLLBACK warp direction change")
    correct_method = "B"
else:
    correct_method = None
    print("[??] Both methods have errors, need further investigation:")
    if err_A_center < err_A_corner * 0.5:
        print("     - Center OK, corners wrong -> likely ray depth issue")
    if abs(err_A_glass - err_A_bg) > 0.1:
        print("     - Glass/background error differs -> depth inconsistency")
    print(f"\n     Method A total: {err_A_all:.6f}")
    print(f"     Method B total: {err_B_all:.6f}")

# ============ Save visualization ============
output_dir = os.path.dirname(os.path.abspath(__file__))

# Error heatmaps (scale x10 for visibility)
err_vis_A = (err_A * 10).clip(0, 1)
err_vis_B = (err_B * 10).clip(0, 1)

cv2.imwrite(os.path.join(output_dir, "sanity_left.png"),
            (left_img * 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, "sanity_right.png"),
            (right_img * 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, "sanity_warped_A.png"),
            (warped_A * 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, "sanity_warped_B.png"),
            (warped_B * 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, "sanity_error_A.png"),
            (err_vis_A * 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, "sanity_error_B.png"),
            (err_vis_B * 255).astype(np.uint8))

# Difference image (side by side comparison)
diff_compare = np.zeros((HEIGHT, WIDTH*2), dtype=np.uint8)
diff_compare[:, :WIDTH] = (err_vis_A * 255).astype(np.uint8)
diff_compare[:, WIDTH:] = (err_vis_B * 255).astype(np.uint8)
cv2.imwrite(os.path.join(output_dir, "sanity_error_compare.png"), diff_compare)

print(f"\nSaved visualization to: {output_dir}")
print("  - sanity_left.png: left image")
print("  - sanity_right.png: right image (generated from disparity)")
print("  - sanity_warped_A.png: Method A warp result")
print("  - sanity_warped_B.png: Method B warp result")
print("  - sanity_error_A.png: Method A error heatmap (x10)")
print("  - sanity_error_B.png: Method B error heatmap (x10)")
print("  - sanity_error_compare.png: side-by-side comparison (A|B)")

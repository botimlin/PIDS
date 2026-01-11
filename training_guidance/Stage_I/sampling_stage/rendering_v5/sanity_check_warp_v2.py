"""
Sanity Check V2: Using gradient texture for more precise alignment check

Key insight: Use smooth gradients instead of checkerboard to minimize interpolation artifacts
"""

import numpy as np
import cv2
import os

# ============ Config ============
WIDTH = 640
HEIGHT = 480
FOV_DEG = 45.4
BASELINE_MM = 65.0

FOV_RAD = np.radians(FOV_DEG)
FOCAL_PX = (WIDTH / 2) / np.tan(FOV_RAD / 2)

print("=" * 60)
print("Sanity Check V2: Gradient Texture")
print("=" * 60)
print(f"\nFocal length: {FOCAL_PX:.2f} px")

# ============ Create test data ============
# Constant depth (simplest case)
depth = np.ones((HEIGHT, WIDTH), dtype=np.float32) * 0.4  # 400mm

# Smooth gradient texture (horizontal)
y_coords, x_coords = np.meshgrid(np.arange(HEIGHT), np.arange(WIDTH), indexing='ij')
left_img = (x_coords / WIDTH).astype(np.float32)  # 0 to 1 gradient

# Compute disparity
baseline_m = BASELINE_MM / 1000.0
disparity = FOCAL_PX * baseline_m / depth  # constant disparity for constant depth
disp_val = disparity[0, 0]
print(f"Constant disparity: {disp_val:.2f} px")

# ============ Generate right image ============
# For constant depth/disparity, right image is simply left shifted
# right_img[y, x] = left_img[y, x + disp]
xx, yy = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
xx_src = (xx + disparity).astype(np.float32)
yy_src = yy.astype(np.float32)

right_img = cv2.remap(left_img, xx_src, yy_src, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)

# ============ Warp back ============
# Method A: xx - disparity
xx_src_A = (xx - disparity).astype(np.float32)
warped_A = cv2.remap(right_img, xx_src_A, yy_src, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)

# Method B: xx + disparity
xx_src_B = (xx + disparity).astype(np.float32)
warped_B = cv2.remap(right_img, xx_src_B, yy_src, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)

# ============ Compute errors ============
# Center region only (avoid boundary)
margin = int(disp_val) + 10
valid = np.zeros((HEIGHT, WIDTH), dtype=bool)
valid[10:-10, margin:-margin] = True

err_A = np.abs(warped_A - left_img)
err_B = np.abs(warped_B - left_img)

err_A_mean = err_A[valid].mean()
err_B_mean = err_B[valid].mean()
err_A_max = err_A[valid].max()
err_B_max = err_B[valid].max()

print(f"\n" + "=" * 60)
print("RESULTS (Constant Depth = 400mm)")
print("=" * 60)

print(f"\nMethod A: xx - disparity")
print(f"  Mean error: {err_A_mean:.8f}")
print(f"  Max error:  {err_A_max:.8f}")

print(f"\nMethod B: xx + disparity")
print(f"  Mean error: {err_B_mean:.8f}")
print(f"  Max error:  {err_B_max:.8f}")

print(f"\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)

if err_A_mean < 1e-5:
    print("[OK] Method A (xx - disparity) is PERFECT!")
    print("     This confirms v5.1.3 warp direction is correct.")
elif err_B_mean < 1e-5:
    print("[!!] Method B (xx + disparity) is correct!")
    print("     Need to ROLLBACK warp direction change!")
else:
    print("[??] Neither method gives perfect alignment")
    print(f"     Method A: {err_A_mean:.8f}")
    print(f"     Method B: {err_B_mean:.8f}")

# ============ Also test with varying depth ============
print(f"\n" + "=" * 60)
print("Testing with varying depth...")
print("=" * 60)

# Depth varies from 300mm to 500mm
depth_var = 0.3 + 0.2 * (x_coords / WIDTH).astype(np.float32)
disparity_var = FOCAL_PX * baseline_m / depth_var

print(f"Disparity range: {disparity_var.min():.2f} ~ {disparity_var.max():.2f} px")

# Create gradient left image
left_var = (x_coords / WIDTH).astype(np.float32)

# Generate right image
xx_src_var = (xx + disparity_var).astype(np.float32)
right_var = cv2.remap(left_var, xx_src_var, yy_src, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)

# Warp back
xx_src_A_var = (xx - disparity_var).astype(np.float32)
warped_A_var = cv2.remap(right_var, xx_src_A_var, yy_src, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

xx_src_B_var = (xx + disparity_var).astype(np.float32)
warped_B_var = cv2.remap(right_var, xx_src_B_var, yy_src, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

# Errors
margin_var = int(disparity_var.max()) + 10
valid_var = np.zeros((HEIGHT, WIDTH), dtype=bool)
valid_var[10:-10, margin_var:-margin_var] = True

err_A_var = np.abs(warped_A_var - left_var)[valid_var].mean()
err_B_var = np.abs(warped_B_var - left_var)[valid_var].mean()

print(f"\nMethod A error: {err_A_var:.8f}")
print(f"Method B error: {err_B_var:.8f}")

if err_A_var < err_B_var:
    print("\n[OK] Method A (xx - disparity) is better for varying depth too!")
else:
    print("\n[!!] Method B is better for varying depth - need investigation!")

# ============ Save visualization ============
output_dir = os.path.dirname(os.path.abspath(__file__))

cv2.imwrite(os.path.join(output_dir, "sanity2_left.png"),
            (left_img * 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, "sanity2_right.png"),
            (right_img * 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, "sanity2_warped_A.png"),
            (warped_A * 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, "sanity2_warped_B.png"),
            (warped_B * 255).astype(np.uint8))

# Error visualization (scaled x1000 for visibility)
err_vis_A = (err_A * 1000).clip(0, 255).astype(np.uint8)
err_vis_B = (err_B * 1000).clip(0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(output_dir, "sanity2_error_A.png"), err_vis_A)
cv2.imwrite(os.path.join(output_dir, "sanity2_error_B.png"), err_vis_B)

print(f"\nSaved to: {output_dir}/sanity2_*.png")

"""
Sanity Check: Depth Camera Position

Verify that using left camera position for depth gives correct disparity.

Test:
1. Depth from left camera position
2. Depth from middle position (old behavior)
3. Compare alignment after warp
"""

import numpy as np
import cv2

# ============ Config ============
WIDTH = 640
HEIGHT = 480
FOV_DEG = 45.4
BASELINE_MM = 65.0

FOV_RAD = np.radians(FOV_DEG)
FOCAL_PX = (WIDTH / 2) / np.tan(FOV_RAD / 2)

print("=" * 60)
print("Sanity Check: Depth Camera Position")
print("=" * 60)

# Camera positions (in mm, world coordinates)
LEFT_CAM_X = -BASELINE_MM / 2   # -32.5mm
RIGHT_CAM_X = BASELINE_MM / 2   # +32.5mm
MID_CAM_X = 0                   # 0mm (old behavior)

print(f"\nCamera positions (X):")
print(f"  Left camera:   {LEFT_CAM_X:.1f} mm")
print(f"  Right camera:  {RIGHT_CAM_X:.1f} mm")
print(f"  Middle (old):  {MID_CAM_X:.1f} mm")

# ============ 3D Scene ============
# A point at (0, 400, 0) - directly in front, 400mm away
# For simplicity, assume cameras look along +Y axis

POINT_3D = np.array([0, 400, 0])  # mm
print(f"\n3D Point: {POINT_3D} mm")

# Project to each camera
def project(point_3d, cam_x, focal_px, width):
    """Project 3D point to 2D, returns x coordinate"""
    # Assuming camera looks along +Y
    # x_pixel = focal * (X - cam_x) / Y + width/2
    x_rel = point_3d[0] - cam_x
    y_depth = point_3d[1]
    x_pixel = focal_px * x_rel / y_depth + width / 2
    return x_pixel

x_left = project(POINT_3D, LEFT_CAM_X, FOCAL_PX, WIDTH)
x_right = project(POINT_3D, RIGHT_CAM_X, FOCAL_PX, WIDTH)
x_mid = project(POINT_3D, MID_CAM_X, FOCAL_PX, WIDTH)

print(f"\n2D Projection:")
print(f"  Left camera:   x = {x_left:.2f} px")
print(f"  Right camera:  x = {x_right:.2f} px")
print(f"  Middle camera: x = {x_mid:.2f} px")

# Disparity should be x_left - x_right
true_disparity = x_left - x_right
print(f"\nTrue disparity (x_left - x_right): {true_disparity:.2f} px")

# ============ Disparity from depth ============
# depth = distance along camera optical axis (Y in world)
# disparity = focal * baseline / depth

depth_from_left = POINT_3D[1]  # 400mm
depth_from_mid = POINT_3D[1]   # Same! (depth is along Y, not from camera position)

# But wait - "depth" in Mitsuba is ray depth (distance from camera to point)
# For a point at (0, 400, 0):
# - Ray depth from left cam at (-32.5, 0, 0): sqrt(32.5^2 + 400^2) = 401.3mm
# - Ray depth from mid cam at (0, 0, 0): sqrt(0^2 + 400^2) = 400mm
# - Ray depth from right cam at (32.5, 0, 0): sqrt(32.5^2 + 400^2) = 401.3mm

ray_depth_left = np.sqrt((POINT_3D[0] - LEFT_CAM_X)**2 + POINT_3D[1]**2 + POINT_3D[2]**2)
ray_depth_mid = np.sqrt((POINT_3D[0] - MID_CAM_X)**2 + POINT_3D[1]**2 + POINT_3D[2]**2)
ray_depth_right = np.sqrt((POINT_3D[0] - RIGHT_CAM_X)**2 + POINT_3D[1]**2 + POINT_3D[2]**2)

print(f"\nRay depths:")
print(f"  From left camera:  {ray_depth_left:.2f} mm")
print(f"  From middle:       {ray_depth_mid:.2f} mm")
print(f"  From right camera: {ray_depth_right:.2f} mm")

# ============ Key insight ============
# The disparity formula assumes "depth" = perpendicular distance to image plane
# disparity = focal * baseline / Z
# where Z = depth along optical axis (not ray depth!)

# For parallel optical axes looking at +Y:
# Z = Y component of 3D point (same for all cameras!)
Z_depth = POINT_3D[1]  # 400mm

disparity_formula = FOCAL_PX * BASELINE_MM / Z_depth / 1000 * 1000  # in pixels
print(f"\nDisparity from formula (f*b/Z):")
print(f"  Z = {Z_depth} mm")
print(f"  disparity = {FOCAL_PX:.2f} * {BASELINE_MM} / {Z_depth}")
print(f"  disparity = {disparity_formula:.2f} px")

print(f"\nComparison:")
print(f"  True disparity:    {true_disparity:.2f} px")
print(f"  Formula disparity: {disparity_formula:.2f} px")
print(f"  Match: {np.isclose(true_disparity, disparity_formula)}")

# ============ The real question ============
print(f"\n" + "=" * 60)
print("KEY INSIGHT")
print("=" * 60)
print("""
For parallel optical axes:
- Disparity formula uses Z (perpendicular depth), NOT ray depth
- Z is the same regardless of which camera measures it
- So depth camera position doesn't matter for Z!

BUT: Mitsuba's depth output is RAY DEPTH (distance to point)
- Ray depth from left camera != Z
- Ray depth from middle != Z (but closer)

Question: Does pids_renderer use ray depth or Z?
""")

# Check the formula used in renderer
print("=" * 60)
print("Checking what the renderer actually does...")
print("=" * 60)

# If using ray depth:
disp_from_ray_left = FOCAL_PX * BASELINE_MM / ray_depth_left
disp_from_ray_mid = FOCAL_PX * BASELINE_MM / ray_depth_mid

print(f"\nIf using ray depth:")
print(f"  From left cam ray depth:  {disp_from_ray_left:.2f} px")
print(f"  From middle ray depth:    {disp_from_ray_mid:.2f} px")
print(f"  True disparity:           {true_disparity:.2f} px")

# Error
err_ray_left = abs(disp_from_ray_left - true_disparity)
err_ray_mid = abs(disp_from_ray_mid - true_disparity)

print(f"\nErrors:")
print(f"  Using left cam ray depth:   {err_ray_left:.2f} px ({err_ray_left/true_disparity*100:.1f}%)")
print(f"  Using middle ray depth:     {err_ray_mid:.2f} px ({err_ray_mid/true_disparity*100:.1f}%)")

# For extreme case (point at edge of FOV)
print(f"\n" + "=" * 60)
print("Extreme case: Point at edge of FOV")
print("=" * 60)

# Point at left edge of image (x=0), depth 400mm
# In world: X = -Z * tan(FOV/2) = -400 * tan(22.7) = -167mm
POINT_EDGE = np.array([-167, 400, 0])
print(f"3D Point at left edge: {POINT_EDGE} mm")

x_left_edge = project(POINT_EDGE, LEFT_CAM_X, FOCAL_PX, WIDTH)
x_right_edge = project(POINT_EDGE, RIGHT_CAM_X, FOCAL_PX, WIDTH)
true_disp_edge = x_left_edge - x_right_edge

ray_left_edge = np.sqrt((POINT_EDGE[0] - LEFT_CAM_X)**2 + POINT_EDGE[1]**2)
ray_mid_edge = np.sqrt((POINT_EDGE[0] - MID_CAM_X)**2 + POINT_EDGE[1]**2)

disp_ray_left_edge = FOCAL_PX * BASELINE_MM / ray_left_edge
disp_ray_mid_edge = FOCAL_PX * BASELINE_MM / ray_mid_edge

print(f"  True disparity: {true_disp_edge:.2f} px")
print(f"  From left ray:  {disp_ray_left_edge:.2f} px (error: {abs(disp_ray_left_edge - true_disp_edge):.2f} px)")
print(f"  From mid ray:   {disp_ray_mid_edge:.2f} px (error: {abs(disp_ray_mid_edge - true_disp_edge):.2f} px)")

print(f"\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
The disparity formula d = f*b/Z uses Z (perpendicular depth).
If Mitsuba outputs RAY depth instead of Z:
  - Error depends on viewing angle
  - Center of image: minimal error (ray ~ Z)
  - Edge of image: larger error (ray > Z)

This is the "ray depth problem" mentioned in the task!

FIX: Convert ray depth to Z depth before computing disparity:
  Z = ray_depth * cos(angle_from_center)

Or use Mitsuba's 'distance' AOV with proper projection.
""")

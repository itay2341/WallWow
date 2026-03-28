"""Debug script to understand projection issues."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from geometry_engine import CornerGeometry, Camera, LeftWall

# Create geometry
geometry = CornerGeometry(
    wall_width=3.0,
    wall_height=2.5,
    camera_pos=(2.0, 2.0, 1.2),
    camera_fov=75.0,
    camera_resolution=(1920, 1080)
)

print("Camera:", geometry.camera)
print("Camera position:", geometry.camera.position)
print("Camera look_at:", geometry.camera.look_at)
print("Camera forward:", geometry.camera.forward)
print("\nLeft wall bounds:", geometry.left_wall.get_bounds())
print("Right wall bounds:", geometry.right_wall.get_bounds())
print("Ceiling bounds:", geometry.ceiling.get_bounds())

# Test a few points on left wall
print("\n" + "="*60)
print("Testing left wall points (X=0):")
print("="*60)

left_wall = geometry.left_wall
y_min, y_max, z_min, z_max = left_wall.get_bounds()

test_points_3d = np.array([
    [0.0, y_min, z_min],  # Bottom corner
    [0.0, y_max, z_min],  # Bottom edge
    [0.0, y_min, z_max],  # Top corner
    [0.0, y_max, z_max],  # Top edge
    [0.0, (y_min+y_max)/2, (z_min+z_max)/2],  # Center
])

print(f"\n3D points on left wall (Y range: {y_min} to {y_max}, Z range: {z_min} to {z_max}):")
for i, pt in enumerate(test_points_3d):
    print(f"  Point {i}: {pt}")
    
    # Project to image
    pixel = geometry.camera.project_to_image(pt)
    print(f"    -> Pixel: {pixel}")
    
    # Check camera space
    pt_cam = geometry.camera.world_to_camera(pt)
    print(f"    -> Camera space: {pt_cam} (Z>0 means in front: {pt_cam[2] > 0})")
    
print("\n" + "="*60)
print("Testing right wall points (Y=0):")
print("="*60)

right_wall = geometry.right_wall
x_min, x_max, z_min, z_max = right_wall.get_bounds()

test_points_3d = np.array([
    [x_min, 0.0, z_min],  # Bottom corner
    [x_max, 0.0, z_min],  # Bottom edge
    [x_min, 0.0, z_max],  # Top corner
    [x_max, 0.0, z_max],  # Top edge
    [(x_min+x_max)/2, 0.0, (z_min+z_max)/2],  # Center
])

print(f"\n3D points on right wall (X range: {x_min} to {x_max}, Z range: {z_min} to {z_max}):")
for i, pt in enumerate(test_points_3d):
    print(f"  Point {i}: {pt}")
    pixel = geometry.camera.project_to_image(pt)
    print(f"    -> Pixel: {pixel}")
    pt_cam = geometry.camera.world_to_camera(pt)
    print(f"    -> Camera space: {pt_cam} (Z>0: {pt_cam[2] > 0})")

print("\n" + "="*60)
print("Testing ceiling points (Z=2.5):")
print("="*60)

ceiling = geometry.ceiling
x_min, x_max, y_min, y_max = ceiling.get_bounds()

test_points_3d = np.array([
    [x_min, y_min, ceiling.ceiling_height],  # Corner
    [x_max, y_min, ceiling.ceiling_height],  # Edge
    [x_min, y_max, ceiling.ceiling_height],  # Edge
    [x_max, y_max, ceiling.ceiling_height],  # Corner at origin
    [(x_min+x_max)/2, (y_min+y_max)/2, ceiling.ceiling_height],  # Center
])

print(f"\n3D points on ceiling (X range: {x_min} to {x_max}, Y range: {y_min} to {y_max}, Z={ceiling.ceiling_height}):")
for i, pt in enumerate(test_points_3d):
    print(f"  Point {i}: {pt}")
    pixel = geometry.camera.project_to_image(pt)
    print(f"    -> Pixel: {pixel}")
    pt_cam = geometry.camera.world_to_camera(pt)
    print(f"    -> Camera space: {pt_cam} (Z>0: {pt_cam[2] > 0})")

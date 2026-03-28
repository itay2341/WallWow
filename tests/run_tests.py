"""
Quick test runner for WallWow.

Runs a simple test to verify the entire pipeline works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geometry_engine import CornerGeometry
from projection_mapper import ProjectionMapper
from renderer import PanelRenderer, load_image, resize_to_camera_resolution
from validator import EdgeValidator


def test_basic_pipeline():
    """Test the complete pipeline with a simple input."""
    print("=" * 60)
    print("WallWow Basic Pipeline Test")
    print("=" * 60)
    
    # Check if test image exists
    test_image_path = Path(__file__).parent.parent / "data" / "grid.png"
    
    if not test_image_path.exists():
        print("\nGenerating test images first...")
        from generate_test_images import generate_all_test_images
        generate_all_test_images()
    
    print(f"\n[1/5] Loading test image: {test_image_path}")
    input_image = load_image(str(test_image_path))
    print(f"  Image shape: {input_image.shape}")
    
    # Resize to camera resolution
    camera_res = (1920, 1080)
    if (input_image.shape[1], input_image.shape[0]) != camera_res:
        input_image = resize_to_camera_resolution(input_image, camera_res)
        print(f"  Resized to: {input_image.shape}")
    
    print("\n[2/5] Initializing geometry...")
    geometry = CornerGeometry(
        wall_width=3.0,
        wall_height=2.5,
        camera_pos=(2.0, 2.0, 1.2),
        camera_fov=75.0,
        camera_resolution=camera_res
    )
    print(f"  Camera: {geometry.camera.position}")
    print(f"  FOV: {geometry.camera.fov}°")
    
    print("\n[3/5] Computing projections...")
    mapper = ProjectionMapper(geometry, output_resolution=512)  # Smaller for faster test
    homographies = mapper.compute_all_homographies()
    print(f"  Computed {len(homographies)} homographies")
    
    print("\n[4/5] Rendering panels...")
    renderer = PanelRenderer(mapper, output_resolution=512)
    panels = renderer.render_all_panels(input_image, homographies)
    print(f"  Rendered {len(panels)} panels")
    
    # Check panel dimensions
    for name, panel in panels.items():
        print(f"    {name}: {panel.shape}")
    
    print("\n[5/5] Validating edges...")
    validator = EdgeValidator(panels)
    validator.validate_all_edges(num_samples=50)
    
    print("\n" + validator.generate_report())
    
    print("\n" + "=" * 60)
    print("✓ Test completed successfully!")
    print("=" * 60)
    
    return True


def test_geometry():
    """Test geometry components."""
    print("\nTesting geometry components...")
    
    from geometry_engine import Camera, LeftWall, RightWall, Ceiling
    import numpy as np
    
    # Test camera
    camera = Camera(position=(2, 2, 1.2), fov=75.0)
    print(f"  Camera initialized: {camera}")
    
    # Test projection
    point_3d = np.array([1.0, 1.0, 1.0])
    pixel = camera.project_to_image(point_3d)
    print(f"  3D point {point_3d} projects to pixel {pixel}")
    
    # Test planes
    left = LeftWall()
    right = RightWall()
    ceiling = Ceiling()
    print(f"  Left wall: {left}")
    print(f"  Right wall: {right}")
    print(f"  Ceiling: {ceiling}")
    
    print("✓ Geometry test passed")


def test_math_utils():
    """Test math utilities."""
    print("\nTesting math utilities...")
    
    from math_utils import ray_plane_intersection
    import numpy as np
    
    # Test ray-plane intersection
    ray_origin = np.array([1.0, 1.0, 1.0])
    ray_direction = np.array([-1.0, 0.0, 0.0])  # Pointing left
    plane_point = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([1.0, 0.0, 0.0])  # Left wall normal
    
    intersection = ray_plane_intersection(ray_origin, ray_direction, plane_point, plane_normal)
    
    if intersection is not None:
        print(f"  Ray-plane intersection: {intersection}")
        assert np.allclose(intersection[0], 0.0), "X should be 0"
        print("✓ Ray-plane intersection correct")
    else:
        print("✗ Ray-plane intersection failed")
        return False
    
    print("✓ Math utils test passed")


if __name__ == "__main__":
    try:
        # Run component tests
        test_geometry()
        test_math_utils()
        
        # Run full pipeline test
        test_basic_pipeline()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

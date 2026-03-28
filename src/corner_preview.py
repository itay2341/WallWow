#!/usr/bin/env python3
"""
3D Mock Corner Renderer for WallWow
Visualizes generated panels in a realistic 3D corner environment.
Supports both interactive viewing and static image export.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import trimesh
    import pyrender
except ImportError:
    print("ERROR: Required dependencies not installed.")
    print("Please run: pip install pyrender trimesh")
    sys.exit(1)


def create_corner_geometry(wall_width=3.0, wall_height=2.5):
    """
    Create three rectangular meshes forming a 90° corner.
    
    Coordinate system matches WallWow:
    - Origin at corner vertex (0, 0, 0)
    - Left wall: YZ plane, extends in -Y direction
    - Right wall: XZ plane, extends in -X direction
    - Ceiling: XY plane at Z=wall_height
    
    Args:
        wall_width: Width of each wall in meters (default 3.0)
        wall_height: Height of walls in meters (default 2.5)
    
    Returns:
        dict: {'left_wall': mesh, 'right_wall': mesh, 'ceiling': mesh}
    """
    meshes = {}
    
    # Left Wall (YZ plane at X=0)
    # Extends from (0, -wall_width, 0) to (0, 0, wall_height)
    left_vertices = np.array([
        [0, 0, 0],               # Bottom-right corner
        [0, -wall_width, 0],     # Bottom-left corner
        [0, -wall_width, wall_height],  # Top-left corner
        [0, 0, wall_height]      # Top-right corner
    ], dtype=np.float64)
    
    left_faces = np.array([
        [0, 1, 2],  # First triangle
        [0, 2, 3]   # Second triangle
    ])
    
    # UV coordinates (0,0 at bottom-left, 1,1 at top-right)
    left_uvs = np.array([
        [1, 0],  # Bottom-right
        [0, 0],  # Bottom-left
        [0, 1],  # Top-left
        [1, 1]   # Top-right
    ], dtype=np.float64)
    
    meshes['left_wall'] = trimesh.Trimesh(
        vertices=left_vertices,
        faces=left_faces,
        visual=trimesh.visual.TextureVisuals(uv=left_uvs)
    )
    
    # Right Wall (XZ plane at Y=0)
    # Extends from (-wall_width, 0, 0) to (0, 0, wall_height)
    right_vertices = np.array([
        [0, 0, 0],                # Bottom-left corner
        [0, 0, wall_height],      # Top-left corner
        [-wall_width, 0, wall_height],  # Top-right corner
        [-wall_width, 0, 0]       # Bottom-right corner
    ], dtype=np.float64)
    
    right_faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])
    
    right_uvs = np.array([
        [0, 0],  # Bottom-left
        [0, 1],  # Top-left
        [1, 1],  # Top-right
        [1, 0]   # Bottom-right
    ], dtype=np.float64)
    
    meshes['right_wall'] = trimesh.Trimesh(
        vertices=right_vertices,
        faces=right_faces,
        visual=trimesh.visual.TextureVisuals(uv=right_uvs)
    )
    
    # Ceiling (XY plane at Z=wall_height)
    # Extends from (-wall_width, -wall_width, wall_height) to (0, 0, wall_height)
    ceiling_vertices = np.array([
        [0, 0, wall_height],                        # Near corner
        [0, -wall_width, wall_height],              # Near-left
        [-wall_width, -wall_width, wall_height],    # Far corner
        [-wall_width, 0, wall_height]               # Near-right
    ], dtype=np.float64)
    
    ceiling_faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])
    
    ceiling_uvs = np.array([
        [1, 1],  # Near corner
        [1, 0],  # Near-left
        [0, 0],  # Far corner
        [0, 1]   # Near-right
    ], dtype=np.float64)
    
    meshes['ceiling'] = trimesh.Trimesh(
        vertices=ceiling_vertices,
        faces=ceiling_faces,
        visual=trimesh.visual.TextureVisuals(uv=ceiling_uvs)
    )
    
    return meshes


def load_and_apply_textures(meshes, panels_dir):
    """
    Load panel textures and apply them to meshes.
    
    Args:
        meshes: dict with 'left_wall', 'right_wall', 'ceiling' trimesh objects
        panels_dir: Path to directory containing left.png, right.png, top.png
    
    Returns:
        dict: {'left_wall': pyrender.Mesh, 'right_wall': pyrender.Mesh, 'ceiling': pyrender.Mesh}
    """
    panels_path = Path(panels_dir)
    
    # Define panel filenames
    panel_files = {
        'left_wall': panels_path / 'left.png',
        'right_wall': panels_path / 'right.png',
        'ceiling': panels_path / 'top.png'
    }
    
    # Verify all files exist
    for name, filepath in panel_files.items():
        if not filepath.exists():
            raise FileNotFoundError(f"Panel image not found: {filepath}")
    
    pyrender_meshes = {}
    
    for name, mesh in meshes.items():
        # Load texture image
        texture_path = panel_files[name]
        texture_image = Image.open(texture_path)
        
        # Convert to RGB if necessary
        if texture_image.mode != 'RGB':
            texture_image = texture_image.convert('RGB')
        
        # Convert PIL image to numpy array
        texture_array = np.array(texture_image)
        
        # Create pyrender texture
        texture = pyrender.Texture(
            source=texture_array,
            source_channels='RGB'
        )
        
        # Create material with texture
        material = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=texture,
            metallicFactor=0.0,      # Non-metallic (matte surface)
            roughnessFactor=0.8,     # Slightly rough (paper-like)
            alphaMode='OPAQUE'
        )
        
        # Create pyrender mesh with material
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        pyrender_meshes[name] = pyrender_mesh
    
    return pyrender_meshes


def create_scene(pyrender_meshes, camera_pos=(2.0, 2.0, 1.2), camera_target=(-1.0, -1.0, 1.25), fov=75.0):
    """
    Create a pyrender scene with textured meshes, camera, and lighting.
    
    Args:
        pyrender_meshes: dict of pyrender.Mesh objects
        camera_pos: Camera position (x, y, z) in meters
        camera_target: Point camera looks at (x, y, z) in meters
        fov: Vertical field of view in degrees
    
    Returns:
        tuple: (scene, camera_node) for rendering or interactive viewing
    """
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    
    # Add meshes to scene
    for name, mesh in pyrender_meshes.items():
        scene.add(mesh, name=name)
    
    # Create camera
    camera = pyrender.PerspectiveCamera(yfov=np.radians(fov), aspectRatio=16/9)
    
    # Compute camera pose (look-at transformation)
    camera_pos = np.array(camera_pos, dtype=np.float64)
    camera_target = np.array(camera_target, dtype=np.float64)
    
    # Compute camera orientation
    forward = camera_target - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    # World up vector (Z-up)
    world_up = np.array([0, 0, 1], dtype=np.float64)
    
    # Compute right and up vectors
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Build rotation matrix (camera to world)
    rotation = np.column_stack([right, up, -forward])
    
    # Build 4x4 transformation matrix
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = rotation
    camera_pose[:3, 3] = camera_pos
    
    camera_node = scene.add(camera, pose=camera_pose, name='camera')
    
    # Add lighting
    # Key light: Main light from upper-right, bright and directional
    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    key_light_pose = np.eye(4)
    key_light_pose[:3, 3] = [2.0, 2.0, 3.0]  # Position
    scene.add(key_light, pose=key_light_pose, name='key_light')
    
    # Fill light: Softer light from opposite side to reduce harsh shadows
    fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
    fill_light_pose = np.eye(4)
    fill_light_pose[:3, 3] = [-2.0, -2.0, 2.0]
    scene.add(fill_light, pose=fill_light_pose, name='fill_light')
    
    # Rim light: Subtle light from behind to add depth
    rim_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    rim_light_pose = np.eye(4)
    rim_light_pose[:3, 3] = [-1.0, -1.0, 2.5]
    scene.add(rim_light, pose=rim_light_pose, name='rim_light')
    
    return scene, camera_node


def render_static(scene, output_path, resolution=(1920, 1080)):
    """
    Render scene to a static PNG image.
    
    Args:
        scene: pyrender.Scene to render
        output_path: Path to save PNG output
        resolution: (width, height) tuple for output image
    """
    width, height = resolution
    
    # Create offscreen renderer
    renderer = pyrender.OffscreenRenderer(width, height)
    
    try:
        # Render scene
        color, depth = renderer.render(scene)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        image = Image.fromarray(color)
        image.save(output_path)
        
        print(f"✓ Static render saved to: {output_path}")
        
    finally:
        renderer.delete()


def render_interactive(scene):
    """
    Launch interactive 3D viewer with mouse controls.
    
    Controls:
    - Left mouse: Rotate view
    - Right mouse: Pan view
    - Scroll wheel: Zoom
    - Q or ESC: Quit
    
    Args:
        scene: pyrender.Scene to view interactively
    """
    print("\n" + "=" * 60)
    print("Interactive 3D Viewer - Controls:")
    print("  • Left mouse drag: Rotate view")
    print("  • Right mouse drag: Pan view")
    print("  • Scroll wheel: Zoom in/out")
    print("  • Q or ESC: Quit viewer")
    print("=" * 60 + "\n")
    
    try:
        viewer = pyrender.Viewer(
            scene,
            use_raymond_lighting=False,  # We have custom lighting
            run_in_thread=False,
            viewport_size=(1920, 1080),
            registered_keys={
                'q': lambda *args: viewer.close_external()
            }
        )
    except Exception as e:
        print(f"ERROR: Failed to launch interactive viewer: {e}")
        print("This may occur if display is not available or OpenGL context cannot be created.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='WallWow 3D Corner Preview - Visualize panels in realistic 3D corner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive viewer only
  python src/corner_preview.py --panels-dir output_simple --mode interactive
  
  # Static render only
  python src/corner_preview.py --panels-dir output_simple --mode static
  
  # Both modes (default)
  python src/corner_preview.py --panels-dir output_simple
  
  # Custom output path and resolution
  python src/corner_preview.py --panels-dir output --output preview.png --resolution 2560 1440
  
  # Custom camera position
  python src/corner_preview.py --panels-dir output_simple --camera-pos 3.0 3.0 2.0
        """
    )
    
    parser.add_argument(
        '--panels-dir',
        type=str,
        required=True,
        help='Directory containing left.png, right.png, top.png panel images'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'static', 'both'],
        default='both',
        help='Rendering mode: interactive viewer, static image, or both (default: both)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for static render (default: <panels-dir>/corner_preview.png)'
    )
    
    parser.add_argument(
        '--resolution',
        type=int,
        nargs=2,
        default=[1920, 1080],
        metavar=('WIDTH', 'HEIGHT'),
        help='Render resolution in pixels (default: 1920 1080)'
    )
    
    parser.add_argument(
        '--camera-pos',
        type=float,
        nargs=3,
        default=[2.0, 2.0, 1.2],
        metavar=('X', 'Y', 'Z'),
        help='Camera position in meters (default: 2.0 2.0 1.2)'
    )
    
    parser.add_argument(
        '--camera-target',
        type=float,
        nargs=3,
        default=[-1.0, -1.0, 1.25],
        metavar=('X', 'Y', 'Z'),
        help='Point camera looks at in meters (default: -1.0 -1.0 1.25)'
    )
    
    parser.add_argument(
        '--fov',
        type=float,
        default=75.0,
        help='Camera vertical field of view in degrees (default: 75.0)'
    )
    
    parser.add_argument(
        '--wall-width',
        type=float,
        default=3.0,
        help='Wall width in meters (default: 3.0)'
    )
    
    parser.add_argument(
        '--wall-height',
        type=float,
        default=2.5,
        help='Wall height in meters (default: 2.5)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("WallWow 3D Corner Preview")
    print("=" * 60)
    print(f"Panels directory: {args.panels_dir}")
    print(f"Render mode: {args.mode}")
    print(f"Camera position: {args.camera_pos}")
    print(f"Camera target: {args.camera_target}")
    print(f"Resolution: {args.resolution[0]}x{args.resolution[1]}")
    print("=" * 60 + "\n")
    
    # Set output path
    if args.output is None:
        args.output = Path(args.panels_dir) / 'corner_preview.png'
    
    # Step 1: Create corner geometry
    print("[1/4] Creating corner geometry...")
    try:
        meshes = create_corner_geometry(
            wall_width=args.wall_width,
            wall_height=args.wall_height
        )
        print(f"  ✓ Created 3 meshes (left_wall, right_wall, ceiling)")
    except Exception as e:
        print(f"ERROR: Failed to create geometry: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Load and apply textures
    print("\n[2/4] Loading panel textures...")
    try:
        pyrender_meshes = load_and_apply_textures(meshes, args.panels_dir)
        print(f"  ✓ Loaded and applied 3 textures")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load textures: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Create scene
    print("\n[3/4] Building 3D scene...")
    try:
        scene, camera_node = create_scene(
            pyrender_meshes,
            camera_pos=tuple(args.camera_pos),
            camera_target=tuple(args.camera_target),
            fov=args.fov
        )
        print(f"  ✓ Scene created with camera and 3 lights")
    except Exception as e:
        print(f"ERROR: Failed to create scene: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Render based on mode
    print("\n[4/4] Rendering...")
    
    if args.mode in ['static', 'both']:
        try:
            render_static(scene, args.output, resolution=tuple(args.resolution))
        except Exception as e:
            print(f"ERROR: Failed to render static image: {e}")
            import traceback
            traceback.print_exc()
            if args.mode == 'static':
                sys.exit(1)
    
    if args.mode in ['interactive', 'both']:
        try:
            render_interactive(scene)
        except Exception as e:
            print(f"ERROR: Failed to launch interactive viewer: {e}")
            import traceback
            traceback.print_exc()
            if args.mode == 'interactive':
                sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ Preview complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

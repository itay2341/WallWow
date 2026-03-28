"""
WallWow - Main Pipeline

Command-line interface for generating 3D corner illusion panels.

Usage:
    python main.py <input_image> [options]

Example:
    python main.py data/cube.png --fov 80 --output-size 2048
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from geometry_engine import CornerGeometry
from projection_mapper import ProjectionMapper
from simple_mapper import SimplePartitionMapper
from renderer import PanelRenderer, load_image, resize_to_camera_resolution


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="WallWow - Generate 3D corner illusion panels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.png
  python main.py input.png --fov 80 --output-size 2048
  python main.py input.png --camera-pos 2.5 2.5 1.5 --output-dir my_output
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input_image',
        type=str,
        help='Path to input image'
    )
    
    # Optional arguments
    parser.add_argument(
        '--camera-pos',
        type=float,
        nargs=3,
        default=[2.0, 2.0, 1.2],
        metavar=('X', 'Y', 'Z'),
        help='Camera position in meters (default: 2.0 2.0 1.2)'
    )
    
    parser.add_argument(
        '--fov',
        type=float,
        default=75.0,
        help='Camera field of view in degrees (default: 75.0)'
    )
    
    parser.add_argument(
        '--output-size',
        type=int,
        default=1024,
        help='Output panel size in pixels (square) (default: 1024)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory (default: output/)'
    )
    
    parser.add_argument(
        '--wall-width',
        type=float,
        default=3.0,
        help='Physical wall width in meters (default: 3.0)'
    )
    
    parser.add_argument(
        '--wall-height',
        type=float,
        default=2.5,
        help='Physical wall/ceiling height in meters (default: 2.5)'
    )
    
    parser.add_argument(
        '--camera-resolution',
        type=int,
        nargs=2,
        default=[1920, 1080],
        metavar=('WIDTH', 'HEIGHT'),
        help='Camera image resolution (default: 1920 1080)'
    )
    
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Skip generating preview composite'
    )
    
    parser.add_argument(
        '--preview-layout',
        type=str,
        choices=['horizontal', 'grid'],
        default='horizontal',
        help='Preview layout (default: horizontal)'
    )
    
    parser.add_argument(
        '--simple-mode',
        action='store_true',
        help='Use simple image partitioning instead of 3D projection (recommended for better coverage)'
    )
    
    parser.add_argument(
        '--perspective',
        type=float,
        default=0.15,
        help='Perspective strength for simple mode (0-0.3) (default: 0.15)'
    )
    
    return parser.parse_args()


def main():
    """Main pipeline execution."""
    args = parse_arguments()
    
    # Print configuration
    print("=" * 60)
    print("WallWow - 3D Corner Illusion Generator")
    print("=" * 60)
    print(f"Input image: {args.input_image}")
    print(f"Camera position: {args.camera_pos}")
    print(f"Camera FOV: {args.fov}°")
    print(f"Camera resolution: {args.camera_resolution[0]}x{args.camera_resolution[1]}")
    print(f"Output panel size: {args.output_size}x{args.output_size}")
    print(f"Wall dimensions: {args.wall_width}m x {args.wall_height}m")
    print(f"Output directory: {args.output_dir}")
    print(f"Mapping mode: {'Simple Partition' if args.simple_mode else '3D Projection'}")
    if args.simple_mode:
        print(f"Perspective strength: {args.perspective}")
    print("=" * 60)
    
    # Step 1: Load input image
    print("\n[1/6] Loading input image...")
    try:
        input_image = load_image(args.input_image)
        print(f"  Loaded image: {input_image.shape[1]}x{input_image.shape[0]} pixels")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Resize to camera resolution if needed
    if (input_image.shape[1], input_image.shape[0]) != tuple(args.camera_resolution):
        print(f"  Resizing to camera resolution: {args.camera_resolution[0]}x{args.camera_resolution[1]}...")
        input_image = resize_to_camera_resolution(input_image, tuple(args.camera_resolution))
    
    # Step 2: Initialize geometry
    print("\n[2/6] Initializing corner geometry...")
    geometry = CornerGeometry(
        wall_width=args.wall_width,
        wall_height=args.wall_height,
        camera_pos=tuple(args.camera_pos),
        camera_fov=args.fov,
        camera_resolution=tuple(args.camera_resolution)
    )
    print(f"  Camera: {geometry.camera}")
    print(f"  Planes: {len(geometry.planes)}")
    
    # Step 3: Compute homographies
    print("\n[3/6] Computing homographies...")
    if args.simple_mode:
        # Use simple partitioning approach
        mapper = SimplePartitionMapper(geometry, output_resolution=args.output_size)
        
        try:
            homographies = mapper.compute_all_homographies_with_perspective(
                input_image.shape[1],
                input_image.shape[0],
                perspective_strength=args.perspective
            )
            print(f"  Computed {len(homographies)} homographies using simple partition")
        except Exception as e:
            print(f"ERROR: Failed to compute homographies: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Use 3D projection approach
        mapper = ProjectionMapper(geometry, output_resolution=args.output_size)
        
        try:
            homographies = mapper.compute_all_homographies()
            print(f"  Computed {len(homographies)} homographies using 3D projection")
        except Exception as e:
            print(f"ERROR: Failed to compute homographies: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Step 4: Render panels
    print("\n[4/6] Rendering panels...")
    renderer = PanelRenderer(None, output_resolution=args.output_size)  # mapper not needed for rendering
    
    try:
        panels = renderer.render_all_panels(input_image, homographies)
        print(f"  Rendered {len(panels)} panels")
    except Exception as e:
        print(f"ERROR: Failed to render panels: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Save output panels
    print(f"\n[5/6] Saving panels to {args.output_dir}...")
    try:
        file_paths = renderer.save_panels(output_dir=args.output_dir, format='png')
        print(f"  Saved {len(file_paths)} panel images")
    except Exception as e:
        print(f"ERROR: Failed to save panels: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 6: Generate preview composite
    if not args.no_preview:
        print(f"\n[6/6] Generating preview composite...")
        try:
            preview_path = Path(args.output_dir) / f"preview_{args.preview_layout}.png"
            renderer.save_preview(output_path=str(preview_path), layout=args.preview_layout)
        except Exception as e:
            print(f"WARNING: Failed to generate preview: {e}")
    else:
        print("\n[6/6] Skipping preview generation")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ Pipeline completed successfully!")
    print("=" * 60)
    print("\nOutput files:")
    for plane_name, file_path in file_paths.items():
        print(f"  {plane_name:15s} -> {file_path}")
    
    if not args.no_preview:
        print(f"  {'preview':15s} -> {preview_path}")
    
    print("\nNext steps:")
    print("  1. View the generated panels")
    print("  2. Print each panel at the same physical size")
    print("  3. Mount on walls and ceiling at a 90° corner")
    print("  4. View from the camera position for best illusion effect")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

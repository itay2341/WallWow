"""
Test Image Generator for WallWow.

Creates synthetic test images for validating the projection system:
- Grid patterns (to check line continuity)
- Gradients
- Geometric shapes
- Perspective scenes
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple


def create_grid_pattern(
    width: int = 1920,
    height: int = 1080,
    grid_size: int = 100,
    line_thickness: int = 2,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    line_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Create a grid pattern image.
    
    Useful for checking if lines remain continuous across panel boundaries.
    
    Args:
        width: Image width
        height: Image height
        grid_size: Size of grid cells in pixels
        line_thickness: Thickness of grid lines
        bg_color: Background color (B, G, R)
        line_color: Line color (B, G, R)
    
    Returns:
        Grid pattern image
    """
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # Draw vertical lines
    for x in range(0, width, grid_size):
        cv2.line(img, (x, 0), (x, height), line_color, line_thickness)
    
    # Draw horizontal lines
    for y in range(0, height, grid_size):
        cv2.line(img, (0, y), (width, y), line_color, line_thickness)
    
    return img


def create_diagonal_lines(
    width: int = 1920,
    height: int = 1080,
    num_lines: int = 20,
    line_thickness: int = 3,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    line_color: Tuple[int, int, int] = (0, 0, 255)
) -> np.ndarray:
    """
    Create diagonal lines pattern.
    
    Args:
        width: Image width
        height: Image height
        num_lines: Number of diagonal lines
        line_thickness: Thickness of lines
        bg_color: Background color
        line_color: Line color
    
    Returns:
        Diagonal lines image
    """
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # Draw diagonal lines from top-left to bottom-right
    step = width // num_lines
    for i in range(-num_lines, num_lines):
        x_start = i * step
        cv2.line(img, (x_start, 0), (x_start + height, height), line_color, line_thickness)
    
    return img


def create_gradient(
    width: int = 1920,
    height: int = 1080,
    direction: str = "horizontal"
) -> np.ndarray:
    """
    Create a gradient image.
    
    Args:
        width: Image width
        height: Image height
        direction: "horizontal", "vertical", or "radial"
    
    Returns:
        Gradient image
    """
    if direction == "horizontal":
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        gradient = np.tile(gradient, (height, 1))
        img = np.stack([gradient, gradient, gradient], axis=2)
    
    elif direction == "vertical":
        gradient = np.linspace(0, 255, height, dtype=np.uint8)
        gradient = np.tile(gradient, (width, 1)).T
        img = np.stack([gradient, gradient, gradient], axis=2)
    
    elif direction == "radial":
        cx, cy = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        gradient = (dist / max_dist * 255).astype(np.uint8)
        img = np.stack([gradient, gradient, gradient], axis=2)
    
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    return img


def create_checkerboard(
    width: int = 1920,
    height: int = 1080,
    square_size: int = 100
) -> np.ndarray:
    """
    Create a checkerboard pattern.
    
    Args:
        width: Image width
        height: Image height
        square_size: Size of checkerboard squares
    
    Returns:
        Checkerboard image
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            # Determine if this square is black or white
            row = y // square_size
            col = x // square_size
            if (row + col) % 2 == 0:
                color = (255, 255, 255)
            else:
                color = (0, 0, 0)
            
            x_end = min(x + square_size, width)
            y_end = min(y + square_size, height)
            img[y:y_end, x:x_end] = color
    
    return img


def create_perspective_corridor(
    width: int = 1920,
    height: int = 1080
) -> np.ndarray:
    """
    Create a simple perspective corridor image.
    
    Args:
        width: Image width
        height: Image height
    
    Returns:
        Perspective corridor image
    """
    img = np.full((height, width, 3), (200, 200, 200), dtype=np.uint8)
    
    # Vanishing point
    vp_x = width // 2
    vp_y = height // 2
    
    # Draw perspective lines
    corners = [
        (0, 0),
        (width - 1, 0),
        (width - 1, height - 1),
        (0, height - 1)
    ]
    
    for corner in corners:
        cv2.line(img, corner, (vp_x, vp_y), (0, 0, 0), 2)
    
    # Draw horizontal lines at various depths
    for i in range(1, 10):
        t = i / 10  # Interpolation parameter
        
        # Top edge
        x1 = int(t * vp_x)
        x2 = int(width - t * (width - vp_x))
        y = int(t * vp_y)
        cv2.line(img, (x1, y), (x2, y), (50, 50, 50), 1)
        
        # Bottom edge
        y_bottom = int(height - t * (height - vp_y))
        cv2.line(img, (x1, y_bottom), (x2, y_bottom), (50, 50, 50), 1)
    
    return img


def create_colorful_gradient(
    width: int = 1920,
    height: int = 1080
) -> np.ndarray:
    """
    Create a colorful HSV gradient.
    
    Args:
        width: Image width
        height: Image height
    
    Returns:
        Colorful gradient image
    """
    # Create HSV image
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Hue varies horizontally (0-180 for OpenCV)
    hue = np.linspace(0, 180, width, dtype=np.uint8)
    hsv[:, :, 0] = hue
    
    # Saturation varies vertically
    sat = np.linspace(255, 100, height, dtype=np.uint8)
    hsv[:, :, 1] = sat[:, np.newaxis]
    
    # Full value
    hsv[:, :, 2] = 255
    
    # Convert to BGR
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return img


def generate_all_test_images(output_dir: str = "data"):
    """
    Generate all test images.
    
    Args:
        output_dir: Directory to save test images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating test images in {output_dir}/...")
    
    # Grid pattern
    print("  - grid.png")
    grid = create_grid_pattern()
    cv2.imwrite(str(output_path / "grid.png"), grid)
    
    # Diagonal lines
    print("  - diagonal.png")
    diagonal = create_diagonal_lines()
    cv2.imwrite(str(output_path / "diagonal.png"), diagonal)
    
    # Checkerboard
    print("  - checkerboard.png")
    checker = create_checkerboard()
    cv2.imwrite(str(output_path / "checkerboard.png"), checker)
    
    # Gradients
    print("  - gradient_h.png")
    grad_h = create_gradient(direction="horizontal")
    cv2.imwrite(str(output_path / "gradient_h.png"), grad_h)
    
    print("  - gradient_v.png")
    grad_v = create_gradient(direction="vertical")
    cv2.imwrite(str(output_path / "gradient_v.png"), grad_v)
    
    print("  - gradient_radial.png")
    grad_r = create_gradient(direction="radial")
    cv2.imwrite(str(output_path / "gradient_radial.png"), grad_r)
    
    # Colorful gradient
    print("  - rainbow.png")
    rainbow = create_colorful_gradient()
    cv2.imwrite(str(output_path / "rainbow.png"), rainbow)
    
    # Perspective corridor
    print("  - corridor.png")
    corridor = create_perspective_corridor()
    cv2.imwrite(str(output_path / "corridor.png"), corridor)
    
    print(f"\n✓ Generated 8 test images")


if __name__ == "__main__":
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    generate_all_test_images(output_dir)

"""
Panel Renderer for WallWow.

Applies perspective transformations to generate the three output panel images
(left wall, right wall, ceiling) from the input image.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional
from PIL import Image


class PanelRenderer:
    """
    Renders output panel images by applying perspective warps.
    
    Takes homography matrices from ProjectionMapper and uses cv2.warpPerspective
    to generate the three panel images that will be printed and mounted.
    """
    
    def __init__(
        self,
        projection_mapper = None,
        output_resolution: int = 1024
    ):
        """
        Initialize the panel renderer.
        
        Args:
            projection_mapper: ProjectionMapper with computed homographies (optional)
            output_resolution: Output panel size (square: resolution x resolution)
        """
        self.projection_mapper = projection_mapper
        self.output_resolution = output_resolution
        
        # Store rendered panels
        self.panels = {}  # plane_name -> image array
    
    def render_panel(
        self,
        input_image: np.ndarray,
        plane_name: str,
        homography: np.ndarray,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Render a single panel by applying perspective warp.
        
        Args:
            input_image: Source image (H, W, 3) in BGR or RGB format
            plane_name: Name of the plane ("left_wall", "right_wall", "ceiling")
            homography: 3x3 homography matrix mapping panel coords to image coords
            interpolation: OpenCV interpolation method
            border_mode: How to handle out-of-bounds pixels
            border_value: Fill value for out-of-bounds pixels
        
        Returns:
            Rendered panel image (output_resolution, output_resolution, 3)
        """
        output_size = (self.output_resolution, self.output_resolution)
        
        # Apply perspective warp
        # Note: cv2.warpPerspective applies H such that: src_coords = H @ dst_coords
        # This is exactly what we want: input_image_coords = H @ panel_coords
        panel_image = cv2.warpPerspective(
            input_image,
            homography,
            output_size,
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value
        )
        
        # Store the panel
        self.panels[plane_name] = panel_image
        
        return panel_image
    
    def render_all_panels(
        self,
        input_image: np.ndarray,
        homographies: Dict[str, np.ndarray],
        interpolation: int = cv2.INTER_LINEAR
    ) -> Dict[str, np.ndarray]:
        """
        Render all three panels.
        
        Args:
            input_image: Source image (H, W, 3)
            homographies: Dictionary mapping plane names to homography matrices
            interpolation: OpenCV interpolation method
        
        Returns:
            Dictionary mapping plane names to rendered panel images
        """
        panels = {}
        
        for plane_name, H in homographies.items():
            print(f"Rendering {plane_name}...")
            panel = self.render_panel(
                input_image,
                plane_name,
                H,
                interpolation=interpolation
            )
            panels[plane_name] = panel
        
        return panels
    
    def save_panels(
        self,
        output_dir: str = "output",
        format: str = "png",
        quality: int = 95
    ) -> Dict[str, str]:
        """
        Save rendered panels to disk.
        
        Args:
            output_dir: Output directory path
            format: Image format ("png", "jpg", etc.)
            quality: JPEG quality (0-100), ignored for PNG
        
        Returns:
            Dictionary mapping plane names to output file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        
        for plane_name, panel_image in self.panels.items():
            # Convert BGR to RGB if needed (OpenCV uses BGR)
            if panel_image.shape[2] == 3:
                panel_rgb = cv2.cvtColor(panel_image, cv2.COLOR_BGR2RGB)
            else:
                panel_rgb = panel_image
            
            # Determine filename based on plane
            if plane_name == "left_wall":
                filename = f"left.{format}"
            elif plane_name == "right_wall":
                filename = f"right.{format}"
            elif plane_name == "ceiling":
                filename = f"top.{format}"
            else:
                filename = f"{plane_name}.{format}"
            
            file_path = output_path / filename
            
            # Save using PIL for better format support
            pil_image = Image.fromarray(panel_rgb)
            
            if format.lower() in ['jpg', 'jpeg']:
                pil_image.save(file_path, quality=quality, optimize=True)
            else:
                pil_image.save(file_path, optimize=True)
            
            file_paths[plane_name] = str(file_path)
            print(f"  Saved {plane_name} to {file_path}")
        
        return file_paths
    
    def get_panel(self, plane_name: str) -> np.ndarray:
        """
        Get a rendered panel by name.
        
        Args:
            plane_name: Name of the plane
        
        Returns:
            Panel image array
        """
        if plane_name not in self.panels:
            raise ValueError(f"Panel {plane_name} not yet rendered")
        
        return self.panels[plane_name]
    
    def create_preview_composite(
        self,
        layout: str = "horizontal"
    ) -> np.ndarray:
        """
        Create a composite preview image showing all three panels.
        
        Args:
            layout: "horizontal" or "grid"
        
        Returns:
            Composite image
        """
        if len(self.panels) < 3:
            raise ValueError("All three panels must be rendered first")
        
        left = self.panels["left_wall"]
        right = self.panels["right_wall"]
        top = self.panels["ceiling"]
        
        if layout == "horizontal":
            # Arrange panels side by side: [left | right | top]
            composite = np.hstack([left, right, top])
        
        elif layout == "grid":
            # Arrange in 2x2 grid:
            # [left  | right]
            # [top   | blank]
            top_row = np.hstack([left, right])
            blank = np.zeros_like(top)
            bottom_row = np.hstack([top, blank])
            composite = np.vstack([top_row, bottom_row])
        
        else:
            raise ValueError(f"Unknown layout: {layout}")
        
        return composite
    
    def save_preview(
        self,
        output_path: str = "output/preview.png",
        layout: str = "horizontal"
    ):
        """
        Save a composite preview image.
        
        Args:
            output_path: Path to save preview
            layout: "horizontal" or "grid"
        """
        composite = self.create_preview_composite(layout)
        
        # Convert BGR to RGB
        composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        pil_image = Image.fromarray(composite_rgb)
        pil_image.save(output_path, optimize=True)
        
        print(f"Saved preview to {output_path}")


def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to image file
        target_size: Optional (width, height) to resize to
    
    Returns:
        Image array in BGR format (OpenCV convention)
    """
    # Load with OpenCV
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Resize if requested
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    return image


def resize_to_camera_resolution(
    image: np.ndarray,
    camera_resolution: Tuple[int, int]
) -> np.ndarray:
    """
    Resize image to match camera resolution.
    
    Args:
        image: Input image
        camera_resolution: (width, height) in pixels
    
    Returns:
        Resized image
    """
    return cv2.resize(image, camera_resolution, interpolation=cv2.INTER_AREA)

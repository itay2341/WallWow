"""
Projection Mapper for WallWow.

Implements the reverse projection algorithm that determines which part of the
input image should be painted on each physical plane (wall/ceiling) to create
the desired 3D illusion.

Algorithm:
1. For each plane, create a grid of sample points on its surface
2. For each point on the plane:
   - Project it through the camera to determine where it appears in the input image
   - Build correspondence: plane_coords → image_coords
3. Compute perspective transformation matrix for warping
"""

import numpy as np
from typing import Tuple, Dict
import cv2

from geometry_engine import Camera, Plane, CornerGeometry
from math_utils import create_meshgrid_2d


class ProjectionMapper:
    """
    Maps regions of the input image onto physical planes.
    
    For each plane, computes the correspondence between points on the
    physical plane and pixels in the input image, enabling perspective
    warping to generate the panel images.
    """
    
    def __init__(self, geometry: CornerGeometry, output_resolution: int = 1024):
        """
        Initialize the projection mapper.
        
        Args:
            geometry: The corner geometry system (planes + camera)
            output_resolution: Output panel resolution (square: resolution x resolution)
        """
        self.geometry = geometry
        self.output_resolution = output_resolution
        
        # Store computed mappings
        self.mappings = {}  # plane_name -> (src_points, dst_points, homography)
    
    def compute_plane_mapping(
        self,
        plane: Plane,
        sample_density: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the mapping from a physical plane to the input image.
        
        Strategy:
        1. Create a grid of 3D points on the physical plane
        2. Project each point through the camera to get image coordinates
        3. Build correspondence map
        4. Compute homography for perspective warp
        
        Args:
            plane: The physical plane to map
            sample_density: Number of sample points per dimension (default: output_resolution)
        
        Returns:
            Tuple of:
            - plane_coords_2d: 2D coordinates on plane surface (N, 2)
            - image_coords: Corresponding 2D image coordinates (N, 2)
            - valid_mask: Boolean mask indicating valid correspondences (N,)
        """
        if sample_density is None:
            sample_density = self.output_resolution
        
        # Generate 3D points on the plane surface
        plane_points_3d = self._generate_plane_points(plane, sample_density)
        
        # Project 3D points onto camera's image plane
        image_coords = self.geometry.camera.project_to_image(plane_points_3d)
        
        # Determine valid projections (not NaN, within image bounds)
        valid_mask = self._filter_valid_projections(image_coords)
        
        # Extract 2D coordinates on the plane surface (for output panel)
        plane_coords_2d = self._get_plane_2d_coords(plane, plane_points_3d)
        
        return plane_coords_2d, image_coords, valid_mask
    
    def _generate_plane_points(self, plane: Plane, density: int) -> np.ndarray:
        """
        Generate a grid of 3D points on a plane surface.
        
        Args:
            plane: The plane to sample
            density: Number of points per dimension
        
        Returns:
            Array of 3D points (shape: (density * density, 3))
        """
        if plane.name == "left_wall":
            # Left wall: X = 0, vary Y and Z
            y_min, y_max, z_min, z_max = plane.get_bounds()
            y = np.linspace(y_min, y_max, density)
            z = np.linspace(z_min, z_max, density)
            yv, zv = np.meshgrid(y, z)
            x = np.zeros_like(yv)
            points_3d = np.stack([x.ravel(), yv.ravel(), zv.ravel()], axis=1)
            
        elif plane.name == "right_wall":
            # Right wall: Y = 0, vary X and Z
            x_min, x_max, z_min, z_max = plane.get_bounds()
            x = np.linspace(x_min, x_max, density)
            z = np.linspace(z_min, z_max, density)
            xv, zv = np.meshgrid(x, z)
            y = np.zeros_like(xv)
            points_3d = np.stack([xv.ravel(), y.ravel(), zv.ravel()], axis=1)
            
        elif plane.name == "ceiling":
            # Ceiling: Z = height, vary X and Y
            x_min, x_max, y_min, y_max = plane.get_bounds()
            x = np.linspace(x_min, x_max, density)
            y = np.linspace(y_min, y_max, density)
            xv, yv = np.meshgrid(x, y)
            z = np.full_like(xv, plane.ceiling_height)
            points_3d = np.stack([xv.ravel(), yv.ravel(), z.ravel()], axis=1)
            
        else:
            raise ValueError(f"Unknown plane: {plane.name}")
        
        return points_3d
    
    def _get_plane_2d_coords(self, plane: Plane, points_3d: np.ndarray) -> np.ndarray:
        """
        Extract 2D coordinates within the plane's local coordinate system.
        
        Maps 3D plane points to 2D output panel coordinates.
        
        Args:
            plane: The plane
            points_3d: 3D points on the plane (N, 3)
        
        Returns:
            2D coordinates (N, 2) in range [0, output_resolution]
        """
        if plane.name == "left_wall":
            # Map (Y, Z) to (U, V) in output image
            y_min, y_max, z_min, z_max = plane.get_bounds()
            u = (points_3d[:, 1] - y_min) / (y_max - y_min) * self.output_resolution
            v = (points_3d[:, 2] - z_min) / (z_max - z_min) * self.output_resolution
            
        elif plane.name == "right_wall":
            # Map (X, Z) to (U, V) in output image
            x_min, x_max, z_min, z_max = plane.get_bounds()
            u = (points_3d[:, 0] - x_min) / (x_max - x_min) * self.output_resolution
            v = (points_3d[:, 2] - z_min) / (z_max - z_min) * self.output_resolution
            
        elif plane.name == "ceiling":
            # Map (X, Y) to (U, V) in output image
            x_min, x_max, y_min, y_max = plane.get_bounds()
            u = (points_3d[:, 0] - x_min) / (x_max - x_min) * self.output_resolution
            v = (points_3d[:, 1] - y_min) / (y_max - y_min) * self.output_resolution
            
        else:
            raise ValueError(f"Unknown plane: {plane.name}")
        
        coords_2d = np.stack([u, v], axis=1)
        return coords_2d
    
    def _filter_valid_projections(self, image_coords: np.ndarray) -> np.ndarray:
        """
        Filter out invalid projections (NaN, out of bounds).
        
        Args:
            image_coords: Projected image coordinates (N, 2)
        
        Returns:
            Boolean mask (N,) indicating valid projections
        """
        # Check for NaN
        valid = ~np.isnan(image_coords).any(axis=1)
        
        # Check bounds (allow some margin for interpolation)
        cam_width = self.geometry.camera.width
        cam_height = self.geometry.camera.height
        
        valid &= (image_coords[:, 0] >= -10) & (image_coords[:, 0] < cam_width + 10)
        valid &= (image_coords[:, 1] >= -10) & (image_coords[:, 1] < cam_height + 10)
        
        return valid
    
    def compute_homography(
        self,
        plane: Plane,
        sample_for_homography: int = 100
    ) -> np.ndarray:
        """
        Compute the homography matrix for a plane with proper image region partitioning.
        
        The homography maps from output panel coordinates to the appropriate region
        of the input image, ensuring no overlap between planes.
        
        Args:
            plane: The plane to compute homography for
            sample_for_homography: Number of sample points per dimension for homography computation
        
        Returns:
            3x3 homography matrix
        """
        # Get correspondences
        plane_coords, image_coords, valid = self.compute_plane_mapping(
            plane, sample_density=sample_for_homography
        )
        
        # Filter to valid correspondences
        plane_coords_valid = plane_coords[valid]
        image_coords_valid = image_coords[valid]
        
        if len(plane_coords_valid) < 4:
            raise ValueError(f"Not enough valid correspondences for {plane.name}: {len(plane_coords_valid)}")
        
        # Determine the bounding region in image space for this plane
        # This helps partition the image across planes
        img_width = self.geometry.camera.width
        img_height = self.geometry.camera.height
        
        # Define image regions for each plane based on camera view
        # These regions ensure proper partitioning with minimal overlap
        if plane.name == "left_wall":
            # Left third of image
            region_mask = image_coords_valid[:, 0] < (img_width / 3)
        elif plane.name == "right_wall":
            # Right third of image  
            region_mask = image_coords_valid[:, 0] > (2 * img_width / 3)
        elif plane.name == "ceiling":
            # Top third and center of image
            region_mask = (image_coords_valid[:, 1] < (img_height / 2)) & \
                         (image_coords_valid[:, 0] >= (img_width / 3)) & \
                         (image_coords_valid[:, 0] <= (2 * img_width / 3))
        else:
            region_mask = np.ones(len(image_coords_valid), dtype=bool)
        
        # Apply region mask to get plane-specific correspondences
        plane_coords_region = plane_coords_valid[region_mask]
        image_coords_region = image_coords_valid[region_mask]
        
        if len(plane_coords_region) < 4:
            # Fallback to all valid points if region filtering is too aggressive
            print(f"  Warning: Region filtering too aggressive for {plane.name}, using all valid points")
            plane_coords_region = plane_coords_valid
            image_coords_region = image_coords_valid
        
        # Use OpenCV to compute homography (more robust than manual DLT)
        # Homography maps from plane_coords (dst) to image_coords (src)
        # We want: image_coords = H @ plane_coords
        H, mask = cv2.findHomography(plane_coords_region, image_coords_region, cv2.RANSAC, 5.0)
        
        if H is None:
            raise ValueError(f"Failed to compute homography for {plane.name}")
        
        # Store the mapping
        self.mappings[plane.name] = {
            'plane_coords': plane_coords_region,
            'image_coords': image_coords_region,
            'homography': H,
            'inliers': mask
        }
        
        return H
    
    def compute_all_homographies(self) -> Dict[str, np.ndarray]:
        """
        Compute homographies for all three planes.
        
        Returns:
            Dictionary mapping plane names to homography matrices
        """
        homographies = {}
        
        for plane in self.geometry.planes:
            print(f"Computing homography for {plane.name}...")
            H = self.compute_homography(plane)
            homographies[plane.name] = H
            
            # Print statistics
            n_valid = np.sum(self.mappings[plane.name]['inliers'])
            n_total = len(self.mappings[plane.name]['plane_coords'])
            print(f"  Valid correspondences: {n_valid} / {n_total}")
        
        return homographies
    
    def get_mapping(self, plane_name: str) -> Dict:
        """
        Get the stored mapping for a plane.
        
        Args:
            plane_name: Name of the plane
        
        Returns:
            Dictionary with keys: 'plane_coords', 'image_coords', 'homography', 'inliers'
        """
        if plane_name not in self.mappings:
            raise ValueError(f"No mapping computed for {plane_name}. Call compute_homography first.")
        
        return self.mappings[plane_name]
    
    def visualize_correspondences(self, plane_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get correspondence points for visualization.
        
        Args:
            plane_name: Name of the plane
        
        Returns:
            Tuple of (plane_coords, image_coords) for valid inliers
        """
        mapping = self.get_mapping(plane_name)
        inliers = mapping['inliers'].ravel() > 0
        
        plane_coords = mapping['plane_coords'][inliers]
        image_coords = mapping['image_coords'][inliers]
        
        return plane_coords, image_coords

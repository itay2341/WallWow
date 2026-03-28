"""
Simple Image Partitioning Mapper for WallWow.

Alternative approach that divides the input image into three regions
and maps each to a panel, ensuring full coverage and no overlap.
"""

import numpy as np
import cv2
from typing import Dict, Tuple

from geometry_engine import CornerGeometry


class SimplePartitionMapper:
    """
    Maps input image regions to panels using simple geometric partitioning.
    
    This approach divides the input image into distinct regions for each plane,
    ensuring no overlap and full utilization of the input image.
    """
    
    def __init__(self, geometry: CornerGeometry, output_resolution: int = 1024):
        """
        Initialize the simple mapper.
        
        Args:
            geometry: Corner geometry (used for aspect ratios)
            output_resolution: Output panel resolution
        """
        self.geometry = geometry
        self.output_resolution = output_resolution
        self.homographies = {}
    
    def compute_all_homographies_simple(
        self,
        input_width: int,
        input_height: int
    ) -> Dict[str, np.ndarray]:
        """
        Compute homographies by partitioning the input image into three regions.
        
        Strategy:
        - Left wall: Maps from left portion of image
        - Right wall: Maps from right portion of image
        - Ceiling: Maps from top-center portion of image
        
        Args:
            input_width: Width of input image
            input_height: Height of input image
        
        Returns:
            Dictionary of homographies for each plane
        """
        print("Using simple partition mapping strategy...")
        
        # Define regions in input image
        # These ensure each plane gets a distinct portion with no overlap
        
        # Left wall gets left 35% of image
        left_region = self._create_homography_for_region(
            src_x=0,
            src_y=0,  
            src_width=int(input_width * 0.35),
            src_height=input_height,
            output_size=self.output_resolution
        )
        
        # Right wall gets right 35% of image
        right_region = self._create_homography_for_region(
            src_x=int(input_width * 0.65),
            src_y=0,
            src_width=int(input_width * 0.35),
            src_height=input_height,
            output_size=self.output_resolution
        )
        
        # Ceiling gets center-top 30% width, top 40% height
        ceiling_region = self._create_homography_for_region(
            src_x=int(input_width * 0.35),
            src_y=0,
            src_width=int(input_width * 0.30),
            src_height=int(input_height * 0.40),
            output_size=self.output_resolution
        )
        
        self.homographies = {
            'left_wall': left_region,
            'right_wall': right_region,
            'ceiling': ceiling_region
        }
        
        print(f"  Computed simple partitioning for all 3 planes")
        print(f"    Left: Uses left {int(input_width * 0.35)}px of image")
        print(f"    Right: Uses right {int(input_width * 0.35)}px of image")
        print(f"    Ceiling: Uses center-top {int(input_width * 0.30)}x{int(input_height * 0.40)}px")
        
        return self.homographies
    
    def _create_homography_for_region(
        self,
        src_x: int,
        src_y: int,
        src_width: int,
        src_height: int,
        output_size: int
    ) -> np.ndarray:
        """
        Create a homography that maps a rectangular region to output panel.
        
        Args:
            src_x: X coordinate of region in input image
            src_y: Y coordinate of region in input image
            src_width: Width of region
            src_height: Height of region
            output_size: Output panel size (square)
        
        Returns:
            3x3 homography matrix
        """
        # Source (input image) corners
        src_pts = np.float32([
            [src_x, src_y],                           # Top-left
            [src_x + src_width, src_y],               # Top-right
            [src_x + src_width, src_y + src_height],  # Bottom-right
            [src_x, src_y + src_height]               # Bottom-left
        ])
        
        # Destination (output panel) corners
        dst_pts = np.float32([
            [0, 0],                          # Top-left
            [output_size, 0],                # Top-right
            [output_size, output_size],      # Bottom-right
            [0, output_size]                 # Bottom-left
        ])
        
        # Compute perspective transform
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        return H
    
    def compute_all_homographies_with_perspective(
        self,
        input_width: int,
        input_height: int,
        perspective_strength: float = 0.15
    ) -> Dict[str, np.ndarray]:
        """
        Compute homographies with perspective distortion for 3D effect.
        
        This adds perspective warping to simulate the corner viewing angle.
        
        Args:
            input_width: Width of input image
            input_height: Height of input image
            perspective_strength: How much perspective to apply (0-1)
        
        Returns:
            Dictionary of homographies for each plane
        """
        print("Using perspective-aware partition mapping...")
        
        # Left wall - with perspective converging right
        left_src = np.float32([
            [0, 0],
            [int(input_width * 0.40), 0],
            [int(input_width * 0.35), input_height],
            [0, input_height]
        ])
        left_dst = np.float32([
            [int(self.output_resolution * perspective_strength), 0],
            [self.output_resolution, 0],
            [self.output_resolution, self.output_resolution],
            [0, self.output_resolution]
        ])
        left_H = cv2.getPerspectiveTransform(left_src, left_dst)
        
        # Right wall - with perspective converging left
        right_src = np.float32([
            [int(input_width * 0.60), 0],
            [input_width, 0],
            [input_width, input_height],
            [int(input_width * 0.65), input_height]
        ])
        right_dst = np.float32([
            [0, 0],
            [self.output_resolution - int(self.output_resolution * perspective_strength), 0],
            [self.output_resolution, self.output_resolution],
            [0, self.output_resolution]
        ])
        right_H = cv2.getPerspectiveTransform(right_src, right_dst)
        
        # Ceiling - with perspective converging down
        ceiling_src = np.float32([
            [int(input_width * 0.35), 0],
            [int(input_width * 0.65), 0],
            [int(input_width * 0.70), int(input_height * 0.45)],
            [int(input_width * 0.30), int(input_height * 0.45)]
        ])
        ceiling_dst = np.float32([
            [0, 0],
            [self.output_resolution, 0],
            [self.output_resolution - int(self.output_resolution * perspective_strength), self.output_resolution],
            [int(self.output_resolution * perspective_strength), self.output_resolution]
        ])
        ceiling_H = cv2.getPerspectiveTransform(ceiling_src, ceiling_dst)
        
        self.homographies = {
            'left_wall': left_H,
            'right_wall': right_H,
            'ceiling': ceiling_H
        }
        
        print(f"  Computed perspective-aware mapping for all 3 planes")
        print(f"    Perspective strength: {perspective_strength}")
        
        return self.homographies

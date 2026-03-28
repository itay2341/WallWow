"""
Geometry Engine for WallWow

Defines the 3D coordinate system, camera, and physical planes (walls + ceiling)
for the corner illusion system.

Coordinate System:
- Origin at corner vertex where two walls and ceiling meet
- Left wall: YZ plane (X ≤ 0)
- Right wall: XZ plane (Y ≥ 0)
- Ceiling: XY plane (Z = ceiling_height)
- All planes meet at 90° angles
"""

import numpy as np
from typing import Tuple


class Plane:
    """
    Represents a 3D plane in the corner system.
    
    A plane is defined by a point on the plane and a normal vector.
    Plane equation: n · (P - P0) = 0
    where n is the normal, P is any point on the plane, P0 is a known point.
    """
    
    def __init__(self, point: np.ndarray, normal: np.ndarray, name: str):
        """
        Initialize a plane.
        
        Args:
            point: A 3D point on the plane (shape: (3,))
            normal: Normal vector to the plane (shape: (3,))
            name: Human-readable name (e.g., "left_wall")
        """
        self.point = np.array(point, dtype=np.float64)
        self.normal = np.array(normal, dtype=np.float64)
        self.normal = self.normal / np.linalg.norm(self.normal)  # Normalize
        self.name = name
        
        # Compute plane constant d in equation: ax + by + cz = d
        self.d = np.dot(self.normal, self.point)
    
    def __repr__(self):
        return f"Plane(name='{self.name}', normal={self.normal}, d={self.d})"


class LeftWall(Plane):
    """Left wall in YZ plane (X ≤ 0)."""
    
    def __init__(self, width: float = 3.0, height: float = 2.5):
        """
        Args:
            width: Wall width in meters (extending in -Y direction from corner)
            height: Wall height in meters (Z direction)
        """
        point = np.array([0.0, 0.0, 0.0])  # Origin at corner
        normal = np.array([1.0, 0.0, 0.0])  # Points into the room (+X)
        super().__init__(point, normal, "left_wall")
        self.width = width
        self.height = height
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the bounding box of the wall in 3D space.
        
        Returns:
            (y_min, y_max, z_min, z_max) where wall is at X=0
        """
        return (-self.width, 0.0, 0.0, self.height)


class RightWall(Plane):
    """Right wall in XZ plane (Y ≥ 0)."""
    
    def __init__(self, width: float = 3.0, height: float = 2.5):
        """
        Args:
            width: Wall width in meters (extending in -X direction from corner)
            height: Wall height in meters (Z direction)
        """
        point = np.array([0.0, 0.0, 0.0])  # Origin at corner
        normal = np.array([0.0, 1.0, 0.0])  # Points into the room (+Y)
        super().__init__(point, normal, "right_wall")
        self.width = width
        self.height = height
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the bounding box of the wall in 3D space.
        
        Returns:
            (x_min, x_max, z_min, z_max) where wall is at Y=0
        """
        return (-self.width, 0.0, 0.0, self.height)


class Ceiling(Plane):
    """Ceiling in XY plane at Z = height."""
    
    def __init__(self, width_x: float = 3.0, width_y: float = 3.0, height: float = 2.5):
        """
        Args:
            width_x: Ceiling extent in X direction (meters, extends in -X from corner)
            width_y: Ceiling extent in Y direction (meters, extends in -Y from corner)
            height: Ceiling height from floor (Z coordinate, meters)
        """
        point = np.array([0.0, 0.0, height])  # At ceiling height
        normal = np.array([0.0, 0.0, -1.0])  # Points down into room (-Z)
        super().__init__(point, normal, "ceiling")
        self.width_x = width_x
        self.width_y = width_y
        self.ceiling_height = height
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the bounding box of the ceiling in 3D space.
        
        Returns:
            (x_min, x_max, y_min, y_max) where ceiling is at Z=height
        """
        return (-self.width_x, 0.0, -self.width_y, 0.0)


class Camera:
    """
    Virtual camera observing the corner.
    
    The camera observes the corner from a fixed position with a given
    field of view and resolution. It defines the viewing frustum and
    projection parameters.
    """
    
    def __init__(
        self,
        position: Tuple[float, float, float] = (2.0, 2.0, 1.2),
        look_at: Tuple[float, float, float] = (0.0, 0.0, 1.5),
        up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        fov: float = 75.0,
        resolution: Tuple[int, int] = (1920, 1080)
    ):
        """
        Initialize camera.
        
        Args:
            position: Camera position in world coordinates (x, y, z) meters
            look_at: Point the camera is looking at (x, y, z) meters
            up: Up vector for camera orientation (default: Z-up)
            fov: Vertical field of view in degrees
            resolution: Image resolution (width, height) in pixels
        """
        self.position = np.array(position, dtype=np.float64)
        self.look_at = np.array(look_at, dtype=np.float64)
        self.up = np.array(up, dtype=np.float64)
        self.up = self.up / np.linalg.norm(self.up)  # Normalize
        
        self.fov = fov  # Degrees
        self.fov_rad = np.radians(fov)
        self.resolution = resolution
        self.width, self.height = resolution
        self.aspect_ratio = self.width / self.height
        
        # Compute camera coordinate frame
        self._compute_camera_frame()
        
        # Compute projection matrix (intrinsics)
        self._compute_projection_matrix()
    
    def _compute_camera_frame(self):
        """
        Compute the camera's local coordinate frame (forward, right, up vectors).
        """
        # Forward vector (Z-axis in camera space, points toward scene)
        self.forward = self.look_at - self.position
        self.forward = self.forward / np.linalg.norm(self.forward)
        
        # Right vector (X-axis in camera space)
        self.right = np.cross(self.forward, self.up)
        self.right = self.right / np.linalg.norm(self.right)
        
        # Recompute up vector to ensure orthogonality (Y-axis in camera space)
        self.up = np.cross(self.right, self.forward)
        self.up = self.up / np.linalg.norm(self.up)
    
    def _compute_projection_matrix(self):
        """
        Compute the camera's intrinsic matrix (pinhole camera model).
        
        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        
        where fx, fy are focal lengths in pixels and (cx, cy) is the principal point.
        """
        # Focal length in pixels (from FOV)
        # tan(fov/2) = (height/2) / focal_length
        self.focal_length = (self.height / 2.0) / np.tan(self.fov_rad / 2.0)
        
        fx = fy = self.focal_length
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        self.intrinsic_matrix = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)
    
    def get_extrinsic_matrix(self) -> np.ndarray:
        """
        Get the camera's extrinsic matrix (world-to-camera transform).
        
        Returns:
            4x4 transformation matrix [R | t] where R is rotation and t is translation
        """
        # Build rotation matrix from camera frame vectors
        # In camera space: +X right, +Y down, +Z forward (into scene)
        R = np.array([
            self.right,
            -self.up,  # Flip up to make Y point down
            self.forward  # Camera looks along +Z in camera space
        ], dtype=np.float64)
        
        # Translation vector (camera position in world frame)
        t = -R @ self.position
        
        # Construct 4x4 extrinsic matrix
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        
        return extrinsic
    
    def world_to_camera(self, point_world: np.ndarray) -> np.ndarray:
        """
        Transform a 3D point from world coordinates to camera coordinates.
        
        Args:
            point_world: 3D point in world frame (shape: (3,) or (N, 3))
        
        Returns:
            3D point in camera frame (same shape as input)
        """
        single_point = point_world.ndim == 1
        if single_point:
            point_world = point_world.reshape(1, -1)
        
        # Homogeneous coordinates
        ones = np.ones((point_world.shape[0], 1))
        point_world_h = np.hstack([point_world, ones])
        
        # Apply extrinsic matrix
        extrinsic = self.get_extrinsic_matrix()
        point_cam_h = (extrinsic @ point_world_h.T).T
        
        if single_point:
            return point_cam_h[0, :3]
        return point_cam_h[:, :3]
    
    def project_to_image(self, point_world: np.ndarray) -> np.ndarray:
        """
        Project a 3D world point onto the 2D image plane.
        
        Args:
            point_world: 3D point in world frame (shape: (3,) or (N, 3))
        
        Returns:
            2D pixel coordinates (u, v) (shape: (2,) or (N, 2))
            Returns None for points behind the camera
        """
        # Transform to camera coordinates
        point_cam = self.world_to_camera(point_world)
        
        single_point = point_cam.ndim == 1
        if single_point:
            point_cam = point_cam.reshape(1, -1)
        
        # Check if points are in front of camera (Z > 0 in camera frame)
        valid = point_cam[:, 2] > 0
        
        # Perspective projection
        point_2d = np.zeros((point_cam.shape[0], 2))
        point_2d[valid] = point_cam[valid, :2] / point_cam[valid, 2:3]
        
        # Apply intrinsic matrix
        # [u, v, 1]^T = K @ [x/z, y/z, 1]^T
        ones = np.ones((point_2d.shape[0], 1))
        point_2d_h = np.hstack([point_2d, ones])
        pixel_coords = (self.intrinsic_matrix @ point_2d_h.T).T[:, :2]
        
        # Set invalid points to None (represented as NaN)
        pixel_coords[~valid] = np.nan
        
        if single_point:
            return pixel_coords[0] if valid[0] else None
        return pixel_coords
    
    def __repr__(self):
        return (f"Camera(position={self.position}, look_at={self.look_at}, "
                f"fov={self.fov}°, resolution={self.resolution})")


class CornerGeometry:
    """
    Complete corner geometry system.
    
    Encapsulates the three planes (left wall, right wall, ceiling) and
    the virtual camera observing them.
    """
    
    def __init__(
        self,
        wall_width: float = 3.0,
        wall_height: float = 2.5,
        camera_pos: Tuple[float, float, float] = (2.0, 2.0, 1.2),
        camera_fov: float = 75.0,
        camera_resolution: Tuple[int, int] = (1920, 1080)
    ):
        """
        Initialize the complete corner geometry.
        
        Args:
            wall_width: Width of walls in meters
            wall_height: Height of walls/ceiling in meters
            camera_pos: Camera position (x, y, z) in meters
            camera_fov: Camera field of view in degrees
            camera_resolution: Camera image resolution (width, height)
        """
        self.left_wall = LeftWall(width=wall_width, height=wall_height)
        self.right_wall = RightWall(width=wall_width, height=wall_height)
        self.ceiling = Ceiling(width_x=wall_width, width_y=wall_width, height=wall_height)
        
        # Camera looks at center of corner (negative coords since walls extend in -X,-Y)
        look_at = (-wall_width / 3, -wall_width / 3, wall_height / 2)
        self.camera = Camera(
            position=camera_pos,
            look_at=look_at,
            fov=camera_fov,
            resolution=camera_resolution
        )
        
        self.planes = [self.left_wall, self.right_wall, self.ceiling]
    
    def get_plane_by_name(self, name: str) -> Plane:
        """Get a plane by its name."""
        for plane in self.planes:
            if plane.name == name:
                return plane
        raise ValueError(f"Unknown plane name: {name}")
    
    def __repr__(self):
        return (f"CornerGeometry(\n"
                f"  {self.left_wall}\n"
                f"  {self.right_wall}\n"
                f"  {self.ceiling}\n"
                f"  {self.camera}\n"
                f")")

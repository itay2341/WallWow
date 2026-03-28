"""
Mathematical utilities for WallWow.

Provides ray-plane intersection, coordinate transformations, and matrix utilities
for computing projections and homographies.
"""

import numpy as np
from typing import Optional, Tuple


def ray_plane_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    epsilon: float = 1e-6
) -> Optional[np.ndarray]:
    """
    Compute the intersection point between a ray and a plane.
    
    Ray equation: P(t) = ray_origin + t * ray_direction
    Plane equation: (P - plane_point) · plane_normal = 0
    
    Solving for t:
        (ray_origin + t * ray_direction - plane_point) · plane_normal = 0
        t = ((plane_point - ray_origin) · plane_normal) / (ray_direction · plane_normal)
    
    Args:
        ray_origin: Starting point of the ray (shape: (3,))
        ray_direction: Direction vector of the ray (shape: (3,)), normalized
        plane_point: A point on the plane (shape: (3,))
        plane_normal: Normal vector of the plane (shape: (3,)), normalized
        epsilon: Small value to check for parallel ray/plane
    
    Returns:
        Intersection point (3D coordinates) or None if ray is parallel to plane
        or intersection is behind ray origin (t < 0)
    """
    # Ensure inputs are numpy arrays
    ray_origin = np.array(ray_origin, dtype=np.float64)
    ray_direction = np.array(ray_direction, dtype=np.float64)
    plane_point = np.array(plane_point, dtype=np.float64)
    plane_normal = np.array(plane_normal, dtype=np.float64)
    
    # Normalize direction and normal (in case they aren't already)
    ray_direction = ray_direction / (np.linalg.norm(ray_direction) + epsilon)
    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + epsilon)
    
    # Compute denominator
    denom = np.dot(ray_direction, plane_normal)
    
    # Check if ray is parallel to plane
    if abs(denom) < epsilon:
        return None
    
    # Compute t
    t = np.dot(plane_point - ray_origin, plane_normal) / denom
    
    # Check if intersection is in front of ray origin
    if t < 0:
        return None
    
    # Compute intersection point
    intersection = ray_origin + t * ray_direction
    
    return intersection


def ray_plane_intersection_batch(
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    epsilon: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute intersections between multiple rays and a plane (vectorized).
    
    Args:
        ray_origins: Array of ray starting points (shape: (N, 3))
        ray_directions: Array of ray directions (shape: (N, 3))
        plane_point: A point on the plane (shape: (3,))
        plane_normal: Normal vector of the plane (shape: (3,))
        epsilon: Small value for numerical stability
    
    Returns:
        intersections: Array of intersection points (shape: (N, 3))
        valid: Boolean mask indicating valid intersections (shape: (N,))
    """
    # Normalize directions and normal
    ray_directions = ray_directions / (np.linalg.norm(ray_directions, axis=1, keepdims=True) + epsilon)
    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + epsilon)
    
    # Compute denominators (N,)
    denoms = np.dot(ray_directions, plane_normal)
    
    # Check for parallel rays
    parallel = np.abs(denoms) < epsilon
    
    # Compute t values (N,)
    numerators = np.dot(plane_point - ray_origins, plane_normal)
    t_values = np.zeros(len(ray_origins))
    t_values[~parallel] = numerators[~parallel] / denoms[~parallel]
    
    # Check if intersections are in front of ray origins
    behind = t_values < 0
    
    # Compute intersection points
    intersections = ray_origins + t_values[:, np.newaxis] * ray_directions
    
    # Determine valid intersections
    valid = ~parallel & ~behind
    
    return intersections, valid


def compute_homography_from_points(
    src_points: np.ndarray,
    dst_points: np.ndarray
) -> np.ndarray:
    """
    Compute a homography matrix from point correspondences.
    
    Given 4+ point pairs (src -> dst), compute the 3x3 homography matrix H
    such that dst = H @ src (in homogeneous coordinates).
    
    Args:
        src_points: Source points (shape: (N, 2), N >= 4)
        dst_points: Destination points (shape: (N, 2), N >= 4)
    
    Returns:
        Homography matrix H (shape: (3, 3))
    
    Uses Direct Linear Transform (DLT) algorithm.
    """
    assert src_points.shape[0] >= 4, "Need at least 4 point pairs"
    assert src_points.shape == dst_points.shape, "Point arrays must have same shape"
    
    n_points = src_points.shape[0]
    
    # Build matrix A for homogeneous linear equations
    # For each point correspondence, we get 2 equations
    A = []
    
    for i in range(n_points):
        x, y = src_points[i]
        u, v = dst_points[i]
        
        # First equation: -x, -y, -1, 0, 0, 0, ux, uy, u
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        
        # Second equation: 0, 0, 0, -x, -y, -1, vx, vy, v
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    
    A = np.array(A, dtype=np.float64)
    
    # Solve using SVD: A @ h = 0
    # The solution is the right singular vector corresponding to smallest singular value
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    
    # Reshape to 3x3 matrix
    H = h.reshape(3, 3)
    
    # Normalize so that H[2, 2] = 1
    H = H / H[2, 2]
    
    return H


def points_in_plane_bounds(
    points_3d: np.ndarray,
    plane_name: str,
    bounds: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Check if 3D points lie within the bounds of a plane.
    
    Args:
        points_3d: 3D points to check (shape: (N, 3))
        plane_name: Name of the plane ("left_wall", "right_wall", "ceiling")
        bounds: Plane bounds - interpretation depends on plane:
                - left_wall: (y_min, y_max, z_min, z_max)
                - right_wall: (x_min, x_max, z_min, z_max)
                - ceiling: (x_min, x_max, y_min, y_max)
    
    Returns:
        Boolean mask indicating which points are in bounds (shape: (N,))
    """
    if plane_name == "left_wall":
        # Left wall: X ≈ 0, check Y and Z bounds
        y_min, y_max, z_min, z_max = bounds
        in_bounds = (
            (points_3d[:, 1] >= y_min) & (points_3d[:, 1] <= y_max) &
            (points_3d[:, 2] >= z_min) & (points_3d[:, 2] <= z_max)
        )
    elif plane_name == "right_wall":
        # Right wall: Y ≈ 0, check X and Z bounds
        x_min, x_max, z_min, z_max = bounds
        in_bounds = (
            (points_3d[:, 0] >= x_min) & (points_3d[:, 0] <= x_max) &
            (points_3d[:, 2] >= z_min) & (points_3d[:, 2] <= z_max)
        )
    elif plane_name == "ceiling":
        # Ceiling: Z ≈ height, check X and Y bounds
        x_min, x_max, y_min, y_max = bounds
        in_bounds = (
            (points_3d[:, 0] >= x_min) & (points_3d[:, 0] <= x_max) &
            (points_3d[:, 1] >= y_min) & (points_3d[:, 1] <= y_max)
        )
    else:
        raise ValueError(f"Unknown plane name: {plane_name}")
    
    return in_bounds


def create_meshgrid_2d(width: int, height: int) -> np.ndarray:
    """
    Create a 2D meshgrid of pixel coordinates.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        Array of shape (height * width, 2) containing (x, y) pixel coordinates
    """
    x = np.arange(width)
    y = np.arange(height)
    xv, yv = np.meshgrid(x, y)
    
    # Flatten and stack
    coords = np.stack([xv.ravel(), yv.ravel()], axis=1)
    
    return coords


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize a batch of vectors to unit length.
    
    Args:
        vectors: Array of vectors (shape: (N, 3))
    
    Returns:
        Normalized vectors (shape: (N, 3))
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    return vectors / norms


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix that rotates vec1 to vec2.
    
    Uses Rodrigues' rotation formula.
    
    Args:
        vec1: Source vector (shape: (3,))
        vec2: Target vector (shape: (3,))
    
    Returns:
        3x3 rotation matrix
    """
    # Normalize vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Compute rotation axis (cross product)
    axis = np.cross(vec1, vec2)
    axis_norm = np.linalg.norm(axis)
    
    # Compute rotation angle
    cos_angle = np.dot(vec1, vec2)
    
    # Handle special cases
    if axis_norm < 1e-10:
        # Vectors are parallel
        if cos_angle > 0:
            # Same direction - identity
            return np.eye(3)
        else:
            # Opposite direction - 180° rotation around any perpendicular axis
            # Find a perpendicular vector
            perp = np.array([1, 0, 0]) if abs(vec1[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(vec1, perp)
            axis = axis / np.linalg.norm(axis)
            return rotation_matrix_axis_angle(axis, np.pi)
    
    # Normalize axis
    axis = axis / axis_norm
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    
    return rotation_matrix_axis_angle(axis, angle)


def rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Compute rotation matrix from axis-angle representation (Rodrigues' formula).
    
    Args:
        axis: Rotation axis (shape: (3,)), should be normalized
        angle: Rotation angle in radians
    
    Returns:
        3x3 rotation matrix
    """
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    one_minus_cos = 1 - cos_a
    
    R = np.array([
        [cos_a + kx*kx*one_minus_cos,       kx*ky*one_minus_cos - kz*sin_a,  kx*kz*one_minus_cos + ky*sin_a],
        [ky*kx*one_minus_cos + kz*sin_a,    cos_a + ky*ky*one_minus_cos,      ky*kz*one_minus_cos - kx*sin_a],
        [kz*kx*one_minus_cos - ky*sin_a,    kz*ky*one_minus_cos + kx*sin_a,   cos_a + kz*kz*one_minus_cos]
    ])
    
    return R


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply a homography transformation to 2D points.
    
    Args:
        points: 2D points (shape: (N, 2))
        H: Homography matrix (shape: (3, 3))
    
    Returns:
        Transformed 2D points (shape: (N, 2))
    """
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    
    # Apply homography
    transformed_h = (H @ points_h.T).T
    
    # Convert back to Cartesian coordinates
    transformed = transformed_h[:, :2] / transformed_h[:, 2:3]
    
    return transformed

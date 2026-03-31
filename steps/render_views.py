"""Step 4 — Orthographic rendering of 3 views (left, right, top)."""

import logging
import os

# Must be set before any pyrender / OpenGL import
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import trimesh
import pyrender
from PIL import Image

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camera math
# ---------------------------------------------------------------------------

def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return a 4×4 camera-to-world matrix (OpenGL convention)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    # OpenGL: camera looks along -Z in its local frame.
    # Columns of the 3×3 block = camera axes in world coords.
    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = true_up
    mat[:3, 2] = -forward
    mat[:3, 3] = eye
    return mat


def _compute_ortho_mag(mesh: trimesh.Trimesh) -> float:
    """Compute xmag/ymag so the object fills OBJECT_FILL_RATIO of the frame."""
    extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    max_extent = float(extent.max())
    # mag is half the visible width — we want object to fill OBJECT_FILL_RATIO
    return (max_extent / 2.0) / config.OBJECT_FILL_RATIO

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def render_views(
    mesh_path: str,
    output_dir: str = config.DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Render left, right, top orthographic views.

    Returns dict mapping view name → saved image path.
    """
    logger.info("Loading mesh for rendering …")
    tm_mesh = trimesh.load(mesh_path, force="mesh")

    # Build pyrender mesh (with vertex colours if available)
    if tm_mesh.visual and hasattr(tm_mesh.visual, "vertex_colors"):
        pr_mesh = pyrender.Mesh.from_trimesh(tm_mesh, smooth=True)
    else:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.7, 0.7, 0.7, 1.0],
            metallicFactor=0.2,
            roughnessFactor=0.6,
        )
        pr_mesh = pyrender.Mesh.from_trimesh(tm_mesh, material=material, smooth=True)

    mag = _compute_ortho_mag(tm_mesh)
    logger.info("Orthographic mag = %.4f", mag)

    res = config.RENDER_RESOLUTION
    renderer = pyrender.OffscreenRenderer(viewport_width=res, viewport_height=res)

    saved: dict[str, str] = {}
    os.makedirs(output_dir, exist_ok=True)

    for view_name, cam_cfg in config.CAMERAS.items():
        scene = pyrender.Scene(
            ambient_light=np.array([config.AMBIENT_LIGHT_INTENSITY] * 3),
            bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        )
        scene.add(pr_mesh)

        # Orthographic camera — identical for every view
        camera = pyrender.OrthographicCamera(xmag=mag, ymag=mag, znear=0.01, zfar=100.0)
        cam_pose = _look_at(
            np.array(cam_cfg["eye"]),
            np.array(cam_cfg["target"]),
            np.array(cam_cfg["up"]),
        )
        scene.add(camera, pose=cam_pose)

        # Soft directional light co-located with camera
        dl = pyrender.DirectionalLight(
            color=np.ones(3),
            intensity=config.DIRECTIONAL_LIGHT_INTENSITY,
        )
        scene.add(dl, pose=cam_pose)

        # Render
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        # Build alpha from depth buffer
        alpha = (depth > 0).astype(np.uint8) * 255
        rgba = np.dstack([color[:, :, :3], alpha])

        img = Image.fromarray(rgba, "RGBA")
        out_path = os.path.join(output_dir, f"{view_name}.png")
        img.save(out_path)
        saved[view_name] = out_path
        logger.info("Rendered %s → %s", view_name, out_path)

    renderer.delete()
    return saved

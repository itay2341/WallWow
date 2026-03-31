"""Step 3 — Mesh normalisation: centre at origin, uniform scale, clean up."""

import logging

import numpy as np
import trimesh

import config

logger = logging.getLogger(__name__)


def process_mesh(mesh_path: str) -> str:
    """Normalise mesh in-place and return the same path."""

    logger.info("Loading mesh from %s", mesh_path)
    mesh: trimesh.Trimesh = trimesh.load(mesh_path, force="mesh")

    # Centre at origin
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    logger.info("Centred mesh (shifted by %s)", centroid)

    # Uniform scale to fit [-0.5, 0.5]^3
    extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    max_extent = extent.max()
    if max_extent > 0:
        mesh.vertices /= max_extent
    logger.info("Scaled mesh (max extent was %.4f)", max_extent)

    # Remove degenerate faces
    original_faces = len(mesh.faces)
    non_degenerate = mesh.nondegenerate_faces()
    if not non_degenerate.all():
        mesh.update_faces(non_degenerate)
    unique = mesh.unique_faces()
    mesh.update_faces(unique)
    removed = original_faces - len(mesh.faces)
    if removed:
        logger.info("Removed %d degenerate/duplicate faces", removed)

    # Fix normals
    trimesh.repair.fix_normals(mesh)

    mesh.export(mesh_path)
    logger.info("Saved normalised mesh → %s", mesh_path)
    return mesh_path

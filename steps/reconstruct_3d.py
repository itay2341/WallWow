"""Step 2 — Single image → 3D mesh via TripoSR."""

import logging
import os
import sys

import numpy as np
import torch
from PIL import Image

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# rembg helper — remove background & centre the foreground
# ---------------------------------------------------------------------------

def _remove_background(image: Image.Image) -> Image.Image:
    from rembg import remove

    return remove(image)


def _resize_foreground(image: Image.Image, ratio: float) -> Image.Image:
    """Crop to foreground bounding box, then paste centred onto a square canvas."""
    image = image.convert("RGBA")
    arr = np.array(image)
    alpha = arr[:, :, 3]
    coords = np.argwhere(alpha > 0)
    if len(coords) == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = image.crop((x0, y0, x1, y1))

    # Determine new size so object occupies `ratio` of the canvas
    max_side = max(cropped.size)
    new_size = int(max_side / ratio)
    canvas = Image.new("RGBA", (new_size, new_size), (127, 127, 127, 255))
    paste_x = (new_size - cropped.width) // 2
    paste_y = (new_size - cropped.height) // 2
    canvas.paste(cropped, (paste_x, paste_y), cropped)
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def reconstruct_3d(
    image_path: str,
    output_dir: str = config.DEFAULT_OUTPUT_DIR,
    device: str = "cuda",
) -> str:
    """Run TripoSR on *image_path* and return path to the exported mesh."""

    # Add TripoSR to path
    triposr_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "TripoSR")
    if triposr_dir not in sys.path:
        sys.path.insert(0, triposr_dir)

    from tsr.system import TSR  # type: ignore

    # Pre-process image
    logger.info("Removing background …")
    image = Image.open(image_path)
    image = _remove_background(image)
    image = _resize_foreground(image, config.FOREGROUND_RATIO)

    processed_path = os.path.join(output_dir, "processed_input.png")
    image.save(processed_path)
    logger.info("Saved processed input → %s", processed_path)

    # Load model
    logger.info("Loading TripoSR model …")
    model = TSR.from_pretrained(
        config.TRIPOSR_MODEL_ID,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)

    # Convert RGBA → RGB (TripoSR expects 3-channel input)
    if image.mode == "RGBA":
        # Composite over white background
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg.convert("RGB")
    else:
        image = image.convert("RGB")

    # Run inference
    logger.info("Running 3D reconstruction …")
    with torch.no_grad():
        scene_codes = model([image], device=device)

    # Extract mesh
    logger.info("Extracting mesh (MC resolution=%d) …", config.MC_RESOLUTION)
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=config.MC_RESOLUTION)
    mesh = meshes[0]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "mesh.obj")
    mesh.export(out_path)
    logger.info("Saved mesh → %s", out_path)

    # Free VRAM
    del model
    torch.cuda.empty_cache()

    return out_path

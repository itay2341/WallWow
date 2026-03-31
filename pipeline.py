"""Pipeline orchestrator — runs all steps sequentially with VRAM management."""

import logging
import os
import time
from datetime import datetime

import config
from steps.generate_image import generate_image
from steps.decompose_image import decompose_image
from steps.validate import validate

logger = logging.getLogger(__name__)


def run_pipeline(
    prompt: str,
    output_dir: str = config.DEFAULT_OUTPUT_DIR,
    device: str = "cuda",
    input_image: str | None = None,
    skip_validation: bool = False,
) -> dict[str, str]:
    """Execute the full pipeline and return paths to the 3 output images.

    Parameters
    ----------
    prompt : str
        Text description of the desired 3D object.
    output_dir : str
        Directory for all outputs.
    device : str
        PyTorch device ('cuda' or 'cpu').
    input_image : str | None
        If provided, skip image generation and use this image.
    skip_validation : bool
        If True, skip the validation step.
    """
    # Create a timestamped subfolder for this run
    run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Output folder: %s", run_dir)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Text → Image
    # ------------------------------------------------------------------
    if input_image:
        logger.info("Using provided image — skipping generation")
        image_path = input_image
    else:
        logger.info("═══ STEP 1 / 3 : Generating image ═══")
        t1 = time.time()
        image_path = generate_image(prompt, output_dir=run_dir, device=device)
        logger.info("Step 1 done in %.1fs", time.time() - t1)

    # ------------------------------------------------------------------
    # Step 2: Decompose isometric image → 3 face images
    # ------------------------------------------------------------------
    logger.info("═══ STEP 2 / 3 : Decomposing image into 3 faces ═══")
    t2 = time.time()
    image_paths = decompose_image(
        image_path, output_dir=run_dir, resolution=config.RENDER_RESOLUTION,
    )
    logger.info("Step 2 done in %.1fs", time.time() - t2)

    # ------------------------------------------------------------------
    # Step 3: Validation
    # ------------------------------------------------------------------
    if not skip_validation:
        logger.info("═══ STEP 3 / 3 : Validation ═══")
        t3 = time.time()
        passed = validate(image_paths)
        logger.info("Step 3 done in %.1fs", time.time() - t3)
        if not passed:
            logger.warning("Validation did not fully pass — outputs may need review")
    else:
        logger.info("═══ STEP 3 / 3 : Validation SKIPPED ═══")

    total = time.time() - t0
    logger.info("Pipeline finished in %.1fs", total)
    logger.info("Outputs:")
    for name, path in sorted(image_paths.items()):
        logger.info("  %s → %s", name, path)

    return image_paths

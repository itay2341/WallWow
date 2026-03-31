"""Step 3 — Validation: dimensions and fill ratio for decomposed faces."""

import logging

import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)


def _fill_ratio(img: Image.Image) -> float:
    alpha = np.array(img.convert("RGBA"))[:, :, 3] > 0
    return float(alpha.sum() / alpha.size)


def validate(image_paths: dict[str, str]) -> bool:
    """Run validation checks on the 3 face images.  Returns True if all pass."""
    ok = True

    # --- Dimensions ---
    dims = {n: Image.open(p).size for n, p in image_paths.items()}
    if len(set(dims.values())) != 1:
        logger.warning("Dimension mismatch: %s", dims)
        ok = False
    else:
        logger.info("Dimensions OK: %s", list(dims.values())[0])

    # --- Fill ratio ---
    for name in ("top", "right", "left"):
        img = Image.open(image_paths[name])
        fr = _fill_ratio(img)
        if fr < config.FILL_RATIO_MIN:
            logger.warning("%s fill ratio very low: %.2f", name, fr)
            ok = False
        else:
            logger.info("%s fill ratio: %.2f", name, fr)

    if ok:
        logger.info("All validation checks PASSED")
    else:
        logger.warning("Some validation checks FAILED — see warnings above")

    return ok

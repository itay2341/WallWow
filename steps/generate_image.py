"""Step 1 — Text prompt → 2D image via Stable Diffusion XL."""

import logging
import os

import torch
from PIL import Image

import config

logger = logging.getLogger(__name__)


def generate_image(
    prompt: str,
    output_dir: str = config.DEFAULT_OUTPUT_DIR,
    device: str = "cuda",
) -> str:
    """Generate a single 2D image from a text prompt.

    Returns the path to the saved image.
    """
    from diffusers import StableDiffusionXLPipeline

    logger.info("Loading Stable Diffusion XL …")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.SD_MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.enable_model_cpu_offload()

    full_prompt = config.PROMPT_TEMPLATE.format(subject=prompt)
    logger.info("Generating image for: %s", full_prompt)

    result = pipe(
        prompt=full_prompt,
        negative_prompt=config.NEGATIVE_PROMPT,
        num_inference_steps=config.SD_INFERENCE_STEPS,
        guidance_scale=config.SD_GUIDANCE_SCALE,
        height=config.RENDER_RESOLUTION,
        width=config.RENDER_RESOLUTION,
    )
    image: Image.Image = result.images[0]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "generated.png")
    image.save(out_path)
    logger.info("Saved generated image → %s", out_path)

    # Free VRAM
    del pipe
    torch.cuda.empty_cache()

    return out_path

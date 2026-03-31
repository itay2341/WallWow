"""Central configuration for the illusion image generator pipeline."""

# ---------- Resolution & Framing ----------
RENDER_RESOLUTION = 1024          # px, square output images

# ---------- Stable Diffusion ----------
SD_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

PROMPT_TEMPLATE = (
    "{subject}, 3D render of a compact solid cube shape, isometric corner view, "
    "equally showing top face and two side faces, three-quarter elevated angle, "
    "single centered object, clean hard edges, studio lighting, "
    "plain gray background, no clutter, highly detailed, minimalistic"
)

NEGATIVE_PROMPT = (
    "multiple objects, text, watermark, blurry, low quality, "
    "front view, side view, flat view, single face visible, "
    "background clutter, flat, 2D, cartoon, "
    "human, face, animal, organic, noisy, cropped"
)

SD_INFERENCE_STEPS = 30
SD_GUIDANCE_SCALE = 7.5

# ---------- Validation ----------
FILL_RATIO_MIN = 0.05             # warn if any face has less than 5% fill

# ---------- Paths ----------
DEFAULT_OUTPUT_DIR = "output"

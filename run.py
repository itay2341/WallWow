#!/usr/bin/env python3
"""CLI entry point for the illusion image generator."""

import argparse
import logging
import sys

import config
from pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3 geometrically-consistent orthographic images for a 3D illusion.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text description of the 3D object to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.DEFAULT_OUTPUT_DIR,
        help="Directory for output files (default: %(default)s).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override render resolution (default: %d)." % config.RENDER_RESOLUTION,
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="Path to an existing image — skips Stable Diffusion generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="PyTorch device (default: cuda).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the validation step.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args()

    # At least one input source is required
    if not args.prompt and not args.input_image:
        parser.error("Provide at least one of --prompt or --input-image.")

    # Override resolution if requested
    if args.resolution:
        config.RENDER_RESOLUTION = args.resolution

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    run_pipeline(
        prompt=args.prompt or "",
        output_dir=args.output_dir,
        device=args.device,
        input_image=args.input_image,
        skip_validation=args.no_validate,
    )


if __name__ == "__main__":
    main()

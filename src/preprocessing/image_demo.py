"""Run the fundus preprocessing pipeline and save original/enhanced comparisons."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

from src.preprocessing.image_pipeline import (
    PreprocessingConfig,
    preprocess_image,
    save_comparison_visualization,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess one retinal fundus image and save separate original, "
            "enhanced, and comparison files."
        )
    )
    parser.add_argument("image", type=Path, help="Path to a single fundus image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (defaults to <image parent>/preprocessed_<image stem>).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Enhanced output width in pixels (default: 512).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Enhanced output height in pixels (default: 512).",
    )
    parser.add_argument(
        "--disable-clahe",
        action="store_true",
        help="Disable local contrast enhancement.",
    )
    parser.add_argument(
        "--disable-illumination-normalization",
        action="store_true",
        help="Disable low-frequency illumination normalization.",
    )
    parser.add_argument(
        "--background-mode",
        choices=("black", "preserve"),
        default="black",
        help="Set detected non-retinal pixels to black or keep them unchanged.",
    )
    args = parser.parse_args(argv)

    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be positive.")
    source = args.image.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Input image does not exist: {source}")
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else source.parent / f"preprocessed_{source.stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    original_copy = output_dir / f"original{source.suffix or '.img'}"
    enhanced_path = output_dir / "enhanced.png"
    comparison_path = output_dir / "comparison.png"
    outputs = (original_copy, enhanced_path, comparison_path)

    if source in (path.resolve() for path in outputs):
        parser.error("Output directory would overwrite the original image.")
    existing_outputs = [str(path) for path in outputs if path.exists()]
    if existing_outputs:
        parser.error(
            "Refusing to overwrite existing output file(s): "
            + ", ".join(existing_outputs)
        )

    result = preprocess_image(
        source,
        PreprocessingConfig(
            output_size=(args.width, args.height),
            background_mode=args.background_mode,
            apply_clahe=not args.disable_clahe,
            normalize_illumination=not args.disable_illumination_normalization,
        ),
    )
    shutil.copy2(source, original_copy)
    result.enhanced_image.save(enhanced_path)
    save_comparison_visualization(
        result.original_image, result.enhanced_image, comparison_path
    )
    print(f"Original copy: {original_copy}")
    print(f"Enhanced image: {enhanced_path}")
    print(f"Comparison: {comparison_path}")
    print(
        "Normalized array: "
        f"{result.normalized_image.shape} {result.normalized_image.dtype} "
        f"range=[{result.normalized_image.min():.3f}, "
        f"{result.normalized_image.max():.3f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

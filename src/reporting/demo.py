"""Generate a screening report for the first image in a local dataset folder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.dr_classification.classifier import load_model_checkpoint
from src.reporting.screening_report import (
    create_screening_report,
    write_screening_report,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _find_sample_image(dataset_root: Path) -> Path:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
    excluded_parts = {"mask", "masks", "annotation", "annotations", "ground_truth"}
    candidates = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.lower() in excluded_parts for part in path.parts)
    )
    if not candidates:
        raise FileNotFoundError(
            f"No sample fundus image found under local dataset: {dataset_root}"
        )
    return candidates[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create JSON, Markdown, and PDF screening reports from a local dataset sample."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/aptos-20152019-preprocessed"),
        help="Local dataset directory; the first supported image is selected.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/dr_resnet18/best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/screening_demo"),
    )
    parser.add_argument(
        "--calibration-temperature",
        type=float,
        help="Optional temperature fitted on a validation split for this checkpoint.",
    )
    args = parser.parse_args(argv)
    try:
        image_path = _find_sample_image(args.dataset_root)
    except FileNotFoundError as error:
        parser.error(str(error))
    if not args.checkpoint.is_file():
        parser.error(
            f"Trained checkpoint not found: {args.checkpoint}. "
            "Train the classifier or provide --checkpoint."
        )
    model, config, device = load_model_checkpoint(args.checkpoint)
    result = create_screening_report(
        image_path,
        model,
        config,
        device,
        args.output_dir,
        calibration_temperature=args.calibration_temperature,
    )
    paths = write_screening_report(result, args.output_dir)
    print(f"Sample image selected from local dataset: {image_path}")
    print(f"JSON result: {paths['json']}")
    print(f"Human-readable report: {paths['markdown']}")
    print(f"PDF report: {paths['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

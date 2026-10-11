"""Command-line demo for assessing one retinal fundus image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.image_quality_assessment.quality import (
    QualityThresholds,
    assess_image,
    save_diagnostic_visualization,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run engineering-oriented quality checks on one retinal fundus image. "
            "This is not a clinically validated gradability assessment."
        )
    )
    parser.add_argument("image", type=Path, help="Path to a single image file.")
    parser.add_argument(
        "--thresholds-json",
        type=Path,
        help="Optional JSON object overriding named QualityThresholds fields.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full machine-readable result instead of a text report.",
    )
    parser.add_argument(
        "--save-visualization",
        type=Path,
        help="Optional path for an annotated copy of the input image.",
    )
    args = parser.parse_args(argv)

    thresholds = None
    if args.thresholds_json is not None:
        values = json.loads(args.thresholds_json.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            parser.error("--thresholds-json must contain a JSON object.")
        thresholds = QualityThresholds.from_dict(values)

    assessment = assess_image(args.image, thresholds)
    if args.json:
        print(json.dumps(assessment.to_dict(), indent=2))
    else:
        print(assessment.format_report())
    if args.save_visualization is not None:
        save_diagnostic_visualization(
            args.image, args.save_visualization, assessment
        )
        print(f"\nAnnotated image saved to: {args.save_visualization}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

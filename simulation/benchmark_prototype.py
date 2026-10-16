"""Measure the prototype's image preprocessing + DR prediction latency."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dr_classification.classifier import (
    load_model_checkpoint,
    predict_image,
)
from src.image_quality_assessment.quality import assess_image
from src.preprocessing.image_pipeline import preprocess_image


def measure(
    function: Callable[[], object],
    warmups: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmups):
        function()
    measurements = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        measurements.append(time.perf_counter() - started)
    ordered = sorted(measurements)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_seconds_per_image": statistics.median(measurements),
        "p95_seconds_per_image": ordered[p95_index],
        "minimum_seconds_per_image": min(measurements),
        "maximum_seconds_per_image": max(measurements),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Authorized local test image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional trained checkpoint for measuring prediction latency.",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be non-negative and --repeats must be positive.")
    if not args.image.is_file():
        parser.error(f"Image file not found: {args.image}")

    result: dict[str, Any] = {
        "input_image_size_bytes": args.image.stat().st_size,
        "warmup_runs": args.warmups,
        "measured_runs": args.repeats,
        "image_quality_assessment": measure(
            lambda: assess_image(args.image),
            args.warmups,
            args.repeats,
        ),
        "image_preprocessing": measure(
            lambda: preprocess_image(args.image),
            args.warmups,
            args.repeats,
        ),
    }
    with Image.open(args.image) as input_image:
        result["input_image_dimensions"] = {
            "width": input_image.width,
            "height": input_image.height,
        }
    checkpoint = args.checkpoint or Path(
        "models/checkpoints/dr_resnet18/best_model.pt"
    )
    if checkpoint.is_file():
        model, model_config, device = load_model_checkpoint(checkpoint)
        result["prediction"] = {
            "status": "measured",
            "device": str(device),
            **measure(
                lambda: predict_image(args.image, model, model_config, device),
                args.warmups,
                args.repeats,
            ),
        }
    else:
        result["prediction"] = {
            "status": "not measured",
            "reason": (
                f"Trained checkpoint not found: {checkpoint}. "
                "Pass --checkpoint to measure preprocessing plus prediction."
            ),
        }

    result["note"] = (
        "Local engineering timings only. Image-quality assessment and "
        "preprocessing are measured separately. When a checkpoint is supplied, "
        "prediction timing also includes preprocessing. Timings exclude Grad-CAM, "
        "retinal evidence generation, network transmission, clinical review, and "
        "operational delays. Not a clinical or service-capacity guarantee."
    )
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()

"""Run single-image DR severity inference from a saved checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.dr_classification.classifier import load_model_checkpoint, predict_image


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Predict one of five DR severity levels using an experimental model. "
            "Predictions are not a clinical diagnosis."
        )
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional JSON output file.")
    parser.add_argument(
        "--calibration-temperature",
        type=float,
        help=(
            "Optional temperature fitted on the validation split (for example, "
            "from evaluation metrics JSON)."
        ),
    )
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    model, config, device = load_model_checkpoint(args.checkpoint)
    prediction = predict_image(
        args.image,
        model,
        config,
        device,
        calibration_temperature=args.calibration_temperature,
    )
    prediction["performance_scope"] = (
        "Experimental model output; not clinically validated or a diagnosis."
    )
    serialized = json.dumps(prediction, indent=2)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.resolve() in (args.image.resolve(), args.checkpoint.resolve()):
            parser.error("Output path must not overwrite the input image or checkpoint.")
        if args.output.exists():
            parser.error(f"Refusing to overwrite existing output file: {args.output}")
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Inference result saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

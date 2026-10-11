"""Command-line entry point for auditing and splitting the configured dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.preprocessing.dataset import (
    aptos_preprocessed_config,
    load_dataset,
    summarize_dataset,
)
from src.preprocessing.splitting import create_splits, save_split_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the APTOS preprocessed dataset and create split metadata."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/aptos-20152019-preprocessed"),
        help="Directory containing trainData.csv, testData.csv, and image folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/metadata"),
        help="Directory for the summary and split metadata (no images are copied).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()

    config = aptos_preprocessed_config(args.dataset_root)
    try:
        args.output_dir.resolve().relative_to(config.root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output-dir must not be inside the source dataset directory.")

    records = load_dataset(config)
    summary = summarize_dataset(records, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    invalid_count = (
        len(summary["missing_images"])
        + len(summary["corrupted_images"])
        + len(summary["unreadable_images"])
    )
    if invalid_count:
        raise RuntimeError(
            f"Found {invalid_count} missing, corrupted, or unreadable images. "
            f"See {summary_path}; split metadata was not written."
        )

    splits = create_splits(
        records,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    save_split_metadata(
        splits,
        args.output_dir / "split_metadata.csv",
        args.output_dir / "split_config.json",
        config,
        args.validation_fraction,
        args.seed,
    )
    print(f"Dataset summary: {summary_path}")
    print(f"Split metadata: {args.output_dir / 'split_metadata.csv'}")
    print(
        "Split counts: "
        + ", ".join(f"{name}={len(items)}" for name, items in splits.items())
    )


if __name__ == "__main__":
    main()

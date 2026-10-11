"""Evaluate a saved classifier on the held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from src.dr_classification.classifier import (
    FundusClassificationDataset,
    calculate_metrics,
    class_weights,
    load_model_checkpoint,
    save_confusion_matrix,
    validate_split_separation,
)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataset_root: str | Path,
    split_csv: str | Path,
    output_dir: str | Path,
    split: str = "test",
) -> dict[str, object]:
    model, config, device = load_model_checkpoint(checkpoint_path)
    datasets = {
        name: FundusClassificationDataset(dataset_root, split_csv, name, config)
        for name in ("train", "validation", "test")
    }
    validate_split_separation(datasets)
    dataset = datasets[split]
    loss_weights = class_weights(
        datasets["train"].class_counts(config.num_classes),
        device,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += float(loss) * batch_size
            total_examples += batch_size
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = calculate_metrics(targets, predictions)
    metrics["loss"] = total_loss / total_examples
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / f"{split}_metrics.json"
    metrics_payload = {
        **metrics,
        "split": split,
        "checkpoint": str(Path(checkpoint_path)),
        "performance_scope": (
            "Experimental held-out dataset results; not clinically validated."
        ),
        "class_names": list(config.class_names),
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    matrix = metrics["confusion_matrix"]
    matrix_csv = output_path / f"{split}_confusion_matrix.csv"
    with matrix_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *config.class_names])
        for name, row in zip(config.class_names, matrix):
            writer.writerow([name, *row])
    matrix_image = output_path / f"{split}_confusion_matrix.png"
    save_confusion_matrix(matrix, config.class_names, matrix_image)
    return {
        "metrics": metrics_path,
        "confusion_matrix_csv": matrix_csv,
        "confusion_matrix_image": matrix_image,
        **metrics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an experimental DR classifier on a dataset split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/aptos-20152019-preprocessed"),
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("data/metadata/split_metadata.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/dr_classification"))
    parser.add_argument(
        "--split",
        choices=("test", "validation"),
        default="test",
        help="Use test for final evaluation; validation is for development only.",
    )
    args = parser.parse_args(argv)
    result = evaluate_checkpoint(
        args.checkpoint,
        args.dataset_root,
        args.split_csv,
        args.output_dir,
        args.split,
    )
    print(f"Metrics: {result['metrics']}")
    print(f"Confusion matrix CSV: {result['confusion_matrix_csv']}")
    print(f"Confusion matrix image: {result['confusion_matrix_image']}")
    print(
        "Measured metrics: "
        + ", ".join(
            f"{key}={result[key]:.4f}"
            for key in ("loss", "accuracy", "precision", "recall", "f1")
        )
    )
    print("Results are experimental and are not clinical validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

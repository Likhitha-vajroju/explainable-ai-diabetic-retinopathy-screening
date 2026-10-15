"""Evaluate a saved classifier on validation or held-out test data."""

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
    class_weights,
    load_model_checkpoint,
    validate_split_separation,
)
from src.dr_classification.calibration import (
    apply_temperature,
    calibration_summary,
    fit_temperature,
    save_reliability_diagram,
)
from src.dr_classification.evaluation_metrics import evaluate_predictions
from src.dr_classification.evaluation_reporting import (
    build_evaluation_report,
    save_evaluation_dashboard,
)


def _class_distribution(dataset: FundusClassificationDataset) -> dict[str, object]:
    counts = dataset.class_counts(5)
    nonzero = [count for count in counts if count > 0]
    return {
        "counts": {str(index): count for index, count in enumerate(counts)},
        "largest_to_smallest_nonzero_ratio": (
            max(nonzero) / min(nonzero) if nonzero else None
        ),
    }


def _checkpoint_epoch(checkpoint_path: str | Path) -> int | None:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    return int(epoch) if epoch is not None else None


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataset_root: str | Path,
    split_csv: str | Path,
    output_dir: str | Path,
    split: str = "test",
) -> dict[str, object]:
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be 'validation' or 'test'.")
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
    probability_rows: list[list[float]] = []
    logit_rows: list[list[float]] = []
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
            logit_rows.extend(logits.cpu().tolist())
            probability_rows.extend(torch.softmax(logits, dim=1).cpu().tolist())

    if split == "test":
        calibration_dataset = datasets["validation"]
        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        calibration_targets: list[int] = []
        calibration_logits: list[list[float]] = []
        with torch.inference_mode():
            for images, labels in calibration_loader:
                logits = model(
                    images.to(device, non_blocking=device.type == "cuda")
                )
                calibration_targets.extend(labels.tolist())
                calibration_logits.extend(logits.cpu().tolist())
        temperature_fit_split = "validation"
    else:
        calibration_targets = targets
        calibration_logits = logit_rows
        temperature_fit_split = split

    temperature = fit_temperature(calibration_logits, calibration_targets)
    calibrated_probability_rows = apply_temperature(logit_rows, temperature).tolist()
    raw_calibration = calibration_summary(probability_rows, targets)
    calibrated_calibration = calibration_summary(calibrated_probability_rows, targets)

    metrics = evaluate_predictions(targets, probability_rows, config.class_names)
    metrics["confidence_calibration"] = {
        "method": "Multiclass scalar temperature scaling fitted by validation NLL.",
        "temperature": temperature,
        "fit_split": temperature_fit_split,
        "evaluation_split": split,
        "fit_and_evaluation_split_overlap": temperature_fit_split == split,
        "raw": raw_calibration,
        "calibrated": calibrated_calibration,
        "interpretation": (
            "Calibrated probabilities are experimental estimates of model "
            "confidence, not clinical certainty."
        ),
    }
    metrics["loss"] = total_loss / total_examples
    metrics["loss_definition"] = (
        "Mean weighted cross-entropy using inverse-frequency weights from train only."
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / f"{split}_metrics.json"
    metrics_payload = {
        **metrics,
        "split": split,
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_epoch": _checkpoint_epoch(checkpoint_path),
        "performance_scope": (
            "Experimental held-out dataset results; not clinically validated."
        ),
        "class_names": list(config.class_names),
        "dataset_split_support": {
            name: len(split_dataset) for name, split_dataset in datasets.items()
        },
        "class_imbalance": {
            name: _class_distribution(split_dataset)
            for name, split_dataset in datasets.items()
        },
        "targets": {
            "referable_sensitivity_strictly_above": 0.90,
            "referable_specificity_strictly_above": 0.85,
        },
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
    dashboard_path = output_path / f"{split}_evaluation_dashboard.png"
    save_evaluation_dashboard(metrics, config.class_names, dashboard_path)
    reliability_path = output_path / f"{split}_reliability_diagram.png"
    save_reliability_diagram(
        raw_calibration,
        calibrated_calibration,
        reliability_path,
    )
    report_path = output_path / f"{split}_evaluation_report.md"
    report_path.write_text(
        build_evaluation_report(metrics_payload),
        encoding="utf-8",
    )
    return {
        "metrics": metrics_path,
        "confusion_matrix_csv": matrix_csv,
        "evaluation_dashboard": dashboard_path,
        "reliability_diagram": reliability_path,
        "report": report_path,
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/dr_classification"),
    )
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
    print(f"Evaluation dashboard: {result['evaluation_dashboard']}")
    print(f"Reliability diagram: {result['reliability_diagram']}")
    print(f"Evaluation report: {result['report']}")
    binary = result["binary_referable"]
    sensitivity = binary["sensitivity"]
    specificity = binary["specificity"]
    sensitivity_status = (
        "NOT EVALUABLE"
        if sensitivity is None
        else "MET"
        if sensitivity > 0.90
        else "NOT MET"
    )
    specificity_status = (
        "NOT EVALUABLE"
        if specificity is None
        else "MET"
        if specificity > 0.85
        else "NOT MET"
    )
    print(
        "Referable DR measured: "
        f"sensitivity={sensitivity}, specificity={specificity}, "
        f"precision={binary['precision']}, f1={binary['f1']}, "
        f"roc_auc={binary['roc_auc']}"
    )
    print(
        f"Targets (>90% sensitivity / >85% specificity): "
        f"{sensitivity_status} / {specificity_status}."
    )
    print("Results are experimental; they are not clinical validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

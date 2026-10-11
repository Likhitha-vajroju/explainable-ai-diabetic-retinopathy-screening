"""Training and validation loops for the five-level DR classifier."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dr_classification.classifier import (
    ClassificationConfig,
    FundusClassificationDataset,
    build_model,
    calculate_metrics,
    checkpoint_payload,
    class_weights,
    save_training_curves,
    resolve_device,
    seed_everything,
    validate_split_separation,
)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    targets: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())

    metrics = calculate_metrics(targets, predictions)
    metrics["loss"] = total_loss / total_examples
    return metrics


def _write_history(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        raise ValueError("No training history to write.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_model(
    dataset_root: str | Path,
    split_csv: str | Path,
    output_dir: str | Path,
    config: ClassificationConfig,
) -> dict[str, Any]:
    """Train on train rows, select by validation macro F1, and never use test rows."""
    seed_everything(config.seed)
    device = resolve_device(config.device)

    datasets = {
        split: FundusClassificationDataset(dataset_root, split_csv, split, config)
        for split in ("train", "validation", "test")
    }
    validate_split_separation(datasets)
    counts = datasets["train"].class_counts(config.num_classes)
    weights = class_weights(counts, device)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_validation_f1 = -1.0
    checkpoint_path = output_path / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device
        )
        history_row = {"epoch": float(epoch)}
        history_row.update(
            {f"train_{key}": float(train_metrics[key]) for key in
             ("loss", "accuracy", "precision", "recall", "f1")}
        )
        history_row.update(
            {f"validation_{key}": float(validation_metrics[key]) for key in
             ("loss", "accuracy", "precision", "recall", "f1")}
        )
        history.append(history_row)
        print(
            f"Epoch {epoch}/{config.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_accuracy={train_metrics['accuracy']:.4f} "
            f"train_precision={train_metrics['precision']:.4f} "
            f"train_recall={train_metrics['recall']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} "
            f"validation_loss={validation_metrics['loss']:.4f} "
            f"validation_accuracy={validation_metrics['accuracy']:.4f} "
            f"validation_precision={validation_metrics['precision']:.4f} "
            f"validation_recall={validation_metrics['recall']:.4f} "
            f"validation_f1={validation_metrics['f1']:.4f}"
        )
        if validation_metrics["f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["f1"]
            torch.save(
                checkpoint_payload(model, config, epoch, validation_metrics),
                checkpoint_path,
            )

    history_path = output_path / "history.csv"
    _write_history(history, history_path)
    curves_path = output_path / "training_curves.png"
    save_training_curves(history, curves_path)
    return {
        "checkpoint": checkpoint_path,
        "history": history_path,
        "curves": curves_path,
        "best_validation_f1": best_validation_f1,
    }

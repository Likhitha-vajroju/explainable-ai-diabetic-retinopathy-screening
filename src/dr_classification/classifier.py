"""Transfer-learning model, dataset, metrics, and single-image inference."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models

from src.preprocessing.image_pipeline import (
    PreprocessingConfig,
    preprocess_image,
)


DEFAULT_CLASS_TO_INDEX = {str(index): index for index in range(5)}
DEFAULT_CLASS_NAMES = (
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
)
SUPPORTED_ARCHITECTURES = ("resnet18", "resnet50", "efficientnet_b0")
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]


@dataclass(frozen=True)
class ClassificationConfig:
    """Validated settings for DR transfer learning and preprocessing."""

    architecture: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 5
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    class_to_index: Mapping[str, int] = field(
        default_factory=lambda: DEFAULT_CLASS_TO_INDEX.copy()
    )
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
    preprocessing: PreprocessingConfig = field(
        default_factory=lambda: PreprocessingConfig(output_size=(224, 224))
    )

    def __post_init__(self) -> None:
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {', '.join(SUPPORTED_ARCHITECTURES)}."
            )
        if self.num_classes != 5:
            raise ValueError("The DR classifier requires exactly five output classes.")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32 pixels.")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
        if sorted(self.class_to_index.values()) != list(range(self.num_classes)):
            raise ValueError("class_to_index must map exactly five classes to indices 0-4.")
        if len(self.class_names) != self.num_classes or any(
            not name for name in self.class_names
        ):
            raise ValueError("class_names must provide one non-empty name per class.")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ClassificationConfig:
        data = dict(values)
        preprocessing_values = dict(data.pop("preprocessing", {}))
        image_size = data.get("image_size", 224)
        preprocessing_values.setdefault("output_size", (image_size, image_size))
        if "output_size" in preprocessing_values:
            output_size = preprocessing_values["output_size"]
            preprocessing_values["output_size"] = (
                tuple(output_size) if output_size is not None else None
            )
        if preprocessing_values["output_size"] is not None:
            preprocessing_values["output_size"] = (image_size, image_size)
        if "clahe_tile_grid" in preprocessing_values:
            preprocessing_values["clahe_tile_grid"] = tuple(
                preprocessing_values["clahe_tile_grid"]
            )
        data["preprocessing"] = PreprocessingConfig(**preprocessing_values)
        if "class_names" in data:
            data["class_names"] = tuple(data["class_names"])
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> ClassificationConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Classification configuration must be a JSON object.")
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["class_to_index"] = dict(self.class_to_index)
        values["class_names"] = list(self.class_names)
        return values


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device)


def build_model(config: ClassificationConfig) -> nn.Module:
    """Build a supported torchvision classifier with a five-class output head."""
    if config.architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if config.pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    elif config.architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if config.pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    else:
        weights = models.EfficientNet_B0_Weights.DEFAULT if config.pretrained else None
        model = models.efficientnet_b0(weights=weights)
        classifier = model.classifier[-1]
        model.classifier[-1] = nn.Linear(classifier.in_features, config.num_classes)
    return model


class FundusClassificationDataset(Dataset[tuple[torch.Tensor, int]]):
    """Read split-metadata rows and apply the project's existing preprocessing."""

    def __init__(
        self,
        dataset_root: str | Path,
        split_csv: str | Path,
        split: str,
        config: ClassificationConfig,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.split_csv = Path(split_csv)
        self.split = split
        self.config = config
        self.rows: list[dict[str, str]] = []
        with self.split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"split", "target_label", "image_path", "group_id"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    "Split CSV must contain columns: "
                    + ", ".join(sorted(required))
                )
            self.rows = [row for row in reader if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"Split metadata contains no rows for split {split!r}.")

        self.samples: list[tuple[Path, int]] = []
        self.group_ids: set[str] = set()
        for row in self.rows:
            label = row.get("target_label") or row.get("label", "")
            if label not in config.class_to_index:
                raise ValueError(
                    f"Split row label {label!r} is not in class_to_index."
                )
            relative_path = row.get("image_path", "")
            if not relative_path:
                raise ValueError(f"Split row has an empty image_path for label {label!r}.")
            image_path = (self.dataset_root / relative_path).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Fundus image does not exist: {image_path}")
            self.samples.append((image_path, config.class_to_index[label]))
            self.group_ids.add(row["group_id"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        result = preprocess_image(image_path, self.config.preprocessing)
        pixels = np.ascontiguousarray(result.normalized_image.transpose(2, 0, 1))
        pixels = np.ascontiguousarray((pixels - IMAGENET_MEAN) / IMAGENET_STD)
        return torch.from_numpy(pixels), label

    def class_counts(self, num_classes: int) -> list[int]:
        counts = Counter(label for _, label in self.samples)
        return [counts[index] for index in range(num_classes)]


def validate_split_separation(
    datasets: Mapping[str, FundusClassificationDataset],
) -> None:
    names = tuple(datasets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left, right = datasets[left_name], datasets[right_name]
            left_ids = {
                row.get("sample_id", "")
                for row in left.rows
                if row.get("sample_id")
            }
            right_ids = {
                row.get("sample_id", "")
                for row in right.rows
                if row.get("sample_id")
            }
            duplicate_ids = left_ids & right_ids
            if duplicate_ids:
                raise ValueError(
                    f"Sample IDs overlap between {left_name} and {right_name}: "
                    f"{sorted(duplicate_ids)[:5]}"
                )
            overlap = left.group_ids & right.group_ids
            if overlap:
                raise ValueError(
                    f"Group IDs overlap between {left_name} and {right_name}: "
                    f"{sorted(overlap)[:5]}"
                )


def class_weights(class_counts: list[int], device: torch.device) -> torch.Tensor:
    """Return inverse-frequency balanced weights estimated only from train labels."""
    if len(class_counts) != 5 or any(count < 1 for count in class_counts):
        raise ValueError(
            "Training split must contain at least one example from each of five classes; "
            f"observed counts: {class_counts}."
        )
    total = sum(class_counts)
    number_of_classes = len(class_counts)
    return torch.tensor(
        [total / (number_of_classes * count) for count in class_counts],
        dtype=torch.float32,
        device=device,
    )


def calculate_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    losses: list[float] | None = None,
) -> dict[str, Any]:
    """Calculate accuracy and fixed-five-class macro precision, recall, and F1."""
    target_array = np.asarray(targets, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    if target_array.shape != prediction_array.shape or target_array.size == 0:
        raise ValueError("Targets and predictions must be non-empty and have equal shape.")
    if (
        np.any(target_array < 0)
        or np.any(target_array >= 5)
        or np.any(prediction_array < 0)
        or np.any(prediction_array >= 5)
    ):
        raise ValueError("Targets and predictions must use class indices 0 through 4.")

    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (target_array, prediction_array), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted_count = confusion.sum(axis=0)
    support = confusion.sum(axis=1)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(5, dtype=np.float64),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(5, dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(5, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    result: dict[str, Any] = {
        "accuracy": float(np.mean(target_array == prediction_array)),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "support": int(target_array.size),
        "per_class": [
            {
                "class_index": index,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(5)
        ],
        "confusion_matrix": confusion.tolist(),
    }
    if losses:
        result["loss"] = float(np.mean(losses))
    return result


def prediction_result(
    probabilities: list[float] | np.ndarray | torch.Tensor,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Format predicted severity, probabilities, and uncalibrated max-softmax confidence."""
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (5,) or len(class_names) != 5:
        raise ValueError("Inference requires five probabilities and five class names.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Probabilities must be finite, non-negative values.")
    total = float(values.sum())
    if not np.isclose(total, 1.0, atol=1e-5):
        raise ValueError("Class probabilities must sum to 1.")
    predicted_index = int(np.argmax(values))
    return {
        "predicted_dr_level": predicted_index,
        "predicted_label": class_names[predicted_index],
        "class_probabilities": [
            {
                "dr_level": index,
                "label": class_names[index],
                "probability": float(values[index]),
            }
            for index in range(5)
        ],
        "raw_confidence": float(values[predicted_index]),
        "confidence_interpretation": (
            "Maximum softmax probability; uncalibrated and not a clinical confidence."
        ),
    }


def checkpoint_payload(
    model: nn.Module,
    config: ClassificationConfig,
    epoch: int,
    validation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "epoch": epoch,
        "validation_metrics": dict(validation_metrics),
        "performance_scope": (
            "Experimental validation-split results only; not clinical validation."
        ),
    }


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, ClassificationConfig, torch.device]:
    target_device = device or torch.device("cpu")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=target_device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=target_device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain a model_state_dict.")
    config = ClassificationConfig.from_dict(checkpoint["config"])
    inference_device = resolve_device(config.device) if device is None else target_device
    model = build_model(
        ClassificationConfig.from_dict({**config.to_dict(), "pretrained": False})
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(inference_device)
    model.eval()
    return model, config, inference_device


def predict_image(
    image_path: str | Path,
    model: nn.Module,
    config: ClassificationConfig,
    device: torch.device,
) -> dict[str, Any]:
    processed = preprocess_image(image_path, config.preprocessing)
    pixels = np.ascontiguousarray(processed.normalized_image.transpose(2, 0, 1))
    pixels = np.ascontiguousarray((pixels - IMAGENET_MEAN) / IMAGENET_STD)
    input_tensor = torch.from_numpy(pixels).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]
    return prediction_result(
        probabilities.detach().cpu().numpy(),
        config.class_names,
    )


def save_confusion_matrix(
    confusion_matrix: list[list[int]],
    class_names: tuple[str, ...],
    output_path: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(confusion_matrix, cmap="Blues")
    axis.set(
        xticks=range(5),
        yticks=range(5),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted DR level",
        ylabel="True DR level",
        title="Experimental DR classification confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for row in range(5):
        for column in range(5):
            axis.text(column, row, str(confusion_matrix[row][column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_training_curves(history: list[dict[str, float]], output_path: str | Path) -> None:
    if not history:
        raise ValueError("Cannot save training curves without epoch history.")
    metrics = ("loss", "accuracy", "precision", "recall", "f1")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    epochs = [int(row["epoch"]) for row in history]
    for axis, metric in zip(axes.flat, metrics):
        axis.plot(epochs, [row[f"train_{metric}"] for row in history], label="train")
        axis.plot(
            epochs, [row[f"validation_{metric}"] for row in history], label="validation"
        )
        axis.set(title=metric.replace("_", " ").title(), xlabel="Epoch")
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes.flat[-1].axis("off")
    figure.suptitle("Experimental training and validation metrics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

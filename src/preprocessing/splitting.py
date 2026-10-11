"""Reproducible, group-aware stratification for dataset records."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.preprocessing.dataset import DatasetConfig, DatasetRecord


def _partition_groups(
    records: list[DatasetRecord], validation_fraction: float, seed: int
) -> tuple[list[DatasetRecord], list[DatasetRecord]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    grouped_records: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        grouped_records[record.group_id].append(record)
    if len(grouped_records) < 2:
        raise ValueError("At least two distinct groups are required for a split.")

    total_by_label = Counter(record.label for record in records)
    group_items = list(grouped_records.items())
    rng = random.Random(seed)
    tie_breakers = {group_id: rng.random() for group_id, _ in group_items}

    def rarity(item: tuple[str, list[DatasetRecord]]) -> tuple[float, int, float]:
        group_id, group_records = item
        group_counts = Counter(record.label for record in group_records)
        rarity_score = sum(
            count / total_by_label[label] for label, count in group_counts.items()
        )
        return rarity_score, len(group_records), tie_breakers[group_id]

    group_items.sort(key=rarity, reverse=True)
    target_validation = {
        label: count * validation_fraction for label, count in total_by_label.items()
    }
    target_train = {
        label: count * (1 - validation_fraction)
        for label, count in total_by_label.items()
    }
    assigned = {"train": Counter(), "validation": Counter()}
    target_sizes = {
        "train": len(records) * (1 - validation_fraction),
        "validation": len(records) * validation_fraction,
    }
    targets = {"train": target_train, "validation": target_validation}
    assigned_sizes = {"train": 0, "validation": 0}
    split_records: dict[str, list[DatasetRecord]] = {
        "train": [],
        "validation": [],
    }

    for _, group_records in group_items:
        group_counts = Counter(record.label for record in group_records)
        scores: dict[str, float] = {}
        for candidate in ("train", "validation"):
            total_error = 0.0
            for split_name in ("train", "validation"):
                class_error = 0.0
                for label in total_by_label:
                    increment = group_counts[label] if split_name == candidate else 0
                    target = targets[split_name][label]
                    projected = assigned[split_name][label] + increment
                    class_error += ((projected - target) / max(target, 1.0)) ** 2
                size_increment = len(group_records) if split_name == candidate else 0
                projected_size = assigned_sizes[split_name] + size_increment
                size_error = (
                    (projected_size - target_sizes[split_name])
                    / max(target_sizes[split_name], 1.0)
                ) ** 2
                total_error += class_error + 0.25 * size_error
            scores[candidate] = total_error

        chosen = min(
            ("train", "validation"),
            key=lambda split_name: (scores[split_name], split_name),
        )
        split_records[chosen].extend(group_records)
        assigned[chosen].update(group_counts)
        assigned_sizes[chosen] += len(group_records)

    validation_sample_ids = {
        record.sample_id for record in split_records["validation"]
    }
    validation_groups = {
        group_id
        for group_id, group_records in grouped_records.items()
        if group_records[0].sample_id in validation_sample_ids
    }
    validation_counts = Counter(
        record.label for record in split_records["validation"]
    )
    validation_size = len(split_records["validation"])

    # Improve class and size balance by moving whole groups only.
    while True:
        best_delta = 0.0
        best_group: str | None = None
        best_direction = 0
        for group_id in sorted(grouped_records):
            group_counts = Counter(
                record.label for record in grouped_records[group_id]
            )
            direction = -1 if group_id in validation_groups else 1
            delta = 0.0
            for label, amount in group_counts.items():
                target_validation_count = target_validation[label]
                target_training_count = target_train[label]
                current_validation = validation_counts[label]
                updated_validation = current_validation + direction * amount
                current_training = total_by_label[label] - current_validation
                updated_training = total_by_label[label] - updated_validation
                delta += (
                    (updated_validation - target_validation_count)
                    / max(target_validation_count, 1.0)
                ) ** 2 - (
                    (current_validation - target_validation_count)
                    / max(target_validation_count, 1.0)
                ) ** 2
                delta += (
                    (updated_training - target_training_count)
                    / max(target_training_count, 1.0)
                ) ** 2 - (
                    (current_training - target_training_count)
                    / max(target_training_count, 1.0)
                ) ** 2

            updated_size = validation_size + direction * len(
                grouped_records[group_id]
            )
            current_training_size = len(records) - validation_size
            updated_training_size = len(records) - updated_size
            delta += 0.25 * (
                (
                    (updated_size - target_sizes["validation"])
                    / max(target_sizes["validation"], 1.0)
                )
                ** 2
                - (
                    (validation_size - target_sizes["validation"])
                    / max(target_sizes["validation"], 1.0)
                )
                ** 2
                + (
                    (updated_training_size - target_sizes["train"])
                    / max(target_sizes["train"], 1.0)
                )
                ** 2
                - (
                    (current_training_size - target_sizes["train"])
                    / max(target_sizes["train"], 1.0)
                )
                ** 2
            )
            if delta < best_delta:
                best_delta = delta
                best_group = group_id
                best_direction = direction

        if best_group is None:
            break
        group_counts = Counter(
            record.label for record in grouped_records[best_group]
        )
        validation_counts.update(
            {
                label: best_direction * amount
                for label, amount in group_counts.items()
            }
        )
        validation_size += best_direction * len(grouped_records[best_group])
        if best_direction < 0:
            validation_groups.remove(best_group)
        else:
            validation_groups.add(best_group)

    train_records = [
        record for record in records if record.group_id not in validation_groups
    ]
    validation_records = [
        record for record in records if record.group_id in validation_groups
    ]
    if not train_records or not validation_records:
        raise ValueError("Unable to populate both train and validation splits.")
    return train_records, validation_records


def create_splits(
    records: list[DatasetRecord],
    validation_fraction: float = 0.15,
    seed: int = 42,
    training_source: str = "train",
    testing_source: str = "test",
) -> dict[str, list[DatasetRecord]]:
    """Split the published training source and preserve its held-out test source."""
    identifiers = [record.sample_id for record in records]
    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(identifiers).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"Duplicate sample identifiers cannot be split: {duplicate_ids}")
    if any(record.image_path is None for record in records):
        raise ValueError("Resolve missing images before creating dataset splits.")

    training_records = [record for record in records if record.source == training_source]
    testing_records = [record for record in records if record.source == testing_source]
    unknown_records = [
        record
        for record in records
        if record.source not in {training_source, testing_source}
    ]
    if not training_records or not testing_records or unknown_records:
        raise ValueError(
            "Records must belong to exactly the configured training and testing sources."
        )

    train, validation = _partition_groups(training_records, validation_fraction, seed)
    groups_by_split = {
        "train": {record.group_id for record in train},
        "validation": {record.group_id for record in validation},
        "test": {record.group_id for record in testing_records},
    }
    if (
        groups_by_split["train"] & groups_by_split["validation"]
        or groups_by_split["train"] & groups_by_split["test"]
        or groups_by_split["validation"] & groups_by_split["test"]
    ):
        raise ValueError("A patient/group identifier appears in multiple splits.")
    return {"train": train, "validation": validation, "test": testing_records}


def save_split_metadata(
    splits: dict[str, list[DatasetRecord]],
    csv_path: Path | str,
    config_path: Path | str,
    dataset_config: DatasetConfig,
    validation_fraction: float,
    seed: int,
) -> None:
    """Write paths and labels only; source image files remain in place."""
    csv_path = Path(csv_path)
    config_path = Path(config_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "source",
                "sample_id",
                "label",
                "target_label",
                "group_id",
                "grouping_method",
                "image_path",
            ),
        )
        writer.writeheader()
        for split_name, split_records in splits.items():
            for record in sorted(split_records, key=lambda item: item.sample_id):
                image_path = (
                    record.image_path.relative_to(dataset_config.root).as_posix()
                    if record.image_path is not None
                    else ""
                )
                writer.writerow(
                    {
                        "split": split_name,
                        "source": record.source,
                        "sample_id": record.sample_id,
                        "label": record.label,
                        "target_label": record.target_label,
                        "group_id": record.group_id,
                        "grouping_method": record.grouping_method,
                        "image_path": image_path,
                    }
                )

    config_payload = {
        "dataset_root": dataset_config.root.as_posix(),
        "sources": [
            {
                "name": source.name,
                "labels_file": source.labels_file.as_posix(),
                "image_dir": source.image_dir.as_posix(),
            }
            for source in dataset_config.sources
        ],
        "id_column": dataset_config.id_column,
        "label_column": dataset_config.label_column,
        "patient_id_regex": dataset_config.patient_id_regex,
        "label_mapping": dict(dataset_config.label_mapping or {}),
        "validation_fraction_of_training_source": validation_fraction,
        "seed": seed,
        "strategy": (
            "grouped stratification of the published training source; "
            "published testing source preserved unchanged; unmatched IDs use "
            "sample-level groups"
        ),
        "split_counts": {
            name: len(split_records) for name, split_records in splits.items()
        },
    }
    config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

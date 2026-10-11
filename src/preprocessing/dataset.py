"""Load and audit the published APTOS 2015/2019 preprocessed dataset layout."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class DatasetSource:
    name: str
    labels_file: Path
    image_dir: Path


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    sources: tuple[DatasetSource, ...]
    id_column: str = "id_code"
    label_column: str = "diagnosis"
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    patient_id_regex: str | None = None
    label_mapping: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.sources:
            raise ValueError("At least one dataset source is required.")
        if len({source.name for source in self.sources}) != len(self.sources):
            raise ValueError("Dataset source names must be unique.")
        if self.patient_id_regex is not None:
            pattern = re.compile(self.patient_id_regex)
            if "patient" not in pattern.groupindex:
                raise ValueError("patient_id_regex must define a named 'patient' group.")


@dataclass(frozen=True)
class DatasetRecord:
    sample_id: str
    label: str
    target_label: str
    image_path: Path | None
    source: str
    group_id: str
    grouping_method: str = "sample_id"


def aptos_preprocessed_config(root: Path | str) -> DatasetConfig:
    """Return paths and columns observed in the Kaggle dataset file listing."""
    return DatasetConfig(
        root=Path(root),
        sources=(
            DatasetSource(
                name="train",
                labels_file=Path("trainData.csv"),
                image_dir=Path("trainImgs") / "trainImgs",
            ),
            DatasetSource(
                name="test",
                labels_file=Path("testData.csv"),
                image_dir=Path("testImgs") / "testImgs",
            ),
        ),
        patient_id_regex=r"^(?P<patient>.+)_(?:left|right)$",
    )


def map_label(label: str, label_mapping: Mapping[str, str] | None = None) -> str:
    """Keep raw labels by default; apply only a complete, explicit mapping."""
    if label_mapping is None:
        return label
    try:
        return label_mapping[label]
    except KeyError as exc:
        raise ValueError(f"No explicit mapping was provided for label {label!r}.") from exc


def _group_id(sample_id: str, patient_id_regex: str | None) -> tuple[str, str]:
    if patient_id_regex is None:
        return sample_id, "sample_id"
    match = re.fullmatch(patient_id_regex, sample_id, flags=re.IGNORECASE)
    if match is None:
        return sample_id, "sample_id_fallback"
    return match.group("patient"), "laterality_regex"


def _index_images(directory: Path, extensions: tuple[str, ...]) -> dict[str, list[Path]]:
    image_index: dict[str, list[Path]] = defaultdict(list)
    if not directory.is_dir():
        return image_index
    accepted_extensions = {extension.lower() for extension in extensions}
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in accepted_extensions:
            image_index[path.stem.casefold()].append(path)
    return image_index


def load_dataset(config: DatasetConfig) -> list[DatasetRecord]:
    """Read configured CSV annotations and resolve image paths without copying files."""
    records: list[DatasetRecord] = []
    for source in config.sources:
        labels_path = config.root / source.labels_file
        if not labels_path.is_file():
            raise FileNotFoundError(f"Label CSV not found: {labels_path}")

        image_dir = config.root / source.image_dir
        image_index = _index_images(image_dir, config.image_extensions)
        with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Label CSV has no header: {labels_path}")
            required_columns = {config.id_column, config.label_column}
            missing_columns = required_columns.difference(reader.fieldnames)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"{labels_path} is missing required columns: {missing}")

            for row_number, row in enumerate(reader, start=2):
                sample_id = (row.get(config.id_column) or "").strip()
                raw_label = (row.get(config.label_column) or "").strip()
                if not sample_id or not raw_label:
                    raise ValueError(
                        f"Empty image identifier or label in {labels_path}, row {row_number}."
                    )

                candidates = image_index.get(sample_id.casefold(), [])
                if len(candidates) > 1:
                    paths = ", ".join(str(path) for path in sorted(candidates))
                    raise ValueError(
                        f"Multiple image files match identifier {sample_id!r}: {paths}"
                    )
                image_path = candidates[0] if candidates else None
                group_id, grouping_method = _group_id(
                    sample_id, config.patient_id_regex
                )
                records.append(
                    DatasetRecord(
                        sample_id=sample_id,
                        label=raw_label,
                        target_label=map_label(raw_label, config.label_mapping),
                        image_path=image_path,
                        source=source.name,
                        group_id=group_id,
                        grouping_method=grouping_method,
                    )
                )
    return records


def load_image(record: DatasetRecord, color_mode: str = "RGB") -> Image.Image:
    """Open one record's image and return a detached Pillow image."""
    if record.image_path is None:
        raise FileNotFoundError(f"No image file was found for {record.sample_id!r}.")
    with Image.open(record.image_path) as image:
        return image.convert(color_mode)


def _relative_path(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_imbalance(counts: Counter[str]) -> dict[str, int | float | None]:
    values = list(counts.values())
    return {
        "largest_class_count": max(values, default=0),
        "smallest_class_count": min(values, default=0),
        "largest_to_smallest_ratio": (
            max(values) / min(values) if values and min(values) > 0 else None
        ),
    }


def summarize_dataset(
    records: list[DatasetRecord], config: DatasetConfig
) -> dict[str, object]:
    """Inspect image readability, dimensions, formats, missing files, and duplicates."""
    raw_counts = Counter(record.label for record in records)
    target_counts = Counter(record.target_label for record in records)
    grouping_counts = Counter(record.grouping_method for record in records)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        source_counts[record.source][record.label] += 1
    duplicate_ids = [
        {"sample_id": sample_id, "count": count}
        for sample_id, count in sorted(Counter(r.sample_id for r in records).items())
        if count > 1
    ]

    dimensions: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    missing_images: list[dict[str, str]] = []
    corrupted_images: list[dict[str, str]] = []
    unreadable_images: list[dict[str, str]] = []
    referenced_paths: set[Path] = set()
    hashes: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"paths": set(), "sample_ids": set()}
    )
    readable_count = 0
    discovered_count = 0

    for source in config.sources:
        directory = config.root / source.image_dir
        for image_path in _index_images(directory, config.image_extensions).values():
            discovered_count += len(image_path)

    for record in records:
        if record.image_path is None:
            missing_images.append(
                {"sample_id": record.sample_id, "source": record.source}
            )
            continue

        referenced_paths.add(record.image_path.resolve())
        try:
            with Image.open(record.image_path) as image:
                image.load()
                width, height = image.size
                image_format = (image.format or record.image_path.suffix).upper()
            readable_count += 1
            dimensions[f"{width}x{height}"] += 1
            formats[image_format] += 1
            file_hash = _sha256(record.image_path)
            hashes[file_hash]["paths"].add(_relative_path(record.image_path, config.root) or "")
            hashes[file_hash]["sample_ids"].add(record.sample_id)
        except PermissionError as exc:
            unreadable_images.append(
                {
                    "sample_id": record.sample_id,
                    "path": _relative_path(record.image_path, config.root) or "",
                    "error": str(exc),
                }
            )
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            corrupted_images.append(
                {
                    "sample_id": record.sample_id,
                    "path": _relative_path(record.image_path, config.root) or "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    orphan_images: list[str] = []
    for source in config.sources:
        directory = config.root / source.image_dir
        for paths in _index_images(directory, config.image_extensions).values():
            orphan_images.extend(
                _relative_path(path, config.root) or ""
                for path in paths
                if path.resolve() not in referenced_paths
            )

    duplicate_content = [
        {
            "sha256": file_hash,
            "paths": sorted(details["paths"]),
            "sample_ids": sorted(details["sample_ids"]),
        }
        for file_hash, details in sorted(hashes.items())
        if len(details["paths"]) > 1
    ]
    return {
        "total_labeled_images": len(records),
        "readable_images": readable_count,
        "discovered_image_files": discovered_count,
        "class_counts_raw": dict(sorted(raw_counts.items())),
        "class_counts_by_source": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(source_counts.items())
        },
        "class_counts_after_explicit_mapping": dict(sorted(target_counts.items())),
        "class_imbalance": _class_imbalance(raw_counts),
        "class_imbalance_by_source": {
            source: _class_imbalance(counts)
            for source, counts in sorted(source_counts.items())
        },
        "grouping_methods": dict(sorted(grouping_counts.items())),
        "sample_level_group_fallbacks": [
            {"sample_id": record.sample_id, "source": record.source}
            for record in records
            if record.grouping_method == "sample_id_fallback"
        ],
        "image_formats": dict(sorted(formats.items())),
        "image_dimensions": dict(sorted(dimensions.items())),
        "missing_images": missing_images,
        "corrupted_images": corrupted_images,
        "unreadable_images": unreadable_images,
        "duplicate_ids": duplicate_ids,
        "duplicate_image_content": duplicate_content,
        "unlabeled_image_files": sorted(orphan_images),
    }

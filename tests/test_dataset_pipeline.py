from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.preprocessing.dataset import (
    DatasetConfig,
    DatasetRecord,
    DatasetSource,
    load_dataset,
    load_image,
    map_label,
    summarize_dataset,
)
from src.preprocessing.splitting import create_splits, save_split_metadata


class DatasetLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.image_dir = self.root / "trainImgs" / "trainImgs"
        self.image_dir.mkdir(parents=True)
        self.labels_file = self.root / "trainData.csv"

    def _write_rows(self, rows: list[tuple[str, str]]) -> None:
        with self.labels_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("id_code", "diagnosis"))
            writer.writerows(rows)

    def _config(self, label_mapping: dict[str, str] | None = None) -> DatasetConfig:
        return DatasetConfig(
            root=self.root,
            sources=(
                DatasetSource(
                    name="train",
                    labels_file=Path("trainData.csv"),
                    image_dir=Path("trainImgs") / "trainImgs",
                ),
            ),
            patient_id_regex=r"^(?P<patient>.+)_(?:left|right)$",
            label_mapping=label_mapping,
        )

    def test_loads_csv_labels_and_groups_laterality(self) -> None:
        self._write_rows([("123_left", "0"), ("123_right", "4")])
        (self.image_dir / "123_left.png").touch()
        (self.image_dir / "123_right.png").touch()

        records = load_dataset(self._config())

        self.assertEqual([record.label for record in records], ["0", "4"])
        self.assertEqual([record.target_label for record in records], ["0", "4"])
        self.assertEqual([record.group_id for record in records], ["123", "123"])
        self.assertTrue(all(record.image_path is not None for record in records))

    def test_opaque_id_falls_back_to_sample_level_group(self) -> None:
        self._write_rows([("000c1434d8d7", "2")])
        config = self._config()

        records = load_dataset(config)
        summary = summarize_dataset(records, config)

        self.assertEqual(records[0].group_id, "000c1434d8d7")
        self.assertEqual(records[0].grouping_method, "sample_id_fallback")
        self.assertEqual(summary["grouping_methods"], {"sample_id_fallback": 1})
        self.assertEqual(
            summary["sample_level_group_fallbacks"],
            [{"sample_id": "000c1434d8d7", "source": "train"}],
        )

    def test_mapping_is_explicit_and_rejects_unmapped_labels(self) -> None:
        self.assertEqual(map_label("0"), "0")
        self.assertEqual(map_label("0", {"0": "documented-class"}), "documented-class")
        with self.assertRaisesRegex(ValueError, "No explicit mapping"):
            map_label("4", {"0": "documented-class"})

    def test_summary_reports_corrupted_image(self) -> None:
        self._write_rows([("123_left", "0")])
        (self.image_dir / "123_left.png").write_text(
            "not an image", encoding="utf-8"
        )
        config = self._config()
        summary = summarize_dataset(load_dataset(config), config)

        self.assertEqual(summary["total_labeled_images"], 1)
        self.assertEqual(len(summary["corrupted_images"]), 1)
        self.assertEqual(summary["class_counts_raw"], {"0": 1})
        self.assertEqual(summary["class_counts_by_source"], {"train": {"0": 1}})

    def test_missing_image_is_reported(self) -> None:
        self._write_rows([("123_left", "0")])
        config = self._config()

        summary = summarize_dataset(load_dataset(config), config)

        self.assertEqual(
            summary["missing_images"],
            [{"sample_id": "123_left", "source": "train"}],
        )

    def test_load_image_uses_requested_color_mode(self) -> None:
        image_path = self.image_dir / "123_left.png"
        image = MagicMock()
        image.convert.return_value = "converted-image"
        image_context = MagicMock()
        image_context.__enter__.return_value = image
        record = DatasetRecord(
            sample_id="123_left",
            label="0",
            target_label="0",
            image_path=image_path,
            source="train",
            group_id="123",
        )

        with patch(
            "src.preprocessing.dataset.Image.open", return_value=image_context
        ):
            loaded = load_image(record, color_mode="RGB")

        self.assertEqual(loaded, "converted-image")
        image.convert.assert_called_once_with("RGB")


class DatasetSplittingTests(unittest.TestCase):
    @staticmethod
    def _record(
        sample_id: str, label: str, source: str = "train"
    ) -> DatasetRecord:
        return DatasetRecord(
            sample_id=sample_id,
            label=label,
            target_label=label,
            image_path=Path("images") / f"{sample_id}.png",
            source=source,
            group_id=sample_id.rsplit("_", maxsplit=1)[0],
        )

    def test_split_is_reproducible_stratified_and_group_disjoint(self) -> None:
        records = [
            self._record(f"patient-{patient}_{eye}", str(patient % 2))
            for patient in range(20)
            for eye in ("left", "right")
        ]
        records.extend(
            self._record(f"test-{patient}_{eye}", str(patient % 2), "test")
            for patient in range(4)
            for eye in ("left", "right")
        )

        first = create_splits(records, validation_fraction=0.25, seed=17)
        second = create_splits(records, validation_fraction=0.25, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first["test"]), 8)

        groups = {
            name: {record.group_id for record in split_records}
            for name, split_records in first.items()
        }
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])

        for split_name in ("train", "validation"):
            labels = {record.label for record in first[split_name]}
            self.assertEqual(labels, {"0", "1"})
        validation_counts = Counter(record.label for record in first["validation"])
        train_counts = Counter(record.label for record in first["train"])
        self.assertLessEqual(abs(validation_counts["0"] - 5), 1)
        self.assertLessEqual(abs(validation_counts["1"] - 5), 1)
        self.assertLessEqual(abs(train_counts["0"] - 15), 1)
        self.assertLessEqual(abs(train_counts["1"] - 15), 1)

    def test_rejects_missing_images_before_splitting(self) -> None:
        records = [
            self._record("patient-1_left", "0"),
            self._record("patient-2_left", "1"),
            self._record("test-1_left", "0", "test"),
        ]
        records[0] = DatasetRecord(
            sample_id=records[0].sample_id,
            label=records[0].label,
            target_label=records[0].target_label,
            image_path=None,
            source=records[0].source,
            group_id=records[0].group_id,
        )

        with self.assertRaisesRegex(ValueError, "Resolve missing images"):
            create_splits(records)

    def test_saves_split_paths_and_configuration_without_copying_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = []
            for source, count in (("train", 8), ("test", 4)):
                for patient in range(count):
                    sample_id = f"{source}-{patient}_left"
                    label = str(patient % 2)
                    records.append(
                        DatasetRecord(
                            sample_id=sample_id,
                            label=label,
                            target_label=label,
                            image_path=root / source / f"{sample_id}.png",
                            source=source,
                            group_id=f"{source}-{patient}",
                        )
                    )
            splits = create_splits(records, validation_fraction=0.25, seed=11)
            config = DatasetConfig(
                root=root,
                sources=(
                    DatasetSource("train", Path("train.csv"), Path("train")),
                    DatasetSource("test", Path("test.csv"), Path("test")),
                ),
            )
            metadata_path = root / "metadata" / "splits.csv"
            config_path = root / "metadata" / "config.json"

            save_split_metadata(
                splits,
                metadata_path,
                config_path,
                config,
                validation_fraction=0.25,
                seed=11,
            )

            with metadata_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), len(records))
            self.assertEqual(saved_config["seed"], 11)
            self.assertEqual(saved_config["split_counts"]["test"], 4)
            self.assertFalse(any(root.rglob("*.png")))


if __name__ == "__main__":
    unittest.main()

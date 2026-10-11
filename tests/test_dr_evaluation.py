from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dr_classification.evaluation_metrics import (
    evaluate_predictions,
    roc_curve_auc,
)
from src.dr_classification.evaluation_reporting import (
    build_evaluation_report,
    save_evaluation_dashboard,
)
from src.dr_classification.classifier import DEFAULT_CLASS_NAMES


class RocEvaluationTests(unittest.TestCase):
    def test_roc_auc_perfect_and_tied_rankings(self) -> None:
        perfect = roc_curve_auc([0, 1], [0.1, 0.9])
        tied = roc_curve_auc([0, 1], [0.5, 0.5])

        self.assertEqual(perfect["auc"], 1.0)
        self.assertEqual(tied["auc"], 0.5)
        self.assertEqual(perfect["fpr"][0], 0.0)
        self.assertEqual(perfect["tpr"][0], 0.0)

    def test_auc_is_undefined_when_a_split_lacks_one_binary_class(self) -> None:
        result = roc_curve_auc([1, 1], [0.2, 0.8])

        self.assertIsNone(result["auc"])
        self.assertFalse(result["defined"])
        self.assertIn("both positive and negative", result["reason"])

    def test_roc_rejects_non_finite_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            roc_curve_auc([0, 1], [0.2, np.nan])


class ClassificationEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = [0, 1, 2, 3, 4]
        self.probabilities = np.eye(5, dtype=np.float64).tolist()

    def test_five_class_and_binary_referable_metrics(self) -> None:
        result = evaluate_predictions(
            self.targets,
            self.probabilities,
            DEFAULT_CLASS_NAMES,
        )
        referable = result["binary_referable"]

        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_precision"], 1.0)
        self.assertEqual(result["macro_sensitivity"], 1.0)
        self.assertEqual(result["macro_specificity_ovr"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["macro_roc_auc_ovr"], 1.0)
        self.assertEqual(result["confusion_matrix"], np.eye(5, dtype=int).tolist())
        self.assertEqual(referable["confusion_matrix"], [[2, 0], [0, 3]])
        self.assertEqual(referable["sensitivity"], 1.0)
        self.assertEqual(referable["recall"], 1.0)
        self.assertEqual(referable["specificity"], 1.0)
        self.assertEqual(referable["precision"], 1.0)
        self.assertEqual(referable["f1"], 1.0)
        self.assertEqual(referable["roc_auc"], 1.0)
        self.assertIn("unchanged five-class argmax", referable["decision_rule"])
        json.dumps(result, allow_nan=False)

    def test_unbalanced_binary_predictions_are_grouped_without_rethresholding(self) -> None:
        targets = [0, 1, 2, 2, 4]
        probabilities = [
            [0.7, 0.1, 0.1, 0.05, 0.05],
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.2, 0.3, 0.2, 0.2],
            [0.35, 0.30, 0.20, 0.10, 0.05],
            [0.05, 0.05, 0.15, 0.25, 0.50],
        ]

        result = evaluate_predictions(targets, probabilities, DEFAULT_CLASS_NAMES)
        referable = result["binary_referable"]

        self.assertEqual(referable["confusion_matrix"], [[2, 0], [1, 2]])
        self.assertAlmostEqual(referable["sensitivity"], 2 / 3)
        self.assertEqual(referable["specificity"], 1.0)
        self.assertEqual(referable["roc_auc"], 1.0)
        self.assertEqual(
            result["confusion_matrix"][2][0],
            1,
            "The original five-class argmax prediction is retained.",
        )

    def test_undefined_class_auc_is_reported_as_none(self) -> None:
        result = evaluate_predictions(
            [0, 0],
            [[0.8, 0.05, 0.05, 0.05, 0.05], [0.7, 0.1, 0.1, 0.05, 0.05]],
            DEFAULT_CLASS_NAMES,
        )

        self.assertIsNone(result["binary_referable"]["roc_auc"])
        self.assertIsNone(result["per_class"][0]["roc_auc_ovr"])
        self.assertEqual(result["macro_roc_auc_defined_classes"], 0)

    def test_binary_sensitivity_is_undefined_without_referable_examples(self) -> None:
        result = evaluate_predictions(
            [0, 1],
            [
                [0.8, 0.1, 0.05, 0.03, 0.02],
                [0.1, 0.7, 0.1, 0.05, 0.05],
            ],
            DEFAULT_CLASS_NAMES,
        )

        self.assertIsNone(result["binary_referable"]["sensitivity"])
        self.assertIsNone(result["binary_referable"]["f1"])
        self.assertIsNone(result["binary_referable"]["roc_auc"])

    def test_rejects_invalid_probability_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            evaluate_predictions(
                [0],
                [[0.1, 0.1, 0.1, 0.1, 0.1]],
                DEFAULT_CLASS_NAMES,
            )

    def test_report_states_measured_targets_and_limitations(self) -> None:
        metrics = evaluate_predictions(
            self.targets,
            self.probabilities,
            DEFAULT_CLASS_NAMES,
        )
        payload = {
            **metrics,
            "loss": 0.25,
            "split": "test",
            "checkpoint": "model.pt",
            "checkpoint_epoch": 3,
            "class_names": list(DEFAULT_CLASS_NAMES),
            "dataset_split_support": {"train": 100, "validation": 20, "test": 5},
            "class_imbalance": {
                split: {
                    "counts": {
                        str(index): count
                        for index, count in enumerate(counts)
                    },
                    "largest_to_smallest_nonzero_ratio": ratio,
                }
                for split, counts, ratio in (
                    ("train", [30, 20, 20, 15, 15], 2.0),
                    ("validation", [6, 4, 4, 3, 3], 2.0),
                    ("test", [1, 1, 1, 1, 1], 1.0),
                )
            },
            "targets": {
                "referable_sensitivity_strictly_above": 0.90,
                "referable_specificity_strictly_above": 0.85,
            },
        }

        report = build_evaluation_report(payload)

        self.assertIn("Sensitivity / recall", report)
        self.assertIn("MET", report)
        self.assertIn("possible dataset bias", report.lower())
        self.assertIn("clinical validation", report.lower())
        self.assertIn("Class counts", report)

    def test_report_does_not_claim_targets_when_values_are_below_targets(self) -> None:
        metrics = evaluate_predictions(
            [0, 1, 2, 3],
            [
                [0.05, 0.05, 0.80, 0.05, 0.05],
                [0.05, 0.05, 0.80, 0.05, 0.05],
                [0.80, 0.05, 0.05, 0.05, 0.05],
                [0.05, 0.05, 0.05, 0.80, 0.05],
            ],
            DEFAULT_CLASS_NAMES,
        )
        payload = {
            **metrics,
            "loss": 1.0,
            "split": "test",
            "checkpoint": "model.pt",
            "checkpoint_epoch": 1,
            "class_names": list(DEFAULT_CLASS_NAMES),
            "dataset_split_support": {"train": 10, "validation": 5, "test": 5},
            "class_imbalance": {
                split: {
                    "counts": {str(index): 1 for index in range(5)},
                    "largest_to_smallest_nonzero_ratio": 1.0,
                }
                for split in ("train", "validation", "test")
            },
            "targets": {
                "referable_sensitivity_strictly_above": 0.90,
                "referable_specificity_strictly_above": 0.85,
            },
        }

        report = build_evaluation_report(payload)

        self.assertIn("| 50.00% | >90% | NOT MET |", report)
        self.assertIn("| 0.00% | >85% | NOT MET |", report)

    def test_dashboard_is_saved(self) -> None:
        metrics = evaluate_predictions(
            self.targets,
            self.probabilities,
            DEFAULT_CLASS_NAMES,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "dashboard.png"

            save_evaluation_dashboard(metrics, DEFAULT_CLASS_NAMES, output)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

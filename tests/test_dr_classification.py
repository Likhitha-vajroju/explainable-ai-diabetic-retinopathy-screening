from __future__ import annotations

import unittest

import numpy as np
import torch

from src.dr_classification.classifier import (
    ClassificationConfig,
    calculate_metrics,
    class_weights,
    prediction_result,
)


class ClassificationInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ClassificationConfig(pretrained=False)

    def test_inference_output_contains_level_probabilities_and_raw_confidence(self) -> None:
        probabilities = torch.tensor([0.05, 0.10, 0.65, 0.15, 0.05])

        result = prediction_result(probabilities, self.config.class_names)

        self.assertEqual(result["predicted_dr_level"], 2)
        self.assertEqual(result["predicted_label"], "Moderate")
        self.assertEqual(len(result["class_probabilities"]), 5)
        self.assertAlmostEqual(
            sum(item["probability"] for item in result["class_probabilities"]),
            1.0,
        )
        self.assertAlmostEqual(result["raw_confidence"], 0.65)
        self.assertIn("uncalibrated", result["confidence_interpretation"])

    def test_rejects_invalid_inference_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            prediction_result([0.1, 0.1, 0.1, 0.1, 0.1], self.config.class_names)

    def test_balanced_class_weights_use_train_counts(self) -> None:
        weights = class_weights([10, 20, 30, 40, 50], torch.device("cpu"))

        self.assertEqual(weights.shape, (5,))
        self.assertGreater(float(weights[0]), float(weights[4]))
        self.assertAlmostEqual(float(weights[0] / weights[4]), 50 / 10)

    def test_metrics_and_confusion_matrix_have_five_classes(self) -> None:
        metrics = calculate_metrics(
            np.array([0, 1, 2, 3, 4, 4]),
            np.array([0, 2, 2, 3, 1, 4]),
        )

        self.assertEqual(metrics["accuracy"], 4 / 6)
        self.assertEqual(len(metrics["confusion_matrix"]), 5)
        self.assertEqual(len(metrics["per_class"]), 5)
        self.assertEqual(
            sum(sum(row) for row in metrics["confusion_matrix"]),
            6,
        )

    def test_model_configuration_round_trips_json_compatible_values(self) -> None:
        config = ClassificationConfig.from_dict(
            {
                **self.config.to_dict(),
                "image_size": 128,
                "preprocessing": {
                    "output_size": [128, 128],
                    "clahe_tile_grid": [4, 4],
                },
            }
        )

        self.assertEqual(config.image_size, 128)
        self.assertEqual(config.preprocessing.output_size, (128, 128))
        self.assertEqual(config.preprocessing.clahe_tile_grid, (4, 4))


if __name__ == "__main__":
    unittest.main()

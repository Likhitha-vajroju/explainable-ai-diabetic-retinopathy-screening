from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dr_classification.calibration import (
    apply_temperature,
    calibration_summary,
    fit_temperature,
    save_reliability_diagram,
)
from src.dr_classification.decision_support import (
    REFERABLE_RECOMMENDATION,
    screening_result,
)


class TemperatureScalingTests(unittest.TestCase):
    def test_temperature_scaling_returns_probabilities_and_reduces_overconfidence(self) -> None:
        logits = np.array(
            [
                [4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 4.0],
            ],
            dtype=np.float64,
        )
        targets = [0, 1, 2, 3, 3]

        temperature = fit_temperature(logits, targets)
        calibrated = apply_temperature(logits, temperature)

        self.assertGreater(temperature, 0)
        self.assertEqual(calibrated.shape, (5, 5))
        self.assertTrue(np.allclose(calibrated.sum(axis=1), 1.0))
        raw_probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        raw_probabilities /= raw_probabilities.sum(axis=1, keepdims=True)
        raw_confidence = raw_probabilities.max(axis=1).mean()
        calibrated_confidence = calibrated.max(axis=1).mean()
        self.assertLess(calibrated_confidence, raw_confidence)

    def test_calibration_summary_reports_ece_and_reliability_bins(self) -> None:
        probabilities = np.array(
            [
                [0.9, 0.025, 0.025, 0.025, 0.025],
                [0.1, 0.8, 0.05, 0.025, 0.025],
            ]
        )

        result = calibration_summary(probabilities, [0, 0], number_of_bins=10)

        self.assertAlmostEqual(
            result["expected_calibration_error"],
            (0.1 + 0.8) / 2,
        )
        self.assertEqual(sum(item["count"] for item in result["bins"]), 2)
        self.assertIn("negative_log_likelihood", result)
        self.assertIn("brier_score", result)

    def test_reliability_diagram_is_written(self) -> None:
        probabilities = np.eye(5)
        summary = calibration_summary(probabilities, list(range(5)))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reliability.png"

            save_reliability_diagram(summary, summary, output)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_rejects_invalid_calibration_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            fit_temperature([[1.0, 0.0]], [0])
        with self.assertRaisesRegex(ValueError, "positive"):
            apply_temperature([0, 0, 0, 0, 0], 0)


class ReferableDecisionSupportTests(unittest.TestCase):
    def test_levels_zero_and_one_are_non_referable(self) -> None:
        for level in (0, 1):
            with self.subTest(level=level):
                probabilities = np.full(5, 0.01)
                probabilities[level] = 0.96

                result = screening_result(probabilities)

                self.assertEqual(result["dr_grade"]["level"], level)
                self.assertFalse(result["referable"])
                self.assertEqual(result["referable_status"], "non-referable")
                self.assertIsNone(result["calibrated_confidence"])

    def test_levels_two_through_four_recommend_ophthalmologist_review(self) -> None:
        for level in (2, 3, 4):
            with self.subTest(level=level):
                probabilities = np.full(5, 0.01)
                probabilities[level] = 0.96

                result = screening_result(probabilities)

                self.assertTrue(result["referable"])
                self.assertEqual(result["referable_status"], "referable")
                self.assertEqual(
                    result["referral_recommendation"],
                    REFERABLE_RECOMMENDATION,
                )
                self.assertIn("not an independent diagnosis", result["limitation"])

    def test_includes_raw_probabilities_and_optional_calibrated_confidence(self) -> None:
        raw = [0.05, 0.05, 0.6, 0.2, 0.1]
        calibrated = [0.1, 0.1, 0.45, 0.2, 0.15]

        result = screening_result(raw, calibrated)

        self.assertEqual(len(result["raw_probabilities"]), 5)
        self.assertAlmostEqual(result["raw_confidence"], 0.6)
        self.assertAlmostEqual(result["calibrated_confidence"], 0.45)
        self.assertIn("not clinical certainty", result["confidence_note"])

    def test_rejects_invalid_probability_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            screening_result([0.1] * 5)
        with self.assertRaisesRegex(ValueError, "Calibrated probabilities"):
            screening_result([0.2] * 5, [0.1] * 5)


if __name__ == "__main__":
    unittest.main()

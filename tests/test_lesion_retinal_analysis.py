from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.lesion_retinal_analysis import (
    EVIDENCE_WARNING,
    evaluate_binary_mask,
    localize_optic_disc,
    save_evidence_visualization,
    segment_vessels,
)


def make_fundus_with_vessels(size: int = 128) -> Image.Image:
    image = Image.new("RGB", (size, size), "black")
    draw = ImageDraw.Draw(image)
    margin = size // 10
    draw.ellipse((margin, margin, size - margin, size - margin), fill=(145, 95, 65))
    draw.line((size // 2, size // 3, size // 2, size - margin), fill=(35, 22, 18), width=3)
    draw.line((size // 2, size // 2, size // 4, size // 3), fill=(35, 22, 18), width=2)
    return image


class RetinalAnalysisTests(unittest.TestCase):
    def test_vessel_mask_has_original_dimensions_and_detects_dark_lines(self) -> None:
        image = make_fundus_with_vessels()

        mask = segment_vessels(image)

        self.assertEqual(mask.shape, (image.height, image.width))
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertGreater(int(mask.sum()), 0)
        self.assertFalse(mask[0, 0])

    def test_optic_disc_localizer_returns_bright_region_candidate(self) -> None:
        image = make_fundus_with_vessels()
        ImageDraw.Draw(image).ellipse((82, 52, 98, 68), fill=(245, 220, 175))

        candidate = localize_optic_disc(image)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertIn("not a confirmed optic disc", candidate.label)
        self.assertGreater(candidate.area_pixels, 0)
        self.assertGreater(candidate.center_xy[0], 70)

    def test_binary_metrics_compare_against_supplied_annotations(self) -> None:
        prediction = np.array([[1, 1], [0, 0]], dtype=bool)
        ground_truth = np.array([[1, 0], [1, 0]], dtype=bool)

        metrics = evaluate_binary_mask(prediction, ground_truth)

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["dice"], 0.5)
        self.assertEqual(metrics["iou"], 1 / 3)
        self.assertEqual(metrics["sensitivity"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(metrics["pixel_accuracy"], 0.5)

    def test_metrics_require_equal_sized_two_dimensional_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal-size 2D"):
            evaluate_binary_mask(np.zeros((2, 2)), np.zeros((3, 2)))

    def test_visualization_separates_algorithmic_and_annotated_evidence(self) -> None:
        image = make_fundus_with_vessels()
        vessel_mask = segment_vessels(image)
        annotated_mask = np.zeros((image.height, image.width), dtype=bool)
        annotated_mask[45:50, 55:60] = True

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "retinal-evidence.png"
            saved = save_evidence_visualization(
                image,
                output,
                vessel_mask=vessel_mask,
                annotated_regions={"microaneurysm-like": annotated_mask},
            )

            self.assertEqual(saved, output)
            self.assertTrue(output.is_file())
            with Image.open(output) as visualization:
                self.assertGreater(visualization.width, image.width)
                self.assertGreater(visualization.height, 0)
        self.assertIn("not a confirmed clinical finding", EVIDENCE_WARNING)


if __name__ == "__main__":
    unittest.main()

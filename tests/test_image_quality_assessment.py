from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from src.image_quality_assessment import (
    MetricStatus,
    QualityStatus,
    QualityThresholds,
    assess_image,
    save_diagnostic_visualization,
)


def make_fundus_like_image(size: int = 192) -> Image.Image:
    image = Image.new("RGB", (size, size), "black")
    draw = ImageDraw.Draw(image)
    margin = size // 10
    draw.ellipse(
        (margin, margin, size - margin, size - margin),
        fill=(105, 65, 40),
    )
    step = max(2, size // 48)
    for y in range(margin + 2, size - margin, step):
        draw.line((margin + 2, y, size - margin - 2, y), fill=(190, 145, 95))
    for x in range(margin + 2, size - margin, step):
        draw.line((x, margin + 2, x, size - margin - 2), fill=(65, 35, 25))
    return image


class QualityThresholdTests(unittest.TestCase):
    def test_rejects_out_of_order_thresholds(self) -> None:
        with self.assertRaisesRegex(ValueError, "Brightness thresholds"):
            QualityThresholds(brightness_min_mean=220)

    def test_rejects_unknown_json_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown quality threshold"):
            QualityThresholds.from_dict({"clinical_cutoff": 0.5})


class ImageQualityAssessmentTests(unittest.TestCase):
    def test_assesses_all_requested_indicators_and_serializes(self) -> None:
        thresholds = QualityThresholds(
            focus_min_variance=1,
            focus_borderline_variance=0.5,
            brightness_min_mean=20,
            brightness_max_mean=220,
            brightness_borderline_min=10,
            brightness_borderline_max=240,
            contrast_min_stddev=5,
            contrast_borderline_stddev=2,
            fov_min_area_fraction=0.4,
            fov_borderline_area_fraction=0.2,
            fov_min_dimension_fraction=0.6,
            fov_borderline_dimension_fraction=0.4,
        )
        result = assess_image(make_fundus_like_image(), thresholds)

        self.assertEqual(
            [measurement.name for measurement in result.measurements],
            ["focus", "illumination", "contrast", "field_of_view", "artifacts"],
        )
        self.assertGreaterEqual(result.score, 0)
        self.assertLessEqual(result.score, 100)
        self.assertIn(result.status, QualityStatus)
        payload = result.to_dict()
        self.assertEqual(payload["status"], result.status.value)
        json.dumps(payload)
        self.assertIn("Image Quality Score:", result.format_report())

    def test_blur_reduces_focus_measurement(self) -> None:
        sharp = make_fundus_like_image()
        blurred = sharp.filter(ImageFilter.GaussianBlur(radius=4))

        sharp_result = assess_image(sharp)
        blurred_result = assess_image(blurred)

        sharp_focus = sharp_result.measurements[0]
        blurred_focus = blurred_result.measurements[0]
        self.assertGreater(sharp_focus.value, blurred_focus.value)
        self.assertGreaterEqual(sharp_focus.score, blurred_focus.score)

    def test_black_image_reports_multiple_quality_reasons(self) -> None:
        result = assess_image(Image.new("RGB", (64, 64), "black"))
        reasons = " ".join(result.reasons)

        self.assertEqual(result.status, QualityStatus.UNGRADABLE)
        self.assertIn("insufficient focus", reasons)
        self.assertIn("inadequate illumination", reasons)
        self.assertIn("field of view", reasons)
        self.assertTrue(
            all(
                measurement.status is MetricStatus.FAIL
                for measurement in result.measurements
            )
        )

    def test_white_image_flags_clipping_artifact(self) -> None:
        result = assess_image(Image.new("RGB", (64, 64), "white"))
        artifacts = result.measurements[-1]

        self.assertEqual(artifacts.status, MetricStatus.FAIL)
        self.assertIn("clipped", artifacts.reason or "")

    def test_black_background_around_field_is_not_counted_as_clipping(self) -> None:
        image = Image.new("RGB", (128, 128), "black")
        ImageDraw.Draw(image).ellipse((8, 8, 120, 120), fill=(110, 70, 45))

        result = assess_image(image)

        self.assertEqual(result.measurements[1].value, 79.0)
        self.assertEqual(result.measurements[4].value, 0.0)
        self.assertEqual(result.measurements[4].status, MetricStatus.PASS)

    def test_reports_small_retinal_field(self) -> None:
        image = Image.new("RGB", (128, 128), "black")
        ImageDraw.Draw(image).ellipse((52, 52, 76, 76), fill=(120, 70, 35))

        result = assess_image(image)
        field_of_view = result.measurements[3]

        self.assertEqual(field_of_view.status, MetricStatus.FAIL)
        self.assertIn("field of view", field_of_view.reason or "")

    def test_invalid_image_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Could not read image"):
            assess_image(Path("missing-image-file.png"))

    def test_saves_annotated_visualization(self) -> None:
        image = make_fundus_like_image()
        result = assess_image(image)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "quality.png"

            save_diagnostic_visualization(image, output, result)

            with Image.open(output) as annotated:
                self.assertEqual(annotated.width, image.width)
                self.assertGreater(annotated.height, image.height)


if __name__ == "__main__":
    unittest.main()

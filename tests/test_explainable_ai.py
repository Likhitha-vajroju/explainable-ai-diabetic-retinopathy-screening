from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.dr_classification.classifier import ClassificationConfig, build_model
from src.explainable_ai import GRADCAM_WARNING, generate_gradcam
from src.preprocessing.image_pipeline import PreprocessingConfig


class GradCAMTests(unittest.TestCase):
    def test_generates_original_size_heatmap_and_visualization(self) -> None:
        config = ClassificationConfig(
            pretrained=False,
            image_size=64,
            preprocessing=PreprocessingConfig(output_size=(64, 64)),
        )
        model = build_model(config)
        model.eval()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "fundus.png"
            visualization_path = root / "gradcam.png"
            image = Image.new("RGB", (80, 60), (20, 35, 25))
            ImageDraw.Draw(image).ellipse((10, 5, 70, 55), fill=(145, 75, 60))
            image.save(image_path)

            result = generate_gradcam(
                image_path,
                model,
                config,
                torch.device("cpu"),
                visualization_path,
                target_class_index=3,
            )

            self.assertEqual(result.heatmap.shape, (60, 80))
            self.assertTrue(np.isfinite(result.heatmap).all())
            self.assertGreaterEqual(float(result.heatmap.min()), 0.0)
            self.assertLessEqual(float(result.heatmap.max()), 1.0)
            self.assertTrue(visualization_path.is_file())
            self.assertGreater(visualization_path.stat().st_size, 0)
            self.assertEqual(result.explanation["target_class"]["index"], 3)
            self.assertIn("predicted_class", result.explanation)
            self.assertGreaterEqual(result.explanation["confidence"], 0.0)
            self.assertLessEqual(result.explanation["confidence"], 1.0)
            self.assertEqual(
                result.explanation["visualization_path"], str(visualization_path)
            )
            self.assertEqual(result.explanation["warning"], GRADCAM_WARNING)


if __name__ == "__main__":
    unittest.main()

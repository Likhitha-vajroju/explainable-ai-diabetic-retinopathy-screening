from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.preprocessing import (
    PreprocessingConfig,
    load_rgb_image,
    preprocess_image,
    save_comparison_visualization,
)
from src.preprocessing.image_demo import main as run_demo


def make_fundus_image(size: tuple[int, int] = (160, 120)) -> Image.Image:
    image = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 10, size[0] - 10, size[1] - 10), fill=(105, 65, 40))
    for x in range(20, size[0] - 20, 8):
        draw.line((x, 20, x + 10, size[1] - 20), fill=(190, 140, 90), width=2)
    return image


class ImageLoadingTests(unittest.TestCase):
    def test_loads_detached_rgb_image_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "input.png"
            make_fundus_image().save(path)

            loaded = load_rgb_image(path)
            with Image.open(path) as original:
                self.assertEqual(original.mode, "RGB")
                self.assertEqual(loaded.size, original.size)
                self.assertEqual(loaded.getpixel((0, 0)), original.getpixel((0, 0)))

    def test_rejects_missing_image_with_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Could not load image"):
            load_rgb_image(Path("missing-fundus-image.png"))


class PreprocessingTests(unittest.TestCase):
    def test_resizes_without_distorting_aspect_and_keeps_original_separate(self) -> None:
        image = make_fundus_image()
        original_pixels = np.asarray(image).copy()

        result = preprocess_image(
            image,
            PreprocessingConfig(
                output_size=(128, 128),
                apply_clahe=False,
                normalize_illumination=False,
                noise_reduction_radius=0,
            ),
        )

        self.assertEqual(result.enhanced_image.size, (128, 128))
        self.assertEqual(result.original_image.size, image.size)
        self.assertTrue(np.array_equal(np.asarray(image), original_pixels))
        self.assertEqual(result.normalized_image.shape, (128, 128, 3))
        self.assertEqual(result.normalized_image.dtype, np.float32)
        self.assertGreaterEqual(float(result.normalized_image.min()), 0.0)
        self.assertLessEqual(float(result.normalized_image.max()), 1.0)

    def test_can_disable_pixel_scaling(self) -> None:
        result = preprocess_image(
            make_fundus_image(),
            PreprocessingConfig(
                output_size=None,
                normalize_pixels=False,
                apply_clahe=False,
                normalize_illumination=False,
                noise_reduction_radius=0,
            ),
        )

        self.assertGreaterEqual(float(result.normalized_image.min()), 0.0)
        self.assertLessEqual(float(result.normalized_image.max()), 255.0)

    def test_preserves_dimensions_when_resize_is_disabled(self) -> None:
        image = make_fundus_image((97, 61))

        result = preprocess_image(image, PreprocessingConfig(output_size=None))

        self.assertEqual(result.enhanced_image.size, image.size)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive dimensions"):
            PreprocessingConfig(output_size=(128, 0))

    def test_comparison_visualization_contains_both_images(self) -> None:
        original = make_fundus_image()
        result = preprocess_image(original, PreprocessingConfig(output_size=(80, 80)))
        with tempfile.TemporaryDirectory() as temporary_directory:
            comparison_path = Path(temporary_directory) / "comparison.png"

            save_comparison_visualization(
                result.original_image, result.enhanced_image, comparison_path
            )

            with Image.open(comparison_path) as comparison:
                self.assertEqual(
                    comparison.width,
                    result.original_image.width + result.enhanced_image.width + 16,
                )
                self.assertGreater(comparison.height, result.original_image.height)

    def test_demo_copies_source_and_refuses_to_overwrite_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "fundus.png"
            output_dir = root / "review"
            make_fundus_image().save(source)
            original_bytes = source.read_bytes()

            with redirect_stdout(StringIO()):
                self.assertEqual(
                    run_demo([str(source), "--output-dir", str(output_dir)]),
                    0,
                )

            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertEqual(
                (output_dir / "original.png").read_bytes(), original_bytes
            )
            self.assertTrue((output_dir / "enhanced.png").is_file())
            self.assertTrue((output_dir / "comparison.png").is_file())
            with (
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit),
            ):
                run_demo([str(source), "--output-dir", str(output_dir)])
            self.assertEqual(source.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()

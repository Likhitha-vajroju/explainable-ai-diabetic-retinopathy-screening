"""Configurable retinal fundus image preprocessing and enhancement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


@dataclass(frozen=True)
class PreprocessingConfig:
    """Engineering parameters for fundus image preprocessing."""

    output_size: tuple[int, int] | None = (512, 512)
    preserve_aspect_ratio: bool = True
    background_mode: Literal["black", "preserve"] = "black"
    retinal_mask_threshold: int = 12
    normalize_illumination: bool = True
    illumination_blur_fraction: float = 0.08
    illumination_target: float = 128.0
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    noise_reduction_radius: float = 0.6
    normalize_pixels: bool = True

    def __post_init__(self) -> None:
        if self.output_size is not None and (
            len(self.output_size) != 2 or any(size < 1 for size in self.output_size)
        ):
            raise ValueError("output_size must contain two positive dimensions.")
        if self.background_mode not in ("black", "preserve"):
            raise ValueError("background_mode must be 'black' or 'preserve'.")
        if not 0 <= self.retinal_mask_threshold <= 255:
            raise ValueError("retinal_mask_threshold must be in [0, 255].")
        if not 0 < self.illumination_blur_fraction <= 1:
            raise ValueError("illumination_blur_fraction must be in (0, 1].")
        if not 1 <= self.illumination_target <= 255:
            raise ValueError("illumination_target must be in [1, 255].")
        if self.clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be greater than zero.")
        if (
            len(self.clahe_tile_grid) != 2
            or any(tile_count < 1 for tile_count in self.clahe_tile_grid)
        ):
            raise ValueError("clahe_tile_grid must contain two positive tile counts.")
        if self.noise_reduction_radius < 0:
            raise ValueError("noise_reduction_radius cannot be negative.")


@dataclass(frozen=True)
class PreprocessingResult:
    """Original, enhanced, and normalized image representations."""

    original_image: Image.Image
    enhanced_image: Image.Image
    normalized_image: np.ndarray


def load_rgb_image(image: str | Path | Image.Image) -> Image.Image:
    """Load an image as a detached RGB copy without changing its source file."""
    try:
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert("RGB").copy()
        with Image.open(image) as source:
            return ImageOps.exif_transpose(source).convert("RGB").copy()
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load image: {error}") from error


def _resize_image(image: Image.Image, config: PreprocessingConfig) -> Image.Image:
    if config.output_size is None:
        return image.copy()
    width, height = config.output_size
    resampling = Image.Resampling.LANCZOS
    if not config.preserve_aspect_ratio:
        return image.resize((width, height), resampling)

    resized = ImageOps.contain(image, (width, height), method=resampling)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(
        resized,
        ((width - resized.width) // 2, (height - resized.height) // 2),
    )
    return canvas


def _clahe_luminance(
    luminance: np.ndarray,
    mask: np.ndarray,
    clip_limit: float,
    requested_grid: tuple[int, int],
) -> np.ndarray:
    """Apply contrast-limited adaptive histogram equalization to uint8 luminance."""
    height, width = luminance.shape
    tiles_x = min(requested_grid[0], width)
    tiles_y = min(requested_grid[1], height)
    x_edges = np.linspace(0, width, tiles_x + 1, dtype=int)
    y_edges = np.linspace(0, height, tiles_y + 1, dtype=int)
    lookup_tables = np.empty((tiles_y, tiles_x, 256), dtype=np.uint8)

    for tile_y in range(tiles_y):
        y_start, y_end = y_edges[tile_y], y_edges[tile_y + 1]
        for tile_x in range(tiles_x):
            x_start, x_end = x_edges[tile_x], x_edges[tile_x + 1]
            tile_mask = mask[y_start:y_end, x_start:x_end]
            tile = luminance[y_start:y_end, x_start:x_end][tile_mask]
            if tile.size == 0:
                lookup_tables[tile_y, tile_x] = np.arange(256, dtype=np.uint8)
                continue

            histogram = np.bincount(tile, minlength=256)
            clip_count = max(1, int(clip_limit * tile.size / 256))
            excess = int(np.maximum(histogram - clip_count, 0).sum())
            histogram = np.minimum(histogram, clip_count)
            histogram += excess // 256
            histogram[: excess % 256] += 1
            cumulative = histogram.cumsum()
            nonzero = np.flatnonzero(cumulative)
            if nonzero.size == 0 or cumulative[-1] == cumulative[nonzero[0]]:
                lookup_tables[tile_y, tile_x] = np.arange(256, dtype=np.uint8)
            else:
                scaled = (cumulative - cumulative[nonzero[0]]) * (
                    255.0 / (cumulative[-1] - cumulative[nonzero[0]])
                )
                lookup_tables[tile_y, tile_x] = np.clip(
                    scaled, 0, 255
                ).astype(np.uint8)

    tile_x_position = np.clip(
        (np.arange(width, dtype=np.float32) + 0.5) * tiles_x / width - 0.5,
        0,
        tiles_x - 1,
    )
    tile_y_position = np.clip(
        (np.arange(height, dtype=np.float32) + 0.5) * tiles_y / height - 0.5,
        0,
        tiles_y - 1,
    )
    x0 = tile_x_position.astype(int)
    y0 = tile_y_position.astype(int)
    x1 = np.minimum(x0 + 1, tiles_x - 1)
    y1 = np.minimum(y0 + 1, tiles_y - 1)
    wx = tile_x_position - x0
    wy = tile_y_position - y0

    top_left = lookup_tables[y0[:, None], x0[None, :], luminance]
    top_right = lookup_tables[y0[:, None], x1[None, :], luminance]
    bottom_left = lookup_tables[y1[:, None], x0[None, :], luminance]
    bottom_right = lookup_tables[y1[:, None], x1[None, :], luminance]
    top = top_left * (1 - wx[None, :]) + top_right * wx[None, :]
    bottom = bottom_left * (1 - wx[None, :]) + bottom_right * wx[None, :]
    return np.clip(top * (1 - wy[:, None]) + bottom * wy[:, None], 0, 255).astype(
        np.uint8
    )


def preprocess_image(
    image: str | Path | Image.Image,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Preprocess one image while retaining separate input and output images.

    Geometric resizing preserves aspect ratio by default. Enhancement is applied
    to luminance only, and the returned normalized array is RGB float32 in [0, 1].
    """
    settings = config or PreprocessingConfig()
    original = load_rgb_image(image)
    working = _resize_image(original, settings)
    source_pixels = np.asarray(working, dtype=np.uint8)
    ycbcr = np.asarray(working.convert("YCbCr"), dtype=np.uint8)
    luminance = ycbcr[:, :, 0].copy()
    mask = luminance > settings.retinal_mask_threshold
    if not mask.any():
        mask = np.ones(luminance.shape, dtype=bool)

    if settings.normalize_illumination:
        blur_radius = max(
            1.0,
            min(working.size) * settings.illumination_blur_fraction,
        )
        illumination = np.asarray(
            Image.fromarray(luminance).filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
            ),
            dtype=np.float32,
        )
        normalized_luminance = luminance.astype(np.float32)
        normalized_luminance[mask] *= (
            settings.illumination_target
            / np.maximum(illumination[mask], 1.0)
        )
        luminance = np.clip(normalized_luminance, 0, 255).astype(np.uint8)

    if settings.apply_clahe:
        luminance = _clahe_luminance(
            luminance,
            mask,
            settings.clahe_clip_limit,
            settings.clahe_tile_grid,
        )

    if settings.noise_reduction_radius > 0:
        luminance = np.asarray(
            Image.fromarray(luminance).filter(
                ImageFilter.GaussianBlur(radius=settings.noise_reduction_radius)
            ),
            dtype=np.uint8,
        )

    enhanced_ycbcr = ycbcr.copy()
    enhanced_ycbcr[:, :, 0] = luminance
    enhanced = Image.fromarray(enhanced_ycbcr, mode="YCbCr").convert("RGB")
    enhanced_pixels = np.asarray(enhanced, dtype=np.uint8).copy()
    if settings.background_mode == "black":
        enhanced_pixels[~mask] = 0
    else:
        enhanced_pixels[~mask] = source_pixels[~mask]
    enhanced = Image.fromarray(enhanced_pixels, mode="RGB")

    normalized = enhanced_pixels.astype(np.float32)
    if settings.normalize_pixels:
        normalized /= 255.0
    return PreprocessingResult(
        original_image=original.copy(),
        enhanced_image=enhanced,
        normalized_image=normalized,
    )


def save_comparison_visualization(
    original: Image.Image,
    enhanced: Image.Image,
    output_path: str | Path,
) -> None:
    """Save original and enhanced images side-by-side with descriptive labels."""
    gap = 16
    label_height = 28
    height = max(original.height, enhanced.height) + label_height
    canvas = Image.new(
        "RGB", (original.width + enhanced.width + gap, height), (28, 28, 28)
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Original", fill=(255, 255, 255))
    draw.text((original.width + gap + 8, 8), "Enhanced", fill=(255, 255, 255))
    canvas.paste(original, (0, label_height))
    canvas.paste(enhanced, (original.width + gap, label_height))
    canvas.save(output_path)

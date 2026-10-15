"""Standalone, heuristic retinal-structure and annotated-evidence utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.preprocessing.image_pipeline import load_rgb_image


EVIDENCE_WARNING = (
    "Algorithmic or annotated image evidence is not a confirmed clinical finding "
    "or diagnosis."
)

_REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "microaneurysm-like": (255, 60, 60),
    "exudate-like": (255, 230, 0),
    "hemorrhage-like": (255, 0, 180),
    "other annotated evidence": (80, 220, 255),
}


@dataclass(frozen=True)
class OpticDiscCandidate:
    """Bright-region candidate; its coordinates are not a clinical finding."""

    center_xy: tuple[float, float]
    bounding_box_xyxy: tuple[int, int, int, int]
    area_pixels: int
    method: str
    label: str = "algorithmic bright-region candidate; not a confirmed optic disc"


def _retinal_field(rgb: np.ndarray) -> np.ndarray:
    """Estimate the photographed field from non-dark pixels."""
    return np.max(rgb, axis=2) > 12


def segment_vessels(
    image: str | Path | Image.Image,
    *,
    threshold_percentile: float = 88.0,
) -> np.ndarray:
    """Return a heuristic dark-line vessel candidate mask, with original dimensions.

    A locally blurred green channel estimates background illumination. Dark
    structures with a larger-than-percentile local contrast are retained only
    when they have nearby supporting pixels. This is a visualization-oriented
    algorithm, not a clinically validated vessel segmenter.
    """
    if not 50.0 <= threshold_percentile < 100.0:
        raise ValueError("threshold_percentile must be in [50, 100).")
    rgb = np.asarray(load_rgb_image(image), dtype=np.uint8)
    field = _retinal_field(rgb)
    if not field.any():
        return np.zeros(field.shape, dtype=bool)

    green = Image.fromarray(rgb[:, :, 1], mode="L")
    radius = max(2.0, min(green.size) * 0.025)
    background = np.asarray(green.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    local_dark_contrast = np.maximum(background - rgb[:, :, 1].astype(np.float32), 0.0)
    field_contrast = local_dark_contrast[field]
    threshold = max(5.0, float(np.percentile(field_contrast, threshold_percentile)))
    candidates = (local_dark_contrast >= threshold) & field

    neighbor_count = np.zeros(field.shape, dtype=np.uint8)
    for dy, dx in (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    ):
        source_y = slice(max(0, -dy), min(field.shape[0], field.shape[0] - dy))
        source_x = slice(max(0, -dx), min(field.shape[1], field.shape[1] - dx))
        target_y = slice(max(0, dy), min(field.shape[0], field.shape[0] + dy))
        target_x = slice(max(0, dx), min(field.shape[1], field.shape[1] + dx))
        neighbor_count[target_y, target_x] += candidates[source_y, source_x]
    return candidates & (neighbor_count >= 2)


def localize_optic_disc(
    image: str | Path | Image.Image,
    *,
    bright_percentile: float = 99.0,
    minimum_area_fraction: float = 0.0005,
    maximum_area_fraction: float = 0.12,
) -> OpticDiscCandidate | None:
    """Estimate a bright-region candidate that may correspond to the optic disc.

    The largest contiguous-like bright pixel cluster is approximated by a
    bounding box around high-intensity, yellow-white pixels. Returns ``None``
    when no plausible-size cluster is found; this is not optic-disc
    segmentation or clinical localization.
    """
    if not 90.0 <= bright_percentile < 100.0:
        raise ValueError("bright_percentile must be in [90, 100).")
    if not 0 < minimum_area_fraction < maximum_area_fraction <= 1:
        raise ValueError("Area fractions must satisfy 0 < minimum < maximum <= 1.")
    rgb = np.asarray(load_rgb_image(image), dtype=np.uint8)
    field = _retinal_field(rgb)
    if not field.any():
        return None

    luminance = (
        0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    )
    threshold = max(150.0, float(np.percentile(luminance[field], bright_percentile)))
    bright = (
        field
        & (luminance >= threshold)
        & (rgb[:, :, 0] > 1.08 * rgb[:, :, 2])
        & (rgb[:, :, 1] > 1.05 * rgb[:, :, 2])
    )
    ys, xs = np.nonzero(bright)
    area = int(xs.size)
    total_pixels = int(field.size)
    if not minimum_area_fraction <= area / total_pixels <= maximum_area_fraction:
        return None
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    return OpticDiscCandidate(
        center_xy=(float(xs.mean()), float(ys.mean())),
        bounding_box_xyxy=(left, top, right, bottom),
        area_pixels=area,
        method="high-percentile yellow-white intensity heuristic",
    )


def evaluate_binary_mask(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, int | float | None]:
    """Compare binary masks; only meaningful when supplied annotations are valid."""
    predicted = np.asarray(prediction, dtype=bool)
    truth = np.asarray(ground_truth, dtype=bool)
    if predicted.ndim != 2 or truth.ndim != 2 or predicted.shape != truth.shape:
        raise ValueError("Prediction and ground-truth masks must be equal-size 2D arrays.")

    true_positive = int(np.count_nonzero(predicted & truth))
    false_positive = int(np.count_nonzero(predicted & ~truth))
    false_negative = int(np.count_nonzero(~predicted & truth))
    true_negative = int(np.count_nonzero(~predicted & ~truth))
    union = true_positive + false_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "dice": (
            1.0
            if true_positive + false_positive + false_negative == 0
            else 2 * true_positive
            / (2 * true_positive + false_positive + false_negative)
        ),
        "iou": 1.0 if union == 0 else true_positive / union,
        "sensitivity": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        ),
        "specificity": (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive
            else None
        ),
        "pixel_accuracy": (
            (true_positive + true_negative) / predicted.size
            if predicted.size
            else None
        ),
    }


def save_evidence_visualization(
    image: str | Path | Image.Image,
    output_path: str | Path,
    *,
    vessel_mask: np.ndarray | None = None,
    optic_disc: OpticDiscCandidate | None = None,
    annotated_regions: Mapping[str, np.ndarray] | None = None,
) -> Path:
    """Save original, algorithmic vessel, and separately annotated evidence panels.

    ``annotated_regions`` accepts pixel masks supplied by the caller, e.g.
    annotations from a documented public retinal dataset. This function does
    not infer lesions or validate provenance of those annotations.
    """
    original_image = load_rgb_image(image)
    original = np.asarray(original_image, dtype=np.uint8)
    expected_shape = (original_image.height, original_image.width)
    output = Path(output_path)
    if isinstance(image, (str, Path)) and Path(image).resolve() == output.resolve():
        raise ValueError("Visualization output path must not overwrite the input image.")

    vessel = None if vessel_mask is None else np.asarray(vessel_mask, dtype=bool)
    if vessel is not None and vessel.shape != expected_shape:
        raise ValueError(f"vessel_mask must have shape {expected_shape}.")
    regions = dict(annotated_regions or {})
    validated_regions: dict[str, np.ndarray] = {}
    for name, mask in regions.items():
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != expected_shape:
            raise ValueError(f"Annotated region {name!r} must have shape {expected_shape}.")
        validated_regions[name] = mask_array

    vessel_overlay = original.astype(np.float32).copy()
    if vessel is not None:
        vessel_color = np.array((0, 255, 255), dtype=np.float32)
        vessel_overlay[vessel] = (
            0.35 * vessel_overlay[vessel] + 0.65 * vessel_color
        )
    vessel_image = Image.fromarray(np.uint8(np.clip(vessel_overlay, 0, 255)))
    if optic_disc is not None:
        draw = ImageDraw.Draw(vessel_image)
        draw.rectangle(optic_disc.bounding_box_xyxy, outline=(255, 255, 0), width=2)

    region_overlay = original.astype(np.float32).copy()
    for name, mask in validated_regions.items():
        color = np.asarray(
            _REGION_COLORS.get(name, _REGION_COLORS["other annotated evidence"]),
            dtype=np.float32,
        )
        region_overlay[mask] = 0.35 * region_overlay[mask] + 0.65 * color

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].imshow(original)
    axes[0].set_title("Original fundus image")
    axes[1].imshow(vessel_image)
    axes[1].set_title("Algorithmic vessel / disc candidates")
    axes[2].imshow(np.uint8(np.clip(region_overlay, 0, 255)))
    axes[2].set_title(
        "Caller-supplied annotated evidence"
        if validated_regions
        else "No lesion annotations supplied"
    )
    for axis in axes:
        axis.set_axis_off()
    region_legend = ", ".join(validated_regions) or "none supplied"
    figure.suptitle(
        "Retinal evidence visualization (not a diagnosis)",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        f"Annotated regions: {region_legend}. {EVIDENCE_WARNING}",
        ha="center",
        fontsize=8,
        color="#8a1c1c",
    )
    try:
        figure.savefig(output, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output

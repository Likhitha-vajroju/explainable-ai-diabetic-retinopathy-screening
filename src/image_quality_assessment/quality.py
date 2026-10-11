"""Engineering-oriented image quality checks for retinal fundus photographs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageStat


class QualityStatus(str, Enum):
    ACCEPTABLE = "ACCEPTABLE"
    BORDERLINE = "BORDERLINE"
    UNGRADABLE = "UNGRADABLE"


class MetricStatus(str, Enum):
    PASS = "PASS"
    BORDERLINE = "BORDERLINE"
    FAIL = "FAIL"


@dataclass(frozen=True)
class QualityThresholds:
    """Configurable heuristic cutoffs; defaults are not clinically validated."""

    focus_min_variance: float = 20.0
    focus_borderline_variance: float = 10.0
    brightness_min_mean: float = 45.0
    brightness_max_mean: float = 210.0
    brightness_borderline_min: float = 25.0
    brightness_borderline_max: float = 235.0
    contrast_min_stddev: float = 25.0
    contrast_borderline_stddev: float = 12.0
    fov_min_area_fraction: float = 0.20
    fov_borderline_area_fraction: float = 0.10
    fov_min_dimension_fraction: float = 0.35
    fov_borderline_dimension_fraction: float = 0.20
    clipped_max_fraction: float = 0.10
    clipped_borderline_fraction: float = 0.30
    ungradable_score_min: float = 40.0
    acceptable_score_min: float = 70.0
    fov_background_threshold: int = 12

    def __post_init__(self) -> None:
        if not (
            0 <= self.brightness_borderline_min
            < self.brightness_min_mean
            < self.brightness_max_mean
            < self.brightness_borderline_max
            <= 255
        ):
            raise ValueError("Brightness thresholds must form ordered ranges in [0, 255].")
        if not (
            0
            <= self.focus_borderline_variance
            < self.focus_min_variance
        ):
            raise ValueError("Focus thresholds must be non-negative and ordered.")
        if not (
            0
            <= self.contrast_borderline_stddev
            < self.contrast_min_stddev
        ):
            raise ValueError("Contrast thresholds must be non-negative and ordered.")
        if not (
            0 <= self.fov_borderline_area_fraction < self.fov_min_area_fraction <= 1
        ):
            raise ValueError("FOV area thresholds must be ordered fractions in [0, 1].")
        if not (
            0
            <= self.fov_borderline_dimension_fraction
            < self.fov_min_dimension_fraction
            <= 1
        ):
            raise ValueError("FOV dimension thresholds must be ordered fractions in [0, 1].")
        if not (
            0
            <= self.clipped_max_fraction
            < self.clipped_borderline_fraction
            <= 1
        ):
            raise ValueError("Clipping thresholds must be ordered fractions in [0, 1].")
        if not 0 <= self.ungradable_score_min <= self.acceptable_score_min <= 100:
            raise ValueError("Overall score thresholds must be ordered in [0, 100].")
        if not 0 <= self.fov_background_threshold <= 255:
            raise ValueError("FOV background threshold must be in [0, 255].")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> QualityThresholds:
        """Create validated thresholds from a JSON-compatible mapping."""
        expected = set(cls.__dataclass_fields__)
        unknown = set(values) - expected
        if unknown:
            raise ValueError(f"Unknown quality threshold(s): {', '.join(sorted(unknown))}")
        return cls(**values)


@dataclass(frozen=True)
class QualityMeasurement:
    name: str
    value: float
    unit: str
    status: MetricStatus
    score: float
    detail: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass(frozen=True)
class QualityAssessment:
    image_size: tuple[int, int]
    score: int
    status: QualityStatus
    measurements: tuple[QualityMeasurement, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "score": self.score,
            "status": self.status.value,
            "measurements": [measurement.to_dict() for measurement in self.measurements],
            "reasons": list(self.reasons),
        }

    def format_report(self) -> str:
        lines = [
            f"Image Quality Score: {self.score}/100",
            f"Status: {self.status.value}",
            "",
        ]
        lines.extend(
            f"{measurement.name.replace('_', ' ').title()}: {measurement.status.value} "
            f"({measurement.value:.2f} {measurement.unit})"
            for measurement in self.measurements
        )
        if self.reasons:
            lines.extend(("", "Reasons:", ""))
            lines.extend(f"- {reason}" for reason in self.reasons)
        return "\n".join(lines)


def _load_rgb(image: str | Path | Image.Image) -> Image.Image:
    try:
        if isinstance(image, Image.Image):
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        else:
            with Image.open(image) as source:
                rgb = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read image: {error}") from error
    if rgb.width < 3 or rgb.height < 3:
        raise ValueError("Image must be at least 3 pixels wide and high.")
    return rgb


def _measurement(
    name: str,
    value: float,
    unit: str,
    status: MetricStatus,
    score: float,
    detail: str,
    reason: str | None,
) -> QualityMeasurement:
    return QualityMeasurement(
        name=name,
        value=round(float(value), 4),
        unit=unit,
        status=status,
        score=round(max(0.0, min(100.0, score)), 2),
        detail=detail,
        reason=reason,
    )


def assess_image(
    image: str | Path | Image.Image,
    thresholds: QualityThresholds | None = None,
) -> QualityAssessment:
    """Measure focus, illumination, contrast, field of view, and clipping artifacts.

    The returned score is a configurable engineering screening metric, not a
    clinically validated measure of image gradability.
    """
    config = thresholds or QualityThresholds()
    rgb = _load_rgb(image)
    gray = ImageOps.grayscale(rgb)
    width, height = gray.size
    pixel_count = width * height

    field_mask = gray.point(
        lambda value: 255 if value > config.fov_background_threshold else 0
    )
    mask_histogram = field_mask.histogram()
    field_pixels = mask_histogram[255]
    bounds = field_mask.getbbox()
    if bounds is None:
        bounds = (0, 0, 0, 0)
        area_fraction = width_fraction = height_fraction = 0.0
    else:
        left, top, right, bottom = bounds
        area_fraction = ((right - left) * (bottom - top)) / pixel_count
        width_fraction = (right - left) / width
        height_fraction = (bottom - top) / height

    eroded_mask = field_mask.filter(ImageFilter.MinFilter(3))
    edge_response = gray.filter(ImageFilter.FIND_EDGES)
    focus_stats = ImageStat.Stat(edge_response, mask=eroded_mask)
    focus_variance = focus_stats.var[0] if field_pixels else 0.0

    field_histogram = gray.histogram(mask=field_mask)
    intensity_sum = sum(level * count for level, count in enumerate(field_histogram))
    mean_brightness = intensity_sum / field_pixels if field_pixels else 0.0
    artifact_histogram = field_histogram if field_pixels else gray.histogram()
    artifact_pixel_count = field_pixels if field_pixels else pixel_count
    clipped_pixels = sum(artifact_histogram[:2]) + sum(artifact_histogram[-2:])
    clipped_fraction = clipped_pixels / artifact_pixel_count
    contrast_stddev = ImageStat.Stat(gray, mask=field_mask).stddev[0] if field_pixels else 0.0

    measurements: list[QualityMeasurement] = []

    if focus_variance >= config.focus_min_variance:
        focus_status = MetricStatus.PASS
    elif focus_variance >= config.focus_borderline_variance:
        focus_status = MetricStatus.BORDERLINE
    else:
        focus_status = MetricStatus.FAIL
    focus_score = (
        100.0
        if focus_status is MetricStatus.PASS
        else 100.0
        * (focus_variance - config.focus_borderline_variance)
        / (config.focus_min_variance - config.focus_borderline_variance)
    )
    measurements.append(
        _measurement(
            "focus",
            focus_variance,
            "edge variance",
            focus_status,
            focus_score,
            "Variance of an edge-filter response inside the detected retinal field.",
            "insufficient focus" if focus_status is not MetricStatus.PASS else None,
        )
    )

    if config.brightness_min_mean <= mean_brightness <= config.brightness_max_mean:
        brightness_status = MetricStatus.PASS
        brightness_score = 100.0
    elif (
        config.brightness_borderline_min
        <= mean_brightness
        <= config.brightness_borderline_max
    ):
        brightness_status = MetricStatus.BORDERLINE
        if mean_brightness < config.brightness_min_mean:
            brightness_score = 100.0 * (
                mean_brightness - config.brightness_borderline_min
            ) / (config.brightness_min_mean - config.brightness_borderline_min)
        else:
            brightness_score = 100.0 * (
                config.brightness_borderline_max - mean_brightness
            ) / (config.brightness_borderline_max - config.brightness_max_mean)
    else:
        brightness_status = MetricStatus.FAIL
        brightness_score = 0.0
    illumination_reason = None
    if brightness_status is not MetricStatus.PASS:
        illumination_reason = (
            "inadequate illumination (image too dark)"
            if mean_brightness < config.brightness_min_mean
            else "inadequate illumination (image too bright)"
        )
    measurements.append(
        _measurement(
            "illumination",
            mean_brightness,
            "mean gray level (0-255)",
            brightness_status,
            brightness_score,
            "Mean grayscale brightness inside the detected retinal field.",
            illumination_reason,
        )
    )

    if contrast_stddev >= config.contrast_min_stddev:
        contrast_status = MetricStatus.PASS
    elif contrast_stddev >= config.contrast_borderline_stddev:
        contrast_status = MetricStatus.BORDERLINE
    else:
        contrast_status = MetricStatus.FAIL
    contrast_score = 100.0 * min(
        1.0, contrast_stddev / config.contrast_min_stddev
    )
    measurements.append(
        _measurement(
            "contrast",
            contrast_stddev,
            "standard deviation",
            contrast_status,
            contrast_score,
            "Grayscale intensity spread inside the detected retinal field.",
            "low retinal contrast"
            if contrast_status is not MetricStatus.PASS
            else None,
        )
    )

    area_ok = area_fraction >= config.fov_min_area_fraction
    dimensions_ok = min(width_fraction, height_fraction) >= config.fov_min_dimension_fraction
    area_borderline = area_fraction >= config.fov_borderline_area_fraction
    dimensions_borderline = (
        min(width_fraction, height_fraction)
        >= config.fov_borderline_dimension_fraction
    )
    if area_ok and dimensions_ok:
        fov_status = MetricStatus.PASS
    elif area_borderline and dimensions_borderline:
        fov_status = MetricStatus.BORDERLINE
    else:
        fov_status = MetricStatus.FAIL
    fov_score = 100.0 * min(
        1.0,
        area_fraction / config.fov_min_area_fraction,
        min(width_fraction, height_fraction) / config.fov_min_dimension_fraction,
    )
    measurements.append(
        _measurement(
            "field_of_view",
            area_fraction * 100.0,
            "% of frame (bounding box)",
            fov_status,
            fov_score,
            (
                f"Detected field occupies {area_fraction:.1%} by bounding-box area; "
                f"dimensions are {width_fraction:.1%} x {height_fraction:.1%}."
            ),
            "inadequate retinal field of view"
            if fov_status is not MetricStatus.PASS
            else None,
        )
    )

    if clipped_fraction <= config.clipped_max_fraction:
        artifact_status = MetricStatus.PASS
        artifact_score = 100.0
    elif clipped_fraction <= config.clipped_borderline_fraction:
        artifact_status = MetricStatus.BORDERLINE
        artifact_score = 100.0 * (
            config.clipped_borderline_fraction - clipped_fraction
        ) / (config.clipped_borderline_fraction - config.clipped_max_fraction)
    else:
        artifact_status = MetricStatus.FAIL
        artifact_score = 0.0
    measurements.append(
        _measurement(
            "artifacts",
            clipped_fraction * 100.0,
            "% clipped pixels",
            artifact_status,
            artifact_score,
            (
                "Fraction of detected field pixels near pure black or white; "
                "uses the full image if no field is detected. Other artifact "
                "types are not detected."
            ),
            "substantial clipped/overexposed or underexposed regions"
            if artifact_status is not MetricStatus.PASS
            else None,
        )
    )

    score = round(sum(item.score for item in measurements) / len(measurements))
    if (
        any(item.status is MetricStatus.FAIL for item in measurements)
        or score < config.ungradable_score_min
    ):
        status = QualityStatus.UNGRADABLE
    elif (
        all(item.status is MetricStatus.PASS for item in measurements)
        and score >= config.acceptable_score_min
    ):
        status = QualityStatus.ACCEPTABLE
    else:
        status = QualityStatus.BORDERLINE

    reasons = tuple(
        dict.fromkeys(
            item.reason for item in measurements if item.reason is not None
        )
    )
    return QualityAssessment(
        image_size=(width, height),
        score=score,
        status=status,
        measurements=tuple(measurements),
        reasons=reasons,
    )


def save_diagnostic_visualization(
    image: str | Path | Image.Image,
    output_path: str | Path,
    assessment: QualityAssessment,
) -> None:
    """Save a simple annotated copy as a visual companion to the text report."""
    annotated = _load_rgb(image)
    banner_height = max(56, 24 + 18 * len(assessment.measurements))
    canvas = Image.new("RGB", (annotated.width, annotated.height + banner_height), "black")
    canvas.paste(annotated, (0, 0))
    draw = ImageDraw.Draw(canvas)
    status_color = {
        QualityStatus.ACCEPTABLE: (35, 160, 75),
        QualityStatus.BORDERLINE: (220, 150, 25),
        QualityStatus.UNGRADABLE: (200, 45, 45),
    }[assessment.status]
    draw.rectangle(
        (0, annotated.height, annotated.width, canvas.height),
        fill=(24, 24, 24),
    )
    draw.text(
        (8, annotated.height + 5),
        f"{assessment.status.value} - {assessment.score}/100",
        fill=status_color,
    )
    for index, item in enumerate(assessment.measurements):
        draw.text(
            (8, annotated.height + 24 + index * 16),
            f"{item.name}: {item.status.value}",
            fill=(235, 235, 235),
        )
    canvas.save(output_path)

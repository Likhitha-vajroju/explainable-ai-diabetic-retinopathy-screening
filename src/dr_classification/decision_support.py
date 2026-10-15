"""Project-defined referable DR screening-result formatting."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_CLASS_NAMES = (
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
)
REFERABLE_RECOMMENDATION = "Referable screening result — ophthalmologist review recommended."
NON_REFERABLE_RECOMMENDATION = (
    "Non-referable screening result under this project definition; "
    "this does not rule out disease or replace clinical assessment."
)
SCREENING_LIMITATION = (
    "This is experimental decision support, not an independent diagnosis or "
    "treatment decision."
)


def screening_result(
    raw_probabilities: list[float] | np.ndarray,
    calibrated_probabilities: list[float] | np.ndarray | None = None,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> dict[str, Any]:
    """Create a screening result using argmax grade and project referral grouping.

    Levels 0/1 are grouped as non-referable, and levels 2/3/4 as referable.
    If calibrated probabilities are given, their maximum probability is
    reported for the raw-argmax grade (temperature scaling preserves the argmax).
    """
    raw = np.asarray(raw_probabilities, dtype=np.float64)
    if raw.shape != (5,) or len(class_names) != 5:
        raise ValueError("Screening requires five probabilities and five class names.")
    if (
        not np.isfinite(raw).all()
        or np.any(raw < 0)
        or not np.isclose(raw.sum(), 1.0, atol=1e-5)
    ):
        raise ValueError("Raw probabilities must be finite, non-negative, and sum to 1.")
    calibrated: np.ndarray | None = None
    if calibrated_probabilities is not None:
        calibrated = np.asarray(calibrated_probabilities, dtype=np.float64)
        if (
            calibrated.shape != (5,)
            or not np.isfinite(calibrated).all()
            or np.any(calibrated < 0)
            or not np.isclose(calibrated.sum(), 1.0, atol=1e-5)
        ):
            raise ValueError(
                "Calibrated probabilities must be five finite non-negative values "
                "that sum to 1."
            )

    grade = int(raw.argmax())
    referable = grade >= 2
    return {
        "dr_grade": {"level": grade, "label": class_names[grade]},
        "raw_probabilities": [
            {"level": level, "label": class_names[level], "probability": float(value)}
            for level, value in enumerate(raw)
        ],
        "raw_confidence": float(raw[grade]),
        "calibrated_confidence": (
            float(calibrated[grade]) if calibrated is not None else None
        ),
        "confidence_note": (
            "Calibrated probability is an experimental estimate of model "
            "confidence, not clinical certainty."
            if calibrated is not None
            else "No calibrated confidence is available; raw confidence is uncalibrated."
        ),
        "referable": referable,
        "referable_status": "referable" if referable else "non-referable",
        "referral_recommendation": (
            REFERABLE_RECOMMENDATION
            if referable
            else NON_REFERABLE_RECOMMENDATION
        ),
        "decision_definition": (
            "Project grouping: levels 0 and 1 are non-referable; "
            "levels 2, 3, and 4 are referable."
        ),
        "limitation": SCREENING_LIMITATION,
    }

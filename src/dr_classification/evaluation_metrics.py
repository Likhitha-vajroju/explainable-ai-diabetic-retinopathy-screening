"""Five-class and binary referable-DR evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def roc_curve_auc(
    targets: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Compute a tie-aware binary ROC curve and trapezoidal ROC-AUC."""
    target_array = np.asarray(targets, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    if (
        target_array.ndim != 1
        or score_array.ndim != 1
        or target_array.shape != score_array.shape
        or target_array.size == 0
    ):
        raise ValueError("ROC targets and scores must be non-empty equal-length vectors.")
    if not np.isin(target_array, (0, 1)).all():
        raise ValueError("Binary ROC targets must contain only 0 and 1.")
    if not np.isfinite(score_array).all():
        raise ValueError("ROC scores must be finite.")

    positives = int(target_array.sum())
    negatives = int(target_array.size - positives)
    if positives == 0 or negatives == 0:
        return {
            "auc": None,
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "defined": False,
            "reason": "ROC-AUC requires both positive and negative examples.",
            "positive_support": positives,
            "negative_support": negatives,
        }

    order = np.argsort(-score_array, kind="stable")
    sorted_scores = score_array[order]
    sorted_targets = target_array[order]
    cumulative_true_positives = np.cumsum(sorted_targets)
    cumulative_false_positives = np.cumsum(1 - sorted_targets)
    last_at_score = np.r_[
        np.flatnonzero(np.diff(sorted_scores)),
        target_array.size - 1,
    ]
    true_positives = cumulative_true_positives[last_at_score]
    false_positives = cumulative_false_positives[last_at_score]
    tpr = np.r_[0.0, true_positives / positives]
    fpr = np.r_[0.0, false_positives / negatives]
    thresholds = [None, *sorted_scores[last_at_score].tolist()]
    auc = float(np.trapezoid(tpr, fpr))
    return {
        "auc": auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds,
        "defined": True,
        "reason": None,
        "positive_support": positives,
        "negative_support": negatives,
    }


def _binary_confusion_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    true_positive = int(np.sum((targets == 1) & (predictions == 1)))
    true_negative = int(np.sum((targets == 0) & (predictions == 0)))
    false_positive = int(np.sum((targets == 0) & (predictions == 1)))
    false_negative = int(np.sum((targets == 1) & (predictions == 0)))
    precision: float | None = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    sensitivity: float | None = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else None
    )
    specificity: float | None = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else None
    )
    if sensitivity is None or true_positive + false_positive == 0:
        f1: float | None = None
    else:
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if precision is not None and precision + sensitivity > 0
            else 0.0
        )
    return {
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "support": int(targets.size),
        "referable_support": int(targets.sum()),
        "non_referable_support": int(targets.size - targets.sum()),
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
        "decision_rule": "Referable when unchanged five-class argmax predicts level 2, 3, or 4.",
    }


def evaluate_predictions(
    targets: list[int] | np.ndarray,
    probabilities: list[list[float]] | np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Evaluate five-class and binary referable performance from model scores."""
    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if target_array.ndim != 1 or target_array.size == 0:
        raise ValueError("Evaluation targets must be a non-empty vector.")
    if probability_array.shape != (target_array.size, 5):
        raise ValueError("Evaluation probabilities must have shape (n_samples, 5).")
    if len(class_names) != 5:
        raise ValueError("Exactly five class names are required.")
    if not np.isin(target_array, np.arange(5)).all():
        raise ValueError("Evaluation targets must use class indices 0 through 4.")
    if not np.isfinite(probability_array).all() or np.any(probability_array < 0):
        raise ValueError("Evaluation probabilities must be finite and non-negative.")
    if not np.allclose(probability_array.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each row of class probabilities must sum to 1.")

    predictions = probability_array.argmax(axis=1)
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (target_array, predictions), 1)
    support = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_counts,
        out=np.zeros(5, dtype=np.float64),
        where=predicted_counts != 0,
    )
    sensitivity = np.divide(
        true_positive,
        support,
        out=np.zeros(5, dtype=np.float64),
        where=support != 0,
    )
    true_negative = confusion.sum() - support - predicted_counts + true_positive
    specificity = np.divide(
        true_negative,
        true_negative + (predicted_counts - true_positive),
        out=np.zeros(5, dtype=np.float64),
        where=(true_negative + predicted_counts - true_positive) != 0,
    )
    f1 = np.divide(
        2 * precision * sensitivity,
        precision + sensitivity,
        out=np.zeros(5, dtype=np.float64),
        where=(precision + sensitivity) != 0,
    )

    per_class: list[dict[str, Any]] = []
    roc_curves: dict[str, Any] = {}
    class_auc_values: list[float] = []
    for class_index, class_name in enumerate(class_names):
        roc = roc_curve_auc(
            target_array == class_index,
            probability_array[:, class_index],
        )
        roc_curves[str(class_index)] = roc
        if roc["auc"] is not None:
            class_auc_values.append(roc["auc"])
        per_class.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "precision": float(precision[class_index]),
                "recall": float(sensitivity[class_index]),
                "sensitivity": float(sensitivity[class_index]),
                "specificity": float(specificity[class_index]),
                "f1": float(f1[class_index]),
                "support": int(support[class_index]),
                "roc_auc_ovr": roc["auc"],
                "roc_auc_defined": roc["defined"],
                "roc_auc_reason": roc["reason"],
            }
        )

    binary_targets = (target_array >= 2).astype(np.int64)
    binary_predictions = (predictions >= 2).astype(np.int64)
    referable_scores = probability_array[:, 2:5].sum(axis=1)
    binary_metrics = _binary_confusion_metrics(binary_targets, binary_predictions)
    binary_roc = roc_curve_auc(binary_targets, referable_scores)
    binary_metrics["roc_auc"] = binary_roc["auc"]
    binary_metrics["roc_auc_defined"] = binary_roc["defined"]
    binary_metrics["roc_auc_reason"] = binary_roc["reason"]
    binary_metrics["roc_curve"] = binary_roc

    return {
        "accuracy": float(np.mean(predictions == target_array)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(sensitivity.mean()),
        "macro_sensitivity": float(sensitivity.mean()),
        "macro_specificity_ovr": float(specificity.mean()),
        "macro_f1": float(f1.mean()),
        "macro_roc_auc_ovr": (
            float(np.mean(class_auc_values)) if class_auc_values else None
        ),
        "macro_roc_auc_defined_classes": len(class_auc_values),
        "support": int(target_array.size),
        "class_support": {
            class_names[index]: int(support[index]) for index in range(5)
        },
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "roc_curves_ovr": roc_curves,
        "binary_referable": binary_metrics,
        "binary_referable_score": (
            "Sum of probabilities for levels 2, 3, and 4; used for ROC-AUC only."
        ),
    }

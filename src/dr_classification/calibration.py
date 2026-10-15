"""Post-hoc multiclass temperature scaling and confidence calibration metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


def _validated_logits_targets(
    logits: np.ndarray | torch.Tensor,
    targets: list[int] | np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_tensor = torch.as_tensor(logits, dtype=torch.float64).detach().cpu()
    targets_tensor = torch.as_tensor(targets, dtype=torch.long).detach().cpu()
    if (
        logits_tensor.ndim != 2
        or logits_tensor.shape[1] != 5
        or logits_tensor.shape[0] == 0
        or targets_tensor.shape != (logits_tensor.shape[0],)
    ):
        raise ValueError("Calibration requires logits of shape (n, 5) and n targets.")
    if not torch.isfinite(logits_tensor).all():
        raise ValueError("Calibration logits must be finite.")
    if not torch.isin(targets_tensor, torch.arange(5)).all():
        raise ValueError("Calibration targets must use class indices 0 through 4.")
    return logits_tensor, targets_tensor


def fit_temperature(
    validation_logits: np.ndarray | torch.Tensor,
    validation_targets: list[int] | np.ndarray | torch.Tensor,
) -> float:
    """Fit a positive scalar temperature by minimizing validation NLL.

    Fit only on a calibration/validation partition, then apply the returned
    temperature to a disjoint evaluation or inference partition.
    """
    logits, targets = _validated_logits_targets(validation_logits, validation_targets)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        loss = F.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature_value = float(log_temperature.detach().exp())
    if not np.isfinite(temperature_value) or temperature_value <= 0:
        raise RuntimeError("Temperature fitting produced an invalid value.")
    return temperature_value


def apply_temperature(
    logits: np.ndarray | torch.Tensor,
    temperature: float,
) -> np.ndarray:
    """Convert five-class logits to temperature-scaled probabilities."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive value.")
    values = torch.as_tensor(logits, dtype=torch.float64).detach().cpu()
    if values.ndim not in (1, 2) or values.shape[-1] != 5:
        raise ValueError("Logits must have shape (5,) or (n, 5).")
    if not torch.isfinite(values).all():
        raise ValueError("Logits must be finite.")
    return torch.softmax(values / temperature, dim=-1).cpu().numpy()


def calibration_summary(
    probabilities: np.ndarray | list[list[float]],
    targets: list[int] | np.ndarray,
    *,
    number_of_bins: int = 10,
) -> dict[str, object]:
    """Compute top-label ECE, Brier score, NLL, and reliability-bin values."""
    scores = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[1] != 5 or scores.shape[0] == 0:
        raise ValueError("Calibration probabilities must have shape (n, 5).")
    if labels.shape != (scores.shape[0],) or not np.isin(labels, np.arange(5)).all():
        raise ValueError("Calibration targets must be n class indices from 0 through 4.")
    if (
        not np.isfinite(scores).all()
        or np.any(scores < 0)
        or not np.allclose(scores.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("Each calibration probability row must be finite and sum to 1.")
    if number_of_bins < 1:
        raise ValueError("number_of_bins must be at least 1.")

    predicted = scores.argmax(axis=1)
    confidence = scores.max(axis=1)
    correct = predicted == labels
    bin_edges = np.linspace(0.0, 1.0, number_of_bins + 1)
    bins: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(number_of_bins):
        lower, upper = float(bin_edges[index]), float(bin_edges[index + 1])
        in_bin = (confidence >= lower) & (
            confidence <= upper if index == number_of_bins - 1 else confidence < upper
        )
        count = int(in_bin.sum())
        if count:
            average_confidence = float(confidence[in_bin].mean())
            accuracy = float(correct[in_bin].mean())
            ece += count / labels.size * abs(accuracy - average_confidence)
        else:
            average_confidence = 0.0
            accuracy = 0.0
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": average_confidence,
                "accuracy": accuracy,
            }
        )

    one_hot = np.eye(5, dtype=np.float64)[labels]
    clipped = np.clip(scores, np.finfo(np.float64).tiny, 1.0)
    return {
        "expected_calibration_error": float(ece),
        "brier_score": float(np.mean(np.sum((scores - one_hot) ** 2, axis=1))),
        "negative_log_likelihood": float(-np.log(clipped[np.arange(labels.size), labels]).mean()),
        "support": int(labels.size),
        "number_of_bins": number_of_bins,
        "bins": bins,
    }


def save_reliability_diagram(
    raw_summary: dict[str, object],
    calibrated_summary: dict[str, object],
    output_path: str | Path,
) -> None:
    """Save raw/calibrated top-label reliability curves and confidence histogram."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot((0, 1), (0, 1), "--", color="gray", label="Perfect reliability")
    for summary, label, color in (
        (raw_summary, "Raw", "tab:blue"),
        (calibrated_summary, "Temperature-scaled", "tab:orange"),
    ):
        bins = summary["bins"]
        occupied = [item for item in bins if item["count"]]
        axes[0].plot(
            [item["mean_confidence"] for item in occupied],
            [item["accuracy"] for item in occupied],
            marker="o",
            label=f"{label} (ECE={summary['expected_calibration_error']:.3f})",
            color=color,
        )
        axes[1].bar(
            [item["mean_confidence"] for item in occupied],
            [item["count"] for item in occupied],
            width=0.06,
            alpha=0.45,
            label=label,
            color=color,
        )
    axes[0].set(
        title="Top-label reliability (not clinical certainty)",
        xlabel="Mean predicted class probability",
        ylabel="Observed accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].set(
        title="Samples per confidence bin",
        xlabel="Mean predicted class probability",
        ylabel="Number of samples",
        xlim=(0, 1),
    )
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.suptitle("Experimental DR classifier confidence calibration")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)

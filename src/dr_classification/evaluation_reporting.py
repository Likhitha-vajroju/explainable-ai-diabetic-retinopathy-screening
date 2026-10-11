"""Create presentation-ready evaluation plots and a transparent Markdown report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def save_evaluation_dashboard(
    metrics: dict[str, Any],
    class_names: tuple[str, ...],
    output_path: str | Path,
) -> None:
    """Save five-class/binary confusion matrices and ROC curves in one dashboard."""
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    figure.suptitle(
        "Experimental DR classifier evaluation - not clinical validation",
        fontsize=16,
        fontweight="bold",
    )
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        row_totals,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_totals != 0,
    )
    axis = axes[0, 0]
    axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    axis.set(
        title="Five-class confusion matrix (row-normalized; counts shown)",
        xlabel="Predicted DR level",
        ylabel="True DR level",
        xticks=range(5),
        yticks=range(5),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for row in range(5):
        for column in range(5):
            axis.text(
                column,
                row,
                f"{confusion[row, column]}\n{normalized[row, column]:.0%}",
                ha="center",
                va="center",
                color="white" if normalized[row, column] > 0.55 else "black",
                fontsize=8,
            )

    axis = axes[0, 1]
    binary_confusion = np.asarray(
        metrics["binary_referable"]["confusion_matrix"],
        dtype=np.int64,
    )
    axis.imshow(binary_confusion, cmap="Oranges")
    axis.set(
        title="Referable DR confusion matrix (counts)",
        xlabel="Predicted",
        ylabel="True",
        xticks=(0, 1),
        yticks=(0, 1),
        xticklabels=("Non-referable", "Referable"),
        yticklabels=("Non-referable", "Referable"),
    )
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(binary_confusion[row, column]),
                ha="center",
                va="center",
                fontsize=12,
            )

    axis = axes[1, 0]
    any_multiclass_curve = False
    for class_index, class_name in enumerate(class_names):
        curve = metrics["roc_curves_ovr"][str(class_index)]
        if not curve["defined"]:
            continue
        axis.plot(
            curve["fpr"],
            curve["tpr"],
            label=f"{class_name} (AUC={curve['auc']:.3f})",
        )
        any_multiclass_curve = True
    axis.plot((0, 1), (0, 1), linestyle="--", color="gray", label="Chance")
    axis.set(
        title="Five-class one-vs-rest ROC curves",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.grid(True, alpha=0.25)
    if any_multiclass_curve:
        axis.legend(loc="lower right", fontsize=8)
    else:
        axis.text(0.5, 0.5, "ROC-AUC undefined for this split", ha="center")

    axis = axes[1, 1]
    binary_roc = metrics["binary_referable"]["roc_curve"]
    if binary_roc["defined"]:
        axis.plot(
            binary_roc["fpr"],
            binary_roc["tpr"],
            label=f"Referable DR (AUC={binary_roc['auc']:.3f})",
            color="darkred",
        )
        axis.plot((0, 1), (0, 1), linestyle="--", color="gray", label="Chance")
        axis.legend(loc="lower right")
    else:
        axis.text(0.5, 0.5, binary_roc["reason"], ha="center", wrap=True)
    axis.set(
        title="Binary referable DR ROC curve",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.grid(True, alpha=0.25)

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _format_metric(value: float | int | None, percent: bool = False) -> str:
    if value is None:
        return "Not defined (both outcome classes are required)"
    if percent:
        return f"{value:.2%}"
    return f"{value:.4f}"


def build_evaluation_report(payload: dict[str, Any]) -> str:
    """Render metrics, split composition, target status, and limitations."""
    class_names = payload["class_names"]
    binary = payload["binary_referable"]
    targets = payload["targets"]
    sensitivity_target = targets["referable_sensitivity_strictly_above"]
    specificity_target = targets["referable_specificity_strictly_above"]
    sensitivity_met = (
        binary["sensitivity"] is not None
        and binary["sensitivity"] > sensitivity_target
    )
    specificity_met = (
        binary["specificity"] is not None
        and binary["specificity"] > specificity_target
    )
    rows = [
        "# Diabetic retinopathy classifier evaluation",
        "",
        "> **Experimental benchmark results only. This report is not clinical validation,",
        "> does not establish suitability for patient care, and is not a diagnosis.**",
        "",
        "## Evaluation setup",
        "",
        f"- Evaluated split: `{payload['split']}`.",
        f"- Checkpoint: `{payload['checkpoint']}`.",
        f"- Selected checkpoint epoch: `{payload.get('checkpoint_epoch')}`.",
        f"- Evaluated samples: {payload['support']}.",
        "- Five-class predictions: unmodified model argmax.",
        "- Binary referable decision: argmax level 2, 3, or 4; no prediction threshold was tuned.",
        "- Binary ROC-AUC score: sum of model probabilities for levels 2, 3, and 4.",
        "- Five-class precision, sensitivity/recall, specificity, and F1 are macro averages; specificity is one-vs-rest.",
        "- ROC-AUC is computed one-vs-rest per class where both positive and negative examples exist.",
        "",
        "## Dataset split and class imbalance",
        "",
        "| Split | Samples | Class counts (levels 0-4) | Largest/smallest non-empty class ratio |",
        "| --- | ---: | --- | ---: |",
    ]
    imbalance = payload["class_imbalance"]
    supports = payload["dataset_split_support"]
    for split_name in ("train", "validation", "test"):
        distribution = imbalance[split_name]
        counts = distribution["counts"]
        count_text = ", ".join(str(counts[str(index)]) for index in range(5))
        ratio = distribution["largest_to_smallest_nonzero_ratio"]
        rows.append(
            f"| {split_name} | {supports[split_name]} | {count_text} | "
            f"{_format_metric(ratio)} |"
        )
    rows.extend(
        (
            "",
            "Class order and names: "
            + ", ".join(
                f"{index} = {name}" for index, name in enumerate(class_names)
            )
            + ".",
            "The training loss uses inverse-frequency class weights computed from "
            "training labels only. Evaluation metrics are computed on the selected "
            "split without changing predictions.",
            "",
            "## Five-class results",
            "",
            "| Metric | Measured value |",
            "| --- | ---: |",
            f"| Loss (train-weighted cross-entropy) | {_format_metric(payload.get('loss'))} |",
            f"| Accuracy | {_format_metric(payload['accuracy'], percent=True)} |",
            f"| Macro precision | {_format_metric(payload['macro_precision'], percent=True)} |",
            f"| Macro recall / sensitivity | {_format_metric(payload['macro_sensitivity'], percent=True)} |",
            f"| Macro one-vs-rest specificity | {_format_metric(payload['macro_specificity_ovr'], percent=True)} |",
            f"| Macro F1 | {_format_metric(payload['macro_f1'], percent=True)} |",
            f"| Macro one-vs-rest ROC-AUC | {_format_metric(payload['macro_roc_auc_ovr'])} "
            f"({payload['macro_roc_auc_defined_classes']}/5 classes defined) |",
            "",
            "| Level | Class | Support | Precision | Sensitivity/recall | Specificity | F1 | ROC-AUC OVR |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for class_metrics in payload["per_class"]:
        rows.append(
            f"| {class_metrics['class_index']} | {class_metrics['class_name']} | "
            f"{class_metrics['support']} | "
            f"{_format_metric(class_metrics['precision'], percent=True)} | "
            f"{_format_metric(class_metrics['sensitivity'], percent=True)} | "
            f"{_format_metric(class_metrics['specificity'], percent=True)} | "
            f"{_format_metric(class_metrics['f1'], percent=True)} | "
            f"{_format_metric(class_metrics['roc_auc_ovr'])} |"
        )
    rows.extend(
        (
            "",
            "### Five-class confusion matrix (rows=true, columns=predicted)",
            "",
            "| True / Predicted | "
            + " | ".join(f"{index}: {name}" for index, name in enumerate(class_names))
            + " |",
            "| --- | " + " | ".join("---:" for _ in class_names) + " |",
        )
    )
    for index, row in enumerate(payload["confusion_matrix"]):
        rows.append(
            f"| {index}: {class_names[index]} | "
            + " | ".join(str(count) for count in row)
            + " |"
        )

    rows.extend(
        (
            "",
            "## Binary referable DR results",
            "",
            "**Non-referable:** levels 0 and 1. **Referable:** levels 2, 3, and 4.",
            "The confusion matrix and thresholded metrics use the unchanged five-class argmax "
            "grouped into these two sets. ROC-AUC uses the summed referable probability "
            "and is threshold-independent.",
            "",
            "| Metric | Measured value | Target | Target status |",
            "| --- | ---: | ---: | --- |",
            f"| Sensitivity / recall | {_format_metric(binary['sensitivity'], percent=True)} "
            f"| >{sensitivity_target:.0%} | "
            f"{'MET' if sensitivity_met else 'NOT MET' if binary['sensitivity'] is not None else 'NOT EVALUABLE'} |",
            f"| Specificity | {_format_metric(binary['specificity'], percent=True)} "
            f"| >{specificity_target:.0%} | "
            f"{'MET' if specificity_met else 'NOT MET' if binary['specificity'] is not None else 'NOT EVALUABLE'} |",
            f"| Precision | {_format_metric(binary['precision'], percent=True)} | Not specified | N/A |",
            f"| F1 | {_format_metric(binary['f1'], percent=True)} | Not specified | N/A |",
            f"| ROC-AUC | {_format_metric(binary['roc_auc'])} | Not specified | N/A |",
            f"| Support (non-referable / referable) | {binary['non_referable_support']} / "
            f"{binary['referable_support']} | N/A | N/A |",
            "",
            "### Binary confusion matrix (rows=true, columns=predicted)",
            "",
            "| True / Predicted | Non-referable | Referable |",
            "| --- | ---: | ---: |",
            f"| Non-referable | {binary['confusion_matrix'][0][0]} | "
            f"{binary['confusion_matrix'][0][1]} |",
            f"| Referable | {binary['confusion_matrix'][1][0]} | "
            f"{binary['confusion_matrix'][1][1]} |",
            "",
            "The >90% sensitivity and >85% specificity values are target requirements, "
            "not expected or guaranteed results. No predictions or thresholds were "
            "modified to meet them.",
            "",
            "## Limitations and possible dataset bias",
            "",
            "- Dataset labels and the mapping from codes to severity levels must be verified "
            "against authoritative source documentation before clinical interpretation.",
            "- The dataset documentation reports an unknown license; confirm terms before "
            "redistribution or use.",
            "- Camera, acquisition protocol, image quality, compression, site, age, "
            "ethnicity, and disease-spectrum representation may differ from deployment "
            "settings; subgroup metadata may not be available to measure those effects.",
            "- Patient/group separation depends on inferred identifier prefixes. Opaque IDs "
            "fall back to sample-level grouping, so patient-level leakage cannot be ruled out.",
            "- Class imbalance, selection bias, label noise, duplicate images, and the "
            "published split design can affect measured performance. Review dataset audit "
            "and class supports before interpreting aggregate scores.",
            "- A held-out dataset benchmark estimates performance only on that split under "
            "its data and label assumptions. Clinical validation additionally requires "
            "representative external cohorts, prespecified protocols, calibration and "
            "subgroup analysis, workflow evaluation, and appropriate clinical oversight.",
            "- Softmax probabilities and ROC-AUC do not establish calibrated confidence, "
            "clinical benefit, sensitivity/specificity in another population, or fitness "
            "for screening.",
            "",
            "## Artifacts",
            "",
            "See the companion evaluation dashboard PNG and confusion-matrix CSV in the "
            "same output directory. The machine-readable metrics JSON contains the "
            "full ROC coordinates and split counts.",
            "",
        )
    )
    return "\n".join(rows)

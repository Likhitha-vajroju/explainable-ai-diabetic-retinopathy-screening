"""Orchestrate the existing research modules into screening-review reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import torch

from src.dr_classification.classifier import (
    ClassificationConfig,
    predict_image,
)
from src.dr_classification.decision_support import SCREENING_LIMITATION
from src.explainable_ai import generate_gradcam
from src.image_quality_assessment import assess_image
from src.lesion_retinal_analysis import (
    EVIDENCE_WARNING,
    localize_optic_disc,
    save_evidence_visualization,
    segment_vessels,
)
from src.preprocessing.image_pipeline import preprocess_image


MEDICAL_DISCLAIMER = (
    "This is an AI-assisted screening prototype, not a definitive diagnosis. "
    "It does not determine treatment or replace assessment by a qualified "
    "ophthalmologist. Model performance depends on the dataset, labels, and "
    "validation conditions; results may not generalize to other populations, "
    "devices, or workflows."
)


def _recommendation(quality_status: str, referable: bool) -> str:
    if quality_status == "UNGRADABLE":
        return (
            "Image quality is flagged as ungradable by engineering heuristics. "
            "Consider recapturing the image and arrange qualified clinical review; "
            "do not rely on this model output."
        )
    if referable:
        return "Referable screening result — ophthalmologist review recommended."
    return (
        "Non-referable under this project definition. Continue the established "
        "screening pathway; this result does not rule out disease or replace "
        "clinical assessment."
    )


def build_screening_result(
    *,
    image_size: tuple[int, int],
    quality: dict[str, Any],
    preprocessing: dict[str, Any],
    prediction: dict[str, Any],
    gradcam: dict[str, Any],
    retinal_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Create a report payload without adding patient identity or demographics."""
    screening = prediction["screening_result"]
    return {
        "report_type": "AI-assisted diabetic retinopathy screening prototype",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "patient_information": None,
        "image": {"width": image_size[0], "height": image_size[1]},
        "image_quality": quality,
        "preprocessing": preprocessing,
        "dr_prediction": {
            "grade": prediction["predicted_dr_level"],
            "grade_label": prediction["predicted_label"],
            "raw_probabilities": prediction["class_probabilities"],
            "raw_confidence": prediction["raw_confidence"],
            "calibrated_confidence": screening["calibrated_confidence"],
            "confidence_note": screening["confidence_note"],
        },
        "referable_screening": {
            "status": screening["referable_status"],
            "referable": screening["referable"],
            "decision_definition": screening["decision_definition"],
            "recommendation": _recommendation(
                quality["status"], screening["referable"]
            ),
        },
        "gradcam": gradcam,
        "retinal_lesion_evidence": retinal_evidence,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
        "decision_support_limitation": SCREENING_LIMITATION,
    }


def _markdown_report(result: dict[str, Any]) -> str:
    prediction = result["dr_prediction"]
    quality = result["image_quality"]
    preprocessing = result["preprocessing"]
    referral = result["referable_screening"]
    rows = [
        "# AI-assisted diabetic retinopathy screening report",
        "",
        f"> **{MEDICAL_DISCLAIMER}**",
        "",
        f"Generated (UTC): {result['generated_at_utc']}",
        "",
        "## Image quality",
        "",
        f"- Engineering quality status: **{quality['status']}**",
        f"- Engineering score: {quality['score']}/100 (not clinically validated).",
        f"- Image dimensions: {result['image']['width']} × {result['image']['height']}.",
        f"- Findings: {', '.join(quality['reasons']) or 'No heuristic flags.'}",
        "",
        "## Preprocessing",
        "",
        f"- Status: **{preprocessing['status']}**.",
        f"- Operations: {preprocessing['description']}",
        "",
        "## DR classifier screening result",
        "",
        f"- Predicted grade: **Level {prediction['grade']} — "
        f"{prediction['grade_label']}**.",
        f"- Raw maximum class probability: {prediction['raw_confidence']:.1%}.",
        f"- Calibrated confidence for predicted grade: "
        f"{'not available.' if prediction['calibrated_confidence'] is None else format(prediction['calibrated_confidence'], '.1%') + '.'}",
        f"- Confidence interpretation: {prediction['confidence_note']}",
        "",
        "| DR level | Class | Raw probability |",
        "| ---: | --- | ---: |",
    ]
    for probability in prediction["raw_probabilities"]:
        rows.append(
            f"| {probability['dr_level']} | {probability['label']} | "
            f"{probability['probability']:.2%} |"
        )
    gradcam = result["gradcam"]
    retinal = result["retinal_lesion_evidence"]
    rows.extend(
        (
            "",
            "## Referral support and recommended next action",
            "",
            f"- Status: **{referral['status']}** under this project's grouping.",
            f"- Next action: {referral['recommendation']}",
            f"- Grouping: {referral['decision_definition']}",
            "",
            "## Visual evidence",
            "",
            f"- Grad-CAM: {gradcam['status']}. "
            + (
                f"Visualization: `{gradcam['visualization_path']}`. "
                f"{gradcam.get('warning', '')}"
                if gradcam["status"] == "available"
                else gradcam.get("message", "No visualization generated.")
            ),
            f"- Retinal algorithm/annotation evidence: {retinal['status']}. "
            + (
                f"Visualization: `{retinal['visualization_path']}`. "
                f"{retinal['warning']}"
                if retinal["status"] == "available"
                else retinal.get("message", "No retinal evidence visualization generated.")
            ),
            "- Grad-CAM attention is separate from algorithmic structure candidates "
            "and from externally annotated lesion evidence. None is a confirmed "
            "clinical lesion or diagnosis.",
            "",
            "## Limitations",
            "",
            f"- {SCREENING_LIMITATION}",
            "- Model performance depends on the dataset, label quality, population, "
            "devices, and validation protocol; it is not established for other settings.",
            "- No patient identity or demographic information was provided or inferred.",
            "",
        )
    )
    return "\n".join(rows)


def _save_pdf(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction = result["dr_prediction"]
    referral = result["referable_screening"]
    quality = result["image_quality"]
    with PdfPages(output_path) as pdf:
        figure = plt.figure(figsize=(8.27, 11.69))
        axis = figure.add_axes((0.09, 0.06, 0.82, 0.88))
        axis.axis("off")
        lines = [
            "AI-ASSISTED DIABETIC RETINOPATHY",
            "SCREENING REPORT",
            "",
            "PROTOTYPE — NOT A DEFINITIVE DIAGNOSIS",
            "",
            f"Generated UTC: {result['generated_at_utc']}",
            "",
            "IMAGE QUALITY",
            f"Status: {quality['status']} | Engineering score: {quality['score']}/100",
            f"Dimensions: {result['image']['width']} x {result['image']['height']}",
            "",
            "PREPROCESSING",
            f"{result['preprocessing']['status']}: {result['preprocessing']['description']}",
            "",
            "DR CLASSIFIER",
            f"Predicted grade: Level {prediction['grade']} - {prediction['grade_label']}",
            f"Raw maximum probability: {prediction['raw_confidence']:.1%}",
            "Calibrated confidence: "
            + (
                "not available"
                if prediction["calibrated_confidence"] is None
                else f"{prediction['calibrated_confidence']:.1%}"
            ),
            "",
            "CLASS PROBABILITIES",
        ]
        lines.extend(
            f"Level {item['dr_level']} - {item['label']}: {item['probability']:.1%}"
            for item in prediction["raw_probabilities"]
        )
        lines.extend(
            (
                "",
                "REFERABLE SCREENING",
                f"Status: {referral['status']}",
                f"Next action: {referral['recommendation']}",
                "",
                "VISUAL EVIDENCE",
                f"Grad-CAM: {result['gradcam']['status']}",
                f"Retinal evidence: {result['retinal_lesion_evidence']['status']}",
                "",
                "MEDICAL DISCLAIMER",
                MEDICAL_DISCLAIMER,
                "",
                SCREENING_LIMITATION,
                "Grad-CAM and algorithmic evidence are not confirmed clinical lesions.",
                "No patient identity or demographic details were provided or inferred.",
            )
        )
        axis.text(
            0,
            1,
            "\n".join(lines),
            va="top",
            fontsize=9,
            linespacing=1.35,
            wrap=True,
        )
        figure.suptitle("Screening summary", fontsize=15, fontweight="bold")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        evidence_paths = [
            ("Grad-CAM visualization", result["gradcam"].get("visualization_path")),
            (
                "Retinal algorithm/annotation visualization",
                result["retinal_lesion_evidence"].get("visualization_path"),
            ),
        ]
        available_paths: list[tuple[str, Path]] = []
        for title, path in evidence_paths:
            if not path:
                continue
            image_path = Path(path)
            if not image_path.is_absolute():
                image_path = output_path.parent / image_path
            if image_path.is_file():
                available_paths.append((title, image_path))
        if available_paths:
            figure, axes = plt.subplots(
                len(available_paths),
                1,
                figsize=(8.27, 11.69),
                squeeze=False,
            )
            for axis, (title, image_path) in zip(axes[:, 0], available_paths):
                with Image.open(image_path) as image:
                    axis.imshow(image.convert("RGB"))
                axis.set_title(title)
                axis.axis("off")
            figure.suptitle("Visual evidence — not a diagnosis", fontsize=14)
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


def write_screening_report(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the machine-readable JSON, Markdown report, and PDF summary."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": destination / "screening_result.json",
        "markdown": destination / "screening_report.md",
        "pdf": destination / "screening_report.pdf",
    }
    paths["json"].write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths["markdown"].write_text(_markdown_report(result), encoding="utf-8")
    _save_pdf(result, paths["pdf"])
    return paths


def create_screening_report(
    image_path: str | Path,
    model: torch.nn.Module,
    config: ClassificationConfig,
    device: torch.device,
    output_dir: str | Path,
    calibration_temperature: float | None = None,
) -> dict[str, Any]:
    """Run image quality, preprocessing, classifier, Grad-CAM and retinal evidence.

    Grad-CAM and retinal analysis are optional evidence steps: expected
    image/algorithm failures are recorded in the report without blocking the
    classifier result.
    """
    source = Path(image_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input fundus image does not exist: {source}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    quality_result = assess_image(source)
    preprocessing_result = preprocess_image(source, config.preprocessing)
    prediction = predict_image(
        source,
        model,
        config,
        device,
        calibration_temperature=calibration_temperature,
    )

    gradcam_path = destination / "gradcam.png"
    try:
        gradcam_result = generate_gradcam(
            source, model, config, device, gradcam_path
        )
        gradcam = {
            "status": "available",
            **gradcam_result.explanation,
        }
    except (OSError, RuntimeError, ValueError) as error:
        gradcam = {
            "status": "unavailable",
            "message": f"Optional Grad-CAM generation failed: {error}",
            "visualization_path": None,
        }

    retinal_path = destination / "retinal_evidence.png"
    try:
        vessel_mask = segment_vessels(source)
        optic_disc = localize_optic_disc(source)
        save_evidence_visualization(
            source,
            retinal_path,
            vessel_mask=vessel_mask,
            optic_disc=optic_disc,
        )
        retinal_evidence = {
            "status": "available",
            "visualization_path": str(retinal_path),
            "vessel_candidate_pixels": int(np.count_nonzero(vessel_mask)),
            "optic_disc_candidate": (
                {
                    "center_xy": list(optic_disc.center_xy),
                    "bounding_box_xyxy": list(optic_disc.bounding_box_xyxy),
                    "label": optic_disc.label,
                }
                if optic_disc is not None
                else None
            ),
            "lesion_annotations": "none supplied; no lesion detection claimed",
            "warning": EVIDENCE_WARNING,
        }
    except (OSError, RuntimeError, ValueError) as error:
        retinal_evidence = {
            "status": "unavailable",
            "visualization_path": None,
            "lesion_annotations": "none supplied; no lesion detection claimed",
            "message": f"Optional retinal analysis failed: {error}",
            "warning": EVIDENCE_WARNING,
        }

    result = build_screening_result(
        image_size=preprocessing_result.original_image.size,
        quality=quality_result.to_dict(),
        preprocessing={
            "status": "complete",
            "description": (
                f"Configured preprocessing applied. Output size: "
                f"{config.preprocessing.output_size}; aspect ratio preserved: "
                f"{config.preprocessing.preserve_aspect_ratio}; illumination "
                f"normalization: {config.preprocessing.normalize_illumination}; "
                f"CLAHE: {config.preprocessing.apply_clahe}; noise reduction "
                f"radius: {config.preprocessing.noise_reduction_radius}."
            ),
            "processed_size": {
                "width": preprocessing_result.enhanced_image.width,
                "height": preprocessing_result.enhanced_image.height,
            },
        },
        prediction=prediction,
        gradcam=gradcam,
        retinal_evidence=retinal_evidence,
    )
    result["artifacts"] = {
        "gradcam_visualization": gradcam["visualization_path"],
        "retinal_evidence_visualization": retinal_evidence["visualization_path"],
    }
    return result

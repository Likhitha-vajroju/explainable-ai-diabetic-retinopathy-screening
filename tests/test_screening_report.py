from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw
from torch import nn

from src.dr_classification.classifier import ClassificationConfig
from src.preprocessing.image_pipeline import PreprocessingConfig
from src.reporting.screening_report import (
    MEDICAL_DISCLAIMER,
    build_screening_result,
    create_screening_report,
    write_screening_report,
)


class _FeatureBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv2 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv2(inputs))


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer4 = nn.Sequential(_FeatureBlock())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.layer4(inputs)
        return self.fc(self.pool(features).flatten(1))


class ScreeningReportTests(unittest.TestCase):
    def make_result(self) -> dict[str, object]:
        probabilities = [
            {"dr_level": index, "label": name, "probability": probability}
            for index, (name, probability) in enumerate(
                (
                    ("No DR", 0.05),
                    ("Mild", 0.05),
                    ("Moderate", 0.65),
                    ("Severe", 0.15),
                    ("Proliferative DR", 0.10),
                )
            )
        ]
        return build_screening_result(
            image_size=(100, 80),
            quality={
                "status": "ACCEPTABLE",
                "score": 82,
                "reasons": [],
                "measurements": [],
            },
            preprocessing={
                "status": "complete",
                "description": "Synthetic unit-test preprocessing configuration.",
                "processed_size": {"width": 64, "height": 64},
            },
            prediction={
                "predicted_dr_level": 2,
                "predicted_label": "Moderate",
                "class_probabilities": probabilities,
                "raw_confidence": 0.65,
                "screening_result": {
                    "referable_status": "referable",
                    "referable": True,
                    "calibrated_confidence": 0.52,
                    "confidence_note": "Experimental confidence, not clinical certainty.",
                    "decision_definition": "Test project definition.",
                    "referral_recommendation": (
                        "Referable screening result — ophthalmologist review recommended."
                    ),
                },
            },
            gradcam={
                "status": "unavailable",
                "visualization_path": None,
                "message": "Optional Grad-CAM not available in unit test.",
            },
            retinal_evidence={
                "status": "unavailable",
                "visualization_path": None,
                "message": "Optional retinal analysis not available in unit test.",
            },
        )

    def test_report_payload_has_no_invented_patient_information(self) -> None:
        result = self.make_result()

        self.assertIsNone(result["patient_information"])
        self.assertEqual(result["image"]["width"], 100)
        self.assertEqual(result["dr_prediction"]["grade_label"], "Moderate")
        self.assertEqual(result["dr_prediction"]["calibrated_confidence"], 0.52)
        self.assertTrue(result["referable_screening"]["referable"])
        self.assertIn("ophthalmologist review recommended", result["referable_screening"]["recommendation"].lower())
        self.assertIn("not a definitive diagnosis", result["medical_disclaimer"])

    def test_writes_machine_readable_and_human_readable_reports(self) -> None:
        result = self.make_result()

        with tempfile.TemporaryDirectory() as directory:
            paths = write_screening_report(result, directory)

            self.assertTrue(paths["json"].is_file())
            self.assertTrue(paths["markdown"].is_file())
            self.assertTrue(paths["pdf"].is_file())
            loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(loaded["patient_information"], None)
            report_text = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("AI-assisted diabetic retinopathy screening report", report_text)
            self.assertIn("not a definitive diagnosis", report_text.lower())
            self.assertIn("model performance depends", report_text.lower())
            self.assertIn("ophthalmologist review recommended", report_text.lower())
            self.assertIn(MEDICAL_DISCLAIMER, report_text)
            self.assertGreater(paths["pdf"].stat().st_size, 0)

    def test_full_pipeline_records_optional_evidence_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "sample.png"
            image = Image.new("RGB", (80, 64), "black")
            draw = ImageDraw.Draw(image)
            draw.ellipse((5, 4, 75, 60), fill=(130, 80, 55))
            draw.line((40, 20, 40, 55), fill=(35, 20, 15), width=2)
            image.save(image_path)
            config = ClassificationConfig(
                pretrained=False,
                image_size=64,
                preprocessing=PreprocessingConfig(output_size=(64, 64)),
            )
            model = _TinyClassifier()

            with (
                patch(
                    "src.reporting.screening_report.generate_gradcam",
                    side_effect=RuntimeError("Grad-CAM smoke failure"),
                ),
                patch(
                    "src.reporting.screening_report.segment_vessels",
                    side_effect=RuntimeError("Retinal smoke failure"),
                ),
            ):
                result = create_screening_report(
                    image_path,
                    model,
                    config,
                    torch.device("cpu"),
                    root / "report",
                )

            self.assertIn(result["image_quality"]["status"], ("ACCEPTABLE", "BORDERLINE", "UNGRADABLE"))
            self.assertEqual(result["preprocessing"]["status"], "complete")
            self.assertIn("grade", result["dr_prediction"])
            self.assertEqual(result["gradcam"]["status"], "unavailable")
            self.assertIn("Grad-CAM smoke failure", result["gradcam"]["message"])
            self.assertEqual(result["retinal_lesion_evidence"]["status"], "unavailable")
            self.assertIn("Retinal smoke failure", result["retinal_lesion_evidence"]["message"])
            self.assertTrue(result["referable_screening"]["recommendation"])


if __name__ == "__main__":
    unittest.main()

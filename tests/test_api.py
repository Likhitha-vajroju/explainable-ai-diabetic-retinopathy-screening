from __future__ import annotations

import asyncio
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.api.main import _validate_image, create_app


def encoded_image(
    image_format: str = "JPEG",
    size: tuple[int, int] = (48, 48),
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (80, 110, 90)).save(buffer, format=image_format)
    return buffer.getvalue()


class ApiImageValidationTests(unittest.TestCase):
    def test_accepts_supported_image_and_reports_dimensions(self) -> None:
        image_format, dimensions = _validate_image(encoded_image())

        self.assertEqual(image_format, "JPEG")
        self.assertEqual(dimensions, (48, 48))

    def test_rejects_unreadable_image_content(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            _validate_image(b"not an image")

        self.assertEqual(raised.exception.status_code, 422)

    def test_rejects_unsupported_image_format(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            _validate_image(encoded_image("GIF"))

        self.assertEqual(raised.exception.status_code, 415)

    def test_rejects_truncated_image_content(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            _validate_image(b"\xff\xd8\xff\xe0truncated-jpeg")

        self.assertEqual(raised.exception.status_code, 422)

    def test_rejects_dimensions_below_minimum(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            _validate_image(encoded_image(size=(31, 48)))

        self.assertEqual(raised.exception.status_code, 413)


class ApiReadinessTests(unittest.TestCase):
    def test_health_and_analysis_report_missing_model(self) -> None:
        async def exercise_api(checkpoint: Path) -> tuple[dict[str, object], int, str]:
            application = create_app(checkpoint_path=checkpoint)
            async with application.router.lifespan_context(application):
                transport = ASGITransport(app=application)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    health_response = await client.get("/api/health")
                    analysis_response = await client.post(
                        "/api/analyze",
                        files={
                            "image": ("fundus.jpg", encoded_image(), "image/jpeg")
                        },
                    )
                    return (
                        health_response.json(),
                        analysis_response.status_code,
                        analysis_response.json()["detail"],
                    )

        with tempfile.TemporaryDirectory() as directory:
            health, status_code, detail = asyncio.run(
                exercise_api(Path(directory) / "missing.pt")
            )

        self.assertFalse(health["model_ready"])
        self.assertEqual(health["status"], "model_unavailable")
        self.assertEqual(status_code, 503)
        self.assertIn("checkpoint", detail)

    def test_health_reports_unloadable_model_checkpoint(self) -> None:
        async def check_health(checkpoint: Path) -> dict[str, object]:
            application = create_app(checkpoint_path=checkpoint)
            async with application.router.lifespan_context(application):
                await application.state.model_loading_task
                transport = ASGITransport(app=application)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.get("/api/health")
                    return response.json()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid.pt"
            checkpoint.write_bytes(b"not a model checkpoint")
            with patch(
                "src.api.main.load_model_checkpoint",
                side_effect=ValueError("invalid checkpoint content"),
            ):
                health = asyncio.run(check_health(checkpoint))

        self.assertFalse(health["model_ready"])
        self.assertEqual(health["status"], "model_unavailable")
        self.assertIn("invalid checkpoint content", health["model_error"])

    def test_health_is_available_while_checkpoint_loads(self) -> None:
        async def exercise_api(checkpoint: Path) -> tuple[dict, dict]:
            application = create_app(checkpoint_path=checkpoint)
            release_loader = threading.Event()

            def slow_loader(_checkpoint: Path) -> tuple[object, object, str]:
                release_loader.wait(timeout=5)
                return object(), object(), "cpu"

            with patch(
                "src.api.main.load_model_checkpoint",
                side_effect=slow_loader,
            ):
                async with application.router.lifespan_context(application):
                    transport = ASGITransport(app=application)
                    async with AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                    ) as client:
                        loading_response = await client.get("/api/health")
                        release_loader.set()
                        await application.state.model_loading_task
                        ready_response = await client.get("/api/health")
                        return loading_response.json(), ready_response.json()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "available.pt"
            checkpoint.touch()
            loading, ready = asyncio.run(exercise_api(checkpoint))

        self.assertEqual(loading["status"], "loading")
        self.assertTrue(loading["model_loading"])
        self.assertFalse(loading["model_ready"])
        self.assertEqual(ready["status"], "ready")
        self.assertTrue(ready["model_ready"])

    def test_rejects_unsupported_upload_before_analysis(self) -> None:
        async def upload_unsupported(checkpoint: Path) -> tuple[int, int]:
            application = create_app(checkpoint_path=checkpoint)
            async with application.router.lifespan_context(application):
                application.state.model = object()
                transport = ASGITransport(app=application)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    unsupported = await client.post(
                        "/api/analyze",
                        files={
                            "image": ("fundus.gif", encoded_image("GIF"), "image/gif")
                        },
                    )
                    mismatch = await client.post(
                        "/api/analyze",
                        files={
                            "image": ("fundus.png", encoded_image(), "image/png")
                        },
                    )
                    return unsupported.status_code, mismatch.status_code

        with tempfile.TemporaryDirectory() as directory:
            unsupported_status, mismatch_status = asyncio.run(
                upload_unsupported(Path(directory) / "missing.pt")
            )

        self.assertEqual(unsupported_status, 415)
        self.assertEqual(mismatch_status, 415)

    def test_rejects_corrupt_and_empty_uploads(self) -> None:
        async def upload_invalid(checkpoint: Path) -> tuple[int, int]:
            application = create_app(checkpoint_path=checkpoint)
            async with application.router.lifespan_context(application):
                application.state.model = object()
                transport = ASGITransport(app=application)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    corrupt = await client.post(
                        "/api/analyze",
                        files={"image": ("fundus.jpg", b"\xff\xd8truncated", "image/jpeg")},
                    )
                    empty = await client.post(
                        "/api/analyze",
                        files={"image": ("fundus.jpg", b"", "image/jpeg")},
                    )
                    return corrupt.status_code, empty.status_code

        with tempfile.TemporaryDirectory() as directory:
            corrupt_status, empty_status = asyncio.run(
                upload_invalid(Path(directory) / "missing.pt")
            )

        self.assertEqual(corrupt_status, 422)
        self.assertEqual(empty_status, 422)

    def test_analysis_pipeline_returns_result_and_artifacts(self) -> None:
        async def exercise_api(checkpoint: Path) -> dict:
            application = create_app(checkpoint_path=checkpoint)
            async with application.router.lifespan_context(application):
                application.state.model = object()
                application.state.config = SimpleNamespace(preprocessing=None)
                application.state.device = "cpu"
                transport = ASGITransport(app=application)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/api/analyze",
                        files={
                            "image": (
                                "fundus.jpg",
                                encoded_image(),
                                "image/jpeg",
                            )
                        },
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    return response.json()

        def report_writer(result: dict, output_dir: Path) -> dict[str, Path]:
            output_dir.mkdir(parents=True, exist_ok=True)
            paths = {}
            for name in ("pdf", "markdown", "json"):
                path = output_dir / f"report.{name}"
                path.write_bytes(f"{name} content".encode())
                paths[name] = path
            return paths

        result = {
            "image": {"width": 48, "height": 48},
            "gradcam": {"status": "unavailable", "message": "not configured"},
            "retinal_lesion_evidence": {
                "status": "unavailable",
                "message": "not configured",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("src.api.main.create_screening_report", return_value=result),
                patch(
                    "src.api.main.preprocess_image",
                    return_value=SimpleNamespace(
                        enhanced_image=Image.new("RGB", (48, 48))
                    ),
                ),
                patch("src.api.main.write_screening_report", side_effect=report_writer),
            ):
                payload = asyncio.run(
                    exercise_api(Path(directory) / "missing.pt")
                )

        self.assertEqual(payload["result"]["image"]["format"], "JPEG")
        self.assertEqual(payload["upload"]["width"], 48)
        self.assertIn("preprocessed_image", payload["artifacts"])
        self.assertIn("screening_report_pdf", payload["artifacts"])

    def test_analysis_failure_returns_logged_sanitized_error(self) -> None:
        async def exercise_api(checkpoint: Path) -> tuple[int, dict]:
            application = create_app(checkpoint_path=checkpoint)
            async with application.router.lifespan_context(application):
                application.state.model = object()
                application.state.config = SimpleNamespace(preprocessing=None)
                application.state.device = "cpu"
                transport = ASGITransport(app=application)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/api/analyze",
                        files={"image": ("fundus.jpg", encoded_image(), "image/jpeg")},
                    )
                    return response.status_code, response.json()

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "src.api.main.create_screening_report",
                side_effect=RuntimeError("internal details"),
            ):
                status_code, body = asyncio.run(
                    exercise_api(Path(directory) / "missing.pt")
                )

        self.assertEqual(status_code, 500)
        self.assertNotIn("internal details", body["detail"])
        self.assertIn("service logs", body["detail"])


if __name__ == "__main__":
    unittest.main()

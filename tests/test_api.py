from __future__ import annotations

import asyncio
import io
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.api.main import _validate_image, create_app


def encoded_image(image_format: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 48), (80, 110, 90)).save(buffer, format=image_format)
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


if __name__ == "__main__":
    unittest.main()

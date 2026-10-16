from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from src.api.main import create_app


class ScreeningHistoryApiTests(unittest.TestCase):
    def test_records_screening_results_and_returns_patient_history(self) -> None:
        async def exercise_api(directory: Path) -> tuple[dict, dict, int]:
            application = create_app(
                checkpoint_path=directory / "missing.pt",
                history_db_path=directory / "history.sqlite3",
            )
            async with application.router.lifespan_context(application):
                transport = ASGITransport(app=application)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    first_response = await client.post(
                        "/api/screening-history",
                        json={
                            "anonymous_patient_id": "ANON-42",
                            "dr_level": 2,
                            "referable": True,
                            "confidence": 0.84,
                            "image_quality_score": 78,
                        },
                    )
                    second_response = await client.post(
                        "/api/screening-history",
                        json={
                            "anonymous_patient_id": "ANON-42",
                            "dr_level": 1,
                            "referable": False,
                            "confidence": 0.91,
                            "image_quality_score": 88,
                            "reviewer_decision": "agrees",
                        },
                    )
                    history_response = await client.get(
                        "/api/screening-history/ANON-42"
                    )
                    return (
                        first_response.json(),
                        history_response.json(),
                        second_response.status_code,
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            first, history, second_status = asyncio.run(
                exercise_api(Path(temporary_directory))
            )

        self.assertEqual(first["anonymous_patient_id"], "ANON-42")
        self.assertIsNone(first["reviewer_decision"])
        self.assertEqual(second_status, 201)
        self.assertEqual(history["anonymous_patient_id"], "ANON-42")
        self.assertEqual(len(history["records"]), 2)
        self.assertEqual(history["records"][0]["dr_level"], 1)
        self.assertEqual(history["records"][0]["reviewer_decision"], "agrees")
        self.assertEqual(history["records"][1]["dr_level"], 2)
        self.assertTrue(history["records"][1]["referable"])
        self.assertEqual(history["records"][1]["confidence"], 0.84)
        self.assertEqual(history["records"][1]["image_quality_score"], 78)
        self.assertTrue(history["records"][0]["screened_at"].endswith("+00:00"))

    def test_rejects_identifying_id_and_out_of_range_screening_values(self) -> None:
        async def exercise_api(directory: Path) -> tuple[int, int]:
            application = create_app(
                checkpoint_path=directory / "missing.pt",
                history_db_path=directory / "history.sqlite3",
            )
            async with application.router.lifespan_context(application):
                transport = ASGITransport(app=application)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    identifying_id = await client.post(
                        "/api/screening-history",
                        json={
                            "anonymous_patient_id": "Jane Doe",
                            "dr_level": 1,
                            "referable": False,
                            "confidence": 0.9,
                            "image_quality_score": 90,
                        },
                    )
                    invalid_level = await client.post(
                        "/api/screening-history",
                        json={
                            "anonymous_patient_id": "ANON-42",
                            "dr_level": 5,
                            "referable": True,
                            "confidence": 0.9,
                            "image_quality_score": 90,
                        },
                    )
                    return identifying_id.status_code, invalid_level.status_code

        with tempfile.TemporaryDirectory() as temporary_directory:
            identifying_status, level_status = asyncio.run(
                exercise_api(Path(temporary_directory))
            )

        self.assertEqual(identifying_status, 422)
        self.assertEqual(level_status, 422)


if __name__ == "__main__":
    unittest.main()

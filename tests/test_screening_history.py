from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from src.api.main import create_app


class ScreeningHistoryApiTests(unittest.TestCase):
    def test_records_screening_results_and_returns_patient_history(self) -> None:
        async def exercise_api(directory: Path) -> tuple[dict, dict, int, dict]:
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
                        },
                    )
                    review_response = await client.put(
                        f"/api/screening-history/{second_response.json()['id']}/review",
                        json={"reviewer_decision": "ai_prediction_accepted"},
                    )
                    history_response = await client.get(
                        "/api/screening-history/ANON-42"
                    )
                    return (
                        first_response.json(),
                        history_response.json(),
                        second_response.status_code,
                        review_response.json(),
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            first, history, second_status, reviewed = asyncio.run(
                exercise_api(Path(temporary_directory))
            )

        self.assertEqual(first["anonymous_patient_id"], "ANON-42")
        self.assertIsNone(first["reviewer_decision"])
        self.assertIsNone(first["reviewed_at"])
        self.assertEqual(second_status, 201)
        self.assertEqual(reviewed["reviewer_decision"], "ai_prediction_accepted")
        self.assertIsNotNone(reviewed["reviewed_at"])
        self.assertEqual(history["anonymous_patient_id"], "ANON-42")
        self.assertEqual(len(history["records"]), 2)
        self.assertEqual(history["records"][0]["dr_level"], 1)
        self.assertEqual(
            history["records"][0]["reviewer_decision"],
            "ai_prediction_accepted",
        )
        self.assertIsNotNone(history["records"][0]["reviewed_at"])
        self.assertEqual(history["records"][1]["dr_level"], 2)
        self.assertTrue(history["records"][1]["referable"])
        self.assertEqual(history["records"][1]["confidence"], 0.84)
        self.assertEqual(history["records"][1]["image_quality_score"], 78)
        self.assertTrue(history["records"][0]["screened_at"].endswith("+00:00"))

    def test_rejects_identifying_id_and_out_of_range_screening_values(self) -> None:
        async def exercise_api(directory: Path) -> tuple[int, int, int, int]:
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
                    invalid_confidence = await client.post(
                        "/api/screening-history",
                        content=json.dumps({
                            "anonymous_patient_id": "ANON-42",
                            "dr_level": 1,
                            "referable": False,
                            "confidence": float("nan"),
                            "image_quality_score": 90,
                        }),
                        headers={"Content-Type": "application/json"},
                    )
                    invalid_lookup = await client.get(
                        "/api/screening-history/invalid%20case"
                    )
                    return (
                        identifying_id.status_code,
                        invalid_level.status_code,
                        invalid_confidence.status_code,
                        invalid_lookup.status_code,
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            identifying_status, level_status, confidence_status, lookup_status = asyncio.run(
                exercise_api(Path(temporary_directory))
            )

        self.assertEqual(identifying_status, 422)
        self.assertEqual(level_status, 422)
        self.assertEqual(confidence_status, 422)
        self.assertEqual(lookup_status, 422)

    def test_accepts_each_reviewer_outcome(self) -> None:
        async def exercise_api(directory: Path) -> list[dict]:
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
                    decisions = (
                        "ai_prediction_accepted",
                        "ai_prediction_modified",
                        "image_requires_manual_review",
                        "image_ungradable",
                    )
                    reviewed = []
                    for index, decision in enumerate(decisions):
                        created = await client.post(
                            "/api/screening-history",
                            json={
                                "anonymous_patient_id": f"CASE-{index}",
                                "dr_level": 1,
                                "referable": False,
                                "confidence": 0.9,
                                "image_quality_score": 90,
                            },
                        )
                        response = await client.put(
                            f"/api/screening-history/{created.json()['id']}/review",
                            json={"reviewer_decision": decision},
                        )
                        self.assertEqual(response.status_code, 200)
                        reviewed.append(response.json())
                    return reviewed

        with tempfile.TemporaryDirectory() as temporary_directory:
            reviewed_records = asyncio.run(exercise_api(Path(temporary_directory)))

        self.assertEqual(
            [record["reviewer_decision"] for record in reviewed_records],
            [
                "ai_prediction_accepted",
                "ai_prediction_modified",
                "image_requires_manual_review",
                "image_ungradable",
            ],
        )
        self.assertTrue(all(record["reviewed_at"] for record in reviewed_records))

    def test_migrates_existing_history_and_rejects_unknown_review_record(self) -> None:
        async def exercise_api(directory: Path) -> tuple[dict, int, int]:
            database = directory / "history.sqlite3"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    """
                    CREATE TABLE screening_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        anonymous_patient_id TEXT NOT NULL,
                        screened_at TEXT NOT NULL,
                        dr_level INTEGER NOT NULL,
                        referable INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        image_quality_score REAL NOT NULL,
                        reviewer_decision TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO screening_records (
                        anonymous_patient_id, screened_at, dr_level, referable,
                        confidence, image_quality_score, reviewer_decision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("ANON-OLD", "2026-01-01T00:00:00+00:00", 2, 1, 0.8, 70, "agrees"),
                )
                connection.commit()

            application = create_app(
                checkpoint_path=directory / "missing.pt",
                history_db_path=database,
            )
            async with application.router.lifespan_context(application):
                transport = ASGITransport(app=application)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    history = await client.get("/api/screening-history/ANON-OLD")
                    missing_record = await client.put(
                        "/api/screening-history/999/review",
                        json={"reviewer_decision": "image_ungradable"},
                    )
                    invalid_decision = await client.put(
                        "/api/screening-history/1/review",
                        json={"reviewer_decision": "not_a_decision"},
                    )
                    return history.json(), missing_record.status_code, invalid_decision.status_code

        with tempfile.TemporaryDirectory() as temporary_directory:
            history, missing_status, invalid_status = asyncio.run(
                exercise_api(Path(temporary_directory))
            )

        self.assertEqual(history["records"][0]["reviewer_decision"], "ai_prediction_accepted")
        self.assertIsNone(history["records"][0]["reviewed_at"])
        self.assertEqual(missing_status, 404)
        self.assertEqual(invalid_status, 422)


if __name__ == "__main__":
    unittest.main()

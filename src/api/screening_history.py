"""Local SQLite storage for minimal screening-result history."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ScreeningHistoryStore:
    """Persist screening outputs without storing images or personal information."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS screening_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anonymous_patient_id TEXT NOT NULL,
                    screened_at TEXT NOT NULL,
                    dr_level INTEGER NOT NULL CHECK (dr_level BETWEEN 0 AND 4),
                    referable INTEGER NOT NULL CHECK (referable IN (0, 1)),
                    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
                    image_quality_score REAL NOT NULL
                        CHECK (image_quality_score BETWEEN 0 AND 100),
                    reviewer_decision TEXT CHECK (
                        reviewer_decision IS NULL
                        OR reviewer_decision IN (
                            'agrees',
                            'disagrees',
                            'requires_additional_review'
                        )
                    )
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_screening_records_patient_date
                ON screening_records (anonymous_patient_id, screened_at DESC, id DESC)
                """
            )
            connection.commit()

    def create_record(
        self,
        *,
        anonymous_patient_id: str,
        dr_level: int,
        referable: bool,
        confidence: float,
        image_quality_score: float,
        reviewer_decision: str | None,
    ) -> dict[str, Any]:
        screened_at = datetime.now(timezone.utc).isoformat()
        values = (
            anonymous_patient_id,
            screened_at,
            dr_level,
            int(referable),
            confidence,
            image_quality_score,
            reviewer_decision,
        )
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO screening_records (
                        anonymous_patient_id,
                        screened_at,
                        dr_level,
                        referable,
                        confidence,
                        image_quality_score,
                        reviewer_decision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                row = connection.execute(
                    "SELECT * FROM screening_records WHERE id = ?",
                    (cursor.lastrowid,),
                ).fetchone()
        return self._serialize(row)

    def get_patient_records(self, anonymous_patient_id: str) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT * FROM screening_records
                WHERE anonymous_patient_id = ?
                ORDER BY screened_at DESC, id DESC
                """,
                (anonymous_patient_id,),
            ).fetchall()
        return [self._serialize(row) for row in rows]

    @staticmethod
    def _serialize(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "anonymous_patient_id": row["anonymous_patient_id"],
            "screened_at": row["screened_at"],
            "dr_level": row["dr_level"],
            "referable": bool(row["referable"]),
            "confidence": row["confidence"],
            "image_quality_score": row["image_quality_score"],
            "reviewer_decision": row["reviewer_decision"],
        }

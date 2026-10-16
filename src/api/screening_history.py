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
            existing = connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type = 'table' AND name = 'screening_records'
                """
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    CREATE TABLE screening_records (
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
                            'ai_prediction_accepted',
                            'ai_prediction_modified',
                            'image_requires_manual_review',
                            'image_ungradable'
                        )
                    ),
                    reviewed_at TEXT
                    )
                    """
                )
            elif "reviewed_at" not in existing[0]:
                connection.execute(
                    "DROP INDEX IF EXISTS idx_screening_records_patient_date"
                )
                connection.execute(
                    """
                    CREATE TABLE screening_records_new (
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
                                'ai_prediction_accepted',
                                'ai_prediction_modified',
                                'image_requires_manual_review',
                                'image_ungradable'
                            )
                        ),
                        reviewed_at TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO screening_records_new (
                        id, anonymous_patient_id, screened_at, dr_level, referable,
                        confidence, image_quality_score, reviewer_decision
                    )
                    SELECT
                        id, anonymous_patient_id, screened_at, dr_level, referable,
                        confidence, image_quality_score,
                        CASE reviewer_decision
                            WHEN 'agrees' THEN 'ai_prediction_accepted'
                            WHEN 'disagrees' THEN 'ai_prediction_modified'
                            WHEN 'requires_additional_review'
                                THEN 'image_requires_manual_review'
                            ELSE reviewer_decision
                        END
                    FROM screening_records
                    """
                )
                connection.execute("DROP TABLE screening_records")
                connection.execute(
                    "ALTER TABLE screening_records_new RENAME TO screening_records"
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
    ) -> dict[str, Any]:
        screened_at = datetime.now(timezone.utc).isoformat()
        values = (
            anonymous_patient_id,
            screened_at,
            dr_level,
            int(referable),
            confidence,
            image_quality_score,
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
                        image_quality_score
                    ) VALUES (?, ?, ?, ?, ?, ?)
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

    def record_reviewer_decision(
        self,
        record_id: int,
        reviewer_decision: str,
    ) -> dict[str, Any] | None:
        reviewed_at = datetime.now(timezone.utc).isoformat()
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE screening_records
                    SET reviewer_decision = ?, reviewed_at = ?
                    WHERE id = ?
                    """,
                    (reviewer_decision, reviewed_at, record_id),
                )
                if cursor.rowcount == 0:
                    return None
                row = connection.execute(
                    "SELECT * FROM screening_records WHERE id = ?",
                    (record_id,),
                ).fetchone()
        return self._serialize(row)

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
            "reviewed_at": row["reviewed_at"],
        }

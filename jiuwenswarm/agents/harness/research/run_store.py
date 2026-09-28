from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from jiuwenswarm.agents.harness.research.provenance.store import ScientificProvenanceStore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class StageRecord:
    stage: str
    status: str
    payload: dict[str, Any]
    updated_at: str


class ResearchRunStore:
    """Persist workflow state and provenance into the requested MVP output files."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.root_dir / "research.db"
        self.audit_path = self.root_dir / "audit.jsonl"
        self.provenance = ScientificProvenanceStore(
            self.root_dir,
            sqlite_filename=self.sqlite_path.name,
            audit_filename=self.audit_path.name,
        )
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_stage (
                    stage TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def set_metadata(self, key: str, value: Any) -> None:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_metadata (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (key, json.dumps(value, ensure_ascii=False, sort_keys=True), now),
            )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM run_metadata WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return default
        return json.loads(row["value_json"])

    def set_stage(self, stage: str, *, status: str, payload: dict[str, Any]) -> StageRecord:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_stage (stage, status, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(stage) DO UPDATE SET
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (stage, status, json.dumps(payload, ensure_ascii=False, sort_keys=True), now),
            )
        self.append_audit_event(
            "workflow_stage",
            {"stage": stage, "status": status, "payload": payload},
        )
        return StageRecord(stage=stage, status=status, payload=payload, updated_at=now)

    def get_stage(self, stage: str) -> StageRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT stage, status, payload_json, updated_at FROM run_stage WHERE stage = ?",
                (stage,),
            ).fetchone()
        if row is None:
            return None
        return StageRecord(
            stage=row["stage"],
            status=row["status"],
            payload=json.loads(row["payload_json"]),
            updated_at=row["updated_at"],
        )

    def append_audit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "timestamp": utc_now_iso(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


__all__ = ["ResearchRunStore", "StageRecord", "utc_now_iso"]

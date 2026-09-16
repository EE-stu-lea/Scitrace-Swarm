from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Iterator

from .graph import ScientificEvidenceGraph
from .models import EvidenceEdge, NodeKind, ProvenanceNode, edge_from_dict, node_from_dict


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class StorePaths:
    sqlite_path: Path
    audit_jsonl_path: Path


class ScientificProvenanceStore:
    """SQLite-backed provenance store with append-only JSONL audit output."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        sqlite_filename: str = "scientific_provenance.sqlite3",
        audit_filename: str = "scientific_provenance.audit.jsonl",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.paths = StorePaths(
            sqlite_path=self.root_dir / sqlite_filename,
            audit_jsonl_path=self.root_dir / audit_filename,
        )
        self._init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.paths.sqlite_path)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provenance_nodes (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provenance_edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def upsert_node(self, node: ProvenanceNode) -> ProvenanceNode:
        payload_json = json.dumps(node.to_dict(), ensure_ascii=False, sort_keys=True)
        now = _utc_timestamp()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO provenance_nodes (id, kind, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind=excluded.kind,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (node.id, node.kind.value, payload_json, now, now),
            )
        self.append_audit_event("upsert_node", node.to_dict())
        return node

    def upsert_edge(self, edge: EvidenceEdge) -> EvidenceEdge:
        payload_json = json.dumps(edge.to_dict(), ensure_ascii=False, sort_keys=True)
        now = _utc_timestamp()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO provenance_edges (id, source_id, target_id, relation, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_id=excluded.source_id,
                    target_id=excluded.target_id,
                    relation=excluded.relation,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    edge.id,
                    edge.source_id,
                    edge.target_id,
                    edge.relation.value,
                    payload_json,
                    now,
                    now,
                ),
            )
        self.append_audit_event("upsert_edge", edge.to_dict())
        return edge

    def get_node(self, node_id: str) -> ProvenanceNode | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM provenance_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
        if row is None:
            return None
        return node_from_dict(json.loads(row[0]))

    def list_nodes(self, *, kind: NodeKind | None = None) -> list[ProvenanceNode]:
        query = "SELECT payload_json FROM provenance_nodes"
        params: tuple[object, ...] = ()
        if kind is not None:
            query += " WHERE kind = ?"
            params = (kind.value,)
        query += " ORDER BY id"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [node_from_dict(json.loads(row[0])) for row in rows]

    def list_edges(self) -> list[EvidenceEdge]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM provenance_edges ORDER BY id"
            ).fetchall()
        return [edge_from_dict(json.loads(row[0])) for row in rows]

    def load_graph(self) -> ScientificEvidenceGraph:
        graph = ScientificEvidenceGraph()
        for node in self.list_nodes():
            graph.add_node(node)
        for edge in self.list_edges():
            graph.add_edge(edge)
        return graph

    def append_audit_event(self, event_type: str, payload: dict[str, object]) -> None:
        event = {
            "timestamp": _utc_timestamp(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.paths.audit_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

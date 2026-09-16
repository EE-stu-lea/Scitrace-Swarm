from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import time
from typing import Any

from jiuwenswarm.agents.harness.research.provenance.graph import ScientificEvidenceGraph
from jiuwenswarm.agents.harness.research.provenance.models import (
    Claim,
    EvidenceEdge,
    ExperimentArtifact,
    LiteratureEvidence,
    NodeKind,
    ProvenanceNode,
    edge_from_dict,
    node_from_dict,
)
from jiuwenswarm.agents.harness.research.provenance.store import ScientificProvenanceStore
from jiuwenswarm.agents.harness.research.rails._compat import (
    AgentCallbackContext,
    DeepAgentRail,
)
from jiuwenswarm.common.utils import get_agent_workspace_dir

_TOOL_START_KEY = "__scientific_provenance_tool_started_at__"


class ScientificProvenanceRail(DeepAgentRail):
    """Capture research provenance and expose explicit recording helpers."""

    priority = 58

    def __init__(
        self,
        *,
        enabled: bool = False,
        output_root: str | Path | None = None,
        capture_tool_audits: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.output_root = Path(output_root) if output_root else get_agent_workspace_dir() / "research"
        self.capture_tool_audits = bool(capture_tool_audits)
        self._stores: dict[str, ScientificProvenanceStore] = {}

    def _session_id(self, ctx: AgentCallbackContext | None = None) -> str:
        if ctx is not None:
            session = getattr(ctx, "session", None)
            get_session_id = getattr(session, "get_session_id", None)
            if callable(get_session_id):
                value = get_session_id()
                if value:
                    return str(value)
            inputs = getattr(ctx, "inputs", None)
            conversation_id = (
                inputs.get("conversation_id")
                if isinstance(inputs, dict)
                else getattr(inputs, "conversation_id", None)
            )
            if conversation_id:
                return str(conversation_id)
            extra = getattr(ctx, "extra", None)
            if isinstance(extra, dict) and extra.get("session_id"):
                return str(extra["session_id"])
        return "default"

    def _store(self, session_id: str) -> ScientificProvenanceStore:
        store = self._stores.get(session_id)
        if store is None:
            store = ScientificProvenanceStore(self.output_root / session_id)
            self._stores[session_id] = store
        return store

    def graph(self, session_id: str = "default") -> ScientificEvidenceGraph:
        return self._store(session_id).load_graph()

    def record_node(self, node: ProvenanceNode, *, session_id: str = "default") -> ProvenanceNode:
        self._store(session_id).upsert_node(node)
        return node

    def record_edge(self, edge: EvidenceEdge, *, session_id: str = "default") -> EvidenceEdge:
        self._store(session_id).upsert_edge(edge)
        return edge

    def record_literature_evidence(self, *, session_id: str = "default", **kwargs: Any) -> LiteratureEvidence:
        node = LiteratureEvidence(**kwargs)
        self.record_node(node, session_id=session_id)
        return node

    def record_artifact(self, *, session_id: str = "default", **kwargs: Any) -> ExperimentArtifact:
        node = ExperimentArtifact(**kwargs)
        self.record_node(node, session_id=session_id)
        return node

    def record_claim(self, *, session_id: str = "default", **kwargs: Any) -> Claim:
        node = Claim(**kwargs)
        self.record_node(node, session_id=session_id)
        return node

    def ingest_payload(self, payload: dict[str, Any], *, session_id: str = "default") -> dict[str, list[str]]:
        store = self._store(session_id)
        recorded_nodes: list[str] = []
        recorded_edges: list[str] = []
        for raw_node in self._iter_items(payload.get("nodes")):
            node = raw_node if isinstance(raw_node, ProvenanceNode) else node_from_dict(raw_node)
            store.upsert_node(node)
            recorded_nodes.append(node.id)
        for raw_edge in self._iter_items(payload.get("edges")):
            edge = raw_edge if isinstance(raw_edge, EvidenceEdge) else edge_from_dict(raw_edge)
            store.upsert_edge(edge)
            recorded_edges.append(edge.id)
        if payload.get("audit"):
            store.append_audit_event("scientific_payload", dict(payload.get("audit") or {}))
        return {"node_ids": recorded_nodes, "edge_ids": recorded_edges}

    @staticmethod
    def _iter_items(value: Any) -> Iterable[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    async def before_tool_call(self, ctx: AgentCallbackContext) -> None:
        if not self.enabled or not self.capture_tool_audits:
            return
        extra = getattr(ctx, "extra", None)
        if isinstance(extra, dict):
            extra[_TOOL_START_KEY] = time.perf_counter()

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        if not self.enabled:
            return
        session_id = self._session_id(ctx)
        store = self._store(session_id)
        inputs = getattr(ctx, "inputs", None)
        tool_name = getattr(inputs, "tool_name", "") if inputs is not None else ""
        tool_args = getattr(inputs, "tool_args", {}) if inputs is not None else {}
        tool_result = getattr(inputs, "tool_result", None) if inputs is not None else None
        payload = self._extract_scientific_payload(tool_args, tool_result)
        if payload:
            self.ingest_payload(payload, session_id=session_id)
        if self.capture_tool_audits:
            extra = getattr(ctx, "extra", None)
            started_at = extra.get(_TOOL_START_KEY) if isinstance(extra, dict) else None
            duration_seconds = 0.0
            if isinstance(started_at, (int, float)):
                duration_seconds = max(0.0, time.perf_counter() - started_at)
            store.append_audit_event(
                "tool_audit",
                {
                    "tool_name": tool_name,
                    "tool_args": tool_args if isinstance(tool_args, dict) else {},
                    "duration_seconds": duration_seconds,
                    "result_type": type(tool_result).__name__,
                },
            )

    async def on_tool_exception(self, ctx: AgentCallbackContext) -> None:
        if not self.enabled:
            return
        session_id = self._session_id(ctx)
        store = self._store(session_id)
        inputs = getattr(ctx, "inputs", None)
        store.append_audit_event(
            "tool_exception",
            {
                "tool_name": getattr(inputs, "tool_name", "") if inputs is not None else "",
                "tool_args": getattr(inputs, "tool_args", {}) if inputs is not None else {},
                "exception": str(getattr(ctx, "exception", "") or ""),
            },
        )

    @staticmethod
    def _extract_scientific_payload(tool_args: Any, tool_result: Any) -> dict[str, Any] | None:
        for candidate in (tool_result, tool_args):
            if isinstance(candidate, dict):
                payload = candidate.get("scientific_provenance")
                if isinstance(payload, dict):
                    return payload
        return None

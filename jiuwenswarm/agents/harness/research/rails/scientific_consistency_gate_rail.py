from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from jiuwenswarm.agents.harness.research.provenance.models import (
    Claim,
    ClaimStatus,
    ExperimentArtifact,
    LiteratureEvidence,
)
from jiuwenswarm.agents.harness.research.provenance.store import ScientificProvenanceStore
from jiuwenswarm.agents.harness.research.rails._compat import (
    AgentCallbackContext,
    DeepAgentRail,
)
from jiuwenswarm.agents.harness.research.provenance.validator import ScientificValidator
from jiuwenswarm.common.utils import get_agent_workspace_dir

_POSITIVE_CONCLUSION_PATTERN = re.compile(
    r"\b("
    r"support(?:s|ed)?|"
    r"improv(?:e|es|ed|ement)|"
    r"outperform(?:s|ed)?|"
    r"significant(?:ly)?|"
    r"effective|"
    r"state-of-the-art|"
    r"achieve(?:s|d)?"
    r")\b",
    re.IGNORECASE,
)


class ScientificConsistencyGateRail(DeepAgentRail):
    """Validate citations, claims, and experiment metrics before final output."""

    priority = 60

    def __init__(
        self,
        *,
        enabled: bool = False,
        output_root: str | Path | None = None,
        block_on_error: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.output_root = Path(output_root) if output_root else get_agent_workspace_dir() / "research"
        self.block_on_error = bool(block_on_error)
        self.validator = ScientificValidator()
        self._stores: dict[str, ScientificProvenanceStore] = {}

    def _store(self, session_id: str) -> ScientificProvenanceStore:
        store = self._stores.get(session_id)
        if store is None:
            store = ScientificProvenanceStore(self.output_root / session_id)
            self._stores[session_id] = store
        return store

    def check_payload(self, payload: dict[str, Any], *, session_id: str = "default") -> dict[str, Any]:
        graph = self._store(session_id).load_graph()
        errors: list[dict[str, Any]] = [
            error.__dict__
            for error in self.validator.validate_graph(graph)
        ]

        citation_ids = self._citation_ids(payload)
        for citation_id in citation_ids:
            node = graph.get_node(citation_id)
            if not isinstance(node, LiteratureEvidence):
                errors.append(
                    {
                        "code": "citation_missing",
                        "message": f"Citation `{citation_id}` does not resolve to a LiteratureEvidence node.",
                        "node_id": citation_id,
                        "severity": "error",
                    }
                )

        claim_ids = self._claim_ids(payload)
        draft_text = str(payload.get("draft_text") or "")
        for claim_id in claim_ids:
            node = graph.get_node(claim_id)
            if not isinstance(node, Claim):
                errors.append(
                    {
                        "code": "claim_missing",
                        "message": f"Claim `{claim_id}` does not exist in the evidence graph.",
                        "node_id": claim_id,
                        "severity": "error",
                    }
                )
                continue

            for validation_error in self.validator.validate_claim(node, graph):
                errors.append(validation_error.__dict__)

            if node.status in {ClaimStatus.REFUTED, ClaimStatus.INCONCLUSIVE} and self._promotes_claim(
                node, draft_text
            ):
                errors.append(
                    {
                        "code": "refuted_claim_promoted",
                        "message": (
                            f"Claim `{claim_id}` is `{node.status.value}` in provenance but the draft frames it as supported."
                        ),
                        "node_id": claim_id,
                        "severity": "error",
                    }
                )

        for item in self._metric_checks(payload):
            artifact_id = item.get("artifact_id", "")
            node = graph.get_node(str(artifact_id))
            if not isinstance(node, ExperimentArtifact):
                errors.append(
                    {
                        "code": "artifact_missing",
                        "message": f"Artifact `{artifact_id}` does not exist in the evidence graph.",
                        "node_id": artifact_id,
                        "severity": "error",
                    }
                )
                continue

            actual_value = node.metric_value
            claimed_value = item.get("claimed_value", item.get("value"))
            if not self._values_match(actual_value, claimed_value):
                errors.append(
                    {
                        "code": "metric_mismatch",
                        "message": (
                            f"Artifact `{artifact_id}` records `{actual_value}` but the draft payload claims `{claimed_value}`."
                        ),
                        "node_id": artifact_id,
                        "severity": "error",
                    }
                )

        return {
            "ok": not errors,
            "session_id": session_id,
            "error_count": len(errors),
            "errors": errors,
        }

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        if not self.enabled:
            return
        session_id = self._session_id(ctx)
        inputs = getattr(ctx, "inputs", None)
        tool_args = getattr(inputs, "tool_args", {}) if inputs is not None else {}
        tool_result = getattr(inputs, "tool_result", None) if inputs is not None else None
        payload = self._extract_payload(tool_args, tool_result)
        if payload is None:
            return

        result = self.check_payload(payload, session_id=session_id)
        self._store(session_id).append_audit_event("consistency_gate", result)

        extra = getattr(ctx, "extra", None)
        if isinstance(extra, dict):
            extra["scientific_consistency_gate"] = result

        if result["ok"] or not self.block_on_error:
            return

        blocked_result = {
            "success": False,
            "blocked": True,
            "gate": "scientific_consistency_gate",
            "errors": result["errors"],
        }
        if isinstance(inputs, dict):
            inputs["tool_result"] = blocked_result
        elif inputs is not None:
            setattr(inputs, "tool_result", blocked_result)

    @staticmethod
    def _extract_payload(tool_args: Any, tool_result: Any) -> dict[str, Any] | None:
        for candidate in (tool_result, tool_args):
            if isinstance(candidate, dict):
                payload = candidate.get("scientific_consistency_gate")
                if isinstance(payload, dict):
                    return payload
        return None

    @staticmethod
    def _citation_ids(payload: dict[str, Any]) -> list[str]:
        citations = payload.get("citations")
        if isinstance(citations, list):
            return [str(item).strip() for item in citations if str(item).strip()]
        if isinstance(citations, dict):
            return [str(value).strip() for value in citations.values() if str(value).strip()]
        return []

    @staticmethod
    def _claim_ids(payload: dict[str, Any]) -> list[str]:
        claims = payload.get("claims")
        claim_ids: list[str] = []
        if isinstance(claims, list):
            for item in claims:
                if isinstance(item, str) and item.strip():
                    claim_ids.append(item.strip())
                elif isinstance(item, dict) and str(item.get("claim_id") or "").strip():
                    claim_ids.append(str(item.get("claim_id")).strip())
        return claim_ids

    @staticmethod
    def _metric_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
        reported = payload.get("reported_metrics")
        if isinstance(reported, list):
            return [dict(item) for item in reported if isinstance(item, dict)]
        return []

    @staticmethod
    def _promotes_claim(claim: Claim, draft_text: str) -> bool:
        if not draft_text or not _POSITIVE_CONCLUSION_PATTERN.search(draft_text):
            return False
        text = claim.text.strip().lower().rstrip(".!?:;")
        lowered_draft = draft_text.lower()
        return not text or text in lowered_draft

    @staticmethod
    def _values_match(actual: Any, claimed: Any) -> bool:
        if actual == claimed:
            return True
        try:
            return abs(float(actual) - float(claimed)) <= 1e-9
        except (TypeError, ValueError):
            return str(actual).strip() == str(claimed).strip()

    @staticmethod
    def _session_id(ctx: AgentCallbackContext | None = None) -> str:
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
        return "default"


__all__ = ["ScientificConsistencyGateRail"]

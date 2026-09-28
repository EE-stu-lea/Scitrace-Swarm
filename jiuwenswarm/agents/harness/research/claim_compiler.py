from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from jiuwenswarm.agents.harness.research.provenance.models import (
    Claim,
    ClaimStatus,
    ExperimentArtifact,
    LiteratureEvidence,
)


class ClaimCompilerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClaimCompilerInput:
    topic: str
    evidence: list[LiteratureEvidence]
    artifacts: list[ExperimentArtifact]
    section: str = "results"
    status_threshold_supported: float = 0.75
    status_threshold_refuted: float = 0.45


def infer_claim_status(
    artifacts: list[ExperimentArtifact],
    *,
    supported_threshold: float = 0.75,
    refuted_threshold: float = 0.45,
) -> ClaimStatus:
    status_hints = {
        str(item.metadata.get("claim_status") or "").strip().lower()
        for item in artifacts
        if isinstance(item.metadata, dict)
    }
    status_hints.discard("")
    if len(status_hints) == 1:
        hinted = next(iter(status_hints))
        if hinted in {ClaimStatus.SUPPORTED.value, ClaimStatus.REFUTED.value, ClaimStatus.INCONCLUSIVE.value}:
            return ClaimStatus(hinted)
    numeric_values = [float(item.metric_value) for item in artifacts if isinstance(item.metric_value, (int, float))]
    if not numeric_values:
        return ClaimStatus.INCONCLUSIVE
    best = max(numeric_values)
    if best >= supported_threshold:
        return ClaimStatus.SUPPORTED
    if best <= refuted_threshold:
        return ClaimStatus.REFUTED
    return ClaimStatus.INCONCLUSIVE


def sanitize_claim_payload(
    payload: dict[str, Any],
    *,
    compiler_input: ClaimCompilerInput,
) -> Claim:
    evidence_ids = [item.id for item in compiler_input.evidence]
    artifact_ids = [item.id for item in compiler_input.artifacts]
    text = str(payload.get("text") or "").strip()
    if not text:
        raise ClaimCompilerError("claim text is empty")
    referenced_evidence = [
        item for item in payload.get("evidence_ids", evidence_ids)
        if str(item).strip() in evidence_ids
    ]
    referenced_artifacts = [
        item for item in payload.get("artifact_ids", artifact_ids)
        if str(item).strip() in artifact_ids
    ]
    if not referenced_evidence and not referenced_artifacts:
        referenced_evidence = evidence_ids[:1]
        referenced_artifacts = artifact_ids[:1]
    status = infer_claim_status(
        compiler_input.artifacts,
        supported_threshold=compiler_input.status_threshold_supported,
        refuted_threshold=compiler_input.status_threshold_refuted,
    )
    return Claim(
        text=text,
        status=status,
        evidence_ids=referenced_evidence,
        artifact_ids=referenced_artifacts,
        confidence=_coerce_confidence(payload.get("confidence")),
        section=str(payload.get("section") or compiler_input.section).strip(),
        validation_errors=[],
        metadata={
            "compiler_source": str(payload.get("compiler_source") or "claim_compiler"),
            "writer_locked_status": status.value,
        },
    )


def conservative_claim_text(
    *,
    topic: str,
    status: ClaimStatus,
    evidence: list[LiteratureEvidence],
    artifacts: list[ExperimentArtifact],
) -> str:
    top_title = evidence[0].title if evidence else topic
    if status is ClaimStatus.SUPPORTED:
        return f"The available evidence and recorded experiment artifacts support the tested hypothesis about {topic.lower()}."
    if status is ClaimStatus.REFUTED:
        return f"The recorded experiment artifacts refute the tested hypothesis about {topic.lower()}."
    return f"The current evidence on {top_title} remains inconclusive, and the observed experiment outputs do not justify a strong claim."


def load_claim_payload(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClaimCompilerError(f"invalid claim compiler JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ClaimCompilerError("claim compiler must return a JSON object")
    return payload


def _coerce_confidence(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return max(0.0, min(1.0, parsed))
    return None


__all__ = [
    "ClaimCompilerError",
    "ClaimCompilerInput",
    "conservative_claim_text",
    "infer_claim_status",
    "load_claim_payload",
    "sanitize_claim_payload",
]

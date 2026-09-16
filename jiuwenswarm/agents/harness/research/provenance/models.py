from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            key: _normalize_for_hash(val)
            for key, val in asdict(value).items()
        }
    if isinstance(value, dict):
        return {
            str(key): _normalize_for_hash(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_hash(item) for item in value]
    return value


def canonical_payload_hash(payload: dict[str, Any]) -> str:
    normalized = _normalize_for_hash(payload)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_evidence_id(payload: dict[str, Any]) -> str:
    return f"evidence_{canonical_payload_hash(payload)[:20]}"


def stable_claim_id(payload: dict[str, Any]) -> str:
    return f"claim_{canonical_payload_hash(payload)[:20]}"


def stable_edge_id(source_id: str, target_id: str, relation: str) -> str:
    return f"edge_{canonical_payload_hash({'source_id': source_id, 'target_id': target_id, 'relation': relation})[:20]}"


class NodeKind(str, Enum):
    RESEARCH_QUESTION = "ResearchQuestion"
    HYPOTHESIS = "Hypothesis"
    LITERATURE_EVIDENCE = "LiteratureEvidence"
    EXPERIMENT_PLAN = "ExperimentPlan"
    EXPERIMENT_ARTIFACT = "ExperimentArtifact"
    CLAIM = "Claim"
    REVISION_DECISION = "RevisionDecision"


class EdgeRelation(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    TESTS = "tests"
    DERIVED_FROM = "derived_from"
    REVISES = "revises"


class ClaimStatus(str, Enum):
    PROPOSED = "proposed"
    TESTED = "tested"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class ProvenanceNode:
    id: str
    kind: NodeKind
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _normalize_for_hash(asdict(self))


@dataclass(frozen=True)
class ResearchQuestion(ProvenanceNode):
    text: str = ""

    def __init__(
        self,
        *,
        text: str,
        id: str | None = None,
        created_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {"text": text.strip()}
        resolved_id = id or f"rq_{canonical_payload_hash(payload)[:20]}"
        super().__init__(
            id=resolved_id,
            kind=NodeKind.RESEARCH_QUESTION,
            created_at=created_at or utc_now(),
            metadata=dict(metadata or {}),
        )
        object.__setattr__(self, "text", text.strip())


@dataclass(frozen=True)
class Hypothesis(ProvenanceNode):
    text: str = ""
    rank: int | None = None

    def __init__(
        self,
        *,
        text: str,
        rank: int | None = None,
        id: str | None = None,
        created_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {"text": text.strip(), "rank": rank}
        resolved_id = id or f"hyp_{canonical_payload_hash(payload)[:20]}"
        super().__init__(
            id=resolved_id,
            kind=NodeKind.HYPOTHESIS,
            created_at=created_at or utc_now(),
            metadata=dict(metadata or {}),
        )
        object.__setattr__(self, "text", text.strip())
        object.__setattr__(self, "rank", rank)


@dataclass(frozen=True)
class LiteratureEvidence(ProvenanceNode):
    title: str = ""
    authors: tuple[str, ...] = field(default_factory=tuple)
    year: int | None = None
    source_url: str = ""
    doi: str = ""
    retrieved_at: datetime = field(default_factory=utc_now)
    summary: str = ""
    source_checksum: str = ""
    reliability_score: float | None = None

    def __init__(
        self,
        *,
        title: str,
        authors: list[str] | tuple[str, ...],
        year: int | None,
        source_url: str,
        doi: str = "",
        retrieved_at: datetime | None = None,
        summary: str = "",
        source_checksum: str = "",
        reliability_score: float | None = None,
        id: str | None = None,
        created_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        normalized_authors = tuple(str(author).strip() for author in authors if str(author).strip())
        resolved_retrieved_at = retrieved_at or utc_now()
        checksum = source_checksum or sha256_text(
            f"{title.strip()}|{'|'.join(normalized_authors)}|{year}|{source_url.strip()}|{doi.strip()}"
        )
        payload = {
            "title": title.strip(),
            "authors": normalized_authors,
            "year": year,
            "source_url": source_url.strip(),
            "doi": doi.strip().lower(),
            "source_checksum": checksum,
        }
        super().__init__(
            id=id or stable_evidence_id(payload),
            kind=NodeKind.LITERATURE_EVIDENCE,
            created_at=created_at or utc_now(),
            metadata=dict(metadata or {}),
        )
        object.__setattr__(self, "title", title.strip())
        object.__setattr__(self, "authors", normalized_authors)
        object.__setattr__(self, "year", year)
        object.__setattr__(self, "source_url", source_url.strip())
        object.__setattr__(self, "doi", doi.strip())
        object.__setattr__(self, "retrieved_at", resolved_retrieved_at)
        object.__setattr__(self, "summary", summary.strip())
        object.__setattr__(self, "source_checksum", checksum)
        object.__setattr__(self, "reliability_score", reliability_score)


@dataclass(frozen=True)
class ExperimentPlan(ProvenanceNode):
    experiment_id: str = ""
    hypothesis_id: str = ""
    command: str = ""
    dataset: str = ""
    metric_names: tuple[str, ...] = field(default_factory=tuple)
    seed: int | None = None

    def __init__(
        self,
        *,
        experiment_id: str,
        hypothesis_id: str = "",
        command: str = "",
        dataset: str = "",
        metric_names: list[str] | tuple[str, ...] | None = None,
        seed: int | None = None,
        id: str | None = None,
        created_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        resolved_metrics = tuple(str(item).strip() for item in (metric_names or []) if str(item).strip())
        payload = {
            "experiment_id": experiment_id.strip(),
            "hypothesis_id": hypothesis_id.strip(),
            "command": command.strip(),
            "dataset": dataset.strip(),
            "metric_names": resolved_metrics,
            "seed": seed,
        }
        super().__init__(
            id=id or f"plan_{canonical_payload_hash(payload)[:20]}",
            kind=NodeKind.EXPERIMENT_PLAN,
            created_at=created_at or utc_now(),
            metadata=dict(metadata or {}),
        )
        object.__setattr__(self, "experiment_id", experiment_id.strip())
        object.__setattr__(self, "hypothesis_id", hypothesis_id.strip())
        object.__setattr__(self, "command", command.strip())
        object.__setattr__(self, "dataset", dataset.strip())
        object.__setattr__(self, "metric_names", resolved_metrics)
        object.__setattr__(self, "seed", seed)


@dataclass(frozen=True)
class ExperimentArtifact(ProvenanceNode):
    experiment_id: str = ""
    file_path: str = ""
    sha256: str = ""
    command: str = ""
    seed: int | None = None
    dataset: str = ""
    metric_name: str = ""
    metric_value: float | int | str | None = None

    def __init__(
        self,
        *,
        experiment_id: str,
        file_path: str,
        sha256: str,
        command: str,
        seed: int | None,
        dataset: str,
        metric_name: str,
        metric_value: float | int | str | None,
        id: str | None = None,
        created_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "experiment_id": experiment_id.strip(),
            "file_path": str(file_path).strip(),
            "sha256": sha256.strip().lower(),
            "command": command.strip(),
            "seed": seed,
            "dataset": dataset.strip(),
            "metric_name": metric_name.strip(),
            "metric_value": metric_value,
        }
        super().__init__(
            id=id or f"artifact_{canonical_payload_hash(payload)[:20]}",
            kind=NodeKind.EXPERIMENT_ARTIFACT,
            created_at=created_at or utc_now(),
            metadata=dict(metadata or {}),
        )
        object.__setattr__(self, "experiment_id", experiment_id.strip())
        object.__setattr__(self, "file_path", str(file_path).strip())
        object.__setattr__(self, "sha256", sha256.strip().lower())
        object.__setattr__(self, "command", command.strip())
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "dataset", dataset.strip())
        object.__setattr__(self, "metric_name", metric_name.strip())
        object.__setattr__(self, "metric_value", metric_value)


@dataclass(frozen=True)
class Claim(ProvenanceNode):
    text: str = ""
    status: ClaimStatus = ClaimStatus.PROPOSED
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    artifact_ids: tuple[str, ...] = field(default_factory=tuple)
    confidence: float | None = None
    section: str = ""
    validation_errors: tuple[str, ...] = field(default_factory=tuple)

    def __init__(
        self,
        *,
        text: str,
        status: ClaimStatus = ClaimStatus.PROPOSED,
        evidence_ids: list[str] | tuple[str, ...] | None = None,
        artifact_ids: list[str] | tuple[str, ...] | None = None,
        confidence: float | None = None,
        section: str = "",
        validation_errors: list[str] | tuple[str, ...] | None = None,
        id: str | None = None,
        created_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        normalized_evidence_ids = tuple(str(item).strip() for item in (evidence_ids or []) if str(item).strip())
        normalized_artifact_ids = tuple(str(item).strip() for item in (artifact_ids or []) if str(item).strip())
        normalized_errors = tuple(str(item).strip() for item in (validation_errors or []) if str(item).strip())
        payload = {
            "text": text.strip(),
            "section": section.strip(),
        }
        super().__init__(
            id=id or stable_claim_id(payload),
            kind=NodeKind.CLAIM,
            created_at=created_at or utc_now(),
            metadata=dict(metadata or {}),
        )
        object.__setattr__(self, "text", text.strip())
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence_ids", normalized_evidence_ids)
        object.__setattr__(self, "artifact_ids", normalized_artifact_ids)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "section", section.strip())
        object.__setattr__(self, "validation_errors", normalized_errors)


@dataclass(frozen=True)
class RevisionDecision(ProvenanceNode):
    claim_id: str = ""
    decision: str = ""
    rationale: str = ""

    def __init__(
        self,
        *,
        claim_id: str,
        decision: str,
        rationale: str = "",
        id: str | None = None,
        created_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "claim_id": claim_id.strip(),
            "decision": decision.strip(),
            "rationale": rationale.strip(),
        }
        super().__init__(
            id=id or f"rev_{canonical_payload_hash(payload)[:20]}",
            kind=NodeKind.REVISION_DECISION,
            created_at=created_at or utc_now(),
            metadata=dict(metadata or {}),
        )
        object.__setattr__(self, "claim_id", claim_id.strip())
        object.__setattr__(self, "decision", decision.strip())
        object.__setattr__(self, "rationale", rationale.strip())


@dataclass(frozen=True)
class EvidenceEdge:
    source_id: str
    target_id: str
    relation: EdgeRelation
    id: str = ""
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(
                self,
                "id",
                stable_edge_id(self.source_id, self.target_id, self.relation.value),
            )

    def to_dict(self) -> dict[str, Any]:
        return _normalize_for_hash(asdict(self))


_NODE_TYPE_BY_KIND: dict[NodeKind, type[ProvenanceNode]] = {
    NodeKind.RESEARCH_QUESTION: ResearchQuestion,
    NodeKind.HYPOTHESIS: Hypothesis,
    NodeKind.LITERATURE_EVIDENCE: LiteratureEvidence,
    NodeKind.EXPERIMENT_PLAN: ExperimentPlan,
    NodeKind.EXPERIMENT_ARTIFACT: ExperimentArtifact,
    NodeKind.CLAIM: Claim,
    NodeKind.REVISION_DECISION: RevisionDecision,
}


def node_from_dict(payload: dict[str, Any]) -> ProvenanceNode:
    raw = dict(payload)
    raw_kind = NodeKind(raw.pop("kind"))
    node_type = _NODE_TYPE_BY_KIND[raw_kind]
    for key in ("created_at", "retrieved_at"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            raw[key] = datetime.fromisoformat(value)
    if raw_kind is NodeKind.CLAIM and "status" in raw and not isinstance(raw["status"], ClaimStatus):
        raw["status"] = ClaimStatus(raw["status"])
    return node_type(**raw)


def edge_from_dict(payload: dict[str, Any]) -> EvidenceEdge:
    raw = dict(payload)
    if isinstance(raw.get("created_at"), str):
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
    if not isinstance(raw.get("relation"), EdgeRelation):
        raw["relation"] = EdgeRelation(raw["relation"])
    return EvidenceEdge(**raw)


def dataclass_field_names(model_type: type[Any]) -> set[str]:
    return {field_info.name for field_info in fields(model_type)}

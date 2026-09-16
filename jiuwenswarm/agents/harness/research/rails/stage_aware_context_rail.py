from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jiuwenswarm.agents.harness.research.provenance.graph import ScientificEvidenceGraph
from jiuwenswarm.agents.harness.research.provenance.models import (
    Claim,
    ClaimStatus,
    ExperimentArtifact,
    ExperimentPlan,
    Hypothesis,
    LiteratureEvidence,
    NodeKind,
    ProvenanceNode,
    ResearchQuestion,
    RevisionDecision,
)
from jiuwenswarm.agents.harness.research.provenance.store import ScientificProvenanceStore
from jiuwenswarm.agents.harness.research.rails._compat import (
    AgentCallbackContext,
    DeepAgentRail,
    PromptSection,
)
from jiuwenswarm.common.utils import get_agent_workspace_dir


_SECTION_NAME = "scientific_stage_context"
_STAGE_ALIASES: dict[str, str] = {
    "topic intake": "ideation",
    "literature research": "ideation",
    "evidence verification": "ideation",
    "gap analysis": "ideation",
    "ideation": "ideation",
    "hypothesis ranking": "planning",
    "planning": "planning",
    "experiment planning": "planning",
    "coding and execution": "experiment",
    "artifact ingestion": "experiment",
    "experiment": "experiment",
    "claim compilation": "writing",
    "paper writing": "writing",
    "targeted revision": "writing",
    "writing": "writing",
    "automated review": "reviewing",
    "consistency gate": "reviewing",
    "final audit": "reviewing",
    "reviewing": "reviewing",
}
_STAGE_KIND_WEIGHTS: dict[str, dict[NodeKind, float]] = {
    "ideation": {
        NodeKind.RESEARCH_QUESTION: 1.00,
        NodeKind.HYPOTHESIS: 1.00,
        NodeKind.LITERATURE_EVIDENCE: 0.95,
        NodeKind.EXPERIMENT_PLAN: 0.35,
        NodeKind.EXPERIMENT_ARTIFACT: 0.25,
        NodeKind.CLAIM: 0.45,
        NodeKind.REVISION_DECISION: 0.30,
    },
    "planning": {
        NodeKind.RESEARCH_QUESTION: 0.75,
        NodeKind.HYPOTHESIS: 0.95,
        NodeKind.LITERATURE_EVIDENCE: 0.70,
        NodeKind.EXPERIMENT_PLAN: 1.00,
        NodeKind.EXPERIMENT_ARTIFACT: 0.45,
        NodeKind.CLAIM: 0.50,
        NodeKind.REVISION_DECISION: 0.35,
    },
    "experiment": {
        NodeKind.RESEARCH_QUESTION: 0.40,
        NodeKind.HYPOTHESIS: 0.55,
        NodeKind.LITERATURE_EVIDENCE: 0.40,
        NodeKind.EXPERIMENT_PLAN: 0.90,
        NodeKind.EXPERIMENT_ARTIFACT: 1.00,
        NodeKind.CLAIM: 0.70,
        NodeKind.REVISION_DECISION: 0.40,
    },
    "writing": {
        NodeKind.RESEARCH_QUESTION: 0.35,
        NodeKind.HYPOTHESIS: 0.40,
        NodeKind.LITERATURE_EVIDENCE: 0.85,
        NodeKind.EXPERIMENT_PLAN: 0.55,
        NodeKind.EXPERIMENT_ARTIFACT: 0.90,
        NodeKind.CLAIM: 1.00,
        NodeKind.REVISION_DECISION: 0.50,
    },
    "reviewing": {
        NodeKind.RESEARCH_QUESTION: 0.30,
        NodeKind.HYPOTHESIS: 0.40,
        NodeKind.LITERATURE_EVIDENCE: 0.95,
        NodeKind.EXPERIMENT_PLAN: 0.55,
        NodeKind.EXPERIMENT_ARTIFACT: 0.95,
        NodeKind.CLAIM: 1.00,
        NodeKind.REVISION_DECISION: 0.95,
    },
}


@dataclass(frozen=True)
class StageContextItem:
    node_id: str
    kind: str
    score: float
    estimated_tokens: int
    summary: str


class StageAwareContextRail(DeepAgentRail):
    """Inject stage-specific scientific context with an explicit token budget."""

    priority = 56
    section_priority = 118

    def __init__(
        self,
        *,
        enabled: bool = False,
        output_root: str | Path | None = None,
        token_budget: int = 1800,
        include_conflicts: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.output_root = Path(output_root) if output_root else get_agent_workspace_dir() / "research"
        self.token_budget = max(64, int(token_budget))
        self.include_conflicts = bool(include_conflicts)
        self.system_prompt_builder = None
        self._stores: dict[str, ScientificProvenanceStore] = {}

    def init(self, agent) -> None:
        self.system_prompt_builder = getattr(agent, "system_prompt_builder", None)

    def uninit(self, agent) -> None:
        _ = agent
        if self.system_prompt_builder is not None:
            self.system_prompt_builder.remove_section(_SECTION_NAME)
        self.system_prompt_builder = None

    def _store(self, session_id: str) -> ScientificProvenanceStore:
        store = self._stores.get(session_id)
        if store is None:
            store = ScientificProvenanceStore(self.output_root / session_id)
            self._stores[session_id] = store
        return store

    def graph(self, session_id: str = "default") -> ScientificEvidenceGraph:
        return self._store(session_id).load_graph()

    def build_stage_context(
        self,
        session_id: str,
        *,
        stage: str,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        graph = self.graph(session_id)
        normalized_stage = self._normalize_stage(stage)
        budget = max(64, int(token_budget or self.token_budget))
        ranked = self._rank_items(graph, normalized_stage)

        used_tokens = 0
        selected_items: list[StageContextItem] = []
        selected_node_ids: list[str] = []
        for item in ranked:
            if used_tokens and used_tokens + item.estimated_tokens > budget:
                continue
            used_tokens += item.estimated_tokens
            selected_items.append(item)
            selected_node_ids.append(item.node_id)
            if used_tokens >= budget:
                break

        return {
            "stage": normalized_stage,
            "token_budget": budget,
            "used_tokens": used_tokens,
            "selected_node_ids": selected_node_ids,
            "items": [item.__dict__ for item in selected_items],
        }

    def render_context(self, bundle: dict[str, Any], *, language: str = "en") -> str:
        stage = str(bundle.get("stage") or "writing")
        items = bundle.get("items") or []
        if not items:
            return ""

        if language == "cn":
            lines = [
                "# 科研阶段上下文",
                "",
                f"- 当前阶段: `{stage}`",
                f"- Token 预算: `{bundle.get('token_budget', 0)}`",
                f"- 已选上下文估算: `{bundle.get('used_tokens', 0)}`",
                "",
            ]
        else:
            lines = [
                "# Stage-Aware Scientific Context",
                "",
                f"- stage: `{stage}`",
                f"- token_budget: `{bundle.get('token_budget', 0)}`",
                f"- estimated_tokens: `{bundle.get('used_tokens', 0)}`",
                "",
            ]

        for item in items:
            lines.append(
                f"- [{item['kind']}] `{item['node_id']}` score=`{item['score']:.3f}`: {item['summary']}"
            )
        lines.append("")
        return "\n".join(lines)

    async def before_model_call(self, ctx: AgentCallbackContext) -> None:
        if not self.enabled or self.system_prompt_builder is None:
            return

        session_id = self._session_id(ctx)
        stage = self._resolve_stage(ctx)
        bundle = self.build_stage_context(session_id, stage=stage)
        content = self.render_context(
            bundle,
            language=getattr(self.system_prompt_builder, "language", "en") or "en",
        )
        self.system_prompt_builder.remove_section(_SECTION_NAME)
        if not content:
            return

        self.system_prompt_builder.add_section(
            PromptSection(
                name=_SECTION_NAME,
                content={"cn": content, "en": content},
                priority=self.section_priority,
            )
        )
        extra = getattr(ctx, "extra", None)
        if isinstance(extra, dict):
            extra[_SECTION_NAME] = bundle

    def _rank_items(self, graph: ScientificEvidenceGraph, stage: str) -> list[StageContextItem]:
        items: list[StageContextItem] = []
        for node in graph.nodes.values():
            summary = self._summary(node)
            estimated_tokens = self._estimate_tokens(summary)
            score = self._score_node(graph, node, stage)
            items.append(
                StageContextItem(
                    node_id=node.id,
                    kind=node.kind.value,
                    score=score,
                    estimated_tokens=estimated_tokens,
                    summary=summary,
                )
            )
        return sorted(
            items,
            key=lambda item: (-item.score / max(item.estimated_tokens, 1), -item.score, item.node_id),
        )

    def _score_node(self, graph: ScientificEvidenceGraph, node: ProvenanceNode, stage: str) -> float:
        weight = _STAGE_KIND_WEIGHTS.get(stage, _STAGE_KIND_WEIGHTS["writing"]).get(node.kind, 0.1)
        reliability = self._reliability(node)
        coverage = 1.0
        conflict_bonus = 0.0

        if isinstance(node, Claim):
            support_count = len(graph.claim_support_ids(node))
            coverage += min(0.6, 0.1 * support_count)
            if node.status in {ClaimStatus.REFUTED, ClaimStatus.INCONCLUSIVE} and self.include_conflicts:
                conflict_bonus += 0.45 if stage in {"ideation", "reviewing"} else 0.15
        elif isinstance(node, LiteratureEvidence) and self.include_conflicts:
            if self._has_conflict_edges(graph, node.id):
                conflict_bonus += 0.25
        elif isinstance(node, ExperimentArtifact):
            if node.metric_value not in ("", None):
                coverage += 0.2
        elif isinstance(node, ExperimentPlan):
            if node.metric_names:
                coverage += 0.15
        elif isinstance(node, RevisionDecision):
            conflict_bonus += 0.25 if stage == "reviewing" else 0.0

        return weight * (0.7 + reliability) * coverage + conflict_bonus

    @staticmethod
    def _has_conflict_edges(graph: ScientificEvidenceGraph, node_id: str) -> bool:
        return any(edge.relation.value == "contradicts" for edge in graph.incoming(node_id) + graph.outgoing(node_id))

    @staticmethod
    def _reliability(node: ProvenanceNode) -> float:
        if isinstance(node, LiteratureEvidence):
            value = node.reliability_score
            return float(value) if isinstance(value, (int, float)) else 0.55
        if isinstance(node, ExperimentArtifact):
            return 0.9 if node.sha256 and node.command else 0.65
        if isinstance(node, ExperimentPlan):
            return 0.75 if node.command else 0.55
        if isinstance(node, Claim):
            return {
                ClaimStatus.SUPPORTED: 0.95,
                ClaimStatus.TESTED: 0.8,
                ClaimStatus.PROPOSED: 0.5,
                ClaimStatus.INCONCLUSIVE: 0.45,
                ClaimStatus.REFUTED: 0.4,
            }.get(node.status, 0.5)
        return 0.6

    @staticmethod
    def _summary(node: ProvenanceNode) -> str:
        if isinstance(node, LiteratureEvidence):
            authors = ", ".join(node.authors[:3])
            year = node.year if node.year is not None else "n/a"
            return f"{node.title} ({year}) by {authors or 'unknown authors'}; summary={node.summary or '-'}"
        if isinstance(node, ExperimentArtifact):
            return (
                f"experiment_id={node.experiment_id}, dataset={node.dataset or '-'}, "
                f"{node.metric_name or 'metric'}={node.metric_value!s}"
            )
        if isinstance(node, ExperimentPlan):
            return (
                f"experiment_id={node.experiment_id}, dataset={node.dataset or '-'}, "
                f"metrics={', '.join(node.metric_names) or '-'}"
            )
        if isinstance(node, Claim):
            return f"status={node.status.value}, section={node.section or '-'}, text={node.text}"
        if isinstance(node, Hypothesis):
            return f"rank={node.rank if node.rank is not None else '-'}, text={node.text}"
        if isinstance(node, ResearchQuestion):
            return node.text
        if isinstance(node, RevisionDecision):
            return f"claim_id={node.claim_id}, decision={node.decision}, rationale={node.rationale or '-'}"
        return node.id

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        compact = str(text or "").strip()
        if not compact:
            return 24
        return max(24, min(320, len(compact) // 4 + 8))

    @staticmethod
    def _normalize_stage(stage: str | None) -> str:
        value = str(stage or "").strip().lower()
        return _STAGE_ALIASES.get(value, "writing")

    def _resolve_stage(self, ctx: AgentCallbackContext) -> str:
        for source in (getattr(ctx, "inputs", None), getattr(ctx, "extra", None)):
            if isinstance(source, dict):
                for key in ("research_stage", "stage", "workflow_stage"):
                    value = source.get(key)
                    if value:
                        return self._normalize_stage(str(value))
            else:
                for key in ("research_stage", "stage", "workflow_stage"):
                    value = getattr(source, key, None)
                    if value:
                        return self._normalize_stage(str(value))
        return "writing"

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


__all__ = ["StageAwareContextRail"]

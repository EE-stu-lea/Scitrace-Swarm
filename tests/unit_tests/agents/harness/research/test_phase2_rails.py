from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[5]


def _ensure_module(name: str) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        sys.modules[name] = module
    return module


def _ensure_package(name: str) -> ModuleType:
    module = _ensure_module(name)
    if not hasattr(module, "__path__"):
        module.__path__ = []  # type: ignore[attr-defined]
    return module


def _load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_openjiuwen_stubs() -> None:
    for name in [
        "openjiuwen",
        "openjiuwen.core",
        "openjiuwen.core.single_agent",
        "openjiuwen.core.single_agent.rail",
        "openjiuwen.harness",
        "openjiuwen.harness.rails",
        "openjiuwen.agent_teams",
        "openjiuwen.agent_teams.harness",
        "openjiuwen.agent_teams.schema",
    ]:
        _ensure_package(name)

    rail_base = _ensure_module("openjiuwen.core.single_agent.rail.base")
    rail_base.AgentCallbackContext = object

    harness_rail_base = _ensure_module("openjiuwen.harness.rails.base")

    class _DeepAgentRail:
        priority = 50

        def __init__(self, *args, **kwargs):
            pass

        def init(self, agent) -> None:
            _ = agent

        def uninit(self, agent) -> None:
            _ = agent

    harness_rail_base.DeepAgentRail = _DeepAgentRail

    prompts = _ensure_module("openjiuwen.harness.prompts")

    @dataclass
    class _PromptSection:
        name: str
        content: dict[str, str]
        priority: int = 0

    prompts.PromptSection = _PromptSection

    manifest = _ensure_module("openjiuwen.agent_teams.harness.manifest")

    class _ElementKind:
        RAIL = "rail"

    def _harness_element(*args, **kwargs):
        builder = kwargs.get("builder")
        if builder is not None:
            return builder

        def decorator(func):
            return func

        return decorator

    manifest.ElementKind = _ElementKind
    manifest.harness_element = _harness_element

    schema_build_context = _ensure_module("openjiuwen.agent_teams.schema.build_context")

    @dataclass
    class _BuildContext:
        member_name: str = ""
        role: str = ""
        workspace: object | None = None
        member_card_id: str = ""
        language: str = "en"

    schema_build_context.BuildContext = _BuildContext

    paths = _ensure_module("openjiuwen.agent_teams.paths")
    paths.SKILL_VISIBILITY_FILENAME = "skills-visibility.json"

    def _member_skill_visibility_path(team_id: str, member_name: str) -> str:
        return f"/tmp/{team_id}/{member_name}/skills-visibility.json"

    paths.member_skill_visibility_path = _member_skill_visibility_path


_install_openjiuwen_stubs()
_ensure_package("jiuwenswarm.agents.swarm")
_ensure_package("jiuwenswarm.agents.swarm.providers")

ScientificProvenanceRail = importlib.import_module(
    "jiuwenswarm.agents.harness.research.rails.scientific_provenance_rail"
).ScientificProvenanceRail
ScientificConsistencyGateRail = importlib.import_module(
    "jiuwenswarm.agents.harness.research.rails.scientific_consistency_gate_rail"
).ScientificConsistencyGateRail
StageAwareContextRail = importlib.import_module(
    "jiuwenswarm.agents.harness.research.rails.stage_aware_context_rail"
).StageAwareContextRail
ResourceAuditRail = importlib.import_module(
    "jiuwenswarm.agents.harness.research.rails.resource_audit_rail"
).ResourceAuditRail
ScientificProvenanceStore = importlib.import_module(
    "jiuwenswarm.agents.harness.research.provenance.store"
).ScientificProvenanceStore
models = importlib.import_module("jiuwenswarm.agents.harness.research.provenance.models")
SwarmBuildContext = _load_module(
    "jiuwenswarm.agents.swarm.context",
    "jiuwenswarm/agents/swarm/context.py",
).SwarmBuildContext
research_rails = _load_module(
    "jiuwenswarm.agents.swarm.providers.research_rails",
    "jiuwenswarm/agents/swarm/providers/research_rails.py",
)


class _FakeSession:
    def __init__(self, session_id: str) -> None:
        self._session_id = session_id

    def get_session_id(self) -> str:
        return self._session_id


class _FakeUsage:
    def __init__(self, payload: dict[str, int]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, int]:
        return dict(self._payload)


class _FakeResponse:
    def __init__(self, usage_payload: dict[str, int]) -> None:
        self.usage_metadata = _FakeUsage(usage_payload)


class _FakePromptBuilder:
    def __init__(self, language: str = "en") -> None:
        self.language = language
        self.sections: dict[str, object] = {}

    def add_section(self, section) -> None:
        self.sections[section.name] = section

    def remove_section(self, name: str) -> None:
        self.sections.pop(name, None)


def _run(awaitable) -> object:
    return asyncio.run(awaitable)


def test_scientific_provenance_rail_ingests_payload_and_audits(tmp_path) -> None:
    rail = ScientificProvenanceRail(enabled=True, output_root=tmp_path)
    session = _FakeSession("sess-prov")
    payload = {
        "scientific_provenance": {
            "nodes": [
                {
                    "kind": "LiteratureEvidence",
                    "title": "Evidence Graphs for Research",
                    "authors": ["Alice"],
                    "year": 2026,
                    "source_url": "https://example.org/paper",
                    "summary": "Paper summary",
                },
                {
                    "kind": "Claim",
                    "text": "Evidence graphs improve claim grounding.",
                    "evidence_ids": [],
                    "artifact_ids": [],
                    "section": "results",
                    "status": "proposed",
                },
            ],
            "audit": {"source": "unit-test"},
        }
    }
    extra: dict[str, object] = {}
    ctx = SimpleNamespace(
        session=session,
        inputs=SimpleNamespace(
            tool_name="web_fetch",
            tool_args={},
            tool_result=payload,
            conversation_id="sess-prov",
        ),
        extra=extra,
        exception=None,
    )

    _run(rail.before_tool_call(ctx))
    _run(rail.after_tool_call(ctx))

    graph = rail.graph("sess-prov")
    assert len(graph.nodes) == 2
    audit_file = tmp_path / "sess-prov" / "scientific_provenance.audit.jsonl"
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert any(json.loads(line)["event_type"] == "tool_audit" for line in lines)


def test_stage_aware_context_rail_respects_stage_and_budget(tmp_path) -> None:
    evidence = models.LiteratureEvidence(
        title="Grounded Research Systems",
        authors=["Alice"],
        year=2026,
        source_url="https://example.org/grounded",
        summary="Evidence-backed agents produce more reliable papers.",
        reliability_score=0.9,
    )
    hypothesis = models.Hypothesis(text="Evidence graphs improve research writing.", rank=1)
    artifact = models.ExperimentArtifact(
        experiment_id="exp-1",
        file_path="results/exp-1.json",
        sha256="abc123",
        command="python train.py --seed 7",
        seed=7,
        dataset="SciBench",
        metric_name="accuracy",
        metric_value=0.91,
    )
    claim = models.Claim(
        text="Evidence graphs improve research writing quality.",
        evidence_ids=[evidence.id],
        artifact_ids=[artifact.id],
        section="results",
        status=models.ClaimStatus.SUPPORTED,
    )
    store = ScientificProvenanceStore(tmp_path / "sess-stage")
    for node in (evidence, hypothesis, artifact, claim):
        store.upsert_node(node)

    rail = StageAwareContextRail(enabled=True, output_root=tmp_path, token_budget=80)
    ideation = rail.build_stage_context("sess-stage", stage="ideation")
    writing = rail.build_stage_context("sess-stage", stage="writing")

    assert ideation["used_tokens"] <= 80
    assert writing["used_tokens"] <= 80
    assert {evidence.id, hypothesis.id}.issubset(set(ideation["selected_node_ids"][:2]))
    assert {artifact.id, claim.id}.issubset(set(writing["selected_node_ids"][:2]))

    builder = _FakePromptBuilder(language="en")
    rail.init(SimpleNamespace(system_prompt_builder=builder))
    ctx = SimpleNamespace(
        session=_FakeSession("sess-stage"),
        inputs=SimpleNamespace(conversation_id="sess-stage"),
        extra={"research_stage": "writing"},
    )
    _run(rail.before_model_call(ctx))
    assert "scientific_stage_context" in builder.sections
    assert ctx.extra["scientific_stage_context"]["stage"] == "writing"


def test_scientific_consistency_gate_blocks_invalid_payload(tmp_path) -> None:
    evidence = models.LiteratureEvidence(
        title="Negative Results Matter",
        authors=["Bob"],
        year=2025,
        source_url="https://example.org/negative",
        summary="The tested hypothesis is not supported by the data.",
        reliability_score=0.88,
    )
    artifact = models.ExperimentArtifact(
        experiment_id="exp-neg",
        file_path="results/exp-neg.json",
        sha256="def456",
        command="python evaluate.py --seed 9",
        seed=9,
        dataset="SciBench",
        metric_name="accuracy",
        metric_value=0.81,
    )
    claim = models.Claim(
        text="The method substantially improves benchmark accuracy.",
        evidence_ids=[evidence.id],
        artifact_ids=[artifact.id],
        section="results",
        status=models.ClaimStatus.REFUTED,
    )
    store = ScientificProvenanceStore(tmp_path / "sess-gate")
    for node in (evidence, artifact, claim):
        store.upsert_node(node)

    rail = ScientificConsistencyGateRail(enabled=True, output_root=tmp_path, block_on_error=True)
    payload = {
        "scientific_consistency_gate": {
            "draft_text": "The method substantially improves benchmark accuracy and achieves significant gains.",
            "citations": ["missing-citation"],
            "claims": [claim.id],
            "reported_metrics": [
                {
                    "artifact_id": artifact.id,
                    "metric_name": "accuracy",
                    "claimed_value": 0.92,
                }
            ],
        }
    }
    result = rail.check_payload(payload["scientific_consistency_gate"], session_id="sess-gate")
    codes = {item["code"] for item in result["errors"]}
    assert {"citation_missing", "metric_mismatch", "refuted_claim_promoted"}.issubset(codes)

    ctx = SimpleNamespace(
        session=_FakeSession("sess-gate"),
        inputs=SimpleNamespace(
            tool_name="paper_writer",
            tool_args={},
            tool_result=payload,
            conversation_id="sess-gate",
        ),
        extra={},
        exception=None,
    )
    _run(rail.after_tool_call(ctx))
    assert ctx.inputs.tool_result["blocked"] is True
    assert len(ctx.inputs.tool_result["errors"]) >= 3


def test_resource_audit_rail_tracks_usage_and_writes_reports(tmp_path) -> None:
    rail = ResourceAuditRail(
        enabled=True,
        output_root=tmp_path,
        member_name="leader",
        role="leader",
    )
    session = _FakeSession("sess-audit")
    invoke_extra: dict[str, object] = {}
    invoke_ctx = SimpleNamespace(
        session=session,
        inputs=SimpleNamespace(conversation_id="sess-audit"),
        extra=invoke_extra,
    )
    _run(rail.before_invoke(invoke_ctx))

    tool_extra: dict[str, object] = {}
    tool_ctx = SimpleNamespace(
        session=session,
        inputs=SimpleNamespace(
            tool_name="shell",
            tool_result={"success": False, "status": "error"},
            conversation_id="sess-audit",
        ),
        extra=tool_extra,
        exception=None,
    )
    _run(rail.before_tool_call(tool_ctx))
    _run(rail.after_tool_call(tool_ctx))

    model_ctx = SimpleNamespace(
        session=session,
        inputs=SimpleNamespace(
            response=_FakeResponse({"input_tokens": 11, "output_tokens": 5, "total_tokens": 16}),
            conversation_id="sess-audit",
        ),
        agent=SimpleNamespace(_config=SimpleNamespace(model_name="qwen-test")),
        extra={},
    )
    _run(rail.after_model_call(model_ctx))
    _run(rail.after_invoke(invoke_ctx))

    snapshot = rail.snapshot("sess-audit")
    assert snapshot.invoke_count == 1
    assert snapshot.tool_calls == 1
    assert snapshot.tool_failures == 1
    assert snapshot.model_calls == 1
    assert snapshot.total_tokens == 16
    assert snapshot.model_calls_by_name["qwen-test"] == 1

    json_path = tmp_path / "sess-audit" / "resource_report.json"
    markdown_path = tmp_path / "sess-audit" / "resource_report.md"
    assert json_path.exists()
    assert markdown_path.exists()
    content = json.loads(json_path.read_text(encoding="utf-8"))
    assert content["total_tokens"] == 16


def test_research_provider_gates_disabled_and_builds_enabled(tmp_path) -> None:
    context = SwarmBuildContext(
        session_id="sess-provider",
        project_dir=str(tmp_path),
        member_name="leader",
        role="leader",
    )

    disabled = research_rails.build_scientific_provenance_rail({"enabled": False}, context)
    enabled = research_rails.build_scientific_provenance_rail({"enabled": True}, context)
    stage_enabled = research_rails.build_stage_aware_context_rail({"enabled": True}, context)
    gate_enabled = research_rails.build_scientific_consistency_gate_rail({"enabled": True}, context)
    audit_enabled = research_rails.build_resource_audit_rail({"enabled": True}, context)

    assert disabled is None
    assert enabled is not None
    assert stage_enabled is not None
    assert gate_enabled is not None
    assert audit_enabled is not None
    assert enabled.output_root == tmp_path / ".scitrace" / "provenance"
    assert stage_enabled.output_root == tmp_path / ".scitrace" / "provenance"
    assert gate_enabled.output_root == tmp_path / ".scitrace" / "provenance"
    assert audit_enabled.reporter.output_root == tmp_path / ".scitrace" / "resource_audit"

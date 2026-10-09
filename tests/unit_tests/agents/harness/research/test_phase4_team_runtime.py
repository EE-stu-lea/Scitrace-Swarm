from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import pytest

from ._stubs import install_openjiuwen_stubs

install_openjiuwen_stubs()

from examples.scitrace_research.workflow.pipeline import load_yaml_config, run_pipeline
from jiuwenswarm.agents.harness.research.claim_compiler import ClaimCompilerInput
from jiuwenswarm.agents.harness.research.run_store import ResearchRunStore
from jiuwenswarm.agents.harness.research.runtime import ResearchStageHarness
from jiuwenswarm.agents.harness.research.team_contract import get_team_stage_contract
from jiuwenswarm.agents.harness.research.team_runtime import (
    DeterministicTeamBackend,
    JiuwenSwarmResearchTeamRuntime,
    OpenJiuwenTeamBackend,
    _extract_top_level_json,
)


class _SequenceRuntime:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls = 0

    async def complete_json(self, **kwargs) -> dict:
        _ = kwargs
        self.calls += 1
        return self.payloads.pop(0)

    async def complete_text(self, **kwargs) -> str:
        _ = kwargs
        return "draft"

    def usage_summary(self) -> dict:
        return {}

    def runtime_info(self) -> dict:
        return {}

    def call_count(self) -> int:
        return self.calls


def test_stage_harness_retries_valid_json_with_wrong_topic_schema() -> None:
    runtime = _SequenceRuntime([
        {"topic": "wrong shape"},
        {"research_question": "rq", "hypothesis": "hyp"},
    ])
    harness = ResearchStageHarness(runtime, max_retries=2)
    try:
        payload = harness.generate_research_brief(topic="topic")
    finally:
        harness.close()
    assert payload == {"research_question": "rq", "hypothesis": "hyp"}
    assert runtime.calls == 2


def test_team_json_extractor_accepts_coordination_prefix_and_suffix() -> None:
    payload = _extract_top_level_json('handoff follows:\n{"result":{"ok":true}}\ncomplete')
    assert payload == {"result": {"ok": True}}


def test_experiment_plan_normalizes_structured_hypothesis() -> None:
    runtime = _SequenceRuntime([
        {
            "hypothesis": {"hypothesis_id": "hyp_1", "statement": "Grounded planning improves traceability."},
            "controls": "fixed benchmark",
            "metrics": ["citation_support_rate"],
            "limitations": ["controlled setting"],
        }
    ])
    harness = ResearchStageHarness(runtime, max_retries=1)
    try:
        payload = harness.design_experiment(payload={"trusted_plan": {}})
    finally:
        harness.close()
    assert payload["hypothesis"] == "Grounded planning improves traceability."
    assert payload["controls"] == ["fixed benchmark"]


def test_experiment_plan_preserves_structured_controls_as_list_entry() -> None:
    controls = {
        "design": "matched budget and split",
        "conditions": [{"name": "fixed_rrf"}, {"name": "calibrated"}],
    }
    runtime = _SequenceRuntime([{
        "hypothesis": {"statement": "Calibration changes retrieval quality."},
        "controls": controls,
        "metrics": {"primary": "evidence_recall", "secondary": ["mrr", "ndcg"]},
        "limitations": ["two datasets"],
    }])
    harness = ResearchStageHarness(runtime, max_retries=1)
    try:
        payload = harness.design_experiment(payload={"trusted_plan": {}})
    finally:
        harness.close()
    assert payload["controls"] == [controls]
    assert payload["metrics"] == [{"primary": "evidence_recall", "secondary": ["mrr", "ndcg"]}]
    assert runtime.calls == 1


def test_stage_harness_normalizes_team_claim_array() -> None:
    runtime = _SequenceRuntime([
        {"topic": "topic", "status": "supported", "claims": [{"text": "grounded claim", "confidence": 0.8}]}
    ])
    harness = ResearchStageHarness(runtime, allow_claim_fallback=False)
    try:
        payload = harness.compile_claim(ClaimCompilerInput(topic="topic", evidence=[], artifacts=[]))
    finally:
        harness.close()
    assert payload["text"] == "grounded claim"
    assert payload["section"] == "results"


def test_stage_harness_normalizes_nested_team_claim_object() -> None:
    runtime = _SequenceRuntime([
        {"claim": {"text": "nested grounded claim", "confidence": 0.7, "section": "results"}}
    ])
    harness = ResearchStageHarness(runtime, allow_claim_fallback=False, max_retries=1)
    try:
        payload = harness.compile_claim(ClaimCompilerInput(topic="topic", evidence=[], artifacts=[]))
    finally:
        harness.close()
    assert payload["text"] == "nested grounded claim"
    assert payload["confidence"] == 0.7


def _team_mock_config() -> dict:
    path = (
        Path(__file__).resolve().parents[5]
        / "examples"
        / "scitrace_research"
        / "configs"
        / "team_mock.yaml"
    )
    return load_yaml_config(path)


def test_team_runtime_returns_valid_json_handoff(tmp_path) -> None:
    call_log = tmp_path / "model_calls.jsonl"
    runtime = JiuwenSwarmResearchTeamRuntime(
        backend=DeterministicTeamBackend(config=_team_mock_config(), output_root=tmp_path, research_id="team_case"),
        research_id="team_case",
        output_root=tmp_path,
        call_log_path=call_log,
    )

    payload = asyncio.run(
        runtime.complete_json(
            stage="topic_intake",
            operation="research_question",
            system_prompt="Return JSON only.",
            user_prompt="Evidence-aware context governance for scientific research agents",
        )
    )

    assert "research_question" in payload
    assert "hypothesis" in payload
    logged = json.loads(call_log.read_text(encoding="utf-8").splitlines()[-1])
    assert {"prompt_tokens", "completion_tokens", "total_tokens"} <= logged.keys()


def test_team_runtime_strips_markdown_handoff_but_keeps_structured_refs(tmp_path) -> None:
    runtime = JiuwenSwarmResearchTeamRuntime(
        backend=DeterministicTeamBackend(config=_team_mock_config(), output_root=tmp_path, research_id="team_case"),
        research_id="team_case",
        output_root=tmp_path,
    )
    prompt = json.dumps(
        {
            "topic": "Evidence-aware context governance for scientific research agents",
            "claims": [
                {
                    "id": "claim_1",
                    "status": "supported",
                    "text": "Evidence-aware context governance improves claim grounding.",
                }
            ],
            "evidence": [{"id": "evidence_1"}],
            "artifact": {
                "id": "artifact_1",
                "metric_name": "citation_support_rate",
                "metric_value": 0.87,
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    markdown = asyncio.run(
        runtime.complete_text(
            stage="draft",
            operation="write_draft",
            system_prompt="Write markdown only.",
            user_prompt=prompt,
        )
    )

    assert "## Team Handoff" not in markdown
    assert "## Structured References" in markdown
    assert "citation_support_rate" in markdown


def test_scitrace_mock_team_end_to_end_workflow_runs(tmp_path) -> None:
    result = run_pipeline(
        topic="Evidence-aware context governance for scientific research agents",
        config=_team_mock_config(),
        output_dir=tmp_path / "team_demo001",
        mode="mock_team",
        resume=False,
        dry_run=False,
    )

    assert result["status"] == "success"
    output_dir = Path(result["output_dir"])
    assert (output_dir / "team_manifest.json").exists()
    assert (output_dir / "team_handoffs.jsonl").exists()
    assert (output_dir / "team_events.jsonl").exists()

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["mode"] == "mock_team"

    validation = json.loads((output_dir / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["ok"] is True

    usage = json.loads((output_dir / "resource_usage.json").read_text(encoding="utf-8"))
    assert usage["model_calls"] == 5
    assert usage["invoke_count"] == 5
    assert usage["session_id"] == "scitrace-team-team_demo001"
    assert usage["role"] == "native_team"
    assert "research_leader" in usage["member_name"]
    assert usage["tool_calls"] == 2
    assert usage["tool_failures"] == 0
    assert usage["total_tool_seconds"] > 0
    assert usage["total_experiment_seconds"] > 0
    assert usage["tools_by_name"] == {"experiment_execute": 1, "literature_search": 1}
    assert (output_dir / "review_report.json").exists()
    assert usage["extra"]["workflow_model_calls"] == 5

    team_manifest = json.loads((output_dir / "team_manifest.json").read_text(encoding="utf-8"))
    assert team_manifest["research_id"] == "team_demo001"
    assert team_manifest["session_id"] == "scitrace-team-team_demo001"
    assert team_manifest["roles"] == [
        "research_leader",
        "literature_ideation_agent",
        "experiment_agent",
        "writer_agent",
        "auditor_agent",
    ]

    handoffs = [
        json.loads(line)
        for line in (output_dir / "team_handoffs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(handoffs) == 5
    for entry in handoffs:
        payload = entry["payload"]
        contract = get_team_stage_contract(entry["stage"], entry["operation"])
        assert payload["research_id"] == "team_demo001"
        assert payload["to_member"] == "research_leader"
        assert payload["from_member"] == contract.primary_member
        assert payload["write_scope"] == list(contract.write_scope)
        assert payload["permissions_ack"] is True


class _InvalidHandoffBackend:
    def runtime_info(self) -> dict:
        return {
            "backend": "invalid-team",
            "provider": "mock_team",
            "model_name": "invalid-team",
            "api_base": "",
        }

    async def invoke(self, *, stage: str, operation: str, query: str) -> str:
        _ = query
        contract = get_team_stage_contract(stage, operation)
        return json.dumps(
            {
                "stage": stage,
                "operation": operation,
                "from_member": contract.primary_member,
                "to_member": "research_leader",
                "research_id": "wrong_research_id",
                "permissions_ack": True,
                "write_scope": list(contract.write_scope),
                "result": {
                    "research_question": "bad handoff",
                    "hypothesis": "should fail validation",
                },
            },
            ensure_ascii=False,
        )


def test_team_runtime_rejects_mismatched_research_id_in_handoff(tmp_path) -> None:
    runtime = JiuwenSwarmResearchTeamRuntime(
        backend=_InvalidHandoffBackend(),
        research_id="expected_research_id",
        output_root=tmp_path,
        call_log_path=tmp_path / "model_calls.jsonl",
    )

    with pytest.raises(ValueError, match="research_id mismatch"):
        asyncio.run(
            runtime.complete_json(
                stage="topic_intake",
                operation="research_question",
                system_prompt="Return JSON only.",
                user_prompt="Evidence-aware context governance for scientific research agents",
            )
        )

    call_log = (tmp_path / "model_calls.jsonl").read_text(encoding="utf-8")
    assert "research_id mismatch" in call_log


def test_run_store_uses_wal_and_busy_timeout_for_team_resume(tmp_path) -> None:
    store = ResearchRunStore(tmp_path / "team_state")

    with store._connect() as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]

    assert str(journal_mode).lower() == "wal"
    assert int(busy_timeout) >= 30000


def test_openjiuwen_backend_returns_on_terminal_answer_chunk(tmp_path, monkeypatch) -> None:
    class _Chunk:
        def __init__(self, *, role: str, type_: str, payload: dict):
            self.role = role
            self.type = type_
            self.payload = payload

    answer_text = json.dumps(
        {
            "stage": "claim_compiler",
            "operation": "compile_claim",
            "from_member": "research_leader",
            "to_member": "research_leader",
            "research_id": "team_case",
            "permissions_ack": True,
            "write_scope": ["claims"],
            "result": {
                "text": "Evidence-aware context governance improves claim grounding.",
                "confidence": 0.83,
                "section": "results",
            },
        },
        ensure_ascii=False,
    )

    async def _fake_stream(*args, **kwargs):
        _ = args, kwargs
        yield _Chunk(
            role="leader",
            type_="llm_output",
            payload={"output": {"output": "partial"}},
        )
        yield _Chunk(
            role="leader",
            type_="answer",
            payload={"output": {"output": answer_text}},
        )
        raise AssertionError("stream should stop after terminal answer chunk")

    class _Runner:
        @staticmethod
        def run_agent_team_streaming(*args, **kwargs):
            return _fake_stream(*args, **kwargs)

    backend = OpenJiuwenTeamBackend(config=_team_mock_config(), output_root=tmp_path)
    backend._team_spec = object()
    monkeypatch.setitem(sys.modules, "openjiuwen.core.runner", type("M", (), {"Runner": _Runner}))

    result = asyncio.run(
        backend.invoke(
            stage="claim_compiler",
            operation="compile_claim",
            query="test",
        )
    )

    assert json.loads(result)["operation"] == "compile_claim"


def test_openjiuwen_backend_ignores_non_json_leader_answer_until_structured_result(tmp_path, monkeypatch) -> None:
    class _Chunk:
        def __init__(self, *, role: str, type_: str, payload: dict):
            self.role = role
            self.type = type_
            self.payload = payload

    waiting_text = "正在处理 topic_intake / research_question 阶段，尚未形成结构化结果。"
    final_text = json.dumps(
        {
            "stage": "topic_intake",
            "operation": "research_question",
            "from_member": "research_leader",
            "to_member": "research_leader",
            "research_id": "team_case",
            "permissions_ack": True,
            "write_scope": ["research_question", "hypothesis"],
            "result": {
                "research_question": "How should evidence-aware context governance be designed for scientific research agents?",
                "hypothesis": "Evidence-aware context governance improves claim grounding while preserving answer coverage.",
            },
        },
        ensure_ascii=False,
    )

    async def _fake_stream(*args, **kwargs):
        _ = args, kwargs
        yield _Chunk(
            role="leader",
            type_="answer",
            payload={"output": {"output": waiting_text}},
        )
        yield _Chunk(
            role="leader",
            type_="answer",
            payload={"output": {"output": final_text}},
        )

    class _Runner:
        @staticmethod
        def run_agent_team_streaming(*args, **kwargs):
            return _fake_stream(*args, **kwargs)

    backend = OpenJiuwenTeamBackend(config=_team_mock_config(), output_root=tmp_path)
    backend._team_spec = object()
    monkeypatch.setitem(sys.modules, "openjiuwen.core.runner", type("M", (), {"Runner": _Runner}))

    result = asyncio.run(
        backend.invoke(
            stage="topic_intake",
            operation="research_question",
            query="test",
        )
    )

    payload = json.loads(result)
    assert payload["operation"] == "research_question"
    assert payload["result"]["research_question"].startswith("How should evidence-aware")


def test_research_stage_harness_reuses_single_asyncio_runner() -> None:
    class _Runtime:
        def __init__(self) -> None:
            self.loop_ids: list[int] = []

        async def complete_json(self, *, stage: str, operation: str, system_prompt: str, user_prompt: str) -> dict:
            _ = stage, operation, system_prompt, user_prompt
            self.loop_ids.append(id(asyncio.get_running_loop()))
            return {
                "research_question": "rq",
                "hypothesis": "hyp",
            }

        async def complete_text(self, *, stage: str, operation: str, system_prompt: str, user_prompt: str) -> str:
            _ = stage, operation, system_prompt, user_prompt
            self.loop_ids.append(id(asyncio.get_running_loop()))
            return "# draft"

        def usage_summary(self) -> dict:
            return {}

        def runtime_info(self) -> dict:
            return {}

        def call_count(self) -> int:
            return 0

    runtime = _Runtime()
    harness = ResearchStageHarness(runtime=runtime)

    brief = harness.generate_research_brief(topic="topic")
    draft = harness.write_draft(prompt="prompt")
    harness.close()

    assert brief["research_question"] == "rq"
    assert draft == "# draft"
    assert len(runtime.loop_ids) == 2
    assert runtime.loop_ids[0] == runtime.loop_ids[1]


def test_openjiuwen_backend_uses_fresh_session_per_stage(tmp_path, monkeypatch) -> None:
    class _Chunk:
        def __init__(self, *, role: str, type_: str, payload: dict):
            self.role = role
            self.type = type_
            self.payload = payload

    answer_text = json.dumps(
        {
            "stage": "claim_compiler",
            "operation": "compile_claim",
            "from_member": "research_leader",
            "to_member": "research_leader",
            "research_id": "team_case",
            "permissions_ack": True,
            "write_scope": ["claims"],
            "result": {"text": "claim text", "confidence": 0.8, "section": "results"},
        },
        ensure_ascii=False,
    )

    calls: list[tuple[object, object]] = []

    async def _fake_stream(*args, **kwargs):
        calls.append((kwargs.get("agent_team"), kwargs.get("session")))
        yield _Chunk(
            role="leader",
            type_="answer",
            payload={"output": {"output": answer_text}},
        )

    class _Runner:
        @staticmethod
        def run_agent_team_streaming(*args, **kwargs):
            return _fake_stream(*args, **kwargs)

    backend = OpenJiuwenTeamBackend(config=_team_mock_config(), output_root=tmp_path)
    sentinel = object()
    backend._team_spec = sentinel
    monkeypatch.setitem(sys.modules, "openjiuwen.core.runner", type("M", (), {"Runner": _Runner}))

    asyncio.run(backend.invoke(stage="claim_compiler", operation="compile_claim", query="first"))
    asyncio.run(backend.invoke(stage="claim_compiler", operation="compile_claim", query="second"))

    # Every stage passes the spec and a fresh (incremented) session, avoiding
    # RESUME_FROM_PAUSE which yields an empty stream on a concluded leader.
    assert calls[0][0] is sentinel
    assert calls[1][0] is sentinel
    assert calls[0][1] != calls[1][1]

"""Provider-shaped offline regressions; these tests never call a model service."""
from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from ._stubs import install_openjiuwen_stubs

install_openjiuwen_stubs()

from jiuwenswarm.agents.harness.research.team_runtime import (
    DeterministicTeamBackend,
    JiuwenSwarmResearchTeamRuntime,
    OpenJiuwenTeamBackend,
    TeamResourceBudgetExceeded,
    TeamProviderError,
    _provider_failure_message,
    _usage_from_value,
    _stage_timeout_seconds,
)


def _chunk(kind, payload, role="leader"):
    return SimpleNamespace(type=kind, payload=payload, role=role)


def _backend(tmp_path, monkeypatch, chunks, team=None):
    closed, stopped, started = [], [], []

    async def stream(**kwargs):
        started.append(kwargs["session"])
        try:
            for chunk in chunks:
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk
        finally:
            closed.append(True)

    async def stop(**kwargs):
        stopped.append(kwargs)
        return True

    runner = SimpleNamespace(run_agent_team_streaming=stream, stop_agent_team=stop)
    monkeypatch.setitem(sys.modules, "openjiuwen.core.runner", SimpleNamespace(Runner=runner))
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.research.team_runtime._resolved_default_model",
        lambda config: {"provider": "offline", "model_name": "fixture", "api_base": ""},
    )
    backend = OpenJiuwenTeamBackend(config={"team": team or {}}, output_root=tmp_path)
    backend._team_spec = SimpleNamespace(team_name="fixture_team")
    return backend, closed, stopped, started


def _invoke(backend):
    return asyncio.run(backend.invoke(stage="claim_compiler", operation="compile_claim", query="fixture"))


def test_direct_specialist_uses_one_native_model_call_with_provider_receipt(tmp_path, monkeypatch):
    from jiuwenswarm.agents.harness.research.team_contract import get_team_stage_contract

    contract = get_team_stage_contract("experiment_plan", "design_experiment")
    calls = []

    class Model:
        async def invoke(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=json.dumps({
                    "stage": contract.stage, "operation": contract.operation,
                    "from_member": contract.primary_member, "to_member": "research_leader",
                    "research_id": "case", "permissions_ack": True,
                    "write_scope": list(contract.write_scope),
                    "result": {"hypothesis": "grounded", "controls": ["fixed"],
                               "metrics": ["recall"], "limitations": ["small sample"]},
                }),
                usage={"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
            )

    backend = OpenJiuwenTeamBackend(
        config={"team": {"direct_specialist_stages": ["experiment_plan"],
                         "stage_resource_limits": {"experiment_plan": {
                             "max_provider_calls": 1, "max_total_tokens": 1000}}}},
        output_root=tmp_path,
    )
    backend._direct_model = Model()
    runtime = JiuwenSwarmResearchTeamRuntime(backend=backend, research_id="case",
                                             output_root=tmp_path,
                                             call_log_path=tmp_path / "model_calls.jsonl")
    result = asyncio.run(runtime.complete_json(
        stage="experiment_plan", operation="design_experiment",
        system_prompt="Use checked evidence.", user_prompt="Review trusted plan.",
    ))
    assert result["hypothesis"] == "grounded"
    assert len(calls) == 1
    assert "Do not coordinate, delegate, poll tasks" in calls[0]["messages"][1]["content"]
    assert backend.last_usage()["total_tokens"] == 160
    receipt = json.loads((tmp_path / "provider_usage.jsonl").read_text())
    assert receipt["usage_source"] == "native_model_response"
    assert receipt["role"] == "experiment_agent"
    assert json.loads((tmp_path / "model_calls.jsonl").read_text())["execution_path"] == "direct_specialist"


def test_direct_specialist_missing_usage_fails_closed(tmp_path):
    class Model:
        async def invoke(self, **kwargs):
            return SimpleNamespace(content="answer", usage=None)

    backend = OpenJiuwenTeamBackend(
        config={"team": {"direct_specialist_stages": ["experiment_plan"],
                         "stage_resource_limits": {"experiment_plan": {"max_total_tokens": 1000}}}},
        output_root=tmp_path,
    )
    backend._direct_model = Model()
    with pytest.raises(TeamResourceBudgetExceeded, match="usage is missing"):
        asyncio.run(backend.invoke(stage="experiment_plan", operation="design_experiment", query="x"))
    assert backend.last_usage()["missing_usage_calls"] == 1


def test_direct_writer_returns_full_markdown_handoff_with_long_output_ceiling(tmp_path):
    from jiuwenswarm.agents.harness.research.team_contract import get_team_stage_contract

    contract = get_team_stage_contract("draft", "write_draft")
    calls = []

    class Model:
        async def invoke(self, **kwargs):
            calls.append(kwargs)
            metadata = {
                "stage": contract.stage, "operation": contract.operation,
                "from_member": contract.primary_member, "to_member": "research_leader",
                "research_id": "case", "permissions_ack": True,
                "write_scope": list(contract.write_scope),
            }
            return SimpleNamespace(
                content="# Title\n\n## Method\n\nGrounded text.\n\n## Team Handoff\n"
                        + "```json\n" + json.dumps(metadata) + "\n```",
                usage_metadata={"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
            )

    backend = OpenJiuwenTeamBackend(
        config={"team": {"direct_specialist_stages": ["draft"]}}, output_root=tmp_path,
    )
    backend._direct_model = Model()
    runtime = JiuwenSwarmResearchTeamRuntime(backend=backend, research_id="case", output_root=tmp_path)
    manuscript = asyncio.run(runtime.complete_text(
        stage="draft", operation="write_draft", system_prompt="Use verified figures.",
        user_prompt="Write the complete manuscript.",
    ))
    assert manuscript.startswith("# Title")
    assert "## Team Handoff" not in manuscript
    assert calls[0]["max_tokens"] == 12000
    assert backend.last_usage()["total_tokens"] == 300


def test_direct_specialist_stops_on_permanent_provider_error_without_leaking_message(tmp_path):
    class Model:
        async def invoke(self, **kwargs):
            raise RuntimeError("Error code: 402 - Insufficient Balance sk-fixture-secret")

    backend = OpenJiuwenTeamBackend(
        config={"team": {"direct_specialist_stages": ["experiment_plan"]}}, output_root=tmp_path,
    )
    backend._direct_model = Model()
    with pytest.raises(TeamProviderError, match="Provider HTTP 402") as captured:
        asyncio.run(backend.invoke(stage="experiment_plan", operation="design_experiment", query="x"))
    assert "sk-fixture-secret" not in str(captured.value)


def test_stage_timeouts_do_not_expand_short_stages():
    config = {"team": {"invocation_timeout_seconds": 900, "stage_timeout_seconds": {"draft": 1800}}}
    assert _stage_timeout_seconds(config, "draft") == 1800
    assert _stage_timeout_seconds(config, "claim_compiler") == 900


@pytest.mark.parametrize("kind", ["retry", "controller_dict", "controller_model"])
@pytest.mark.parametrize("role", ["leader", "teammate"])
def test_balance_failure_stops_team_at_first_observed_error(tmp_path, monkeypatch, kind, role):
    from pydantic import BaseModel

    message = "[181001] model call failed: APIStatusError: Error code: 402 - Insufficient Balance sk-fixture-secret"
    payload = {"type": "task_failed", "data": [{"type": "text", "text": message}]}
    if kind == "controller_model":
        class Payload(BaseModel):
            type: str
            data: list[dict]
        payload = Payload(**payload)
    failed = (_chunk("llm_output", {"retrying": True, "content": message}, role)
              if kind == "retry" else _chunk("controller_output", payload, role))
    backend, closed, stopped, _ = _backend(tmp_path, monkeypatch, [
        _chunk("llm_usage", {"input_tokens": 100, "output_tokens": 10}),
        failed,
        AssertionError("must not consume subsequent work"),
    ])
    with pytest.raises(TeamProviderError, match="HTTP 402: Insufficient Balance") as caught:
        _invoke(backend)
    assert "sk-fixture" not in str(caught.value)
    assert closed and stopped
    assert backend.last_usage()["total_tokens"] == 110
    log = (tmp_path / "team_events.jsonl").read_text()
    assert "provider_fatal_error" in log and "sk-fixture" not in log


@pytest.mark.parametrize("status", [401, 402, 403, 404])
def test_nonretryable_provider_status_is_recognized(status):
    chunk = _chunk("llm_output", {"retrying": True, "content": f"Error code: {status}"})
    assert f"HTTP {status}" in _provider_failure_message(chunk)


@pytest.mark.parametrize("chunk", [
    _chunk("llm_output", {"content": "Error code: 402"}),
    _chunk("answer", {"content": "Error code: 402"}),
    _chunk("controller_output", {"type": "task_completion", "data": [{"text": "Error code: 402"}]}),
    _chunk("llm_output", {"retrying": True, "content": "Error code: 429"}),
    _chunk("llm_output", {"retrying": True, "content": "Error code: 500"}),
])
def test_prose_success_and_transient_status_do_not_trigger_fatal_error(chunk):
    assert not _provider_failure_message(chunk)


@pytest.mark.parametrize("kind", ["text", "json"])
def test_harness_does_not_retry_fatal_provider_failure(kind):
    from jiuwenswarm.agents.harness.research.runtime import ResearchStageHarness
    calls = []

    async def fail(**kwargs):
        calls.append(kwargs)
        raise TeamProviderError("Provider HTTP 402: Insufficient Balance")

    harness = ResearchStageHarness(SimpleNamespace(complete_text=fail, complete_json=fail), max_retries=3)
    try:
        with pytest.raises(TeamProviderError):
            getattr(harness, f"_retry_{kind}")(
                stage="draft", operation="write_draft", system_prompt="fixture", user_prompt="fixture",
            )
        assert len(calls) == 1
    finally:
        harness.close()


@pytest.mark.parametrize("timeout", [0, -1, float('inf'), float('nan'), True])
def test_stage_deadline_must_be_finite_and_positive(timeout):
    with pytest.raises(ValueError):
        _stage_timeout_seconds({"team": {"stage_timeout_seconds": {"draft": timeout}}}, "draft")


def test_timeout_identifies_stage_and_stops_team_without_losing_usage(tmp_path, monkeypatch):
    backend, closed, stopped, _ = _backend(tmp_path, monkeypatch, [
        _chunk("llm_usage", {"usage_metadata": {"input_tokens": 100, "output_tokens": 10}}),
        TimeoutError(),
    ], team={"invocation_timeout_seconds": 900})
    with pytest.raises(TimeoutError, match=r"claim_compiler.compile_claim.*900s"):
        _invoke(backend)
    assert backend.last_usage()["total_tokens"] == 110
    assert closed and stopped


def test_context_snapshots_are_not_billed_and_numeric_receipts_are_replayable(tmp_path, monkeypatch):
    chunks = [
        _chunk("context.usage", {"context_window": {"input_tokens": 100},
                                "kv_cache": {"request": {"input_tokens": 100}}}),
        _chunk("llm_usage", {"usage_metadata": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}}),
        _chunk("context.usage", {"context_window": {"input_tokens": 200},
                                "kv_cache": {"session": {"input_tokens": 999999}}}),
        _chunk("llm_usage", {"usage_metadata": {"input_tokens": 200, "output_tokens": 20, "total_tokens": 220},
                             "api_key": "must-not-be-written"}),
        _chunk("answer", {"content": '{"result":{"ok":true}}'}),
    ]
    backend, closed, stopped, _ = _backend(tmp_path, monkeypatch, chunks)
    assert json.loads(_invoke(backend))["result"]["ok"]
    usage = backend.last_usage()
    assert usage["prompt_tokens"] == 300
    assert usage["completion_tokens"] == 30
    assert usage["total_tokens"] == 330
    assert usage["provider_calls"] == 2
    assert usage["missing_usage_calls"] == 0
    log = (tmp_path / "provider_usage.jsonl").read_text()
    assert "must-not-be-written" not in log
    assert sum(json.loads(line)["total_tokens"] for line in log.splitlines()) == 330
    assert closed == [True]
    assert not stopped


@pytest.mark.parametrize("value", [
    None,
    {"context_window": {"input_tokens": 100}},
    {"input_tokens": -1, "output_tokens": 2},
    {"input_tokens": True, "output_tokens": 2},
    {"input_tokens": "garbage", "output_tokens": 2},
    {"input_tokens": float("nan"), "output_tokens": 2},
    {"input_tokens": 1.5, "output_tokens": 2},
    {"input_tokens": 5, "output_tokens": 2, "total_tokens": 100},
    {"input_tokens": 5},
])
def test_invalid_or_estimated_usage_is_unknown_not_a_fabricated_count(value):
    assert _usage_from_value(value) == {}


def test_provider_aliases_and_explicit_zero():
    assert _usage_from_value({"usage": {"prompt_tokens": "10", "completion_tokens": 0}}) == {
        "prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10,
    }


def test_invalid_usage_receipt_is_flagged(tmp_path, monkeypatch):
    backend, *_ = _backend(tmp_path, monkeypatch, [
        _chunk("llm_usage", {"usage_metadata": {"input_tokens": "bad"}}),
        _chunk("answer", {"content": '{"result":{}}'}),
    ])
    _invoke(backend)
    assert backend.last_usage()["missing_usage_calls"] == 1
    assert backend.last_usage()["provider_calls"] == 1
    assert backend.last_usage()["total_tokens"] == 0
    record = json.loads((tmp_path / "provider_usage.jsonl").read_text())
    assert record["usage_status"] == "missing_or_invalid"


@pytest.mark.parametrize("budget", [
    {"max_invocation_total_tokens": 110}, {"max_invocation_provider_calls": 1},
])
def test_budget_stops_stream_and_native_team_and_blocks_retry(tmp_path, monkeypatch, budget):
    backend, closed, stopped, started = _backend(tmp_path, monkeypatch, [
        _chunk("llm_usage", {"usage_metadata": {"input_tokens": 100, "output_tokens": 10}}),
        _chunk("llm_usage", {"usage_metadata": {"input_tokens": 900, "output_tokens": 90}}),
        _chunk("answer", {"content": '{"result":{}}'}),
    ], team=budget)
    with pytest.raises(TeamResourceBudgetExceeded, match="budget reached"):
        _invoke(backend)
    assert backend.last_usage()["total_tokens"] == 110
    assert backend.last_usage()["provider_calls"] == 1
    assert closed == [True]
    assert stopped == [{"team_name": "fixture_team", "session_id": started[0]}]
    with pytest.raises(TeamResourceBudgetExceeded, match="retry is blocked"):
        _invoke(backend)
    assert len(started) == 1
    assert backend.last_usage()["total_tokens"] == 0  # never bill the previous call twice


def test_streamed_tokens_are_neither_persisted_nor_used_as_final_answer(tmp_path, monkeypatch):
    backend, closed, _, _ = _backend(tmp_path, monkeypatch, [
        *[_chunk("llm_reasoning", {"content": "private reasoning"}) for _ in range(1000)],
        *[_chunk("llm_output", {"content": "token"}) for _ in range(1000)],
        _chunk("answer", {"content": '{"result":{}}'}),
    ])
    assert _invoke(backend) == '{"result":{}}'
    records = (tmp_path / "team_events.jsonl").read_text().splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["stream_chunk_count"] == 2001
    assert closed == [True]


def test_native_missing_usage_never_falls_back_to_process_global_tracker(tmp_path, monkeypatch):
    backend, *_ = _backend(tmp_path, monkeypatch, [])
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.research.team_runtime.get_llm_token_usage_summary",
        lambda: {"total": {"total_tokens": 123456}},
    )
    runtime = JiuwenSwarmResearchTeamRuntime(backend=backend, research_id="fixture")
    usage = runtime._capture_usage(stage="review", operation="review_draft")
    assert usage["total_tokens"] == 0
    assert usage["usage_source"] == "native_provider_llm_usage"


def test_token_guard_fails_closed_on_unusable_provider_receipt(tmp_path, monkeypatch):
    backend, closed, stopped, _ = _backend(tmp_path, monkeypatch, [
        _chunk("llm_usage", {"usage_metadata": {"input_tokens": "missing"}}),
    ], team={"max_invocation_total_tokens": 1000})
    with pytest.raises(TeamResourceBudgetExceeded, match="cannot be enforced"):
        _invoke(backend)
    assert backend.last_usage()["usage_status"] == "missing_or_invalid"
    assert closed == [True]
    assert len(stopped) == 1


def test_restarted_backends_do_not_reuse_persistent_session_ids(tmp_path, monkeypatch):
    chunks = [_chunk("answer", {"content": '{"result":{}}'})]
    first, *_ = _backend(tmp_path, monkeypatch, chunks)
    second, *_ = _backend(tmp_path, monkeypatch, chunks)
    assert first._session_id != second._session_id


def test_failed_native_cleanup_preserves_budget_exception_and_receipt(tmp_path, monkeypatch):
    backend, closed, _, _ = _backend(tmp_path, monkeypatch, [
        _chunk("llm_usage", {"usage_metadata": {"input_tokens": 100, "output_tokens": 10}}),
    ], team={"max_invocation_provider_calls": 1})

    async def failing_stop(**kwargs):
        raise RuntimeError("cleanup failed with private details")

    monkeypatch.setattr(sys.modules["openjiuwen.core.runner"].Runner, "stop_agent_team", failing_stop)
    with pytest.raises(TeamResourceBudgetExceeded):
        _invoke(backend)
    assert backend.last_usage()["total_tokens"] == 110
    assert closed == [True]
    log = (tmp_path / "team_events.jsonl").read_text()
    assert "team_cleanup_error" in log
    assert "private details" not in log


@pytest.mark.parametrize("failure", [RuntimeError("provider disconnected"), asyncio.CancelledError()])
def test_all_stream_failures_stop_native_team_and_keep_partial_usage(tmp_path, monkeypatch, failure):
    backend, closed, stopped, started = _backend(tmp_path, monkeypatch, [
        _chunk("llm_usage", {"usage_metadata": {"input_tokens": 100, "output_tokens": 10}}),
        failure,
    ])
    with pytest.raises(type(failure)) as caught:
        _invoke(backend)
    assert caught.value is failure
    assert backend.last_usage()["total_tokens"] == 110
    assert backend.last_usage()["provider_calls"] == 1
    assert closed == [True]
    assert stopped == [{"team_name": "fixture_team", "session_id": started[0]}]


def test_failed_cleanup_does_not_replace_provider_error(tmp_path, monkeypatch):
    failure = RuntimeError("provider disconnected")
    backend, closed, *_ = _backend(tmp_path, monkeypatch, [
        _chunk("llm_usage", {"usage_metadata": {"input_tokens": 100, "output_tokens": 10}}),
        failure,
    ])

    async def failing_stop(**kwargs):
        raise OSError("private cleanup details")

    monkeypatch.setattr(sys.modules["openjiuwen.core.runner"].Runner, "stop_agent_team", failing_stop)
    with pytest.raises(RuntimeError) as caught:
        _invoke(backend)
    assert caught.value is failure
    assert closed == [True]
    assert backend.last_usage()["total_tokens"] == 110
    log = (tmp_path / "team_events.jsonl").read_text()
    assert "team_cleanup_error" in log
    assert "private cleanup details" not in log


@pytest.mark.parametrize("kind", ["json", "text"])
def test_success_log_failure_never_double_counts_provider_receipts(tmp_path, monkeypatch, kind):
    """A failed audit write must not turn one billed call into two receipts."""
    backend = DeterministicTeamBackend(config={}, output_root=tmp_path, research_id="fixture")
    monkeypatch.setattr(backend, "last_usage", lambda: {
        "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
        "provider_calls": 1, "missing_usage_calls": 0,
        "usage_source": "fixture_provider_receipt", "usage_status": "observed_receipts",
    }, raising=False)
    runtime = JiuwenSwarmResearchTeamRuntime(backend=backend, research_id="fixture")
    log_attempts = []
    failure = OSError("audit storage unavailable")

    def fail_log(**kwargs):
        log_attempts.append(kwargs["status"])
        raise failure

    monkeypatch.setattr(runtime, "_record_call", fail_log)
    if kind == "json":
        invocation = runtime.complete_json(stage="topic_intake", operation="research_question",
                                           system_prompt="fixture", user_prompt="fixture")
    else:
        invocation = runtime.complete_text(stage="draft", operation="write_draft",
                                           system_prompt="fixture", user_prompt="{}")
    with pytest.raises(OSError) as caught:
        asyncio.run(invocation)
    assert caught.value is failure
    summary = runtime.usage_summary()
    assert summary["total"]["total_tokens"] == 110
    assert summary["total"]["provider_calls"] == 1
    assert summary["total"]["request_count"] == 1
    assert len(summary["records"]) == 1
    assert log_attempts == ["success"]


def test_resource_report_and_resume_keep_provider_scope(tmp_path):
    import time
    from examples.scitrace_research.workflow.pipeline import _durable_usage_summary, _fill_resource_snapshot
    from jiuwenswarm.agents.harness.research.reporting.resource_reporter import ResourceAuditSnapshot

    records = [
        {"stage": "draft", "model_name": "fixture", "prompt_tokens": 100, "completion_tokens": 10,
         "total_tokens": 110, "provider_calls": 3, "missing_usage_calls": 0,
         "usage_source": "native_provider_llm_usage", "usage_status": "observed_receipts"},
        {"stage": "review", "model_name": "fixture", "prompt_tokens": 20, "completion_tokens": 5,
         "total_tokens": 25, "provider_calls": 2, "missing_usage_calls": 1,
         "usage_source": "native_provider_llm_usage", "usage_status": "missing_or_invalid"},
    ]
    path = tmp_path / "model_calls.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    summary = _durable_usage_summary(path, SimpleNamespace())
    snapshot = ResourceAuditSnapshot(session_id="offline")
    _fill_resource_snapshot(snapshot, usage_summary=summary, topic="test", claim_count=0,
                            evidence_count=0, artifact_count=0, started=time.perf_counter())
    assert summary["total"]["provider_calls"] == 5
    assert snapshot.model_calls == 2
    assert snapshot.extra["observed_provider_calls"] == 5
    assert snapshot.extra["missing_usage_calls"] == 1
    assert snapshot.extra["usage_sources"] == ["native_provider_llm_usage"]
    assert snapshot.extra["legacy_usage_records"] == 0

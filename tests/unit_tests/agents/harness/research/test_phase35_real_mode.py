from __future__ import annotations

import json
from pathlib import Path
import asyncio

import pytest

from ._stubs import install_openjiuwen_stubs

install_openjiuwen_stubs()

from examples.scitrace_research.workflow import pipeline as pipeline_module
from examples.scitrace_research.workflow.pipeline import WorkflowInterrupted, load_yaml_config, run_pipeline
from jiuwenswarm.agents.harness.research.model_diagnostics import (
    _exception_chain,
    _suggestions,
    diagnose_model_connection,
    resolve_model_config_diagnostics,
    run_level1_sdk_probe,
)
from jiuwenswarm.agents.harness.research.provenance.models import LiteratureEvidence
from jiuwenswarm.agents.harness.research.reporting.resource_reporter import ResourceAuditSnapshot
from jiuwenswarm.agents.harness.research.rails import ScientificProvenanceRail
from jiuwenswarm.agents.harness.research.run_store import ResearchRunStore
from jiuwenswarm.agents.harness.research.runtime import (
    JiuwenSwarmResearchModelRuntime,
    MockResearchModelRuntime,
)


def _config() -> dict:
    path = (
        Path(__file__).resolve().parents[5]
        / "examples"
        / "scitrace_research"
        / "configs"
        / "mvp.yaml"
    )
    return load_yaml_config(path)


def _diagnostics_payload() -> dict:
    return {
        "resolved_model": {
            "provider": "OpenAI",
            "model_name": "test-model",
            "api_base": "https://example.test/v1",
            "api_key": {"configured": True},
        },
        "level1_sdk_call": {
            "sync_client": {"success": True, "error_chain": []},
            "async_client": {"success": True, "error_chain": []},
        },
        "level2_jiuwenswarm_call": {"success": True, "error_chain": []},
    }


def test_strict_real_fails_before_team_when_diagnostics_report_insufficient_balance(monkeypatch, tmp_path) -> None:
    failed = _diagnostics_payload()
    for result in (
        failed["level1_sdk_call"]["sync_client"],
        failed["level1_sdk_call"]["async_client"],
        failed["level2_jiuwenswarm_call"],
    ):
        result.update({
            "success": False,
            "error_chain": [{"type": "APIStatusError", "message": "Insufficient Balance", "status_code": 402}],
        })
    monkeypatch.setattr(pipeline_module, "write_model_diagnostics", lambda path: failed)
    runtime_built = False

    def unexpected_runtime(*args, **kwargs):
        nonlocal runtime_built
        runtime_built = True
        raise AssertionError("runtime must not be built after a failed diagnostic")

    monkeypatch.setattr(pipeline_module, "build_runtime", unexpected_runtime)
    output = tmp_path / "insufficient_balance"
    with pytest.raises(RuntimeError, match="insufficient balance"):
        run_pipeline(
            topic="topic",
            config=_config(),
            output_dir=output,
            mode="real_team",
            strict_real=True,
            no_mock_fallback=True,
        )
    assert runtime_built is False
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["failure_stage"] == "runtime_init"
    assert manifest["model_call_count"] == 0


class _RealLiteratureProvider:
    def search(self, topic: str):
        _ = topic
        return [
            type(
                "RealResult",
                (),
                {
                    "title": "Real Metadata",
                    "authors": ("Alice",),
                    "year": 2026,
                    "doi": "10.1000/real",
                    "source_url": "https://example.test/real",
                    "summary": "real metadata",
                    "reliability_score": 0.9,
                    "provider": "openalex",
                    "verified_doi": True,
                    "verified_url": True,
                },
            )()
        ]


def test_env_priority_is_reported_for_blank_defaults(monkeypatch) -> None:
    monkeypatch.setenv("API_BASE", "https://example.test/v1")
    monkeypatch.setenv("API_KEY", "secret-key")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("MODEL_PROVIDER", "OpenAI")
    raw = {
        "models": {
            "defaults": [
                {
                    "model_client_config": {
                        "api_base": "",
                        "api_key": "",
                        "model_name": "",
                        "client_provider": "",
                    },
                    "model_config_obj": {},
                }
            ]
        }
    }
    resolved = {
        "models": {
            "defaults": [
                {
                    "model_client_config": {
                        "api_base": "https://example.test/v1",
                        "api_key": "secret-key",
                        "model_name": "test-model",
                        "client_provider": "OpenAI",
                    },
                    "model_config_obj": {},
                    "is_default": True,
                }
            ]
        }
    }
    monkeypatch.setattr("jiuwenswarm.agents.harness.research.model_diagnostics.get_config_raw", lambda: raw)
    monkeypatch.setattr("jiuwenswarm.agents.harness.research.model_diagnostics.get_config", lambda: resolved)
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.research.model_diagnostics.get_default_models",
        lambda config=None: resolved["models"]["defaults"],
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.research.model_diagnostics.LLMConfig.from_default_model",
        lambda: type(
            "FakeLLMConfig",
            (),
            {
                "model": "test-model",
                "model_client_config": resolved["models"]["defaults"][0]["model_client_config"],
            },
        )(),
    )

    payload = resolve_model_config_diagnostics()

    assert payload["sources"]["api_base"] == "environment"
    assert payload["sources"]["api_key"] == "environment"
    assert payload["resolved"]["model_name"] == "test-model"


def test_api_key_is_redacted_from_diagnostics(monkeypatch) -> None:
    secret = "super-secret-key"
    monkeypatch.setenv("API_KEY", secret)
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.research.model_diagnostics.resolve_model_config_diagnostics",
        lambda: {
            "resolved": {
                "provider": "OpenAI",
                "model_name": "test-model",
                "api_base": "https://example.test/v1",
                "api_key_configured": True,
            },
            "timeout_seconds": 10.0,
            "verify_ssl": True,
            "env_presence": {"API_KEY": True},
            "sources": {"api_key": "environment"},
        },
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.research.model_diagnostics.collect_dns_tcp_tls",
        lambda *args, **kwargs: {"api_base_parts": {"path": "/v1"}, "dns": {"success": True}, "tcp": {"success": True}, "tls": {"success": True}},
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.research.model_diagnostics.collect_proxy_info",
        lambda api_base: {"selected_proxy": "", "detected_proxy_env": {}},
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.research.model_diagnostics.run_level1_sdk_probe",
        lambda *args, **kwargs: {
            "provider": "OpenAI",
            "sync_client": {"success": False, "content": "", "error_chain": [{"type": "RuntimeError", "message": f"bad {secret}", "status_code": None}]},
            "async_client": {"success": False, "content": "", "error_chain": []},
        },
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.research.model_diagnostics.run_level2_jiuwenswarm_probe",
        lambda **kwargs: {"success": False, "content": "", "error_chain": [{"type": "RuntimeError", "message": f"bad {secret}", "status_code": None}]},
    )

    payload = diagnose_model_connection()
    text = json.dumps(payload, ensure_ascii=False)

    assert secret not in text


def test_strict_real_forbids_fixture_fallback(monkeypatch, tmp_path) -> None:
    provenance = ScientificProvenanceRail(enabled=True, output_root=tmp_path / "out", capture_tool_audits=False)
    resource_snapshot = ResourceAuditSnapshot(session_id="")

    class FakeProvider:
        def search(self, topic: str):
            _ = topic
            return [
                type(
                    "FakeResult",
                    (),
                    {
                        "title": "Fixture",
                        "authors": ("Alice",),
                        "year": 2026,
                        "doi": "10.1/x",
                        "source_url": "https://example.test/paper",
                        "summary": "fixture",
                        "reliability_score": 1.0,
                        "provider": "fixture",
                        "verified_doi": True,
                        "verified_url": True,
                    },
                )()
            ]

    monkeypatch.setattr(pipeline_module, "_build_literature_provider", lambda *args, **kwargs: FakeProvider())

    with pytest.raises(RuntimeError, match="strict-real forbids literature fixture/mock fallback"):
        pipeline_module._run_literature_research(
            topic="topic",
            config={
                "literature": {
                    "provider": "auto",
                    "fixture_path": "unused",
                }
            },
            evidence_dir=tmp_path / "evidence",
            provenance=provenance,
            resource_snapshot=resource_snapshot,
            strict_real=True,
            no_mock_fallback=True,
        )


def test_provider_model_mismatch_suggestion_present() -> None:
    suggestions = _suggestions(
        {
            "resolved_model": {
                "provider": "OpenAI",
                "api_base": "https://api.anthropic.com/v1",
            },
            "config_resolution": {"sources": {"api_key": "yaml"}, "env_presence": {"API_KEY": True}},
            "network": {
                "api_base_parts": {"scheme": "https", "path": "/v1"},
                "dns": {"success": True},
                "tcp": {"success": True},
                "tls": {"success": True},
            },
            "proxy": {"selected_proxy": ""},
            "level1_sdk_call": {"sync_client": {"success": False}, "async_client": {"success": False}},
            "level2_jiuwenswarm_call": {"success": False},
        }
    )

    assert any("provider" in item.lower() and "anthropic" in item.lower() for item in suggestions)


def test_level1_success_level2_failure_is_localized() -> None:
    suggestions = _suggestions(
        {
            "resolved_model": {
                "provider": "OpenAI",
                "api_base": "https://example.test/v1",
            },
            "config_resolution": {"sources": {"api_key": "yaml"}, "env_presence": {"API_KEY": True}},
            "network": {
                "api_base_parts": {"scheme": "https", "path": "/v1"},
                "dns": {"success": True},
                "tcp": {"success": True},
                "tls": {"success": True},
            },
            "proxy": {"selected_proxy": ""},
            "level1_sdk_call": {"sync_client": {"success": True}, "async_client": {"success": True}},
            "level2_jiuwenswarm_call": {"success": False},
        }
    )

    assert any("Level 1 成功而 Level 2 失败" in item for item in suggestions)


def test_level1_treats_deepseek_as_openai_compatible(monkeypatch) -> None:
    calls: list[str] = []

    def _sync(client_config, model_name, timeout_seconds):
        calls.append(f"sync:{client_config['client_provider']}:{model_name}:{int(timeout_seconds)}")
        return {"success": True, "content": "pong", "error_chain": []}

    async def _async(client_config, model_name, timeout_seconds):
        calls.append(f"async:{client_config['client_provider']}:{model_name}:{int(timeout_seconds)}")
        return {"success": True, "content": "pong", "error_chain": []}

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.research.model_diagnostics._run_openai_level1_sync",
        _sync,
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.research.model_diagnostics._run_openai_level1_async",
        _async,
    )

    payload = run_level1_sdk_probe(
        {
            "client_provider": "DeepSeek",
            "api_base": "https://api.deepseek.com",
            "api_key": "key",
        },
        "deepseek-v4-flash",
        timeout_seconds=15.0,
    )

    assert payload["sync_client"]["success"] is True
    assert payload["async_client"]["success"] is True
    assert calls == [
        "sync:DeepSeek:deepseek-v4-flash:15",
        "async:DeepSeek:deepseek-v4-flash:15",
    ]


def test_network_failure_saves_failed_checkpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pipeline_module, "write_model_diagnostics", lambda path: _diagnostics_payload())

    class FailingRuntime(MockResearchModelRuntime):
        async def complete_json(self, **kwargs):
            raise RuntimeError("network down")

        def runtime_info(self) -> dict[str, Any]:
            return {"backend": "jiuwenswarm", "provider": "OpenAI", "model_name": "test-model", "api_base": "https://example.test/v1"}

    monkeypatch.setattr(
        pipeline_module,
        "build_runtime",
        lambda mode, **kwargs: FailingRuntime(call_log_path=kwargs.get("call_log_path")),
    )

    with pytest.raises(RuntimeError, match="network down"):
        run_pipeline(
            topic="topic",
            config=_config(),
            output_dir=tmp_path / "real_fail",
            mode="real",
            strict_real=True,
            no_mock_fallback=True,
        )

    store = ResearchRunStore(tmp_path / "real_fail")
    stage = store.get_stage("topic_intake")
    assert stage is not None
    assert stage.status == "failed"


def test_resume_does_not_repeat_completed_real_stages(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pipeline_module, "write_model_diagnostics", lambda path: _diagnostics_payload())
    monkeypatch.setattr(pipeline_module, "_build_literature_provider", lambda *args, **kwargs: _RealLiteratureProvider())
    calls: list[str] = []

    class CountingRuntime(MockResearchModelRuntime):
        def runtime_info(self) -> dict[str, Any]:
            return {"backend": "jiuwenswarm", "provider": "OpenAI", "model_name": "test-model", "api_base": "https://example.test/v1"}

        async def complete_json(self, **kwargs):
            calls.append(f"{kwargs['stage']}.{kwargs['operation']}")
            return await super().complete_json(**kwargs)

        async def complete_text(self, **kwargs):
            calls.append(f"{kwargs['stage']}.{kwargs['operation']}")
            return await super().complete_text(**kwargs)

    monkeypatch.setattr(
        pipeline_module,
        "build_runtime",
        lambda mode, **kwargs: CountingRuntime(call_log_path=kwargs.get("call_log_path")),
    )

    config = _config()
    config.setdefault("literature", {})
    config["literature"]["provider"] = "auto"
    config.setdefault("debug", {})
    config["debug"]["stop_after_stage"] = "literature_research"

    with pytest.raises(WorkflowInterrupted):
        run_pipeline(
            topic="topic",
            config=config,
            output_dir=tmp_path / "resume_real",
            mode="real",
            strict_real=True,
            no_mock_fallback=True,
        )

    resumed = _config()
    resumed.setdefault("literature", {})
    resumed["literature"]["provider"] = "auto"
    monkeypatch.setattr(
        pipeline_module,
        "build_runtime",
        lambda mode, **kwargs: CountingRuntime(call_log_path=kwargs.get("call_log_path")),
    )
    run_pipeline(
        topic="topic",
        config=resumed,
        output_dir=tmp_path / "resume_real",
        mode="real",
        resume=True,
        strict_real=True,
        no_mock_fallback=True,
    )

    assert calls.count("topic_intake.research_question") == 1


def test_real_manifest_fields_are_complete(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pipeline_module, "write_model_diagnostics", lambda path: _diagnostics_payload())
    monkeypatch.setattr(pipeline_module, "_build_literature_provider", lambda *args, **kwargs: _RealLiteratureProvider())
    monkeypatch.setattr(
        pipeline_module,
        "build_runtime",
        lambda mode, **kwargs: MockResearchModelRuntime(call_log_path=kwargs.get("call_log_path")),
    )

    config = _config()
    config.setdefault("literature", {})
    config["literature"]["provider"] = "auto"
    run_pipeline(
        topic="topic",
        config=config,
        output_dir=tmp_path / "manifest_case",
        mode="real",
        strict_real=True,
        no_mock_fallback=True,
    )

    manifest = json.loads((tmp_path / "manifest_case" / "run_manifest.json").read_text(encoding="utf-8"))
    for key in (
        "run_id",
        "mode",
        "strict_real",
        "provider",
        "model_name",
        "literature_provider",
        "mock_fallback_used",
        "fixture_used",
        "model_call_count",
        "retrieval_call_count",
        "start_time",
        "end_time",
        "status",
        "failure_stage",
    ):
        assert key in manifest


def test_model_call_counter_excludes_diagnostics_entries(tmp_path) -> None:
    path = tmp_path / "model_calls.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"stage": "diagnostics", "operation": "level1_sync"}),
                json.dumps({"stage": "diagnostics", "operation": "level2"}),
                json.dumps({"stage": "topic_intake", "operation": "research_question"}),
                json.dumps({"stage": "draft", "operation": "write_draft"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert pipeline_module._count_model_call_records(path, include_diagnostics=True) == 4
    assert pipeline_module._count_model_call_records(path, include_diagnostics=False) == 2


def test_real_runtime_records_usage_via_response_observer(monkeypatch, tmp_path) -> None:
    observed: list[tuple[str, str]] = []

    def _observer_factory(_config):
        def _observe(response, stage, operation):
            _ = response
            observed.append((stage, operation))
        return _observe

    class _FakeModel:
        async def invoke(self, **kwargs):
            _ = kwargs
            return {"content": "pong"}

    class _FakeConfig:
        model = "test-model"
        model_client_config = {"client_provider": "OpenAI", "api_base": "https://example.test/v1"}

        def create_model(self):
            return _FakeModel()

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.research.runtime.create_model_response_observer",
        _observer_factory,
    )

    runtime = JiuwenSwarmResearchModelRuntime(_FakeConfig(), call_log_path=tmp_path / "calls.jsonl")
    content = asyncio.run(
        runtime.complete_text(
            stage="draft",
            operation="write_draft",
            system_prompt="Return pong.",
            user_prompt="ping",
        )
    )

    assert content == "pong"
    assert observed == [("draft", "write_draft")]


def test_async_root_cause_chain_is_preserved() -> None:
    try:
        try:
            raise OSError("socket closed")
        except OSError as exc:
            raise RuntimeError("async invoke failed") from exc
    except RuntimeError as exc:
        chain = _exception_chain(exc)

    assert chain[0]["type"] == "RuntimeError"
    assert chain[1]["type"] == "OSError"


def test_token_or_call_budget_exceeded_stops_safely(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        pipeline_module,
        "build_runtime",
        lambda mode, **kwargs: MockResearchModelRuntime(call_log_path=kwargs.get("call_log_path")),
    )
    config = _config()
    config["budget"] = {"max_model_calls": 0}

    with pytest.raises(RuntimeError, match="model call budget exceeded"):
        run_pipeline(
            topic="topic",
            config=config,
            output_dir=tmp_path / "budget_case",
            mode="mock",
        )

    manifest = json.loads((tmp_path / "budget_case" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure_stage"] == "topic_intake"
    store = ResearchRunStore(tmp_path / "budget_case")
    checkpoint = store.get_stage("topic_intake")
    assert checkpoint is not None
    assert checkpoint.status == "completed"

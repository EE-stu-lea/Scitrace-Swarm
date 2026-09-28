from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol

from jiuwenswarm.agents.harness.research.team_contract import (
    TEAM_TEMPLATE_ID,
    build_team_query,
    get_team_stage_contract,
    validate_team_handoff_metadata,
)
from jiuwenswarm.symphony.llm import get_llm_token_usage_summary, reset_llm_token_usage


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _extract_top_level_json(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.split("```json", 1)[1]
        cleaned = cleaned.split("```", 1)[0]
        cleaned = cleaned.strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Native Team implementations may prefix a coordination sentence or
        # suffix status text even when instructed to return JSON only. Decode
        # the first complete object instead of keeping the stream alive.
        start = cleaned.find("{")
        if start < 0:
            raise
        payload, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(payload, dict):
        raise ValueError("team runtime must return a top-level JSON object")
    return payload


def _extract_markdown_with_handoff(text: str) -> tuple[str, dict[str, Any]]:
    marker = "## Team Handoff"
    if marker not in text:
        raise ValueError("team runtime markdown response is missing `## Team Handoff`")
    body, tail = text.split(marker, 1)
    if "```json" not in tail:
        raise ValueError("team handoff block must contain fenced JSON")
    raw = tail.split("```json", 1)[1].split("```", 1)[0].strip()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("team handoff metadata must be a JSON object")
    return body.rstrip(), payload


def _chunk_role(chunk: Any) -> str:
    role = getattr(chunk, "role", None)
    value = getattr(role, "value", role)
    return str(value or "").strip().lower()


def _chunk_text(chunk: Any) -> str:
    payload = getattr(chunk, "payload", None)
    if isinstance(payload, dict):
        output = payload.get("output")
        if isinstance(output, dict):
            text = output.get("output") or output.get("content") or output.get("text")
            if isinstance(text, str) and text.strip():
                return text
        if isinstance(output, str) and output.strip():
            return output
        for key in ("content", "response_content", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    content = getattr(chunk, "content", None)
    return content if isinstance(content, str) else ""


def _is_terminal_answer_chunk(chunk: Any) -> bool:
    chunk_type = str(getattr(chunk, "type", "") or "").strip().lower()
    return chunk_type == "answer"


def _usage_from_value(value: Any) -> dict[str, int]:
    """Best-effort extraction for OpenAI/Anthropic-style usage payloads."""
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump()
        except Exception:
            pass
    if not isinstance(value, dict):
        value = getattr(value, "__dict__", {})
    if not isinstance(value, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = int(value.get("prompt_tokens", value.get("input_tokens", 0)) or 0)
    completion = int(value.get("completion_tokens", value.get("output_tokens", 0)) or 0)
    total = int(value.get("total_tokens", prompt + completion) or (prompt + completion))
    best = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
    for child in value.values():
        candidate = _usage_from_value(child)
        if candidate["total_tokens"] > best["total_tokens"]:
            best = candidate
    return best


def _looks_like_terminal_result(*, text: str, result_kind: str) -> bool:
    if not text.strip():
        return False
    if result_kind == "json":
        try:
            _extract_top_level_json(text)
            return True
        except Exception:
            return False
    if result_kind == "markdown":
        return "## Team Handoff" in text and "```json" in text
    return False


def _resolved_default_model(config: dict[str, Any]) -> dict[str, str]:
    from jiuwenswarm.common.config import get_default_models, resolve_env_vars

    resolved = resolve_env_vars(config)
    defaults = get_default_models(resolved)
    first = defaults[0] if defaults else {}
    client = first.get("model_client_config") or {}
    return {
        "provider": str(client.get("client_provider") or "").strip(),
        "model_name": str(client.get("model_name") or "").strip(),
        "api_base": str(client.get("api_base") or "").strip().rstrip("/"),
    }


class TeamInvocationBackend(Protocol):
    async def invoke(self, *, stage: str, operation: str, query: str) -> str:
        raise NotImplementedError

    def runtime_info(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class DeterministicTeamBackend:
    """Mock backend for Phase 4 Team integration tests."""

    config: dict[str, Any]
    output_root: Path
    research_id: str | None = None

    async def invoke(self, *, stage: str, operation: str, query: str) -> str:
        _append_jsonl(
            self.output_root / "team_events.jsonl",
            {
                "timestamp": _utc_now_iso(),
                "backend": "mock_team",
                "stage": stage,
                "operation": operation,
                "query_preview": query[:200],
            },
        )
        contract = get_team_stage_contract(stage, operation)
        research_id = str(self.research_id or self.output_root.name)
        if (stage, operation) == ("topic_intake", "research_question"):
            return json.dumps(
                {
                    "stage": stage,
                    "operation": operation,
                    "from_member": contract.primary_member,
                    "to_member": "research_leader",
                    "research_id": research_id,
                    "permissions_ack": True,
                    "write_scope": list(contract.write_scope),
                    "result": {
                        "research_question": "Evidence-aware context governance for scientific research agents",
                        "hypothesis": "Evidence-aware context governance improves claim grounding while preserving conservative reporting.",
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        if (stage, operation) == ("claim_compiler", "compile_claim"):
            return json.dumps(
                {
                    "stage": stage,
                    "operation": operation,
                    "from_member": contract.primary_member,
                    "to_member": "research_leader",
                    "research_id": research_id,
                    "permissions_ack": True,
                    "write_scope": list(contract.write_scope),
                    "result": {
                        "text": "Evidence-aware context governance improves claim grounding for autonomous research agents.",
                        "confidence": 0.83,
                        "section": "results",
                        "compiler_source": "team_mock",
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        if (stage, operation) == ("experiment_plan", "design_experiment"):
            return json.dumps(
                {
                    "stage": stage,
                    "operation": operation,
                    "from_member": contract.primary_member,
                    "to_member": "research_leader",
                    "research_id": research_id,
                    "permissions_ack": True,
                    "write_scope": list(contract.write_scope),
                    "result": {
                        "hypothesis": "Evidence-aware retrieval improves top-1 support selection.",
                        "controls": ["fixed benchmark", "deterministic seed"],
                        "metrics": ["citation_support_rate"],
                        "limitations": ["controlled benchmark; external validity is not established"],
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        if (stage, operation) == ("review", "review_draft"):
            return json.dumps(
                {
                    "stage": stage,
                    "operation": operation,
                    "from_member": contract.primary_member,
                    "to_member": "research_leader",
                    "research_id": research_id,
                    "permissions_ack": True,
                    "write_scope": list(contract.write_scope),
                    "result": {
                        "summary": "No critical consistency errors detected.",
                        "severity": "info",
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        if (stage, operation) == ("draft", "write_draft"):
            payload = json.loads(query.split("# Stage User Prompt", 1)[1].strip())
            artifact = payload.get("artifact") or {}
            metric_value = artifact.get("metric_value")
            markdown = "\n".join(
                [
                    "# SciTrace-Swarm Team Draft",
                    "",
                    "## Topic",
                    str(payload.get("topic") or ""),
                    "",
                    "## Results",
                    *[
                        f"- `{claim['id']}` [{claim['status']}] {claim['text']}"
                        for claim in payload.get("claims", [])
                    ],
                    "",
                    "## Experiment Result",
                    f"- `{artifact.get('id', '')}` {artifact.get('metric_name', '')} = {metric_value}",
                    "",
                    "## Structured References",
                    "```json",
                    json.dumps(
                        {
                            "citations": [item["id"] for item in payload.get("evidence", [])],
                            "claims": [{"claim_id": item["id"]} for item in payload.get("claims", [])],
                            "reported_metrics": [
                                {
                                    "artifact_id": artifact.get("id", ""),
                                    "metric_name": artifact.get("metric_name", ""),
                                    "claimed_value": metric_value,
                                }
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    "```",
                    "",
                    "## Team Handoff",
                    "```json",
                    json.dumps(
                        {
                            "stage": stage,
                            "operation": operation,
                            "from_member": contract.primary_member,
                            "to_member": "research_leader",
                            "research_id": research_id,
                            "permissions_ack": True,
                            "write_scope": list(contract.write_scope),
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    "```",
                ]
            )
            return markdown
        if (stage, operation) == ("revision", "targeted_revision"):
            draft = query.split("# Stage User Prompt", 1)[1].strip()
            original = draft.split("# Structured Errors", 1)[0].rstrip()
            return "\n".join(
                [
                    original,
                    "",
                    "## Team Handoff",
                    "```json",
                    json.dumps(
                        {
                            "stage": stage,
                            "operation": operation,
                            "from_member": contract.primary_member,
                            "to_member": "research_leader",
                            "research_id": research_id,
                            "permissions_ack": True,
                            "write_scope": list(contract.write_scope),
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    "```",
                ]
            )
        raise RuntimeError(f"unsupported team mock stage: {stage}.{operation}")

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": "jiuwenswarm-team-mock",
            "provider": "mock_team",
            "model_name": "mock-team-research",
            "api_base": "",
        }


class OpenJiuwenTeamBackend:
    """Real JiuwenSwarm native Team backend."""

    def __init__(self, *, config: dict[str, Any], output_root: Path, project_dir: str | None = None) -> None:
        self._config = config
        self._output_root = output_root
        self._project_dir = project_dir
        self._team_spec = None
        self._session_counter = 0
        self._session_id = f"scitrace-team-{output_root.name}"
        self._last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._runtime_info = {
            "backend": "jiuwenswarm-team",
            **_resolved_default_model(config),
        }

    def runtime_info(self) -> dict[str, Any]:
        return dict(self._runtime_info)

    def last_usage(self) -> dict[str, int]:
        return dict(self._last_usage)

    def _ensure_team_spec(self):
        if self._team_spec is not None:
            return self._team_spec
        from openjiuwen.agent_teams.schema.blueprint import TeamAgentSpec

        from jiuwenswarm.agents.harness.team.config_loader import load_team_spec_dict
        from jiuwenswarm.agents.swarm import enrich_team_spec_for_swarm
        from jiuwenswarm.common.config import resolve_env_vars

        team_config = resolve_env_vars(self._config)
        template_id = str((team_config.get("team") or {}).get("template_id") or TEAM_TEMPLATE_ID).strip() or TEAM_TEMPLATE_ID
        spec_dict = load_team_spec_dict(config_base=team_config, template_id=template_id, strict_template=True)
        spec = TeamAgentSpec.model_validate(spec_dict)
        enrich_team_spec_for_swarm(
            spec,
            session_id=self._session_id,
            mode="team",
            project_dir=self._project_dir,
            request_metadata={"mode": "team"},
        )
        self._team_spec = spec
        return spec

    async def invoke(self, *, stage: str, operation: str, query: str) -> str:
        from openjiuwen.core.runner import Runner

        spec = self._ensure_team_spec()
        contract = get_team_stage_contract(stage, operation)
        # 每个 stage 用独立的 team session（冷启动）。复用已暂停的 persistent
        # session 会触发 RESUME_FROM_PAUSE，但 leader 协调已在前一轮结束，resume
        # 后不会产生新的 leader 输出（空流）。SEG 共享发生在 pipeline 层，不依赖
        # team session，因此每个阶段冷启动一个 team 是安全的。
        self._session_counter += 1
        self._last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        session_id = f"{self._session_id}-{self._session_counter}"
        leader_outputs: list[str] = []
        any_outputs: list[str] = []
        invocation_timeout = float((self._config.get("team") or {}).get("invocation_timeout_seconds", 420) or 420)
        stream_chunks = 0
        async with asyncio.timeout(invocation_timeout):
            async for chunk in Runner.run_agent_team_streaming(
                agent_team=spec,
                inputs={"query": query},
                session=session_id,
            ):
                role = _chunk_role(chunk)
                text = _chunk_text(chunk)
                chunk_type = str(getattr(chunk, "type", "") or "").strip().lower()
                stream_chunks += 1
                if "usage" in chunk_type:
                    usage = _usage_from_value(getattr(chunk, "payload", None))
                    for key in self._last_usage:
                        self._last_usage[key] += usage[key]
                # Per-token llm_output events can create multi-megabyte logs and
                # obscure the handoff. Persist lifecycle events and terminal
                # answers; retain only a count for token chunks.
                if chunk_type != "llm_output":
                    _append_jsonl(
                        self._output_root / "team_events.jsonl",
                        {
                            "timestamp": _utc_now_iso(),
                            "stage": stage,
                            "operation": operation,
                            "type": str(getattr(chunk, "type", "") or ""),
                            "role": role,
                            "stream_chunk_count": stream_chunks,
                            "payload_preview": text[:2000] if chunk_type == "answer" else text[:300],
                        },
                    )
                if not text.strip():
                    continue
                any_outputs.append(text)
                if role == "leader":
                    leader_outputs.append(text)
                    if _is_terminal_answer_chunk(chunk) and _looks_like_terminal_result(
                        text=text,
                        result_kind=contract.result_kind,
                    ):
                        return text.strip()
        final = (leader_outputs[-1] if leader_outputs else (any_outputs[-1] if any_outputs else "")).strip()
        if not final:
            raise RuntimeError(f"native team returned no final text for {stage}.{operation}")
        return final


class JiuwenSwarmResearchTeamRuntime:
    """Team-backed research runtime that keeps SciTrace's stage contracts structured."""

    def __init__(
        self,
        *,
        backend: TeamInvocationBackend,
        research_id: str,
        call_log_path: str | Path | None = None,
        output_root: str | Path | None = None,
    ) -> None:
        self._backend = backend
        self._research_id = research_id
        self._call_log = Path(call_log_path) if call_log_path else None
        self._output_root = Path(output_root) if output_root else None
        self._request_count = 0
        self._usage_records: list[dict[str, Any]] = []

    def _record_call(
        self, *, stage: str, operation: str, status: str,
        usage: dict[str, Any] | None = None, error: Exception | None = None,
    ) -> None:
        info = self.runtime_info()
        usage = usage or {}
        _append_jsonl(
            self._call_log,
            {
                "timestamp": _utc_now_iso(),
                "backend": info.get("backend", "jiuwenswarm-team"),
                "provider": info.get("provider", ""),
                "model_name": info.get("model_name", ""),
                "api_base": info.get("api_base", ""),
                "stage": stage,
                "operation": operation,
                "request_index": self._request_count,
                "status": status,
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
                "error_chain": [] if error is None else [{"type": type(error).__name__, "message": str(error)}],
            },
        )

    def _capture_usage(self, *, stage: str, operation: str) -> dict[str, Any]:
        summary = get_llm_token_usage_summary()
        total = summary.get("total") or {}
        backend_usage = getattr(self._backend, "last_usage", lambda: {})()
        if int(backend_usage.get("total_tokens", 0) or 0) > 0:
            total = backend_usage
        record = {
            "stage": stage,
            "operation": operation,
            "model": self.runtime_info().get("model_name", ""),
            "prompt_tokens": int(total.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(total.get("completion_tokens", 0) or 0),
            "total_tokens": int(total.get("total_tokens", 0) or 0),
        }
        self._usage_records.append(record)
        return record

    async def complete_json(
        self,
        *,
        stage: str,
        operation: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        contract = get_team_stage_contract(stage, operation)
        query = build_team_query(
            contract=contract,
            research_id=self._research_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        self._request_count += 1
        try:
            reset_llm_token_usage()
            raw = await self._backend.invoke(stage=stage, operation=operation, query=query)
            payload = _extract_top_level_json(raw)
            validate_team_handoff_metadata(payload, contract=contract, research_id=self._research_id)
            result = payload.get("result")
            if not isinstance(result, dict):
                raise ValueError("team JSON handoff must contain an object `result`")
            _append_jsonl(
                self._output_root / "team_handoffs.jsonl" if self._output_root else None,
                {"timestamp": _utc_now_iso(), "stage": stage, "operation": operation, "payload": payload},
            )
            usage = self._capture_usage(stage=stage, operation=operation)
            self._record_call(stage=stage, operation=operation, status="success", usage=usage)
            return result
        except Exception as exc:
            usage = self._capture_usage(stage=stage, operation=operation)
            self._record_call(stage=stage, operation=operation, status="error", usage=usage, error=exc)
            raise

    async def complete_text(
        self,
        *,
        stage: str,
        operation: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        contract = get_team_stage_contract(stage, operation)
        query = build_team_query(
            contract=contract,
            research_id=self._research_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        self._request_count += 1
        try:
            reset_llm_token_usage()
            raw = await self._backend.invoke(stage=stage, operation=operation, query=query)
            markdown, metadata = _extract_markdown_with_handoff(raw)
            validate_team_handoff_metadata(metadata, contract=contract, research_id=self._research_id)
            _append_jsonl(
                self._output_root / "team_handoffs.jsonl" if self._output_root else None,
                {"timestamp": _utc_now_iso(), "stage": stage, "operation": operation, "payload": metadata},
            )
            usage = self._capture_usage(stage=stage, operation=operation)
            self._record_call(stage=stage, operation=operation, status="success", usage=usage)
            return markdown
        except Exception as exc:
            usage = self._capture_usage(stage=stage, operation=operation)
            self._record_call(stage=stage, operation=operation, status="error", usage=usage, error=exc)
            raise

    def usage_summary(self) -> dict[str, Any]:
        prompt_tokens = sum(int(item.get("prompt_tokens", 0) or 0) for item in self._usage_records)
        completion_tokens = sum(int(item.get("completion_tokens", 0) or 0) for item in self._usage_records)
        total_tokens = sum(int(item.get("total_tokens", 0) or 0) for item in self._usage_records)
        return {
            "total": {
                "request_count": self._request_count,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "by_stage": {},
            "by_operation": {},
            "records": list(self._usage_records),
        }

    def runtime_info(self) -> dict[str, Any]:
        return self._backend.runtime_info()

    def call_count(self) -> int:
        return self._request_count


__all__ = [
    "DeterministicTeamBackend",
    "JiuwenSwarmResearchTeamRuntime",
    "OpenJiuwenTeamBackend",
    "TEAM_TEMPLATE_ID",
]

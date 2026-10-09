from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Protocol
from uuid import uuid4

from jiuwenswarm.agents.harness.research.team_contract import (
    TEAM_TEMPLATE_ID,
    TEAM_STAGE_CONTRACTS,
    ROLE_RESEARCH_LEADER,
    build_direct_specialist_query,
    build_team_query,
    get_team_stage_contract,
    validate_team_handoff_metadata,
)
from jiuwenswarm.symphony.llm import get_llm_token_usage_summary, reset_llm_token_usage
from jiuwenswarm.agents.harness.research.stage_policy import (
    apply_stage_capability_profile, prompt_audit, scope_team_spec, stage_resource_limits,
)


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
    """Read a single provider receipt, never search context/session estimates.

    Native core emits both ``context.usage`` and ``llm_usage`` per request.
    Only the latter is a billing receipt; recursive token discovery silently
    counts the same input twice and can mistake estimates for billed tokens.
    """
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        return {}
    value = value.get("usage_metadata", value.get("usage", value))
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        return {}

    def count(key: str, alias: str) -> int | None:
        raw = value.get(key, value.get(alias))
        if isinstance(raw, bool) or raw is None:
            return None
        if isinstance(raw, str) and raw.strip().isdigit():
            return int(raw.strip())
        if isinstance(raw, int) and raw >= 0:
            return raw
        return None

    prompt = count("input_tokens", "prompt_tokens")
    completion = count("output_tokens", "completion_tokens")
    total = count("total_tokens", "total_tokens")
    if prompt is None or completion is None:
        return {}
    if "total_tokens" in value and total is None:
        return {}
    if total is not None and total != prompt + completion:
        return {}
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


class TeamResourceBudgetExceeded(RuntimeError):
    """Observed stream usage reached its configured invocation limit."""


class TeamProviderError(RuntimeError):
    """Provider rejected a request that cannot succeed by automatic retry."""


def _provider_failure_message(chunk: Any) -> str:
    """Inspect framework failure frames, never ordinary model prose.

    Native Team wraps all model errors in 181001 and emits retry-marked
    llm_output before eventually forwarding controller_output. Inspect both
    so an exhausted account does not wait for ten retries or a stage timeout.
    Return only fixed diagnostics; provider text may contain credentials.
    """
    kind = getattr(chunk, "type", "")
    payload = getattr(chunk, "payload", None)

    def field(value: Any, key: str, default: Any = None) -> Any:
        return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)

    text = ""
    if kind == "llm_output" and field(payload, "retrying") is True:
        text = str(field(payload, "content", ""))
    elif kind == "controller_output":
        status = field(payload, "type")
        if getattr(status, "value", status) != "task_failed":
            return ""
        for item in field(payload, "data") or []:
            text += "\n" + str(field(item, "text", ""))
    reasons = {
        "401": "authentication failed; check the API key",
        "402": "Insufficient Balance; replenish the provider account or use a funded API key",
        "403": "access denied; check provider permissions",
        "404": "endpoint or model not found; check the API base and model name",
    }
    match = re.search(r"(?:error code\s*:|HTTP(?: status)?\s*:?|status_code\s*[=:])\s*(401|402|403|404)\b", text, re.I)
    if not match:
        return ""
    code = match.group(1)
    return f"Provider HTTP {code}: {reasons[code]}. Automatic retry stopped; completed checkpoints are retained."


def _direct_provider_failure_message(error: Exception) -> str:
    """Classify only permanent status codes without persisting provider text."""
    text = str(error)
    match = re.search(r"(?:error code\s*:|HTTP(?: status)?\s*:?|status_code\s*[=:])\s*(401|402|403|404)\b", text, re.I)
    if not match:
        return ""
    reasons = {
        "401": "authentication failed; check the API key",
        "402": "Insufficient Balance; replenish the provider account or use a funded API key",
        "403": "access denied; check provider permissions",
        "404": "endpoint or model not found; check the API base and model name",
    }
    code = match.group(1)
    return f"Provider HTTP {code}: {reasons[code]}. Automatic retry stopped; completed checkpoints are retained."


def _stage_timeout_seconds(config: dict[str, Any], stage: str) -> float:
    """Give long manuscript stages their own finite deadline."""
    import math

    team = config.get("team") or {}
    overrides = team.get("stage_timeout_seconds") or {}
    if not isinstance(overrides, dict):
        raise ValueError("team.stage_timeout_seconds must be a mapping")
    value = overrides.get(stage, team.get("invocation_timeout_seconds", 420))
    if isinstance(value, bool):
        raise ValueError("Team stage timeout must be a finite positive number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Team stage timeout must be a finite positive number")
    return timeout


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
                        "findings": [],
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
    """Real Team backend with bounded direct specialist calls when configured."""

    def __init__(self, *, config: dict[str, Any], output_root: Path, project_dir: str | None = None) -> None:
        self._config = config
        self._output_root = output_root
        self._project_dir = project_dir
        self._team_spec = None
        self._stage_specs: dict[tuple[str, str], Any] = {}
        self._active_stage = ""
        self._session_counter = 0
        self._session_id = f"scitrace-team-{output_root.name}-{uuid4().hex[:12]}"
        self._last_usage = self._empty_usage()
        self._budget_exhausted = False
        direct_stages = (config.get("team") or {}).get("direct_specialist_stages", [])
        if not isinstance(direct_stages, list) or any(not isinstance(stage, str) for stage in direct_stages):
            raise ValueError("team.direct_specialist_stages must be a list of stage names")
        allowed = {contract.stage for contract in TEAM_STAGE_CONTRACTS.values()
                   if contract.primary_member != ROLE_RESEARCH_LEADER}
        if len(set(direct_stages)) != len(direct_stages) or set(direct_stages) - allowed:
            raise ValueError("team.direct_specialist_stages contains a duplicate or unsupported stage")
        self._direct_stages = frozenset(direct_stages)
        self._direct_model = None
        self._runtime_info = {
            "backend": "jiuwenswarm-team",
            "stage_scoped_team": bool((config.get("team") or {}).get("stage_scoped_team", False)),
            "stage_dispatch": {stage: ("direct_specialist" if stage in self._direct_stages else "native_team")
                               for stage in sorted({contract.stage for contract in TEAM_STAGE_CONTRACTS.values()})},
            **_resolved_default_model(config),
        }

    def runtime_info(self) -> dict[str, Any]:
        return dict(self._runtime_info)

    @staticmethod
    def _empty_usage() -> dict[str, Any]:
        return {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "provider_calls": 0, "missing_usage_calls": 0,
            "usage_source": "native_provider_llm_usage",
        }

    def last_usage(self) -> dict[str, Any]:
        return {
            **self._last_usage,
            "usage_status": (
                "missing_or_invalid" if self._last_usage["missing_usage_calls"]
                else "observed_receipts" if self._last_usage["provider_calls"]
                else "no_provider_receipts_observed"
            ),
        }

    def uses_direct_stage(self, stage: str) -> bool:
        return stage in self._direct_stages

    def _ensure_direct_model(self):
        if self._direct_model is None:
            from jiuwenswarm.common.config import get_default_models, resolve_env_vars
            from jiuwenswarm.symphony.llm import LLMConfig

            entries = get_default_models(resolve_env_vars(self._config))
            if not entries:
                raise ValueError("direct specialist route requires a configured default model")
            entry = next((item for item in entries if item.get("is_default") is True), entries[0])
            self._direct_model = LLMConfig.from_model_entry(entry).create_model()
        return self._direct_model

    async def _invoke_direct(self, *, stage: str, operation: str, query: str) -> str:
        from jiuwenswarm.symphony.llm import extract_message_content

        contract = get_team_stage_contract(stage, operation)
        self._session_counter += 1
        session_id = f"{self._session_id}-direct-{self._session_counter}"
        try:
            request_options = {"temperature": 0.0, "top_p": 1.0}
            if contract.result_kind == "markdown":
                # A complete ICLR draft can exceed the provider's shorter
                # default response cap. This is a ceiling, not a target.
                request_options["max_tokens"] = 12000
            async with asyncio.timeout(_stage_timeout_seconds(self._config, stage)):
                response = await self._ensure_direct_model().invoke(
                    messages=[
                        {"role": "system", "content": (
                            f"You are the SciTrace {contract.primary_member}. "
                            "Follow the stage's evidence and output contract exactly."
                        )},
                        {"role": "user", "content": query},
                    ],
                    **request_options,
                )
        except TimeoutError as exc:
            raise TimeoutError(
                f"Direct specialist timed out during {stage}.{operation}; "
                "completed workflow checkpoints are retained."
            ) from exc
        except Exception as exc:
            permanent_error = _direct_provider_failure_message(exc)
            if permanent_error:
                raise TeamProviderError(permanent_error) from None
            raise
        usage = getattr(response, "usage", None)
        if usage is None:
            usage = getattr(response, "usage_metadata", None)
        if usage is None:
            usage = getattr(getattr(response, "raw", None), "usage", None)
        receipt = _usage_from_value(usage)
        self._last_usage["usage_source"] = "native_model_response"
        self._last_usage["provider_calls"] = 1
        if receipt:
            for key, value in receipt.items():
                self._last_usage[key] += value
        else:
            self._last_usage["missing_usage_calls"] = 1
        _append_jsonl(self._output_root / "provider_usage.jsonl", {
            "timestamp": _utc_now_iso(), "session_id": session_id,
            "stage": stage, "operation": operation, "role": contract.primary_member,
            "provider_call_index": 1, "usage_source": "native_model_response",
            "usage_status": "reported" if receipt else "missing_or_invalid", **receipt,
        })
        limits = stage_resource_limits(self._config, stage)
        if not receipt and limits.get("max_total_tokens") is not None:
            self._budget_exhausted = True
            raise TeamResourceBudgetExceeded(
                "direct specialist token budget cannot be enforced: provider usage is missing or invalid"
            )
        for option, metric in (("max_total_tokens", "total_tokens"),
                               ("max_provider_calls", "provider_calls")):
            limit = limits.get(option)
            if limit is not None and self._last_usage[metric] > limit:
                self._budget_exhausted = True
                raise TeamResourceBudgetExceeded(
                    f"direct specialist budget exceeded: {metric}={self._last_usage[metric]} > {limit}"
                )
        content = extract_message_content(response).strip()
        if not content:
            raise ValueError(f"direct specialist returned empty text for {stage}.{operation}")
        _append_jsonl(self._output_root / "team_events.jsonl", {
            "timestamp": _utc_now_iso(), "session_id": session_id,
            "stage": stage, "operation": operation, "role": contract.primary_member,
            "type": "direct_specialist_answer", "response_chars": len(content),
        })
        return content

    def _check_stream_budget(self) -> None:
        limits = stage_resource_limits(self._config, self._active_stage)
        for option, metric in (
            ("max_total_tokens", "total_tokens"),
            ("max_provider_calls", "provider_calls"),
        ):
            limit = limits.get(option)
            if limit is None:
                continue
            if metric == "total_tokens" and self._last_usage["missing_usage_calls"]:
                self._budget_exhausted = True
                raise TeamResourceBudgetExceeded("token budget cannot be enforced: provider usage is missing or invalid")
            if self._last_usage[metric] >= limit:
                self._budget_exhausted = True
                raise TeamResourceBudgetExceeded(
                    f"team stream budget reached: {metric}={self._last_usage[metric]} >= {limit}"
                )

    def _ensure_team_spec(self, stage: str, operation: str):
        scoped = bool((self._config.get("team") or {}).get("stage_scoped_team", False))
        key = (stage, operation)
        if scoped and key in self._stage_specs:
            return self._stage_specs[key]
        if not scoped and self._team_spec is not None:
            return self._team_spec
        from openjiuwen.agent_teams.schema.blueprint import TeamAgentSpec

        from jiuwenswarm.agents.harness.team.config_loader import load_team_spec_dict
        from jiuwenswarm.agents.swarm import enrich_team_spec_for_swarm
        from jiuwenswarm.common.config import resolve_env_vars

        team_config = resolve_env_vars(self._config)
        template_id = str((team_config.get("team") or {}).get("template_id") or TEAM_TEMPLATE_ID).strip() or TEAM_TEMPLATE_ID
        spec_dict = load_team_spec_dict(config_base=team_config, template_id=template_id, strict_template=True)
        if scoped:
            spec_dict = scope_team_spec(spec_dict, get_team_stage_contract(stage, operation))
        spec = TeamAgentSpec.model_validate(spec_dict)
        enrich_team_spec_for_swarm(
            spec,
            session_id=self._session_id,
            mode="team",
            project_dir=self._project_dir,
            request_metadata={"mode": "team"},
        )
        capability_audit = apply_stage_capability_profile(
            spec, (team_config.get("team") or {}).get("stage_capability_profile")
        )
        if capability_audit["profile"] != "default":
            _append_jsonl(self._output_root / "team_capability_audit.jsonl", {
                "timestamp": _utc_now_iso(), "stage": stage,
                "operation": operation, **capability_audit,
            })
        if scoped:
            self._stage_specs[key] = spec
        else:
            self._team_spec = spec
        return spec

    async def invoke(self, *, stage: str, operation: str, query: str) -> str:
        # Reset before rejecting a retry so completed receipts are not charged
        # a second time by the caller's error-accounting path.
        self._last_usage = self._empty_usage()
        self._active_stage = stage
        if self._budget_exhausted:
            raise TeamResourceBudgetExceeded("team stream budget is exhausted; automatic retry is blocked")
        self._check_stream_budget()
        limits = stage_resource_limits(self._config, stage)
        audit = prompt_audit(query, stage=stage, operation=operation)
        limit = limits.get("max_prompt_bytes")
        accepted = limit is None or audit["prompt_bytes"] <= limit
        _append_jsonl(self._output_root / "team_prompt_audit.jsonl", {
            "timestamp": _utc_now_iso(), **audit, "limits": limits,
            "status": "accepted" if accepted else "rejected_before_provider_call",
        })
        if not accepted:
            raise TeamResourceBudgetExceeded(
                f"stage prompt exceeds byte ceiling: {stage} {audit['prompt_bytes']} > {limit}; "
                "no model call made. Evidence was not truncated."
            )
        if self.uses_direct_stage(stage):
            return await self._invoke_direct(stage=stage, operation=operation, query=query)
        from openjiuwen.core.runner import Runner

        spec = self._ensure_team_spec(stage, operation)
        contract = get_team_stage_contract(stage, operation)
        # Stages use fresh team sessions; evidence sharing belongs to the
        # pipeline rather than resuming an already-concluded leader.
        self._session_counter += 1
        session_id = f"{self._session_id}-{self._session_counter}"
        last_leader_output = ""
        last_output = ""
        invocation_timeout = _stage_timeout_seconds(self._config, stage)
        stream_chunks = 0
        stream = Runner.run_agent_team_streaming(
            agent_team=spec, inputs={"query": query}, session=session_id,
        )
        failure = None
        try:
            async with asyncio.timeout(invocation_timeout):
                async for chunk in stream:
                    role = _chunk_role(chunk)
                    text = _chunk_text(chunk)
                    chunk_type = str(getattr(chunk, "type", "") or "").strip().lower()
                    stream_chunks += 1
                    provider_failure = _provider_failure_message(chunk)
                    if provider_failure:
                        _append_jsonl(self._output_root / "team_events.jsonl", {
                            "timestamp": _utc_now_iso(), "session_id": session_id,
                            "stage": stage, "operation": operation, "role": role,
                            "type": "provider_fatal_error", "error": provider_failure,
                            "stream_chunk_count": stream_chunks,
                        })
                        raise TeamProviderError(provider_failure)
                    receipt = None
                    if chunk_type == "llm_usage":
                        receipt = _usage_from_value(getattr(chunk, "payload", None))
                        self._last_usage["provider_calls"] += 1
                        if not receipt:
                            self._last_usage["missing_usage_calls"] += 1
                        for key, value in receipt.items():
                            self._last_usage[key] += value
                        # Numeric receipts make future totals replayable without
                        # retaining prompts, credentials or model reasoning.
                        _append_jsonl(self._output_root / "provider_usage.jsonl", {
                            "timestamp": _utc_now_iso(), "session_id": session_id,
                            "stage": stage, "operation": operation, "role": role,
                            "stream_chunk_count": stream_chunks,
                            "provider_call_index": self._last_usage["provider_calls"],
                            "usage_source": "llm_usage",
                            "usage_status": "reported" if receipt else "missing_or_invalid",
                            **receipt,
                        })
                    # Reasoning is token-streamed too. Retain lifecycle events
                    # and answers, and at most two fallback strings in memory.
                    if chunk_type not in {"llm_output", "llm_reasoning"}:
                        _append_jsonl(self._output_root / "team_events.jsonl", {
                            "timestamp": _utc_now_iso(), "session_id": session_id,
                            "stage": stage, "operation": operation, "type": chunk_type,
                            "role": role, "stream_chunk_count": stream_chunks,
                            "payload_preview": text[:2000] if chunk_type == "answer" else text[:300],
                            **({"provider_usage": receipt} if receipt is not None else {}),
                        })
                    self._check_stream_budget()
                    if not text.strip() or chunk_type in {"llm_output", "llm_reasoning"}:
                        continue
                    last_output = text
                    if role == "leader":
                        last_leader_output = text
                        if _is_terminal_answer_chunk(chunk) and _looks_like_terminal_result(
                            text=text, result_kind=contract.result_kind,
                        ):
                            return text.strip()
        except (Exception, asyncio.CancelledError) as exc:
            failure = exc
            # Closing an iterator alone may leave a persistent team running.
            # This also applies to provider/transport failures: the leader can
            # fail while sibling members are still consuming model resources.
            # Stop the active team using the native lifecycle API as well.
            stop = getattr(Runner, "stop_agent_team", None)
            team_name = getattr(spec, "team_name", None)
            if callable(stop) and team_name:
                try:
                    async with asyncio.timeout(10):
                        await stop(team_name=team_name, session_id=session_id)
                except Exception as cleanup_error:
                    self._record_cleanup_failure(session_id, "stop_agent_team", cleanup_error)
            if isinstance(exc, TimeoutError):
                raise TimeoutError(
                    f"Native Team timed out during {stage}.{operation} "
                    f"(configured stage deadline {invocation_timeout:g}s). "
                    "Completed workflow checkpoints and observed usage are retained; resume the failed stage."
                ) from exc
            raise
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                try:
                    async with asyncio.timeout(10):
                        await close()
                except Exception as cleanup_error:
                    if failure is None:
                        raise
                    self._record_cleanup_failure(session_id, "stream.aclose", cleanup_error)
        final = (last_leader_output or last_output).strip()
        if not final:
            raise RuntimeError(f"native team returned no final text for {stage}.{operation}")
        return final


    def _record_cleanup_failure(self, session_id: str, operation: str, error: Exception) -> None:
        # Best-effort telemetry must never replace the invocation exception.
        try:
            _append_jsonl(self._output_root / "team_events.jsonl", {
                "timestamp": _utc_now_iso(), "session_id": session_id,
                "type": "team_cleanup_error", "operation": operation,
                "error_type": type(error).__name__,
            })
        except OSError:
            pass


class JiuwenSwarmResearchTeamRuntime:
    """Team-backed research runtime that keeps SciTrace's stage contracts structured."""

    def __init__(
        self,
        *,
        backend: TeamInvocationBackend,
        research_id: str,
        call_log_path: str | Path | None = None,
        output_root: str | Path | None = None,
        compact_handoff: bool = False,
    ) -> None:
        self._backend = backend
        self._research_id = research_id
        self._call_log = Path(call_log_path) if call_log_path else None
        self._output_root = Path(output_root) if output_root else None
        self._request_count = 0
        self._usage_records: list[dict[str, Any]] = []
        self._compact_handoff = bool(compact_handoff)

    def _record_call(
        self, *, stage: str, operation: str, status: str,
        usage: dict[str, Any] | None = None, error: BaseException | None = None,
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
                "provider_calls": int(usage.get("provider_calls", 0) or 0),
                "missing_usage_calls": int(usage.get("missing_usage_calls", 0) or 0),
                "usage_source": usage.get("usage_source", "unknown"),
                "usage_status": usage.get("usage_status", "unknown"),
                "call_scope": "team_stage_invocation",
                "execution_path": (info.get("stage_dispatch") or {}).get(stage, "native_team"),
                "error_chain": [] if error is None else [{"type": type(error).__name__, "message": str(error)}],
            },
        )

    def _capture_usage(self, *, stage: str, operation: str) -> dict[str, Any]:
        backend_usage = getattr(self._backend, "last_usage", None)
        if callable(backend_usage):
            # Native usage is authoritative even when absent/zero. Falling
            # back to a process-global tracker could charge unrelated calls.
            total = backend_usage()
        else:
            total = get_llm_token_usage_summary().get("total") or {}
        record = {
            "stage": stage,
            "operation": operation,
            "model": self.runtime_info().get("model_name", ""),
            "provider_calls": int(total.get("provider_calls", 0) or 0),
            "missing_usage_calls": int(total.get("missing_usage_calls", 0) or 0),
            "usage_source": total.get("usage_source", "symphony_tracker"),
            "usage_status": total.get("usage_status", "unknown"),
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
        direct = bool(getattr(self._backend, "uses_direct_stage", lambda _stage: False)(stage))
        query = (build_direct_specialist_query if direct else build_team_query)(
            contract=contract,
            research_id=self._research_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            compact=self._compact_handoff,
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
        except (Exception, asyncio.CancelledError) as exc:
            usage = self._capture_usage(stage=stage, operation=operation)
            self._record_call(stage=stage, operation=operation, status="error", usage=usage, error=exc)
            raise
        # Capture once per invocation. Logging can fail (for example on a full
        # disk); placing success telemetry inside the invocation try would
        # re-enter the error path and count the same provider receipts twice.
        usage = self._capture_usage(stage=stage, operation=operation)
        self._record_call(stage=stage, operation=operation, status="success", usage=usage)
        return result

    async def complete_text(
        self,
        *,
        stage: str,
        operation: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        contract = get_team_stage_contract(stage, operation)
        direct = bool(getattr(self._backend, "uses_direct_stage", lambda _stage: False)(stage))
        query = (build_direct_specialist_query if direct else build_team_query)(
            contract=contract,
            research_id=self._research_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            compact=self._compact_handoff,
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
        except (Exception, asyncio.CancelledError) as exc:
            usage = self._capture_usage(stage=stage, operation=operation)
            self._record_call(stage=stage, operation=operation, status="error", usage=usage, error=exc)
            raise
        usage = self._capture_usage(stage=stage, operation=operation)
        self._record_call(stage=stage, operation=operation, status="success", usage=usage)
        return markdown

    def usage_summary(self) -> dict[str, Any]:
        prompt_tokens = sum(int(item.get("prompt_tokens", 0) or 0) for item in self._usage_records)
        completion_tokens = sum(int(item.get("completion_tokens", 0) or 0) for item in self._usage_records)
        total_tokens = sum(int(item.get("total_tokens", 0) or 0) for item in self._usage_records)
        return {
            "total": {
                "request_count": self._request_count,
                "provider_calls": sum(item.get("provider_calls", 0) for item in self._usage_records),
                "missing_usage_calls": sum(item.get("missing_usage_calls", 0) for item in self._usage_records),
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
    "TeamResourceBudgetExceeded",
]

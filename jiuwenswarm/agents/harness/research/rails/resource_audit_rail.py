from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from jiuwenswarm.agents.harness.research.reporting.resource_reporter import (
    ResourceAuditSnapshot,
    ResourceReporter,
)
from jiuwenswarm.agents.harness.research.rails._compat import (
    AgentCallbackContext,
    DeepAgentRail,
)
from jiuwenswarm.common.utils import get_agent_workspace_dir

_INVOKE_START_KEY = "__resource_audit_invoke_started_at__"
_TOOL_START_KEY = "__resource_audit_tool_started_at__"


class ResourceAuditRail(DeepAgentRail):
    """Collect per-session resource counters and write reports."""

    priority = 62

    def __init__(
        self,
        *,
        enabled: bool = False,
        output_root: str | Path | None = None,
        member_name: str = "",
        role: str = "",
        write_report_on_after_invoke: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        resolved_root = Path(output_root) if output_root else get_agent_workspace_dir() / "research"
        self.reporter = ResourceReporter(resolved_root)
        self.member_name = member_name
        self.role = role
        self.write_report_on_after_invoke = bool(write_report_on_after_invoke)
        self._snapshots: dict[str, ResourceAuditSnapshot] = {}

    def _session_id(self, ctx: AgentCallbackContext | None = None) -> str:
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

    def snapshot(self, session_id: str = "default") -> ResourceAuditSnapshot:
        snapshot = self._snapshots.get(session_id)
        if snapshot is None:
            snapshot = ResourceAuditSnapshot(
                session_id=session_id,
                member_name=self.member_name,
                role=self.role,
            )
            self._snapshots[session_id] = snapshot
        return snapshot

    async def before_invoke(self, ctx: AgentCallbackContext) -> None:
        if not self.enabled:
            return
        session_id = self._session_id(ctx)
        self.snapshot(session_id).invoke_count += 1
        extra = getattr(ctx, "extra", None)
        if isinstance(extra, dict):
            extra[_INVOKE_START_KEY] = time.perf_counter()

    async def after_invoke(self, ctx: AgentCallbackContext) -> None:
        if not self.enabled:
            return
        session_id = self._session_id(ctx)
        snapshot = self.snapshot(session_id)
        extra = getattr(ctx, "extra", None)
        started_at = extra.get(_INVOKE_START_KEY) if isinstance(extra, dict) else None
        if isinstance(started_at, (int, float)):
            snapshot.total_runtime_seconds += max(0.0, time.perf_counter() - started_at)
        if self.write_report_on_after_invoke:
            self.reporter.write_snapshot(snapshot)

    async def before_tool_call(self, ctx: AgentCallbackContext) -> None:
        if not self.enabled:
            return
        extra = getattr(ctx, "extra", None)
        if isinstance(extra, dict):
            extra[_TOOL_START_KEY] = time.perf_counter()

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        if not self.enabled:
            return
        session_id = self._session_id(ctx)
        snapshot = self.snapshot(session_id)
        inputs = getattr(ctx, "inputs", None)
        tool_name = str(getattr(inputs, "tool_name", "") or "")
        snapshot.tool_calls += 1
        if tool_name:
            snapshot.tools_by_name[tool_name] = snapshot.tools_by_name.get(tool_name, 0) + 1

        extra = getattr(ctx, "extra", None)
        started_at = extra.get(_TOOL_START_KEY) if isinstance(extra, dict) else None
        if isinstance(started_at, (int, float)):
            duration_seconds = max(0.0, time.perf_counter() - started_at)
            snapshot.total_tool_seconds += duration_seconds
            if tool_name:
                snapshot.tool_seconds_by_name[tool_name] = (
                    snapshot.tool_seconds_by_name.get(tool_name, 0.0) + duration_seconds
                )

        if self._tool_result_is_failure(getattr(inputs, "tool_result", None)):
            snapshot.tool_failures += 1

    async def on_tool_exception(self, ctx: AgentCallbackContext) -> None:
        if not self.enabled:
            return
        session_id = self._session_id(ctx)
        self.snapshot(session_id).tool_failures += 1

    async def after_model_call(self, ctx: AgentCallbackContext) -> None:
        if not self.enabled:
            return
        session_id = self._session_id(ctx)
        snapshot = self.snapshot(session_id)
        inputs = getattr(ctx, "inputs", None)
        response = getattr(inputs, "response", None) if inputs is not None else None
        usage = self._usage_metadata(response)
        snapshot.model_calls += 1
        model_name = self._model_name(ctx)
        if model_name:
            snapshot.model_calls_by_name[model_name] = snapshot.model_calls_by_name.get(model_name, 0) + 1

        input_tokens = self._coerce_int(usage.get("input_tokens") or usage.get("prompt_tokens"))
        output_tokens = self._coerce_int(usage.get("output_tokens") or usage.get("completion_tokens"))
        total_tokens = self._coerce_int(usage.get("total_tokens"))
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        snapshot.input_tokens += input_tokens
        snapshot.output_tokens += output_tokens
        snapshot.total_tokens += total_tokens
        if model_name:
            snapshot.model_tokens_by_name[model_name] = (
                snapshot.model_tokens_by_name.get(model_name, 0) + total_tokens
            )

    @staticmethod
    def _usage_metadata(response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            dumped = usage.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        return usage if isinstance(usage, dict) else {}

    @staticmethod
    def _coerce_int(value: Any) -> int:
        if isinstance(value, bool) or value is None:
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return 0
        return 0

    @staticmethod
    def _tool_result_is_failure(result: Any) -> bool:
        if isinstance(result, dict):
            if result.get("success") is False:
                return True
            if result.get("is_error") is True:
                return True
            if str(result.get("status") or "").lower() in {"error", "failed", "failure"}:
                return True
            return False
        return False

    @staticmethod
    def _model_name(ctx: AgentCallbackContext) -> str:
        try:
            agent = getattr(ctx, "agent", None)
            config = getattr(agent, "_config", None)
            return str(getattr(config, "model_name", "") or "")
        except Exception:
            return ""

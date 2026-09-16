from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from openjiuwen.core.single_agent.rail.base import AgentCallbackContext
    from openjiuwen.harness.prompts import PromptSection
    from openjiuwen.harness.rails.base import DeepAgentRail
except Exception:  # pragma: no cover - fallback for mock-only environments
    class AgentCallbackContext:  # type: ignore[override]
        pass

    class DeepAgentRail:
        priority = 50

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)

        def init(self, agent: Any) -> None:
            _ = agent

        def uninit(self, agent: Any) -> None:
            _ = agent

    @dataclass
    class PromptSection:
        name: str
        content: dict[str, str]
        priority: int = 0

        def render(self, language: str) -> str:
            return str(
                self.content.get(language)
                or self.content.get("en")
                or next(iter(self.content.values()), "")
            )


__all__ = ["AgentCallbackContext", "DeepAgentRail", "PromptSection"]

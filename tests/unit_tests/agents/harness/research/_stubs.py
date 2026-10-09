from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from types import ModuleType


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


def install_openjiuwen_stubs() -> None:
    try:
        importlib.import_module("openjiuwen.core.foundation.kv_cache")
        importlib.import_module("openjiuwen.harness.rails.base")
        return
    except Exception:
        pass

    for name in [
        "openjiuwen",
        "openjiuwen.core",
        "openjiuwen.core.foundation",
        "openjiuwen.core.foundation.kv_cache",
        "openjiuwen.core.kv_cache",
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

    kv_cache = _ensure_module("openjiuwen.core.foundation.kv_cache")

    @dataclass
    class _KVCacheAffinityConfig:
        enable_kv_cache_release: bool = False
        enable_kv_cache_affinity: bool = False

    kv_cache.KVCacheAffinityConfig = _KVCacheAffinityConfig
    _ensure_module("openjiuwen.core.kv_cache").KVCacheAffinityConfig = _KVCacheAffinityConfig

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

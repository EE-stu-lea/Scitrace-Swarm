"""Pure, auditable policy for bounded native research Team invocations.

This does not summarize evidence or estimate billed tokens. It limits the
available roster and measures the exact submitted UTF-8 prompt; provider
receipts remain the sole source of token accounting.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any

from .team_contract import ROLE_RESEARCH_LEADER, TeamStageContract


_LIMIT_KEYS = {
    "max_total_tokens": "max_invocation_total_tokens",
    "max_provider_calls": "max_invocation_provider_calls",
    "max_prompt_bytes": "max_invocation_prompt_bytes",
}


def stage_resource_limits(config: dict[str, Any], stage: str) -> dict[str, int]:
    """Stage overrides may tighten, never relax, invocation-wide ceilings."""
    team = config.get("team") or {}
    overrides = team.get("stage_resource_limits") or {}
    if not isinstance(overrides, dict):
        raise ValueError("team.stage_resource_limits must be a mapping")
    selected = overrides.get(stage, {})
    if not isinstance(selected, dict):
        raise ValueError(f"team.stage_resource_limits.{stage} must be a mapping")
    unknown = set(selected) - set(_LIMIT_KEYS)
    if unknown:
        raise ValueError(f"unknown stage resource limits: {sorted(unknown)}")
    limits = {}
    for local, global_key in _LIMIT_KEYS.items():
        values = [value for value in (team.get(global_key), selected.get(local)) if value is not None]
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"team resource limit {local} must be a positive integer")
        if values:
            limits[local] = min(values)
    return limits


def prompt_audit(query: str, *, stage: str, operation: str) -> dict[str, Any]:
    """Safe telemetry contains no prompt, key, manuscript, or reasoning text."""
    encoded = query.encode("utf-8")
    return {
        "stage": stage, "operation": operation,
        "prompt_bytes": len(encoded), "prompt_characters": len(query),
        "prompt_sha256": hashlib.sha256(encoded).hexdigest(),
        "measurement": "submitted_query_utf8_bytes_not_provider_tokens",
    }


def scope_team_spec(spec: dict[str, Any], contract: TeamStageContract) -> dict[str, Any]:
    """Restrict a copy of the native spec to the contracted stage owner.

    ``team_mode=predefined`` is a native capability ceiling: core removes the
    dynamic spawn tool. It is stronger than asking a broad team not to spawn.
    Team coordination remains native; the pipeline owns cross-stage context.
    """
    result = deepcopy(spec)
    primary = contract.primary_member
    members = result.get("predefined_members") or []
    selected = [item for item in members if item.get("member_name") == primary]
    if primary != ROLE_RESEARCH_LEADER and len(selected) != 1:
        raise ValueError(f"stage requires exactly one predefined member named {primary}")
    result["predefined_members"] = [] if primary == ROLE_RESEARCH_LEADER else selected
    result["team_mode"] = "predefined"
    # Every workflow stage starts a fresh Team session; temporary lifecycle
    # lets core finalize its coordination loop after a stage result instead
    # of keeping an otherwise idle persistent team alive.
    result["lifecycle"] = "temporary"
    result["enable_group_chat"] = False
    result["enable_team_plan"] = False
    result["enable_hitt"] = False
    result["evolution_enabled"] = False
    result["external_cli_agents"] = []
    result["team_name"] = f"{result.get('team_name', 'research')}_{contract.stage}"
    return result


def apply_stage_capability_profile(spec: Any, profile: str | None) -> dict[str, Any]:
    """Remove platform-wide capabilities from a bounded research Team stage.

    Research evidence and experiments are supplied by the workflow, not by
    ad-hoc member tools. Core injects its TeamToolRail and TeamPolicyRail after
    this spec is assembled, so native task delegation and handoff remain.
    Apply *after* swarm enrichment, which otherwise adds the broad platform
    tool and rail catalogue back to each member.
    """
    if profile in (None, "default"):
        return {"profile": "default", "members": {}}
    if profile != "prompt_only":
        raise ValueError("team.stage_capability_profile must be default or prompt_only")
    audit: dict[str, Any] = {"profile": profile, "members": {}}
    for role in ("leader", "teammate"):
        member = spec.agents.get(role)
        if member is None:
            continue
        before = {
            "tools": len(member.tools or []), "rails": len(member.rails or []),
            "mcps": len(member.mcps or []), "skills": len(member.skills or []),
            "sys_operation": bool(member.enable_sys_operation),
        }
        spec.agents[role] = member.model_copy(update={
            "tools": [], "rails": [], "mcps": [], "subagents": [],
            "skills": [], "sys_operation": None,
            "enable_sys_operation": False, "enable_task_planning": False,
            "enable_skill_discovery": False,
            "enable_subagent_runtime": False,
            "add_general_purpose_agent": False,
            "progressive_tool": None, "approval_required_tools": [],
            "enable_read_image_multimodal": False,
        })
        audit["members"][role] = {"before": before, "after": {
            "tools": 0, "rails": 0, "mcps": 0, "skills": 0,
            "sys_operation": False, "image_modality_probe": False,
        }}
    return audit


__all__ = ["apply_stage_capability_profile", "prompt_audit", "scope_team_spec", "stage_resource_limits"]

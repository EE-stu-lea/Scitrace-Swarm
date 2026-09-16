# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Research rail providers for scientific provenance and resource auditing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openjiuwen.agent_teams.harness.manifest import (
    ElementKind,
    harness_element,
)

from jiuwenswarm.agents.harness.research.rails import (
    ResourceAuditRail,
    ScientificConsistencyGateRail,
    ScientificProvenanceRail,
    StageAwareContextRail,
)
from jiuwenswarm.agents.swarm.context import SwarmBuildContext
from jiuwenswarm.common.utils import get_agent_workspace_dir

SCIENTIFIC_PROVENANCE = "swarm.scientific_provenance"
STAGE_AWARE_CONTEXT = "swarm.stage_aware_context"
SCIENTIFIC_CONSISTENCY_GATE = "swarm.scientific_consistency_gate"
RESOURCE_AUDIT = "swarm.resource_audit"


def _resolve_output_root(
    context: SwarmBuildContext,
    *,
    configured_root: str = "",
    folder_name: str,
) -> Path:
    if str(configured_root or "").strip():
        return Path(str(configured_root).strip())
    workspace_root = getattr(getattr(context, "workspace", None), "root_path", None)
    if workspace_root:
        return Path(str(workspace_root)) / ".scitrace" / folder_name
    if context.project_dir:
        return Path(str(context.project_dir)) / ".scitrace" / folder_name
    if context.team_ws_root:
        return Path(str(context.team_ws_root)) / ".scitrace" / folder_name
    return get_agent_workspace_dir() / "research" / folder_name


@harness_element(
    kind=ElementKind.RAIL,
    name=SCIENTIFIC_PROVENANCE,
    description="Capture scientific provenance nodes, edges, and tool audit events.",
)
def build_scientific_provenance_rail(
    params: dict[str, Any],
    context: SwarmBuildContext,
) -> ScientificProvenanceRail | None:
    config = params if isinstance(params, dict) else {}
    if not bool(config.get("enabled", False)):
        return None
    return ScientificProvenanceRail(
        enabled=True,
        output_root=_resolve_output_root(
            context,
            configured_root=str(config.get("output_root", "") or ""),
            folder_name="provenance",
        ),
        capture_tool_audits=bool(config.get("capture_tool_audits", True)),
    )


@harness_element(
    kind=ElementKind.RAIL,
    name=RESOURCE_AUDIT,
    description="Collect per-session resource usage and write report artifacts.",
)
def build_resource_audit_rail(
    params: dict[str, Any],
    context: SwarmBuildContext,
) -> ResourceAuditRail | None:
    config = params if isinstance(params, dict) else {}
    if not bool(config.get("enabled", False)):
        return None
    return ResourceAuditRail(
        enabled=True,
        output_root=_resolve_output_root(
            context,
            configured_root=str(config.get("output_root", "") or ""),
            folder_name="resource_audit",
        ),
        member_name=str(getattr(context, "member_name", "") or ""),
        role=str(getattr(context, "role", "") or ""),
        write_report_on_after_invoke=bool(config.get("write_report_on_after_invoke", True)),
    )


@harness_element(
    kind=ElementKind.RAIL,
    name=STAGE_AWARE_CONTEXT,
    description="Inject stage-aware scientific context from the evidence graph.",
)
def build_stage_aware_context_rail(
    params: dict[str, Any],
    context: SwarmBuildContext,
) -> StageAwareContextRail | None:
    config = params if isinstance(params, dict) else {}
    if not bool(config.get("enabled", False)):
        return None
    return StageAwareContextRail(
        enabled=True,
        output_root=_resolve_output_root(
            context,
            configured_root=str(config.get("output_root", "") or ""),
            folder_name="provenance",
        ),
        token_budget=int(config.get("token_budget", 1800) or 1800),
        include_conflicts=bool(config.get("include_conflicts", True)),
    )


@harness_element(
    kind=ElementKind.RAIL,
    name=SCIENTIFIC_CONSISTENCY_GATE,
    description="Block paper outputs that are not supported by the evidence graph.",
)
def build_scientific_consistency_gate_rail(
    params: dict[str, Any],
    context: SwarmBuildContext,
) -> ScientificConsistencyGateRail | None:
    config = params if isinstance(params, dict) else {}
    if not bool(config.get("enabled", False)):
        return None
    return ScientificConsistencyGateRail(
        enabled=True,
        output_root=_resolve_output_root(
            context,
            configured_root=str(config.get("output_root", "") or ""),
            folder_name="provenance",
        ),
        block_on_error=bool(config.get("block_on_error", True)),
    )


__all__ = [
    "SCIENTIFIC_PROVENANCE",
    "STAGE_AWARE_CONTEXT",
    "SCIENTIFIC_CONSISTENCY_GATE",
    "RESOURCE_AUDIT",
    "build_scientific_provenance_rail",
    "build_stage_aware_context_rail",
    "build_scientific_consistency_gate_rail",
    "build_resource_audit_rail",
]

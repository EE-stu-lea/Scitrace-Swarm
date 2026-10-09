"""No-provider regressions for bounded, evidence-preserving stage dispatch."""
import asyncio
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from ._stubs import install_openjiuwen_stubs

install_openjiuwen_stubs()

from jiuwenswarm.agents.harness.research.stage_policy import (
    apply_stage_capability_profile, prompt_audit, scope_team_spec, stage_resource_limits,
)
from jiuwenswarm.agents.harness.research.team_contract import build_team_query, get_team_stage_contract
from jiuwenswarm.agents.harness.research.team_runtime import TeamResourceBudgetExceeded
from jiuwenswarm.agents.harness.research.runtime import ResearchStageHarness
from .test_native_resource_accounting import _backend, _chunk, _invoke


def test_prompt_only_profile_removes_generic_member_surface_but_keeps_team_spec():
    class Member:
        def __init__(self, **updates):
            self.__dict__.update({
                "tools": ["search"], "rails": ["generic_rail"], "mcps": ["remote"],
                "skills": ["skill"], "enable_sys_operation": True,
            } | updates)

        def model_copy(self, *, update):
            return Member(**(self.__dict__ | update))

    leader, teammate = Member(), Member()
    spec = SimpleNamespace(agents={"leader": leader, "teammate": teammate},
                           team_mode="predefined", predefined_members=["researcher"])
    audit = apply_stage_capability_profile(spec, "prompt_only")
    assert audit["members"]["leader"]["before"]["tools"] == 1
    assert audit["members"]["teammate"]["before"]["rails"] == 1
    for member in spec.agents.values():
        assert member.tools == member.rails == member.mcps == member.skills == []
        assert member.enable_sys_operation is False
        assert member.enable_read_image_multimodal is False
    assert leader.tools == ["search"] and teammate.rails == ["generic_rail"]
    assert spec.team_mode == "predefined" and spec.predefined_members == ["researcher"]
    with pytest.raises(ValueError, match="stage_capability_profile"):
        apply_stage_capability_profile(spec, "unknown")


def test_scope_limits_native_roster_without_mutating_source_or_permission_policy():
    spec = {"team_name": "fixture", "team_mode": "hybrid", "enable_permissions": True,
            "predefined_members": [{"member_name": name} for name in
                                   ("writer_agent", "auditor_agent", "experiment_agent")]}
    original = deepcopy(spec)
    scoped = scope_team_spec(spec, get_team_stage_contract("draft", "write_draft"))
    assert spec == original
    assert scoped["predefined_members"] == [{"member_name": "writer_agent"}]
    assert scoped["team_mode"] == "predefined"  # native core removes spawn_member
    assert scoped["lifecycle"] == "temporary"
    assert scoped["enable_permissions"] is True
    assert scoped["enable_hitt"] is False and scoped["evolution_enabled"] is False
    assert scoped["team_name"] == "fixture_draft"


def test_leader_owned_claim_stage_has_no_self_delegation():
    contract = get_team_stage_contract("claim_compiler", "compile_claim")
    scoped = scope_team_spec({"predefined_members": [{"member_name": "writer_agent"}]}, contract)
    assert scoped["predefined_members"] == []
    query = build_team_query(contract=contract, research_id="id", system_prompt="sys", user_prompt="data", compact=True)
    assert "without delegating to yourself" in query
    assert "one delegation round" not in query
    assert "writer_agent" not in query


def test_short_topic_intake_is_leader_owned_to_avoid_task_board_polling():
    contract = get_team_stage_contract("topic_intake", "research_question")
    assert contract.primary_member == "research_leader"
    scoped = scope_team_spec({"predefined_members": [{"member_name": "literature_ideation_agent"}]}, contract)
    assert scoped["predefined_members"] == []
    query = build_team_query(contract=contract, research_id="id", system_prompt="sys", user_prompt="data", compact=True)
    assert "without delegating to yourself" in query
    assert "one delegation round" not in query


def test_missing_or_duplicate_required_owner_fails_before_native_team_is_built():
    contract = get_team_stage_contract("review", "review_draft")
    for members in ([], [{"member_name": "auditor_agent"}] * 2):
        with pytest.raises(ValueError, match="exactly one"):
            scope_team_spec({"predefined_members": members}, contract)


def test_compact_handoff_preserves_complete_unicode_manuscript_and_evidence():
    contract = get_team_stage_contract("revision", "targeted_revision")
    user = '# Title\n' + '完整正文 evidence \\(k=5\\).\n' * 3000 + 'END_OF_MANUSCRIPT'
    system = "Verified context is authoritative."
    broad = build_team_query(contract=contract, research_id="id", system_prompt=system, user_prompt=user)
    compact = build_team_query(contract=contract, research_id="id", system_prompt=system, user_prompt=user, compact=True)
    assert compact.split("# Stage User Prompt\n", 1)[1] == user + "\n"
    assert system in compact
    assert len(compact.encode()) < len(broad.encode())
    assert "first line must be the manuscript's `# Title`" in compact


@pytest.mark.parametrize("invalid", [True, 0, -2, 1.5, "10"])
def test_invalid_stage_limits_rejected(invalid):
    with pytest.raises(ValueError):
        stage_resource_limits({"team": {"stage_resource_limits": {"draft": {"max_provider_calls": invalid}}}}, "draft")


def test_stage_limits_tighten_global_limits_and_keep_other_stage_independent():
    config = {"team": {"max_invocation_provider_calls": 12, "max_invocation_total_tokens": 500,
                       "stage_resource_limits": {"draft": {"max_provider_calls": 20, "max_total_tokens": 100}}}}
    assert stage_resource_limits(config, "draft") == {"max_provider_calls": 12, "max_total_tokens": 100}
    assert stage_resource_limits(config, "review") == {"max_provider_calls": 12, "max_total_tokens": 500}
    with pytest.raises(ValueError, match="unknown"):
        stage_resource_limits({"team": {"stage_resource_limits": {"draft": {"max_call": 2}}}}, "draft")


def test_prompt_byte_limit_rejects_before_model_or_native_spec_and_does_not_log_content(tmp_path, monkeypatch):
    backend, closed, stopped, started = _backend(tmp_path, monkeypatch, [], team={"max_invocation_prompt_bytes": 4})
    monkeypatch.setattr(backend, "_ensure_team_spec", lambda *args: pytest.fail("must not construct Team"))
    with pytest.raises(TeamResourceBudgetExceeded, match="no model call made"):
        _invoke(backend)
    assert not closed and not stopped and not started
    assert backend.last_usage()["provider_calls"] == 0
    audit = json.loads((tmp_path / "team_prompt_audit.jsonl").read_text())
    assert audit["status"] == "rejected_before_provider_call"
    assert audit["prompt_sha256"] == hashlib.sha256(b"fixture").hexdigest()
    assert "fixture" not in json.dumps(audit)
    assert prompt_audit("中文字", stage="draft", operation="write_draft")["prompt_bytes"] == 9


def test_stage_receipt_ceiling_stops_native_stream_and_retains_observed_receipts(tmp_path, monkeypatch):
    backend, closed, stopped, started = _backend(tmp_path, monkeypatch, [
        _chunk("llm_usage", {"input_tokens": 100, "output_tokens": 10}),
        AssertionError("stage budget must stop before further provider work"),
    ], team={"max_invocation_provider_calls": 10,
             "stage_resource_limits": {"claim_compiler": {"max_provider_calls": 1}}})
    with pytest.raises(TeamResourceBudgetExceeded):
        _invoke(backend)
    assert closed and stopped and len(started) == 1
    assert backend.last_usage()["total_tokens"] == 110


def test_scoped_brief_receives_fixed_research_contract_without_breaking_legacy_topics():
    captured = []
    async def complete_json(**kwargs):
        captured.append(kwargs)
        return {"research_question": "Does this allocation improve recall?", "hypothesis": "To be tested."}
    harness = ResearchStageHarness(SimpleNamespace(complete_json=complete_json))
    try:
        harness.generate_research_brief(topic="  topic  ")
        harness.generate_research_brief(topic="  topic  ", research_scope={"minimum_depth": 1})
    finally:
        harness.close()
    assert captured[0]["user_prompt"] == "topic"
    assert json.loads(captured[1]["user_prompt"]) == {"topic": "topic", "research_scope": {"minimum_depth": 1}}
    assert "rather than presupposing a positive result" in captured[1]["system_prompt"]

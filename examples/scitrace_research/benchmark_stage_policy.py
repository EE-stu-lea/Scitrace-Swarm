"""Measure prompt packaging and roster changes without making API calls.

This captures the real harness's author/reviewer/revision prompts. Results are
UTF-8 byte counts, not a tokenizer estimate, latency benchmark, quality score,
or prediction of provider billing. Manuscript and verified evidence are kept
complete in both conditions; only the handoff envelope/persona list differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jiuwenswarm.agents.harness.research.runtime import ResearchStageHarness
from jiuwenswarm.agents.harness.research.stage_policy import prompt_audit, scope_team_spec
from jiuwenswarm.agents.harness.research.team_contract import (
    TEAM_STAGE_CONTRACTS, build_team_query, get_team_stage_contract,
)


class _CaptureOnlyRuntime:
    """No model object, endpoint, credential, or network client is constructed."""

    def __init__(self):
        self.calls = []

    async def complete_text(self, **kwargs):
        self.calls.append(kwargs)
        return "capture only; never a generated manuscript"

    async def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return {"summary": "Prompt capture only.", "severity": "none", "findings": []}


def benchmark(manuscript_context: dict, draft: str) -> dict:
    capture = _CaptureOnlyRuntime()
    harness = ResearchStageHarness(capture, max_retries=1)
    harness.set_manuscript_context(manuscript_context)
    try:
        harness.write_draft(prompt=json.dumps(manuscript_context, ensure_ascii=False, sort_keys=True))
        review = harness.review_draft(draft=draft, errors=[])
        harness.review_and_revise(draft=draft, errors=[], review=review)
    finally:
        harness.close()
    rows = []
    for call in capture.calls:
        contract = get_team_stage_contract(call["stage"], call["operation"])
        common = dict(contract=contract, research_id="offline-comparison",
                      system_prompt=call["system_prompt"], user_prompt=call["user_prompt"])
        broad = build_team_query(**common, compact=False)
        compact = build_team_query(**common, compact=True)
        broad_tail = broad.split("# Stage System Prompt\n", 1)[1]
        compact_tail = compact.split("# Stage System Prompt\n", 1)[1]
        if broad_tail != compact_tail:
            raise AssertionError("The policy comparison changed stage evidence or manuscript content")
        before = prompt_audit(broad, stage=call["stage"], operation=call["operation"])
        after = prompt_audit(compact, stage=call["stage"], operation=call["operation"])
        rows.append({"stage": call["stage"], "broad_prompt_bytes": before["prompt_bytes"],
                     "compact_prompt_bytes": after["prompt_bytes"],
                     "saved_envelope_bytes": before["prompt_bytes"] - after["prompt_bytes"],
                     "complete_stage_payload_preserved": True,
                     "stage_payload_sha256": hashlib.sha256(broad_tail.encode()).hexdigest()})
    base_spec = {"team_name": "offline", "predefined_members": [
        {"member_name": name} for name in
        ("literature_ideation_agent", "experiment_agent", "writer_agent", "auditor_agent")
    ]}
    roster = []
    for contract in TEAM_STAGE_CONTRACTS.values():
        scoped = scope_team_spec(base_spec, contract)
        roster.append({"stage": contract.stage, "broad_predefined_specialists": 4,
                       "scoped_predefined_specialists": len(scoped["predefined_members"]),
                       "native_team_mode": scoped["team_mode"],
                       "leader_executes_directly": not scoped["predefined_members"]})
    return {
        "protocol": "offline-stage-policy-v1", "provider_calls": 0,
        "comparison": "same current harness content, broad versus compact handoff envelope",
        "measured_unit": "UTF-8 bytes of the submitted query",
        "limitations": [
            "No live model output quality, request count, latency or billed-token saving was measured.",
            "The native SDK adds system prompts and coordination history beyond this submitted query.",
            "Small packaging reductions do not establish an improvement over historical provider usage.",
            "The production benefit of limiting the native roster needs a matched live ablation.",
        ],
        "prompt_comparison": rows, "roster_comparison": roster,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript-context", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context_bytes = args.manuscript_context.read_bytes()
    draft_bytes = args.draft.read_bytes()
    result = benchmark(json.loads(context_bytes), draft_bytes.decode("utf-8"))
    result["sources"] = {
        "manuscript_context_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "draft_sha256": hashlib.sha256(draft_bytes).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved offline prompt/roster measurement: {args.output}")


if __name__ == "__main__":
    main()

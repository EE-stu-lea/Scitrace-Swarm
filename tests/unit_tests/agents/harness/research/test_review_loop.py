"""Exercise author/reviewer handoffs without a provider or scientific fabrication."""
import hashlib
import json

import pytest

from ._stubs import install_openjiuwen_stubs

install_openjiuwen_stubs()

from examples.scitrace_research.workflow.pipeline import _run_consistency_loop
from jiuwenswarm.agents.harness.research.runtime import ResearchRuntimeError, ResearchStageHarness


class RecordingModel:
    def __init__(self, reviews, revised="## Results\nCorrected result."):
        self.reviews = list(reviews)
        self.revised = revised
        self.calls = []

    async def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.reviews.pop(0)

    async def complete_text(self, **kwargs):
        self.calls.append(kwargs)
        return self.revised

    def usage_summary(self):
        return {"total": {"request_count": len(self.calls), "total_tokens": 10 * len(self.calls)}}


def finding(quote="Wrong result.", severity="error"):
    return {"section": "Results", "quote": quote, "severity": severity,
            "message": "The result conflicts with supplied evidence.",
            "suggested_fix": "Use the supplied result and state its limited scope."}


def review(findings=()):
    return {"summary": "Completed scientific review.", "severity": "error" if findings else "none",
            "findings": list(findings)}


def test_reviewer_sees_late_results_and_verified_evidence_without_hidden_json():
    model = RecordingModel([review()])
    harness = ResearchStageHarness(model, max_retries=1)
    manuscript = "## Introduction\n" + "Background. " * 500 + "\n## Results\nLate result.\n"
    harness.set_manuscript_context({"results": {"measured": 0.42}, "claims": [{"id": "claim_1"}]})
    try:
        harness.review_draft(draft=manuscript + "### 9. Structured References\nAUDIT_ONLY", errors=[])
    finally:
        harness.close()
    payload = json.loads(model.calls[0]["user_prompt"])
    assert payload["public_manuscript"] == manuscript
    assert "Late result." in payload["public_manuscript"]
    assert "AUDIT_ONLY" not in payload["public_manuscript"]
    assert payload["verified_context"]["results"]["measured"] == 0.42


@pytest.mark.parametrize("bad", [
    {},
    {"summary": "Reviewed", "severity": "error", "findings": []},
    {"summary": "Reviewed", "severity": "none", "findings": "none"},
    review([finding(quote="Nonexistent quote")]),
    review([finding(quote="")]),
    review([{**finding(), "suggested_fix": ""}]),
])
def test_unactionable_or_ungrounded_reviews_are_rejected(bad):
    model = RecordingModel([bad])
    harness = ResearchStageHarness(model, max_retries=1)
    try:
        with pytest.raises(ResearchRuntimeError):
            harness.review_draft(draft="## Results\nWrong result.", errors=[])
    finally:
        harness.close()


def test_review_reuse_avoids_a_second_paid_audit_and_keeps_verified_context():
    model = RecordingModel([])
    harness = ResearchStageHarness(model, max_retries=1)
    harness.set_manuscript_context({"implementation_contract": {"minimum_depth": 1}})
    already_reviewed = review([finding()])
    try:
        observed, revised = harness.review_and_revise(
            draft="## Results\nWrong result.", errors=[{"message": "Repair"}], review=already_reviewed,
        )
    finally:
        harness.close()
    assert observed is already_reviewed
    assert revised == model.revised
    assert [call["stage"] for call in model.calls] == ["revision"]
    assert '"minimum_depth":1' in model.calls[0]["user_prompt"]


class PassingGate:
    def check_payload(self, payload, session_id):
        return {"ok": True, "errors": [], "error_count": 0}


class NoClaims:
    def record_node(self, *args, **kwargs):
        raise AssertionError("No claims supplied in this fixture")


def run_loop(tmp_path, model, config):
    harness = ResearchStageHarness(model, max_retries=1)
    try:
        return _run_consistency_loop(
            config=config, harness=harness, gate=PassingGate(), provenance=NoClaims(),
            output_root=tmp_path, literature_payload={"evidence": []},
            claim_payload={"claims": []}, execution_payload={"results": {}},
            initial_draft="## Results\nWrong result.",
        )
    finally:
        harness.close()


def test_reviewer_defect_triggers_revision_even_when_deterministic_gate_passes(tmp_path):
    model = RecordingModel([review([finding(severity="ERROR")]), review()])
    result = run_loop(tmp_path, model, {"revision": {"max_rounds": 1}})
    assert result["validation_report"]["ok"]
    assert [call["stage"] for call in model.calls] == ["review", "revision", "review"]
    assert result["draft"] == model.revised
    first, last = [json.loads((tmp_path / "reviews" / f"round_{i}.json").read_text()) for i in (0, 1)]
    assert first["findings"][0]["severity"] == "error"
    assert last["draft_sha256"] == hashlib.sha256(model.revised.encode()).hexdigest()
    assert last["scope"] == "full_public_manuscript"


def test_zero_revisions_is_respected_and_optional_advice_does_not_trigger_rewrite(tmp_path):
    model = RecordingModel([review([finding()])])
    result = run_loop(tmp_path, model, {"revision": {"max_rounds": 0}})
    assert not result["validation_report"]["ok"]
    assert len(model.calls) == 1
    assert result["validation_report"]["errors"][0]["code"] == "paper_reviewer_finding"
    model = RecordingModel([{**review([finding(severity="warning")]), "severity": "warning"}])
    result = run_loop(tmp_path, model, {"revision": {"max_rounds": 1}})
    assert result["validation_report"]["ok"]
    assert len(model.calls) == 1


@pytest.mark.parametrize("budget", [{"max_model_calls": 1}, {"max_total_tokens": 10}])
def test_budget_exhausted_after_review_blocks_revision(tmp_path, budget):
    model = RecordingModel([review([finding()])])
    with pytest.raises(RuntimeError, match="budget exhausted before stage revision"):
        run_loop(tmp_path, model, {"revision": {"max_rounds": 2}, "budget": budget})
    assert len(model.calls) == 1
    assert (tmp_path / "reviews" / "round_0.json").is_file()


def test_budget_exhausted_after_revision_preserves_draft_and_blocks_next_audit(tmp_path):
    model = RecordingModel([review([finding()])])
    with pytest.raises(RuntimeError, match="budget exhausted before stage review"):
        run_loop(tmp_path, model, {"revision": {"max_rounds": 2}, "budget": {"max_model_calls": 2}})
    assert [call["stage"] for call in model.calls] == ["review", "revision"]
    assert (tmp_path / "draft.md").read_text() == model.revised

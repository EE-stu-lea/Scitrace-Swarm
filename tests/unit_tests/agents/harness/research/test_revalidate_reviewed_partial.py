"""Offline recovery only accepts a paid review of the unchanged Agent draft."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ._stubs import install_openjiuwen_stubs

install_openjiuwen_stubs()

from examples.scitrace_research.revalidate_run import _reviewed_partial_evidence


def _fixture(tmp_path):
    (tmp_path / "reviews").mkdir()
    review = {
        "draft_sha256": "draft-hash", "round": 1,
        "findings": [{"severity": "error", "message":
                      "The method overview figure caption is missing."}],
    }
    (tmp_path / "reviews" / "round_1.json").write_text(json.dumps(review))
    (tmp_path / "review_report.json").write_text(json.dumps(review))
    (tmp_path / "paper_quality_report.json").write_text(json.dumps({
        "ok": False, "errors": [{"code": "paper_method_overview_missing"}],
    }))
    (tmp_path / "model_calls.jsonl").write_text(json.dumps({
        "stage": "review", "status": "success",
    }) + "\n")
    checkpoint = SimpleNamespace(status="failed", payload={
        "error": "model call budget exhausted before stage revision: 1",
    })
    return review, checkpoint


def test_reviewed_partial_recovery_requires_hash_bound_review(tmp_path):
    _, checkpoint = _fixture(tmp_path)
    evidence = _reviewed_partial_evidence(tmp_path, checkpoint, "draft-hash", {"model_call_count": 1})
    assert evidence["round"] == 1
    assert evidence["old_quality_error_codes"] == ["paper_method_overview_missing"]
    with pytest.raises(ValueError, match="bind"):
        _reviewed_partial_evidence(tmp_path, checkpoint, "different-hash", {"model_call_count": 1})


def test_reviewed_partial_recovery_rejects_independent_reviewer_defect(tmp_path):
    review, checkpoint = _fixture(tmp_path)
    review["findings"].append({"severity": "error", "message":
                               "A numerical result contradicts the verified experiment."})
    (tmp_path / "reviews" / "round_1.json").write_text(json.dumps(review))
    (tmp_path / "review_report.json").write_text(json.dumps(review))
    with pytest.raises(ValueError, match="independent error"):
        _reviewed_partial_evidence(tmp_path, checkpoint, "draft-hash", {"model_call_count": 1})

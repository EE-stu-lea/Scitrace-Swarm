"""Check mean-risk decompositions, null denominators, and tamper rejection."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from examples.scitrace_research.workflow.query_diagnostics import (
    ARTIFACT_PATH, DATA_PATH, MANIFEST_PATH, build_query_diagnostics,
    budget_transition_diagnostics, decompose_query_outcomes,
    render_query_diagnostics, verify_query_diagnostics,
)

FIXTURE = Path(__file__).resolve().parents[5] / "examples/scitrace_research/fixtures/route_b2_final"


def test_decomposition_separates_mean_benefit_from_complete_evidence_loss():
    # Equal total capacity 12; two regressions to zero, one larger gain.
    result = decompose_query_outcomes([0, 0, 1, .5], [.1, .1, 0, .5], [2, 2, 5, 3], 3)
    assert result["positive_contribution"] == pytest.approx(.25)
    assert result["negative_contribution"] == pytest.approx(-.05)
    assert result["mean_difference"] == pytest.approx(.2)
    assert result["mean_difference"] == pytest.approx(result["positive_contribution"] + result["negative_contribution"])
    assert (result["wins"], result["ties"], result["losses"]) == (1, 1, 2)
    assert result["rescued_from_no_evidence"] == 1
    assert result["newly_no_evidence"] == 2
    assert result["candidate_no_evidence"] - result["baseline_no_evidence"] == 1
    assert result["conservation_residual"] == 0


def test_unchanged_extra_slots_use_moved_capacity_denominator_and_zero_is_null():
    result = decompose_query_outcomes([.5, .5, 1, 0], [.5, .5, .5, 0], [1, 4, 4, 3], 3)
    assert result["moved_document_slots"] == 2
    assert result["extra_slots_on_unchanged_queries"] == 1
    assert result["extra_slots_on_unchanged_fraction"] == .5
    fixed = decompose_query_outcomes([1, 0], [1, 0], [3, 3], 3)
    assert fixed["extra_slots_on_unchanged_fraction"] is None
    assert fixed["conditional_win_fraction"] is None


@pytest.mark.parametrize("candidate,baseline,allocation,budget,error", [
    ([], [], [], 3, "nonempty"),
    ([1], [1, 0], [3], 3, "paired"),
    ([float("nan")], [0], [3], 3, "finite"),
    ([1.1], [0], [3], 3, "recall"),
    ([1], [1], [2.5], 2.5, "integers"),
    ([1, 0], [1, 0], [3, 2], 3, "equal total"),
    ([1, 0], [0, 1], [2, 4], 3, "depth change"),
])
def test_invalid_pairs_fail_closed(candidate, baseline, allocation, budget, error):
    with pytest.raises(ValueError, match=error):
        decompose_query_outcomes(candidate, baseline, allocation, budget)


def test_transition_audit_does_not_hide_individual_regression_in_better_mean():
    data = {"budgets": [2, 3], "query_ids": ["a", "b"],
            "candidate": np.array([[.5, 0], [0, 1.]]),
            "baseline": np.array([[0, .5], [0, .5]]),
            "allocations": np.array([[3, 1], [1, 5]])}
    row = budget_transition_diagnostics(data)[0]
    assert row["candidate_mean_change"] > 0
    assert row["candidate_regressed"] == 1
    assert row["regression_query_ids"] == ["a"]
    assert row["fixed_regressed"] == 0
    assert row["allocation_decreased"] == 1


def test_saved_fixture_exposes_nfc_corpus_mean_risk_tradeoff_and_preserves_sources():
    before = {p.relative_to(FIXTURE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in FIXTURE.rglob("*") if p.is_file()}
    data = build_query_diagnostics(FIXTURE)
    row = next(row for row in data["summaries"] if row["dataset"] == "nfcorpus" and row["budget"] == 5)
    assert row["mean_difference"] > 0
    assert row["newly_no_evidence"] == 17
    assert row["rescued_from_no_evidence"] == 5
    assert row["candidate_no_evidence"] == row["baseline_no_evidence"] + 12
    for row in data["summaries"]:
        assert row["queries"] == row["wins"] + row["ties"] + row["losses"]
        assert row["mean_difference"] == pytest.approx(row["positive_contribution"] + row["negative_contribution"])
    assert len(data["source_sha256"]) == 6
    assert all(not Path(p).is_absolute() for p in data["source_sha256"])
    json.dumps(data, allow_nan=False)
    after = {p.relative_to(FIXTURE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in FIXTURE.rglob("*") if p.is_file()}
    assert before == after


@pytest.fixture(scope="module")
def rendered_run(tmp_path_factory):
    run = tmp_path_factory.mktemp("query-risk")
    shutil.copytree(FIXTURE, run / ARTIFACT_PATH)
    render_query_diagnostics(run / ARTIFACT_PATH, run / "figures")
    return run


def test_rendered_diagnostics_recompute_and_verify(rendered_run):
    result = verify_query_diagnostics(rendered_run)
    assert result["status"] == "passed"
    assert result["sources_verified"] == 6


@pytest.mark.parametrize("tamper,error", [
    ("data", "artifact hash"), ("figure", "artifact hash"),
    ("source", "source outcomes"), ("rehashed_data", "source outcomes"),
    ("unexpected_path", "unexpected artifact paths"),
])
def test_source_bound_manifest_rejects_tampering(rendered_run, tmp_path, tamper, error):
    run = tmp_path / "run"
    shutil.copytree(rendered_run, run)
    if tamper == "data":
        with (run / DATA_PATH).open("a") as stream:
            stream.write(" ")
    elif tamper == "figure":
        with (run / "figures/query_risk.pdf").open("ab") as stream:
            stream.write(b"tampered")
    elif tamper == "source":
        # Whitespace leaves parsed scientific data unchanged but invalidates provenance.
        with (run / ARTIFACT_PATH / "scifact/results.json").open("a") as stream:
            stream.write(" ")
    else:
        manifest = json.loads((run / MANIFEST_PATH).read_text())
        if tamper == "unexpected_path":
            manifest["data_path"] = "../query_diagnostics.json"
        else:
            data = json.loads((run / DATA_PATH).read_text())
            data["summaries"][0]["wins"] += 1
            (run / DATA_PATH).write_text(json.dumps(data))
            manifest["data_sha256"] = hashlib.sha256((run / DATA_PATH).read_bytes()).hexdigest()
        (run / MANIFEST_PATH).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=error):
        verify_query_diagnostics(run)

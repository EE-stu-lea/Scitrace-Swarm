"""Validate statistical pairing and fail-closed artifact audit behavior."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from examples.scitrace_research.workflow.robustness_analysis import (
    FEATURES, allocation_diagnostics, analyze_bundle, effect_diagnostics,
    interval, load_dataset, paired_bootstrap_matrix, sha256,
)


def write_dataset(path, *, repeats=1, scale=1.):
    path.mkdir(parents=True)
    baseline = {2: [.5, 0, 1, 0], 3: [.5, 0, 1, .5]}
    candidate = {2: [0, 0, 1, .5], 3: [.5, 0, 1, 1]}
    allocations = {2: [1, 2, 2, 3], 3: [2, 3, 3, 4]}
    rows = []
    for budget in (2, 3):
        for repeat in range(repeats):
            for q in range(4):
                rows.append({"query_id": f"{repeat}-{q}", "budget_target": budget,
                             "allocated_budget": allocations[budget][q],
                             "candidate_recall": scale * candidate[budget][q],
                             "convex_fixed_recall": scale * baseline[budget][q],
                             **{f: q / 10 for f in FEATURES}})
    with (path / "per_query_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"config": {"budgets": [2, 3], "bootstrap_samples": 1000},
                "calibration": {"maximum_budget": 10}, "primary_budget": 3,
                "dataset_statistics": {"test_queries": 4 * repeats}, "retrievers": ["synthetic"],
                "methods_by_budget": {str(b): {
                    "knn_calibration": {"recall": np.mean(candidate[b]) * scale, "mean_budget": b,
                                        "total_documents": 4 * repeats * b},
                    "convex_fixed": {"recall": np.mean(baseline[b]) * scale, "mean_budget": b,
                                     "total_documents": 4 * repeats * b},
                    "oracle_allocation": {"recall": 1.}} for b in (2, 3)}}
    (path / "results.json").write_text(json.dumps(metadata))
    return metadata


def rewrite_rows(path, edit):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    edit(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_bootstrap_pairs_methods_and_preserves_cross_budget_dependence():
    baseline = np.arange(5.)[:, None] + np.asarray([0., 3.])
    candidate = baseline + [.2, -.1]
    draws = paired_bootstrap_matrix(candidate, baseline, samples=333, seed=7)
    assert draws["difference"] == pytest.approx(np.tile([.2, -.1], (333, 1)))
    assert draws["baseline"][:, 1] - draws["baseline"][:, 0] == pytest.approx(np.full(333, 3.))
    repeated = paired_bootstrap_matrix(candidate, baseline, samples=333, seed=7, chunk_size=17)
    assert np.array_equal(draws["candidate"], repeated["candidate"])


def test_family_interval_is_wider_and_constant_effect_has_zero_width():
    draws = np.linspace(-2, 3, 1000)
    low, high = interval(draws)
    family_low, family_high = interval(draws, family_size=16)
    assert family_low < low < high < family_high
    assert interval(np.full(1000, .125), family_size=16) == (.125, .125)


def test_gain_loss_diagnostics_keep_ties_and_recover_mean_difference():
    stats = effect_diagnostics([1, .5, .2, 0], [0, .5, .4, .1])
    assert (stats["wins"], stats["ties"], stats["losses"]) == (1, 1, 2)
    assert stats["conditional_win_fraction"] == pytest.approx(1 / 3)
    assert stats["gain_contribution"] + stats["loss_contribution"] == pytest.approx(.175)
    assert effect_diagnostics([1, 1], [1, 1])["conditional_win_fraction"] is None


def test_allocation_accounting_and_dispersion():
    stats = allocation_diagnostics([1, 2, 3, 6], 3)
    assert stats["total_allocated_documents"] == stats["fixed_total_documents"] == 12
    assert stats["documents_reallocated"] == 3
    assert stats["reallocated_fraction_of_total"] == .25
    assert stats["allocation_gini"] == pytest.approx(1 / 3)
    assert allocation_diagnostics([3, 3, 3, 3], 3)["allocation_gini"] == 0


@pytest.mark.parametrize("mutation,reason", [
    (lambda rows: rows.append(rows[0].copy()), "duplicate"),
    (lambda rows: rows.pop(), "same nonempty query set"),
    (lambda rows: rows[0].update(allocated_budget="0"), "range"),
    (lambda rows: rows[0].update(allocated_budget="2"), "equal depth"),
    (lambda rows: rows[0].update(candidate_recall="nan"), "finite"),
    (lambda rows: rows[4].update(jaccard_top10=".123"), "feature changes"),
])
def test_loader_rejects_corrupted_pairing_and_outcomes(tmp_path, mutation, reason):
    write_dataset(tmp_path / "dataset")
    rewrite_rows(tmp_path / "dataset/per_query_results.csv", mutation)
    with pytest.raises(ValueError, match=reason):
        load_dataset(tmp_path / "dataset")


def test_loader_rejects_aggregate_disagreement(tmp_path):
    metadata = write_dataset(tmp_path / "dataset")
    metadata["methods_by_budget"]["2"]["knn_calibration"]["recall"] += .01
    (tmp_path / "dataset/results.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="recomputed"):
        load_dataset(tmp_path / "dataset")


def test_macro_weights_datasets_equally_and_preserves_inputs(tmp_path):
    stats = {}
    for name, repeats, scale in (("scifact", 1, 1.), ("nfcorpus", 2, .5), ("fiqa_lexical", 1, 1.)):
        stats[name] = write_dataset(tmp_path / name, repeats=repeats, scale=scale)
    rows = [{"dataset": name, "budget": 3, "candidate_recall": stats[name]["methods_by_budget"]["3"]["knn_calibration"]["recall"],
             "convex_fixed_recall": stats[name]["methods_by_budget"]["3"]["convex_fixed"]["recall"],
             "difference": stats[name]["methods_by_budget"]["3"]["knn_calibration"]["recall"] - stats[name]["methods_by_budget"]["3"]["convex_fixed"]["recall"]}
            for name in ("scifact", "nfcorpus")]
    (tmp_path / "aggregate.json").write_text(json.dumps({
        "datasets": rows, "macro_candidate_recall": np.mean([r["candidate_recall"] for r in rows]),
        "macro_convex_fixed_recall": np.mean([r["convex_fixed_recall"] for r in rows]),
        "macro_difference": np.mean([r["difference"] for r in rows])}))
    before = {str(p): sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    summary = analyze_bundle(tmp_path, samples=200, seed=42)
    after = {str(p): sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    assert summary["macro_effects"][1]["difference"] == pytest.approx(.09375)
    assert summary["config"]["comparison_family_size"] == 8
    assert summary["validation"]["status"] == "passed"
    json.dumps(summary, allow_nan=False)

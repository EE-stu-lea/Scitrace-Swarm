from __future__ import annotations

import json
from pathlib import Path

import pytest

from jiuwenswarm.agents.harness.research.claim_compiler import infer_claim_status
from jiuwenswarm.agents.harness.research.provenance.models import ClaimStatus, ExperimentArtifact

from examples.scitrace_research.workflow.daber_experiment import (
    allocate_budget,
    oracle_budget_allocation,
    ranking_disagreement,
    run_experiment,
    spearman_correlation,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _dataset(root: Path) -> Path:
    _write_jsonl(
        root / "corpus.jsonl",
        [
            {"_id": "d1", "title": "Protein signaling", "text": "Kinase signaling activates the target protein."},
            {"_id": "d2", "title": "Viral transmission", "text": "Airborne particles transmit the respiratory virus."},
            {"_id": "d3", "title": "Control", "text": "The placebo group showed no measurable change."},
            {"_id": "d4", "title": "Unrelated", "text": "Galaxies contain stars and interstellar dust."},
        ],
    )
    _write_jsonl(
        root / "queries.jsonl",
        [
            {"_id": "q1", "text": "Target protein activation by kinase signaling"},
            {"_id": "q2", "text": "Respiratory virus transmission in airborne particles"},
            {"_id": "q3", "text": "No change was measured in the placebo group"},
        ],
    )
    qrels = root / "qrels" / "test.tsv"
    qrels.parent.mkdir(parents=True)
    qrels.write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\nq3\td3\t1\n",
        encoding="utf-8",
    )
    return root


def test_disagreement_and_budget_are_bounded() -> None:
    rankings = [
        [("a", 1.0), ("b", 0.5)],
        [("a", 1.0), ("c", 0.5)],
        [("d", 1.0), ("c", 0.5)],
    ]
    disagreement = ranking_disagreement(rankings, depth=2)
    assert 0.0 < disagreement <= 1.0
    allocation = allocate_budget({"q1": 0.9, "q2": 0.2, "q3": 0.1}, average_budget=3)
    assert sum(allocation.values()) == 9
    assert all(1 <= value <= 10 for value in allocation.values())
    assert allocation["q1"] >= allocation["q3"]
    assert spearman_correlation([1.0, 2.0, 3.0], [4.0, 8.0, 12.0]) == pytest.approx(1.0)


def test_oracle_uses_exact_budget_and_improves_recall_opportunity() -> None:
    rankings = {
        "q1": [("x", 1.0), ("gold1", 0.9), ("z", 0.8)],
        "q2": [("gold2", 1.0), ("y", 0.9), ("z", 0.8)],
    }
    qrels = {"q1": {"gold1": 1}, "q2": {"gold2": 1}}
    allocation = oracle_budget_allocation(rankings, qrels, average_budget=2, maximum_budget=3)
    assert sum(allocation.values()) == 4
    assert allocation["q1"] >= 2
    assert allocation["q2"] >= 1


def test_artifact_claim_status_overrides_scale_dependent_metric_threshold() -> None:
    artifact = ExperimentArtifact(
        experiment_id="calibrated",
        file_path="results.json",
        sha256="abc",
        command="python run_experiment.py",
        seed=42,
        dataset="SciFact+NFCorpus",
        metric_name="macro_evidence_recall",
        metric_value=0.436984,
        metadata={"claim_status": "inconclusive"},
    )
    assert infer_claim_status([artifact]) is ClaimStatus.INCONCLUSIVE


def test_daber_experiment_writes_auditable_results(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "tiny_scifact")
    output = tmp_path / "results.json"
    payload = run_experiment(
        dataset_dir=dataset,
        split="test",
        budgets=[2, 3],
        seed=7,
        bootstrap_samples=100,
        output_path=output,
    )
    assert output.is_file()
    assert output.with_name("per_query_results.csv").is_file()
    assert payload["experiment_id"] == "daber_scientific_evidence_retrieval"
    assert payload["dataset_statistics"] == {"documents": 4, "queries": 3, "positive_qrels": 3}
    assert set(payload["methods_by_budget"]["3"]) >= {
        "bm25_fixed",
        "rrf_fixed",
        "random_allocation",
        "margin_allocation",
        "daber",
        "daber_without_char",
        "oracle_allocation",
    }
    for method in payload["methods_by_budget"]["3"].values():
        assert method["total_documents"] == 9
        assert 0.0 <= method["recall"] <= 1.0
    assert payload["data_files"]["corpus_sha256"]
    assert "disagreement_vs_first_relevant_rank" in payload["disagreement_diagnostics"]

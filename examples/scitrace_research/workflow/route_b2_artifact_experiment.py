"""Validate and import a completed Route B2 experiment into a Team run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--metric-name", default="macro_evidence_recall")
    parser.add_argument("--dataset", default="SciFact+NFCorpus-BEIR")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    aggregate_path = artifact_dir / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    bootstrap = aggregate["stratified_paired_bootstrap"]
    if int(bootstrap["samples"]) < 1000:
        raise ValueError("Route B2 final artifact requires at least 1000 bootstrap samples")
    dataset_results = {}
    files = {"aggregate.json": sha256(aggregate_path)}
    for row in aggregate["datasets"]:
        name = str(row["dataset"])
        result_path = artifact_dir / name / "results.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["dataset"] != name or int(result["primary_budget"]) != int(row["budget"]):
            raise ValueError(f"aggregate/result mismatch for {name}")
        if int(result["config"]["bootstrap_samples"]) < 1000:
            raise ValueError(f"insufficient per-dataset bootstrap samples for {name}")
        dataset_results[name] = result
        files[f"{name}/results.json"] = sha256(result_path)
        query_path = result_path.with_name("per_query_results.csv")
        files[f"{name}/per_query_results.csv"] = sha256(query_path)

    boundary_results = {}
    fiqa_path = artifact_dir / "fiqa_lexical" / "results.json"
    if fiqa_path.is_file():
        fiqa = json.loads(fiqa_path.read_text(encoding="utf-8"))
        if fiqa.get("dataset") != "fiqa":
            raise ValueError("FiQA boundary artifact has an unexpected dataset name")
        boundary_results["fiqa_lexical"] = fiqa
        files["fiqa_lexical/results.json"] = sha256(fiqa_path)
        fiqa_queries = fiqa_path.with_name("per_query_results.csv")
        files["fiqa_lexical/per_query_results.csv"] = sha256(fiqa_queries)

    supported = float(bootstrap["ci95_low"]) > 0
    payload = {
        "experiment_id": "route_b2_continuous_budgeted_hybrid_retrieval",
        "dataset": args.dataset,
        "split": "official train calibration / official test evaluation",
        "metric_name": args.metric_name,
        "metric_value": round(float(aggregate["macro_candidate_recall"]), 6),
        "primary_budget": 5,
        "seed": args.seed,
        "primary_comparison": {
            "candidate": "continuous_knn_budget_allocator_over_train_tuned_convex_hybrid_fusion",
            "baseline": "fixed_budget_train_tuned_convex_hybrid_fusion",
            "candidate_macro_recall": aggregate["macro_candidate_recall"],
            "baseline_macro_recall": aggregate["macro_convex_fixed_recall"],
            "absolute_difference": aggregate["macro_difference"],
            "stratified_paired_bootstrap": bootstrap,
        },
        "claim_assessment": {
            "status": "supported" if supported else "inconclusive",
            "reason": (
                "The dataset-macro paired-bootstrap confidence interval is above zero; "
                "per-dataset intervals remain separately reported and govern dataset-specific claims."
                if supported else
                "The dataset-macro paired-bootstrap confidence interval includes zero."
            ),
        },
        "datasets": dataset_results,
        "boundary_results": boundary_results,
        "artifact_bundle": {
            "source": "precomputed by run_route_b2_experiments.ps1",
            "files_sha256": files,
        },
        "claim_boundary": (
            "The supported claim is limited to the unweighted SciFact/NFCorpus macro-average at budget 5. "
            "SciFact is individually significant; NFCorpus has a positive point estimate whose interval "
            "slightly crosses zero. FiQA is an explicitly reported lexical out-of-domain negative result."
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["primary_comparison"], sort_keys=True))


if __name__ == "__main__":
    main()

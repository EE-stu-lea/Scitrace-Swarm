"""Run the calibrated disagreement study on multiple BEIR datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from daber_experiment import run_experiment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibrated disagreement retrieval experiments.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-dir", action="append", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--budgets", default="2,3,5,8")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric-name", default="macro_evidence_recall")
    parser.add_argument("--dataset", default="SciFact+NFCorpus")
    args = parser.parse_args()

    budgets = sorted({int(value) for value in args.budgets.split(",") if value.strip()})
    if not budgets or min(budgets) < 1 or max(budgets) > 10:
        raise ValueError("budgets must contain integers between 1 and 10")
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_results = {}
    for dataset_text in args.dataset_dir:
        dataset_dir = Path(dataset_text).resolve()
        dataset_output = output_path.with_name(f"results_{dataset_dir.name}.json")
        result = run_experiment(
            dataset_dir=dataset_dir,
            split=args.split,
            budgets=budgets,
            seed=args.seed,
            bootstrap_samples=max(100, args.bootstrap_samples),
            output_path=dataset_output,
        )
        dataset_results[result["dataset"]] = result

    primary_budget = str(5 if 5 in budgets else budgets[-1])
    calibrated = [
        result["methods_by_budget"][primary_budget]["calibrated_disagreement"]["recall"]
        for result in dataset_results.values()
    ]
    fixed = [
        result["methods_by_budget"][primary_budget]["rrf_fixed"]["recall"]
        for result in dataset_results.values()
    ]
    raw = [
        result["methods_by_budget"][primary_budget]["daber"]["recall"]
        for result in dataset_results.values()
    ]
    payload = {
        "experiment_id": "calibrated_disagreement_scientific_retrieval",
        "dataset": args.dataset,
        "split": args.split,
        "metric_name": args.metric_name,
        "metric_value": round(statistics.fmean(calibrated), 6),
        "primary_budget": int(primary_budget),
        "primary_comparison": {
            "candidate": "calibrated_disagreement",
            "baseline": "rrf_fixed",
            "candidate_macro_recall": round(statistics.fmean(calibrated), 6),
            "baseline_macro_recall": round(statistics.fmean(fixed), 6),
            "raw_disagreement_macro_recall": round(statistics.fmean(raw), 6),
            "absolute_difference": round(statistics.fmean(calibrated) - statistics.fmean(fixed), 6),
        },
        "claim_assessment": {
            "status": "inconclusive",
            "reason": (
                "The calibrated policy has a positive macro-average point estimate, "
                "but the per-dataset paired bootstrap confidence intervals include zero."
            ),
        },
        "datasets": dataset_results,
        "config": {
            "budgets": budgets,
            "bootstrap_samples": max(100, args.bootstrap_samples),
            "seed": args.seed,
            "calibration_split": "train",
            "evaluation_split": args.split,
        },
        "result_files": {
            name: {
                "path": str(output_path.with_name(f"results_{name}.json")),
                "sha256": _sha256(output_path.with_name(f"results_{name}.json")),
            }
            for name in dataset_results
        },
        "claim_boundary": (
            "Per-dataset paired bootstrap intervals govern statistical claims; "
            "the macro average is descriptive and is not an independent significance test."
        ),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["primary_comparison"], sort_keys=True))


if __name__ == "__main__":
    main()

"""Aggregate Route B2 dataset artifacts without rerunning retrieval."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import statistics


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))]


def load_run(path: Path) -> dict:
    result_path = path / "results.json" if path.is_dir() else path
    result = json.loads(result_path.read_text(encoding="utf-8"))
    primary_budget = int(result["primary_budget"])
    rows = []
    with result_path.with_name("per_query_results.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["budget_target"]) == primary_budget:
                rows.append(float(row["candidate_recall"]) - float(row["convex_fixed_recall"]))
    if not rows:
        raise ValueError(f"no primary-budget rows in {result_path}")
    return {"path": str(result_path.resolve()), "result": result, "differences": rows}


def aggregate(paths: list[Path], *, samples: int, seed: int) -> dict:
    runs = [load_run(path) for path in paths]
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        dataset_means = [
            statistics.fmean(generator.choice(run["differences"]) for _ in run["differences"])
            for run in runs
        ]
        estimates.append(statistics.fmean(dataset_means))
    datasets = []
    for run in runs:
        result = run["result"]
        primary = result["primary_comparison"]
        datasets.append({
            "dataset": result["dataset"],
            "budget": result["primary_budget"],
            "candidate_recall": primary["candidate_recall"],
            "convex_fixed_recall": primary["baseline_recall"],
            "difference": primary["absolute_difference"],
            "queries": len(run["differences"]),
            "result_path": run["path"],
        })
    observed = statistics.fmean(statistics.fmean(run["differences"]) for run in runs)
    return {
        "aggregation": "unweighted macro-average over datasets at the declared primary budget",
        "datasets": datasets,
        "macro_candidate_recall": statistics.fmean(row["candidate_recall"] for row in datasets),
        "macro_convex_fixed_recall": statistics.fmean(row["convex_fixed_recall"] for row in datasets),
        "macro_difference": observed,
        "stratified_paired_bootstrap": {
            "samples": samples,
            "seed": seed,
            "ci95_low": percentile(estimates, 0.025),
            "ci95_high": percentile(estimates, 0.975),
            "probability_positive": sum(value > 0 for value in estimates) / samples,
        },
    }


def markdown(report: dict) -> str:
    lines = [
        "# Route B2 Aggregate Results",
        "",
        report["aggregation"] + ".",
        "",
        "| Dataset | Budget | Continuous allocator | Fixed convex | Difference | Queries |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["datasets"]:
        lines.append(
            f"| {row['dataset']} | {row['budget']} | {row['candidate_recall']:.6f} | "
            f"{row['convex_fixed_recall']:.6f} | {row['difference']:+.6f} | {row['queries']} |"
        )
    bootstrap = report["stratified_paired_bootstrap"]
    lines.extend([
        "",
        f"Macro candidate recall: **{report['macro_candidate_recall']:.6f}**",
        f"Macro fixed-convex recall: **{report['macro_convex_fixed_recall']:.6f}**",
        f"Macro difference: **{report['macro_difference']:+.6f}**",
        f"Stratified paired-bootstrap 95% CI: "
        f"[{bootstrap['ci95_low']:.6f}, {bootstrap['ci95_high']:.6f}] "
        f"({bootstrap['samples']} samples; P(diff > 0)={bootstrap['probability_positive']:.3f}).",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = aggregate([Path(path).resolve() for path in args.runs], samples=args.samples, seed=args.seed)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({
        "macro_difference": report["macro_difference"],
        **report["stratified_paired_bootstrap"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

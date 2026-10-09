"""Reanalyze saved Route B2 query outcomes without rerunning retrieval.

All uncertainty is conditional on the observed batch and its fixed allocation.
The resampling unit is the query; the same indices are used across budgets and
paired methods. Dataset strata are resampled independently and macro estimates
weight the two scientific datasets equally. Bonferroni percentile intervals
cover the explicitly declared comparison family approximately; they are not
an exact finite-sample guarantee and cannot remedy data-dependent model choice.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

DATASET_KEYS = ("scifact", "nfcorpus", "fiqa_lexical")
MACRO_KEYS = ("scifact", "nfcorpus")
FEATURES = ("jaccard_top5", "jaccard_top10", "jaccard_top20", "rbo_top20",
            "score_dispersion", "rrf_margin", "log_query_length")
ROUNDING_TOLERANCE = 5.1e-7  # Historical JSON rounds aggregate metrics to six decimals.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(actual: float, expected: float, label: str, *, tolerance: float = ROUNDING_TOLERANCE) -> None:
    if not math.isfinite(expected) or abs(actual - expected) > tolerance:
        raise ValueError(f"{label}: recomputed {actual:.12g}, stored {expected:.12g}")


def load_dataset(directory: Path) -> dict:
    """Validate pairing, budgets, exact capacity, and rounded aggregate recall."""
    metadata = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    budgets = tuple(sorted(int(value) for value in metadata["config"]["budgets"]))
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("configured budgets must be nonempty and unique")
    rows = {}
    with (directory / "per_query_results.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            budget, query = int(row["budget_target"]), row["query_id"]
            if budget not in budgets or not query or (budget, query) in rows:
                raise ValueError(f"invalid/duplicate query-budget pair: {budget}, {query}")
            parsed = {name: float(row[name]) for name in FEATURES}
            parsed.update(candidate=float(row["candidate_recall"]),
                          baseline=float(row["convex_fixed_recall"]),
                          allocation=int(row["allocated_budget"]))
            if not all(math.isfinite(value) for value in parsed.values()):
                raise ValueError("query records must contain finite numbers")
            if not (0 <= parsed["candidate"] <= 1 and 0 <= parsed["baseline"] <= 1):
                raise ValueError("recall must be between zero and one")
            maximum = int(metadata["calibration"]["maximum_budget"])
            if not 1 <= parsed["allocation"] <= maximum:
                raise ValueError("allocated depth is outside implemented range 1..maximum_budget")
            delta = parsed["candidate"] - parsed["baseline"]
            if (parsed["allocation"] - budget) * delta < -1e-12:
                raise ValueError("recall direction contradicts depth change on the same fused ranking")
            if parsed["allocation"] == budget and abs(delta) > 1e-12:
                raise ValueError("equal depth on the same ranking must yield equal recall")
            rows[budget, query] = parsed
    ids_by_budget = {budget: {query for b, query in rows if b == budget} for budget in budgets}
    query_ids = sorted(ids_by_budget[budgets[0]])
    if not query_ids or any(ids_by_budget[b] != set(query_ids) for b in budgets):
        raise ValueError("all budgets must contain exactly the same nonempty query set")
    if len(query_ids) != int(metadata["dataset_statistics"]["test_queries"]):
        raise ValueError("query count differs from recorded test split")
    candidate = np.asarray([[rows[b, q]["candidate"] for b in budgets] for q in query_ids])
    baseline = np.asarray([[rows[b, q]["baseline"] for b in budgets] for q in query_ids])
    allocations = np.asarray([[rows[b, q]["allocation"] for b in budgets] for q in query_ids])
    if np.any(np.diff(baseline, axis=1) < -1e-12):
        raise ValueError("fixed-depth recall must be nondecreasing with budget")
    for column, budget in enumerate(budgets):
        if int(allocations[:, column].sum()) != budget * len(query_ids):
            raise ValueError(f"budget {budget}: total allocated capacity differs from fixed baseline")
        methods = metadata["methods_by_budget"][str(budget)]
        for name, matrix in (("knn_calibration", candidate), ("convex_fixed", baseline)):
            _close(float(matrix[:, column].mean()), float(methods[name]["recall"]),
                   f"{directory.name}/budget={budget}/{name}/recall")
            _close(float(methods[name]["mean_budget"]), budget, "stored mean budget")
            _close(float(methods[name]["total_documents"]), budget * len(query_ids),
                   "stored total document capacity")
    for query in query_ids:
        for feature in FEATURES:
            if any(rows[b, query][feature] != rows[budgets[0], query][feature] for b in budgets):
                raise ValueError(f"query feature changes across budgets: {query}/{feature}")
    return {"metadata": metadata, "budgets": budgets, "query_ids": query_ids,
            "candidate": candidate, "baseline": baseline, "allocations": allocations,
            "features": {name: np.asarray([rows[budgets[0], q][name] for q in query_ids])
                         for name in FEATURES}}


def paired_bootstrap_matrix(candidate, baseline, *, samples: int, seed: int,
                            chunk_size: int = 256) -> dict[str, np.ndarray]:
    """Resample paired query vectors jointly over all budget columns."""
    candidate, baseline = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    if candidate.ndim != 2 or candidate.shape != baseline.shape or not candidate.shape[0]:
        raise ValueError("paired input must be matching nonempty query-by-budget matrices")
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("bootstrap input must be finite")
    if samples < 100 or chunk_size < 1:
        raise ValueError("at least 100 resamples and a positive chunk size are required")
    generator = np.random.default_rng(seed)
    n_queries, n_budgets = candidate.shape
    candidate_draws = np.empty((samples, n_budgets))
    baseline_draws = np.empty_like(candidate_draws)
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = generator.integers(0, n_queries, size=(stop - start, n_queries))
        candidate_draws[start:stop] = candidate[indices].mean(axis=1)
        baseline_draws[start:stop] = baseline[indices].mean(axis=1)
    return {"candidate": candidate_draws, "baseline": baseline_draws,
            "difference": candidate_draws - baseline_draws}


def interval(draws, *, family_size: int = 1, alpha: float = .05) -> tuple[float, float]:
    if family_size < 1 or not 0 < alpha < 1:
        raise ValueError("invalid interval family size or alpha")
    tail = alpha / (2 * family_size)
    low, high = np.quantile(draws, [tail, 1 - tail], method="linear")
    return float(low), float(high)


def effect_diagnostics(candidate, baseline) -> dict:
    candidate, baseline = np.asarray(candidate), np.asarray(baseline)
    differences = candidate - baseline
    wins, losses = int((differences > 1e-12).sum()), int((differences < -1e-12).sum())
    ties = len(differences) - wins - losses
    sd = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    return {"queries": len(differences), "wins": wins, "ties": ties, "losses": losses,
            "win_fraction": wins / len(differences), "loss_fraction": losses / len(differences),
            "tie_fraction": ties / len(differences),
            "conditional_win_fraction": wins / (wins + losses) if wins + losses else None,
            "mean_gain_on_winners": float(differences[differences > 1e-12].mean()) if wins else None,
            "mean_loss_on_losers": float(differences[differences < -1e-12].mean()) if losses else None,
            "paired_difference_sd": sd,
            "paired_standardized_difference_dz": float(differences.mean() / sd) if sd else None,
            "gain_contribution": float(np.maximum(differences, 0).mean()),
            "loss_contribution": float(np.minimum(differences, 0).mean())}


def allocation_diagnostics(allocations, target: int) -> dict:
    allocations = np.asarray(allocations, dtype=float)
    ordered = np.sort(allocations)
    n = len(ordered)
    if not n or np.any(ordered <= 0):
        raise ValueError("allocation diagnostics require positive allocated depths")
    gini = 2 * float(np.dot(np.arange(1, n + 1), ordered)) / (n * float(ordered.sum())) - (n + 1) / n
    return {"queries": n, "total_allocated_documents": int(ordered.sum()),
            "fixed_total_documents": n * target, "mean_depth": float(ordered.mean()),
            "sd_depth": float(ordered.std()), "minimum_depth": int(ordered[0]),
            "median_depth": float(np.median(ordered)), "maximum_depth": int(ordered[-1]),
            "allocation_gini": max(0., gini),
            "below_fixed_fraction": float((ordered < target).mean()),
            "equal_fixed_fraction": float((ordered == target).mean()),
            "above_fixed_fraction": float((ordered > target).mean()),
            "documents_reallocated": int(np.maximum(allocations - target, 0).sum()),
            "reallocated_fraction_of_total": float(np.maximum(allocations - target, 0).sum() / ordered.sum())}


def _stratum(dataset, budget, group_type, group_name, mask, candidate, baseline, allocations):
    count = int(mask.sum())
    if not count:
        return None
    return {"dataset": dataset, "budget": budget, "group_type": group_type,
            "group": group_name, "queries": count,
            "candidate_recall": float(candidate[mask].mean()),
            "baseline_recall": float(baseline[mask].mean()),
            "difference": float((candidate[mask] - baseline[mask]).mean()),
            "mean_depth": float(allocations[mask].mean()),
            "wins": int((candidate[mask] > baseline[mask] + 1e-12).sum()),
            "losses": int((candidate[mask] < baseline[mask] - 1e-12).sum())}


def analyze_bundle(artifact_dir: Path, *, samples: int = 50_000, seed: int = 20261008) -> dict:
    datasets = {name: load_dataset(artifact_dir / name) for name in DATASET_KEYS}
    budgets = datasets[DATASET_KEYS[0]]["budgets"]
    if any(data["budgets"] != budgets for data in datasets.values()):
        raise ValueError("all datasets must share the same evaluated budgets")
    family_size = (len(datasets) + 1) * len(budgets)
    effects, allocations_out, histograms, strata = [], [], [], []
    draws_by_dataset = {}
    for index, (name, data) in enumerate(datasets.items()):
        draws = paired_bootstrap_matrix(data["candidate"], data["baseline"],
                                        samples=samples, seed=seed + index)
        draws_by_dataset[name] = draws
        # Feature-only cut points; post-hoc groups are descriptive, without subgroup significance tests.
        cutpoints = np.quantile(data["features"]["jaccard_top10"], [.25, .5, .75])
        quartiles = np.searchsorted(cutpoints, data["features"]["jaccard_top10"], side="right")
        for column, budget in enumerate(budgets):
            cand, base = data["candidate"][:, column], data["baseline"][:, column]
            depth = data["allocations"][:, column]
            low, high = interval(draws["difference"][:, column])
            adjusted_low, adjusted_high = interval(draws["difference"][:, column], family_size=family_size)
            c_low, c_high = interval(draws["candidate"][:, column])
            b_low, b_high = interval(draws["baseline"][:, column])
            delta = float((cand - base).mean())
            oracle = data["metadata"]["methods_by_budget"][str(budget)]["oracle_allocation"]["recall"]
            headroom = float(oracle - base.mean())
            effects.append({"dataset": name, "budget": budget, "candidate_recall": float(cand.mean()),
                            "baseline_recall": float(base.mean()), "difference": delta,
                            "relative_difference": delta / float(base.mean()) if base.mean() else None,
                            "ci95_low": low, "ci95_high": high,
                            "ci_family95_low": adjusted_low, "ci_family95_high": adjusted_high,
                            "candidate_ci95_low": c_low, "candidate_ci95_high": c_high,
                            "baseline_ci95_low": b_low, "baseline_ci95_high": b_high,
                            "oracle_recall_descriptive": oracle,
                            "oracle_headroom_recovered": delta / headroom if headroom > 1e-12 else None,
                            **effect_diagnostics(cand, base)})
            allocations_out.append({"dataset": name, "budget": budget, **allocation_diagnostics(depth, budget)})
            for value in range(1, int(data["metadata"]["calibration"]["maximum_budget"]) + 1):
                histograms.append({"dataset": name, "budget": budget, "allocated_depth": value,
                                   "queries": int((depth == value).sum()),
                                   "fraction": float((depth == value).mean())})
            groups = [("allocation_change", "below_fixed", depth < budget),
                      ("allocation_change", "equal_fixed", depth == budget),
                      ("allocation_change", "above_fixed", depth > budget)]
            groups += [("jaccard_disagreement_quartile", f"Q{q + 1}", quartiles == q) for q in range(4)]
            for kind, group, mask in groups:
                row = _stratum(name, budget, kind, group, mask, cand, base, depth)
                if row is not None:
                    strata.append(row)
    macro = []
    for column, budget in enumerate(budgets):
        draws = {key: np.mean([draws_by_dataset[name][key][:, column] for name in MACRO_KEYS], axis=0)
                 for key in ("candidate", "baseline", "difference")}
        means = {key: float(np.mean([datasets[name][key][:, column].mean() for name in MACRO_KEYS]))
                 for key in ("candidate", "baseline")}
        low, high = interval(draws["difference"])
        adj_low, adj_high = interval(draws["difference"], family_size=family_size)
        macro.append({"dataset": "scientific_macro", "budget": budget,
                      "candidate_recall": means["candidate"], "baseline_recall": means["baseline"],
                      "difference": means["candidate"] - means["baseline"],
                      "ci95_low": low, "ci95_high": high,
                      "ci_family95_low": adj_low, "ci_family95_high": adj_high,
                      "queries": sum(len(datasets[name]["query_ids"]) for name in MACRO_KEYS)})
    original_aggregate = json.loads((artifact_dir / "aggregate.json").read_text(encoding="utf-8"))
    primary_budget = int(datasets["scifact"]["metadata"]["primary_budget"])
    primary = next(row for row in macro if row["budget"] == primary_budget)
    for new, old in (("candidate_recall", "macro_candidate_recall"),
                     ("baseline_recall", "macro_convex_fixed_recall"),
                     ("difference", "macro_difference")):
        _close(primary[new], float(original_aggregate[old]), f"aggregate/{old}")
    for row in original_aggregate["datasets"]:
        match = next(item for item in effects if item["dataset"] == row["dataset"] and item["budget"] == row["budget"])
        for new, old in (("candidate_recall", "candidate_recall"), ("baseline_recall", "convex_fixed_recall"),
                         ("difference", "difference")):
            _close(match[new], float(row[old]), f"aggregate/{row['dataset']}/{old}",
                   tolerance=1.01e-6 if new == "difference" else ROUNDING_TOLERANCE)
    sources = [artifact_dir / "aggregate.json"]
    sources += [artifact_dir / name / filename for name in DATASET_KEYS
                for filename in ("results.json", "per_query_results.csv")]
    return {"schema_version": 1,
            "analysis_kind": "conditional_saved_query_outcome_reanalysis",
            "config": {"samples": samples, "seed": seed, "budgets": list(budgets),
                       "primary_analysis_budget": primary_budget, "macro_datasets": list(MACRO_KEYS),
                       "comparison_family_size": family_size,
                       "family_definition": "all dataset-budget recall differences plus scientific-macro budget differences",
                       "marginal_interval": "95% paired query percentile bootstrap; numpy linear quantiles",
                       "family_interval": f"Bonferroni percentile intervals at marginal coverage {1 - .05 / family_size:.8f}",
                       "resampling": "same query indices across budgets/methods; independent dataset strata; equal-weight scientific macro",
                       "dataset_seeds": {name: seed + i for i, name in enumerate(DATASET_KEYS)}},
            "validation": {"status": "passed", "source_aggregate_matches_query_recomputation": True,
                           "source_rounding_tolerance": ROUNDING_TOLERANCE,
                           "difference_of_rounded_means_tolerance": 1.01e-6,
                           "same_query_sets_across_budgets": True, "exact_equal_total_budget": True,
                           "recall_direction_consistent_with_depth": True},
            "datasets": {name: {"dataset_statistics": data["metadata"]["dataset_statistics"],
                                 "retrievers": data["metadata"]["retrievers"],
                                 "original_bootstrap_samples": data["metadata"]["config"]["bootstrap_samples"]}
                         for name, data in datasets.items()},
            "effects": effects, "macro_effects": macro, "allocation_diagnostics": allocations_out,
            "allocation_histograms": histograms, "descriptive_strata": strata,
            "sources_sha256": {str(path.relative_to(artifact_dir)): sha256(path) for path in sources},
            "limitations": [
                "No retriever, calibrator, or allocator was rerun; input run artifacts are unchanged.",
                "Intervals condition on the fitted pipeline, observed batch, and realized joint allocation; they do not estimate performance after reallocating a fresh or resampled batch.",
                "Query-level exchangeability is a working assumption; shared allocation and correlated queries limit nominal interval interpretation.",
                "The primary budget was recorded in the earlier artifact, not preregistered; multiplicity adjustment does not remove model-selection or researcher-selection bias.",
                "Bonferroni correction applies to the stated 16-comparison family for four budgets; percentile bootstrap coverage is approximate.",
                "FiQA uses its own training data and lexical retrieval; it is an additional domain evaluation, not unseen-domain transfer.",
                "Only candidate/fixed-convex recall pairs are saved. New paired MRR, nDCG, RRF, adaptive-baseline, or feature-ablation intervals cannot be recovered.",
                "Gain/loss, allocation dispersion, oracle-headroom, and subgroup summaries are descriptive; they do not establish causal mechanisms or demographic fairness.",
                "Equal document capacity does not imply equal token count, compute, latency, money, or downstream answer quality."]}


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "robustness_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    for filename, rows in (("paired_effects.csv", summary["effects"] + summary["macro_effects"]),
                           ("allocation_diagnostics.csv", summary["allocation_diagnostics"]),
                           ("allocation_histograms.csv", summary["allocation_histograms"]),
                           ("query_strata.csv", summary["descriptive_strata"])):
        _write_csv(output_dir / filename, rows)
    lines = ["# Route B2: conditional outcome robustness analysis", "",
             "Saved query outcomes were reanalyzed; retrieval and allocation were not rerun. All source means and budget totals passed validation.", "",
             f"Paired bootstrap: {summary['config']['samples']:,} resamples, seed {summary['config']['seed']}. "
             f"The {summary['config']['comparison_family_size']}-comparison family covers all three datasets and the scientific macro at every recorded budget. "
             "Family intervals use Bonferroni-adjusted percentile tails. They have approximate coverage conditional on the observed allocation.", "",
             "| Dataset | Budget | Candidate | Fixed convex | Difference | Marginal 95% CI | Family 95% CI | Wins/ties/losses |",
             "|---|---:|---:|---:|---:|---|---|---|"]
    for row in summary["effects"] + summary["macro_effects"]:
        outcome = f"{row['wins']}/{row['ties']}/{row['losses']}" if "wins" in row else "—"
        lines.append(f"| {row['dataset']} | {row['budget']} | {row['candidate_recall']:.6f} | {row['baseline_recall']:.6f} | "
                     f"{row['difference']:+.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | "
                     f"[{row['ci_family95_low']:.6f}, {row['ci_family95_high']:.6f}] | {outcome} |")
    lines += ["", "Allocation diagnostics describe who gains/loses context capacity. Gini measures dispersion only. "
              "Feature quartiles and allocation-change groups are descriptive, with no subgroup hypothesis tests.", "", "## Limits", ""]
    lines += [f"- {line}" for line in summary["limitations"]]
    (output_dir / "robustness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    artifact_dir, output_dir = args.artifact_dir.resolve(), args.output_dir.resolve()
    if output_dir == artifact_dir or artifact_dir in output_dir.parents:
        parser.error("output directory must be outside the preserved input artifact directory")
    started = time.perf_counter()
    summary = analyze_bundle(artifact_dir, samples=args.samples, seed=args.seed)
    write_outputs(summary, output_dir)
    manifest = {"python_version": platform.python_version(), "numpy_version": np.__version__,
                "analysis_seconds": time.perf_counter() - started,
                "script_sha256": sha256(Path(__file__)), "input_sha256": summary["sources_sha256"],
                "output_sha256": {p.name: sha256(p) for p in sorted(output_dir.iterdir())
                                  if p.name in {"robustness_summary.json", "robustness_report.md", "paired_effects.csv",
                                                "allocation_diagnostics.csv", "allocation_histograms.csv", "query_strata.csv"}},
                "command": ["python", "-m", "examples.scitrace_research.workflow.robustness_analysis",
                            "--artifact-dir", str(args.artifact_dir), "--output-dir", str(args.output_dir),
                            "--samples", str(args.samples), "--seed", str(args.seed)]}
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"validation": summary["validation"], "macro_effects": summary["macro_effects"],
                      "analysis_seconds": manifest["analysis_seconds"]}, indent=2))


if __name__ == "__main__":
    main()

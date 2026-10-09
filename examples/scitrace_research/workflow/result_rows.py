"""Small verified result views shared by the Writer prompt and export gate."""
from __future__ import annotations

from typing import Any


def _primary_results_table_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Give the Writer exact values for a compact, auditable main-results table."""
    rows = []
    for dataset, label in (("scifact", "SciFact"), ("nfcorpus", "NFCorpus")):
        comparison = results["datasets"][dataset]["primary_comparison"]
        interval = comparison["paired_bootstrap"]
        rows.append({
            "dataset": label,
            "candidate": comparison["candidate_recall"],
            "fixed_baseline": comparison["baseline_recall"],
            "difference": comparison["absolute_difference"],
            "ci95": [interval["ci95_low"], interval["ci95_high"]],
        })
    comparison = results["primary_comparison"]
    interval = comparison["stratified_paired_bootstrap"]
    rows.append({
        "dataset": "Macro",
        "candidate": comparison["candidate_macro_recall"],
        "fixed_baseline": comparison["baseline_macro_recall"],
        "difference": comparison["absolute_difference"],
        "ci95": [interval["ci95_low"], interval["ci95_high"]],
    })
    boundary = (results.get("boundary_results") or {}).get("fiqa_lexical")
    if boundary:
        comparison = boundary["primary_comparison"]
        interval = comparison["paired_bootstrap"]
        rows.append({
            "dataset": "FiQA (lexical)",
            "candidate": comparison["candidate_recall"],
            "fixed_baseline": comparison["baseline_recall"],
            "difference": comparison["absolute_difference"],
            "ci95": [interval["ci95_low"], interval["ci95_high"]],
        })
    return rows

def _adaptive_baseline_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Measured baselines at the recorded primary budget, including negative results."""
    rows = []
    datasets = [(results["datasets"][key], label)
                for key, label in (("scifact", "SciFact"), ("nfcorpus", "NFCorpus"))]
    fiqa = (results.get("boundary_results") or {}).get("fiqa_lexical")
    if fiqa:
        datasets.append((fiqa, "FiQA (lexical)"))
    for data, label in datasets:
        primary = str(data["primary_budget"])
        methods = data["methods_by_budget"][primary]
        rows.append({
            "dataset": label,
            "raw_jaccard": methods["raw_jaccard_allocation"]["recall"],
            "binned_calibration": methods["binned_calibration"]["recall"],
            "knn_calibration": methods["knn_calibration"]["recall"],
        })
    return rows


def _train_subsampling_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize existing 80% train-query resamples without implying new seeds."""
    rows = []
    for dataset, label in (("scifact", "SciFact"), ("nfcorpus", "NFCorpus")):
        stability = results["datasets"][dataset]["train_subsampling_stability"]
        runs = stability["runs"]
        recalls = [float(item["recall"]) for item in runs]
        rows.append({
            "dataset": label,
            "resamples": len(runs),
            "training_fraction": 0.8,
            "training_queries_per_resample": runs[0]["train_queries"],
            "mean_recall": stability["recall_mean"],
            "population_sd": stability["recall_std"],
            "min_recall": min(recalls),
            "max_recall": max(recalls),
            "seeds": [item["seed"] for item in runs],
        })
    return rows

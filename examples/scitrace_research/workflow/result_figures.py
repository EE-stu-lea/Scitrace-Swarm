"""Build reproducible result figures from the imported experiment artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FIGURE_PATH = "figures/recall_by_budget.pdf"


def budget_curve_data(results: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """Return plotted values, failing if a required comparison is unavailable."""
    output: dict[str, dict[str, list[float]]] = {}
    for dataset in ("scifact", "nfcorpus"):
        methods = ((results.get("datasets") or {}).get(dataset) or {}).get("methods_by_budget") or {}
        budgets = sorted(int(key) for key in methods)
        if budgets != [2, 3, 5, 8]:
            raise ValueError(f"Missing budget sweep for {dataset}: {budgets}")
        candidate = [float(methods[str(b)]["knn_calibration"]["recall"]) for b in budgets]
        baseline = [float(methods[str(b)]["convex_fixed"]["recall"]) for b in budgets]
        output[dataset] = {"budget": budgets, "candidate": candidate, "baseline": baseline}
    return output


def write_budget_curve(results: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Save a vector plot and its exact source numbers without editing prose."""
    plotted = budget_curve_data(results)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), layout="constrained")
    for axis, dataset, title in zip(axes, ("scifact", "nfcorpus"), ("SciFact", "NFCorpus")):
        values = plotted[dataset]
        axis.plot(values["budget"], values["candidate"], "o-", label="Batch allocator", linewidth=1.8)
        axis.plot(values["budget"], values["baseline"], "s--", label="Fixed-depth fusion", linewidth=1.5)
        axis.set_title(title)
        axis.set_xticks(values["budget"])
        axis.set_xlabel("Documents admitted per query")
        axis.set_ylabel("Evidence recall")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=7)
    path = run_dir / FIGURE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="pdf", bbox_inches="tight", metadata={"Creator": "SciTrace deterministic result figure"})
    plt.close(figure)
    manifest = {
        "path": FIGURE_PATH,
        "caption_hint": "Evidence recall across document budgets on SciFact and NFCorpus. Panels use separate vertical scales. Points are measured test-set recalls; no confidence intervals are shown for this sweep.",
        "plotted_values": plotted,
        "figure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_results_sha256": hashlib.sha256(
            json.dumps(plotted, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    (run_dir / "figures" / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest

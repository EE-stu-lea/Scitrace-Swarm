"""Build reproducible result figures from the imported experiment artifact."""

from __future__ import annotations

import hashlib
import json
import math
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
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (*candidate, *baseline)):
            raise ValueError(f"{dataset} recall must be finite and in [0, 1]")
        output[dataset] = {"budget": budgets, "candidate": candidate, "baseline": baseline}
    return output


def write_budget_curve(results: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Save a vector plot and its exact source numbers without editing prose."""
    plotted = budget_curve_data(results)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = run_dir / FIGURE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # Local style avoids dependence on whichever figure happened to render
    # first in this process. Fonts remain readable at manuscript width.
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}):
        figure, axes = plt.subplots(1, 2, figsize=(6.4, 2.55), layout="constrained")
        for axis, dataset, title in zip(axes, ("scifact", "nfcorpus"), ("SciFact", "NFCorpus")):
            values = plotted[dataset]
            axis.plot(values["budget"], values["candidate"], "o-", color="#215a8e",
                      label="Batch allocator", linewidth=1.7, markersize=5.5,
                      markeredgecolor="white", markeredgewidth=0.7)
            axis.plot(values["budget"], values["baseline"], "o--", color="#cf6b32",
                      label="Fixed-depth fusion", linewidth=1.4, markersize=5.5,
                      markerfacecolor="white", markeredgewidth=1.1)
            axis.set_title(title)
            axis.set_xticks(values["budget"])
            axis.set_ylabel("Evidence recall")
            axis.grid(axis="y", color="#e0e4e8", linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)
        figure.supxlabel("Mean documents admitted per query", fontsize=9)
        axes[0].legend(loc="lower right", fontsize=8, frameon=False)
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

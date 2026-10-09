"""Build an optional effect plot from persisted, independently checkable results."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


DATASETS = (
    ("scifact", "SciFact", "#215a8e", "o"),
    ("nfcorpus", "NFCorpus", "#cf6b32", "o"),
    ("fiqa_lexical", "FiQA (lexical)", "#548264", "o"),
)
PRIMARY_BUDGET = 5
FIGURE_PATH = "figures/budget_effect.pdf"
DATA_PATH = "figures/budget_effect_data.json"
MANIFEST_PATH = "figures/budget_effect_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_budget_effects(run_dir: Path) -> dict | None:
    """Read the actual Route B2 sweep; reject incomplete or inconsistent data."""
    root = run_dir / "experiments" / "route_b2_artifacts"
    if not root.is_dir():
        return None
    aggregate_path = root / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate_by_dataset = {item["dataset"]: item for item in aggregate["datasets"]}
    sources = {aggregate_path.relative_to(run_dir).as_posix(): _sha256(aggregate_path)}
    series = []
    expected_budgets = None
    for slug, label, color, marker in DATASETS:
        path = root / slug / "results.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        sources[path.relative_to(run_dir).as_posix()] = _sha256(path)
        by_budget = result["methods_by_budget"]
        budgets = sorted(int(value) for value in by_budget)
        if expected_budgets is None:
            expected_budgets = budgets
        elif budgets != expected_budgets:
            raise ValueError("Route B2 budget sweeps use different budget grids")
        if PRIMARY_BUDGET not in budgets:
            raise ValueError("Route B2 primary budget is absent")
        differences = []
        intervals = []
        for budget in budgets:
            methods = by_budget[str(budget)]
            candidate = float(methods["knn_calibration"]["recall"])
            baseline = float(methods["convex_fixed"]["recall"])
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (candidate, baseline)):
                raise ValueError("Route B2 recall must be finite and in [0, 1]")
            differences.append(candidate - baseline)
            interval = result["paired_bootstrap_by_budget"][str(budget)]["convex_fixed"]["recall"]
            low, high = float(interval["ci95_low"]), float(interval["ci95_high"])
            observed = float(interval["observed_difference"])
            if not all(math.isfinite(x) for x in (low, high, observed)) or low > high:
                raise ValueError(f"{slug} budget {budget} has an invalid confidence interval")
            if abs(observed - (candidate - baseline)) > 0.000002:
                raise ValueError(f"{slug} budget {budget} difference differs from bootstrap")
            intervals.append([low, high])
            if budget == PRIMARY_BUDGET and slug in aggregate_by_dataset:
                primary = aggregate_by_dataset[slug]
                if abs(candidate - float(primary["candidate_recall"])) > 0.000002:
                    raise ValueError(f"{slug} primary candidate differs from aggregate")
                if abs(baseline - float(primary["convex_fixed_recall"])) > 0.000002:
                    raise ValueError(f"{slug} primary baseline differs from aggregate")
        bootstrap = result["paired_bootstrap_by_budget"][str(PRIMARY_BUDGET)]["convex_fixed"]["recall"]
        ci_low = float(bootstrap["ci95_low"])
        ci_high = float(bootstrap["ci95_high"])
        observed = float(bootstrap["observed_difference"])
        if not (math.isfinite(ci_low) and math.isfinite(ci_high) and ci_low <= ci_high):
            raise ValueError(f"{slug} has an invalid confidence interval")
        if abs(observed - differences[budgets.index(PRIMARY_BUDGET)]) > 0.000002:
            raise ValueError(f"{slug} primary difference differs from paired bootstrap")
        series.append({
            "dataset": slug,
            "label": label,
            "color": color,
            "marker": marker,
            "budgets": budgets,
            "differences": differences,
            "ci95_by_budget": intervals,
            "primary_ci95": [ci_low, ci_high],
            "bootstrap_samples": int(bootstrap["samples"]),
        })
    return {"primary_budget": PRIMARY_BUDGET, "series": series, "source_sha256": sources,
            "interval_scope": "pointwise query-paired bootstrap, conditional on the realized allocation; not simultaneous or a new-batch evaluation",
            "interval_display": "dots show observed differences; shading shows pointwise 95% intervals at evaluated budgets; lines and shading between budgets are visual guides only",
            "fiqa_scope": "separately fitted finance-domain lexical evaluation; not zero-shot transfer"}


def render_budget_effect_figure(run_dir: Path, figure_dir: Path) -> dict | None:
    data = load_budget_effects(run_dir)
    if data is None:
        return None
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "budget_effect.pdf"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.65), sharey=True, layout="constrained")
        for ax, item in zip(axes, data["series"]):
            budgets = item["budgets"]
            differences = [100 * x for x in item["differences"]]
            low = [100 * x[0] for x in item["ci95_by_budget"]]
            high = [100 * x[1] for x in item["ci95_by_budget"]]
            ax.axhline(0, color="#58616a", linewidth=0.8)
            ax.axvline(PRIMARY_BUDGET, color="#b6bbc0", linewidth=0.8, linestyle=":")
            ax.fill_between(budgets, low, high, color=item["color"], alpha=0.15, zorder=1)
            ax.plot(budgets, differences, color=item["color"], marker=item["marker"],
                    markersize=6, markeredgecolor="white", markeredgewidth=0.9,
                    linewidth=1.6, zorder=5)
            ax.set_title(item["label"], fontsize=9)
            ax.set_xticks(budgets)
            ax.grid(axis="y", color="#e0e4e8", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Recall change\n(percentage points)")
        fig.supxlabel("Mean document budget", fontsize=9)
        fig.legend(handles=[
            Line2D([], [], color="#58616a", marker="o", markersize=6,
                   markeredgecolor="white", label="Observed difference"),
            Patch(facecolor="#58616a", alpha=0.15, label="Pointwise 95% interval"),
        ], loc="lower center", bbox_to_anchor=(0.54, -0.12), ncol=2,
            fontsize=8, frameon=False)
        fig.savefig(figure_path, format="pdf", bbox_inches="tight",
                    metadata={"Creator": "SciTrace deterministic result figure"})
        plt.close(fig)
    data_path = figure_dir / "budget_effect_data.json"
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "path": FIGURE_PATH,
        "data_path": DATA_PATH,
        "figure_sha256": _sha256(figure_path),
        "data_sha256": _sha256(data_path),
        "source_sha256": data["source_sha256"],
        "interval_display": data["interval_display"],
    }
    (figure_dir / "budget_effect_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"figure": figure_path, "data": data_path, "sources": data["source_sha256"], "manifest": manifest}

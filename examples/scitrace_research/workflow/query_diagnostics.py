"""Paired query risk diagnostics from preserved Route B2 results.

These descriptive statistics decompose a measured mean into gains and losses;
like the paired recall intervals, they condition on the fitted policy and the
observed batch. They are not run-to-run LLM reliability or causal effects. No
retrieval, training, or API call is performed by this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .robustness_analysis import DATASET_KEYS, load_dataset

FIGURE_PATH = "figures/query_risk.pdf"
DATA_PATH = "figures/query_diagnostics.json"
MANIFEST_PATH = "figures/query_diagnostics_manifest.json"
ARTIFACT_PATH = "experiments/route_b2_artifacts"
DATASET_LABELS = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa_lexical": "FiQA (lexical)"}
TOLERANCE = 1e-12


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def decompose_query_outcomes(candidate, baseline, allocation, budget: int) -> dict:
    """Keep benefit, harm, capacity movement, and no-evidence events distinct."""
    candidate, baseline = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    allocation = np.asarray(allocation, dtype=float)
    if candidate.ndim != 1 or not len(candidate) or candidate.shape != baseline.shape or candidate.shape != allocation.shape:
        raise ValueError("query diagnostics require paired nonempty one-dimensional arrays")
    if not np.isfinite(np.concatenate((candidate, baseline, allocation))).all():
        raise ValueError("query diagnostics require finite values")
    if np.any(candidate < 0) or np.any(candidate > 1) or np.any(baseline < 0) or np.any(baseline > 1):
        raise ValueError("recall must be in [0, 1]")
    if not np.isfinite(budget) or budget < 1 or budget != int(budget) or np.any(allocation < 1) or np.any(allocation != np.floor(allocation)):
        raise ValueError("allocated depths and fixed budget must be positive integers")
    if abs(float(allocation.sum()) - budget * len(candidate)) > TOLERANCE:
        raise ValueError("candidate and baseline must use equal total document capacity")
    delta = candidate - baseline
    if np.any((allocation - budget) * delta < -TOLERANCE) or np.any((allocation == budget) & (np.abs(delta) > TOLERANCE)):
        raise ValueError("recall change must agree with depth change on the same ranking")
    wins, losses = delta > TOLERANCE, delta < -TOLERANCE
    ties = ~(wins | losses)
    # The denominator is all extra document slots moved above the fixed depth,
    # not total capacity; zero movement has no defined unused-share estimate.
    extra_slots = np.maximum(allocation - budget, 0)
    moved = int(extra_slots.sum())
    unchanged_extra = int(extra_slots[ties].sum())
    no_evidence_base, no_evidence_cand = baseline <= TOLERANCE, candidate <= TOLERANCE
    n = len(candidate)
    return {
        "queries": n, "budget": int(budget),
        "candidate_recall": float(candidate.mean()), "baseline_recall": float(baseline.mean()),
        "mean_difference": float(delta.mean()),
        "positive_contribution": float(np.maximum(delta, 0).mean()),
        "negative_contribution": float(np.minimum(delta, 0).mean()),
        "wins": int(wins.sum()), "ties": int(ties.sum()), "losses": int(losses.sum()),
        "win_fraction": float(wins.mean()), "tie_fraction": float(ties.mean()), "loss_fraction": float(losses.mean()),
        "conditional_win_fraction": float(wins.sum() / (wins.sum() + losses.sum())) if (wins | losses).any() else None,
        "baseline_no_evidence": int(no_evidence_base.sum()), "candidate_no_evidence": int(no_evidence_cand.sum()),
        "rescued_from_no_evidence": int((no_evidence_base & ~no_evidence_cand).sum()),
        "newly_no_evidence": int((~no_evidence_base & no_evidence_cand).sum()),
        "moved_document_slots": moved,
        "extra_slots_on_unchanged_queries": unchanged_extra,
        "extra_slots_on_unchanged_fraction": unchanged_extra / moved if moved else None,
        "total_document_capacity": int(allocation.sum()),
        "conservation_residual": float(allocation.sum() - n * budget),
    }


def budget_transition_diagnostics(data: dict) -> list[dict]:
    """Reveal query regressions that an increasing mean budget could conceal."""
    rows = []
    for left, right in zip(range(len(data["budgets"]) - 1), range(1, len(data["budgets"]))):
        candidate_delta = data["candidate"][:, right] - data["candidate"][:, left]
        fixed_delta = data["baseline"][:, right] - data["baseline"][:, left]
        depths = data["allocations"][:, right] - data["allocations"][:, left]
        rows.append({
            "from_budget": int(data["budgets"][left]), "to_budget": int(data["budgets"][right]),
            "queries": len(candidate_delta),
            "candidate_improved": int((candidate_delta > TOLERANCE).sum()),
            "candidate_unchanged": int((np.abs(candidate_delta) <= TOLERANCE).sum()),
            "candidate_regressed": int((candidate_delta < -TOLERANCE).sum()),
            "fixed_regressed": int((fixed_delta < -TOLERANCE).sum()),
            "allocation_decreased": int((depths < 0).sum()),
            "candidate_mean_change": float(candidate_delta.mean()),
            "fixed_mean_change": float(fixed_delta.mean()),
            "regression_query_ids": [q for q, change in zip(data["query_ids"], candidate_delta) if change < -TOLERANCE],
        })
    return rows


def _cases(data: dict, column: int) -> list[dict]:
    """Deterministic extreme examples; no unavailable query text is invented."""
    candidate, baseline = data["candidate"][:, column], data["baseline"][:, column]
    differences = candidate - baseline
    cases = []
    for outcome, sign in (("loss", 1), ("gain", -1)):
        indices = [i for i, value in enumerate(differences) if sign * value < -TOLERANCE]
        indices.sort(key=lambda i: (sign * differences[i], data["query_ids"][i]))
        for i in indices[:2]:
            cases.append({"query_id": data["query_ids"][i], "selection": f"largest_{outcome}",
                          "budget": int(data["budgets"][column]),
                          "allocated_depth": int(data["allocations"][i, column]),
                          "candidate_recall": float(candidate[i]), "baseline_recall": float(baseline[i]),
                          "difference": float(differences[i])})
    return cases


def build_query_diagnostics(artifact_dir: Path) -> dict:
    """Validate preserved query outcomes before deriving descriptive diagnostics."""
    artifact_dir = Path(artifact_dir)
    summaries, transitions, cases = [], [], []
    primary_budget = None
    budgets = None
    sources = {}
    for dataset in DATASET_KEYS:
        data = load_dataset(artifact_dir / dataset)
        if primary_budget is None:
            primary_budget = int(data["metadata"]["primary_budget"])
            budgets = data["budgets"]
        if primary_budget != int(data["metadata"]["primary_budget"]) or budgets != data["budgets"]:
            raise ValueError("datasets must share the primary budget and sweep grid")
        if primary_budget not in budgets:
            raise ValueError("primary budget must be present in the sweep")
        for column, budget in enumerate(budgets):
            summaries.append({"dataset": dataset, **decompose_query_outcomes(
                data["candidate"][:, column], data["baseline"][:, column],
                data["allocations"][:, column], budget)})
        transitions.extend({"dataset": dataset, **row} for row in budget_transition_diagnostics(data))
        cases.extend({"dataset": dataset, **row} for row in _cases(data, budgets.index(primary_budget)))
        for filename in ("results.json", "per_query_results.csv"):
            relative = f"{dataset}/{filename}"
            sources[relative] = _sha(artifact_dir / relative)
    return {
        "schema_version": 1,
        "analysis_kind": "descriptive_paired_query_risk_on_frozen_allocation",
        "primary_budget": primary_budget, "budgets": list(budgets),
        "validation": {"status": "passed", "source_means_recomputed": True,
                       "equal_total_document_capacity": True, "paired_query_ids": True},
        "summaries": summaries, "budget_transitions": transitions, "selected_examples": cases,
        "source_sha256": sources,
        "definitions": {
            "contribution_identity": "mean(candidate - fixed) = mean(max(delta, 0)) + mean(min(delta, 0))",
            "no_evidence": "query recall is zero under the saved relevance labels",
            "extra_slots_on_unchanged_fraction": "extra slots above fixed depth on queries with unchanged recall / all extra slots above fixed depth",
            "case_selection": "Up to two largest losses and gains per dataset at the recorded primary budget; query-id tie break; not representative sampling.",
            "budget_transition": "Same test query IDs, compared between adjacent evaluated mean budgets; allocation policies were already fitted and recorded.",
        },
        "limitations": [
            "Saved outcomes are reanalyzed; no new retrieval, model fitting, or API generation is performed.",
            "These are descriptive query-level outcomes, not repeated-run reliability estimates, causal mechanisms, or statistical significance tests.",
            "Unchanged recall on extra slots is an observed relevance-label outcome, not proof that documents are useless for downstream answers.",
            "All results condition on the fitted policy and observed jointly allocated batch; new-batch behavior is not evaluated.",
            "FiQA is fitted separately with lexical retrieval; it is neither zero-shot transfer nor directly comparable in retrieval setup.",
            "Document slots do not measure tokens, GPU time, monetary cost, latency, or answer correctness.",
        ],
    }


def render_query_diagnostics(artifact_dir: Path, figure_dir: Path,
                             formats: tuple[str, ...] = ("pdf", "png")) -> dict:
    """Produce one paper figure, exact numbers, and a source-bound manifest."""
    if not formats or set(formats) - {"pdf", "png", "svg"}:
        raise ValueError("diagnostic figure formats must be pdf, png, or svg")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    data = build_query_diagnostics(Path(artifact_dir))
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_path = figure_dir / Path(DATA_PATH).name
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    rows = [next(r for r in data["summaries"] if r["dataset"] == dataset and r["budget"] == data["primary_budget"])
            for dataset in DATASET_KEYS]
    blue, red, grey, ink = "#266C96", "#C26049", "#D8DFE5", "#27333D"
    outputs = {}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.75), gridspec_kw={"width_ratios": [1.25, 1.35, 1.0]})
        fig.subplots_adjust(left=.12, right=.985, top=.80, bottom=.29, wspace=.39)
        y = np.arange(len(rows))
        labels = [DATASET_LABELS[r["dataset"]] for r in rows]
        ax = axes[0]
        gains = np.array([100 * r["positive_contribution"] for r in rows])
        losses = np.array([100 * r["negative_contribution"] for r in rows])
        net = gains + losses
        ax.axvline(0, color="#7E8992", lw=.8, zorder=0)
        ax.barh(y, losses, color=red, height=.48)
        ax.barh(y, gains, color=blue, height=.48)
        ax.scatter(net, y, marker="o", c=ink, s=34, edgecolor="white", linewidth=.8, zorder=4)
        for i, value in enumerate(net):
            ax.annotate(f"{value:+.2f}", (value, i), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7)
        ax.set_yticks(y, labels); ax.set_xlabel("Mean recall change (pp)", labelpad=6)
        ax.set_title("(a) Gain and loss contributions", fontsize=8, pad=12)
        ax.grid(axis="x", alpha=.14)
        ax = axes[1]
        left = np.zeros(len(rows))
        for key, color in (("loss_fraction", red), ("tie_fraction", grey), ("win_fraction", blue)):
            values = np.array([100 * r[key] for r in rows])
            ax.barh(y, values, left=left, color=color, height=.48)
            for i, (value, start) in enumerate(zip(values, left)):
                if value >= 8:
                    ax.text(start + value / 2, i, f"{value:.1f}", va="center", ha="center", fontsize=7,
                            color=ink if key == "tie_fraction" else "white")
            left += values
        ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.xaxis.set_major_formatter(PercentFormatter())
        ax.set_yticks(y, [""] * len(rows)); ax.set_xlabel("Queries: loss / tie / gain", labelpad=6)
        ax.set_title("(b) Paired query outcomes", fontsize=8, pad=12)
        ax = axes[2]
        for i, r in enumerate(rows):
            ax.plot([r["newly_no_evidence"], r["rescued_from_no_evidence"]], [i, i], color=grey, lw=2, zorder=0)
            ax.scatter(r["newly_no_evidence"], i, color=red, marker="o", s=34, edgecolor="white", linewidth=.6, zorder=3)
            ax.scatter(r["rescued_from_no_evidence"], i, color=blue, marker="o", s=34, edgecolor="white", linewidth=.6, zorder=4)
            ax.annotate(f"{r['newly_no_evidence']} / {r['rescued_from_no_evidence']}",
                        ((r["newly_no_evidence"] + r["rescued_from_no_evidence"]) / 2, i),
                        xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7)
        ax.set_yticks(y, [""] * len(rows)); ax.set_xlabel("Queries (new miss / rescue)", labelpad=6, fontsize=7.5)
        ax.set_title("(c) No-evidence transitions", fontsize=8, pad=12)
        ax.set_xlim(-1, max(1, max(r["rescued_from_no_evidence"] for r in rows), max(r["newly_no_evidence"] for r in rows)) + 3)
        ax.grid(axis="x", alpha=.14)
        for ax in axes:
            ax.set_ylim(2.6, -.5)
            ax.tick_params(axis="y", length=0)
        fig.legend(handles=[Patch(facecolor=blue, label="Gain / rescue"), Patch(facecolor=red, label="Loss / new miss"),
                            Patch(facecolor=grey, label="Tie"), Line2D([], [], marker="o", color=ink, linestyle="none", markersize=5, label="Net mean")],
                   loc="lower center", bbox_to_anchor=(.55, .025), ncol=4, fontsize=7, frameon=False)
        fig.suptitle(f"Equal mean depth {data['primary_budget']}: average gains can coexist with query-level losses", y=.995, fontsize=9)
        for suffix in formats:
            path = figure_dir / f"query_risk.{suffix}"
            metadata = {"Creator": "SciTrace verified query diagnostics", "CreationDate": None} if suffix == "pdf" else None
            fig.savefig(path, dpi=220, bbox_inches="tight", metadata=metadata)
            outputs[path.name] = _sha(path)
        plt.close(fig)
    manifest = {"schema_version": 1, "path": FIGURE_PATH, "data_path": DATA_PATH,
                "data_sha256": _sha(data_path), "figure_sha256": outputs.get("query_risk.pdf"),
                "files_sha256": outputs, "source_sha256": data["source_sha256"],
                "source_root": ARTIFACT_PATH, "renderer_sha256": _sha(Path(__file__)),
                "caption_hint": "At the recorded primary mean depth, paired recall gains and losses sum to the net mean change (round dot). Query-level outcomes and zero-recall transitions expose heterogeneity hidden by the mean. Counts refer to different-size test sets. All panels are descriptive, conditional on saved allocations; no new significance tests are shown."}
    (figure_dir / Path(MANIFEST_PATH).name).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"figure": figure_dir / "query_risk.pdf", "data": data_path,
            "sources": data["source_sha256"], "manifest": manifest}


def verify_query_diagnostics(run_dir: Path) -> dict:
    """Fail closed if a figure, manifest, data, or saved source was substituted."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / MANIFEST_PATH).read_text(encoding="utf-8"))
    if manifest.get("path") != FIGURE_PATH or manifest.get("data_path") != DATA_PATH or manifest.get("source_root") != ARTIFACT_PATH:
        raise ValueError("query diagnostics manifest has unexpected artifact paths")
    data_path, figure_path = run_dir / DATA_PATH, run_dir / FIGURE_PATH
    if manifest.get("data_sha256") != _sha(data_path) or manifest.get("figure_sha256") != _sha(figure_path):
        raise ValueError("query diagnostics artifact hash mismatch")
    # Restrict file names before reading anything from the manifest.
    files = manifest.get("files_sha256") or {}
    if "query_risk.pdf" not in files or set(files) - {"query_risk.pdf", "query_risk.png", "query_risk.svg"}:
        raise ValueError("query diagnostics manifest has unexpected figure files")
    for name, expected in files.items():
        if _sha(run_dir / "figures" / name) != expected:
            raise ValueError("query diagnostics figure hash mismatch")
    if manifest.get("renderer_sha256") != _sha(Path(__file__)):
        raise ValueError("query diagnostics renderer hash mismatch; regenerate figures")
    recomputed = build_query_diagnostics(run_dir / ARTIFACT_PATH)
    stored = json.loads(data_path.read_text(encoding="utf-8"))
    if manifest.get("source_sha256") != recomputed["source_sha256"] or _canonical(stored) != _canonical(recomputed):
        raise ValueError("query diagnostics data differ from validated source outcomes")
    return {"status": "passed", "figure_sha256": manifest["figure_sha256"],
            "data_sha256": manifest["data_sha256"], "sources_verified": len(recomputed["source_sha256"])}


def write_diagnostic_report(data: dict, path: Path) -> None:
    rows = [r for r in data["summaries"] if r["budget"] == data["primary_budget"]]
    lines = ["# Paired query diagnostics", "", "Saved outcomes only; no model/API/retrieval rerun.", "",
             "| Dataset | Queries | Net change (pp) | Gain / loss contributions (pp) | Gain / tie / loss queries | Rescued / newly zero recall | Extra slots on unchanged queries |",
             "|---|---:|---:|---|---|---|---|"]
    for r in rows:
        fraction = r["extra_slots_on_unchanged_fraction"]
        unchanged = "undefined (no moved capacity)" if fraction is None else f"{r['extra_slots_on_unchanged_queries']}/{r['moved_document_slots']} ({100*fraction:.1f}%)"
        lines.append(f"| {DATASET_LABELS[r['dataset']]} | {r['queries']} | {100*r['mean_difference']:+.3f} | "
                     f"{100*r['positive_contribution']:+.3f} / {100*r['negative_contribution']:+.3f} | "
                     f"{r['wins']} / {r['ties']} / {r['losses']} | {r['rescued_from_no_evidence']} / {r['newly_no_evidence']} | {unchanged} |")
    lines += ["", "## Scope", ""] + [f"- {value}" for value in data["limitations"]]
    lines += ["", "## Budget-transition audit", "", "These counts compare the same queries at adjacent saved mean budgets.", "",
              "| Dataset | Budgets | Candidate regressions | Fixed-depth regressions | Decreased allocations |", "|---|---|---:|---:|---:|"]
    for row in data["budget_transitions"]:
        lines.append(f"| {DATASET_LABELS[row['dataset']]} | {row['from_budget']} → {row['to_budget']} | {row['candidate_regressed']} | {row['fixed_regressed']} | {row['allocation_decreased']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    artifact_dir, output = args.artifact_dir.resolve(), args.output_dir.resolve()
    if output == artifact_dir or artifact_dir in output.parents:
        parser.error("output must be outside preserved source artifacts")
    rendered = render_query_diagnostics(artifact_dir, output)
    data = json.loads(rendered["data"].read_text(encoding="utf-8"))
    write_diagnostic_report(data, output / "query_diagnostics_report.md")
    for key in ("summaries", "budget_transitions", "selected_examples"):
        with (output / f"query_{key}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[key][0]))
            writer.writeheader(); writer.writerows(data[key])
    print(json.dumps({"validation": data["validation"], "files": list(rendered["manifest"]["files_sha256"])}, indent=2))


if __name__ == "__main__":
    main()

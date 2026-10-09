"""Original method schematic, bound to the checked-in implementation contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .method_contract import ROUTE_B2_METHOD_CONTRACT

FIGURE_PATH = "figures/method_overview.pdf"
MANIFEST_PATH = "figures/method_overview_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_hash() -> str:
    return hashlib.sha256(json.dumps(ROUTE_B2_METHOD_CONTRACT, sort_keys=True).encode()).hexdigest()


def render_method_overview(run_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    target = run_dir / FIGURE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(7.1, 2.85))
        ax.set(xlim=(0, 10), ylim=(0, 4.0)); ax.axis("off")
        def box(x, y, width, text, color="#eef2f6", edge="#7c8995"):
            ax.add_patch(FancyBboxPatch((x, y), width, .74, boxstyle="round,pad=0.06", fc=color, ec=edge, lw=.8))
            ax.text(x + width / 2, y + .37, text, ha="center", va="center", linespacing=1.4)
        def arrow(x1, y1, x2, y2, color="#53616e"):
            ax.annotate("", (x2, y2), (x1, y1), arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0})
        ax.text(.05, 3.85, "A  TRAINING ONLY", weight="bold", color="#326d70")
        box(.1, 2.75, 2.05, "Training queries\n+ relevance labels", "#e8f3ef")
        box(3.1, 2.75, 3.0, "Tune fusion; fit feature scaler\nStore neighbor recall curves", "#e8f3ef")
        box(7.15, 2.75, 2.65, "Freeze fitted configuration\nNo test-label tuning", "#e8f3ef")
        arrow(2.2, 3.12, 3.03, 3.12); arrow(6.15, 3.12, 7.08, 3.12)
        ax.axhline(2.47, color="#b7c5ca", lw=.7, ls="--")
        ax.text(.05, 2.2, "B  KNOWN TEST BATCH", weight="bold", color="#23689b")
        box(.1, 1.15, 1.8, "Retrieve candidates\nFuse rankings")
        box(2.55, 1.15, 1.9, "Query features\nPredict recall curves")
        box(5.1, 1.15, 2.25, "Joint integer allocation\nExact batch budget", "#e4eefb", "#23689b")
        box(8.0, 1.15, 1.8, "Admit prefixes\nEvaluate recall")
        arrow(1.96, 1.52, 2.48, 1.52); arrow(4.51, 1.52, 5.03, 1.52); arrow(7.41, 1.52, 7.93, 1.52)
        ax.plot([8.46, 8.46, 3.5], [2.69, 2.08, 2.08], color="#326d70", lw=1.0)
        arrow(3.5, 2.08, 3.5, 1.94, color="#326d70")
        ax.text(5.0, 2.16, "frozen parameters for prediction", fontsize=7, color="#326d70")
        ax.text(5.98, .72, "Changed component: depth per query", ha="center", color="#23689b", weight="bold")
        ax.text(5.0, .17, "Matched comparison: same queries + fused rankings + total document count; fixed depth vs allocated depth",
                ha="center", fontsize=7.2)
        fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.98)
        fig.savefig(target, bbox_inches="tight", metadata={"CreationDate": None})
        fig.savefig(target.with_suffix(".png"), bbox_inches="tight", dpi=220)
        plt.close(fig)
    record = {"path": FIGURE_PATH, "figure_sha256": sha(target), "contract_sha256": contract_hash(),
              "renderer_sha256": sha(Path(__file__)), "kind": "implementation schematic; no performance estimates"}
    (run_dir / MANIFEST_PATH).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def verify_method_overview(run_dir: Path) -> dict:
    record = json.loads((run_dir / MANIFEST_PATH).read_text(encoding="utf-8"))
    if record.get("path") != FIGURE_PATH or record.get("figure_sha256") != sha(run_dir / FIGURE_PATH):
        raise ValueError("Method overview figure differs from its manifest")
    if record.get("contract_sha256") != contract_hash() or record.get("renderer_sha256") != sha(Path(__file__)):
        raise ValueError("Method overview has a stale implementation contract or renderer")
    return record

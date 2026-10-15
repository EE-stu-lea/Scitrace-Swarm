from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[5]
CHECKLIST_SCRIPT = (
    REPO_ROOT / "examples" / "scitrace_research" / "skills"
    / "pre-submit-checklist" / "scripts" / "checklist.py"
)

_FILLER_SENTENCE = (
    "The allocator assigns integer document depths after rankings are fused "
    "and every comparison conditions on the measured artifact."
)


def _filler(repetitions: int) -> str:
    return " ".join([_FILLER_SENTENCE] * repetitions)


def _run(draft_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKLIST_SCRIPT), "--draft", str(draft_path)],
        capture_output=True, text=True, encoding="utf-8",
    )


def _passing_draft() -> str:
    return "\n".join([
        "# Adaptive Document Depth Allocation for Evidence Retrieval",
        "",
        "## Abstract",
        "",
        "We study allocation of document depth across a known query batch and report measured evidence recall.",
        "",
        "## Introduction",
        "",
        "Adaptive budget allocation matters for retrieval pipelines `evidence_01`. "
        "Prior work studies calibrated retrieval budget allocation `evidence_02`. "
        "Our comparison conditions on the realized batch allocation `evidence_03`.",
        "",
        "## Related Work",
        "",
        "Retrieval benchmarks evaluate heterogeneous domains `evidence_04`, and "
        "adaptive retrieval depth methods allocate budget per query `evidence_05`. "
        "We contrast joint batch allocation with these per-query policies `evidence_06`.",
        "",
        "## Method",
        "",
        "![Allocator overview showing training-only fitting, frozen test inference and the changed integer allocation component](figures/method_overview.pdf)",
        "",
        "Figure 1: The allocator fits recall calibration on training queries only and applies frozen inference to test queries.",
        "",
        _filler(17),
        "",
        "## Experimental Setup",
        "",
        _filler(10),
        "",
        "## Results",
        "",
        "### RQ1: Matched-budget effectiveness",
        "",
        "Table 1: Primary paired comparison of candidate and fixed-baseline evidence recall with pointwise paired-bootstrap 95% intervals at budget five.",
        "",
        "| Dataset | Candidate | Fixed baseline | Difference | 95% interval |",
        "| --- | --- | --- | --- | --- |",
        "| SciFact | 0.7 | 0.6 | 0.1 | [0.05, 0.15] |",
        "| NFCorpus | 0.5 | 0.45 | 0.05 | [-0.02, 0.12] |",
        "| Macro | 0.6 | 0.525 | 0.075 | [0.02, 0.13] |",
        "| FiQA (lexical) | 0.3 | 0.28 | 0.02 | [-0.05, 0.09] |",
        "",
        "Table 2: Descriptive comparison of adaptive allocation baselines at the primary budget without paired intervals.",
        "",
        "| Dataset | Raw Jaccard | Binned calibration | kNN allocation |",
        "| --- | --- | --- | --- |",
        "| SciFact | 0.61 | 0.64 | 0.70 |",
        "| NFCorpus | 0.44 | 0.46 | 0.50 |",
        "",
        _filler(10),
        "",
        "### RQ2: Redistribution and query risk",
        "",
        "Table 3: Verified per-query outcome counts for the fixed-allocation comparison at budget five.",
        "",
        "| Dataset | Wins | Ties | Losses | Rescued | Newly missed |",
        "| --- | --- | --- | --- | --- | --- |",
        "| SciFact | 9 | 4 | 2 | 3 | 1 |",
        "| NFCorpus | 6 | 5 | 3 | 2 | 2 |",
        "",
        "![Per-query contributions to mean recall](figures/query_risk.pdf)",
        "",
        "Figure 2: Rescued and newly missed zero-recall events per dataset, descriptive of the realized fixed allocation.",
        "",
        _filler(10),
        "",
        "### RQ3: Sensitivity and boundary conditions",
        "",
        "Table 4: Measured feature ablation recalls showing the full feature set is not uniformly best.",
        "",
        "| Dataset | All features | Rank overlap | Jaccard only | RBO only |",
        "| --- | --- | --- | --- | --- |",
        "| SciFact | 0.70 | 0.71 | 0.66 | 0.65 |",
        "| NFCorpus | 0.50 | 0.49 | 0.47 | 0.46 |",
        "",
        "![Evidence recall by budget for the candidate allocator and the fixed baseline](figures/recall_by_budget.pdf)",
        "",
        "![Candidate minus fixed-baseline evidence recall for SciFact, NFCorpus and the separately trained lexical FiQA evaluation, with pointwise paired-bootstrap intervals at every swept budget](figures/budget_effect.pdf)",
        "",
        _filler(31),
        "",
        "## Discussion",
        "",
        _filler(10),
        "",
        "## Limitations",
        "",
        "The primary budget was selected without a timestamped protocol, and all intervals condition on the realized batch allocation.",
        "",
        "## Conclusion",
        "",
        "We answered RQ1, RQ2 and RQ3 with measured comparisons.",
        "",
        "## References",
        "",
        "- `evidence_01` Alice. Adaptive retrieval budget allocation. 2025.",
        "- `evidence_02` Bob. Calibrated retrieval-budget allocation for generation. 2024.",
        "- `evidence_03` Cara. Query difficulty prediction for retrieval. 2024.",
        "- `evidence_04` BEIR Group. A heterogeneous benchmark for retrieval models. 2021.",
        "- `evidence_05` Dan. Adaptive retrieval depth learning. 2024.",
        "- `evidence_06` Erin. Know before you fetch calibrated allocation. 2024.",
        "",
    ])


def test_checklist_passes_complete_draft(tmp_path: Path) -> None:
    draft = tmp_path / "draft.md"
    draft.write_text(_passing_draft(), encoding="utf-8")

    process = _run(draft)

    report = json.loads(process.stdout)
    assert all(set(entry) == {"item", "status", "detail"} for entry in report)
    failures = [entry for entry in report if entry["status"] != "pass"]
    assert process.returncode == 0, json.dumps(failures, ensure_ascii=False, indent=2)


def test_checklist_fails_minimal_draft(tmp_path: Path) -> None:
    draft = tmp_path / "draft.md"
    draft.write_text(
        "# Broken Draft\n\n## Abstract\n\nThis draft is incomplete; it is not a zero-shot transfer study.\n",
        encoding="utf-8",
    )

    process = _run(draft)

    assert process.returncode == 1
    report = json.loads(process.stdout)
    assert isinstance(report, list) and report
    assert all(set(entry) == {"item", "status", "detail"} for entry in report)
    items = {entry["item"]: entry for entry in report}
    assert items["manuscript_title"]["status"] == "pass"
    assert items["required_sections"]["status"] == "fail"
    assert "Introduction" in items["required_sections"]["detail"]
    # Forbidden terms are flagged even inside a negated sentence.
    assert items["forbidden_terms_absent"]["status"] == "fail"
    assert "zero-shot" in items["forbidden_terms_absent"]["detail"]
    assert items["section_word_minimums"]["status"] == "fail"
    assert items["body_citations_covered"]["status"] == "fail"


def test_checklist_reports_missing_draft_file(tmp_path: Path) -> None:
    process = _run(tmp_path / "absent.md")
    assert process.returncode == 2
    assert "not found" in process.stderr

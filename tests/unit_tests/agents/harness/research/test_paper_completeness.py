"""Evidence-driven checks for underdeveloped or incorrectly copied manuscripts."""

from examples.scitrace_research.workflow.paper_quality import audit_draft, feature_ablation_rows


def test_typesetting_gate_catches_unsupported_math_before_paid_draft_is_accepted():
    config = {"enabled": True, "validate_typesetting": True}
    bad = "# Title\n\n## Method\n" + r"\[\begin{aligned}x=1\end{aligned}\]"
    report = audit_draft(bad, evidence=[], config=config)
    assert [error["code"] for error in report["errors"]] == ["paper_typesetting_invalid"]
    good = "# Title\n\n## Method\n" + r"\[\sum_{i=1}^{n} k_i = B\]"
    assert audit_draft(good, evidence=[], config=config)["ok"]


def test_section_minimum_counts_nested_prose_but_not_tables_or_audit_padding():
    draft = (
        "# Paper\n## Method\nTraining fits the calibrator.\n"
        "### Allocation\nInteger depths maximize predicted recall within the batch budget.\n"
        "## Results\nA measured comparison follows.\n"
        "| Dataset | All features | Rank overlap | Jaccard only | RBO only |\n"
        "|---|---:|---:|---:|---:|\n| SciFact | 0.8000 | 0.8100 | 0.7000 | 0.6000 |\n"
        "![" + "caption " * 40 + "](figures/result.pdf)\n"
        "\n**Figure 1.** " + "caption " * 40 + "\n\n"
        "```json\n" + '"metadata": "' + "metadata " * 40 + '"\n```\n'
        "## References\n" + "reference " * 40 + "\n"
        "### Structured References\n" + "audit " * 40 + "\n"
        "## Results\n" + "spoofed " * 40
    )
    report = audit_draft(draft, evidence=[], config={"enabled": True,
        "minimum_section_words": {"Method": 10, "Results": 20}})
    assert report["section_word_counts"]["method"] == 13
    assert report["section_word_counts"]["results"] == 4
    assert [error["code"] for error in report["errors"]] == ["paper_section_underdeveloped"]
    assert "Results has 4 prose words" in report["errors"][0]["message"]


def _results():
    def dataset(full, overlap, jaccard, rbo):
        return {"feature_ablation": {key: {"recall": value} for key, value in zip(
            ("all_features", "rank_overlap", "jaccard_only", "rbo_only"),
            (full, overlap, jaccard, rbo),
        )}}
    return {"datasets": {"scifact": dataset(.8, .81, .7, .6)},
            "boundary_results": {"fiqa_lexical": dataset(.2, .21, .1, .09)}}


def _ablation_draft():
    # The accepted columns may be reordered. Values are checked by header,
    # preventing an accidental swap of two features from passing review.
    return (
        "## Results\n\nTable 1. Descriptive feature-ablation evidence at the primary budget.\n\n"
        "| Dataset | Rank overlap | All features | RBO only | Jaccard only |\n"
        "|---|---:|---:|---:|---:|\n"
        "| SciFact | 0.8100 | 0.8000 | 0.6000 | 0.7000 |\n"
        "| FiQA (lexical) | 0.2100 | 0.2000 | 0.0900 | 0.1000 |\n"
    )


def test_ablation_table_is_checked_against_current_artifact_including_boundary_dataset():
    results = _results()
    assert feature_ablation_rows(results)[1]["dataset"] == "FiQA (lexical)"
    config = {"enabled": True, "require_feature_ablation_table": True, "require_table_captions": True}
    assert audit_draft(_ablation_draft(), evidence=[], config=config, results=results)["ok"]
    # A fresh experiment changes only one stored result. The old prose must
    # immediately fail without updating a hardcoded list of accepted values.
    results["boundary_results"]["fiqa_lexical"]["feature_ablation"]["rbo_only"]["recall"] = .08
    errors = audit_draft(_ablation_draft(), evidence=[], config=config, results=results)["errors"]
    assert len(errors) == 1
    assert errors[0]["code"] == "paper_feature_ablation_value_mismatch"
    assert "FiQA" in errors[0]["message"] and "rbo_only" in errors[0]["message"]


def test_qualified_lexical_labels_are_accepted_in_every_table():
    draft = _ablation_draft().replace("FiQA (lexical)", "FiQA (lexical finance-domain evaluation)")
    report = audit_draft(draft, evidence=[], config={"enabled": True,
        "require_feature_ablation_table": True}, results=_results())
    assert report["ok"], report["errors"]
    draft = (
        "## Results\n"
        "| Dataset | Raw Jaccard | Binned calibration | kNN allocation |\n"
        "|---|---:|---:|---:|\n"
        "| FiQA (lexical finance-domain evaluation) | 0.2000 | 0.2100 | 0.2200 |\n"
    )
    report = audit_draft(draft, evidence=[], config={"enabled": True,
        "require_adaptive_baseline_table": True}, adaptive_rows=[{
        "dataset": "FiQA (lexical)", "raw_jaccard": .2, "binned_calibration": .21, "knn_calibration": .22}])
    assert report["ok"], report["errors"]


def test_ablation_missing_evidence_or_rows_cannot_be_filled_by_a_fabricated_table():
    config = {"enabled": True, "require_feature_ablation_table": True}
    report = audit_draft(_ablation_draft(), evidence=[], config=config, results={})
    assert report["errors"][0]["code"] == "paper_feature_ablation_evidence_missing"
    draft = _ablation_draft().replace("| FiQA (lexical) | 0.2100 | 0.2000 | 0.0900 | 0.1000 |\n", "")
    report = audit_draft(draft, evidence=[], config=config, results=_results())
    assert report["errors"][0]["code"] == "paper_feature_ablation_row_missing"


def test_all_results_tables_require_captions_including_feature_ablation():
    table = "| Name | Value |\n|---|---:|\n| measured | 0.5 |\n\n"
    draft = "## Results\n\n"
    draft += "Table 1. Measured primary comparison at the configured budget.\n\n" + table
    draft += "Table 2. Descriptive adaptive baseline comparisons at equal budget.\n\n" + table
    draft += table
    report = audit_draft(draft, evidence=[], config={"enabled": True, "require_table_captions": True})
    assert len(report["errors"]) == 1
    assert "Table 3" in report["errors"][0]["message"]

"""Regression tests for selecting evidence and accepting a generated paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from examples.scitrace_research.workflow.paper_quality import audit_draft, select_literature
from examples.scitrace_research.workflow.paper_narrative import audit_narrative
from examples.scitrace_research.workflow.pipeline import (
    _adaptive_baseline_rows, _durable_usage_summary, _paper_results_view,
    _restore_prior_tool_usage, _train_subsampling_rows,
)
from jiuwenswarm.agents.harness.research.reporting.resource_reporter import ResourceAuditSnapshot


def test_results_table_and_figure_gate_requires_verified_values() -> None:
    row = {"dataset": "SciFact", "candidate": 0.844444, "fixed_baseline": 0.803167,
           "difference": 0.041277, "ci95": [0.016944, 0.0675]}
    base = (
        "# Title\n\n## Results\n\n### Main comparison\n"
        "| Dataset | Candidate | Fixed baseline | Difference | 95% CI |\n"
        "|---|---:|---:|---:|---:|\n"
        "| SciFact | 0.8444 | 0.8032 | 0.0413 | [0.0169, 0.0675] |\n\n"
        "![Evidence recall by budget](figures/recall_by_budget.pdf)\n"
    )
    cfg = {"enabled": True, "require_results_table": True, "require_results_figure": True}
    accepted = audit_draft(base, evidence=[], config=cfg, table_rows=[row])
    assert accepted["ok"], accepted["errors"]
    missing = audit_draft(base.replace("0.8444", "0.9000").replace("![Evidence recall by budget](figures/recall_by_budget.pdf)", ""),
                          evidence=[], config=cfg, table_rows=[row])
    assert {error["code"] for error in missing["errors"]} == {
        "paper_results_table_value_mismatch", "paper_results_figure_missing"
    }


def test_budget_sweep_peak_claim_is_checked_against_measured_gaps() -> None:
    results = {"datasets": {"nfcorpus": {"methods_by_budget": {
        "2": {"knn_calibration": {"recall": 0.11}, "convex_fixed": {"recall": 0.10}},
        "5": {"knn_calibration": {"recall": 0.14}, "convex_fixed": {"recall": 0.135}},
    }}}}
    draft = "# Title\n\n## Results\nOn NFCorpus the largest gap at budget 5 is clear.\n"
    report = audit_draft(draft, evidence=[], config={"enabled": True}, results=results)
    assert [error["code"] for error in report["errors"]] == ["paper_budget_sweep_peak_mismatch"]


def test_adaptive_baseline_and_subsampling_gates_check_artifact_values() -> None:
    draft = (
        "# Title\n\n## Results\n"
        "| Dataset | Raw Jaccard | Binned calibration | kNN allocation |\n"
        "|---|---:|---:|---:|\n"
        "| SciFact | 0.796500 | 0.811944 | 0.844444 |\n"
        "| NFCorpus | 0.129268 | 0.139994 | 0.143550 |\n\n"
        "Five 80% training-query subset resamples are a limited sensitivity check.\n"
        "SciFact mean recall 0.841822, population SD 0.004121.\n"
        "NFCorpus mean recall 0.142211, population SD 0.001372.\n"
    )
    cfg = {"enabled": True, "require_adaptive_baseline_table": True,
           "require_train_subsampling_summary": True}
    adaptive = [
        {"dataset": "SciFact", "raw_jaccard": 0.7965, "binned_calibration": 0.811944, "knn_calibration": 0.844444},
        {"dataset": "NFCorpus", "raw_jaccard": 0.129268, "binned_calibration": 0.139994, "knn_calibration": 0.14355},
    ]
    stability = [
        {"dataset": "SciFact", "mean_recall": 0.841822222, "population_sd": 0.004121369},
        {"dataset": "NFCorpus", "mean_recall": 0.142210701, "population_sd": 0.001371826},
    ]
    assert audit_draft(draft, evidence=[], config=cfg, adaptive_rows=adaptive,
                       stability_rows=stability)["ok"]
    bad = draft.replace("0.139994", "0.149994").replace("0.004121", "0.014121")
    codes = {item["code"] for item in audit_draft(
        bad, evidence=[], config=cfg, adaptive_rows=adaptive, stability_rows=stability,
    )["errors"]}
    assert codes == {"paper_adaptive_baseline_value_mismatch", "paper_train_subsampling_value_missing"}


def test_subsampling_rerun_and_shared_peak_claims_are_rejected() -> None:
    results = {"datasets": {
        "scifact": {"methods_by_budget": {
            "2": {"knn_calibration": {"recall": 0.70}, "convex_fixed": {"recall": 0.69}},
            "5": {"knn_calibration": {"recall": 0.84}, "convex_fixed": {"recall": 0.80}},
            "8": {"knn_calibration": {"recall": 0.87}, "convex_fixed": {"recall": 0.86}},
        }},
        "nfcorpus": {"methods_by_budget": {
            "2": {"knn_calibration": {"recall": 0.11}, "convex_fixed": {"recall": 0.10}},
            "5": {"knn_calibration": {"recall": 0.14}, "convex_fixed": {"recall": 0.135}},
            "8": {"knn_calibration": {"recall": 0.16}, "convex_fixed": {"recall": 0.158}},
        }},
    }}
    draft = (
        "# Title\n\n## Experimental Setup\nAll runs use seed 42.\n"
        "## Results\nThe separation is largest at intermediate depths.\n"
        "We re-ran the pipeline on five 80% resamples of the available training queries.\n"
    )
    report = audit_draft(draft, evidence=[], config={"enabled": True,
        "require_train_subsampling_summary": True}, results=results,
        stability_rows=[{"dataset": "SciFact", "mean_recall": 0.84,
                         "population_sd": 0.01, "seeds": [42, 43, 44, 45, 46]}])
    codes = {item["code"] for item in report["errors"]}
    assert {"paper_train_subsampling_rerun_claim", "paper_seed_scope_ambiguity",
            "paper_budget_sweep_qualitative_mismatch"} <= codes


def test_results_tables_need_agent_written_captions() -> None:
    draft = (
        "# Title\n\n## Results\n"
        "| Dataset | Candidate | Fixed | Difference | 95% CI |\n"
        "|---|---:|---:|---:|---:|\n| SciFact | 0.84 | 0.80 | 0.04 | [0.01, 0.07] |\n\n"
        "| Dataset | Raw | Binned | kNN |\n"
        "|---|---:|---:|---:|\n| SciFact | 0.79 | 0.81 | 0.84 |\n"
    )
    cfg = {"enabled": True, "require_table_captions": True}
    report = audit_draft(draft, evidence=[], config=cfg)
    assert [e["code"] for e in report["errors"]] == ["paper_results_table_caption_missing"] * 2
    captioned = draft.replace("| Dataset | Candidate", "**Table 1: Primary paired comparison.**\n\n| Dataset | Candidate")
    captioned = captioned.replace("| Dataset | Raw", "**Table 2: Descriptive adaptive baselines.**\n\n| Dataset | Raw")
    assert audit_draft(captioned, evidence=[], config=cfg)["ok"]


def test_short_figure_alt_and_adjacent_scientific_caption_are_accepted() -> None:
    draft = (
        "# Title\n\n## Method\n\n"
        "![Method overview](figures/method_overview.pdf)\n\n"
        "*Figure 1: Training and test queries are separated; the frozen model changes only the allocation component.*\n\n"
        "## Results\n\n### RQ1: Effect\n\n### RQ2: Risk\n\n### RQ3: Boundary\n\n"
        "![Query risk](figures/query_risk.pdf)\n\n"
        "*Figure 2: Query gains and losses are descriptive for the realized fixed allocation, not new-batch reliability.*\n\n"
        "![Budget effect](figures/budget_effect.pdf)\n\n"
        "*Figure 3: SciFact, NFCorpus and FiQA effects have pointwise intervals at every budget, not simultaneous bands.*\n\n"
        "## Discussion\n\nThe negative result bounds deployment claims.\n"
    )
    assert audit_narrative(draft) == []
    assert audit_draft(draft, evidence=[], config={"enabled": True,
        "require_budget_effect_figure": True})["ok"]


def test_italic_table_caption_and_verified_fiqa_slug_row_are_accepted() -> None:
    draft = (
        "# Title\n\n## Results\n\n"
        "*Table 1: Verified per-query outcomes at the primary matched document budget.*\n\n"
        "| Dataset | Wins | Ties | Losses | Rescued | Newly missed |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        "| fiqa_lexical | 23 | 601 | 24 | 14 | 18 |\n"
    )
    diagnostics = {"primary_budget": 5, "summaries": [{
        "dataset": "fiqa_lexical", "budget": 5, "wins": 23, "ties": 601,
        "losses": 24, "rescued_from_no_evidence": 14, "newly_no_evidence": 18,
    }]}
    report = audit_draft(draft, evidence=[], config={"enabled": True,
        "require_table_captions": True, "require_query_outcome_table": True},
        query_diagnostics=diagnostics)
    assert report["ok"], report["errors"]


def test_period_captions_and_descriptive_fiqa_label_are_accepted() -> None:
    draft = (
        "## Results\n\n"
        "**Table 1.** Evidence recall under a paired comparison at the primary budget, "
        "with descriptive lexical stress-test evidence and intervals.\n\n"
        "| Dataset | Candidate | Fixed baseline | Difference | Paired 95% CI |\n"
        "|---|---:|---:|---:|---:|\n"
        "| FiQA (lexical stress test) | 0.238800 | 0.242454 | -0.003654 | [-0.015728, 0.007071] |\n\n"
        "**Table 2.** Descriptive adaptive baselines at the primary budget, "
        "with no additional uncertainty claim.\n\n"
        "| Dataset | Raw | Binned | kNN |\n"
        "|---|---:|---:|---:|\n"
        "| SciFact | 0.79 | 0.81 | 0.84 |\n"
    )
    report = audit_draft(
        draft,
        evidence=[],
        config={"enabled": True, "require_results_table": True, "require_table_captions": True},
        table_rows=[{
            "dataset": "FiQA (lexical)", "candidate": 0.238800,
            "fixed_baseline": 0.242454, "difference": -0.003654,
            "ci95": [-0.015728, 0.007071],
        }],
    )
    assert report["ok"], report["errors"]


def test_resumed_usage_includes_original_and_revision_calls(tmp_path: Path) -> None:
    import json

    path = tmp_path / "model_calls.jsonl"
    calls = [
        {"stage": "draft", "model_name": "model", "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        {"stage": "diagnostics", "model_name": "model"},
        {"stage": "revision", "model_name": "model", "prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
    ]
    path.write_text("\n".join(json.dumps(call) for call in calls) + "\n")

    class CurrentOnly:
        def usage_summary(self):
            return {"total": {"total_tokens": 40}}

    total = _durable_usage_summary(path, CurrentOnly())
    assert total["total"] == {
        "request_count": 2,
        "provider_calls": 0,
        "missing_usage_calls": 0,
        "prompt_tokens": 130,
        "completion_tokens": 30,
        "total_tokens": 160,
    }


def test_resumed_usage_retains_prior_tool_work() -> None:
    snapshot = ResourceAuditSnapshot(session_id="revision")
    _restore_prior_tool_usage(snapshot, {
        "started_at": "2026-09-23T00:00:00+00:00",
        "tool_calls": 9,
        "tool_failures": 1,
        "total_tool_seconds": 12.5,
        "total_experiment_seconds": 0.5,
        "tools_by_name": {"literature_search": 8, "experiment_execute": 1},
        "tool_seconds_by_name": {"literature_search": 12, "experiment_execute": 0.5},
    })
    assert snapshot.started_at == "2026-09-23T00:00:00+00:00"
    assert snapshot.tool_calls == 9
    assert snapshot.tool_failures == 1
    assert snapshot.total_tool_seconds == 12.5
    assert snapshot.total_experiment_seconds == 0.5
    assert snapshot.tools_by_name == {"literature_search": 8, "experiment_execute": 1}


@dataclass
class Candidate:
    title: str
    summary: str
    doi: str
    provider: str = "openalex"


def test_literature_selection_excludes_unrelated_and_deduplicates() -> None:
    candidates = [
        Candidate("Global Carbon Budget", "Atmospheric emissions and energy", "10.1/carbon"),
        Candidate("Retrieval budget allocation", "Ranking research for query evidence", "10.1/good"),
        Candidate("Retrieval budget allocation", "Ranking research for query evidence", "10.1/good"),
        Candidate("Scientific query retrieval", "Evidence ranking for documents", "10.1/second"),
    ]
    selected, audit = select_literature(
        candidates,
        focus_terms=["retriev", "budget", "rank", "quer", "evidence"],
        minimum_focus_terms=2,
        max_results=8,
    )
    assert [item.doi for item in selected] == ["10.1/good", "10.1/second"]
    assert len(audit) == 4
    assert audit[0]["accepted_by_focus"] is False


def test_literature_selection_deduplicates_arxiv_and_venue_records_by_title() -> None:
    candidates = [
        Candidate("Adaptive Retrieval", "Query budget allocation", "10.1/venue"),
        Candidate("Adaptive   Retrieval", "Query budget allocation", "10.1/arxiv"),
    ]
    selected, _ = select_literature(
        candidates,
        focus_terms=["retriev", "budget"],
        minimum_focus_terms=2,
        max_results=10,
    )
    assert len(selected) == 1


def test_direct_prior_reserved_even_when_generic_source_scores_higher() -> None:
    candidates = [
        Candidate("Retrieval ranking query benchmark fusion search", "Evidence retrieval", "10.1/generic"),
        Candidate("Adaptive Retrieval-Augmented Generation", "Question complexity routing", "10.1/direct"),
    ]
    selected, audit = select_literature(
        candidates,
        focus_terms=["retriev", "rank", "quer", "budget", "benchmark", "fusion", "adapt"],
        minimum_focus_terms=2,
        max_results=1,
        priority_title_term_groups=[["retriev"], ["allocat", "adapt", "depth"]],
        minimum_priority_results=1,
    )
    assert [item.doi for item in selected] == ["10.1/direct"]
    assert audit[1]["priority_title_match"] is True


def test_writer_receives_exact_primary_values_without_full_trial_grid() -> None:
    results = {
        "primary_comparison": {"absolute_difference": 0.023928751432388123},
        "datasets": {
            "scifact": {
                "primary_comparison": {"candidate_recall": 0.844444},
                "methods_by_budget": {"5": {"knn_calibration": {"recall": 0.844444, "ndcg": 0.7}}},
                "paired_bootstrap_by_budget": {"5": {"convex_fixed": {"recall": {"ci95_low": 0.01}, "ndcg": {"ci95_low": 0.0}}}},
                "fusion": {"selected_weights": {"dense": 0.5}, "top_train_trials": [{"weights": {"dense": 1.0}}] * 500},
            },
        },
    }
    view = _paper_results_view(results)
    assert view["primary_comparison"] == results["primary_comparison"]
    assert view["datasets"]["scifact"]["primary_comparison"] == results["datasets"]["scifact"]["primary_comparison"]
    assert view["datasets"]["scifact"]["recall_by_budget"]["5"]["knn_calibration"] == 0.844444
    assert view["datasets"]["scifact"]["paired_recall_by_budget"]["5"]["convex_fixed"] == {"ci95_low": 0.01}
    assert "top_train_trials" not in str(view)


def test_secondary_results_view_uses_measured_baselines_and_subsamples() -> None:
    def dataset(raw: float, binned: float, candidate: float) -> dict:
        return {
            "primary_budget": 5,
            "methods_by_budget": {"5": {
                "raw_jaccard_allocation": {"recall": raw},
                "binned_calibration": {"recall": binned},
                "knn_calibration": {"recall": candidate},
            }},
            "train_subsampling_stability": {
                "recall_mean": 0.84, "recall_std": 0.01,
                "runs": [{"seed": seed, "train_queries": 80, "recall": value}
                         for seed, value in ((42, 0.83), (43, 0.85))],
            },
        }
    results = {"datasets": {
        "scifact": dataset(0.79, 0.81, 0.84),
        "nfcorpus": dataset(0.12, 0.13, 0.14),
    }}
    assert _adaptive_baseline_rows(results)[0] == {
        "dataset": "SciFact", "raw_jaccard": 0.79,
        "binned_calibration": 0.81, "knn_calibration": 0.84,
    }
    assert _train_subsampling_rows(results)[1]["seeds"] == [42, 43]
    assert _train_subsampling_rows(results)[1]["min_recall"] == 0.83
    assert len(_paper_results_view(results)["datasets"]["scifact"]["train_subsampling_stability"]["runs"]) == 2


def test_paper_gate_requires_relevant_in_text_citations_and_batch_scope() -> None:
    config = {
        "enabled": True,
        "focus_terms": ["retriev", "budget"],
        "minimum_focus_terms": 2,
        "min_references": 1,
        "min_body_references": 1,
        "required_sections": ["Abstract", "Method", "References"],
        "required_patterns": [{"code": "batch_missing", "pattern": r"\bbatch\b"}],
        "forbidden_patterns": [{"code": "pathway_claim", "pattern": r"allocat[^.\n]{0,100}pathways?"}],
    }
    evidence = [
        {"id": "evidence_abc123", "title": "Retrieval budget allocation", "summary": "Ranking scientific evidence"},
        {"id": "evidence_def456", "title": "Global Carbon Budget", "summary": "Atmospheric emissions"},
    ]
    bad = (
        "## Abstract\nWe allocate document budget across pathways.\n"
        "## Method\nWe use a retrieval system.\n"
        "## References\n- [evidence_def456] Global Carbon Budget.\n"
    )
    bad_codes = {error["code"] for error in audit_draft(bad, evidence=evidence, config=config)["errors"]}
    assert {"paper_reference_off_topic", "paper_in_text_citations_insufficient", "batch_missing", "pathway_claim"} <= bad_codes

    good = (
        "## Abstract\nWe optimize document depth over a known batch.\n"
        "## Method\nThe evidence context budget is allocated after fusion [`evidence_abc123`]. "
        "We do not allocate budget across pathways.\n"
        "## References\n- [evidence_abc123] Retrieval budget allocation.\n"
    )
    assert audit_draft(good, evidence=evidence, config=config)["ok"] is True


def test_existing_route_b2_draft_is_not_accepted_as_submission() -> None:
    root = Path(__file__).resolve().parents[5]
    prior_draft = root / "runs" / "route_b2_team_real_20260922_155530" / "draft.md"
    prior_evidence = root / "runs" / "route_b2_team_real_20260922_155530" / "evidence" / "literature.json"
    if not prior_draft.exists() or not prior_evidence.exists():
        return
    import json

    literature = json.loads(prior_evidence.read_text(encoding="utf-8"))["evidence"]
    report = audit_draft(
        prior_draft.read_text(encoding="utf-8"),
        evidence=literature,
        config={
            "enabled": True,
            "focus_terms": ["retriev", "rank", "quer", "budget", "benchmark", "fusion", "rag", "evidence", "search"],
            "minimum_focus_terms": 2,
            "min_references": 4,
            "min_body_references": 4,
            "required_sections": ["Abstract", "Introduction", "Related Work", "Method", "Experimental Setup", "Results", "Limitations", "Conclusion", "References"],
            "required_patterns": [{"code": "batch_missing", "pattern": r"\bbatch\b"}],
            "forbidden_patterns": [{"code": "pathway_claim", "pattern": r"allocat[^.\n]{0,100}pathways?"}],
        },
    )
    assert not report["ok"]
    codes = {error["code"] for error in report["errors"]}
    assert "paper_reference_off_topic" in codes
    assert "pathway_claim" in codes


def test_public_prose_cannot_expose_audit_ids_or_continuous_integer_depth() -> None:
    draft = (
        "# Paper\n## Abstract\nClaim `claim_abc123` records the result.\n"
        "## Method\nThe allocated depth is a continuous function of features.\n"
        "## References\n"
    )
    report = audit_draft(
        draft,
        evidence=[],
        config={
            "enabled": True,
            "forbidden_patterns": [{
                "code": "paper_continuous_depth_claim",
                "pattern": r"(?:allocat(?:ed|ion)?\s+depth|document\s+depth)[^.\n]{0,100}\bcontinuous\s+function\b",
            }],
        },
    )
    codes = {error["code"] for error in report["errors"]}
    assert {"paper_internal_ids_in_prose", "paper_continuous_depth_claim"} <= codes


def test_negated_continuous_depth_statement_is_not_blocked() -> None:
    rule = {
        "code": "paper_continuous_depth_claim",
        "pattern": r"(?:allocat(?:ed|ion)?\s+depth|document\s+depth)[^.\n]{0,100}\bcontinuous\s+function\b",
    }
    for statement in (
        "The allocated document depth is always a discrete integer, never a continuous function of features.",
        "The document depth is an integer decision, not a continuous function of the features.",
    ):
        report = audit_draft("# Paper\n## Method\n" + statement, evidence=[], config={"enabled": True, "forbidden_patterns": [rule]})
        assert "paper_continuous_depth_claim" not in {error["code"] for error in report["errors"]}
    positive = audit_draft(
        "# Paper\n## Method\nThe document depth is a continuous function of features.",
        evidence=[], config={"enabled": True, "forbidden_patterns": [rule]},
    )
    assert "paper_continuous_depth_claim" in {error["code"] for error in positive["errors"]}


def test_pathway_negations_inside_the_matched_span_are_not_blocked() -> None:
    rule = {
        "code": "paper_pathway_allocation_claim",
        "pattern": r"(?:reallocat|allocat)[^.\n]{0,100}(?:pathways?|sub-quer(?:y|ies))",
    }
    for statement in (
        "The allocator does not rerank documents, rerun retrievers, or "
        "allocate budget among retrievers, retrieval pathways, or sub-queries.",
        "The allocator therefore redistributes a known total capacity rather "
        "than choosing a retrieval pathway per query.",
        "Route B2 is a batch-level document-depth allocation method, not a "
        "per-query passage-memory or retrieval-pathway router.",
    ):
        report = audit_draft(
            "# Paper\n## Method\n" + statement,
            evidence=[], config={"enabled": True, "forbidden_patterns": [rule]},
        )
        assert "paper_pathway_allocation_claim" not in {error["code"] for error in report["errors"]}
    positive = audit_draft(
        "# Paper\n## Method\nThe allocator allocates budget across retrieval pathways.",
        evidence=[], config={"enabled": True, "forbidden_patterns": [rule]},
    )
    assert "paper_pathway_allocation_claim" in {error["code"] for error in positive["errors"]}


def test_forbidden_phrase_with_built_in_negation_stays_enforceable() -> None:
    rule = {
        "code": "paper_dataset_claim_conflict",
        "pattern": r"\bdo\s+not\s+extend\s+(?:it|the\s+(?:effect|claim))\s+to\s+any\s+individual\s+dataset\b",
    }
    report = audit_draft(
        "# Paper\n## Discussion\nWe do not extend the effect to any individual dataset.",
        evidence=[], config={"enabled": True, "forbidden_patterns": [rule]},
    )
    assert "paper_dataset_claim_conflict" in {error["code"] for error in report["errors"]}


def test_direct_adaptive_retrieval_prior_is_required() -> None:
    draft = (
        "# Paper\n## Related Work\n"
        "A general retrieval benchmark is cited [`evidence_abc123`].\n"
        "## References\n- `evidence_abc123` Benchmark.\n"
    )
    config = {
        "enabled": True,
        "required_reference_topics": [{
            "code": "paper_direct_prior_missing",
            "title_term_groups": [["retriev"], ["budget", "allocat", "adapt", "depth"]],
        }],
    }
    benchmark = {"id": "evidence_abc123", "title": "Information Retrieval Benchmark", "summary": ""}
    result = audit_draft(draft, evidence=[benchmark], config=config)
    assert "paper_direct_prior_missing" in {error["code"] for error in result["errors"]}
    direct_prior = dict(benchmark, title="Adaptive Retrieval Budget Allocation")
    assert audit_draft(draft, evidence=[direct_prior], config=config)["ok"] is True


def test_paper_gate_rejects_duplicate_bibliography_records_for_same_work() -> None:
    draft = (
        "# Paper\n## Related Work\n"
        "We cite both records [`evidence_abc123`] [`evidence_def456`].\n"
        "## References\n- `evidence_abc123` Venue.\n- `evidence_def456` arXiv.\n"
    )
    evidence = [
        {"id": "evidence_abc123", "title": "Adaptive Retrieval", "summary": ""},
        {"id": "evidence_def456", "title": "Adaptive  Retrieval", "summary": ""},
    ]
    result = audit_draft(draft, evidence=evidence, config={"enabled": True})
    assert "paper_reference_duplicate_work" in {error["code"] for error in result["errors"]}

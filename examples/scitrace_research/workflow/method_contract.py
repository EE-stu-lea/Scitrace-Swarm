"""Explicit implementation facts for Route B2 writer and deterministic review.

These statements describe the checked-in experiment, not general retrieval
claims. They are passed as structured context instead of inferred from names.
"""
from __future__ import annotations

import re

ROUTE_B2_METHOD_CONTRACT = {
    "allocation": {"minimum_depth": 1, "maximum_depth": 10,
                   "budget_unit": "mean documents per query across the known batch",
                   "solver": "exact dynamic programming for the sum of predicted recalls",
                   "constraint": "sum of integer depths equals batch_size times mean budget; individual query depths can differ",
                   "scope": "allocate prefixes of the already-fused convex ranking, without reranking or rerunning retrievers"},
    "features": {
        "jaccard_top5_top10_top20": "mean pairwise 1-Jaccard between retriever top-k sets",
        "rbo_top20": "mean pairwise 1-RBO, depth 20, persistence 0.9",
        "rrf_margin": "1 - clipped((s1-s2)/abs(s1)), using top two RRF scores",
        "score_dispersion": "mean across union of top-20 documents of population SD of per-retriever min-max scores; missing scores zero",
        "log_query_length": "log(1 + number of word tokens)",
    },
    "calibration": {
        "standardization": "feature means and population standard deviations fitted on training queries only; zero scales replaced with one",
        "neighbors": "max(10, round(sqrt(number of calibration training queries))); not tuned on test qrels",
        "prediction": "inverse Euclidean-distance-weighted mean of neighbor recall curves with epsilon 1e-6; cumulative maximum ensures nondecreasing predictions",
        "fusion_tuning": "simplex grid step 0.25 selected by mean training recall across supplied budgets",
    },
    "feature_ablation": {
        "all_features": "all seven features",
        "rank_overlap": "three Jaccard disagreements and RBO disagreement (four features)",
        "jaccard_only": "only top-10 Jaccard disagreement (one feature), not all three Jaccard features",
        "rbo_only": "only depth-20 RBO disagreement (one feature)",
        "raw_jaccard_baseline": "direct budget allocation from top-10 Jaccard disagreement; distinct from the jaccard_only kNN ablation",
        "statistical_scope": "measured descriptive recalls; no paired uncertainty estimate against the feature ablations is supplied",
    },
    "fiqa": "independently trained lexical finance-domain evaluation; not zero-shot out-of-domain transfer",
    "training_subset_check": "restrict kNN reference set to 80%; fusion weights, scaler and neighbor count remain fixed",
    "uncertainty": "all budgets already have pointwise paired-bootstrap intervals; query resampling conditions on realized rankings and allocation, not fresh-batch reallocation",
    "randomness": "configuration seed controls sampling; paired-bootstrap seeds are derived from configuration seed plus budget/metric/baseline offsets, not one identical seed for every comparison",
    "resource_scope": "document-context budget only; not retrieval compute, token length or downstream answer quality",
}


def _negated(text: str, match: re.Match[str], *, missing_intervals: bool = False) -> bool:
    """Avoid rejecting an explicit correction of a known erroneous claim."""
    start = max(text.rfind(".", 0, match.start()), text.rfind("\n", 0, match.start()),
                text.rfind(";", 0, match.start())) + 1
    context = text[max(start, match.start() - 60):match.end()]
    if missing_intervals:
        # Here "without/no intervals" is the erroneous assertion itself, not
        # a negation of that assertion ("not only at budget five" still is).
        context = re.sub(r"\b(?:without|no)\s+intervals?\b", "intervals", context,
                         flags=re.IGNORECASE)
    return bool(re.search(r"\b(?:not|never|no|without|cannot|can't|doesn't|don't)\b", context, re.IGNORECASE))


def _training_scope_is_asserted(match: re.Match[str]) -> bool:
    qualifier = re.search(r"\b(?:independently|separately|own|dataset[- ]specific)\b", match.group(), re.IGNORECASE)
    prefix = match.group()[:qualifier.start()] if qualifier else ""
    # Local negation attaches to the training qualifier. A correct explanation
    # may negate zero-shot transfer earlier in the same sentence.
    return not bool(re.search(r"\b(?:not|never|no)\s+(?:(?:been|being|a|an|its|any|ever)\s+){0,2}$",
                              prefix, re.IGNORECASE))


def audit_method_contract(public_draft: str) -> list[dict[str, str]]:
    """Catch observed regressions; absence of a match is not scientific proof."""
    rules = [
        ("paper_zero_depth_claim", r"(?:depth|integer|range)[^\n.]{0,100}\b0\s*(?:\.\.|[-–]|to|\\ldots|…)[^\n]{0,10}\b10\b",
         "Route B2 allocates integer depths 1 through 10; zero depth is infeasible."),
        ("paper_per_query_fixed_allocation", r"(?:each|every)\s+(?:test\s+)?query[^.\n]{0,65}\b(?:exactly|receives|contributes)\s+(?:two|three|five|eight|[2358])\s+documents",
         "The adaptive allocation fixes the batch mean, while individual query depths vary. Only the fixed-depth baseline uses the same depth for every query."),
        ("paper_missing_sweep_intervals_claim", r"(?:intervals?[^.\n]{0,50}(?:computed|available)\s+only\s+(?:at|for)\s+budget\s*5|other\s+(?:sweep\s+)?(?:points|budgets)[^.\n]{0,35}(?:no\s+interval|without\s+interval)|only\s+budget\s*5\s+has[^.\n]{0,30}interval)",
         "Stored paired-bootstrap intervals are available at every swept budget."),
        ("paper_audit_metadata_in_prose", r"\bStructured References\s+(?:JSON\s+)?block\b|\bFigure\s+\d+\s*\([^\n)]*figures/",
         "Describe the scientific figure or supplementary artifact without internal JSON block names or file paths."),
        ("paper_fiqa_zero_shot_claim", r"\bFiQA\b[^.\n]{0,100}\bzero[- ]shot\b|\bzero[- ]shot\b[^.\n]{0,100}\bFiQA\b",
         "FiQA has its own training queries and fitted calibration. Describe it as a separately trained lexical finance-domain evaluation, not zero-shot transfer."),
        ("paper_subsampling_refit_claim", r"(?:subset|subsampl\w*)[^.\n]{0,90}\b(?:refit|retrain|retun)\w*\s+(?:the\s+)?(?:scaler|standardiz\w*|fusion\s+weights|neighbou?r\s+count)",
         "Training-subset sensitivity restricts the kNN reference set; it does not refit the scaler, fusion weights or neighbor count."),
    ]
    errors = []
    for code, pattern, message in rules:
        for match in re.finditer(pattern, public_draft, re.IGNORECASE):
            if code != "paper_audit_metadata_in_prose" and _negated(
                public_draft, match, missing_intervals=code == "paper_missing_sweep_intervals_claim"
            ):
                continue
            if code == "paper_per_query_fixed_allocation":
                start = max(public_draft.rfind(".", 0, match.start()), public_draft.rfind("\n", 0, match.start())) + 1
                end = re.search(r"[.\n]", public_draft[match.end():])
                sentence = public_draft[start:match.end() + end.start() if end else len(public_draft)]
                if re.search(r"\b(?:fixed[- ]depth|baseline)\b", sentence, re.IGNORECASE):
                    continue
            errors.append({"code": code, "message": message, "node_id": "", "severity": "error"})
            break
    training_scope = list(re.finditer(
        r"\bFiQA\b[^.\n]{0,180}\b(?:independently|separately|own|dataset[- ]specific)\b[^.\n]{0,70}\b(?:train\w*|fit\w*|fitted|calibrat\w*)\b"
        r"|\b(?:independently|separately|dataset[- ]specific)\b[^.\n]{0,70}\b(?:train\w*|fit\w*|fitted|calibrat\w*)\b[^.\n]{0,100}\bFiQA\b",
        public_draft, re.IGNORECASE,
    ))
    if re.search(r"\bFiQA\b", public_draft, re.IGNORECASE) and not any(
        _training_scope_is_asserted(match) for match in training_scope
    ):
        errors.append({"code": "paper_fiqa_training_scope_missing", "node_id": "", "severity": "error",
                       "message": "State explicitly that FiQA uses its own training queries and separately fitted lexical calibration; a domain change alone does not establish zero-shot transfer."})
    visible_prose = re.sub(r"(!?\[[^\]]*\])\([^)]*\)", r"\1", public_draft)
    if re.search(r"\bTeam Handoff\b|(?<![:/\w])(?:figures|experiments|handoffs|evidence)/[\w./-]+|\b(?:run_manifest|model_calls|resource_usage)\.(?:jsonl?|csv)\b", visible_prose, re.IGNORECASE):
        errors.append({"code": "paper_internal_workflow_in_prose", "node_id": "", "severity": "error",
                       "message": "Remove Team Handoff markers and internal run paths from manuscript prose; retain scientific captions and keep machine provenance in the audit artifact."})
    return errors

"""A question-led manuscript contract; requirements, never generated findings."""
from __future__ import annotations

import re

RESEARCH_QUESTIONS = {
    "RQ1": "Does predicted-recall allocation improve evidence recall at a matched mean document budget?",
    "RQ2": "Which queries gain or lose evidence, and how does redistribution affect zero-recall outcomes?",
    "RQ3": "How sensitive are the results to features, training subsets, budgets and the separately trained lexical FiQA boundary?",
}

EXPERIMENT_CONTROLS = [
    {"comparison": "Fixed vs kNN allocation", "held_fixed": "queries, fused rankings, total admitted documents",
     "varied": "per-query integer depths", "evidence": "paired query outcomes and conditional intervals"},
    {"comparison": "Raw Jaccard / binned / kNN", "held_fixed": "dataset, fused rankings, mean budget 5",
     "varied": "allocation policy", "evidence": "stored descriptive recalls; no paired baseline intervals"},
    {"comparison": "Feature subsets", "held_fixed": "dataset and evaluation budget",
     "varied": "features supplied to recall calibration", "evidence": "stored ablation point estimates"},
    {"comparison": "80% training subsets", "held_fixed": "fusion, standardizer, neighbor count",
     "varied": "kNN reference subset", "evidence": "five subset draws; not independent full-run seeds"},
    {"comparison": "Lexical FiQA", "held_fixed": "allocation formulation",
     "varied": "corpus, own training data, lexical retrievers", "evidence": "boundary condition; confounded domain/retriever change"},
]


def narrative_context() -> dict:
    return {
        "research_questions": RESEARCH_QUESTIONS,
        "controlled_comparisons": EXPERIMENT_CONTROLS,
        "results_subsections": ["RQ1: Matched-budget effectiveness", "RQ2: Redistribution and query risk",
                                "RQ3: Sensitivity and boundary conditions"],
        "argument_order": "Question -> controlled comparison -> measured evidence -> bounded interpretation.",
        "reference_design_only": "Use an overview diagram, explicit controls and decomposed effects. Do not borrow findings or reliability/aptitude metrics from the reference conversation paper.",
        "appendix_scope": "Optional appendix after References for extended protocol details; the core questions, method, main effects and negative result must remain in the main text.",
    }


def has_substantive_figure_caption(draft: str, filename: str) -> bool:
    """Accept a descriptive image label or a real adjacent Figure caption.

    Manuscripts commonly use a short alt label for accessibility and put the
    scientific explanation in the next italic paragraph. The earlier check
    counted only alt-label characters and rejected such complete figures.
    """
    image = re.compile(r"!\[(?P<alt>[^\]\n]+)\]\(figures/" + re.escape(filename) + r"\)")
    for match in image.finditer(draft):
        if len(re.sub(r"\W+", "", match.group("alt"))) >= 35:
            return True
        caption = re.match(
            r"\s*\*{0,2}Figure\s+\d+\s*[:.]\s*(?P<body>[^\n]+)",
            draft[match.end():], flags=re.IGNORECASE,
        )
        if caption and len(re.sub(r"\W+", "", caption.group("body").rstrip("* "))) >= 35:
            return True
    return False


def audit_narrative(public_draft: str) -> list[dict]:
    """Catch missing argument structure and orchestration leakage, not judge novelty."""
    errors = []
    def add(code, message):
        errors.append({"code": code, "message": message, "node_id": "", "severity": "error"})
    if not re.match(r"\s*#{1,2}\s+\S", public_draft):
        add("paper_coordinator_preamble", "Start with the paper title. Remove delegation/status chatter before it.")
    headings = re.findall(r"(?m)^#{3,4}\s+(.+)$", public_draft)
    for rq in RESEARCH_QUESTIONS:
        if not any(re.search(r"\b" + rq + r"\b", title, re.I) for title in headings):
            add("paper_research_question_missing", f"Add a substantive Results subsection labelled {rq} and answer its supplied research question.")
    if not has_substantive_figure_caption(public_draft, "method_overview.pdf"):
        add("paper_method_overview_missing", "Embed figures/method_overview.pdf in Method with a caption explaining the frozen training/test boundary and changed allocation component.")
    if not has_substantive_figure_caption(public_draft, "query_risk.pdf"):
        add("paper_query_risk_figure_missing", "Embed figures/query_risk.pdf in Results with a caption explaining query gains/losses and the descriptive, fixed-allocation scope.")
    if not re.search(r"(?mi)^#{2,3}\s+(?:\d+[.)]?\s+)?Discussion\s*$", public_draft):
        add("paper_discussion_missing", "Add Discussion linking observed query risk and negative results to deployment implications without claiming measured downstream improvements.")
    return errors

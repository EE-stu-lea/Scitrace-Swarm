"""Deterministic research-paper checks applied before accepting an Agent draft.

These checks make failures actionable for the Writer Agent. They do not write or
rewrite paper prose, and they never replace the scientific provenance gate.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


def focus_matches(title: str, summary: str, terms: Iterable[str]) -> list[str]:
    """Return the configured topical stems present in title or abstract.

    Terms are explicit configuration, not a learned score or a provider's
    uncalibrated relevance field. Keeping the decision deterministic allows a
    reviewer to replay the selection from the cached literature metadata.
    """
    text = f"{title} {summary}".casefold()
    return sorted({
        term.casefold() for term in terms
        if term and re.search(r"\b" + re.escape(term.casefold()) + r"\w*\b", text)
    })


def literature_key(item: Any) -> str:
    title = re.sub(r"\s+", " ", str(getattr(item, "title", "") or "").strip().casefold())
    if title:
        # Providers often expose the same work once as a conference record and
        # once as an arXiv record with different DOI fields. The normalized
        # title is the stable cross-provider identity for paper selection.
        return f"title:{title}"
    doi = str(getattr(item, "doi", "") or "").strip().casefold()
    return f"doi:{doi}"


def title_covers_groups(title: str, groups: list[list[str]]) -> bool:
    """Match each required title concept using configured word stems."""
    return bool(groups) and all(
        bool(focus_matches(title, "", group)) for group in groups if group
    ) and all(bool(group) for group in groups)


def select_literature(
    candidates: Iterable[Any],
    *,
    focus_terms: list[str],
    minimum_focus_terms: int,
    max_results: int,
    priority_title_term_groups: list[list[str]] | None = None,
    minimum_priority_results: int = 0,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Deduplicate and rank verified provider results with a replayable audit."""
    best: dict[str, tuple[int, int, Any, bool]] = {}
    audit: list[dict[str, Any]] = []
    groups = priority_title_term_groups or []
    for index, item in enumerate(candidates):
        matches = focus_matches(item.title, item.summary, focus_terms)
        title_matches = focus_matches(item.title, "", focus_terms)
        relevant = len(matches) >= minimum_focus_terms
        priority = title_covers_groups(item.title, groups)
        score = len(matches) + len(title_matches)
        audit.append({
            "title": item.title,
            "doi": item.doi,
            "provider": item.provider,
            "matched_focus_terms": matches,
            "title_matches": title_matches,
            "accepted_by_focus": relevant,
            "priority_title_match": priority,
            "selection_score": score,
        })
        if not relevant:
            continue
        key = literature_key(item)
        prior = best.get(key)
        if prior is None or score > prior[0]:
            best[key] = (score, index, item, priority)
    ranked = sorted(best.values(), key=lambda entry: (-entry[0], entry[1]))
    priority_entries = [entry for entry in ranked if entry[3]][:max(0, minimum_priority_results)]
    selected_entries = list(priority_entries)
    for entry in ranked:
        if len(selected_entries) >= max_results:
            break
        if entry not in selected_entries:
            selected_entries.append(entry)
    selected = [entry[2] for entry in selected_entries]
    return selected, audit


def _error(code: str, message: str, node_id: str = "") -> dict[str, str]:
    return {"code": code, "message": message, "node_id": node_id, "severity": "error"}


def audit_draft(
    draft: str,
    *,
    evidence: list[dict[str, Any]],
    config: dict[str, Any],
    table_rows: list[dict[str, Any]] | None = None,
    adaptive_rows: list[dict[str, Any]] | None = None,
    stability_rows: list[dict[str, Any]] | None = None,
    results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check review-critical structure, literature relevance, and method scope."""
    if not bool(config.get("enabled", False)):
        return {"ok": True, "error_count": 0, "errors": [], "enabled": False}
    errors: list[dict[str, str]] = []
    required_sections = [str(x) for x in config.get("required_sections", [])]
    headings = [
        re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", line.strip()).casefold()
        for line in re.findall(r"^#{1,4}\s+(.+)$", draft, flags=re.MULTILINE)
    ]
    if re.search(r"^\*\*Abstract\*\*", draft, flags=re.IGNORECASE | re.MULTILINE):
        headings.append("abstract")
    for section in required_sections:
        if not any(section.casefold() == heading for heading in headings):
            errors.append(_error("paper_section_missing", f"Add a substantive {section} section."))

    public_draft = draft.split("## Structured References", 1)[0]
    results_heading = re.search(
        r"(?m)^(?P<marks>#{1,4})\s+(?:\d+(?:\.\d+)*[.)]?\s+)?Results\s*$",
        public_draft,
    )
    results_text = ""
    if results_heading:
        tail = public_draft[results_heading.end():]
        end = len(tail)
        for heading in re.finditer(r"(?m)^(?P<marks>#{1,4})\s+", tail):
            if len(heading.group("marks")) <= len(results_heading.group("marks")):
                end = heading.start()
                break
        results_text = tail[:end]
    if config.get("require_results_table"):
        table_match = re.search(
            r"(?m)^\|[^\n]+\|\s*\n\|[\s:|\-]+\|\s*\n(?P<rows>(?:\|[^\n]+\|\s*\n?)+)",
            results_text,
        )
        if table_match is None:
            errors.append(_error("paper_results_table_missing", "Add a Markdown results table with candidate, fixed baseline, difference, and 95% interval."))
        else:
            headers = [cell.strip().casefold() for cell in table_match.group().splitlines()[0].strip("|").split("|")]
            header_text = " ".join(headers)
            if not all(term in header_text for term in ("candidate", "baseline", "difference", "95")):
                errors.append(_error("paper_results_table_columns", "The results table needs candidate, baseline, difference, and 95% interval columns."))
            rows = [line.strip("|").split("|") for line in table_match.group("rows").splitlines() if line.strip()]
            for expected in table_rows or []:
                dataset = str(expected["dataset"])
                row = next((cells for cells in rows if cells and dataset.casefold() in cells[0].casefold()), None)
                if row is None:
                    errors.append(_error("paper_results_table_row_missing", f"Add the {dataset} row to the results table."))
                    continue
                numbers = [float(number) for number in re.findall(r"(?<![\w.])[-+]?\d*\.\d+", " ".join(row[1:]))]
                targets = [float(expected[key]) for key in ("candidate", "fixed_baseline", "difference")]
                targets += [float(x) for x in expected["ci95"]]
                if len(numbers) < 5 or any(abs(actual - target) > 0.00006 for actual, target in zip(numbers[:5], targets)):
                    errors.append(_error("paper_results_table_value_mismatch", f"The {dataset} row does not match the verified candidate, baseline, difference, and 95% interval values."))
    if config.get("require_results_figure") and not re.search(
        r"!\[[^\]\n]+\]\(figures/recall_by_budget\.pdf\)", results_text
    ):
        errors.append(_error("paper_results_figure_missing", "Embed the verified budget-sweep figure in Results using figures/recall_by_budget.pdf."))
    if config.get("require_table_captions"):
        tables = list(re.finditer(r"(?m)^\|[^\n]+\|\s*\n\|[\s:|\-]+\|", results_text))
        for index, table in enumerate(tables[:2], start=1):
            preceding = results_text[max(0, table.start() - 240):table.start()]
            if not re.search(rf"\bTable\s+{index}\s*:\s*[^\n]+", preceding, flags=re.IGNORECASE):
                errors.append(_error("paper_results_table_caption_missing", f"Place an Agent-written Table {index} caption immediately above its Results table."))
    if config.get("require_adaptive_baseline_table"):
        adaptive_table = next((match for match in re.finditer(
            r"(?m)^\|[^\n]+\|\s*\n\|[\s:|\-]+\|\s*\n(?P<rows>(?:\|[^\n]+\|\s*\n?)+)",
            results_text,
        ) if all(term in match.group().splitlines()[0].casefold() for term in ("raw", "binned", "knn"))), None)
        if adaptive_table is None:
            errors.append(_error("paper_adaptive_baseline_table_missing", "Add a Results table for raw Jaccard, binned calibration, and kNN allocation at the primary budget."))
        else:
            rows = [line.strip("|").split("|") for line in adaptive_table.group("rows").splitlines() if line.strip()]
            for expected in adaptive_rows or []:
                dataset = str(expected["dataset"])
                row = next((cells for cells in rows if cells and dataset.casefold() in cells[0].casefold()), None)
                if row is None:
                    errors.append(_error("paper_adaptive_baseline_row_missing", f"Add the {dataset} adaptive-baseline row."))
                    continue
                numbers = [float(number) for number in re.findall(r"(?<![\w.])[-+]?\d*\.\d+", " ".join(row[1:]))]
                targets = [float(expected[key]) for key in ("raw_jaccard", "binned_calibration", "knn_calibration")]
                if len(numbers) < 3 or any(abs(actual - target) > 0.00006 for actual, target in zip(numbers[:3], targets)):
                    errors.append(_error("paper_adaptive_baseline_value_mismatch", f"The {dataset} adaptive-baseline row does not match the measured recalls."))
    if config.get("require_train_subsampling_summary"):
        stability_context = results_text.casefold()
        if re.search(r"\b(?:re[- ]?ran|reran|re[- ]?executed)\s+(?:the\s+)?(?:full\s+)?(?:pipeline|experiment)\b", stability_context):
            errors.append(_error("paper_train_subsampling_rerun_claim", "The artifact performed five training-subset allocations inside one completed experiment; do not say the full pipeline or experiment was rerun five times."))
        if stability_rows and any(len(set(row.get("seeds") or [])) > 1 for row in stability_rows) and re.search(
            r"\ball\s+runs\s+(?:use|used|have|had)\s+seed\s*42\b", public_draft, flags=re.IGNORECASE,
        ):
            errors.append(_error("paper_seed_scope_ambiguity", "Distinguish the primary experiment/bootstrap seed 42 from training-subset sampling seeds 42 through 46."))
        if not re.search(r"train(?:ing)?[^.\n]{0,80}(?:subsampl|subset)|(?:subsampl|subset)[^.\n]{0,80}train", stability_context):
            errors.append(_error("paper_train_subsampling_context_missing", "Describe the five 80% training-query subset resamples as a limited sensitivity check."))
        if not re.search(r"\b80\s*%|\b0\.8\b", stability_context):
            errors.append(_error("paper_train_subsampling_fraction_missing", "State that each sensitivity run used 80% of training queries."))
        for expected in stability_rows or []:
            dataset = str(expected["dataset"])
            target_mean = float(expected["mean_recall"])
            target_sd = float(expected["population_sd"])
            fragments = [match.group() for match in re.finditer(
                re.escape(dataset) + r"[^\n]{0,260}", results_text, flags=re.IGNORECASE,
            )]
            has_values = any(
                any(abs(value - target_mean) <= 0.00006 for value in
                    [float(number) for number in re.findall(r"(?<![\w.])\d*\.\d+", fragment)])
                and any(abs(value - target_sd) <= 0.00006 for value in
                    [float(number) for number in re.findall(r"(?<![\w.])\d*\.\d+", fragment)])
                for fragment in fragments
            )
            if not has_values:
                errors.append(_error("paper_train_subsampling_value_missing", f"Report the measured {dataset} 80% subset mean and population SD accurately in Results."))
    if results:
        for match in re.finditer(
            r"\b(?P<dataset>SciFact|NFCorpus)\b[^.\n]{0,220}?"
            r"\b(?:largest|biggest)\s+(?:recall\s+)?(?:gap|gain|improvement)\s+at\s+budget\s+(?P<budget>\d+)\b",
            results_text,
            flags=re.IGNORECASE,
        ):
            dataset = match.group("dataset").casefold()
            methods = ((results.get("datasets") or {}).get(dataset) or {}).get("methods_by_budget") or {}
            if not methods:
                continue
            gaps = {
                int(budget): float(values["knn_calibration"]["recall"])
                - float(values["convex_fixed"]["recall"])
                for budget, values in methods.items()
            }
            actual_budget = max(gaps, key=gaps.get)
            claimed_budget = int(match.group("budget"))
            if claimed_budget != actual_budget:
                errors.append(_error(
                    "paper_budget_sweep_peak_mismatch",
                    f"{match.group('dataset')} has its largest measured candidate-minus-baseline recall gap at budget {actual_budget}, not {claimed_budget}; correct or omit the peak-budget claim.",
                ))
        if re.search(r"\b(?:separation|gap|advantage)\s+(?:is|was)\s+largest\s+at\s+intermediate\s+(?:depths|budgets)\b", results_text, flags=re.IGNORECASE):
            for dataset, data in (results.get("datasets") or {}).items():
                methods = data.get("methods_by_budget") or {}
                if len(methods) < 3:
                    continue
                gaps = {int(budget): float(values["knn_calibration"]["recall"]) - float(values["convex_fixed"]["recall"])
                        for budget, values in methods.items()}
                peak = max(gaps, key=gaps.get)
                if peak in (min(gaps), max(gaps)):
                    errors.append(_error("paper_budget_sweep_qualitative_mismatch", f"The largest measured gap for {dataset} is at endpoint budget {peak}; describe the two panels separately."))
                    break
    internal_ids = sorted(set(re.findall(
        r"\b(?:rq|hyp|claim|artifact)_[0-9a-f]+\b", public_draft
    )))
    if internal_ids:
        errors.append(_error(
            "paper_internal_ids_in_prose",
            "Keep research-question, hypothesis, claim, and artifact IDs only in the Structured References JSON block; write the visible paper without audit identifiers.",
        ))
    reference_section = re.split(
        r"^#{1,4}\s+(?:\d+(?:\.\d+)*[.)]?\s+)?References\s*$",
        public_draft,
        maxsplit=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    references = reference_section[1] if len(reference_section) == 2 else ""
    cited_ids = sorted(set(re.findall(r"evidence_[0-9a-f]+", references)))
    min_refs = int(config.get("min_references", 0) or 0)
    if len(cited_ids) < min_refs:
        errors.append(_error(
            "paper_references_insufficient",
            f"References include {len(cited_ids)} traceable sources; require at least {min_refs} directly relevant sources.",
        ))
    body = reference_section[0]
    # The PDF exporter maps backtick-delimited evidence IDs to numbered citations.
    body_cited_ids = set(re.findall(r"`(evidence_[0-9a-f]+)`", body))
    min_body_refs = int(config.get("min_body_references", 0) or 0)
    if len(body_cited_ids & set(cited_ids)) < min_body_refs:
        errors.append(_error(
            "paper_in_text_citations_insufficient",
            f"Only {len(body_cited_ids & set(cited_ids))} bibliography sources are cited in the paper body; require at least {min_body_refs}.",
        ))
    for citation_id in sorted(body_cited_ids - set(cited_ids)):
        errors.append(_error(
            "paper_in_text_citation_missing_reference",
            "An in-text evidence ID has no corresponding bibliography entry.", citation_id,
        ))
    nodes = {str(node.get("id") or ""): node for node in evidence if isinstance(node, dict)}
    cited_titles: dict[str, str] = {}
    for citation_id in cited_ids:
        node = nodes.get(citation_id) or {}
        normalized_title = re.sub(r"\s+", " ", str(node.get("title") or "").strip().casefold())
        if normalized_title and normalized_title in cited_titles:
            errors.append(_error(
                "paper_reference_duplicate_work",
                "The bibliography cites the same work more than once; retain one verified record.",
                citation_id,
            ))
        elif normalized_title:
            cited_titles[normalized_title] = citation_id
    focus_terms = [str(x) for x in config.get("focus_terms", [])]
    minimum_focus_terms = int(config.get("minimum_focus_terms", 0) or 0)
    for citation_id in cited_ids:
        node = nodes.get(citation_id)
        if node is None:
            errors.append(_error("paper_reference_unknown", "Reference has no literature provenance record.", citation_id))
            continue
        matches = focus_matches(str(node.get("title") or ""), str(node.get("summary") or ""), focus_terms)
        if len(matches) < minimum_focus_terms:
            errors.append(_error(
                "paper_reference_off_topic",
                f"Reference '{node.get('title')}' matches only {matches}; replace it with a directly relevant, verified source.",
                citation_id,
            ))

    for requirement in config.get("required_reference_topics", []):
        if not isinstance(requirement, dict):
            continue
        groups = requirement.get("title_term_groups")
        if not isinstance(groups, list) or not groups:
            continue
        def covers_groups(node: dict[str, Any]) -> bool:
            return title_covers_groups(str(node.get("title") or ""), groups)
        count = sum(1 for citation_id in cited_ids if citation_id in nodes and covers_groups(nodes[citation_id]))
        minimum = int(requirement.get("minimum", 1) or 1)
        if count < minimum:
            errors.append(_error(
                str(requirement.get("code") or "paper_direct_prior_missing"),
                str(requirement.get("message") or "Cite a verified directly comparable prior study."),
            ))

    for rule in config.get("forbidden_patterns", []):
        if not isinstance(rule, dict) or not rule.get("pattern"):
            continue
        matches = re.finditer(str(rule["pattern"]), public_draft, flags=re.IGNORECASE | re.DOTALL)
        asserted = False
        for match in matches:
            sentence_start = max(public_draft.rfind(".", 0, match.start()), public_draft.rfind("\n", 0, match.start()))
            prefix = public_draft[max(sentence_start + 1, match.start() - 60):match.start()]
            # The depth rule matches through "continuous function". A writer
            # can correctly say "depth is never a continuous function", with
            # the negation inside the matched span rather than in its prefix.
            matched_text = match.group()
            negated_depth = (
                str(rule.get("code")) == "paper_continuous_depth_claim"
                and bool(re.search(
                    r"\b(?:not|never|no|without|cannot|can't|doesn't|don't)\b"
                    r"[^.\n]{0,70}\bcontinuous\s+function\b",
                    matched_text, flags=re.IGNORECASE,
                ))
            )
            if not negated_depth and not re.search(r"\b(?:not|never|no|without|cannot|can't|doesn't|don't)\b", prefix, flags=re.IGNORECASE):
                asserted = True
                break
        if asserted:
            errors.append(_error(
                str(rule.get("code") or "paper_method_misdescribed"),
                str(rule.get("message") or "Revise the method statement to match the experiment implementation."),
            ))
    for rule in config.get("required_patterns", []):
        if not isinstance(rule, dict) or not rule.get("pattern"):
            continue
        if not re.search(str(rule["pattern"]), public_draft, flags=re.IGNORECASE | re.DOTALL):
            errors.append(_error(
                str(rule.get("code") or "paper_scope_missing"),
                str(rule.get("message") or "State the experiment scope and limits explicitly."),
            ))
    return {"ok": not errors, "error_count": len(errors), "errors": errors, "enabled": True,
            "reference_count": len(cited_ids)}

"""Pre-submit structural self-check for a SciTrace manuscript draft.

Usage:
    python scripts/checklist.py --draft <draft.md>

Prints a JSON array of ``{"item", "status", "detail"}`` entries and exits 0
only when every item passes (exit 1 otherwise, exit 2 for a missing file).
The validator is deliberately self-contained: it imports nothing from the
workflow packages so it stays runnable from any checkout, while mirroring
the deterministic acceptance gate's parsing rules (section extraction, prose
word counting, table/caption detection). Numeric artifact agreement is the
gate's job; this checklist covers structure, citations and safe wording.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED_SECTIONS = (
    "Abstract", "Introduction", "Related Work", "Method", "Experimental Setup",
    "Results", "Discussion", "Limitations", "Conclusion", "References",
)
MINIMUM_SECTION_WORDS = {"Method": 300, "Experimental Setup": 180, "Results": 550, "Discussion": 180}
RQ_SUBSECTIONS = ("RQ1", "RQ2", "RQ3")
MIN_REFERENCES = 6
MIN_CAPTION_CHARS = 20
MIN_FIGURE_CAPTION_CHARS = 35

_TABLE_RE = re.compile(r"(?m)^\|[^\n]+\|\s*\n\|[\s:|\-]+\|\s*\n(?P<rows>(?:\|[^\n]+\|\s*\n?)+)")

# Terms the Writer must avoid entirely, including inside negated sentences.
# The acceptance gate additionally exempts explicit negations, so removing the
# term outright keeps the draft safe under both the gate and this checklist.
FORBIDDEN_PATTERNS = (
    ("zero_shot", re.compile(r"\bzero[- ]shot\b", re.IGNORECASE),
     'Remove "zero-shot" even in negated sentences; describe FiQA as a separately '
     'trained finance-domain lexical evaluation.'),
    ("pathway_allocation",
     re.compile(r"\ballocat\w*[^.\n]{0,100}\b(?:pathways?|sub-quer(?:y|ies))\b"
                r"|\b(?:pathways?|sub-quer(?:y|ies))\b[^.\n]{0,100}\ballocat\w*", re.IGNORECASE),
     'Never pair allocation with pathways or sub-queries; write that document depth is '
     'allocated jointly across a known batch after fusion.'),
    ("continuous_depth",
     re.compile(r"\bcontinuous\b[^.\n]{0,60}\bdepth\b|\bdepth\b[^.\n]{0,60}\bcontinuous\b", re.IGNORECASE),
     'Avoid "continuous" near "depth", even to negate it; state discrete integer depths '
     'and real-valued recall predictions.'),
    ("preregistration", re.compile(r"\bpreregist\w*|\bpre[- ]registered\b|\bpre[- ]specified\b", re.IGNORECASE),
     'Drop preregistration wording, including negations; state that the primary budget '
     'lacks a timestamped protocol.'),
)


def _public_manuscript(draft: str) -> str:
    """Keep the audit-only Structured References JSON block out of every check."""
    return re.split(
        r"^#{1,4}\s+(?:\d+(?:\.\d+)*[.)]?\s+)?Structured References\s*$",
        draft, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE,
    )[0]


def _split_references(public: str) -> tuple[str, str]:
    parts = re.split(
        r"^#{1,4}\s+(?:\d+(?:\.\d+)*[.)]?\s+)?References\s*$",
        public, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE,
    )
    return (parts[0], parts[1]) if len(parts) == 2 else (public, "")


def _section_texts(text: str) -> dict[str, str]:
    """A section includes its subsections but ends before a sibling/parent."""
    normalized = re.sub(r"^\*\*Abstract\*\*\s*$", "## Abstract", text,
                        flags=re.IGNORECASE | re.MULTILINE)
    matches = list(re.finditer(r"(?m)^(?P<marks>#{1,4})\s+(?P<title>.+?)\s*$", normalized))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", match.group("title")).casefold()
        end = next((other.start() for other in matches[index + 1:]
                    if len(other.group("marks")) <= len(match.group("marks"))), len(normalized))
        sections[title] = normalized[match.end():end]
    return sections


def _prose_word_count(text: str) -> int:
    """Count explanatory prose, not tables, captions, references or metadata."""
    text = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s*(?:\|.*|#{1,6}\s+.*)$", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = "\n\n".join(paragraph for paragraph in re.split(r"\n\s*\n", text)
                       if not re.match(r"\s*(?:\*\*)?(?:Figure|Table)\s+\d+\s*[:.]", paragraph,
                                       flags=re.IGNORECASE))
    text = re.sub(r"\[[^\]]*\]\((?:https?://)?[^)]*\)", "", text)
    text = re.sub(r"\b(?:evidence|claim|artifact|rq|hyp)_[\w]+\b|https?://\S+", "", text)
    text = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]", "", text, flags=re.DOTALL)
    return len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", text))


def _table_has_caption(text: str, table_start: int, index: int) -> bool:
    preceding = text[:table_start].rstrip().split("\n\n")[-1].strip()
    caption = re.match(
        rf"(?:\*\*)?Table\s+{index}\s*[:.]\s*(?:\*\*)?\s*(?P<body>.+)",
        preceding.lstrip("*_ "),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return caption is not None and len(re.sub(r"\W+", "", caption.group("body"))) >= MIN_CAPTION_CHARS


def _figure_with_caption(text: str, filename: str, *, require_caption: bool) -> bool:
    image = re.compile(r"!\[(?P<alt>[^\]\n]+)\]\(figures/" + re.escape(filename) + r"\)")
    for match in image.finditer(text):
        if not require_caption:
            return True
        if len(re.sub(r"\W+", "", match.group("alt"))) >= MIN_FIGURE_CAPTION_CHARS:
            return True
        caption = re.match(
            r"\s*\*{0,2}Figure\s+\d+\s*[:.]\s*(?P<body>[^\n]+)",
            text[match.end():], flags=re.IGNORECASE,
        )
        if caption and len(re.sub(r"\W+", "", caption.group("body").rstrip("* "))) >= MIN_FIGURE_CAPTION_CHARS:
            return True
    return False


def _find_query_outcome_table(results_text: str) -> re.Match[str] | None:
    names = {"wins": "win", "ties": "tie", "losses": "loss", "rescued": "rescue", "newly": "newly"}
    for match in _TABLE_RE.finditer(results_text):
        headers = [cell.strip().casefold() for cell in match.group().splitlines()[0].strip("|").split("|")]
        indices = {key: next((i for i, header in enumerate(headers) if name in header), None)
                   for key, name in names.items()}
        if all(index is not None for index in indices.values()) and len(set(indices.values())) == len(names):
            return match
    return None


def _find_ablation_table(results_text: str) -> re.Match[str] | None:
    for match in _TABLE_RE.finditer(results_text):
        headers = [re.sub(r"[-_]", " ", cell.strip().casefold())
                   for cell in match.group().splitlines()[0].strip("|").split("|")]
        indices = {key: next((i for i, header in enumerate(headers) if all(term in header for term in terms)), None)
                   for key, terms in {"all_features": ("feature",), "rank_overlap": ("rank", "overlap"),
                                      "jaccard_only": ("jaccard",), "rbo_only": ("rbo",)}.items()}
        if all(index is not None for index in indices.values()) and len(set(indices.values())) == 4:
            return match
    return None


def run_checks(draft: str) -> list[dict[str, str]]:
    """Return the ordered checklist report for one manuscript draft."""
    report: list[dict[str, str]] = []

    def add(item: str, ok: bool, detail: str) -> None:
        report.append({"item": item, "status": "pass" if ok else "fail", "detail": detail})

    public = _public_manuscript(draft)
    body, references = _split_references(public)

    title_ok = bool(re.match(r"\s*#{1,2}\s+\S", public))
    add("manuscript_title", title_ok,
        "Public manuscript starts with its `# Title` heading."
        if title_ok else "Start the manuscript with its `# Title` heading; remove any preamble before it.")

    headings = [re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", line.strip()).casefold()
                for line in re.findall(r"^#{1,4}\s+(.+)$", public, flags=re.MULTILINE)]
    if re.search(r"^\*\*Abstract\*\*", public, flags=re.IGNORECASE | re.MULTILINE):
        headings.append("abstract")
    missing_sections = [section for section in REQUIRED_SECTIONS if section.casefold() not in headings]
    add("required_sections", not missing_sections,
        "All required sections are present." if not missing_sections else
        "Add substantive sections for: " + ", ".join(missing_sections) + ".")

    sections = _section_texts(public)
    counts = {name: _prose_word_count(text) for name, text in sections.items() if name != "references"}
    deficits = [f"{name} has {counts.get(name.casefold(), 0)} prose words (minimum {minimum})"
                for name, minimum in MINIMUM_SECTION_WORDS.items()
                if counts.get(name.casefold(), 0) < int(minimum)]
    add("section_word_minimums", not deficits,
        "Section prose word minimums are satisfied." if not deficits else
        "; ".join(deficits) + ". Tables, image captions, references and metadata do not count.")

    sub_headings = re.findall(r"(?m)^#{3,4}\s+(.+)$", public)
    missing_rqs = [rq for rq in RQ_SUBSECTIONS
                   if not any(re.search(r"\b" + rq + r"\b", title, re.IGNORECASE) for title in sub_headings)]
    add("rq_subsections", not missing_rqs,
        "RQ1-RQ3 Results subsections are labelled." if not missing_rqs else
        "Add substantive Results subsections labelled RQ1, RQ2 and RQ3 (for example `### RQ1: Matched-budget effectiveness`).")

    results_text = sections.get("results", "")
    tables = list(_TABLE_RE.finditer(results_text))
    primary_ok = False
    if tables:
        header_text = " ".join(cell.strip().casefold()
                               for cell in tables[0].group().splitlines()[0].strip("|").split("|"))
        primary_ok = all(term in header_text for term in ("candidate", "baseline", "difference", "95"))
    add("primary_results_table", primary_ok,
        "The first Results table carries candidate, baseline, difference and 95% interval columns."
        if primary_ok else
        "Make the first Results table the primary paired comparison with candidate, fixed-baseline, "
        "difference and 95% interval columns.")

    adaptive_ok = any(
        all(term in match.group().splitlines()[0].casefold() for term in ("raw", "binned", "knn"))
        for match in tables
    )
    add("adaptive_baseline_table", adaptive_ok,
        "A Results table compares Raw Jaccard, binned calibration and kNN allocation."
        if adaptive_ok else
        "Add a Results table with Raw Jaccard, Binned calibration and kNN allocation columns at the primary budget.")

    add("query_outcome_table", _find_query_outcome_table(results_text) is not None,
        "A Results table reports Wins, Ties, Losses, Rescued and Newly missed counts."
        if _find_query_outcome_table(results_text) is not None else
        "Add Table 3 with Dataset, Wins, Ties, Losses, Rescued and Newly missed columns.")

    add("feature_ablation_table", _find_ablation_table(results_text) is not None,
        "A Results table reports the All-features / Rank-overlap / Jaccard-only / RBO-only ablations."
        if _find_ablation_table(results_text) is not None else
        "Add a Results table with All features, Rank overlap, Jaccard only and RBO only columns.")

    caption_failures = [f"Table {index}" for index, table in enumerate(tables, start=1)
                        if not _table_has_caption(results_text, table.start(), index)]
    add("results_table_captions", len(tables) >= 4 and not caption_failures,
        "Every Results table has a substantive numbered caption in appearance order."
        if len(tables) >= 4 and not caption_failures else
        ("Results needs at least four tables (primary, adaptive baselines, query outcomes, ablations). "
         if len(tables) < 4 else "Place an Agent-written caption immediately above: " + ", ".join(caption_failures) + "."))

    method_text = sections.get("method", "")
    add("figure_method_overview", _figure_with_caption(method_text, "method_overview.pdf", require_caption=True),
        "figures/method_overview.pdf is embedded in Method with a substantive caption."
        if _figure_with_caption(method_text, "method_overview.pdf", require_caption=True) else
        "Embed figures/method_overview.pdf once in Method with a caption explaining training-only fitting, "
        "frozen test inference and the changed integer allocation component.")

    recall_count = len(re.findall(r"!\[[^\]\n]+\]\(figures/recall_by_budget\.pdf\)", results_text))
    add("figure_recall_by_budget", recall_count == 1,
        "figures/recall_by_budget.pdf is embedded exactly once in Results."
        if recall_count == 1 else
        f"Embed figures/recall_by_budget.pdf exactly once in Results (found {recall_count} times).")

    add("figure_budget_effect", _figure_with_caption(results_text, "budget_effect.pdf", require_caption=True),
        "figures/budget_effect.pdf is embedded in Results with a substantive caption."
        if _figure_with_caption(results_text, "budget_effect.pdf", require_caption=True) else
        "Embed figures/budget_effect.pdf in Results with a caption describing the three datasets and "
        "pointwise paired-bootstrap 95% intervals at every budget.")

    add("figure_query_risk", _figure_with_caption(results_text, "query_risk.pdf", require_caption=True),
        "figures/query_risk.pdf is embedded in Results with a substantive caption."
        if _figure_with_caption(results_text, "query_risk.pdf", require_caption=True) else
        "Embed figures/query_risk.pdf once in Results with a caption explaining per-query gains/losses and the "
        "descriptive fixed-allocation scope.")

    cited_ids = sorted(set(re.findall(r"evidence_[0-9a-f]+", references)))
    add("references_count", len(cited_ids) >= MIN_REFERENCES,
        f"References list {len(cited_ids)} traceable evidence IDs."
        if len(cited_ids) >= MIN_REFERENCES else
        f"References list only {len(cited_ids)} traceable evidence IDs; require at least {MIN_REFERENCES} "
        "directly relevant verified sources.")

    body_cited = set(re.findall(r"`(evidence_[0-9a-f]+)`", body))
    covered = body_cited & set(cited_ids)
    uncovered = sorted(body_cited - set(cited_ids))
    add("body_citations_covered", len(covered) >= MIN_REFERENCES and not uncovered,
        f"{len(covered)} bibliography sources are cited in the body with backtick evidence IDs."
        if len(covered) >= MIN_REFERENCES and not uncovered else
        (f"Only {len(covered)} bibliography sources are cited in the body (require {MIN_REFERENCES}); "
         + ("in-text IDs missing a bibliography entry: " + ", ".join(uncovered) if uncovered
            else "cite more verified sources in the body using backtick evidence IDs") + "."))

    violations: list[str] = []
    for code, pattern, advice in FORBIDDEN_PATTERNS:
        match = pattern.search(body)
        if match:
            violations.append(f"{code} (\"{match.group()[:60]}\"): {advice}")
    add("forbidden_terms_absent", not violations,
        "No forbidden term appears in the public prose." if not violations else
        " | ".join(violations))

    internal_ids = sorted(set(re.findall(r"\b(?:rq|hyp|claim|artifact)_[0-9a-f]+\b", public)))
    add("internal_ids_hidden", not internal_ids,
        "Research-question, hypothesis, claim and artifact IDs stay in the Structured References block."
        if not internal_ids else
        "Remove internal IDs from visible prose: " + ", ".join(internal_ids) + ".")

    visible = re.sub(r"(!?\[[^\]]*\])\([^)]*\)", r"\1", public)
    leaked = re.search(
        r"\bTeam Handoff\b|(?<![:/\w])(?:figures|experiments|handoffs|evidence)/[\w./-]+"
        r"|\b(?:run_manifest|model_calls|resource_usage)\.(?:jsonl?|csv)\b",
        visible, re.IGNORECASE,
    )
    add("workflow_metadata_hidden", leaked is None,
        "No Team Handoff marker or internal run path appears in prose."
        if leaked is None else
        f"Remove orchestration leakage from prose: \"{leaked.group()[:60]}\"; keep machine provenance in the audit artifact.")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-submit structural self-check for a SciTrace manuscript draft.")
    parser.add_argument("--draft", required=True, help="Path to the draft markdown file.")
    args = parser.parse_args(argv)

    path = Path(args.draft)
    if not path.is_file():
        print(f"draft file not found: {path}", file=sys.stderr)
        return 2
    report = run_checks(path.read_text(encoding="utf-8-sig"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] == "pass" for item in report) else 1


if __name__ == "__main__":
    sys.exit(main())

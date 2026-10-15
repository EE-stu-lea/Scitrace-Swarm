"""One-hop citation-graph expansion over OpenAlex for the literature stage.

The first retrieval round is keyword-based; this module upgrades it with a
seed-driven citation-graph hop. Every network call is a real OpenAlex request
and counts toward ``retrieval_call_count``. Any expansion failure degrades to
a ``skipped`` outcome so the stage never loses its base results.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Callable

from jiuwenswarm.agents.harness.research.literature import (
    LiteratureSearchResult,
    OpenAlexLiteratureProvider,
    _coerce_result,
    _http_get_json,
    _score_result,
)

logger = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_MAX_EXPANSION_RESULTS = 20
DEFAULT_MAX_SEEDS = 3
DEFAULT_PER_SEED_LIMIT = 10
DEFAULT_RELATED_PER_SEED = 5

FetchJson = Callable[[str, dict[str, Any], float], dict[str, Any]]


class CitationExpansionError(RuntimeError):
    """An OpenAlex expansion request failed after exhausting its retries."""


def openalex_work_id(result: LiteratureSearchResult) -> str:
    """Extract the canonical ``W...`` OpenAlex work id from a provider result."""
    raw = result.raw_payload if isinstance(result.raw_payload, dict) else {}
    candidates = [raw.get("id")]
    ids = raw.get("ids")
    if isinstance(ids, dict):
        candidates.append(ids.get("openalex"))
    for value in candidates:
        token = str(value or "").strip().rsplit("/", 1)[-1]
        if re.fullmatch(r"W\d+", token):
            return token
    return ""


def select_seed_papers(
    results: list[LiteratureSearchResult],
    *,
    max_seeds: int = DEFAULT_MAX_SEEDS,
) -> list[LiteratureSearchResult]:
    """Pick the most relevant filtered results that expose an OpenAlex work id."""
    eligible = [result for result in results if openalex_work_id(result)]
    ranked = sorted(eligible, key=lambda result: (-_score_result(result), result.title.casefold()))
    return ranked[: max(0, int(max_seeds))]


def related_work_ids(
    seed: LiteratureSearchResult,
    *,
    limit: int = DEFAULT_RELATED_PER_SEED,
) -> list[str]:
    """Read the seed's ``related_works`` list without an extra network call."""
    raw = seed.raw_payload if isinstance(seed.raw_payload, dict) else {}
    related = raw.get("related_works")
    if not isinstance(related, list):
        return []
    ids: list[str] = []
    for value in related:
        token = str(value or "").strip().rsplit("/", 1)[-1]
        if re.fullmatch(r"W\d+", token) and token not in ids:
            ids.append(token)
        if len(ids) >= max(0, int(limit)):
            break
    return ids


def _parse_openalex_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Reuse the canonical OpenAlex field mapping from the search provider."""
    # ``_parse_items`` only reads its payload, so bypass ``__init__`` (which
    # creates a cache directory) and keep a single source of parsing truth.
    provider = OpenAlexLiteratureProvider.__new__(OpenAlexLiteratureProvider)
    return provider._parse_items(payload)


def parse_expansion_payload(payload: Any) -> list[LiteratureSearchResult]:
    """Coerce an OpenAlex response (results envelope or single work) to results."""
    if not isinstance(payload, dict):
        return []
    if not isinstance(payload.get("results"), list):
        payload = {"results": [payload]} if payload.get("id") else {"results": []}
    return [
        _coerce_result(item, provider="openalex")
        for item in _parse_openalex_items(payload)
        if isinstance(item, dict)
    ]


def _default_fetch_json(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    return _http_get_json(url, params=params, timeout=timeout)


def _fetch_with_retries(
    fetch_json: FetchJson,
    url: str,
    params: dict[str, Any],
    *,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(max(0, int(retries)) + 1):
        try:
            return fetch_json(url, dict(params), float(timeout_seconds))
        except Exception as exc:  # noqa: BLE001 - retry any transport failure
            response = getattr(exc, "response", None)
            if getattr(response, "status_code", None) == 404:
                # A 404 is deterministic (e.g. OpenAlex lists a related work it
                # cannot serve), so retrying cannot succeed and only wastes
                # the expansion API budget. Fail fast; callers isolate it.
                raise CitationExpansionError(
                    f"OpenAlex expansion request failed: {exc}"
                ) from exc
            last_error = exc
    raise CitationExpansionError(f"OpenAlex expansion request failed: {last_error}") from last_error


def expand_seed_by_citations(
    seed: LiteratureSearchResult,
    *,
    fetch_json: FetchJson,
    timeout_seconds: float,
    retries: int,
    per_seed_limit: int = DEFAULT_PER_SEED_LIMIT,
    related_per_seed: int = DEFAULT_RELATED_PER_SEED,
) -> tuple[list[LiteratureSearchResult], int]:
    """One-hop expansion of a single seed via its citing and related works.

    ``cited_by`` uses the OpenAlex ``cites:<id>`` filter (works citing the
    seed); ``related`` fetches the seed's ``related_works`` records directly.
    """
    seed_id = openalex_work_id(seed)
    if not seed_id:
        raise CitationExpansionError("seed result exposes no OpenAlex work id")
    candidates: list[LiteratureSearchResult] = []
    api_calls = 0
    citing = _fetch_with_retries(
        fetch_json,
        OPENALEX_WORKS_URL,
        {"filter": f"cites:{seed_id}", "per-page": max(1, int(per_seed_limit))},
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    api_calls += 1
    candidates.extend(parse_expansion_payload(citing))
    for work_id in related_work_ids(seed, limit=related_per_seed):
        try:
            related = _fetch_with_retries(
                fetch_json,
                f"{OPENALEX_WORKS_URL}/{work_id}",
                {},
                timeout_seconds=timeout_seconds,
                retries=retries,
            )
        except CitationExpansionError as exc:
            # OpenAlex ``related_works`` may contain dangling ids that the same
            # API 404s on. Skip the dead reference and keep expanding the rest.
            api_calls += 1
            logger.warning("skipping dangling related work %s: %s", work_id, exc)
            continue
        api_calls += 1
        candidates.extend(parse_expansion_payload(related))
    return candidates, api_calls


def _work_keys(result: LiteratureSearchResult) -> set[str]:
    """Identity keys (OpenAlex id, DOI, normalized title) for cross-dedup."""
    keys: set[str] = set()
    work_id = openalex_work_id(result)
    if work_id:
        keys.add(f"openalex:{work_id}")
    doi = str(result.doi or "").strip().casefold()
    if doi:
        keys.add(f"doi:{doi}")
    title = re.sub(r"\s+", " ", str(result.title or "").strip().casefold())
    if title:
        keys.add(f"title:{title}")
    return keys


def expansion_work_key(result: LiteratureSearchResult) -> str:
    """Primary stable key used for the seed-to-expansion provenance mapping."""
    work_id = openalex_work_id(result)
    if work_id:
        return f"openalex:{work_id}"
    doi = str(result.doi or "").strip().casefold()
    if doi:
        return f"doi:{doi}"
    return f"title:{re.sub(r'\s+', ' ', str(result.title or '').strip().casefold())}"


def _match_count(text: str, terms: list[str]) -> int:
    return len({
        str(term).casefold()
        for term in terms
        if term and re.search(r"\b" + re.escape(str(term).casefold()) + r"\w*\b", text)
    })


def filter_and_merge_expansion(
    base_results: list[LiteratureSearchResult],
    expansion_candidates: list[tuple[LiteratureSearchResult, str]],
    *,
    focus_terms: list[str],
    minimum_focus_terms: int,
    max_expansion_results: int,
) -> tuple[list[LiteratureSearchResult], dict[str, str]]:
    """Re-filter expansion candidates and dedupe them against the base results.

    Relevance re-uses the stage's focus_terms rule (title+summary stem
    matches); a candidate is dropped when any identity key (OpenAlex id, DOI,
    normalized title) already exists in the base results or an earlier kept
    candidate. Kept papers keep a mapping from their primary work key to the
    seed that produced them.
    """
    existing: set[str] = set()
    for result in base_results:
        existing |= _work_keys(result)
    scored: list[tuple[int, int, LiteratureSearchResult, str]] = []
    for index, (candidate, seed_id) in enumerate(expansion_candidates):
        if _match_count(f"{candidate.title} {candidate.summary}", focus_terms) < minimum_focus_terms:
            continue
        score = (
            _match_count(f"{candidate.title} {candidate.summary}", focus_terms)
            + _match_count(candidate.title, focus_terms)
        )
        scored.append((score, index, candidate, seed_id))
    kept: list[LiteratureSearchResult] = []
    provenance: dict[str, str] = {}
    for _, _, candidate, seed_id in sorted(scored, key=lambda entry: (-entry[0], entry[1])):
        if len(kept) >= max(0, int(max_expansion_results)):
            break
        keys = _work_keys(candidate)
        if keys & existing:
            continue
        existing |= keys
        kept.append(candidate)
        provenance[expansion_work_key(candidate)] = seed_id
    return kept, provenance


@dataclass(frozen=True)
class CitationExpansionOutcome:
    """Result of one expansion attempt; ``skipped`` never blocks the stage."""

    status: str
    seeds: tuple[LiteratureSearchResult, ...]
    kept: tuple[LiteratureSearchResult, ...]
    expanded_candidate_count: int
    api_call_count: int
    provenance: dict[str, str]
    error: str = ""

    def to_audit_event(self) -> dict[str, Any]:
        event: dict[str, Any] = {
            "event": "literature_citation_expansion",
            "status": self.status,
            "seed_count": len(self.seeds),
            "seed_openalex_ids": [openalex_work_id(seed) for seed in self.seeds],
            "expanded_count": self.expanded_candidate_count,
            "kept_count": len(self.kept),
            "kept_provenance": dict(self.provenance),
            "api_call_count": self.api_call_count,
        }
        if self.error:
            event["error"] = self.error
        return event


def run_citation_expansion(
    base_results: list[LiteratureSearchResult],
    *,
    focus_terms: list[str],
    minimum_focus_terms: int = 2,
    max_expansion_results: int = DEFAULT_MAX_EXPANSION_RESULTS,
    max_seeds: int = DEFAULT_MAX_SEEDS,
    per_seed_limit: int = DEFAULT_PER_SEED_LIMIT,
    related_per_seed: int = DEFAULT_RELATED_PER_SEED,
    timeout_seconds: float = 20.0,
    retries: int = 2,
    fetch_json: FetchJson | None = None,
) -> CitationExpansionOutcome:
    """Expand the filtered first-round results by one citation-graph hop.

    Any failure (no expandable seeds, network error after retries) degrades
    to ``status="skipped"`` and the literature stage continues with its base
    results only.
    """
    fetch = fetch_json or _default_fetch_json
    api_calls = 0

    def counted_fetch(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        nonlocal api_calls
        api_calls += 1
        return fetch(url, params, timeout)

    try:
        seeds = select_seed_papers(base_results, max_seeds=max_seeds)
        if not seeds:
            return CitationExpansionOutcome(
                "skipped", (), (), 0, 0, {},
                "no first-round result exposes an OpenAlex work id",
            )
        candidates: list[tuple[LiteratureSearchResult, str]] = []
        usable_seeds: list[LiteratureSearchResult] = []
        seed_failures: list[str] = []
        for seed in seeds:
            try:
                found, _ = expand_seed_by_citations(
                    seed,
                    fetch_json=counted_fetch,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    per_seed_limit=per_seed_limit,
                    related_per_seed=related_per_seed,
                )
            except CitationExpansionError as exc:
                # Per-seed isolation: one dead seed (e.g. its cites: lookup is
                # unreachable) must not discard the other seeds' expansions.
                logger.warning(
                    "citation expansion: seed %s failed: %s", openalex_work_id(seed), exc
                )
                seed_failures.append(f"{openalex_work_id(seed)}: {exc}")
                continue
            usable_seeds.append(seed)
            candidates.extend((candidate, openalex_work_id(seed)) for candidate in found)
        if not usable_seeds:
            return CitationExpansionOutcome(
                "skipped", (), (), 0, api_calls, {},
                "all candidate seeds failed citation expansion ("
                + "; ".join(seed_failures)
                + ")",
            )
        kept, provenance = filter_and_merge_expansion(
            base_results,
            candidates,
            focus_terms=focus_terms,
            minimum_focus_terms=minimum_focus_terms,
            max_expansion_results=max_expansion_results,
        )
        return CitationExpansionOutcome(
            "expanded", tuple(usable_seeds), tuple(kept), len(candidates), api_calls, provenance,
        )
    except Exception as exc:  # noqa: BLE001 - expansion must never block the stage
        logger.warning("citation expansion skipped after failure: %s", exc)
        return CitationExpansionOutcome(
            "skipped", (), (), 0, api_calls, {}, f"{type(exc).__name__}: {exc}"
        )


__all__ = [
    "CitationExpansionError",
    "CitationExpansionOutcome",
    "DEFAULT_MAX_EXPANSION_RESULTS",
    "OPENALEX_WORKS_URL",
    "expand_seed_by_citations",
    "expansion_work_key",
    "filter_and_merge_expansion",
    "openalex_work_id",
    "parse_expansion_payload",
    "related_work_ids",
    "run_citation_expansion",
    "select_seed_papers",
]

from __future__ import annotations

import types
from typing import Any

import pytest

from ._stubs import install_openjiuwen_stubs

install_openjiuwen_stubs()

from jiuwenswarm.agents.harness.research.citation_expansion import (
    CitationExpansionError,
    _fetch_with_retries,
    expand_seed_by_citations,
    expansion_work_key,
    filter_and_merge_expansion,
    openalex_work_id,
    parse_expansion_payload,
    run_citation_expansion,
    select_seed_papers,
)
from jiuwenswarm.agents.harness.research.literature import LiteratureSearchResult

FOCUS_TERMS = ["retriev", "budget", "adapt"]


def _result(
    title: str,
    *,
    work_id: str = "",
    doi: str = "",
    summary: str = "",
    related: list[str] | None = None,
    reliability: float | None = 0.5,
    verified: bool = True,
) -> LiteratureSearchResult:
    raw: dict[str, Any] = {}
    if work_id:
        raw["id"] = f"https://openalex.org/{work_id}"
    if related is not None:
        raw["related_works"] = related
    return LiteratureSearchResult(
        title=title,
        authors=("Alice",),
        year=2026,
        doi=doi,
        source_url="https://example.org/a",
        summary=summary,
        reliability_score=reliability,
        provider="openalex",
        raw_payload=raw,
        verified_doi=verified,
        verified_url=verified,
    )


def _work(work_id: str, title: str, summary_words: list[str], *, doi: str = "",
          relevance: float = 0.5) -> dict[str, Any]:
    return {
        "id": f"https://openalex.org/{work_id}",
        "display_name": title,
        "authorships": [{"author": {"display_name": "Bob"}}],
        "publication_year": 2025,
        "doi": f"https://doi.org/{doi}" if doi else None,
        "primary_location": {"landing_page_url": "https://example.org/paper"},
        "abstract_inverted_index": {word: [index] for index, word in enumerate(summary_words)},
        "relevance_score": relevance,
    }


class _FakeFetch:
    """Records calls and replays canned responses or raises canned errors."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append((url, dict(params)))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeHTTPError(RuntimeError):
    """Mimic ``requests`` HTTP errors: the status code rides on ``.response``."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"{status_code} Client Error")
        self.response = types.SimpleNamespace(status_code=status_code)


def test_openalex_work_id_extraction() -> None:
    assert openalex_work_id(_result("A", work_id="W1")) == "W1"
    nested = LiteratureSearchResult(
        title="B", authors=(), year=None, doi="", source_url="", summary="",
        reliability_score=None, provider="openalex",
        raw_payload={"ids": {"openalex": "https://openalex.org/W7"}},
    )
    assert openalex_work_id(nested) == "W7"
    assert openalex_work_id(_result("C")) == ""


def test_select_seed_papers_ranks_by_relevance_and_requires_id() -> None:
    strong = _result("Adaptive Budget", work_id="W1", reliability=0.9)
    weak = _result("Weak Paper", work_id="W2", reliability=0.4)
    untracked = _result("Untracked Best", reliability=0.99)  # no OpenAlex id
    selected = select_seed_papers([untracked, weak, strong], max_seeds=2)
    assert [openalex_work_id(item) for item in selected] == ["W1", "W2"]
    assert [openalex_work_id(item) for item in select_seed_papers([weak, strong], max_seeds=1)] == ["W1"]


def test_parse_expansion_payload_envelope_and_single_work() -> None:
    envelope = {"results": [
        _work("W2", "Second Paper", ["retrieval", "budget"], doi="10.1/w2"),
        _work("W3", "Third Paper", ["adaptive"]),
    ]}
    parsed = parse_expansion_payload(envelope)
    assert [item.title for item in parsed] == ["Second Paper", "Third Paper"]
    assert parsed[0].doi == "10.1/w2"
    assert parsed[0].provider == "openalex"
    single = parse_expansion_payload(_work("W9", "Solo Paper", ["adapt"]))
    assert [item.title for item in single] == ["Solo Paper"]
    assert openalex_work_id(single[0]) == "W9"


def test_filter_and_merge_expansion_filters_dedupes_and_traces() -> None:
    base = [_result("Base Paper", work_id="W1", doi="10.1/base", summary="retrieval budget adapt")]
    candidates = [
        (_result("Cats", summary="cats"), "W9"),  # below minimum_focus_terms
        (_result("Retrieval Budget Again", doi="10.1/base", summary="retrieval budget adapt"), "W9"),  # duplicate DOI
        (_result("Base Paper", doi="10.1/other", summary="retrieval budget adapt"), "W9"),  # duplicate title
        (_result("Adaptive Retrieval Budget Allocation", work_id="W50", doi="10.1/d",
                  summary="adaptive retrieval budget allocation"), "W9"),
        (_result("Budget Evidence Search", work_id="W51", doi="10.1/e",
                  summary="budget search evidence retrieval"), "W8"),
    ]
    kept, provenance = filter_and_merge_expansion(
        base, candidates,
        focus_terms=FOCUS_TERMS, minimum_focus_terms=2, max_expansion_results=2,
    )
    assert [item.title for item in kept] == ["Adaptive Retrieval Budget Allocation", "Budget Evidence Search"]
    assert provenance == {
        expansion_work_key(kept[0]): "W9",
        expansion_work_key(kept[1]): "W8",
    }
    single, _ = filter_and_merge_expansion(
        base, candidates, focus_terms=FOCUS_TERMS, minimum_focus_terms=2, max_expansion_results=1,
    )
    assert [item.title for item in single] == ["Adaptive Retrieval Budget Allocation"]


def test_run_citation_expansion_with_mock_fetch() -> None:
    seed = _result(
        "Adaptive Retrieval Budget Allocation", work_id="W100", doi="10.1/seed",
        summary="adaptive retrieval budget allocation",
        related=["https://openalex.org/W200"], reliability=0.9,
    )
    citing = {"results": [
        _work("W300", "Budget Evidence Search Study", ["budget", "evidence", "search", "retrieval"], doi="10.1/x300"),
        _work("W301", "Unrelated Cats", ["cats"]),
    ]}
    related_work = _work("W200", "Adaptive Budget Retrieval Survey", ["adaptive", "budget", "retrieval"],
                          doi="10.1/x200")
    fetch = _FakeFetch([citing, related_work])

    outcome = run_citation_expansion(
        [seed], focus_terms=FOCUS_TERMS, minimum_focus_terms=2, fetch_json=fetch,
    )
    assert outcome.status == "expanded"
    assert outcome.api_call_count == 2
    assert outcome.expanded_candidate_count == 3
    assert [openalex_work_id(item) for item in outcome.kept] == ["W200", "W300"]
    assert outcome.provenance == {"openalex:W200": "W100", "openalex:W300": "W100"}
    assert outcome.seeds == (seed,)
    assert fetch.calls[0][1]["filter"] == "cites:W100"
    assert fetch.calls[1][0].endswith("/works/W200")

    event = outcome.to_audit_event()
    assert event["event"] == "literature_citation_expansion"
    assert event["status"] == "expanded"
    assert event["seed_openalex_ids"] == ["W100"]
    assert event["kept_count"] == 2
    assert event["api_call_count"] == 2


def test_run_citation_expansion_degrades_on_fetch_error() -> None:
    seed = _result("Adaptive Retrieval Budget", work_id="W100", summary="adaptive retrieval budget")
    fetch = _FakeFetch([RuntimeError("network down"), RuntimeError("network down")])

    outcome = run_citation_expansion(
        [seed], focus_terms=FOCUS_TERMS, minimum_focus_terms=2,
        max_expansion_results=5, retries=1, fetch_json=fetch,
    )
    assert outcome.status == "skipped"
    assert outcome.error.startswith("all candidate seeds failed citation expansion")
    assert "network down" in outcome.error
    assert outcome.kept == ()
    assert outcome.api_call_count == 2
    assert len(fetch.calls) == 2  # one retry, then the dead seed is isolated


def test_fetch_with_retries_fails_fast_on_404() -> None:
    fetch = _FakeFetch([_FakeHTTPError(404)])
    with pytest.raises(CitationExpansionError):
        _fetch_with_retries(
            fetch, "https://api.openalex.org/works/W999", {},
            timeout_seconds=1, retries=3,
        )
    assert len(fetch.calls) == 1  # a 404 is deterministic: retrying wastes budget


def test_expand_seed_by_citations_skips_dangling_related_work() -> None:
    seed = _result(
        "Adaptive Retrieval Budget", work_id="W100", doi="10.1/seed",
        summary="adaptive retrieval budget",
        related=["https://openalex.org/W200", "https://openalex.org/W999"],
    )
    related_work = _work("W200", "Adaptive Budget Retrieval Survey",
                         ["adaptive", "budget", "retrieval"], doi="10.1/x200")
    fetch = _FakeFetch([
        {"results": []},        # cites:W100 lookup
        related_work,           # live related work
        _FakeHTTPError(404),    # dangling id OpenAlex itself listed
    ])
    candidates, api_calls = expand_seed_by_citations(
        seed, fetch_json=fetch, timeout_seconds=1, retries=2,
    )
    assert [item.title for item in candidates] == ["Adaptive Budget Retrieval Survey"]
    assert api_calls == 3  # the dead reference still consumed one real call
    assert len(fetch.calls) == 3  # and was not retried


def test_run_citation_expansion_isolates_dead_seed() -> None:
    dead = _result("Dead Seed Paper", work_id="W1", reliability=0.99,
                   summary="adaptive retrieval budget")
    live = _result("Live Seed Paper", work_id="W2", reliability=0.8,
                   summary="adaptive retrieval budget")
    citing = {"results": [
        _work("W300", "Budget Evidence Search Study",
              ["budget", "evidence", "search", "retrieval"], doi="10.1/x300"),
    ]}
    fetch = _FakeFetch([RuntimeError("network down"), RuntimeError("network down"), citing])
    outcome = run_citation_expansion(
        [dead, live], focus_terms=FOCUS_TERMS, minimum_focus_terms=2,
        retries=1, fetch_json=fetch,
    )
    assert outcome.status == "expanded"
    assert [openalex_work_id(seed) for seed in outcome.seeds] == ["W2"]
    assert [openalex_work_id(item) for item in outcome.kept] == ["W300"]
    assert outcome.api_call_count == 3  # failed attempts and the live seed's call


def test_run_citation_expansion_skips_without_seeds() -> None:
    base = [_result("No Id Paper", summary="retrieval budget")]
    outcome = run_citation_expansion(base, focus_terms=FOCUS_TERMS, minimum_focus_terms=2)
    assert outcome.status == "skipped"
    assert "no first-round result" in outcome.error
    assert outcome.api_call_count == 0
    assert outcome.to_audit_event()["status"] == "skipped"


def test_expand_seed_by_citations_raises_after_retries() -> None:
    seed = _result("Adaptive Retrieval Budget", work_id="W100", summary="adaptive retrieval budget")
    with pytest.raises(CitationExpansionError):
        expand_seed_by_citations(
            seed, fetch_json=_FakeFetch([RuntimeError("boom")]),
            timeout_seconds=1, retries=0,
        )
    with pytest.raises(CitationExpansionError):
        expand_seed_by_citations(
            _result("No Id"), fetch_json=_FakeFetch([]),
            timeout_seconds=1, retries=0,
        )

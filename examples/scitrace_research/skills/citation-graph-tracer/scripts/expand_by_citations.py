"""One-hop OpenAlex citation-graph expansion from a first-round seed list.

Usage:
    python scripts/expand_by_citations.py --seeds seeds.json --config filter.json --out expanded.json

``seeds.json`` holds the first-round literature records (a JSON list, or an
object with a ``results`` list). Each record may expose its OpenAlex work id
via ``openalex_id``, ``id``, ``ids: {"openalex": ...}`` or ``raw_payload``,
plus ``title``/``summary`` and optional ``related_works``. ``filter.json``
supplies ``focus_terms`` and the expansion limits. Only api.openalex.org is
contacted. The output JSON reports status, kept records and seed provenance;
any failure degrades to ``status="skipped"`` (exit code 1) instead of raising.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULTS: dict[str, Any] = {
    "focus_terms": [],
    "minimum_focus_terms": 2,
    "max_expansion_results": 20,
    "max_seeds": 3,
    "per_seed_limit": 10,
    "related_per_seed": 5,
    "timeout_seconds": 20.0,
    "retries": 2,
}


class _Skipped(RuntimeError):
    """Any expansion failure; the caller degrades it to status="skipped"."""


def _work_id(record: dict[str, Any]) -> str:
    """Extract the canonical ``W...`` OpenAlex work id from a record."""
    containers = [record]
    raw = record.get("raw_payload")
    if isinstance(raw, dict):
        containers.append(raw)
    for container in containers:
        candidates = [container.get("openalex_id"), container.get("id")]
        ids = container.get("ids")
        if isinstance(ids, dict):
            candidates.append(ids.get("openalex"))
        for value in candidates:
            token = str(value or "").strip().rsplit("/", 1)[-1]
            if re.fullmatch(r"W\d+", token):
                return token
    return ""


def _related_ids(record: dict[str, Any], limit: int) -> list[str]:
    related = record.get("related_works")
    if related is None and isinstance(record.get("raw_payload"), dict):
        related = record["raw_payload"].get("related_works")
    ids: list[str] = []
    for value in related or []:
        token = str(value or "").strip().rsplit("/", 1)[-1]
        if re.fullmatch(r"W\d+", token) and token not in ids:
            ids.append(token)
        if len(ids) >= max(0, int(limit)):
            break
    return ids


def _focus_matches(text: str, terms: list[str]) -> list[str]:
    """Configured topical stems present in the text (stem word boundary)."""
    haystack = str(text or "").casefold()
    return sorted({
        str(term).casefold()
        for term in terms
        if str(term).strip() and re.search(r"\b" + re.escape(str(term).casefold()) + r"\w*\b", haystack)
    })


def _select_seeds(records: list[dict[str, Any]], terms: list[str], max_seeds: int) -> list[dict[str, Any]]:
    """Pick the most focus-relevant records that expose an OpenAlex work id."""
    eligible: list[tuple[int, str, dict[str, Any]]] = []
    for record in records:
        if not isinstance(record, dict) or not _work_id(record):
            continue
        matches = _focus_matches(f"{record.get('title', '')} {record.get('summary', '')}", terms)
        title_matches = _focus_matches(str(record.get("title") or ""), terms)
        eligible.append((len(matches) + len(title_matches),
                         str(record.get("title") or "").casefold(), record))
    eligible.sort(key=lambda entry: (-entry[0], entry[1]))
    return [entry[2] for entry in eligible[: max(0, int(max_seeds))]]


def _fetch_json(url: str, params: dict[str, Any], timeout: float, retries: int) -> dict[str, Any]:
    full_url = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    last_error: Exception | None = None
    for _ in range(max(0, int(retries)) + 1):
        try:
            request = urllib.request.Request(full_url, headers={"User-Agent": "SciTrace-Swarm/0.1"})
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - retry any transport failure
            last_error = exc
    raise _Skipped(f"OpenAlex request failed after retries: {last_error}")


def _abstract_from_inverted(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    words: dict[int, str] = {}
    for token, positions in index.items():
        if isinstance(token, str) and isinstance(positions, list):
            for position in positions:
                if isinstance(position, int):
                    words[position] = token
    return " ".join(words[position] for position in sorted(words))


def _record_from_work(work: Any) -> dict[str, Any] | None:
    """Map one OpenAlex work payload to a flat literature record."""
    if not isinstance(work, dict):
        return None
    authors: list[str] = []
    for authorship in work.get("authorships") or []:
        name = ((authorship or {}).get("author") or {}).get("display_name")
        if isinstance(name, str) and name.strip():
            authors.append(name.strip())
    return {
        "title": str(work.get("display_name") or ""),
        "authors": authors,
        "year": work.get("publication_year"),
        "doi": str(work.get("doi") or "").removeprefix("https://doi.org/").lower(),
        "source_url": str(((work.get("primary_location") or {}).get("landing_page_url")) or work.get("id") or ""),
        "summary": _abstract_from_inverted(work.get("abstract_inverted_index")),
        "openalex_id": _work_id(work),
        "related_works": [str(value) for value in (work.get("related_works") or []) if isinstance(value, str)],
    }


def _record_keys(record: dict[str, Any]) -> set[str]:
    """Identity keys (OpenAlex id, DOI, normalized title) for cross-dedup."""
    keys: set[str] = set()
    work_id = str(record.get("openalex_id") or "")
    if work_id:
        keys.add(f"openalex:{work_id}")
    doi = str(record.get("doi") or "").strip().casefold()
    if doi:
        keys.add(f"doi:{doi}")
    title = re.sub(r"\s+", " ", str(record.get("title") or "").strip().casefold())
    if title:
        keys.add(f"title:{title}")
    return keys


def _primary_key(record: dict[str, Any]) -> str:
    if record.get("openalex_id"):
        return f"openalex:{record['openalex_id']}"
    doi = str(record.get("doi") or "").strip().casefold()
    if doi:
        return f"doi:{doi}"
    return f"title:{re.sub(r'\s+', ' ', str(record.get('title') or '').strip().casefold())}"


def run_expansion(seed_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Expand the first-round records by one citation hop; never raises."""
    try:
        terms = [str(term) for term in config.get("focus_terms") or []]
        minimum = int(config.get("minimum_focus_terms", DEFAULTS["minimum_focus_terms"]) or 0)
        max_results = int(config.get("max_expansion_results", DEFAULTS["max_expansion_results"]) or 0)
        max_seeds = int(config.get("max_seeds", DEFAULTS["max_seeds"]) or 0)
        per_seed = int(config.get("per_seed_limit", DEFAULTS["per_seed_limit"]) or 0)
        related_limit = int(config.get("related_per_seed", DEFAULTS["related_per_seed"]) or 0)
        timeout = float(config.get("timeout_seconds", DEFAULTS["timeout_seconds"]) or 20.0)
        retries = int(config.get("retries", DEFAULTS["retries"]) or 0)

        seeds = _select_seeds(seed_records, terms, max_seeds)
        if not seeds:
            return {"status": "skipped",
                    "error": "no first-round record exposes an OpenAlex work id",
                    "seed_openalex_ids": [], "kept": [], "kept_provenance": {},
                    "api_call_count": 0}

        existing: set[str] = set()
        for record in seed_records:
            if isinstance(record, dict):
                existing |= _record_keys(record)
        candidates: list[tuple[dict[str, Any], str]] = []
        candidate_count = 0
        api_calls = 0
        for seed in seeds:
            seed_id = _work_id(seed)
            citing = _fetch_json(
                OPENALEX_WORKS_URL,
                {"filter": f"cites:{seed_id}", "per-page": max(1, per_seed)},
                timeout, retries,
            )
            api_calls += 1
            for raw in citing.get("results") or []:
                record = _record_from_work(raw)
                if record is not None:
                    candidate_count += 1
                    candidates.append((record, seed_id))
            for work_id in _related_ids(seed, related_limit):
                related = _fetch_json(f"{OPENALEX_WORKS_URL}/{work_id}", {}, timeout, retries)
                api_calls += 1
                record = _record_from_work(related)
                if record is not None:
                    candidate_count += 1
                    candidates.append((record, seed_id))

        filtered: list[tuple[int, dict[str, Any], str]] = []
        for record, seed_id in candidates:
            matches = _focus_matches(f"{record.get('title', '')} {record.get('summary', '')}", terms)
            if len(matches) < minimum:
                continue
            title_matches = _focus_matches(str(record.get("title") or ""), terms)
            filtered.append((len(matches) + len(title_matches), record, seed_id))
        filtered.sort(key=lambda entry: (-entry[0], entry[1].get("title", "").casefold()))

        kept: list[dict[str, Any]] = []
        provenance: dict[str, str] = {}
        for _, record, seed_id in filtered:
            if len(kept) >= max_results:
                break
            keys = _record_keys(record)
            if keys & existing:
                continue
            existing |= keys
            record["focus_matches"] = _focus_matches(
                f"{record.get('title', '')} {record.get('summary', '')}", terms)
            record["seed_openalex_id"] = seed_id
            kept.append(record)
            provenance[_primary_key(record)] = seed_id
        return {
            "status": "expanded",
            "seed_count": len(seeds),
            "seed_openalex_ids": [_work_id(seed) for seed in seeds],
            "expanded_candidate_count": candidate_count,
            "kept": kept,
            "kept_provenance": provenance,
            "api_call_count": api_calls,
        }
    except Exception as exc:  # noqa: BLE001 - expansion must never raise
        return {"status": "skipped", "error": f"{type(exc).__name__}: {exc}",
                "seed_openalex_ids": [], "kept": [], "kept_provenance": {}, "api_call_count": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-hop OpenAlex citation expansion for a first-round seed list.")
    parser.add_argument("--seeds", required=True, help="JSON file with the first-round literature records.")
    parser.add_argument("--config", required=True, help="JSON file with focus_terms and expansion limits.")
    parser.add_argument("--out", required=True, help="Output JSON path for the expansion report.")
    args = parser.parse_args(argv)

    try:
        # utf-8-sig tolerates the byte-order mark that Windows editors prepend.
        seeds_payload = json.loads(Path(args.seeds).read_text(encoding="utf-8-sig"))
        config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}", file=sys.stderr)
        return 2
    records = seeds_payload if isinstance(seeds_payload, list) else seeds_payload.get("results", [])
    if not isinstance(records, list):
        print("seeds JSON must be a list or an object with a `results` list", file=sys.stderr)
        return 2
    report = run_expansion([record for record in records if isinstance(record, dict)], config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status={report['status']} kept={len(report['kept'])} api_calls={report['api_call_count']} -> {out_path}")
    return 0 if report["status"] == "expanded" else 1


if __name__ == "__main__":
    sys.exit(main())

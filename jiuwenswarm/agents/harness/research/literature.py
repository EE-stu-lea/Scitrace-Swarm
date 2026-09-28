from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_doi(value: str) -> str:
    raw = str(value or "").strip()
    raw = raw.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    return raw.lower()


def _normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _dedupe_key(entry: dict[str, Any]) -> str:
    doi = _normalize_doi(str(entry.get("doi") or ""))
    if doi:
        return f"doi:{doi}"
    title = _normalize_title(str(entry.get("title") or ""))
    year = str(entry.get("year") or "").strip()
    digest = hashlib.sha256(f"{title}|{year}".encode("utf-8")).hexdigest()[:20]
    return f"title:{digest}"


@dataclass(frozen=True)
class LiteratureSearchResult:
    title: str
    authors: tuple[str, ...]
    year: int | None
    doi: str
    source_url: str
    summary: str
    reliability_score: float | None
    provider: str
    raw_payload: dict[str, Any]
    verified_doi: bool = False
    verified_url: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "doi": self.doi,
            "source_url": self.source_url,
            "summary": self.summary,
            "reliability_score": self.reliability_score,
            "provider": self.provider,
            "raw_payload": self.raw_payload,
            "verified_doi": self.verified_doi,
            "verified_url": self.verified_url,
        }


class LiteratureProviderError(RuntimeError):
    pass


class LiteratureProvider:
    def search(self, topic: str) -> list[LiteratureSearchResult]:
        raise NotImplementedError


class FixtureLiteratureProvider(LiteratureProvider):
    def __init__(self, fixture_path: str | Path, *, cache_root: str | Path | None = None) -> None:
        self.fixture_path = Path(fixture_path)
        self.cache_root = Path(cache_root) if cache_root else None

    def search(self, topic: str) -> list[LiteratureSearchResult]:
        _ = topic
        payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("results", [])
        return [_coerce_result(record, provider="fixture") for record in records if isinstance(record, dict)]


class RequestsLiteratureProvider(LiteratureProvider):
    provider_name = "requests"

    def __init__(
        self,
        *,
        cache_root: str | Path,
        max_results: int = 5,
        timeout_seconds: float = 20.0,
        retries: int = 2,
        verify_links: bool = True,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.max_results = max(1, int(max_results))
        self.timeout_seconds = float(timeout_seconds)
        self.retries = max(0, int(retries))
        self.verify_links = bool(verify_links)

    def search(self, topic: str) -> list[LiteratureSearchResult]:
        topic = str(topic or "").strip()
        if not topic:
            return []
        payload = self._request(topic)
        self._cache_raw_response(topic, payload)
        items = self._parse_items(payload)
        return self._normalize_and_filter(items)

    def _request(self, topic: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return _http_get_json(
                    self._url(topic),
                    params=self._params(topic),
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
        raise LiteratureProviderError(f"{self.provider_name} search failed: {last_error}") from last_error

    def _cache_raw_response(self, topic: str, payload: dict[str, Any]) -> None:
        digest = hashlib.sha256(topic.encode("utf-8")).hexdigest()[:16]
        path = self.cache_root / f"{self.provider_name}_{digest}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _normalize_and_filter(self, items: list[dict[str, Any]]) -> list[LiteratureSearchResult]:
        deduped: dict[str, LiteratureSearchResult] = {}
        for item in items:
            result = _coerce_result(item, provider=self.provider_name)
            verified_doi, verified_url = self._verify_links(result.doi, result.source_url)
            if not verified_doi and not verified_url:
                continue
            result = LiteratureSearchResult(
                title=result.title,
                authors=result.authors,
                year=result.year,
                doi=result.doi,
                source_url=result.source_url,
                summary=result.summary,
                reliability_score=result.reliability_score,
                provider=result.provider,
                raw_payload=result.raw_payload,
                verified_doi=verified_doi,
                verified_url=verified_url,
            )
            key = _dedupe_key(result.to_dict())
            existing = deduped.get(key)
            if existing is None or _score_result(result) > _score_result(existing):
                deduped[key] = result
            if len(deduped) >= self.max_results:
                break
        return list(deduped.values())

    def _verify_links(self, doi: str, url: str) -> tuple[bool, bool]:
        if not self.verify_links:
            return bool(_normalize_doi(doi)), bool(str(url or "").strip())
        verified_doi = False
        normalized_doi = _normalize_doi(doi)
        if normalized_doi:
            verified_doi = self._can_fetch(f"https://doi.org/{normalized_doi}")
        verified_url = False
        normalized_url = str(url or "").strip()
        if normalized_url:
            verified_url = self._can_fetch(normalized_url)
        return verified_doi, verified_url

    def _can_fetch(self, url: str) -> bool:
        return _http_can_fetch(url, timeout=min(self.timeout_seconds, 10.0))

    def _url(self, topic: str) -> str:
        raise NotImplementedError

    def _params(self, topic: str) -> dict[str, Any]:
        raise NotImplementedError

    def _parse_items(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError


class OpenAlexLiteratureProvider(RequestsLiteratureProvider):
    provider_name = "openalex"

    def _url(self, topic: str) -> str:
        _ = topic
        return "https://api.openalex.org/works"

    def _params(self, topic: str) -> dict[str, Any]:
        return {
            "search": topic,
            "per-page": self.max_results * 2,
            "sort": "relevance_score:desc",
        }

    def _parse_items(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        results = payload.get("results")
        if not isinstance(results, list):
            return []
        items: list[dict[str, Any]] = []
        for raw in results:
            if not isinstance(raw, dict):
                continue
            authorships = raw.get("authorships") or []
            authors = []
            if isinstance(authorships, list):
                for author_item in authorships:
                    author = ((author_item or {}).get("author") or {}).get("display_name")
                    if isinstance(author, str) and author.strip():
                        authors.append(author.strip())
            items.append(
                {
                    "title": raw.get("display_name") or "",
                    "authors": authors,
                    "year": raw.get("publication_year"),
                    "doi": _normalize_doi(str(raw.get("doi") or "")),
                    "source_url": str(raw.get("primary_location", {}).get("landing_page_url") or raw.get("id") or ""),
                    "summary": _openalex_abstract(raw.get("abstract_inverted_index")),
                    "reliability_score": raw.get("relevance_score"),
                    "raw_payload": raw,
                }
            )
        return items


class CrossrefLiteratureProvider(RequestsLiteratureProvider):
    provider_name = "crossref"

    def _url(self, topic: str) -> str:
        _ = topic
        return "https://api.crossref.org/works"

    def _params(self, topic: str) -> dict[str, Any]:
        return {
            "query": topic,
            "rows": self.max_results * 2,
            "sort": "relevance",
        }

    def _parse_items(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        items_raw = (payload.get("message") or {}).get("items")
        if not isinstance(items_raw, list):
            return []
        items: list[dict[str, Any]] = []
        for raw in items_raw:
            if not isinstance(raw, dict):
                continue
            author_list = raw.get("author") or []
            authors = []
            if isinstance(author_list, list):
                for author in author_list:
                    given = str((author or {}).get("given") or "").strip()
                    family = str((author or {}).get("family") or "").strip()
                    name = " ".join(part for part in (given, family) if part)
                    if name:
                        authors.append(name)
            created = raw.get("published-print") or raw.get("published-online") or {}
            year = None
            parts = created.get("date-parts")
            if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
                try:
                    year = int(parts[0][0])
                except (TypeError, ValueError):
                    year = None
            items.append(
                {
                    "title": ((raw.get("title") or [""])[0]) if isinstance(raw.get("title"), list) else raw.get("title"),
                    "authors": authors,
                    "year": year,
                    "doi": _normalize_doi(str(raw.get("DOI") or "")),
                    "source_url": str((raw.get("URL") or "")),
                    "summary": _strip_tags(str(raw.get("abstract") or "")),
                    "reliability_score": None,
                    "raw_payload": raw,
                }
            )
        return items


class ResilientLiteratureProvider(LiteratureProvider):
    def __init__(
        self,
        *,
        providers: list[LiteratureProvider],
        fallback: LiteratureProvider | None = None,
    ) -> None:
        self.providers = providers
        self.fallback = fallback

    def search(self, topic: str) -> list[LiteratureSearchResult]:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                results = provider.search(topic)
                if results:
                    return results
            except Exception as exc:
                last_error = exc
                continue
        if self.fallback is not None:
            return self.fallback.search(topic)
        raise LiteratureProviderError(f"literature search failed: {last_error}") from last_error


def _coerce_result(raw: dict[str, Any], *, provider: str) -> LiteratureSearchResult:
    title = str(raw.get("title") or "").strip()
    authors = tuple(str(item).strip() for item in raw.get("authors", []) if str(item).strip())
    year = raw.get("year")
    try:
        year = int(year) if year not in ("", None) else None
    except (TypeError, ValueError):
        year = None
    return LiteratureSearchResult(
        title=title,
        authors=authors,
        year=year,
        doi=_normalize_doi(str(raw.get("doi") or "")),
        source_url=str(raw.get("source_url") or "").strip(),
        summary=str(raw.get("summary") or "").strip(),
        reliability_score=(
            float(raw.get("reliability_score"))
            if isinstance(raw.get("reliability_score"), (int, float))
            else None
        ),
        provider=provider,
        raw_payload=dict(raw.get("raw_payload") or raw),
        verified_doi=bool(raw.get("verified_doi", False)),
        verified_url=bool(raw.get("verified_url", False)),
    )


def _score_result(result: LiteratureSearchResult) -> float:
    return float(result.reliability_score or 0.0) + (0.5 if result.verified_doi else 0.0) + (
        0.25 if result.verified_url else 0.0
    )


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value or "").strip()


def _openalex_abstract(inverted_index: Any) -> str:
    if not isinstance(inverted_index, dict):
        return ""
    words: dict[int, str] = {}
    for token, positions in inverted_index.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                words[position] = token
    return " ".join(words[index] for index in sorted(words))


def _http_get_json(url: str, *, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    if requests is not None:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    full_url = url
    if params:
        full_url = f"{url}?{urlencode(params)}"
    request = Request(full_url, headers={"User-Agent": "SciTrace-Swarm/0.1"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _http_can_fetch(url: str, *, timeout: float) -> bool:
    if requests is not None:
        try:
            response = requests.get(url, timeout=timeout, allow_redirects=True)
            return 200 <= response.status_code < 400
        except Exception:
            return False
    try:
        request = Request(url, headers={"User-Agent": "SciTrace-Swarm/0.1"})
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            status = getattr(response, "status", 200)
            return 200 <= int(status) < 400
    except (HTTPError, URLError, ValueError):
        return False


__all__ = [
    "CrossrefLiteratureProvider",
    "FixtureLiteratureProvider",
    "LiteratureProvider",
    "LiteratureProviderError",
    "LiteratureSearchResult",
    "OpenAlexLiteratureProvider",
    "ResilientLiteratureProvider",
]

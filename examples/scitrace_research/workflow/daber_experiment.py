"""Disagreement-aware budgeted retrieval on BEIR-format scientific datasets.

The experiment is deliberately self-contained so it can be copied into the
restricted SciTrace-Swarm experiment workspace. It implements three lexical
retrievers (BM25, word TF-IDF, and character 3-gram TF-IDF), reciprocal-rank
fusion, fixed and adaptive evidence budgets, paired bootstrap confidence
intervals, and machine-readable per-query outputs using only the Python
standard library.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time
from typing import Callable, Iterable


WORD_RE = re.compile(r"[a-z0-9]+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(payload)
    return rows


def load_beir(dataset_dir: Path, split: str) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"
    for required in (corpus_path, queries_path, qrels_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing BEIR dataset file: {required}")

    corpus = {}
    for row in _read_jsonl(corpus_path):
        doc_id = str(row.get("_id") or row.get("id") or "").strip()
        text = " ".join(part for part in (str(row.get("title") or ""), str(row.get("text") or "")) if part).strip()
        if doc_id and text:
            corpus[doc_id] = text
    queries = {}
    for row in _read_jsonl(queries_path):
        query_id = str(row.get("_id") or row.get("id") or "").strip()
        text = str(row.get("text") or "").strip()
        if query_id and text:
            queries[query_id] = text

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with qrels_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader, None)
        if not header:
            raise ValueError(f"empty qrels file: {qrels_path}")
        for row in reader:
            if len(row) < 3:
                continue
            query_id, doc_id, score_text = row[:3]
            try:
                score = int(float(score_text))
            except ValueError:
                continue
            if score > 0 and query_id in queries and doc_id in corpus:
                qrels[query_id][doc_id] = score
    queries = {query_id: text for query_id, text in queries.items() if qrels.get(query_id)}
    if not corpus or not queries:
        raise ValueError("BEIR dataset contains no usable corpus/query/qrel records")
    return corpus, queries, dict(qrels)


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def char_ngrams(text: str, n: int = 3) -> list[str]:
    normalized = " ".join(word_tokens(text))
    if len(normalized) < n:
        return [normalized] if normalized else []
    return [normalized[index : index + n] for index in range(len(normalized) - n + 1)]


class BM25Retriever:
    def __init__(self, corpus: dict[str, str], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus)
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.doc_lengths = {}
        document_frequency = Counter()
        for doc_id, text in corpus.items():
            counts = Counter(word_tokens(text))
            self.doc_lengths[doc_id] = sum(counts.values())
            for term, frequency in counts.items():
                self.postings[term].append((doc_id, frequency))
                document_frequency[term] += 1
        self.average_length = statistics.fmean(self.doc_lengths.values()) if self.doc_lengths else 1.0
        self.idf = {
            term: math.log(1.0 + (self.doc_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def rank(self, query: str, limit: int) -> list[tuple[str, float]]:
        scores: dict[str, float] = defaultdict(float)
        for term, query_frequency in Counter(word_tokens(query)).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, term_frequency in self.postings.get(term, []):
                length = self.doc_lengths[doc_id]
                denominator = term_frequency + self.k1 * (1.0 - self.b + self.b * length / self.average_length)
                scores[doc_id] += query_frequency * idf * (term_frequency * (self.k1 + 1.0) / denominator)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


class TfidfRetriever:
    def __init__(self, corpus: dict[str, str], tokenizer: Callable[[str], list[str]]) -> None:
        self.tokenizer = tokenizer
        self.doc_count = len(corpus)
        raw_counts: dict[str, Counter] = {}
        document_frequency = Counter()
        for doc_id, text in corpus.items():
            counts = Counter(tokenizer(text))
            raw_counts[doc_id] = counts
            document_frequency.update(counts.keys())
        self.idf = {
            term: math.log((self.doc_count + 1.0) / (frequency + 1.0)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self.postings: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self.doc_norms = {}
        for doc_id, counts in raw_counts.items():
            weighted = {
                term: (1.0 + math.log(frequency)) * self.idf[term]
                for term, frequency in counts.items()
            }
            self.doc_norms[doc_id] = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
            for term, value in weighted.items():
                self.postings[term].append((doc_id, value))

    def rank(self, query: str, limit: int) -> list[tuple[str, float]]:
        counts = Counter(self.tokenizer(query))
        query_weights = {
            term: (1.0 + math.log(frequency)) * self.idf[term]
            for term, frequency in counts.items()
            if term in self.idf
        }
        query_norm = math.sqrt(sum(value * value for value in query_weights.values()))
        if query_norm == 0.0:
            return []
        dot_products: dict[str, float] = defaultdict(float)
        for term, query_weight in query_weights.items():
            for doc_id, document_weight in self.postings.get(term, []):
                dot_products[doc_id] += query_weight * document_weight
        scores = {
            doc_id: dot / (query_norm * self.doc_norms[doc_id])
            for doc_id, dot in dot_products.items()
        }
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def reciprocal_rank_fusion(rankings: Iterable[list[tuple[str, float]]], *, constant: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def ranking_disagreement(rankings: list[list[tuple[str, float]]], *, depth: int = 10) -> float:
    sets = [set(doc_id for doc_id, _score in ranking[:depth]) for ranking in rankings]
    similarities = []
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            union = sets[left] | sets[right]
            similarities.append(len(sets[left] & sets[right]) / len(union) if union else 1.0)
    return 1.0 - statistics.fmean(similarities) if similarities else 0.0


def score_margin_uncertainty(ranking: list[tuple[str, float]]) -> float:
    if len(ranking) < 2:
        return 1.0
    first = abs(ranking[0][1])
    gap = ranking[0][1] - ranking[1][1]
    return 1.0 - max(0.0, min(1.0, gap / first)) if first else 1.0


def allocate_budget(weights: dict[str, float], *, average_budget: int, maximum_budget: int = 10) -> dict[str, int]:
    query_ids = sorted(weights)
    if not query_ids:
        return {}
    average_budget = max(1, min(int(average_budget), maximum_budget))
    allocation = {query_id: 1 for query_id in query_ids}
    remaining = average_budget * len(query_ids) - len(query_ids)
    adjusted = {query_id: max(1e-6, float(weights[query_id])) for query_id in query_ids}
    while remaining > 0:
        eligible = [query_id for query_id in query_ids if allocation[query_id] < maximum_budget]
        if not eligible:
            break
        selected = max(
            eligible,
            key=lambda query_id: (adjusted[query_id] / allocation[query_id], query_id),
        )
        allocation[selected] += 1
        remaining -= 1
    return allocation


def fixed_budget(query_ids: Iterable[str], budget: int) -> dict[str, int]:
    return {query_id: int(budget) for query_id in query_ids}


def _rankdata(values: list[float]) -> list[float]:
    """Return average ranks for tied values, using one-based ranks."""
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def spearman_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Spearman inputs must have the same non-zero length")
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    denominator = left_scale * right_scale
    return numerator / denominator if denominator else 0.0


def bootstrap_correlation(
    left: list[float],
    right: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    observed = spearman_correlation(left, right)
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        indices = [generator.randrange(len(left)) for _ in left]
        estimates.append(
            spearman_correlation(
                [left[index] for index in indices],
                [right[index] for index in indices],
            )
        )
    estimates.sort()
    return {
        "spearman_rho": observed,
        "ci95_low": estimates[max(0, int(0.025 * samples) - 1)],
        "ci95_high": estimates[min(samples - 1, int(0.975 * samples))],
        "samples": samples,
    }


def oracle_budget_allocation(
    rankings: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    *,
    average_budget: int,
    maximum_budget: int = 10,
) -> dict[str, int]:
    """Find the recall-maximizing per-query allocation by dynamic programming.

    This is an analysis-only upper bound because it uses test relevance labels.
    Every query receives at least one document and the total budget exactly
    matches the fixed baseline whenever the configured maximum permits it.
    """
    values_by_query = {}
    for query_id in sorted(qrels):
        relevant = qrels[query_id]
        ranked_ids = [doc_id for doc_id, _score in rankings[query_id][:maximum_budget]]
        values = [0.0] * (maximum_budget + 1)
        matched = 0
        for budget in range(1, maximum_budget + 1):
            if budget <= len(ranked_ids) and ranked_ids[budget - 1] in relevant:
                matched += 1
            values[budget] = matched / len(relevant)
        values_by_query[query_id] = values
    return optimal_allocation_from_values(
        values_by_query,
        average_budget=average_budget,
        maximum_budget=maximum_budget,
    )


def optimal_allocation_from_values(
    values_by_query: dict[str, list[float]],
    *,
    average_budget: int,
    maximum_budget: int = 10,
) -> dict[str, int]:
    """Exact DP with vectorized state updates and compact backtracking.

    Query order, ascending depth order and strict-improvement tie breaking
    match the scalar reference. NumPy is optional: the standard-library
    experiment remains usable without it. Complexity remains O(n B K), but
    the total-budget loop is executed in native array operations.
    """
    for name, value in (("average_budget", average_budget), ("maximum_budget", maximum_budget)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not values_by_query:
        return {}
    query_ids = sorted(values_by_query)
    for query_id in query_ids:
        values = values_by_query[query_id]
        if len(values) <= maximum_budget or not all(math.isfinite(x) for x in values[:maximum_budget + 1]):
            raise ValueError("each recall curve must provide finite values at depths 0..maximum_budget")
    try:
        import numpy as np
    except ImportError:
        return _optimal_allocation_scalar(values_by_query, average_budget=average_budget,
                                          maximum_budget=maximum_budget)
    target = min(average_budget, maximum_budget) * len(query_ids)
    previous = np.full(target + 1, -np.inf, dtype=np.float64)
    previous[0] = 0.0
    # Only depth decisions are retained; value states use two rolling rows.
    choices = np.zeros((len(query_ids), target + 1), dtype=np.min_scalar_type(maximum_budget))
    for query_index, query_id in enumerate(query_ids, start=1):
        current = np.full(target + 1, -np.inf, dtype=np.float64)
        maximum_total = min(target, query_index * maximum_budget)
        for depth in range(1, min(maximum_budget, maximum_total) + 1):
            start = max(query_index, depth)
            if start > maximum_total:
                continue
            candidates = previous[start-depth:maximum_total+1-depth] + values_by_query[query_id][depth]
            view = current[start:maximum_total+1]
            improved = candidates > view
            np.copyto(view, candidates, where=improved)
            decisions = choices[query_index-1, start:maximum_total+1]
            decisions[improved] = depth
        previous = current
    if not np.isfinite(previous[target]):
        raise ValueError("allocation could not satisfy the requested budget")
    allocation = {}
    remaining = target
    for query_index in range(len(query_ids)-1, -1, -1):
        depth = int(choices[query_index, remaining])
        allocation[query_ids[query_index]] = depth
        remaining -= depth
    return allocation


def _optimal_allocation_scalar(
    values_by_query: dict[str, list[float]],
    *,
    average_budget: int,
    maximum_budget: int = 10,
) -> dict[str, int]:
    """Maximize summed query values for an exact global document budget."""
    query_ids = sorted(values_by_query)
    target = min(average_budget * len(query_ids), maximum_budget * len(query_ids))
    negative_infinity = float("-inf")
    previous = [negative_infinity] * (target + 1)
    previous[0] = 0.0
    choices: list[list[int]] = []
    for query_index, query_id in enumerate(query_ids, start=1):
        values = values_by_query[query_id]
        current = [negative_infinity] * (target + 1)
        row = [0] * (target + 1)
        minimum_total = query_index
        maximum_total = min(target, query_index * maximum_budget)
        for total in range(minimum_total, maximum_total + 1):
            best_value = negative_infinity
            best_budget = 0
            for budget in range(1, min(maximum_budget, total) + 1):
                prior = previous[total - budget]
                if prior == negative_infinity:
                    continue
                candidate = prior + values[budget]
                if candidate > best_value:
                    best_value = candidate
                    best_budget = budget
            current[total] = best_value
            row[total] = best_budget
        previous = current
        choices.append(row)
    if previous[target] == negative_infinity:
        raise ValueError("oracle allocation could not satisfy the requested budget")
    allocation = {}
    remaining = target
    for query_index in range(len(query_ids) - 1, -1, -1):
        budget = choices[query_index][remaining]
        allocation[query_ids[query_index]] = budget
        remaining -= budget
    return allocation


def fit_disagreement_calibrator(
    disagreement: dict[str, float],
    rankings: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    *,
    maximum_budget: int = 10,
    bins: int = 5,
) -> dict:
    """Estimate monotone mean recall curves in disagreement quantile bins."""
    ordered = sorted(qrels, key=lambda query_id: (disagreement[query_id], query_id))
    bin_count = min(bins, len(ordered))
    groups = []
    for bin_index in range(bin_count):
        start = bin_index * len(ordered) // bin_count
        end = (bin_index + 1) * len(ordered) // bin_count
        groups.append(ordered[start:end])
    boundaries = [
        (disagreement[groups[index][-1]] + disagreement[groups[index + 1][0]]) / 2.0
        for index in range(bin_count - 1)
    ]
    curves = []
    for members in groups:
        cumulative = [0.0] * (maximum_budget + 1)
        for query_id in members:
            relevant = qrels[query_id]
            ranked_ids = [doc_id for doc_id, _score in rankings[query_id][:maximum_budget]]
            matched = 0
            for budget in range(1, maximum_budget + 1):
                if budget <= len(ranked_ids) and ranked_ids[budget - 1] in relevant:
                    matched += 1
                cumulative[budget] += matched / len(relevant)
        curves.append([value / len(members) for value in cumulative])
    return {
        "boundaries": boundaries,
        "recall_curves": curves,
        "queries_per_bin": [len(group) for group in groups],
        "maximum_budget": maximum_budget,
    }


def calibrated_disagreement_allocation(
    disagreement: dict[str, float],
    calibrator: dict,
    *,
    average_budget: int,
) -> dict[str, int]:
    maximum_budget = int(calibrator["maximum_budget"])
    values_by_query = {}
    for query_id, value in disagreement.items():
        bin_index = bisect_right(calibrator["boundaries"], value)
        values_by_query[query_id] = calibrator["recall_curves"][bin_index]
    return optimal_allocation_from_values(
        values_by_query,
        average_budget=average_budget,
        maximum_budget=maximum_budget,
    )


def disagreement_diagnostics(
    disagreement: dict[str, float],
    rankings: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    *,
    depth: int,
    bootstrap_samples: int,
    seed: int,
    bins: int = 5,
) -> dict:
    query_ids = sorted(qrels)
    first_relevant_rank = {}
    failure_at_depth = {}
    for query_id in query_ids:
        relevant = qrels[query_id]
        rank = next(
            (
                index
                for index, (doc_id, _score) in enumerate(rankings[query_id], start=1)
                if doc_id in relevant
            ),
            len(rankings[query_id]) + 1,
        )
        first_relevant_rank[query_id] = rank
        failure_at_depth[query_id] = int(rank > depth)
    disagreement_values = [disagreement[query_id] for query_id in query_ids]
    rank_values = [float(first_relevant_rank[query_id]) for query_id in query_ids]
    failure_values = [float(failure_at_depth[query_id]) for query_id in query_ids]
    ordered = sorted(query_ids, key=lambda query_id: (disagreement[query_id], query_id))
    calibration = []
    bin_count = min(bins, len(ordered))
    for bin_index in range(bin_count):
        start = bin_index * len(ordered) // bin_count
        end = (bin_index + 1) * len(ordered) // bin_count
        members = ordered[start:end]
        calibration.append(
            {
                "bin": bin_index + 1,
                "queries": len(members),
                "mean_disagreement": statistics.fmean(disagreement[query_id] for query_id in members),
                "mean_first_relevant_rank": statistics.fmean(first_relevant_rank[query_id] for query_id in members),
                f"failure_rate_at_{depth}": statistics.fmean(failure_at_depth[query_id] for query_id in members),
            }
        )
    return {
        "difficulty_definition": "first relevant rank in the fused RRF ranking; larger means harder",
        "failure_definition": f"no relevant document in the first {depth} fused results",
        "disagreement_vs_first_relevant_rank": bootstrap_correlation(
            disagreement_values,
            rank_values,
            samples=bootstrap_samples,
            seed=seed,
        ),
        f"disagreement_vs_failure_at_{depth}": bootstrap_correlation(
            disagreement_values,
            failure_values,
            samples=bootstrap_samples,
            seed=seed + 1,
        ),
        "calibration_bins_low_to_high_disagreement": calibration,
    }


def _dcg(relevances: list[int]) -> float:
    return sum((2**relevance - 1.0) / math.log2(rank + 1.0) for rank, relevance in enumerate(relevances, start=1))


def evaluate_rankings(
    rankings: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    allocations: dict[str, int],
) -> tuple[dict[str, float], dict[str, dict[str, float | int | list[str]]]]:
    per_query = {}
    for query_id in sorted(qrels):
        budget = allocations[query_id]
        retrieved = [doc_id for doc_id, _score in rankings[query_id][:budget]]
        relevant = qrels[query_id]
        matched = [doc_id for doc_id in retrieved if doc_id in relevant]
        recall = len(matched) / len(relevant)
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                reciprocal_rank = 1.0 / rank
                break
        observed = [relevant.get(doc_id, 0) for doc_id in retrieved]
        ideal = sorted(relevant.values(), reverse=True)[:budget]
        ideal_dcg = _dcg(ideal)
        ndcg = _dcg(observed) / ideal_dcg if ideal_dcg else 0.0
        per_query[query_id] = {
            "budget": budget,
            "recall": recall,
            "hit": int(bool(matched)),
            "reciprocal_rank": reciprocal_rank,
            "ndcg": ndcg,
            "retrieved": retrieved,
            "matched": matched,
        }
    metrics = {
        "recall": statistics.fmean(item["recall"] for item in per_query.values()),
        "hit_rate": statistics.fmean(item["hit"] for item in per_query.values()),
        "mrr": statistics.fmean(item["reciprocal_rank"] for item in per_query.values()),
        "ndcg": statistics.fmean(item["ndcg"] for item in per_query.values()),
        "mean_budget": statistics.fmean(item["budget"] for item in per_query.values()),
        "total_documents": sum(int(item["budget"]) for item in per_query.values()),
    }
    return metrics, per_query


def paired_bootstrap(
    candidate: dict[str, dict],
    baseline: dict[str, dict],
    *,
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    query_ids = sorted(set(candidate) & set(baseline))
    differences = [float(candidate[query_id][metric]) - float(baseline[query_id][metric]) for query_id in query_ids]
    observed = statistics.fmean(differences)
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(statistics.fmean(generator.choice(differences) for _ in differences))
    estimates.sort()
    low = estimates[max(0, int(0.025 * samples) - 1)]
    high = estimates[min(samples - 1, int(0.975 * samples))]
    probability_positive = sum(value > 0 for value in estimates) / samples
    return {
        "observed_difference": observed,
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_probability_positive": probability_positive,
        "samples": samples,
    }


def run_experiment(
    *,
    dataset_dir: Path,
    split: str,
    budgets: list[int],
    seed: int,
    bootstrap_samples: int,
    output_path: Path,
) -> dict:
    corpus, queries, qrels = load_beir(dataset_dir, split)
    maximum_rank = max(100, max(budgets) * 5)
    started = time.perf_counter()
    retrievers = {
        "bm25": BM25Retriever(corpus),
        "word_tfidf": TfidfRetriever(corpus, word_tokens),
        "char_3gram_tfidf": TfidfRetriever(corpus, char_ngrams),
    }
    index_seconds = time.perf_counter() - started

    rankings_by_method: dict[str, dict[str, list[tuple[str, float]]]] = {
        name: {} for name in retrievers
    }
    fused_rankings = {}
    fused_without_char = {}
    disagreement = {}
    disagreement_without_char = {}
    margin_uncertainty = {}
    query_started = time.perf_counter()
    for query_id, query in queries.items():
        component_rankings = []
        for name, retriever in retrievers.items():
            ranking = retriever.rank(query, maximum_rank)
            rankings_by_method[name][query_id] = ranking
            component_rankings.append(ranking)
        fused = reciprocal_rank_fusion(component_rankings)
        fused_rankings[query_id] = fused
        no_char_components = [rankings_by_method["bm25"][query_id], rankings_by_method["word_tfidf"][query_id]]
        fused_without_char[query_id] = reciprocal_rank_fusion(no_char_components)
        disagreement[query_id] = ranking_disagreement(component_rankings)
        disagreement_without_char[query_id] = ranking_disagreement(no_char_components)
        margin_uncertainty[query_id] = score_margin_uncertainty(fused)
    query_seconds = time.perf_counter() - query_started

    random_generator = random.Random(seed)
    random_weights = {query_id: random_generator.random() for query_id in queries}
    calibrator = None
    calibration_seconds = 0.0
    calibration_split = "train"
    calibration_qrels_path = dataset_dir / "qrels" / f"{calibration_split}.tsv"
    if split != calibration_split and calibration_qrels_path.is_file():
        calibration_started = time.perf_counter()
        _calibration_corpus, calibration_queries, calibration_qrels = load_beir(
            dataset_dir,
            calibration_split,
        )
        calibration_rankings = {}
        calibration_disagreement = {}
        calibration_limit = max(10, max(budgets))
        for query_id, query in calibration_queries.items():
            components = [
                retriever.rank(query, calibration_limit)
                for retriever in retrievers.values()
            ]
            calibration_rankings[query_id] = reciprocal_rank_fusion(components)
            calibration_disagreement[query_id] = ranking_disagreement(components)
        calibrator = fit_disagreement_calibrator(
            calibration_disagreement,
            calibration_rankings,
            calibration_qrels,
        )
        calibrator["split"] = calibration_split
        calibrator["queries"] = len(calibration_queries)
        calibrator["qrels_sha256"] = _sha256(calibration_qrels_path)
        calibration_seconds = time.perf_counter() - calibration_started
    diagnostics = disagreement_diagnostics(
        disagreement,
        fused_rankings,
        qrels,
        depth=5,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 9000,
    )
    methods_by_budget = {}
    bootstrap_by_budget = {}
    calibrated_bootstrap_by_budget = {}
    per_query_export = []
    for budget in budgets:
        allocations = {
            "bm25_fixed": fixed_budget(queries, budget),
            "word_tfidf_fixed": fixed_budget(queries, budget),
            "char_3gram_fixed": fixed_budget(queries, budget),
            "rrf_fixed": fixed_budget(queries, budget),
            "random_allocation": allocate_budget(random_weights, average_budget=budget),
            "margin_allocation": allocate_budget(margin_uncertainty, average_budget=budget),
            "daber": allocate_budget(disagreement, average_budget=budget),
            "daber_without_char": allocate_budget(disagreement_without_char, average_budget=budget),
            "oracle_allocation": oracle_budget_allocation(
                fused_rankings,
                qrels,
                average_budget=budget,
            ),
        }
        if calibrator is not None:
            allocations["calibrated_disagreement"] = calibrated_disagreement_allocation(
                disagreement,
                calibrator,
                average_budget=budget,
            )
        method_rankings = {
            "bm25_fixed": rankings_by_method["bm25"],
            "word_tfidf_fixed": rankings_by_method["word_tfidf"],
            "char_3gram_fixed": rankings_by_method["char_3gram_tfidf"],
            "rrf_fixed": fused_rankings,
            "random_allocation": fused_rankings,
            "margin_allocation": fused_rankings,
            "daber": fused_rankings,
            "daber_without_char": fused_without_char,
            "oracle_allocation": fused_rankings,
        }
        if calibrator is not None:
            method_rankings["calibrated_disagreement"] = fused_rankings
        budget_results = {}
        per_query_results = {}
        for method_name, rankings in method_rankings.items():
            metrics, per_query = evaluate_rankings(rankings, qrels, allocations[method_name])
            budget_results[method_name] = {key: round(value, 6) for key, value in metrics.items()}
            per_query_results[method_name] = per_query
        methods_by_budget[str(budget)] = budget_results
        bootstrap_by_budget[str(budget)] = {
            metric: paired_bootstrap(
                per_query_results["daber"],
                per_query_results["rrf_fixed"],
                metric=metric,
                samples=bootstrap_samples,
                seed=seed + budget * 100 + index,
            )
            for index, metric in enumerate(("recall", "hit", "reciprocal_rank", "ndcg"))
        }
        if calibrator is not None:
            calibrated_bootstrap_by_budget[str(budget)] = {
                metric: paired_bootstrap(
                    per_query_results["calibrated_disagreement"],
                    per_query_results["rrf_fixed"],
                    metric=metric,
                    samples=bootstrap_samples,
                    seed=seed + budget * 200 + index,
                )
                for index, metric in enumerate(("recall", "hit", "reciprocal_rank", "ndcg"))
            }
        for query_id in sorted(queries):
            per_query_export.append(
                {
                    "query_id": query_id,
                    "budget_target": budget,
                    "disagreement": round(disagreement[query_id], 6),
                    "fixed_budget": allocations["rrf_fixed"][query_id],
                    "daber_budget": allocations["daber"][query_id],
                    "oracle_budget": allocations["oracle_allocation"][query_id],
                    "calibrated_budget": (
                        allocations["calibrated_disagreement"][query_id]
                        if calibrator is not None
                        else ""
                    ),
                    "fixed_recall": round(float(per_query_results["rrf_fixed"][query_id]["recall"]), 6),
                    "daber_recall": round(float(per_query_results["daber"][query_id]["recall"]), 6),
                    "fixed_hit": int(per_query_results["rrf_fixed"][query_id]["hit"]),
                    "daber_hit": int(per_query_results["daber"][query_id]["hit"]),
                }
            )

    primary_budget = str(5 if 5 in budgets else budgets[-1])
    primary = methods_by_budget[primary_budget]
    candidate_name = "calibrated_disagreement" if calibrator is not None else "daber"
    candidate_bootstrap = (
        calibrated_bootstrap_by_budget[primary_budget]["recall"]
        if calibrator is not None
        else bootstrap_by_budget[primary_budget]["recall"]
    )
    payload = {
        "experiment_id": "daber_scientific_evidence_retrieval",
        "dataset": dataset_dir.name,
        "split": split,
        "metric_name": "evidence_recall",
        "metric_value": primary[candidate_name]["recall"],
        "primary_budget": int(primary_budget),
        "primary_comparison": {
            "candidate": candidate_name,
            "baseline": "rrf_fixed",
            "candidate_recall": primary[candidate_name]["recall"],
            "baseline_recall": primary["rrf_fixed"]["recall"],
            "absolute_difference": round(primary[candidate_name]["recall"] - primary["rrf_fixed"]["recall"], 6),
            "paired_bootstrap": candidate_bootstrap,
        },
        "dataset_statistics": {
            "documents": len(corpus),
            "queries": len(queries),
            "positive_qrels": sum(len(items) for items in qrels.values()),
        },
        "config": {
            "budgets": budgets,
            "maximum_query_budget": 10,
            "retrievers": ["bm25", "word_tfidf", "char_3gram_tfidf"],
            "fusion": "reciprocal_rank_fusion_k60",
            "disagreement": "mean_pairwise_top10_jaccard_distance",
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
        },
        "methods_by_budget": methods_by_budget,
        "paired_bootstrap_by_budget": bootstrap_by_budget,
        "calibrated_paired_bootstrap_by_budget": calibrated_bootstrap_by_budget,
        "calibration": calibrator,
        "disagreement_diagnostics": diagnostics,
        "runtime": {
            "index_seconds": round(index_seconds, 6),
            "all_queries_seconds": round(query_seconds, 6),
            "calibration_seconds": round(calibration_seconds, 6),
        },
        "data_files": {
            "corpus_sha256": _sha256(dataset_dir / "corpus.jsonl"),
            "queries_sha256": _sha256(dataset_dir / "queries.jsonl"),
            "qrels_sha256": _sha256(dataset_dir / "qrels" / f"{split}.tsv"),
        },
        "limitations": [
            "The method uses lexical retrievers and does not compare against large neural rerankers.",
            "Adaptive allocation optimizes a batch-level evidence budget and may not apply to single-query deployments.",
            "A statistically inconclusive comparison must not be reported as an improvement.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    csv_name = (
        "per_query_results.csv"
        if output_path.stem == "results"
        else f"per_query_{output_path.stem}.csv"
    )
    csv_path = output_path.with_name(csv_name)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_query_export[0]))
        writer.writeheader()
        writer.writerows(per_query_export)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DABER on a local BEIR-format dataset.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--budgets", default="2,3,5,8")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric-name", default="evidence_recall")
    parser.add_argument("--dataset", default="SciFact-BEIR")
    args = parser.parse_args()
    budgets = sorted({int(value) for value in args.budgets.split(",") if value.strip()})
    if not budgets or min(budgets) < 1 or max(budgets) > 10:
        raise ValueError("budgets must contain integers between 1 and 10")
    payload = run_experiment(
        dataset_dir=Path(args.dataset_dir).resolve(),
        split=args.split,
        budgets=budgets,
        seed=args.seed,
        bootstrap_samples=max(100, args.bootstrap_samples),
        output_path=Path(args.output).resolve(),
    )
    print(json.dumps({
        "dataset": payload["dataset"],
        "queries": payload["dataset_statistics"]["queries"],
        "primary_budget": payload["primary_budget"],
        "candidate": payload["primary_comparison"]["candidate"],
        "candidate_recall": payload["primary_comparison"]["candidate_recall"],
        "fixed_rrf_recall": payload["primary_comparison"]["baseline_recall"],
        "difference": payload["primary_comparison"]["absolute_difference"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

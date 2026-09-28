"""Route B2: learned budget allocation with stronger fusion diagnostics.

Adds train-tuned convex score fusion, rank-biased-overlap and score-dispersion
features, continuous k-nearest-neighbour calibration, feature ablations,
train-subsampling stability, and an optional dense retriever. Test qrels are
used only for evaluation and the explicitly label-aware oracle upper bound.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Iterable

import numpy as np

try:
    from .daber_experiment import (
        BM25Retriever, TfidfRetriever, _sha256, allocate_budget,
        calibrated_disagreement_allocation, char_ngrams, evaluate_rankings,
        fit_disagreement_calibrator, fixed_budget, load_beir,
        optimal_allocation_from_values, oracle_budget_allocation,
        paired_bootstrap, ranking_disagreement, reciprocal_rank_fusion,
        score_margin_uncertainty, word_tokens,
    )
except ImportError:  # Allow direct execution from the workflow directory.
    from daber_experiment import (
        BM25Retriever, TfidfRetriever, _sha256, allocate_budget,
        calibrated_disagreement_allocation, char_ngrams, evaluate_rankings,
        fit_disagreement_calibrator, fixed_budget, load_beir,
        optimal_allocation_from_values, oracle_budget_allocation,
        paired_bootstrap, ranking_disagreement, reciprocal_rank_fusion,
        score_margin_uncertainty, word_tokens,
    )


def rank_biased_overlap(left, right, *, depth=20, persistence=0.9) -> float:
    left_seen, right_seen = set(), set()
    weighted = agreement = 0.0
    for rank in range(1, depth + 1):
        if rank <= len(left): left_seen.add(left[rank - 1][0])
        if rank <= len(right): right_seen.add(right[rank - 1][0])
        agreement = len(left_seen & right_seen) / rank
        weighted += agreement * persistence ** (rank - 1)
    return (1.0 - persistence) * weighted + agreement * persistence**depth


def rbo_disagreement(rankings, *, depth=20, persistence=0.9) -> float:
    values = [1.0 - rank_biased_overlap(rankings[a], rankings[b], depth=depth, persistence=persistence)
              for a in range(len(rankings)) for b in range(a + 1, len(rankings))]
    return statistics.fmean(values) if values else 0.0


def normalized_scores(ranking, depth=100) -> dict[str, float]:
    selected = ranking[:depth]
    if not selected: return {}
    values = [float(score) for _doc, score in selected]
    low, high = min(values), max(values)
    if high == low: return {doc: 1.0 for doc, _score in selected}
    return {doc: (float(score) - low) / (high - low) for doc, score in selected}


def score_dispersion(rankings, *, depth=20) -> float:
    maps = [normalized_scores(ranking, depth) for ranking in rankings]
    union = set().union(*maps)
    return statistics.fmean(statistics.pstdev([mapping.get(doc, 0.0) for mapping in maps]) for doc in union) if union else 0.0


def convex_fusion(rankings, weights, *, depth=100):
    scores = defaultdict(float)
    for weight, ranking in zip(weights, rankings):
        for doc, value in normalized_scores(ranking, depth).items():
            scores[doc] += weight * value
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def simplex_grid(dimensions: int, units: int = 4):
    return [[part / units for part in parts]
            for parts in itertools.product(range(units + 1), repeat=dimensions)
            if sum(parts) == units]


class DenseRetriever:
    def __init__(self, corpus, model_name, batch_size=32, cache_path=None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("dense retrieval requires torch and sentence-transformers") from exc
        self.doc_ids = list(corpus)
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self.cache_path = Path(cache_path) if cache_path else None
        self.embeddings = self._load_cached(model_name)
        if self.embeddings is None:
            encoder = getattr(self.model, "encode_document", self.model.encode)
            self.embeddings = encoder(
                [corpus[doc] for doc in self.doc_ids], batch_size=batch_size,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False).astype(np.float32)
            self._save_cached(model_name)

    def _load_cached(self, model_name):
        if not self.cache_path or not self.cache_path.exists():
            return None
        with np.load(self.cache_path, allow_pickle=False) as cached:
            cached_ids = cached["doc_ids"].astype(str).tolist()
            cached_model = str(cached["model_name"].item())
            if cached_ids == self.doc_ids and cached_model == model_name:
                return cached["embeddings"].astype(np.float32)
        return None

    def _save_cached(self, model_name):
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            doc_ids=np.asarray(self.doc_ids),
            model_name=np.asarray(model_name),
            embeddings=self.embeddings,
        )

    def rank_many(self, queries, limit):
        query_ids = list(queries)
        encoder = getattr(self.model, "encode_query", self.model.encode)
        vectors = encoder([queries[q] for q in query_ids],
            batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        output = {}
        for query_id, vector in zip(query_ids, vectors):
            scores = self.embeddings @ vector
            count = min(limit, len(scores))
            if count == len(scores):
                candidates = np.arange(len(scores))
            else:
                candidates = np.argpartition(scores, len(scores) - count)[-count:]
            ordered = candidates[np.argsort(scores[candidates])[::-1]]
            output[query_id] = [(self.doc_ids[int(i)], float(scores[int(i)])) for i in ordered]
        return output


def build_rankings(retrievers, queries, limit):
    output = {}
    for name, retriever in retrievers.items():
        if isinstance(retriever, DenseRetriever):
            output[name] = retriever.rank_many(queries, limit)
        else:
            output[name] = {qid: retriever.rank(query, limit) for qid, query in queries.items()}
    return output


def fuse_all(rankings, query_ids: Iterable[str], weights=None):
    names = list(rankings)
    if weights is None:
        return {qid: reciprocal_rank_fusion([rankings[name][qid] for name in names]) for qid in query_ids}
    return {qid: convex_fusion([rankings[name][qid] for name in names], weights) for qid in query_ids}


def tune_convex(rankings, qrels, budgets):
    names, trials = list(rankings), []
    best_score, best_weights = -1.0, [1 / len(names)] * len(names)
    for weights in simplex_grid(len(names)):
        fused = fuse_all(rankings, qrels, weights)
        score = statistics.fmean(evaluate_rankings(fused, qrels, fixed_budget(qrels, b))[0]["recall"] for b in budgets)
        trials.append({"weights": dict(zip(names, weights)), "train_macro_recall": score})
        if score > best_score: best_score, best_weights = score, weights
    return best_weights, sorted(trials, key=lambda item: -item["train_macro_recall"])


def query_features(queries, components, fused):
    names = list(components)
    feature_names = ["jaccard_top5", "jaccard_top10", "jaccard_top20", "rbo_top20", "score_dispersion", "rrf_margin", "log_query_length"]
    features, named = {}, {}
    for qid, query in queries.items():
        parts = [components[name][qid] for name in names]
        row = {"jaccard_top5": ranking_disagreement(parts, depth=5),
               "jaccard_top10": ranking_disagreement(parts, depth=10),
               "jaccard_top20": ranking_disagreement(parts, depth=20),
               "rbo_top20": rbo_disagreement(parts),
               "score_dispersion": score_dispersion(parts),
               "rrf_margin": score_margin_uncertainty(fused[qid]),
               "log_query_length": math.log1p(len(word_tokens(query)))}
        named[qid] = row
        features[qid] = [row[name] for name in feature_names]
    return feature_names, features, named


def recall_curves(rankings, qrels, maximum_budget=10):
    curves = {}
    for qid, relevant in qrels.items():
        ranked = [doc for doc, _score in rankings[qid][:maximum_budget]]
        matched, curve = 0, [0.0]
        for budget in range(1, maximum_budget + 1):
            if budget <= len(ranked) and ranked[budget - 1] in relevant: matched += 1
            curve.append(matched / len(relevant))
        curves[qid] = curve
    return curves


def standardize(train, test):
    matrix = np.asarray(list(train.values()), dtype=float)
    mean, scale = matrix.mean(0), matrix.std(0)
    scale[scale == 0] = 1
    transform = lambda values: ((np.asarray(values) - mean) / scale).tolist()
    return ({qid: transform(v) for qid, v in train.items()},
            {qid: transform(v) for qid, v in test.items()},
            {"mean": mean.tolist(), "scale": scale.tolist()})


def knn_predictions(train_features, train_curves, test_features, *, neighbors,
                    feature_indices=None, train_subset=None):
    train_ids = sorted(train_subset or set(train_features))
    feature_indices = feature_indices or list(range(len(next(iter(train_features.values())))))
    matrix = np.asarray([[train_features[q][i] for i in feature_indices] for q in train_ids])
    curves = np.asarray([train_curves[q] for q in train_ids])
    count, output = min(max(1, neighbors), len(train_ids)), {}
    for qid, values in test_features.items():
        vector = np.asarray([values[i] for i in feature_indices])
        distances = np.sum((matrix - vector) ** 2, axis=1)
        indices = np.argpartition(distances, count - 1)[:count]
        weights = 1 / (np.sqrt(distances[indices]) + 1e-6)
        output[qid] = np.maximum.accumulate(np.average(curves[indices], axis=0, weights=weights)).tolist()
    return output


def run_experiment(*, dataset_dir, split, budgets, seed, bootstrap_samples,
                   output_path, dense_model="none", dense_batch_size=32,
                   dense_cache=None, max_train_queries=0):
    corpus, test_queries, test_qrels = load_beir(dataset_dir, split)
    _corpus, train_queries, train_qrels = load_beir(dataset_dir, "train")
    available_train_queries = len(train_queries)
    if max_train_queries and len(train_queries) > max_train_queries:
        selected = set(random.Random(seed).sample(sorted(train_queries), max_train_queries))
        train_queries = {qid: train_queries[qid] for qid in sorted(selected)}
        train_qrels = {qid: train_qrels[qid] for qid in sorted(selected)}
    maximum_rank = max(100, max(budgets) * 5)
    started = time.perf_counter()
    retrievers = {"bm25": BM25Retriever(corpus),
                  "word_tfidf": TfidfRetriever(corpus, word_tokens),
                  "char_3gram_tfidf": TfidfRetriever(corpus, char_ngrams)}
    if dense_model.lower() not in {"", "none", "off", "false"}:
        if dense_cache is None:
            safe_model = dense_model.replace("/", "_").replace("\\", "_")
            dense_cache = output_path.with_name(f"dense_embeddings_{safe_model}.npz")
        retrievers["dense"] = DenseRetriever(
            corpus, dense_model, dense_batch_size, cache_path=dense_cache
        )
    index_seconds = time.perf_counter() - started
    ranking_started = time.perf_counter()
    train_components = build_rankings(retrievers, train_queries, maximum_rank)
    test_components = build_rankings(retrievers, test_queries, maximum_rank)
    rrf_train, rrf_test = fuse_all(train_components, train_qrels), fuse_all(test_components, test_qrels)
    weights, trials = tune_convex(train_components, train_qrels, budgets)
    convex_train, convex_test = fuse_all(train_components, train_qrels, weights), fuse_all(test_components, test_qrels, weights)
    ranking_seconds = time.perf_counter() - ranking_started
    feature_names, train_raw, train_named = query_features(train_queries, train_components, rrf_train)
    _names, test_raw, test_named = query_features(test_queries, test_components, rrf_test)
    train_features, test_features, scaler = standardize(train_raw, test_raw)
    train_curves = recall_curves(convex_train, train_qrels)
    neighbors = max(10, round(math.sqrt(len(train_qrels))))
    predictions = knn_predictions(train_features, train_curves, test_features, neighbors=neighbors)
    feature_sets = {"jaccard_only": [1], "rbo_only": [3], "rank_overlap": [0, 1, 2, 3], "all_features": list(range(len(feature_names)))}
    ablation_predictions = {name: knn_predictions(train_features, train_curves, test_features,
                              neighbors=neighbors, feature_indices=indices)
                            for name, indices in feature_sets.items()}
    raw = {qid: test_named[qid]["jaccard_top10"] for qid in test_qrels}
    train_raw_disagreement = {qid: train_named[qid]["jaccard_top10"] for qid in train_qrels}
    binned = fit_disagreement_calibrator(train_raw_disagreement, convex_train, train_qrels)
    methods_by_budget, bootstraps, per_query_rows = {}, {}, []
    for budget in budgets:
        allocations = {"rrf_fixed": fixed_budget(test_qrels, budget),
                       "convex_fixed": fixed_budget(test_qrels, budget),
                       "raw_jaccard_allocation": allocate_budget(raw, average_budget=budget),
                       "binned_calibration": calibrated_disagreement_allocation(raw, binned, average_budget=budget),
                       "knn_calibration": optimal_allocation_from_values(predictions, average_budget=budget),
                       "oracle_allocation": oracle_budget_allocation(convex_test, test_qrels, average_budget=budget)}
        for name, values in ablation_predictions.items():
            allocations[f"knn_{name}"] = optimal_allocation_from_values(values, average_budget=budget)
        rankings = {name: convex_test for name in allocations}; rankings["rrf_fixed"] = rrf_test
        results, rows = {}, {}
        for name, ranking in rankings.items():
            metric, row = evaluate_rankings(ranking, test_qrels, allocations[name])
            results[name] = {key: round(float(value), 6) for key, value in metric.items()}; rows[name] = row
        methods_by_budget[str(budget)] = results
        bootstraps[str(budget)] = {baseline: {metric: paired_bootstrap(rows["knn_calibration"], rows[baseline], metric=metric,
                                    samples=bootstrap_samples, seed=seed + budget * 100 + i + j * 10)
                                  for i, metric in enumerate(("recall", "hit", "reciprocal_rank", "ndcg"))}
                                    for j, baseline in enumerate(("convex_fixed", "rrf_fixed"))}
        for qid in sorted(test_qrels):
            per_query_rows.append({"query_id": qid, "budget_target": budget,
                "allocated_budget": allocations["knn_calibration"][qid],
                "candidate_recall": rows["knn_calibration"][qid]["recall"],
                "convex_fixed_recall": rows["convex_fixed"][qid]["recall"], **test_named[qid]})
    primary_budget = str(5 if 5 in budgets else budgets[-1])
    stability = []
    train_ids = sorted(train_qrels)
    for repeat_seed in range(seed, seed + 5):
        subset = set(random.Random(repeat_seed).sample(train_ids, max(10, round(.8 * len(train_ids)))))
        values = knn_predictions(train_features, train_curves, test_features, neighbors=neighbors, train_subset=subset)
        allocation = optimal_allocation_from_values(values, average_budget=int(primary_budget))
        metric, _ = evaluate_rankings(convex_test, test_qrels, allocation)
        stability.append({"seed": repeat_seed, "train_queries": len(subset), "recall": metric["recall"], "mrr": metric["mrr"], "ndcg": metric["ndcg"]})
    primary = methods_by_budget[primary_budget]
    comparison = bootstraps[primary_budget]["convex_fixed"]["recall"]
    status = "supported" if comparison["ci95_low"] > 0 else "refuted" if comparison["ci95_high"] < 0 else "inconclusive"
    payload = {"experiment_id": "route_b2_learned_budget_allocation", "dataset": dataset_dir.name, "split": split,
        "metric_name": "evidence_recall", "metric_value": primary["knn_calibration"]["recall"], "primary_budget": int(primary_budget),
        "primary_comparison": {"candidate": "knn_calibration", "baseline": "convex_fixed",
            "candidate_recall": primary["knn_calibration"]["recall"], "baseline_recall": primary["convex_fixed"]["recall"],
            "absolute_difference": round(primary["knn_calibration"]["recall"] - primary["convex_fixed"]["recall"], 6), "paired_bootstrap": comparison},
        "claim_assessment": {"status": status, "reason": "Paired bootstrap against train-tuned convex fusion."},
        "dataset_statistics": {"documents": len(corpus), "available_train_queries": available_train_queries,
            "calibration_train_queries": len(train_queries), "test_queries": len(test_queries),
            "test_positive_qrels": sum(map(len, test_qrels.values()))},
        "retrievers": list(retrievers), "dense_model": None if "dense" not in retrievers else dense_model,
        "fusion": {"selected_weights": dict(zip(retrievers, weights)), "top_train_trials": trials[:10]},
        "calibration": {"method": "distance_weighted_knn_expected_recall_curves", "neighbors": neighbors, "feature_names": feature_names, "standardizer": scaler, "maximum_budget": 10},
        "methods_by_budget": methods_by_budget, "paired_bootstrap_by_budget": bootstraps,
        "feature_ablation": {name: primary[f"knn_{name}"] for name in feature_sets},
        "train_subsampling_stability": {"runs": stability, "recall_mean": statistics.fmean(x["recall"] for x in stability), "recall_std": statistics.pstdev(x["recall"] for x in stability)},
        "runtime": {"index_seconds": index_seconds, "ranking_and_tuning_seconds": ranking_seconds},
        "config": {"budgets": budgets, "seed": seed, "bootstrap_samples": bootstrap_samples,
            "convex_grid_step": .25, "max_train_queries": max_train_queries},
        "data_files": {"corpus_sha256": _sha256(dataset_dir / "corpus.jsonl"), "queries_sha256": _sha256(dataset_dir / "queries.jsonl"), "train_qrels_sha256": _sha256(dataset_dir / "qrels/train.tsv"), "test_qrels_sha256": _sha256(dataset_dir / "qrels" / f"{split}.tsv")},
        "limitations": ["Dense retrieval is included only when configured.", "The allocator requires official train qrels.", "Claims are controlled by per-dataset confidence intervals."]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    with output_path.with_name("per_query_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_query_rows[0])); writer.writeheader(); writer.writerows(per_query_rows)
    return payload


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test"); parser.add_argument("--budgets", default="2,3,5,8")
    parser.add_argument("--bootstrap-samples", type=int, default=1000); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dense-model", default="none"); parser.add_argument("--dense-batch-size", type=int, default=32)
    parser.add_argument("--dense-cache")
    parser.add_argument("--max-train-queries", type=int, default=0)
    parser.add_argument("--metric-name", default="evidence_recall"); parser.add_argument("--dataset", default="BEIR")
    args = parser.parse_args(); budgets = sorted({int(x) for x in args.budgets.split(",") if x.strip()})
    payload = run_experiment(dataset_dir=Path(args.dataset_dir).resolve(), split=args.split, budgets=budgets,
        seed=args.seed, bootstrap_samples=max(100, args.bootstrap_samples), output_path=Path(args.output).resolve(),
        dense_model=args.dense_model, dense_batch_size=args.dense_batch_size,
        dense_cache=Path(args.dense_cache).resolve() if args.dense_cache else None,
        max_train_queries=max(0, args.max_train_queries))
    print(json.dumps({"dataset": payload["dataset"], **payload["primary_comparison"]}, sort_keys=True))


if __name__ == "__main__": main()

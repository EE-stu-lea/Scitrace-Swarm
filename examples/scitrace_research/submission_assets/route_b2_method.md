# Route B2: Continuous Calibration for Budgeted Hybrid Retrieval

Route B2 is a separate experimental line. It does not replace the accepted
Route B run or its paper. Its purpose is to answer the concrete weaknesses in
the first Stanford Agentic Reviewer report with a stronger, fully specified
retrieval study.

## Query-time algorithm

For every query, BM25, word TF-IDF, character 3-gram TF-IDF, and (when enabled)
BGE-small each retrieve the top 100 documents. A convex score fusion is fitted
on the official training split. Each retriever's top-100 scores are min-max
normalized per query, and the fused score is

`fusion(d, q) = sum_r alpha_r * normalized_score_r(d, q)`.

The non-negative weights sum to one. They are selected from a simplex grid with
step 0.25 by maximizing mean training recall across budgets 2, 3, 5, and 8.
Test labels never enter fusion tuning.

The allocator computes seven query features without labels:

1. pairwise Jaccard disagreement at depths 5, 10, and 20;
2. pairwise rank-biased-overlap disagreement at depth 20 with persistence 0.9;
3. dispersion of normalized retrieval scores across retrievers;
4. the margin uncertainty of the RRF ranking; and
5. log query length.

Features are standardized using training-split means and standard deviations.
For a test query, distance-weighted k-nearest neighbours in the training split
predict a cumulative recall curve `v_q(b)` for budgets 0 through 10. The
neighbour count is `max(10, round(sqrt(number_of_training_queries)))`; weights
are inverse Euclidean distance. Predicted curves are forced to be monotone.

Given target average budget `B`, dynamic programming solves

`argmax sum_q v_q(b_q)` subject to `sum_q b_q = B * number_of_queries` and
`1 <= b_q <= 10`.

This is the exact operational rule: the controller reallocates output depth
across queries while the total number of retrieved documents stays identical
to the fixed-budget baseline. It does not split a query's budget among
retrievers; retrievers first produce a single fused ranking, and the controller
chooses how deeply to read that ranking.

## Baselines and ablations

- fixed-budget RRF;
- train-tuned convex score fusion with a fixed budget;
- raw Jaccard-proportional allocation;
- the original five-bin disagreement calibration;
- continuous kNN calibration;
- kNN using Jaccard only, RBO only, all rank-overlap features, or all features;
- a label-aware oracle allocation reported only as an upper bound.

Every method receives exactly the same total document budget. Evaluation
reports evidence recall, hit rate, MRR, and nDCG at budgets 2, 3, 5, and 8.
Paired query bootstrap compares the continuous allocator with both fixed convex
fusion and fixed RRF. Five training-query subsamples measure calibration
stability. The primary result is the macro-average over datasets at budget 5;
averages across budgets are reported separately and never use the same label.

## Current lexical pilot

The first no-dense pilot uses 200 bootstrap resamples. At budget 5:

| Dataset | Continuous allocator | Fixed convex fusion | Difference | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|
| SciFact | 0.778667 | 0.760500 | +0.018167 | [-0.003667, 0.039167] |
| NFCorpus | 0.129846 | 0.122912 | +0.006934 | [0.001474, 0.013683] |

These are development results, not the final paper values. The final run must
use 1,000 bootstrap resamples, record the configured dense model, include a
third dataset or an explicitly marked cross-dataset test, and regenerate all
tables directly from the final JSON artifacts.

## Current hybrid pilot

The BGE-small hybrid run uses the same train/test separation and total-budget
constraint. The independently tuned convex weights were identical on SciFact
and NFCorpus: BM25 0.25, character TF-IDF 0.25, BGE 0.50, and word TF-IDF 0.

| Dataset | Continuous allocator | Fixed convex fusion | Fixed RRF | Difference vs convex |
|---|---:|---:|---:|---:|
| SciFact | 0.844444 | 0.803167 | 0.771556 | +0.041277 |
| NFCorpus | 0.143550 | 0.136970 | 0.135638 | +0.006580 |

At budget 5, the two-dataset macro recall is 0.493997 for the continuous
allocator and 0.470069 for fixed convex fusion. A 1,000-sample stratified paired
bootstrap gives a macro difference of +0.023929 with 95% CI
[0.012170, 0.037179] and P(difference > 0) = 1.000. The unified final artifacts
are stored in `examples/scitrace_research/fixtures/route_b2_final`; each per-dataset result also
contains 1,000 paired bootstrap samples and complete runtime/config metadata.

## FiQA boundary condition

FiQA is retained as an out-of-domain lexical stress test because it has an
official train/test split and a substantially larger corpus (about 57k
documents). To keep calibration tractable, 1,000 of 5,498 available training
queries are selected with seed 42; the complete test split is evaluated.

At budget 5, the continuous allocator reaches 0.238800 recall, compared with
0.242454 for fixed convex fusion and 0.221172 for fixed RRF. The difference
against fixed convex fusion is -0.003654 with 95% paired-bootstrap CI
[-0.015728, 0.007071]. This result is inconclusive and slightly negative. It is
reported as a boundary condition: the learned allocator improves both
scientific retrieval datasets in the hybrid setting but is not claimed to
improve every domain or retriever configuration.

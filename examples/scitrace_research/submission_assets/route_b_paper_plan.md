# Route B paper plan

## Scope

The competition artifact is the SciTrace-Swarm paper-generation system. The
paper generated for evaluation studies an independent information-retrieval
question:

> **When Disagreement Needs Calibration: Budgeted Scientific Evidence
> Retrieval Across SciFact and NFCorpus**

The paper must not present JiuwenSwarm, the Scientific Evidence Graph, or the
competition harness as its scientific contribution. Those belong in the
technical package and framework-contribution documentation.

## Research question and method

Under a fixed batch-level document budget, can disagreement among BM25, word
TF-IDF, and character 3-gram TF-IDF identify scientific queries that deserve
more retrieved evidence?

The evaluated systems are fixed-budget Reciprocal Rank Fusion (RRF), random
allocation, score-margin allocation, raw disagreement allocation, a train-only
calibrated disagreement policy, and a label-aware oracle upper bound. The
calibrated policy learns five disagreement-quantile recall curves on each
dataset's train split and applies them to the test split. Test qrels are used
only for evaluation and the oracle analysis.

## Verified results

All values below are mean evidence recall. Paired bootstrap uses 1,000 samples.

| Dataset | Budget | Fixed RRF | Raw disagreement | Calibrated | Oracle | Calibrated minus fixed 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 2 | 0.6322 | 0.6219 | 0.6322 | 0.8010 | [0.0000, 0.0000] |
| SciFact | 3 | 0.6852 | 0.6789 | 0.6836 | 0.8010 | [-0.0183, 0.0150] |
| SciFact | 5 | 0.7359 | 0.7159 | 0.7453 | 0.8010 | [-0.0072, 0.0284] |
| SciFact | 8 | 0.7960 | 0.7724 | 0.7860 | 0.8010 | [-0.0267, 0.0033] |
| NFCorpus | 2 | 0.0856 | 0.0818 | 0.0913 | 0.1378 | [-0.0006, 0.0120] |
| NFCorpus | 3 | 0.1061 | 0.0994 | 0.1067 | 0.1513 | [-0.0083, 0.0093] |
| NFCorpus | 5 | 0.1274 | 0.1203 | 0.1286 | 0.1566 | [-0.0072, 0.0074] |
| NFCorpus | 8 | 0.1469 | 0.1371 | 0.1503 | 0.1566 | [-0.0019, 0.0085] |

Disagreement has a positive but modest association with first relevant rank:
SciFact Spearman rho is 0.1363 (95% CI [0.0138, 0.2499]); NFCorpus rho is
0.2680 (95% CI [0.1617, 0.3629]). Raw disagreement allocation nevertheless
reduces recall at every tested budget. Train-only calibration removes most of
that loss and yields small positive point estimates at SciFact budget 5 and
NFCorpus budgets 2, 3, 5, and 8. These calibrated improvements are trends, not
statistically significant wins, because their confidence intervals include
zero.

## Claims the paper may make

1. Retriever disagreement is a measurable but weak proxy for scientific-query
   difficulty on both evaluated datasets.
2. Treating raw disagreement as a direct allocation weight is consistently
   worse than fixed-budget RRF in this experiment.
3. Train-only calibration recovers the raw policy's losses and sometimes gives
   small positive test-set point estimates, but the current experiment does not
   establish a statistically significant improvement over fixed RRF.
4. The oracle results show that adaptive allocation has remaining headroom, so
   better calibrated signals are a concrete direction for future work.

The paper must not claim state-of-the-art performance, neural-retriever
coverage, statistically significant calibrated gains, or generalization beyond
SciFact and NFCorpus.

## Audit artifacts

- SciFact result: `runs/daber_scifact_calibrated_20260921/results.json`
  - SHA-256: `51C40616D7F0A7FAC5C8A64D402837C585B2D131220BB06DEC085EC151F77CDD`
- NFCorpus result: `runs/daber_nfcorpus_calibrated_20260921/results.json`
  - SHA-256: `B8902FD8BB53E149D8CDFF273C6BFEF47D1522B3D0D19C03BB79097E4BDE9D9F`
- Experiment implementation:
  `examples/scitrace_research/workflow/daber_experiment.py`
- Multi-dataset entry point:
  `examples/scitrace_research/workflow/daber_multidataset_experiment.py`
- Team configuration:
  `examples/scitrace_research/configs/team_real_route_b.yaml`

The exact PDF produced by the final Team run must be uploaded to Stanford
Agentic Reviewer. A token issued for an older PDF does not correspond to this
paper.

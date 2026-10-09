# final: a question-led, evidence-bound paper

The design reference is Laban et al., *LLMs Get Lost in Multi-Turn Conversation*,
ICLR 2026. We borrow organization, not its data, claims, figure artwork, task,
or percentile-based aptitude/unreliability metrics. Our stored retrieval outcomes
are not repeated stochastic attempts on the same task.

## Main argument

1. **Introduction:** the matched document-context budget problem, a bounded
   takeaway, three research questions and specific contributions.
2. **Related Work:** foundational benchmarks/scientific claim verification;
   hybrid retrieval; per-query adaptive retrieval versus joint batch allocation.
   Reserved title matching selects only actually retrieved records. If a paper
   is unavailable, do not invent its metadata or evidence ID.
3. **Method:** feature construction, training-only fitting, real-valued predicted
   recall curves, discrete allocation and exact capacity constraint. Figure 1
   highlights the changed component and the train/test boundary.
4. **Experimental Setup:** datasets/splits, metrics, comparators, controls,
   conditional bootstrap, and explicit import of completed artifacts.
5. **Results:** answer RQ1–RQ3 below. Do not just restate every table entry.
6. **Discussion:** why mean improvement does not guarantee every query retains
   evidence; what is observed versus a conjectured mechanism; deployment limits.
7. **Limitations / Conclusion / References**, optional appendix for detailed
   protocol. No essential negative finding should be relegated to an appendix.

| Question | Evidence and figure | Interpretation boundary |
|---|---|---|
| RQ1: Effect at matched mean budget? | Table 1 paired main effects; Table 2 adaptive baselines; circular-point budget-effect figure and recall sweep | Main macro effect, SciFact secondary result, inconclusive NFCorpus; no uncertainty against adaptive baselines |
| RQ2: Who gains/loses evidence? | Table 3 exact query outcome counts; gain/loss contribution and zero-recall transition figure | Descriptive outcomes for the realized allocation; no causal or repeated-run reliability claim |
| RQ3: Sensitivity and boundaries? | Table 4 feature ablations; five training-subset draws; FiQA negative result | Not full-run seeds, not zero-shot transfer, no demonstrated downstream quality or compute benefit |

## Figure contract

- `method_overview.pdf`: original schematic, no fabricated measurements; bound
  to the implementation contract and renderer SHA-256.
- `budget_effect.pdf`: circular estimates over shaded pointwise intervals;
  the zero line and shared vertical scale make heterogeneous effects legible.
  Lines/shading between measured budgets are visual guides, not observations.
- `recall_by_budget.pdf`: absolute recall, separate dataset scales clearly noted.
- `query_risk.pdf`: positive/negative contributions to the same mean difference,
  win/tie/loss shares, and newly-zero versus rescued query counts. Exact saved
  per-query outcomes are recomputed before export, including capacity checks.

Captions must define the comparison, metric/units, data and uncertainty scope.
Figures use a fixed allowlist; arbitrary image paths are rejected. Their data,
sources and export copies are hash-bound. PDF floats may use available space;
they do not silently replace manuscript prose.

## Newly exposed result (retrospective analysis)

At mean depth 5, NFCorpus has 62 improving, 215 tied and 46 worsening queries.
Five queries move from zero recall to positive recall, while 17 go the other
way. Its positive mean difference therefore coexists with 12 additional
zero-recall queries. This follows directly from saved outcomes, is not a new
retrieval run, and has no newly claimed significance test.

For SciFact, the corresponding gain/tie/loss counts are 18/279/3 and
rescue/newly-zero counts 14/2. For lexical FiQA they are 23/601/24 and 14/18.
These counts are checked before draft acceptance. Counts do not establish
answer correctness, fairness, latency, or token savings.

## Writing and validation

The scoped research brief, complete manuscript context, and full-draft reviewer
share the method contract, RQs and exact diagnostic rows. Deterministic checks
catch missing RQ sections, coordination chatter, missing figures and altered
counts. They complement semantic review; they do not certify novelty or a
conference-level score. Suggested 3,000–3,800 body words are planning guidance,
not a length reward. Never pad a paper to reproduce the reference's page count.

Relevant primary sources for live retrieval and background verification:

- [BEIR](https://arxiv.org/abs/2104.08663)
- [Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/)
- [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/)

These links guide retrieval; their presence here is not a live-provider receipt.

# Reproduce a Route B2 paper from the Agent

The PDF must come from a fresh `real_team` run. Do not edit `draft.md`, `paper.tex`, or `paper.pdf` after validation. The exporter checks that the draft SHA-256 matches both the successful validation report and run manifest.

The Team imports the checked Route B2 result fixture at `examples/scitrace_research/fixtures/route_b2_final`. Its importer verifies the aggregate and per-query artifact hashes. This Team run does **not** rerun BM25, TF-IDF, dense retrieval, or the bootstrap experiment. The experiment implementation is `workflow/route_b2_experiment.py`; `run_route_b2_experiments.ps1` documents a full rerun using the public BEIR SciFact and NFCorpus datasets. Benchmark corpora and dense ranking caches are not committed to this repository. A full rerun must supply those files and the specified BGE-small model. The FiQA lexical stress test is a separate boundary result.

## Generate a new draft and PDF

Set `API_BASE`, `API_KEY`, `MODEL_NAME`, and `MODEL_PROVIDER` in the environment. Install the repository's runtime dependencies, pinned `matplotlib` for the deterministic figures, and `pdflatex` with the bundled ICLR style. From the repository root, use a **new** run directory; resuming the old successful run will reuse its old stage checkpoints.

```bash
python -m examples.scitrace_research.run_team \
  --topic "Continuous difficulty calibration for budgeted hybrid scientific evidence retrieval" \
  --config examples/scitrace_research/configs/team_real_route_b2.yaml \
  --output runs/route_b2_quality_new \
  --mode real_team --strict-real --no-mock-fallback

python -m examples.scitrace_research.paper_export \
  --run-dir runs/route_b2_quality_new
```

If the literature search finds fewer than four relevant verified sources, or the generated draft fails the scientific or paper quality gate, the run fails. The current paper gate requires the main results table, adaptive-baseline table, and two embedded vector figures with substantive Agent-written captions. The budget-sweep figure is generated from `experiments/results.json`; the three-dataset effect figure is generated from the imported Route B2 aggregate and per-dataset results. Both are prepared before the Writer Agent receives its prompt. Their plotted data, source hashes, and PDF hashes are recorded in `figures/`. The effect figure now displays the existing pointwise paired-bootstrap intervals at every budget. FiQA is separately fitted on its own training data, not a zero-shot transfer test. The intervals condition on the original allocation and are not fresh-batch uncertainty. Inspect `evidence/literature_selection.json`, `paper_quality_report.json`, and `validation_report.json`, then start a new run after correcting the retrieval configuration or Agent prompts. Do not copy an old PDF into the new run.

If a deterministic gate has a confirmed implementation bug, `python -m examples.scitrace_research.revalidate_run --run-dir runs/<failed-run>` replays both gates against the unchanged Agent draft. It accepts either a completed consistency checkpoint bound to the failed validation hash, or a narrowly eligible partial loop stopped by the global request ceiling after a paid review that binds the current draft hash. The partial route rejects unrelated reviewer defects or old quality errors outside the corrected parser checks. It makes no model call, verifies the complete per-call token log, and preserves the original manifest, draft, quality report, review report, and WAL-consistent database as `revalidation_original_*`. A passing replay records the correction in `audit.jsonl`; the original resource report remains unchanged. Then use the normal `paper_export` command. This path is not for repairing a scientifically invalid draft.

## Check the provenance of the exported PDF

The following files should exist in the new run:

- `run_config.yaml`, `run_manifest.json`, `model_calls.jsonl`, `resource_usage.json`
- `evidence/raw/`, `evidence/literature.json`, `evidence/literature_selection.json`
- `draft.md`, `paper_quality_report.json`, `validation_report.json`
- `figures/recall_by_budget.pdf`, `figures/figure_manifest.json`
- `figures/budget_effect.pdf`, `figures/budget_effect_data.json`, `figures/budget_effect_manifest.json`
- `paper/paper.tex`, `paper/paper.pdf`, `paper/paper_manifest.json`

`run_manifest.json` and `validation_report.json` must both say `status=success` / `ok=true` and contain the same `draft_sha256`. `paper/paper_manifest.json` records the exact draft, literature, figures, TeX, and PDF hashes. The Markdown-to-LaTeX conversion only typesets the accepted Agent draft and hides audit identifiers; it does not replace the paper's prose. A human should still review the cited work and scientific interpretation before submission.

## Revise a scientifically inconsistent Agent draft

Preserve the failed run for audit, correct the workflow configuration or Agent instructions, and start a new Team run in a new output directory. The Agent must revise the manuscript; do not hand-edit `draft.md`, `paper.tex`, or `paper.pdf`. After any gate update, the PDF exporter replays the current paper-quality checks against the hash-bound Agent draft. A historical `status=success` and stored quality report cannot bypass the current checks. The token guard is a runaway protection rather than a contest limit, and actual usage remains recorded in the run artifacts.

## 2026-10-08 implementation corrections

The writer receives a structured Route B2 implementation contract and the example enables its deterministic checks. Depths are 1..10; budget is a batch mean; overlap-named fields are disagreement features; training-subset sensitivity changes only the kNN library. A historical success flag does not make the original prose accurate under these revised checks. Keep the historical run immutable and generate a fresh draft for an official Agent submission.

The standalone saved-query reanalysis is available as `workflow/robustness_analysis.py`, and publication plots as `workflow/publication_figures.py`. The local `improvements/` manuscript is an explicitly separated Codex-assisted revision, not the output of a new real Team run. It uses 50,000 bootstrap draws and a 16-comparison family correction. Do not mix these intervals with the original fixture's 1,000/200-draw intervals without identifying the analysis version.

Native Team usage now records only provider `llm_usage` receipts in `provider_usage.jsonl`. Stage call counts and observed provider receipt counts are distinct. Configuration guards operate after receipts arrive and can overshoot. Missing receipts and legacy accounting must remain visible in resource reports.

The final recovery configuration uses a hybrid execution route: native Team for
leader-owned topic framing and claim compilation, and one named openJiuwen model
request per attempt for experiment planning, drafting, review and revision. The
direct stages use the same scientific prompts and checked handoff fields, log
native response usage to `provider_usage.jsonl`, and retain stage limits. This
route was introduced after a failed experiment-plan attempt spent ten provider
calls on Team task-board coordination. It has passed local tests and a no-API
resume preflight; paper quality and actual cost remain unmeasured until the
resumed run completes. The manuscript stages use a 12,000-token output ceiling
to allow the requested full-length paper, not as a token-consumption target.

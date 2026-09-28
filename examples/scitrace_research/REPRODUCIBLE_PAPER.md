# Reproduce a Route B2 paper from the Agent

The PDF must come from a fresh `real_team` run. Do not edit `draft.md`, `paper.tex`, or `paper.pdf` after validation. The exporter checks that the draft SHA-256 matches both the successful validation report and run manifest.

The Team imports the checked Route B2 result fixture at `examples/scitrace_research/fixtures/route_b2_final`. Its importer verifies the aggregate and per-query artifact hashes. This Team run does **not** rerun BM25, TF-IDF, dense retrieval, or the bootstrap experiment. The experiment implementation is `workflow/route_b2_experiment.py`; `run_route_b2_experiments.ps1` documents a full rerun using the public BEIR SciFact and NFCorpus datasets. Benchmark corpora and dense ranking caches are not committed to this repository. A full rerun must supply those files and the specified BGE-small model. The FiQA lexical stress test is a separate boundary result.

## Generate a new draft and PDF

Set `API_BASE`, `API_KEY`, `MODEL_NAME`, and `MODEL_PROVIDER` in the environment. Install the repository's runtime dependencies, `matplotlib` for the deterministic budget-sweep figure, and `pdflatex` with the bundled ICLR style. From the repository root, use a **new** run directory; resuming the old successful run will reuse its old stage checkpoints.

```bash
python -m examples.scitrace_research.run_team \
  --topic "Continuous difficulty calibration for budgeted hybrid scientific evidence" \
  --config examples/scitrace_research/configs/team_real_route_b2.yaml \
  --output runs/route_b2_quality_new \
  --mode real_team --strict-real --no-mock-fallback

python -m examples.scitrace_research.paper_export \
  --run-dir runs/route_b2_quality_new
```

If the literature search finds fewer than four relevant verified sources, or the generated draft fails the scientific or paper quality gate, the run fails. The current paper gate requires a main results table whose values match the experiment artifact and an embedded vector budget-sweep figure. The figure is generated from `experiments/results.json` before the Writer Agent receives its prompt; its exact data and hash are recorded in `figures/figure_manifest.json`. Inspect `evidence/literature_selection.json`, `paper_quality_report.json`, and `validation_report.json`, then start a new run after correcting the retrieval configuration or Agent prompts. Do not copy an old PDF into the new run.

If a deterministic gate has a confirmed implementation bug, `python -m examples.scitrace_research.revalidate_run --run-dir runs/<failed-run>` replays both gates against the unchanged Agent draft in a completed real Team checkpoint. It makes no model call, requires the original draft hash and complete per-call token log, and preserves the original failed manifest and validation report as `revalidation_original_*.json`. A passing replay records the correction in `audit.jsonl`; the original run's token overrun remains in the resource report. Then use the normal `paper_export` command. This path is not for repairing a scientifically invalid draft.

## Check the provenance of the exported PDF

The following files should exist in the new run:

- `run_config.yaml`, `run_manifest.json`, `model_calls.jsonl`, `resource_usage.json`
- `evidence/raw/`, `evidence/literature.json`, `evidence/literature_selection.json`
- `draft.md`, `paper_quality_report.json`, `validation_report.json`
- `figures/recall_by_budget.pdf`, `figures/figure_manifest.json`
- `paper/paper.tex`, `paper/paper.pdf`, `paper/paper_manifest.json`

`run_manifest.json` and `validation_report.json` must both say `status=success` / `ok=true` and contain the same `draft_sha256`. `paper/paper_manifest.json` records the exact draft, literature, TeX, and PDF hashes. The Markdown-to-LaTeX conversion only typesets the accepted Agent draft and hides audit identifiers; it does not replace the paper's prose. A human should still review the cited work and scientific interpretation before submission.

## Revise a scientifically inconsistent Agent draft

Preserve the failed run for audit, correct the workflow configuration or Agent instructions, and start a new Team run in a new output directory. The Agent must revise the manuscript; do not hand-edit `draft.md`, `paper.tex`, or `paper.pdf`. After any gate update, the PDF exporter replays the current paper-quality checks against the hash-bound Agent draft. A historical `status=success` and stored quality report cannot bypass the current checks. The token guard is a runaway protection rather than a contest limit, and actual usage remains recorded in the run artifacts.

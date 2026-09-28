# Agent paper export and mechanism evaluation

Run `run_submission.ps1 -TeamName NAME` after setting model environment variables. It now runs the Team and exports the actual draft, then stops for external review. No Reviewer token is required at this stage.

For an existing successful run:

```powershell
.\.venv_real\Scripts\python.exe -m examples.scitrace_research.paper_export --run-dir runs/YOUR_RUN
```

Upload that run's `paper/paper.pdf` to Reviewer. After receiving the token, invoke `build_submission --run-dir runs/YOUR_RUN --team-name NAME --access-token TOKEN --output-root submission_FINAL`. Append `--pr-url OFFICIAL_PR_URL` when a public contribution PR is available. Packaging verifies the draft, literature, LaTeX and PDF hashes and copies the PDF without recompilation. Use a fresh output directory for each package. Token-to-review correspondence must be confirmed on the external review site; a local hash cannot authenticate a remote token.

The exporter typesets Agent prose instead of generating fixed prose. Structured References and Team Handoff are excluded from the paper body; referenced evidence metadata is appended verbatim as source metadata. Unsupported body blocks produce an explicit error. The exporter's success does not imply that all scientific claims or paper quality are correct.

## Offline mechanism experiments

```powershell
.\.venv_real\Scripts\python.exe -m examples.scitrace_research.evaluate_gate --output runs/gate_ablation/results.json
```

This invokes the actual ScientificConsistencyGateRail on persisted synthetic SEG records: one clean control and ten faulty payloads, compared with an explicitly gate-free accept-all baseline. It also compares StageAwareContextRail against random-order packing using the same token estimates and budgets 64/96/128/256, with five random seeds per budget. Coverage measures retention of the required artifact and claim, not downstream scientific accuracy. The raw cases and selected-budget statistics are saved to JSON.

Observed gate result: 9/10 faulty payloads detected, zero false positives on the single clean control. The prose-only numeric drift is missed and retained as a negative result. These small controlled experiments do not establish generalization, real API cost savings, or Team superiority. Those require separate real-model experiments. Do not add these results to an already reviewed PDF without regeneration and re-review.

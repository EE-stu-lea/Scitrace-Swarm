# Grounded manuscript generation and review

The Route B2 configuration separates manuscript completeness from scientific evidence. The experiment importer supplies existing measured results; it does not perform a fresh retrieval experiment. `paper_quality` requirements apply to the live Agent output and are replayed before PDF export.

The Writer receives a manuscript plan, explicit implementation facts, verified literature and compact result views. Required tables cover primary paired comparisons, adaptive baselines including the negative FiQA comparison, and all reported feature ablations. Minimum section prose counts identify omitted explanations; they exclude tables, captions, code, references and audit metadata and must not be treated as a scientific quality score.

Use `\( ... \)` for inline mathematics and standalone `\[ ... \]` paragraphs for display formulas, with ASCII LaTeX notation. The exporter also supports dollar delimiters, validates a limited scientific command set and rejects file/control commands. Write objective and constraint equations separately; arbitrary environments such as `aligned` and `cases` are unsupported. Export preserves scientific text; it does not invent equations or rewrite claims.

The consistency loop runs deterministic checks and a full-public-manuscript review for each draft. The Reviewer must return a summary, severity and findings array. Every finding requires an exact quote, section, explanation and proposed correction. Error findings enter targeted revision; optional warnings remain in the audit without forcing another paid rewrite. A revision uses the existing review and receives the same authoritative evidence, instead of requesting a duplicate review or relying on the failed draft as its sole context. Each revised draft is checked again, subject to the configured revision and resource limits.

`manuscript_context.json` preserves the authoring evidence. `reviews/round_N.json` binds each review to its draft SHA-256; `review_report.json` contains the latest review. Deterministic checks remain necessary because an LLM reviewer can still make incorrect suggestions or miss mistakes. Successful local validation is not an official conference or competition review score.

The whole-run budget is checked before review/revision calls. A zero revision allowance performs one audit but no rewrite. Observed token limits are post-receipt guards, so an in-flight Team invocation may overshoot. Full-text review can cost more than the previous truncated audit; this change makes review useful and auditable, without claiming an end-to-end cost reduction.

Runtime exceptions stop the native Team and preserve observed usage. An audit-write failure remains a visible error and does not capture the same provider receipt twice. The regression suites use fixture streams and model responses; real provider integration must be measured separately.

Run the research test directory from `source/` with the project development dependencies. See `configs/team_real_route_b2.yaml` for the enabled manuscript requirements and `run_route_b2.sh` for the module entry points. Existing run artifacts should remain immutable; start a new output directory to measure a changed pipeline.

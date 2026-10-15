# Scitrace-Swarm

**Multi-agent orchestration for research paper generation**

This repository is a secondary development of [JiuwenSwarm](https://github.com/openJiuwen-ai/jiuwenswarm). It adds **Scitrace-Swarm**, a research workflow connecting literature retrieval, restricted experiments or verified result import, claim provenance, drafting, review, quality checks, and PDF export. Specialized agents collaborate across the stages, with checkpoints and audit records for inspection.

> JiuwenSwarm's general Agent, desktop, and channel features come from the upstream project. This repository focuses on Scitrace-Swarm. See the [upstream documentation](https://github.com/openJiuwen-ai/jiuwenswarm#readme) for the original product and its general setup.

[中文说明](README_CN.md) · [Research workflow guide](examples/scitrace_research/README.md) · [System design](examples/scitrace_research/docs/final_system_design.md) · [Paper design](examples/scitrace_research/docs/final_paper_design.md)

## Added capabilities

| Stage | Scitrace-Swarm implementation |
| --- | --- |
| Agent coordination | A research leader coordinates literature, experiment, writing, and review stages with role and handoff contracts. |
| Literature and evidence | Search and filter papers, optionally expand one citation hop through OpenAlex, and audit sources and calls. |
| Experiments and provenance | Run restricted experiments or import verified results; link papers, claims, and artifacts in a provenance graph. |
| Manuscript workflow | Draft from structured evidence, review findings, apply targeted revisions, and run deterministic quality checks. |
| Skills | Local pre-submission checklist, statistical notation, and citation tracing Skills; writing stages can load the relevant guidance. |
| Delivery | Track resource use and checkpoints, bind artifacts with hashes, and export an ICLR-style PDF after validation. |

Research claims, citations, and experiment results still need human review.

## Repository layout

- [`jiuwenswarm/agents/harness/research/`](jiuwenswarm/agents/harness/research/): research harness, Team runtime, stage contracts, provenance, and auditing.
- [`examples/scitrace_research/`](examples/scitrace_research/): workflow configuration, experiment examples, quality gates, and paper export.
- [`examples/scitrace_research/skills/`](examples/scitrace_research/skills/): local Scitrace-Swarm Skill packages.
- [`tests/unit_tests/agents/harness/research/`](tests/unit_tests/agents/harness/research/): research workflow tests.

## Getting started

Use Python 3.11–3.13 and install the repository and Route B2 dependencies from the repository root:

```bash
python -m pip install -e .
python -m pip install -r examples/scitrace_research/requirements-route-b2.txt
```

Follow the [research workflow guide](examples/scitrace_research/README.md) to set model environment variables. Check the run requirements before starting a real workflow:

```bash
bash examples/scitrace_research/launch_paper.sh --check
```

The default example imports hash-checked Route B2 results included in the repository; it **does not rerun retrieval experiments**. See the [reproducibility notes](examples/scitrace_research/REPRODUCIBLE_PAPER.md) for the exact boundary and full experiment path. PDF export also requires `pdflatex`.

Run the research workflow tests:

```bash
python -m pytest tests/unit_tests/agents/harness/research -q
```

## Origin and license

This repository retains the original JiuwenSwarm code and history. Scitrace-Swarm adds a research paper generation workflow on top. Refer to [JiuwenSwarm](https://github.com/openJiuwen-ai/jiuwenswarm) for upstream documentation and contributor attribution. See [LICENSE](LICENSE) for the license.

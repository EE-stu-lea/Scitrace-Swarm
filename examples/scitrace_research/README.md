# SciTrace Research

The **final** iteration adds question-led manuscript generation, verified query
risk diagnostics and scoped native Team execution. See
[paper design](docs/final_paper_design.md) and [system design](docs/final_system_design.md).
For writing from saved artifacts, install `requirements-paper.txt`; the dense
experiment stack is only needed when rerunning retrieval.

SciTrace Research is a JiuwenSwarm Team workflow for producing an auditable
scientific short paper. It connects literature retrieval, restricted experiment
execution or verified artifact import, claim provenance, Agent writing,
consistency review, resource accounting, and ICLR PDF export.

## Repository layout

- `jiuwenswarm/agents/harness/research/`: reusable research harness, rails,
  provenance graph, Team runtime, and resource reporting.
- `examples/scitrace_research/`: workflow configuration, example experiments,
  quality gates, submission builder, and PDF exporter.
- `tests/unit_tests/agents/harness/research/`: focused unit and integration tests.

Runtime output is written under `runs/` and is excluded from Git.

## Install

Use Python 3.11 and install the repository plus the optional Route B2
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r examples/scitrace_research/requirements-route-b2.txt
```

Install `pdflatex` separately to export the ICLR PDF.

## Run a real Team workflow

Provide model settings through the environment. Never commit API credentials.

```bash
export API_BASE='https://your-provider.example/v1'
export API_KEY='your-secret-key'
export MODEL_NAME='your-model-name'
export MODEL_PROVIDER='openai'

bash examples/scitrace_research/launch_paper.sh --check
bash examples/scitrace_research/launch_paper.sh
```

The default workflow imports the hash-checked Route B2 fixture included under
`fixtures/route_b2_final`; it does not claim to rerun retrieval. See
`REPRODUCIBLE_PAPER.md` for the exact reproducibility boundary and full
experiment entry point.

## Test

```bash
python -m pytest -q tests/unit_tests/agents/harness/research
```

The generated paper remains eligible for export only while its Agent draft,
validation report, paper-quality report, and export manifest have matching
SHA-256 provenance.

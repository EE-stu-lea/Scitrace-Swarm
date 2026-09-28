from __future__ import annotations

import json
from pathlib import Path

import pytest

from ._stubs import install_openjiuwen_stubs

install_openjiuwen_stubs()

from jiuwenswarm.agents.harness.research.experiment_executor import (
    ExperimentExecutorError,
    SafeExperimentExecutor,
)
from jiuwenswarm.agents.harness.research.literature import (
    FixtureLiteratureProvider,
    OpenAlexLiteratureProvider,
    ResilientLiteratureProvider,
)


def test_literature_deduplication(monkeypatch, tmp_path) -> None:
    provider = OpenAlexLiteratureProvider(
        cache_root=tmp_path / "raw",
        max_results=5,
        timeout_seconds=1,
        retries=0,
        verify_links=True,
    )
    monkeypatch.setattr(
        provider,
        "_request",
        lambda topic: {
            "results": [
                {
                    "display_name": "Evidence Graphs for Science",
                    "authorships": [{"author": {"display_name": "Alice"}}],
                    "publication_year": 2026,
                    "doi": "https://doi.org/10.1000/xyz",
                    "primary_location": {"landing_page_url": "https://example.org/a"},
                    "abstract_inverted_index": {"evidence": [0], "graphs": [1]},
                    "relevance_score": 0.9,
                },
                {
                    "display_name": "Evidence Graphs for Science",
                    "authorships": [{"author": {"display_name": "Alice"}}],
                    "publication_year": 2026,
                    "doi": "https://doi.org/10.1000/xyz",
                    "primary_location": {"landing_page_url": "https://example.org/b"},
                    "abstract_inverted_index": {"duplicate": [0]},
                    "relevance_score": 0.8,
                },
            ]
        },
    )
    monkeypatch.setattr(provider, "_can_fetch", lambda url: True)

    results = provider.search("evidence graphs")
    assert len(results) == 1
    assert results[0].doi == "10.1000/xyz"


def test_literature_validation_fallback(monkeypatch, tmp_path) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(
            [
                {
                    "title": "Fixture Paper",
                    "authors": ["Alice"],
                    "year": 2025,
                    "doi": "10.1/fallback",
                    "source_url": "https://example.org/fallback",
                    "summary": "Fixture fallback entry.",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    fallback = FixtureLiteratureProvider(fixture_path)
    provider = OpenAlexLiteratureProvider(
        cache_root=tmp_path / "raw",
        max_results=5,
        timeout_seconds=1,
        retries=0,
        verify_links=True,
    )
    monkeypatch.setattr(
        provider,
        "_request",
        lambda topic: {
            "results": [
                {
                    "display_name": "Bad Link Paper",
                    "authorships": [{"author": {"display_name": "Bob"}}],
                    "publication_year": 2026,
                    "doi": "https://doi.org/10.9999/bad",
                    "primary_location": {"landing_page_url": "https://bad.invalid/paper"},
                    "abstract_inverted_index": {"bad": [0]},
                    "relevance_score": 0.5,
                }
            ]
        },
    )
    monkeypatch.setattr(provider, "_can_fetch", lambda url: False)
    resilient = ResilientLiteratureProvider(providers=[provider], fallback=fallback)

    results = resilient.search("bad links")
    assert len(results) == 1
    assert results[0].title == "Fixture Paper"


def test_experiment_timeout(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script_path = workspace / "slow.py"
    script_path.write_text(
        "import argparse, time\n"
        "parser=argparse.ArgumentParser(); parser.add_argument('--output', required=True); args=parser.parse_args()\n"
        "time.sleep(2)\n",
        encoding="utf-8",
    )
    executor = SafeExperimentExecutor(workspace=workspace, timeout_seconds=1)

    with pytest.raises(ExperimentExecutorError):
        executor.execute(
            {
                "kind": "python_inline",
                "script_path": str(script_path),
                "args": [],
                "seed": 1,
            },
            output_dir=tmp_path / "out",
        )


def test_reject_non_workspace_script(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('bad')\n", encoding="utf-8")
    executor = SafeExperimentExecutor(workspace=workspace, timeout_seconds=5)

    with pytest.raises(ExperimentExecutorError):
        executor.execute(
            {
                "kind": "python_inline",
                "script_path": str(outside),
                "args": [],
                "seed": 1,
            },
            output_dir=tmp_path / "out",
        )


def test_real_retrieval_experiment_produces_distinct_metric(tmp_path) -> None:
    from examples.scitrace_research.workflow.pipeline import _experiment_script

    script_text = _experiment_script()
    assert "mock_metric_value" not in script_text

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script_path = workspace / "run_experiment.py"
    script_path.write_text(script_text, encoding="utf-8")

    executor = SafeExperimentExecutor(workspace=workspace, timeout_seconds=30)
    result = executor.execute(
        {
            "kind": "python_inline",
            "script_path": str(script_path),
            "seed": 42,
            "args": ["--metric-name", "citation_support_rate", "--dataset", "SciTraceBench"],
        },
        output_dir=workspace / "out",
    )

    payload = result.payload
    assert isinstance(payload["metric_value"], float)
    assert payload["metric_value"] == payload["vector_support_rate"]
    # The evidence-aware (stem-based) method should beat the keyword baseline
    # on this benchmark, because the baseline is lured by distractor evidence.
    assert payload["vector_support_rate"] > payload["rule_baseline_support_rate"]

from __future__ import annotations

import json
from pathlib import Path

from ._stubs import install_openjiuwen_stubs

install_openjiuwen_stubs()

from examples.scitrace_research.run import main as cli_main
from examples.scitrace_research.workflow.pipeline import (
    WorkflowInterrupted,
    _complete_structured_references,
    _extract_structured_references,
    load_yaml_config,
    run_pipeline,
)


def _config() -> dict:
    path = (
        Path(__file__).resolve().parents[5]
        / "examples"
        / "scitrace_research"
        / "configs"
        / "mvp.yaml"
    )
    return load_yaml_config(path)


def test_scitrace_mock_end_to_end_workflow_runs(tmp_path) -> None:
    result = run_pipeline(
        topic="Evidence-aware context engineering for autonomous research agents",
        config=_config(),
        output_dir=tmp_path / "demo001",
        mode="mock",
        resume=False,
        dry_run=False,
    )

    assert result["status"] == "success"
    output_dir = Path(result["output_dir"])
    assert (output_dir / "run_config.yaml").exists()
    assert (output_dir / "research.db").exists()
    assert (output_dir / "audit.jsonl").exists()
    assert (output_dir / "evidence" / "literature.json").exists()
    assert (output_dir / "experiments" / "plan.json").exists()
    assert (output_dir / "experiments" / "stdout.log").exists()
    assert (output_dir / "experiments" / "stderr.log").exists()
    assert (output_dir / "experiments" / "results.json").exists()
    assert (output_dir / "claims.json").exists()
    assert (output_dir / "draft.md").exists()
    assert (output_dir / "validation_report.json").exists()
    assert (output_dir / "resource_usage.json").exists()
    assert (output_dir / "resource_report.md").exists()

    validation = json.loads((output_dir / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["ok"] is True
    draft = (output_dir / "draft.md").read_text(encoding="utf-8")
    assert "## Structured References" in draft
    assert "citation_support_rate" in draft


def test_complete_run_artifact_directory_structure(tmp_path) -> None:
    result = run_pipeline(
        topic="Evidence-aware context engineering for autonomous research agents",
        config=_config(),
        output_dir=tmp_path / "artifact_dir_case",
        mode="mock",
        resume=False,
        dry_run=False,
    )

    output_dir = Path(result["output_dir"])
    expected_paths = [
        output_dir / "run_config.yaml",
        output_dir / "research.db",
        output_dir / "audit.jsonl",
        output_dir / "evidence",
        output_dir / "evidence" / "literature.json",
        output_dir / "experiments",
        output_dir / "experiments" / "plan.json",
        output_dir / "experiments" / "stdout.log",
        output_dir / "experiments" / "stderr.log",
        output_dir / "experiments" / "results.json",
        output_dir / "claims.json",
        output_dir / "draft.md",
        output_dir / "validation_report.json",
        output_dir / "resource_usage.json",
        output_dir / "resource_report.md",
    ]
    missing = [str(path) for path in expected_paths if not path.exists()]
    assert missing == []


def test_cli_dry_run(tmp_path) -> None:
    config_path = (
        Path(__file__).resolve().parents[5]
        / "examples"
        / "scitrace_research"
        / "configs"
        / "mvp.yaml"
    )
    exit_code = cli_main(
        [
            "--topic",
            "Evidence-aware context engineering for autonomous research agents",
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "dry_run"),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "dry_run" / "run_config.yaml").exists()


def test_resume_continues_after_interruption(tmp_path) -> None:
    config = _config()
    config.setdefault("debug", {})
    config["debug"]["stop_after_stage"] = "literature_research"
    output_dir = tmp_path / "resume_case"

    try:
        run_pipeline(
            topic="Evidence-aware context engineering for autonomous research agents",
            config=config,
            output_dir=output_dir,
            mode="mock",
            resume=False,
            dry_run=False,
        )
    except WorkflowInterrupted:
        pass
    else:
        raise AssertionError("workflow should have interrupted")

    failed_usage = json.loads((output_dir / "resource_usage.json").read_text(encoding="utf-8"))
    assert failed_usage["extra"]["run_status"] == "failed"
    assert failed_usage["extra"]["failure_stage"] == "literature_research"
    assert failed_usage["extra"]["partial_report"] is True

    resumed = _config()
    result = run_pipeline(
        topic="Evidence-aware context engineering for autonomous research agents",
        config=resumed,
        output_dir=output_dir,
        mode="mock",
        resume=True,
        dry_run=False,
    )
    assert result["status"] == "success"
    usage = json.loads((output_dir / "resource_usage.json").read_text(encoding="utf-8"))
    assert usage["total_tokens"] == 0


def test_writer_metric_drift_is_blocked_by_gate(tmp_path) -> None:
    config = _config()
    config.setdefault("writer", {})
    config["writer"]["force_metric_drift"] = True
    config.setdefault("revision", {})
    config["revision"]["max_rounds"] = 0

    result = run_pipeline(
        topic="Evidence-aware context engineering for autonomous research agents",
        config=config,
        output_dir=tmp_path / "drift_case",
        mode="mock",
        resume=False,
        dry_run=False,
    )
    assert result["status"] == "failed"
    validation = json.loads((tmp_path / "drift_case" / "validation_report.json").read_text(encoding="utf-8"))
    codes = {item["code"] for item in validation["errors"]}
    assert "metric_mismatch" in codes


def test_gate_fails_after_revision_limit(tmp_path) -> None:
    config = _config()
    config.setdefault("debug", {})
    config["debug"]["force_persistent_gate_failure"] = True
    config.setdefault("revision", {})
    config["revision"]["max_rounds"] = 1

    result = run_pipeline(
        topic="Evidence-aware context engineering for autonomous research agents",
        config=config,
        output_dir=tmp_path / "revision_fail_case",
        mode="mock",
        resume=False,
        dry_run=False,
    )
    assert result["status"] == "failed"
    validation = json.loads((tmp_path / "revision_fail_case" / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["failed_reason"] == "max_revision_rounds_exceeded"


def test_extract_structured_references_from_real_style_array_payload() -> None:
    draft = """# Draft

## Structured References

```json
[
  {
    "id": "evidence_1",
    "type": "LiteratureEvidence"
  },
  {
    "id": "claim_1",
    "type": "Claim"
  },
  {
    "id": "artifact_1",
    "type": "ExperimentArtifact",
    "metric_name": "citation_support_rate",
    "metric_value": 0.87
  }
]
```
"""
    payload = _extract_structured_references(draft)
    assert payload == {
        "citations": ["evidence_1"],
        "claims": [{"claim_id": "claim_1"}],
        "reported_metrics": [
            {
                "artifact_id": "artifact_1",
                "metric_name": "citation_support_rate",
                "claimed_value": 0.87,
            }
        ],
    }


def test_complete_structured_references_backfills_missing_claims_and_metric() -> None:
    completed = _complete_structured_references(
        {"citations": ["evidence_1"]},
        literature_payload={"evidence": [{"id": "evidence_1"}, {"id": "evidence_2"}]},
        claim_payload={"claims": [{"id": "claim_1"}]},
        execution_payload={
            "artifact": {
                "id": "artifact_1",
                "metric_name": "citation_support_rate",
                "metric_value": 0.87,
            }
        },
    )

    assert completed["citations"] == ["evidence_1"]
    assert completed["claims"] == [{"claim_id": "claim_1"}]
    assert completed["reported_metrics"] == [
        {
            "artifact_id": "artifact_1",
            "metric_name": "citation_support_rate",
            "claimed_value": 0.87,
        }
    ]


def test_complete_structured_references_preserves_explicit_metric_drift() -> None:
    completed = _complete_structured_references(
        {
            "citations": ["evidence_1"],
            "claims": [{"claim_id": "claim_1"}],
            "reported_metrics": [
                {
                    "artifact_id": "artifact_1",
                    "metric_name": "citation_support_rate",
                    "claimed_value": 0.92,
                }
            ],
        },
        literature_payload={"evidence": [{"id": "evidence_1"}]},
        claim_payload={"claims": [{"id": "claim_1"}]},
        execution_payload={
            "artifact": {
                "id": "artifact_1",
                "metric_name": "citation_support_rate",
                "metric_value": 0.87,
            }
        },
    )

    assert completed["reported_metrics"][0]["claimed_value"] == 0.92

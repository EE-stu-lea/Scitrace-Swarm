"""Offline replay of deterministic gates for an unchanged Agent-generated draft.

Use this only after a gate implementation bug has been fixed. The original
failed manifest and validation are retained and the audit trail records the
recheck; no model call or draft rewrite occurs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from jiuwenswarm.agents.harness.research.rails import ScientificConsistencyGateRail
from jiuwenswarm.agents.harness.research.run_store import ResearchRunStore, utc_now_iso
from jiuwenswarm.agents.harness.research.reporting.resource_reporter import ResourceAuditSnapshot, ResourceReporter

from .workflow.paper_quality import audit_draft
from .workflow.pipeline import (
    _adaptive_baseline_rows,
    _complete_structured_references,
    _extract_structured_references,
    _primary_results_table_rows,
    _train_subsampling_rows,
    load_yaml_config,
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def revalidate(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    store = ResearchRunStore(run_dir)
    original_manifest = _read_json(run_dir / "run_manifest.json")
    original_validation = _read_json(run_dir / "validation_report.json")
    if original_manifest.get("status") != "failed" or original_manifest.get("failure_stage") != "consistency_loop":
        raise ValueError("Only a failed consistency-loop run can be revalidated")
    if original_manifest.get("mode") != "real_team" or not original_manifest.get("strict_real"):
        raise ValueError("Revalidation requires a strict real Team run")
    checkpoint = store.get_stage("consistency_loop")
    if checkpoint is None or checkpoint.status != "completed":
        raise ValueError("No completed consistency-loop checkpoint")
    draft_path = run_dir / "draft.md"
    draft_bytes = draft_path.read_bytes()
    draft_hash = hashlib.sha256(draft_bytes).hexdigest()
    if checkpoint.payload.get("draft") != draft_bytes.decode("utf-8"):
        raise ValueError("The Agent draft differs from its durable checkpoint")
    if original_validation.get("draft_sha256") != draft_hash:
        raise ValueError("The failed validation does not bind to this Agent draft")

    config = load_yaml_config(run_dir / "run_config.yaml")
    literature = _read_json(run_dir / "evidence" / "literature.json")
    claims = _read_json(run_dir / "claims.json")
    results = _read_json(run_dir / "experiments" / "results.json")
    structured = _complete_structured_references(
        _extract_structured_references(draft_bytes.decode("utf-8")),
        literature_payload=literature,
        claim_payload=claims,
        execution_payload=results,
    )
    structured["draft_text"] = draft_bytes.decode("utf-8")
    scientific = ScientificConsistencyGateRail(
        enabled=True, output_root=run_dir, block_on_error=False
    ).check_payload(structured, session_id="")
    quality = audit_draft(
        draft_bytes.decode("utf-8"),
        evidence=literature["evidence"],
        config=config.get("paper_quality") or {},
        table_rows=_primary_results_table_rows(results)
        if (config.get("paper_quality") or {}).get("require_results_table") else None,
        adaptive_rows=_adaptive_baseline_rows(results)
        if (config.get("paper_quality") or {}).get("require_adaptive_baseline_table") else None,
        stability_rows=_train_subsampling_rows(results)
        if (config.get("paper_quality") or {}).get("require_train_subsampling_summary") else None,
        results=results,
    )
    if not scientific["ok"] or not quality["ok"]:
        raise ValueError(json.dumps({"scientific": scientific["errors"], "quality": quality["errors"]}))

    calls = []
    for line in (run_dir / "model_calls.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("stage") != "diagnostics":
            if not all(isinstance(item.get(key), int) for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
                raise ValueError("Workflow model call lacks durable token usage")
            calls.append(item)
    if len(calls) != original_manifest.get("model_call_count"):
        raise ValueError("Model call log does not match the failed manifest")

    (run_dir / "revalidation_original_manifest.json").write_text(
        json.dumps(original_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "revalidation_original_validation.json").write_text(
        json.dumps(original_validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    now = utc_now_iso()
    report = {
        "ok": True,
        "errors": [],
        "draft_sha256": draft_hash,
        "revalidation": {
            "time": now,
            "reason": "Corrected false-positive check for negated continuous-depth statements",
            "original_status": "failed",
            "original_failure_stage": "consistency_loop",
            "model_calls_during_revalidation": 0,
            "scientific_gate": scientific,
            "paper_quality": quality,
        },
        "attempts": original_validation.get("attempts", []),
    }
    (run_dir / "paper_quality_report.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    snapshot = ResourceAuditSnapshot(session_id=f"scitrace-team-{run_dir.name}", role="native_team")
    snapshot.model_calls = len(calls)
    snapshot.invoke_count = len(calls)
    snapshot.input_tokens = sum(item["prompt_tokens"] for item in calls)
    snapshot.output_tokens = sum(item["completion_tokens"] for item in calls)
    snapshot.total_tokens = sum(item["total_tokens"] for item in calls)
    snapshot.retries = sum(item.get("status") == "error" for item in calls)
    for item in calls:
        name = str(item.get("model_name") or "unknown")
        snapshot.model_calls_by_name[name] = snapshot.model_calls_by_name.get(name, 0) + 1
        snapshot.model_tokens_by_name[name] = snapshot.model_tokens_by_name.get(name, 0) + item["total_tokens"]
    snapshot.total_runtime_seconds = (
        datetime.fromisoformat(original_manifest["end_time"])
        - datetime.fromisoformat(original_manifest["start_time"])
    ).total_seconds()
    snapshot.extra.update({
        "workflow": "scitrace_mvp",
        "workflow_model_calls": len(calls),
        "revalidated_without_model_calls": True,
        "original_run_exceeded_local_budget": snapshot.total_tokens > int(
            (config.get("budget") or {}).get("max_total_tokens") or 0
        ),
    })
    ResourceReporter(
        run_dir, json_filename="resource_usage.json", markdown_filename="resource_report.md", nest_by_session=False
    ).write_snapshot(snapshot)
    manifest = dict(original_manifest)
    manifest.update({
        "status": "success",
        "failure_stage": "",
        "draft_sha256": draft_hash,
        "revalidated_at": now,
        "revalidated_from_failure": True,
        "original_failure_stage": "consistency_loop",
        "budget_overrun": f"Original run used {snapshot.total_tokens} tokens; local guard was "
        f"{(config.get('budget') or {}).get('max_total_tokens')}",
    })
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    store.set_stage("consistency_loop", status="completed", payload={"draft": draft_bytes.decode("utf-8"), "validation_report": report})
    store.set_metadata("status", "success")
    store.append_audit_event("offline_gate_revalidation", {
        "draft_sha256": draft_hash,
        "original_status": "failed",
        "original_failure_stage": "consistency_loop",
        "model_calls_during_revalidation": 0,
        "total_original_tokens": snapshot.total_tokens,
    })
    if hashlib.sha256(draft_path.read_bytes()).hexdigest() != draft_hash:
        raise RuntimeError("Agent draft changed during revalidation")
    return {"status": "success", "run_dir": str(run_dir), "draft_sha256": draft_hash,
            "total_tokens": snapshot.total_tokens, "model_calls": len(calls)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(revalidate(args.run_dir), indent=2))

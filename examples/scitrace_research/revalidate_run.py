"""Offline replay of deterministic gates for an unchanged Agent-generated draft.

Use this only after a gate implementation bug has been fixed. The original
failed manifest and validation are retained and the audit trail records the
recheck; no model call or draft rewrite occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3

from jiuwenswarm.agents.harness.research.rails import ScientificConsistencyGateRail
from jiuwenswarm.agents.harness.research.run_store import ResearchRunStore, utc_now_iso

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


_PARSING_FALSE_POSITIVES = {
    "paper_method_overview_missing": ("method overview", "caption"),
    "paper_query_risk_figure_missing": ("query risk", "caption"),
    "paper_query_outcome_row_missing": ("fiqa", "outcome"),
    "paper_budget_effect_figure_missing": ("budget effect", "caption"),
    "paper_results_table_caption_missing": ("table", "caption"),
    "paper_typesetting_invalid": ("math",),
}


def _reviewed_partial_evidence(run_dir: Path, checkpoint, draft_hash: str,
                               original_manifest: dict) -> dict:
    """Accept a failed loop only when its latest paid review binds this draft.

    The branch is deliberately narrow: it repairs validator false positives,
    never turns a model-reported scientific concern into an automatic pass.
    """
    error = str((checkpoint.payload or {}).get("error") or "")
    if checkpoint.status != "failed" or not re.fullmatch(
        r"model call budget exhausted before stage revision: \d+", error
    ):
        raise ValueError("No completed consistency-loop checkpoint or eligible reviewed draft")
    reviews = list((run_dir / "reviews").glob("round_*.json"))
    if not reviews:
        raise ValueError("No paid review binds the partial-stage draft")
    def round_number(path: Path) -> int:
        match = re.fullmatch(r"round_(\d+)\.json", path.name)
        return int(match.group(1)) if match else -1
    latest_path = max(reviews, key=round_number)
    latest = _read_json(latest_path)
    if round_number(latest_path) < 0 or latest.get("draft_sha256") != draft_hash:
        raise ValueError("Latest paid review does not bind the current Agent draft")
    current_review = _read_json(run_dir / "review_report.json")
    if current_review != latest:
        raise ValueError("Latest review report differs from its round artifact")
    original_quality = _read_json(run_dir / "paper_quality_report.json")
    old_errors = original_quality.get("errors") or []
    old_codes = [str(item.get("code") or "") for item in old_errors if isinstance(item, dict)]
    if not old_codes or len(old_codes) != len(old_errors) or set(old_codes) - _PARSING_FALSE_POSITIVES.keys():
        raise ValueError("Partial-stage draft has errors beyond corrected parsing checks")
    findings = latest.get("findings") or []
    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("Malformed paid review finding")
        if str(finding.get("severity") or "").casefold() != "error":
            continue
        message = str(finding.get("message") or "").casefold()
        if not any(all(term in message for term in _PARSING_FALSE_POSITIVES[code]) for code in old_codes):
            raise ValueError("Paid review contains an independent error; do not revalidate offline")
    calls = [json.loads(line) for line in (run_dir / "model_calls.jsonl").read_text().splitlines()
             if line.strip()]
    workflow_calls = [item for item in calls if item.get("stage") != "diagnostics"]
    if len(workflow_calls) != original_manifest.get("model_call_count"):
        raise ValueError("Model call log does not match the failed manifest")
    if not workflow_calls or (workflow_calls[-1].get("stage"), workflow_calls[-1].get("status")) != ("review", "success"):
        raise ValueError("The latest paid operation was not a completed review")
    return {"round": round_number(latest_path), "old_quality_error_codes": old_codes,
            "review_draft_sha256": draft_hash, "review_error_count": sum(
                str(item.get("severity") or "").casefold() == "error" for item in findings)}


def revalidate(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    store = ResearchRunStore(run_dir)
    original_manifest = _read_json(run_dir / "run_manifest.json")
    original_validation_path = run_dir / "validation_report.json"
    original_validation = _read_json(original_validation_path) if original_validation_path.is_file() else None
    if original_manifest.get("status") != "failed" or original_manifest.get("failure_stage") != "consistency_loop":
        raise ValueError("Only a failed consistency-loop run can be revalidated")
    if original_manifest.get("mode") != "real_team" or not original_manifest.get("strict_real"):
        raise ValueError("Revalidation requires a strict real Team run")
    checkpoint = store.get_stage("consistency_loop")
    if checkpoint is None:
        raise ValueError("No consistency-loop checkpoint")
    draft_path = run_dir / "draft.md"
    draft_bytes = draft_path.read_bytes()
    draft_hash = hashlib.sha256(draft_bytes).hexdigest()
    partial_evidence = None
    if checkpoint.status == "completed":
        if checkpoint.payload.get("draft") != draft_bytes.decode("utf-8"):
            raise ValueError("The Agent draft differs from its durable checkpoint")
        if original_validation is None or original_validation.get("draft_sha256") != draft_hash:
            raise ValueError("The failed validation does not bind to this Agent draft")
    else:
        partial_evidence = _reviewed_partial_evidence(run_dir, checkpoint, draft_hash, original_manifest)

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
    from .workflow.query_diagnostics import build_query_diagnostics
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
        query_diagnostics=build_query_diagnostics(run_dir / "experiments" / "route_b2_artifacts")
        if (config.get("paper_quality") or {}).get("require_query_outcome_table") else None,
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
    total_tokens = sum(item["total_tokens"] for item in calls)
    resource_path = run_dir / "resource_usage.json"
    resource_bytes = resource_path.read_bytes()
    resource_usage = json.loads(resource_bytes)
    if (resource_usage.get("total_tokens") != total_tokens
            or resource_usage.get("model_calls") != len(calls)):
        raise ValueError("Existing resource report does not match durable model calls")

    shutil.copyfile(run_dir / "run_manifest.json", run_dir / "revalidation_original_manifest.json")
    shutil.copyfile(draft_path, run_dir / "revalidation_original_draft.md")
    if original_validation_path.is_file():
        shutil.copyfile(original_validation_path, run_dir / "revalidation_original_validation.json")
    shutil.copyfile(run_dir / "paper_quality_report.json", run_dir / "revalidation_original_paper_quality.json")
    if partial_evidence is not None:
        shutil.copyfile(run_dir / "review_report.json", run_dir / "revalidation_original_review.json")
    source_db = run_dir / "research.db"
    with sqlite3.connect(f"{source_db.as_uri()}?mode=ro", uri=True) as original_db:
        with sqlite3.connect(run_dir / "revalidation_original_research.db") as saved_db:
            original_db.backup(saved_db)
    now = utc_now_iso()
    report = {
        "ok": True,
        "errors": [],
        "draft_sha256": draft_hash,
        "revalidation": {
            "time": now,
            "reason": "Replayed corrected deterministic gates against the unchanged Agent draft",
            "original_status": "failed",
            "original_failure_stage": "consistency_loop",
            "model_calls_during_revalidation": 0,
            "scientific_gate": scientific,
            "paper_quality": quality,
            "resource_report_sha256": hashlib.sha256(resource_bytes).hexdigest(),
            "reviewed_partial_stage": partial_evidence,
        },
        "attempts": (original_validation or {}).get("attempts", []),
    }
    (run_dir / "paper_quality_report.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Revalidation makes no model call. Preserve the original resource report
    # byte-for-byte, including observed provider receipts and resumed-run scope.
    token_limit = int((config.get("budget") or {}).get("max_total_tokens") or 0)
    exceeded = bool(token_limit and total_tokens > token_limit)
    manifest = dict(original_manifest)
    manifest.update({
        "status": "success",
        "failure_stage": "",
        "draft_sha256": draft_hash,
        "revalidated_at": now,
        "revalidated_from_failure": True,
        "original_failure_stage": "consistency_loop",
        "budget_overrun": (
            f"Original run used {total_tokens} tokens; local guard was {token_limit}"
            if exceeded else ""
        ),
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
        "total_original_tokens": total_tokens,
        "reviewed_partial_stage": partial_evidence,
    })
    if hashlib.sha256(draft_path.read_bytes()).hexdigest() != draft_hash:
        raise RuntimeError("Agent draft changed during revalidation")
    if resource_path.read_bytes() != resource_bytes:
        raise RuntimeError("Original resource report changed during revalidation")
    return {"status": "success", "run_dir": str(run_dir), "draft_sha256": draft_hash,
            "total_tokens": total_tokens, "model_calls": len(calls)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(revalidate(args.run_dir), indent=2))

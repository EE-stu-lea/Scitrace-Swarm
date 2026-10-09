"""Validate and audit runtime-only recovery without invalidating scientific checkpoints.

Configuration comparison is symmetric: additions and removals are rejected unless
explicitly covered below. Error messages contain key paths, never config values.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any


_MISSING = object()
_DIRECT_SPECIALIST_RECOVERY = ["experiment_plan", "draft", "review", "revision"]
_GLOBAL_CEILINGS = {
    ("team", "invocation_timeout_seconds"),
    ("team", "max_invocation_provider_calls"),
    ("team", "max_invocation_total_tokens"),
    ("team", "max_invocation_prompt_bytes"),
    ("budget", "max_total_tokens"),
    ("budget", "max_model_calls"),
    ("budget", "max_estimated_cost"),
}
_SNAPSHOT_FILES = (
    "run_config.yaml", "run_manifest.json", "resource_usage.json",
    "console.log", "paper_export.log", "team_events.jsonl", "model_calls.jsonl",
    "provider_usage.jsonl", "model_diagnostics.json", "team_prompt_audit.jsonl",
    "team_capability_audit.jsonl", "team_manifest.json", "audit.jsonl",
)


def _is_ceiling(path: tuple[str, ...]) -> bool:
    return (
        path in _GLOBAL_CEILINGS
        or (len(path) == 3 and path[:2] == ("team", "stage_timeout_seconds"))
        or (len(path) == 4 and path[:2] == ("team", "stage_resource_limits")
            and path[-1] in {"max_total_tokens", "max_provider_calls", "max_prompt_bytes"})
        or (len(path) == 5 and path[:3] == ("modes", "team", "agents")
            and path[3] in {"leader", "teammate"}
            and path[4] in {"max_iterations", "completion_timeout"})
    )


def _finite_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def validate_resume_config(
    previous: dict[str, Any], current: dict[str, Any], *, topic: str
) -> list[dict[str, Any]]:
    """Return approved changes; reject changes to scientific inputs or model config.

    Only numeric ceiling increases and audited routing/capability fixes are
    compatible with existing checkpoints.
    The run-added topic must match exactly. No credentials enter the returned audit.
    """
    old = deepcopy(previous)
    new = deepcopy(current)
    if old.pop("topic", _MISSING) != topic or new.pop("topic", topic) != topic:
        raise ValueError("Resume topic differs from the existing run.")
    changes: list[dict[str, Any]] = []

    def visit(before: Any, after: Any, path: tuple[str, ...]) -> None:
        if isinstance(before, dict) and isinstance(after, dict):
            for key in sorted(before.keys() | after.keys()):
                visit(before.get(key, _MISSING), after.get(key, _MISSING), (*path, str(key)))
            return
        if before is not _MISSING and after is not _MISSING and type(before) is type(after) and before == after:
            return
        profile_upgrade = (
            path == ("team", "stage_capability_profile")
            and (before is _MISSING or before == "default") and after == "prompt_only"
        )
        specialist_recovery = (
            path == ("team", "direct_specialist_stages")
            and before is _MISSING and after == _DIRECT_SPECIALIST_RECOVERY
        )
        ceiling_increase = (
            _is_ceiling(path) and _finite_number(before) and _finite_number(after)
            and after >= before and after > 0
        )
        if not (profile_upgrade or specialist_recovery or ceiling_increase):
            raise ValueError(
                "Resume config change is not a supported runtime-only recovery at "
                + ".".join(path) + "; start a new run for scientific or model changes."
            )
        changes.append({"path": ".".join(path), "before": None if before is _MISSING else before,
                        "after": after, "kind": ("capability_profile" if profile_upgrade else
                                                 "direct_specialist_recovery" if specialist_recovery else
                                                 "ceiling_increase")})

    visit(old, new, ())
    return changes


def _read_config(path: Path) -> dict[str, Any]:
    from ruamel.yaml import YAML

    value = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Resume configuration must be a mapping.")
    return value


def inspect_resume(run_dir: Path, config_path: Path, *, topic: str) -> dict[str, Any]:
    """Read-only preflight, also used by --check; never updates original run files."""
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") not in {"failed", "interrupted"} or manifest.get("mode") != "real_team":
        raise ValueError("Resume requires an existing failed/interrupted real Team run.")
    previous_path = run_dir / "run_config.yaml"
    changes = validate_resume_config(_read_config(previous_path), _read_config(config_path), topic=topic)
    # Read a temporary SQLite copy: a WAL-mode read can otherwise update the
    # original -shm file during `--check`. Include its WAL to see last commits.
    with tempfile.TemporaryDirectory(prefix="scitrace-resume-inspect-") as temp:
        source = run_dir / "research.db"
        copied = Path(temp) / "research.db"
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(source) + suffix)
            if candidate.is_file():
                shutil.copyfile(candidate, Path(str(copied) + suffix))
        with sqlite3.connect(f'{copied.as_uri()}?mode=ro', uri=True) as conn:
            checkpoints = [{"stage": stage, "status": status}
                           for stage, status in conn.execute("select stage,status from run_stage order by rowid")]
    return {
        "policy": "runtime-only-recovery-v1", "config_changes": changes,
        "previous_config_sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
        "next_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "checkpoints": checkpoints,
    }


def snapshot_resume(run_dir: Path, config_path: Path, snapshot_dir: Path, *, topic: str) -> dict[str, Any]:
    """Preserve the pre-resume evidence before the pipeline rewrites its manifest.

    The workflow appends model calls and events; log copies additionally preserve
    the exact old bytes. The database remains the authoritative checkpoint store.
    """
    run_dir = run_dir.resolve()
    snapshot_dir = snapshot_dir.resolve()
    if snapshot_dir.parent != run_dir / "resume_history":
        raise ValueError("Resume snapshots must be directly inside the run's resume_history directory.")
    audit = inspect_resume(run_dir, config_path, topic=topic)
    snapshot_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    files: dict[str, str] = {}
    for name in _SNAPSHOT_FILES:
        source = run_dir / name
        if source.is_file():
            target = snapshot_dir / name
            shutil.copyfile(source, target)
            target.chmod(0o600)
            files[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    # SQLite's backup API folds an uncheckpointed WAL into one consistent
    # database image. Copying research.db alone would omit recent checkpoints.
    db_snapshot = snapshot_dir / "research.db"
    with sqlite3.connect(f'{(run_dir / "research.db").as_uri()}?mode=ro', uri=True) as source_db:
        with sqlite3.connect(db_snapshot) as target_db:
            source_db.backup(target_db)
    db_snapshot.chmod(0o600)
    files["research.db"] = hashlib.sha256(db_snapshot.read_bytes()).hexdigest()
    audit["preserved_files_sha256"] = files
    audit["created_at"] = datetime.now(timezone.utc).isoformat()
    audit["note"] = "Scientific and writing inputs are unchanged; failed stages may be retried under the listed runtime changes."
    audit_path = snapshot_dir / "resume_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_path.chmod(0o600)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--snapshot-dir", type=Path)
    args = parser.parse_args()
    try:
        audit = (snapshot_resume(args.run_dir, args.config, args.snapshot_dir, topic=args.topic)
                 if args.snapshot_dir else inspect_resume(args.run_dir, args.config, topic=args.topic))
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, str(exc) + "\n")
    for checkpoint in audit["checkpoints"]:
        print(f'Checkpoint: {checkpoint["stage"]}: {checkpoint["status"]}')
    for change in audit["config_changes"]:
        print(f'Approved runtime recovery: {change["path"]}: {change["before"]} -> {change["after"]}')
    if args.snapshot_dir:
        print(f'Resume audit: {args.snapshot_dir / "resume_audit.json"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

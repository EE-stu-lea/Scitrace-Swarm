from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ExperimentExecutorError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExperimentExecutionResult:
    plan_path: Path
    stdout_path: Path
    stderr_path: Path
    results_path: Path
    exit_code: int
    timed_out: bool
    duration_seconds: float
    command: list[str]
    seed: int | None
    output_sha256: str
    payload: dict[str, Any]


class SafeExperimentExecutor:
    """Minimal structured experiment runner with workspace and timeout guards."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        timeout_seconds: int = 120,
        allowed_executables: list[str] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = max(1, int(timeout_seconds))
        default_allowed = {
            Path(sys.executable).name.lower(),
            "python",
            "python.exe",
        }
        self.allowed_executables = {
            item.lower() for item in (allowed_executables or list(default_allowed))
        } | default_allowed

    def execute(
        self,
        plan: dict[str, Any],
        *,
        output_dir: str | Path,
    ) -> ExperimentExecutionResult:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        plan_path = output_root / "plan.json"
        stdout_path = output_root / "stdout.log"
        stderr_path = output_root / "stderr.log"
        results_path = output_root / "results.json"
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

        command = self._build_command(plan, results_path)
        seed = plan.get("seed")
        started = time.perf_counter()
        timed_out = False
        exit_code = -1

        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.workspace,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    timeout=self.timeout_seconds,
                    check=False,
                    text=True,
                )
                exit_code = int(completed.returncode)
            except subprocess.TimeoutExpired:
                timed_out = True
                stderr_handle.write(f"Experiment timed out after {self.timeout_seconds} seconds.\n")
                exit_code = -9

        duration_seconds = max(0.0, time.perf_counter() - started)
        if timed_out:
            raise ExperimentExecutorError(f"experiment timed out after {self.timeout_seconds} seconds")
        if exit_code != 0:
            raise ExperimentExecutorError(f"experiment failed with exit code {exit_code}")
        if not results_path.exists():
            raise ExperimentExecutorError("experiment did not write results.json")

        payload = json.loads(results_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ExperimentExecutorError("results.json must contain a JSON object")
        payload.setdefault("command", command)
        payload.setdefault("seed", seed)
        payload.setdefault("generated_at", _utc_now_iso())
        results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

        return ExperimentExecutionResult(
            plan_path=plan_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            results_path=results_path,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_seconds=duration_seconds,
            command=command,
            seed=seed,
            output_sha256=_sha256_file(results_path),
            payload=payload,
        )

    def _build_command(self, plan: dict[str, Any], results_path: Path) -> list[str]:
        kind = str(plan.get("kind") or "python_inline").strip().lower()
        if kind != "python_inline":
            raise ExperimentExecutorError(f"unsupported experiment kind: {kind}")
        script_path = self._resolve_script_path(plan)
        command = [
            str(sys.executable),
            str(script_path),
            "--output",
            str(results_path),
        ]
        seed = plan.get("seed")
        if seed not in (None, ""):
            command.extend(["--seed", str(seed)])
        extra_args = plan.get("args") or []
        if not isinstance(extra_args, list):
            raise ExperimentExecutorError("experiment args must be a list")
        command.extend(str(item) for item in extra_args)
        self._validate_command(command)
        return command

    def _resolve_script_path(self, plan: dict[str, Any]) -> Path:
        script_path = Path(str(plan.get("script_path") or "")).expanduser()
        if not script_path.is_absolute():
            script_path = (self.workspace / script_path).resolve()
        if not script_path.exists():
            raise ExperimentExecutorError(f"script does not exist: {script_path}")
        if self.workspace not in script_path.parents and script_path != self.workspace:
            raise ExperimentExecutorError(f"script is outside workspace: {script_path}")
        return script_path

    def _validate_command(self, command: list[str]) -> None:
        if not command:
            raise ExperimentExecutorError("empty experiment command")
        executable = Path(command[0]).name.lower()
        if executable not in self.allowed_executables:
            raise ExperimentExecutorError(f"executable is not allowed: {command[0]}")
        for item in command[1:]:
            if item.startswith("-"):
                continue
            path = Path(item)
            if path.is_absolute() and self.workspace not in path.resolve().parents and path.resolve() != self.workspace:
                raise ExperimentExecutorError(f"path argument is outside workspace: {item}")


__all__ = ["ExperimentExecutionResult", "ExperimentExecutorError", "SafeExperimentExecutor"]

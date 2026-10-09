"""Safe progress for long calls: checkpoint state and observed receipt totals."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import time


def snapshot(run_dir: Path) -> dict:
    stages = []
    database = run_dir / "research.db"
    if database.exists():
        try:
            with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True, timeout=1) as connection:
                stages = list(connection.execute("select stage,status from run_stage"))
        except sqlite3.Error:
            pass
    current = next((stage for stage, state in stages if state == "running"), "starting / between stages")
    completed = sum(state == "completed" for _, state in stages)
    # Durable model_calls are stage receipts, not estimates or in-flight usage.
    tokens = calls = 0
    log = run_dir / "model_calls.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get("stage") != "diagnostics":
                calls += 1
                value = item.get("total_tokens", 0)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    tokens += value
    return {"current": current, "completed": completed, "stage_calls": calls, "recorded_tokens": tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    while True:
        time.sleep(30)
        try:
            os.kill(args.parent_pid, 0)
        except ProcessLookupError:
            break
        data = snapshot(args.run_dir)
        print(f"[progress {int(time.monotonic()-started)}s] stage={data['current']}; "
              f"completed={data['completed']}; recorded tokens={data['recorded_tokens']:,} "
              "(completed stage receipts; in-flight usage pending)", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from jiuwenswarm.agents.harness.research.model_diagnostics import diagnose_model_connection

from .workflow.pipeline import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose JiuwenSwarm real model connectivity for SciTrace-Swarm.")
    parser.add_argument("--config", required=True, help="Path to the workflow YAML config.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ = load_yaml_config(args.config)
    try:
        payload = diagnose_model_connection()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": str(exc),
                    "config": str(Path(args.config).resolve()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    payload["status"] = "ok"
    payload["workflow_config"] = str(Path(args.config).resolve())
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

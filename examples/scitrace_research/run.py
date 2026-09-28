from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .workflow.pipeline import WorkflowInterrupted, load_yaml_config, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SciTrace-Swarm Phase 3 MVP workflow.")
    parser.add_argument("--topic", required=True, help="Research topic or question.")
    parser.add_argument("--config", required=True, help="Path to the workflow YAML config.")
    parser.add_argument("--output", required=True, help="Output directory for one run.")
    parser.add_argument(
        "--mode",
        choices=["mock", "real", "mock_team", "real_team"],
        help="Override workflow mode.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from research.db stage state.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and workflow without execution.")
    parser.add_argument("--strict-real", action="store_true", help="Fail immediately on any real-mode fallback or provider error.")
    parser.add_argument("--no-mock-fallback", action="store_true", help="Disable all fixture/mock fallback in real mode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_yaml_config(args.config)
    try:
        result = run_pipeline(
            topic=args.topic,
            config=config,
            output_dir=args.output,
            mode=args.mode,
            resume=bool(args.resume),
            dry_run=bool(args.dry_run),
            strict_real=bool(args.strict_real),
            no_mock_fallback=bool(args.no_mock_fallback),
        )
    except WorkflowInterrupted as exc:
        print(
            json.dumps(
                {
                    "status": "interrupted",
                    "message": str(exc),
                    "output_dir": str(Path(args.output).resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": str(exc),
                    "output_dir": str(Path(args.output).resolve()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    # openjiuwen 的 Team 引擎会遗留非 daemon 后台任务（TaskScheduler/EventBus），
    # 用 os._exit 强制退出，避免 CLI 脚本在主流程完成后被残留线程挂住。
    import os

    os._exit(main())

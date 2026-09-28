from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pipeline import load_yaml_config, run_pipeline


def load_case(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_scitrace_mvp(
    case: dict[str, Any],
    *,
    output_root: str | Path,
    session_id: str = "scitrace-mvp",
) -> dict[str, Any]:
    _ = session_id
    config_path = Path(__file__).resolve().parents[1] / "configs" / "mvp.yaml"
    config = load_yaml_config(config_path)
    if "literature" in case:
        fixture_path = Path(output_root) / "tmp_literature_fixture.json"
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(case["literature"], ensure_ascii=False, indent=2), encoding="utf-8")
        config.setdefault("literature", {})
        config["literature"]["provider"] = "mock"
        config["literature"]["fixture_path"] = str(fixture_path)
    if "experiments" in case and case["experiments"]:
        # The experiment stage now computes its metric from the real retrieval
        # benchmark, so any metric_value carried in a fixture case is no longer
        # injected as a mock value.
        pass
    return run_pipeline(
        topic=str(case.get("topic") or "").strip(),
        config=config,
        output_dir=output_root,
        mode="mock",
        resume=False,
        dry_run=False,
    )


__all__ = ["load_case", "run_scitrace_mvp"]

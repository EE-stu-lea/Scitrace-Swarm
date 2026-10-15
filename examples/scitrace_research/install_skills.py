"""Install the SciTrace local Skill packages into the global agent library.

Idempotent: a skill that is already present in the library is skipped instead
of being overwritten, and skills_state.json is never duplicated. Run with the
project interpreter from any working directory:

    python examples/scitrace_research/install_skills.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jiuwenswarm.agents.harness.research.skill_context import (  # noqa: E402
    install_skills_to_library,
)

_SKILL_NAMES = ("citation-graph-tracer", "pre-submit-checklist", "stats-notation")


def main() -> int:
    skills_root = Path(__file__).resolve().parent / "skills"
    missing = [name for name in _SKILL_NAMES if not (skills_root / name / "SKILL.md").is_file()]
    if missing:
        print("missing SKILL.md for: " + ", ".join(missing), file=sys.stderr)
        return 1
    report = install_skills_to_library([skills_root / name for name in _SKILL_NAMES])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"installed={len(report['installed'])} skipped={len(report['skipped'])} -> {report['target']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

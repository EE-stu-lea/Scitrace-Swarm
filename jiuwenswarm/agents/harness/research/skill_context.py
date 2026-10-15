"""Deterministic inlining of local Skill packages into bounded stage prompts.

Draft/review/revision are direct-specialist stages with exactly one model
request and no tool loop, so a Skill cannot be discovered at run time: its
guidance must be inlined into the assembled prompt by deterministic code.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
from typing import Any

logger = logging.getLogger(__name__)

SKILL_FILE_NAME = "SKILL.md"

# Token discipline: an earlier prompt-enrichment experiment inflated a stage
# prompt roughly 24x, so every inlined Skill body is hard-capped. 2600 chars
# keeps a two-Skill stage (the SciTrace default) below ~5.2k extra chars while
# still leaving room for an actionable checklist body.
DEFAULT_MAX_CHARS_PER_SKILL = 2600

_SCITRACE_SKILLS_MARKETPLACE = "scitrace"


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return flat ``key: value`` frontmatter fields plus the markdown body."""
    stripped = text.lstrip("\ufeff")
    if not stripped.startswith("---"):
        raise ValueError("SKILL.md must open with a `---` frontmatter block")
    lines = stripped.splitlines()
    closing = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing is None:
        raise ValueError("SKILL.md frontmatter is not closed by a `---` line")
    fields: dict[str, str] = {}
    for line in lines[1:closing]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        fields[key.strip()] = value.strip().strip("\"'")
    body = "\n".join(lines[closing + 1:]).strip()
    return fields, body


def load_skill_package(skill_root: Path) -> dict[str, str]:
    """Validate and load one Skill package directory.

    Raises:
        FileNotFoundError: the directory or its SKILL.md is missing.
        ValueError: the frontmatter lacks required fields or its declared
            ``name`` does not equal the package directory name.
    """
    root = Path(skill_root)
    skill_md = root / SKILL_FILE_NAME
    if not skill_md.is_file():
        raise FileNotFoundError(f"skill package is missing SKILL.md: {root}")
    fields, body = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
    name = fields.get("name", "").strip()
    description = fields.get("description", "").strip()
    if not name:
        raise ValueError(f"SKILL.md frontmatter must declare `name`: {root}")
    if name != root.name:
        raise ValueError(f"skill name mismatch: frontmatter `{name}` != directory `{root.name}`")
    if not description:
        raise ValueError(f"SKILL.md frontmatter must declare `description`: {root}")
    return {"name": name, "description": description, "body": body}


def build_skills_block(
    skill_names: list[str],
    skills_root: Path,
    max_chars_per_skill: int = DEFAULT_MAX_CHARS_PER_SKILL,
) -> str:
    """Assemble a deterministic markdown block from named Skill packages.

    Skills are sorted by name and each body is capped at
    ``max_chars_per_skill`` characters so repeated runs inline byte-identical
    guidance. Unknown or invalid skill names are logged and skipped instead
    of failing the stage prompt.
    """
    root = Path(skills_root)
    sections: list[str] = []
    for name in sorted({str(item).strip() for item in skill_names if str(item).strip()}):
        try:
            package = load_skill_package(root / name)
        except (OSError, ValueError) as exc:
            logger.warning("skipping unavailable skill `%s`: %s", name, exc)
            continue
        body = package["body"]
        if max_chars_per_skill > 0 and len(body) > max_chars_per_skill:
            body = (
                body[:max_chars_per_skill].rstrip()
                + f"\n[...truncated at {max_chars_per_skill} characters...]"
            )
        sections.append(
            f"### Skill: {package['name']}\n"
            f"{package['description']}\n\n"
            f"{body}"
        )
    if not sections:
        return ""
    return (
        "## Inlined SciTrace Skills\n\n"
        "The stage owner must follow these skill checklists and notation rules exactly.\n\n"
        + "\n\n".join(sections)
    )


def install_skills_to_library(
    skill_roots: list[Path],
    target: Path | None = None,
) -> dict[str, Any]:
    """Copy local Skill packages into the global library and register them.

    Idempotent: an already installed skill name is skipped instead of being
    overwritten. Registration follows the skills_state.json schema used by
    builtin skills, with ``source="local"`` and ``marketplace="scitrace"``.
    """
    if target is None:
        from jiuwenswarm.common.utils import get_agent_skills_dir

        target = get_agent_skills_dir()
    library = Path(target)
    library.mkdir(parents=True, exist_ok=True)
    state_path = library / "skills_state.json"
    state: dict[str, Any] = {"marketplaces": [], "installed_plugins": [], "local_skills": []}
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state = loaded
        except (OSError, ValueError):
            logger.warning("unreadable skills_state.json; recreating it: %s", state_path)
    if not isinstance(state.get("installed_plugins"), list):
        state["installed_plugins"] = []
    installed_at = datetime.now(timezone.utc).isoformat()
    installed: list[str] = []
    skipped: list[str] = []
    existing_names = {
        str(item.get("name") or "")
        for item in state["installed_plugins"]
        if isinstance(item, dict)
    }
    for skill_root in skill_roots:
        package = load_skill_package(Path(skill_root))
        destination = library / package["name"]
        if destination.exists():
            skipped.append(package["name"])
        else:
            shutil.copytree(Path(skill_root), destination)
            installed.append(package["name"])
        if package["name"] not in existing_names:
            state["installed_plugins"].append({
                "name": package["name"],
                "marketplace": _SCITRACE_SKILLS_MARKETPLACE,
                "version": "",
                "commit": "",
                "source": "local",
                "installed_at": installed_at,
            })
            existing_names.add(package["name"])
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "target": str(library),
        "state_file": str(state_path),
        "installed": installed,
        "skipped": skipped,
    }


__all__ = [
    "DEFAULT_MAX_CHARS_PER_SKILL",
    "build_skills_block",
    "install_skills_to_library",
    "load_skill_package",
]

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ._stubs import install_openjiuwen_stubs

install_openjiuwen_stubs()

from jiuwenswarm.agents.harness.research.skill_context import (
    DEFAULT_MAX_CHARS_PER_SKILL,
    build_skills_block,
    install_skills_to_library,
    load_skill_package,
)
from jiuwenswarm.agents.harness.research.team_contract import get_team_stage_contract
from jiuwenswarm.agents.harness.research.team_runtime import JiuwenSwarmResearchTeamRuntime

REPO_ROOT = Path(__file__).resolve().parents[5]
SKILLS_ROOT = REPO_ROOT / "examples" / "scitrace_research" / "skills"


def _make_skill(root: Path, name: str, body: str, *, description: str = "demo skill") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}",
        encoding="utf-8",
    )
    return skill_dir


def test_load_skill_package_valid(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path, "demo-skill", "# Demo\n\nDo the right thing.")
    package = load_skill_package(skill)
    assert package["name"] == "demo-skill"
    assert package["description"] == "demo skill"
    assert package["body"].startswith("# Demo")


def test_load_skill_package_missing_skill_md(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_skill_package(tmp_path / "absent")


def test_load_skill_package_name_mismatch(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path, "dir-name", "body")
    (skill / "SKILL.md").write_text(
        "---\nname: other-name\ndescription: d\n---\n\nbody", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_skill_package(skill)


def test_load_skill_package_requires_description(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path, "no-desc", "body", description="")
    with pytest.raises(ValueError):
        load_skill_package(skill)


def test_build_skills_block_is_deterministic_and_sorted() -> None:
    names = ["stats-notation", "pre-submit-checklist"]
    first = build_skills_block(list(reversed(names)), SKILLS_ROOT)
    second = build_skills_block(names, SKILLS_ROOT)
    assert first == second
    assert first.startswith("## Inlined SciTrace Skills")
    assert first.index("### Skill: pre-submit-checklist") < first.index("### Skill: stats-notation")


def test_repo_skill_bodies_fit_inline_budget() -> None:
    for name in ("citation-graph-tracer", "pre-submit-checklist", "stats-notation"):
        package = load_skill_package(SKILLS_ROOT / name)
        assert len(package["body"]) <= DEFAULT_MAX_CHARS_PER_SKILL, name


def test_build_skills_block_skips_unknown_skill(tmp_path: Path) -> None:
    assert build_skills_block(["missing-skill"], tmp_path) == ""


def test_build_skills_block_skips_invalid_skill(tmp_path: Path) -> None:
    bad = tmp_path / "bad-skill"
    bad.mkdir()
    (bad / "SKILL.md").write_text("no frontmatter here", encoding="utf-8")
    assert build_skills_block(["bad-skill"], tmp_path) == ""


def test_build_skills_block_truncates_long_body(tmp_path: Path) -> None:
    _make_skill(tmp_path, "long-skill", "x" * 5000)
    block = build_skills_block(["long-skill"], tmp_path, max_chars_per_skill=100)
    assert "[...truncated at 100 characters...]" in block
    assert "x" * 101 not in block


def test_install_skills_to_library_is_idempotent(tmp_path: Path) -> None:
    roots = [_make_skill(tmp_path / "src", name, "body") for name in ("skill-a", "skill-b")]
    target = tmp_path / "library"
    first = install_skills_to_library(roots, target=target)
    second = install_skills_to_library(roots, target=target)
    assert first["installed"] == ["skill-a", "skill-b"]
    assert second["installed"] == []
    assert sorted(second["skipped"]) == ["skill-a", "skill-b"]
    state = json.loads((target / "skills_state.json").read_text(encoding="utf-8"))
    entries = [entry for entry in state["installed_plugins"] if entry.get("marketplace") == "scitrace"]
    assert {entry["name"] for entry in entries} == {"skill-a", "skill-b"}
    assert all(entry["source"] == "local" for entry in entries)
    assert (target / "skill-a" / "SKILL.md").is_file()


def test_stage_skills_block_opt_in_via_config(tmp_path: Path) -> None:
    runtime = JiuwenSwarmResearchTeamRuntime(
        backend=None, research_id="case", output_root=tmp_path,
        config={"skills_dir": str(SKILLS_ROOT)},
    )
    draft_block = runtime._stage_skills_block(get_team_stage_contract("draft", "write_draft"))
    assert "### Skill: pre-submit-checklist" in draft_block
    assert "### Skill: stats-notation" in draft_block
    review_block = runtime._stage_skills_block(get_team_stage_contract("review", "review_draft"))
    assert review_block == ""  # v5 ablation: the reviewer prompt stays skill-free


def test_stage_skills_block_empty_without_config(tmp_path: Path) -> None:
    runtime = JiuwenSwarmResearchTeamRuntime(
        backend=None, research_id="case", output_root=tmp_path,
    )
    assert runtime._stage_skills_block(get_team_stage_contract("draft", "write_draft")) == ""


def test_stage_skills_block_resolves_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_skill(tmp_path / "skills", "pre-submit-checklist", "checklist body")
    _make_skill(tmp_path / "skills", "stats-notation", "body text")
    monkeypatch.chdir(tmp_path)
    runtime = JiuwenSwarmResearchTeamRuntime(
        backend=None, research_id="case", output_root=tmp_path,
        config={"skills_dir": "skills"},
    )
    block = runtime._stage_skills_block(get_team_stage_contract("draft", "write_draft"))
    assert "### Skill: stats-notation" in block

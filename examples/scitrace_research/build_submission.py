from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PR_SECTION_PLACEHOLDER = "{{PR_SECTION}}"
QA_TOKEN_PLACEHOLDER = "QA_ONLY_TOKEN_PLACEHOLDER"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tex(value: object) -> str:
    text = str(value or "")
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_sqlite_consistent(source: Path, destination: Path) -> None:
    """Fold a live run's WAL into the database included in the submission."""
    if not source.is_file():
        return
    destination.unlink(missing_ok=True)
    with sqlite3.connect(f'{source.resolve().as_uri()}?mode=ro', uri=True) as original:
        with sqlite3.connect(destination) as packaged:
            original.backup(packaged)


def _valid_official_pr_url(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"https://(?:www\.)?(?:atomgit\.com|gitcode\.com)/openJiuwen/jiuwenswarm/"
            r"(?:pull|pulls|pull_requests|merge_requests)/\d+(?:/discuss)?/?",
            value.strip(),
            flags=re.IGNORECASE,
        )
    )


from .paper_export import export as _paper, verify


def _copy_code(destination: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", "dist")
    shutil.copytree(ROOT / "jiuwenswarm", destination / "jiuwenswarm", ignore=ignore)
    # The official pyproject declares this package-dir; omitting it creates an
    # apparently complete source bundle that cannot be installed with -e .
    shutil.copytree(ROOT / "jiuwenbox", destination / "jiuwenbox", ignore=ignore)
    shutil.copytree(ROOT / "examples" / "scitrace_research", destination / "examples" / "scitrace_research", ignore=ignore)
    shutil.copy2(ROOT / "examples" / "__init__.py", destination / "examples" / "__init__.py")
    test_source = ROOT / "tests" / "unit_tests" / "agents" / "harness" / "research"
    shutil.copytree(test_source, destination / "tests" / "unit_tests" / "agents" / "harness" / "research", ignore=ignore)
    for name in ("pyproject.toml", "README.md", "LICENSE", "OPEN_SOURCE_SOFTWARE_NOTICE.md", "MANIFEST.in"):
        source = ROOT / name
        if source.exists():
            shutil.copy2(source, destination / name)
    shutil.copy2(ROOT / "examples" / "scitrace_research" / "configs" / "team_real_route_b2.yaml", destination / "config.yaml")
    (destination / "requirements.txt").write_text("-e .\n-r examples/scitrace_research/requirements-paper.txt\n", encoding="utf-8")
    (destination / "main.py").write_text(
        "from examples.scitrace_research.run_team import main\n\nif __name__ == '__main__':\n    raise SystemExit(main())\n",
        encoding="utf-8",
    )
    (destination / "run_guide.md").write_text(
        "# Reproduction\n\n"
        "Use Python 3.11 and run `pip install -r requirements.txt`. Set `API_BASE`, "
        "`API_KEY`, `MODEL_NAME`, and `MODEL_PROVIDER`, then execute:\n\n"
        "```bash\n"
        "bash examples/scitrace_research/launch_paper.sh --check\n"
        "bash examples/scitrace_research/launch_paper.sh\n"
        "```\n\n"
        "Use build_submission with the new run's official reviewer token and your real contribution PR URL.\n\n"
        "The workflow performs live literature retrieval, native stage-scoped Team inference, "
        "verified experiment import and query diagnostics, claim compilation, drafting, consistency review, "
        "ICLR PDF export, and final ZIP construction. Never store API credentials in YAML or source files.\n",
        encoding="utf-8",
    )


def build(
    run_dir: Path,
    output_root: Path,
    team_name: str,
    access_token: str,
    pr_url: str = "",
    *,
    allow_non_real: bool = False,
    paper_dir: Path | None = None,
) -> Path:
    required = ["run_manifest.json", "validation_report.json", "draft.md", "claims.json", "resource_usage.json", "experiments/results.json", "evidence/literature.json"]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise ValueError("run directory is incomplete: " + ", ".join(missing))
    manifest = _load(run_dir / "run_manifest.json")
    validation = _load(run_dir / "validation_report.json")
    if manifest.get("status") != "success" or not validation.get("ok"):
        raise ValueError("only a successful, gate-approved run can be packaged")
    if not allow_non_real and (manifest.get("mode") != "real_team" or not manifest.get("strict_real")):
        raise ValueError("competition packages require a strict-real native Team run")
    if manifest.get("strict_real") and (manifest.get("mock_fallback_used") or manifest.get("fixture_used")):
        raise ValueError("strict-real package contains a forbidden fallback")

    selected_paper = paper_dir or run_dir / "paper"
    verify(run_dir, selected_paper)
    if not team_name or Path(team_name).name != team_name or team_name in (".", ".."):
        raise ValueError("team_name must be a single directory name")
    token = access_token.strip()
    has_token = bool(token)
    if not token and not allow_non_real:
        raise ValueError("Agentic Reviewer access token is required for a competition package")
    if token and (len(token) < 16 or any(ch.isspace() for ch in token)):
        raise ValueError("Agentic Reviewer access token has an invalid format")
    pr_url = pr_url.strip()
    has_pr = bool(pr_url)
    if pr_url and not _valid_official_pr_url(pr_url):
        raise ValueError("PR URL must point to a numbered openJiuwen/jiuwenswarm PR on AtomGit/GitCode")
    token = token or QA_TOKEN_PLACEHOLDER
    target = output_root / team_name
    if target.exists():
        raise ValueError("Output already exists; choose a new output-root to preserve reviewed submissions")
    (target / "Agentic Reviewer").mkdir(parents=True)
    (target / "docs").mkdir(parents=True)
    (target / "code").mkdir(parents=True)
    (target / "artifacts").mkdir(parents=True)
    (target / "Agentic Reviewer" / "PaperReview-AccessToken.txt").write_text(token + "\n", encoding="utf-8")
    _copy_code(target / "code")
    transient_ignore = shutil.ignore_patterns("*.sqlite3-shm", "*.sqlite3-wal", "*.db-shm", "*.db-wal")
    def artifact_ignore(src: str, names: list[str]) -> set[str]:
        ignored = set(transient_ignore(src, names))
        if paper_dir is not None and Path(src).resolve() == run_dir.resolve():
            ignored.add("paper")
        return ignored
    artifact_dir = target / "artifacts" / run_dir.name
    shutil.copytree(run_dir, artifact_dir, ignore=artifact_ignore)
    _copy_sqlite_consistent(run_dir / "research.db", artifact_dir / "research.db")
    if paper_dir is not None:
        shutil.copytree(selected_paper, artifact_dir / "paper")
    assets = ROOT / "examples" / "scitrace_research" / "submission_assets"
    for name in ("architecture.md", "module_call.md", "innovation.md"):
        shutil.copy2(assets / "docs" / name, target / "docs" / name)
    contribution = (assets / "framework_contribution.md").read_text(encoding="utf-8")
    if contribution.count(PR_SECTION_PLACEHOLDER) != 1:
        raise ValueError("framework contribution template must contain exactly one PR section placeholder")
    pr_section = (
        "## Public Contribution PR\n\n"
        f"PR URL: {pr_url}\n\n"
        "This URL points to the contribution discussion in the official "
        "`openJiuwen/jiuwenswarm` repository. The packaged source snapshot remains "
        "available for reproduction independently of the current review status.\n"
        if has_pr else ""
    )
    (target / "framework_contribution.md").write_text(
        contribution.replace(PR_SECTION_PLACEHOLDER, pr_section), encoding="utf-8"
    )
    shutil.copy2(assets / "提交说明.md", target / "提交说明.md")
    shutil.copy2(run_dir / "resource_report.md", target / "resource_report.md")
    shutil.copytree(selected_paper, target / "paper")
    (target / "submission_checklist.md").write_text(
        "# Submission Checklist\n\n"
        "- [x] English ICLR PDF generated from the accepted Agent draft\n"
        "- [x] Native JiuwenSwarm Team source, configuration, tests, and reproduction guide\n"
        "- [x] Architecture, module-call, and innovation documents\n"
        "- [x] Traceable strict-real run artifacts and resource report\n"
        f"- [{'x' if has_token else ' '}] Stanford Agentic Reviewer token for this exact PDF\n",
        encoding="utf-8",
    )
    manifest_files = {}
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.name != "submission_manifest.json":
            manifest_files[path.relative_to(target).as_posix()] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
    (target / "submission_manifest.json").write_text(
        json.dumps(
            {
                "team_name": team_name,
                "source_run": run_dir.name,
                "strict_real": bool(manifest.get("strict_real")),
                "mode": manifest.get("mode"),
                "reviewer_token_present": has_token,
                "official_pr_url": pr_url or None,
                "files": manifest_files,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    zip_path = output_root / f"{team_name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_root))
    print(json.dumps({
        "submission_dir": str(target),
        "zip_path": str(zip_path),
        "token_placeholder": token == QA_TOKEN_PLACEHOLDER,
        "pr_included": has_pr,
    }, ensure_ascii=False, indent=2))
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and build the official SciTrace-Swarm submission ZIP.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-root", default="submission_build")
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--access-token", default="")
    parser.add_argument("--pr-url", default="")
    parser.add_argument("--paper-dir", default="", help="Verified export directory whose PDF matches the supplied Reviewer token")
    parser.add_argument("--allow-non-real", action="store_true", help="QA only: permit packaging a mock/non-strict run.")
    args = parser.parse_args()
    build(
        Path(args.run_dir).resolve(),
        Path(args.output_root).resolve(),
        args.team_name.strip(),
        args.access_token,
        args.pr_url,
        allow_non_real=bool(args.allow_non_real),
        paper_dir=Path(args.paper_dir).resolve() if args.paper_dir else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

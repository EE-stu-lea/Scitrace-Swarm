from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import shutil
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIRS = (
    "jiuwenswarm",
    "jiuwenbox",
    "packages",
    "examples/scitrace_research",
    "tests/unit_tests/agents/harness/research",
)
ROOT_FILES = (
    "examples/__init__.py",
    "pyproject.toml",
    "uv.lock",
    "pytest.ini",
    "MANIFEST.in",
    "README.md",
    "README_CN.md",
    "TESTING.md",
    "LICENSE",
    "OPEN_SOURCE_SOFTWARE_NOTICE.md",
)
IGNORE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    ".venv_real",
}
IGNORE_SUFFIXES = {".pyc", ".pyo", ".log", ".zip"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        path = Path(name)
        if name in IGNORE_NAMES or path.suffix.lower() in IGNORE_SUFFIXES:
            ignored.add(name)
        elif name.endswith((".sqlite3-shm", ".sqlite3-wal", ".db-shm", ".db-wal")):
            ignored.add(name)
    return ignored


def _copy_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"required path is missing: {source}")
    if source.is_dir():
        shutil.copytree(source, destination, ignore=_ignore)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _write_reproduction_files(bundle: Path, reference_run_name: str) -> None:
    (bundle / "requirements.txt").write_text("-e .[test]\n", encoding="utf-8")
    (bundle / "REPRODUCE.md").write_text(
        """# SciTrace-Swarm reproducibility package

This archive contains the JiuwenSwarm source snapshot used by SciTrace-Swarm,
the complete research workflow, relevant unit tests, dependency lock file,
offline mechanism evaluation, and a successful strict-real reference run.
It deliberately contains no API key or Stanford Reviewer token.

## Environment

- Windows PowerShell 7 or Windows PowerShell 5.1
- Python 3.11 (the project supports Python >=3.11,<3.14)
- Git and network access for the `openjiuwen` dependency
- Network access to OpenAlex/Crossref and an OpenAI-compatible model endpoint

## Install and verify locally

```powershell
Set-Location <unzipped-package-directory>
.\\verify_reproduction.ps1
```

The script creates `.venv_real`, installs the project, runs the research unit
tests, reruns the deterministic gate/context ablation, and checks the bundled
reference run and package checksums.

## Run a new strict-real five-agent workflow

```powershell
$env:API_BASE="https://your-provider.example/v1"
$env:API_KEY="YOUR_API_KEY"
$env:MODEL_NAME="YOUR_MODEL_NAME"
$env:MODEL_PROVIDER="OpenAI"
.\\verify_reproduction.ps1 -RunReal -TeamName "YOUR_TEAM_NAME"
```

`MODEL_PROVIDER` must match a provider supported by the installed openJiuwen
client. Credentials are read only from environment variables. A fresh run is
written under `runs/`; it performs literature retrieval, native Team inference,
restricted experiment execution, claim compilation, consistency review, and
ICLR PDF export.

## Included evidence

- `reference_run/`: the successful strict-real Team run `{reference_run_name}`
- `evaluation/gate_ablation_results.json`: deterministic rail ablation output
- `docs/`: architecture, module-call, innovation, and contribution descriptions
- `PACKAGE_MANIFEST.json`: size and SHA-256 for every packaged file

The Stanford Reviewer token belongs in the private competition submission only
and is intentionally excluded from this reproducibility archive.
""".format(reference_run_name=reference_run_name),
        encoding="utf-8",
    )
    (bundle / "verify_reproduction.ps1").write_text(
        r'''param(
    [switch]$RunReal,
    [string]$TeamName = "REPRODUCTION"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:JIUWENSWARM_DATA_DIR = Join-Path $PSScriptRoot ".test_jiuwenswarm_data"
New-Item -ItemType Directory -Force -Path $env:JIUWENSWARM_DATA_DIR | Out-Null

if (-not (Test-Path -LiteralPath ".venv_real\Scripts\python.exe")) {
    py -3.11 -m venv .venv_real
}

$Python = Join-Path $PSScriptRoot ".venv_real\Scripts\python.exe"
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m pytest `
    tests/unit_tests/agents/harness/research/test_provenance_models.py `
    tests/unit_tests/agents/harness/research/test_provenance_store.py `
    tests/unit_tests/agents/harness/research/test_phase2_rails.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m examples.scitrace_research.evaluate_gate `
    --output evaluation/reproduced_gate_ablation_results.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m examples.scitrace_research.verify_repro_bundle `
    --bundle-root $PSScriptRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($RunReal) {
    foreach ($Name in @("API_BASE", "API_KEY", "MODEL_NAME", "MODEL_PROVIDER")) {
        if (-not (Test-Path "Env:$Name") -or [string]::IsNullOrWhiteSpace((Get-Item "Env:$Name").Value)) {
            throw "Environment variable $Name is required for -RunReal."
        }
    }
    & ".\examples\scitrace_research\run_submission.ps1" -TeamName $TeamName
}
''',
        encoding="utf-8",
    )


def build(output_root: Path, name: str, reference_run: Path) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("name must be a single directory name")
    required_run_files = (
        "run_manifest.json",
        "validation_report.json",
        "draft.md",
        "claims.json",
        "resource_usage.json",
        "experiments/results.json",
        "evidence/literature.json",
        "paper/paper.pdf",
    )
    missing = [item for item in required_run_files if not (reference_run / item).is_file()]
    if missing:
        raise ValueError("reference run is incomplete: " + ", ".join(missing))
    manifest = json.loads((reference_run / "run_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((reference_run / "validation_report.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "success" or manifest.get("mode") != "real_team":
        raise ValueError("reference run must be a successful real_team run")
    if not manifest.get("strict_real") or not validation.get("ok"):
        raise ValueError("reference run must be strict-real and gate-approved")

    output_root.mkdir(parents=True, exist_ok=True)
    bundle = output_root / name
    archive_path = output_root / f"{name}.zip"
    if bundle.exists() or archive_path.exists():
        raise ValueError("output already exists; choose a new name or output root")
    bundle.mkdir()

    for relative in SOURCE_DIRS:
        _copy_required(ROOT / relative, bundle / relative)
    for relative in ROOT_FILES:
        _copy_required(ROOT / relative, bundle / relative)

    _copy_required(reference_run, bundle / "reference_run")
    _copy_required(ROOT / "runs/gate_ablation/results.json", bundle / "evaluation/gate_ablation_results.json")
    docs_source = ROOT / "examples/scitrace_research/submission_assets"
    for filename in ("architecture.md", "module_call.md", "innovation.md"):
        _copy_required(docs_source / "docs" / filename, bundle / "docs" / filename)
    _copy_required(docs_source / "framework_contribution.md", bundle / "docs/framework_contribution.md")
    _write_reproduction_files(bundle, reference_run.name)

    prohibited = []
    for path in bundle.rglob("*"):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if lower_name in {".env", "paperreview-accesstoken.txt"} or lower_name.endswith(".zip"):
            prohibited.append(path.relative_to(bundle).as_posix())
    if prohibited:
        raise ValueError("prohibited files entered the bundle: " + ", ".join(prohibited))

    files = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "PACKAGE_MANIFEST.json":
            files[path.relative_to(bundle).as_posix()] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
    package_manifest = {
        "package_name": name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_root": ROOT.name,
        "reference_run": reference_run.name,
        "strict_real": True,
        "reviewer_token_included": False,
        "api_credentials_included": False,
        "file_count": len(files),
        "files": files,
    }
    (bundle / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(package_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_root))
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "archive": str(archive_path),
                "archive_sha256": _sha256(archive_path),
                "file_count": len(files) + 1,
                "reference_run": reference_run.name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a sanitized SciTrace-Swarm reproducibility archive.")
    parser.add_argument("--output-root", default="release_packages")
    parser.add_argument("--name", default="369387_scitrace_swarm_repro")
    parser.add_argument(
        "--reference-run",
        default="runs/team_real_submission_20260908_202003",
    )
    args = parser.parse_args()
    build(
        Path(args.output_root).resolve(),
        args.name.strip(),
        Path(args.reference_run).resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

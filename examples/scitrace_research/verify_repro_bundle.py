from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(bundle_root: Path) -> None:
    manifest_path = bundle_root / "PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest["files"].items():
        path = bundle_root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        if path.stat().st_size != expected["size_bytes"]:
            failures.append(f"size mismatch: {relative}")
            continue
        if _sha256(path) != expected["sha256"]:
            failures.append(f"hash mismatch: {relative}")
    reference_manifest = json.loads(
        (bundle_root / "reference_run/run_manifest.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (bundle_root / "reference_run/validation_report.json").read_text(encoding="utf-8")
    )
    if reference_manifest.get("status") != "success":
        failures.append("reference run status is not success")
    if reference_manifest.get("mode") != "real_team" or not reference_manifest.get("strict_real"):
        failures.append("reference run is not strict-real native Team")
    if not validation.get("ok"):
        failures.append("reference run validation gate did not pass")
    if failures:
        raise ValueError("bundle verification failed:\n- " + "\n- ".join(failures))
    print(
        json.dumps(
            {
                "status": "ok",
                "verified_files": len(manifest["files"]),
                "reference_run": manifest["reference_run"],
                "strict_real": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a SciTrace-Swarm reproducibility bundle.")
    parser.add_argument("--bundle-root", default=".")
    args = parser.parse_args()
    verify(Path(args.bundle_root).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fetch or verify the two pinned external MATLAB dependencies.

External checkouts are not part of the distributed source package. MATLAB and
its toolboxes must be obtained separately under the user's applicable licenses.
This script never removes or overwrites an existing dependency checkout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

REPOSITORY = Path(__file__).resolve().parents[1]
DEPENDENCIES = {
    "SpyCode": {
        "url": "https://github.com/barbaLab/SpyCode.git",
        "commit": "fb9bc345d562939a4c6f4c90002e5f689de45db1",
        "license_file": "LICENSE.md",
        "license_notice": "LGPL-3.0; inspect additional file-specific notices",
    },
    "NetworkWideAdaptiveBurstDetection": {
        "url": "https://github.com/kerstinlenk/NetworkWideAdaptiveBurstDetection.git",
        "commit": "9de22b0c2dde0dce29adfa207245e15d1f3805a4",
        "license_file": "LICENSE",
        "license_notice": "GPL-3.0-or-later, as stated in the upstream README",
    },
}


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", "--no-optional-locks", *map(str, args)],
        cwd=cwd, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(f"Git command failed: {result.stderr.strip()}")
    return result.stdout.strip()


def verify(root: Path) -> list[dict]:
    records = []
    for name, spec in DEPENDENCIES.items():
        directory = root / name
        if not (directory / ".git").exists():
            raise RuntimeError(f"Missing dependency checkout: {directory}")
        commit = git("rev-parse", "HEAD", cwd=directory)
        if commit != spec["commit"]:
            raise RuntimeError(f"{name}: expected {spec['commit']}, found {commit}")
        if git("status", "--porcelain", "--untracked-files=no", cwd=directory):
            raise RuntimeError(f"{name}: tracked source differs from the pinned checkout")
        if not (directory / spec["license_file"]).is_file():
            raise RuntimeError(f"{name}: upstream license file is missing")
        records.append(dict(name=name, commit=commit, license_notice=spec["license_notice"]))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=REPOSITORY / "external_dependencies")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.directory.resolve()
    if not args.verify_only:
        root.mkdir(parents=True, exist_ok=True)
        for name, spec in DEPENDENCIES.items():
            directory = root / name
            if directory.exists():
                if not (directory / ".git").exists():
                    raise RuntimeError(f"Refusing to overwrite existing path: {directory}")
                continue
            git("clone", "--filter=blob:none", "--no-checkout", spec["url"], str(directory))
            git("checkout", "--detach", spec["commit"], cwd=directory)
    print(json.dumps({"verified": True, "dependencies": verify(root)}, indent=2))


if __name__ == "__main__":
    main()

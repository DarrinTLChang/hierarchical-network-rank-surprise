"""Verify the release checksums without reading or running any input recording."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IGNORED_DIRECTORIES = {".git", "__pycache__", ".venv", ".pytest_cache", "runs", "validation"}


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    manifest = ROOT / "MANIFEST.sha256"
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or name in expected:
            raise ValueError("Invalid or duplicate manifest path")
        expected[name] = digest
    for name, digest in expected.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise ValueError(f"Missing, changed, or linked release file: {name}")
    actual = {
        p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
        if p.is_file() and p != manifest
        and not any(part in IGNORED_DIRECTORIES for part in p.relative_to(ROOT).parts)
    }
    extra = actual - set(expected)
    if extra:
        raise ValueError(f"Unlisted files (do not include in an upload): {sorted(extra)}")
    print(f"Verified {len(expected)} manifest-listed files; no unlisted release files.")
    print("Checksums verify identity, not permission to publish or data privacy clearance.")


if __name__ == "__main__":
    main()

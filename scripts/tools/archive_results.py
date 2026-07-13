#!/usr/bin/env python3
"""Archive generated results and deduplicate identical files with hard links."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--archive-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_dir = args.archive_dir.resolve()
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved: list[dict[str, str]] = []
    for source in args.sources:
        if not source.exists():
            continue
        destination = archive_dir / source.as_posix().lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"archive destination already exists: {destination}")
        shutil.move(str(source), str(destination))
        moved.append({"source": str(source), "destination": str(destination)})

    canonical_by_hash: dict[str, Path] = {}
    duplicates: list[dict[str, str]] = []
    for path in sorted(p for p in archive_dir.rglob("*") if p.is_file()):
        if path.name == "manifest.json":
            continue
        digest = file_hash(path)
        canonical = canonical_by_hash.get(digest)
        if canonical is None:
            canonical_by_hash[digest] = path
            continue
        if os.stat(path).st_dev != os.stat(canonical).st_dev:
            continue
        path.unlink()
        os.link(canonical, path)
        duplicates.append(
            {
                "sha256": digest,
                "canonical": str(canonical.relative_to(archive_dir)),
                "duplicate": str(path.relative_to(archive_dir)),
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "moved": moved,
        "deduplicated": duplicates,
    }
    (archive_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"archived={len(moved)} deduplicated={len(duplicates)} dir={archive_dir}")


if __name__ == "__main__":
    main()

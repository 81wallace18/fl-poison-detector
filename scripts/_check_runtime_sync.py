#!/usr/bin/env python3
"""Check that the standalone and MONZA detector copies are identical."""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path

MIRRORED_FILES = (
    "cc_mlp.py",
    "context_features.py",
    "features.py",
    "fl_save.py",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=root / "src")
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=root / "PFLlibMonza/system/flcore/detector",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = 0
    for filename in MIRRORED_FILES:
        source = args.source_dir / filename
        runtime = args.runtime_dir / filename
        if not source.exists() or not runtime.exists():
            print(f"MISSING {filename}: source={source.exists()} runtime={runtime.exists()}")
            failures += 1
            continue
        source_text = source.read_text(encoding="utf-8").splitlines()
        runtime_text = runtime.read_text(encoding="utf-8").splitlines()
        if source_text == runtime_text:
            continue
        failures += 1
        print(f"DIFF {filename}")
        print(
            "\n".join(
                difflib.unified_diff(
                    source_text,
                    runtime_text,
                    fromfile=str(source),
                    tofile=str(runtime),
                    lineterm="",
                )
            )
        )
    if failures:
        print(f"runtime sync failed: {failures} file(s) differ")
        return 1
    print(f"runtime sync ok: {len(MIRRORED_FILES)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

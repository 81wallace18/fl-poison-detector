#!/usr/bin/env python3
"""Validate relative links in project Markdown files."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def main() -> int:
    failures: list[str] = []
    markdown_files = [ROOT / "README.md"]
    markdown_files.extend((ROOT / "docs").rglob("*.md"))
    markdown_files.extend((ROOT / "scripts").rglob("*.md"))
    markdown_files.append(ROOT / "PFLlibMonza/README.md")

    for document in markdown_files:
        text = document.read_text(encoding="utf-8")
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / unquote(target)).resolve()
            if not resolved.exists():
                failures.append(
                    f"{document.relative_to(ROOT)} -> {raw_target}"
                )
    if failures:
        print("broken local Markdown links:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"Markdown links ok: {len(markdown_files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

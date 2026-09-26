#!/usr/bin/env python3
"""Check local markdown/html links for missing files.

This is a lightweight static checker intended for Jekyll content.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK_SUFFIXES = {".md", ".html", ".yml", ".yaml", ".scss"}

PATTERN = re.compile(
    r"!\[[^\]]*\]\(([^)]+)\)"
    r"|\[[^\]]+\]\(([^)]+)\)"
    r"|src=[\"']([^\"']+)[\"']"
    r"|href=[\"']([^\"']+)[\"']"
)


def iter_files() -> list[Path]:
    return [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in CHECK_SUFFIXES]


def normalize_target(raw: str) -> str | None:
    target = raw.strip().strip("\"'")
    if not target:
        return None
    if any(target.startswith(prefix) for prefix in ("http://", "https://", "mailto:", "javascript:", "//", "#", "{{", "{%")):
        return None
    target = target.split("#", 1)[0].split("?", 1)[0]
    # Skip non-path markdown references such as "[text](these lecture notes)".
    if "/" not in target and "." not in Path(target).name:
        return None
    return target or None


def resolve_path(file_path: Path, target: str) -> Path:
    if target.startswith("assets/"):
        return ROOT / target
    if target.startswith("/"):
        return ROOT / target.lstrip("/")
    return (file_path.parent / target).resolve()


def main() -> int:
    missing: list[tuple[Path, str, Path]] = []
    for file_path in iter_files():
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for match in PATTERN.finditer(text):
            raw_target = next(group for group in match.groups() if group is not None)
            target = normalize_target(raw_target)
            if target is None:
                continue
            resolved = resolve_path(file_path, target)
            if not resolved.exists():
                missing.append((file_path, target, resolved))

    if not missing:
        print("No missing local links/assets found.")
        return 0

    print("Missing local links/assets detected:")
    for file_path, target, resolved in missing:
        rel_file = file_path.relative_to(ROOT)
        rel_resolved = resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved
        print(f"- {rel_file}: {target} -> {rel_resolved}")

    print(f"\nTotal missing references: {len(missing)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

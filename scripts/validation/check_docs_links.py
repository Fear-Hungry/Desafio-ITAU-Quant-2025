#!/usr/bin/env python3
"""Check internal Markdown links across the repository.

This is a lightweight validation to keep documentation navigable when files
move or names change.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LinkError:
    source: Path
    target: str
    resolved: Path


def _iter_markdown_files() -> list[Path]:
    candidates = [
        REPO_ROOT / "README.md",
        *sorted((REPO_ROOT / "docs").rglob("*.md")),
        *sorted((REPO_ROOT / "configs").rglob("*.md")),
        *sorted((REPO_ROOT / "data").rglob("*.md")),
        *sorted((REPO_ROOT / "scripts").rglob("*.md")),
        *sorted((REPO_ROOT / "tests").rglob("*.md")),
        *sorted((REPO_ROOT / "notebooks").rglob("*.md")),
    ]
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        if path.exists() and path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _extract_links(markdown: str) -> list[str]:
    # Basic inline markdown links: [text](target)
    # We intentionally skip images because they use the same syntax and are
    # commonly remote or optional in repos.
    pattern = re.compile(r"(?<!\!)\[[^\]]+\]\(([^)]+)\)")
    return [m.group(1).strip() for m in pattern.finditer(markdown)]


def _should_ignore(target: str) -> bool:
    lowered = target.lower()
    return lowered.startswith(("http://", "https://", "mailto:", "tel:"))


def _resolve_target(source: Path, target: str) -> Path | None:
    if _should_ignore(target):
        return None

    # Strip surrounding angle brackets sometimes used in Markdown.
    cleaned = target.strip().strip("<>").strip()
    if not cleaned:
        return None

    # Drop anchor fragments: file.md#section
    cleaned = cleaned.split("#", 1)[0].strip()
    if not cleaned:
        return None

    # Ignore pure anchors: (#section)
    if cleaned.startswith("#"):
        return None

    # Ignore code reference links like `arara_quant.module` (not a path).
    if "://" in cleaned:
        return None

    candidate = (source.parent / cleaned).resolve()
    return candidate


def main(argv: list[str] | None = None) -> int:
    _ = argv  # currently unused; reserved for future flags

    errors: list[LinkError] = []
    for md_path in _iter_markdown_files():
        text = md_path.read_text(encoding="utf-8")
        for target in _extract_links(text):
            resolved = _resolve_target(md_path, target)
            if resolved is None:
                continue
            if not resolved.exists():
                errors.append(
                    LinkError(source=md_path.relative_to(REPO_ROOT), target=target, resolved=resolved)
                )

    if errors:
        print("Broken documentation links detected:\n", file=sys.stderr)
        for err in errors:
            print(
                f"- {err.source}: ({err.target}) -> {err.resolved}",
                file=sys.stderr,
            )
        print(f"\nTotal: {len(errors)}", file=sys.stderr)
        return 1

    print("✓ Documentation links look OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

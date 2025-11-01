"""YAML loader with optional fallback when PyYAML is unavailable."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency branch
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - used in tests when PyYAML missing
    yaml = None

__all__ = ["load_yaml_text", "read_yaml"]


def load_yaml_text(text: str) -> dict[str, Any]:
    """Parse YAML/JSON-ish text returning a dictionary."""

    if yaml is not None:
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML root must be a mapping")
        return data
    return _minimal_yaml_load(text)


def read_yaml(path: Path) -> dict[str, Any]:
    return load_yaml_text(path.read_text(encoding="utf-8"))


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    mapping, remaining = _parse_block(lines, indent=0)
    if remaining:
        raise ValueError("Unexpected trailing content while parsing YAML")
    return mapping


def _parse_block(lines: list[str], indent: int) -> tuple[dict[str, Any], list[str]]:
    mapping: dict[str, Any] = {}
    while lines:
        raw = lines[0]
        stripped = raw.lstrip()
        if not stripped or stripped.startswith("#"):
            lines.pop(0)
            continue
        current_indent = len(raw) - len(stripped)
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError("Invalid indentation level in YAML content")
        lines.pop(0)
        key, _, remainder = stripped.partition(":")
        key = key.strip()
        remainder = remainder.strip()
        if not key:
            raise ValueError("Empty key encountered in YAML mapping")
        if not remainder:
            if lines and lines[0].lstrip().startswith("- "):
                value, lines = _parse_list(lines, indent + 2)
            else:
                value, lines = _parse_block(lines, indent + 2)
        else:
            value = _parse_value(remainder)
        mapping[key] = value
    return mapping, lines


def _parse_list(lines: list[str], indent: int) -> tuple[list[Any], list[str]]:
    items: list[Any] = []
    while lines:
        raw = lines[0]
        stripped = raw.lstrip()
        if not stripped:
            lines.pop(0)
            continue
        current_indent = len(raw) - len(stripped)
        if current_indent < indent:
            break
        if not stripped.startswith("- "):
            break
        lines.pop(0)
        value_text = stripped[2:].strip()
        if value_text:
            items.append(_parse_value(value_text))
        else:
            block, lines = _parse_block(lines, indent + 2)
            items.append(block)
    return items, lines


def _parse_value(token: str) -> Any:
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if token.startswith("[") or token.startswith("{"):
        try:
            return ast.literal_eval(token)
        except (ValueError, SyntaxError) as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Cannot parse literal value: {token}") from exc
    try:
        if any(ch in token for ch in (".", "e", "E")):
            return float(token)
        return int(token)
    except ValueError:
        return token.strip("\"'")

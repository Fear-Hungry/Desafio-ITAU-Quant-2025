# Language Policy (English-only)

This repository follows an **English-only** policy for all new and updated content.

## Scope
- Source code: identifiers, docstrings, comments, CLI help, log messages.
- Documentation: `README.md`, `docs/**`, diagrams, and report templates.
- Configuration: keys and field names should be English when introduced/renamed.

## Rationale
- Consistency for an international audience (paper readers, reviewers, and users).
- Lower maintenance cost (single language across code + docs).
- Clearer contribution and review process.

## Rules
- Use English for:
  - Function/class/variable names.
  - CLI output and help strings.
  - Docstrings and user-facing docs.
- Avoid mixing languages in the same file/section.
- Prefer simple, technical English. Avoid slang and idioms.

## Legacy content
Some legacy documentation and research scripts may still contain Portuguese text.
When touching those areas, prefer translating to English as part of the change.

## Practical guidance
- Keep naming consistent: `snake_case` for Python, `kebab-case` for file names.
- Prefer neutral terms: “out-of-sample”, “walk-forward”, “turnover”, “drawdown”.
- For math-heavy notes, keep the math as-is and translate the surrounding prose.


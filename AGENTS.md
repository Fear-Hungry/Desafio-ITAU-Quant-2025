# Repository Guidelines

## Project Structure & Module Organization
- Core library lives under `src/itau_quant/` with thematic packages (`backtesting/`, `optimization/`, `risk/`, `utils/`) mirroring the investment workflow.
- Configuration schemas reside in `configs/` and `src/configs/`; runtime-sensitive overrides belong in `secrets/` (keep this out of version control).
- Datasets are split into `data/raw/`, `data/processed/`, and experiment outputs under `results/`; notebooks for exploratory analysis stay in `notebooks/`.
- Tests target each domain module from `tests/`, while docs, PRD, and quick-start materials sit in `docs/` and the root Markdown files.

## Build, Test, and Development Commands
```bash
poetry install                     # bootstrap a local dev environment
poetry run pytest                  # run the entire test suite (PYTHONPATH already set)
poetry run pytest tests/estimators # focus on estimator validation
poetry run ruff check src tests    # lint for both style and static analysis
poetry run black --check src tests # enforce formatting before pushing
make test                          # alternative for CI-like pytest execution
```

## Coding Style & Naming Conventions
- Target Python 3.11 with `ruff` and `black`; adopt Black’s default formatting (4-space indents, double quotes, trailing commas where applicable).
- Follow snake_case for functions, methods, and local variables; PascalCase for classes; ALL_CAPS for constants and env keys.
- Locate reusable config keys in `itau_quant.config`; prefer `Path` objects over plain strings when handling filesystem paths.
- Keep CLI commands and user-facing strings in English; domain documentation may remain bilingual when already so in the repo.

## Testing Guidelines
- All new features require `pytest` coverage with fixtures or factories colocated in `tests/conftest.py` or module-level fixtures.
- Name test files `test_<module>.py` and structure cases using Arrange-Act-Assert comments when logic is complex.
- For stochastic components, seed via `itau_quant.utils.random` helpers to ensure deterministic runs.
- Execute `poetry run pytest` before opening a PR; include targeted commands (e.g., `pytest tests/risk/test_cvar.py`) in the PR discussion for complex areas.

## Commit & Pull Request Guidelines
- Mirror the existing history: prefer Conventional Commit prefixes (`feat:`, `fix:`, `chore:`) alongside concise, action-oriented subjects; Portuguese context is acceptable in the body.
- Each PR should outline scope, validation commands, and references to `PRD.md` tasks or GitHub issues; attach CLI output or screenshots when UI/report artifacts change.
- Keep branches focused; rebase on `main` before requesting review, and ensure CI (lint + tests) passes locally.

## Security & Configuration Tips
- Never commit credentials or raw client data—store them in `.env` or `secrets/` and document usage in `docs/SECURITY_NOTES.md` if updates are required.
- Validate new YAML configs with `poetry run itau-quant show-settings --json` before publishing to avoid runtime schema regressions.

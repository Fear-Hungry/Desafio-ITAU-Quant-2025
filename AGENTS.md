# Repository Guidelines

## Project Structure & Module Organization
- Source in `src/arara_quant/` (modules: `optimization/`, `backtesting/`, `estimators/`, `portfolio/`, `risk/`, `pipeline/`, `utils/`, CLI in `cli.py`).
- Tests in `tests/` mirroring package subfolders (e.g., `tests/optimization/`).
- Configuration in `configs/*.yaml`; scripts in `scripts/`; notebooks in `notebooks/`.
- Data artifacts in `data/`; run outputs in `results/` and `reports/`; docs in `docs/`.

## Build, Test, and Development Commands
- `make dev-setup` — install deps and pre-commit hooks (Poetry required).
- `make test` / `make test-cov` — run tests (with coverage HTML in `htmlcov/`).
- `make lint` / `make format` / `make type-check` — Ruff, Black, MyPy.
- `make validate` or `make validate-full` — quick/full validation bundle.
- Run CLI: `poetry run arara-quant backtest --config configs/optimizer_example.yaml --no-dry-run`.
- Pipeline helpers: `make data`, `make optimize`, `make backtest`, `make oos`.
- Tip: `make` shows a categorized help menu.

## Coding Style & Naming Conventions
- Python ≥ 3.11, 4-space indentation, max line length 88 (Black/Ruff).
- Tools: Black (format), Ruff (lint), MyPy (types); enforced via pre-commit.
- Naming: modules/files `snake_case`; classes `CamelCase`; functions/vars `snake_case`; constants `UPPER_CASE`.
- Place executable scripts under `scripts/`; expose user flows via `arara_quant.cli`.

## Testing Guidelines
- Framework: PyTest. Tests live in `tests/` and are named `test_*.py`.
- Mirror package structure for discoverability; prefer small, deterministic units.
- Commands: `make test`, `pytest tests/optimization -v`, or `make test-cov`.
- Coverage targets (Codecov): project ≥ 70%, patch ≥ 75%. Generate HTML via `make coverage`.

## Commit & Pull Request Guidelines
- Follow Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:` (e.g., `feat(cli): add walk-forward report`).
- Messages should be imperative, concise; include scope when helpful.
- PRs must include: clear summary, motivation/linked issue, tests/fixtures, and doc/README updates when behavior changes. Attach figures for analysis changes when relevant.
- Ensure CI parity locally: `make validate-full` and `make test-cov-xml` pass. Do not commit secrets or large data artifacts.

## Security & Configuration Tips
- Never commit credentials; keep `.env` local and sanitize `configs/*.yaml`.
- Validate YAML with `make validate-configs`. Prefer `poetry run ...` to use the project venv.
- Reproducibility: record command and config paths in PR descriptions when sharing results from `results/` or `reports/`.


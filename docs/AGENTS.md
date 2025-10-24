# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/itau_quant/`, split into `data`, `optimization`, `backtesting`, and `utils` to isolate ETL, modeling, and orchestration concerns. Tests mirror this layout under `tests/` so every module pairs with a `test_*.py`. Store immutable inputs in `data/raw`, write derived tables to `data/processed`, and keep exploratory work in `notebooks/` with polished outputs archived in `reports/`.

## Build, Test, and Development Commands
- `poetry install` — bootstrap the virtualenv with runtime and dev dependencies.
- `poetry run pytest` — run the unit suite and halt on regressions.
- `poetry run ruff check src tests` — lint code style and imports.
- `poetry run black src tests` — apply canonical formatting.
- `poetry run python -m itau_quant.backtesting.engine` — smoke-test manual backtest entry points when introduced.

## Coding Style & Naming Conventions
Target Python 3.9+, annotate public functions, and prefer small helpers over deeply nested logic. `black` and `ruff` share an 88-character limit; let them dictate layout and import order. Use `snake_case` for modules and functions, `PascalCase` for classes, and ALL_CAPS for config toggles. Rely on `itau_quant.utils.logging_config` for structured logging instead of ad-hoc prints.

## Testing Guidelines
Name new files `test_<module>.py` beside their subject package. Favor parametrized pytest cases for estimator and solver scenarios, asserting portfolio constraints (weights sum to one, bounds respected) and numerical tolerances. Run `poetry run pytest` before pushing and keep fixtures lightweight unless reuse clearly reduces boilerplate.

## Commit & Pull Request Guidelines
Use conventional commits such as `feat:`, `fix:`, or `chore:` and keep subjects imperative under 72 characters. Capture impacted data sources or notebooks in the body for reproducibility. Pull requests need a concise summary, evidence of local tests, and—when strategy behavior shifts—a link to the validating notebook or report snippet.

## Data Handling & Reproducibility
Avoid committing proprietary datasets; instead document retrieval steps or store them under ignored paths. Version generated artefacts only when they accelerate reviews, otherwise regenerate via scripts or notebooks. When notebook logic matures, migrate reusable pieces into `src/` and leave the notebook focused on narrative insight.

#!/usr/bin/env bash
set -euo pipefail
poetry run ruff format .
poetry run ruff check .
poetry run mypy src
poetry run pytest -q
bash ./.codex/checks/check_secrets.sh
poetry run python ./.codex/checks/check_notebooks_clean.py
poetry run python ./.codex/checks/check_quant_rules.py
poetry run python ./.codex/checks/check_changed_files.py

# Guia de Contribuição

Obrigado por considerar contribuir com este repositório.

## Ambiente

- Python: `>=3.11,<3.14` (ver `pyproject.toml`)
- Dependências: Poetry (recomendado)

## Setup

```bash
poetry install
poetry run pre-commit install
```

## Testes

```bash
poetry run pytest -v
```

Atalhos via `Makefile`:

- `make test-fast`
- `make test`
- `make test-cov`

## Qualidade de código

```bash
poetry run ruff check src tests
poetry run black --check src tests
poetry run mypy src --ignore-missing-imports --no-strict-optional
```

Atalho:

```bash
make check-all
```

## Estrutura do repositório

- `src/arara_quant/`: pacote principal (dados, estimadores, otimização, backtesting).
- `configs/`: YAMLs versionados (universos, budgets, experimentos).
- `scripts/`: entrypoints do pipeline e utilitários.
- `tests/`: testes por domínio.
- `docs/`: guias, specs e relatórios.

## Padrões

- Prefira PRs pequenos e revisáveis.
- Adicione/atualize testes quando alterar comportamento.
- Atualize docs quando adicionar flags/configs novas.
- Evite commitar artefatos gerados (ver `.gitignore`).

## Commits e Pull Requests

- Use Conventional Commits quando possível (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- Descreva motivação, trade-offs e como reproduzir/testar.
- Para mudanças que afetam métricas/relatórios, inclua os comandos usados (ex.: `make reproduce-oos`).


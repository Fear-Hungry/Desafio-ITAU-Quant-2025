# Testes (`tests/`)

Os testes cobrem os principais módulos do `src/arara_quant/` (dados,
estimadores, otimização, backtesting e relatórios). A suíte foi desenhada para
rodar majoritariamente **offline** (sem dependência de rede), com casos
específicos isolados quando necessário.

## Rodando localmente

- **Suíte completa**
  - `poetry run pytest -v`
- **Smoke/rápido (recomendado no dia a dia)**
  - `poetry run pytest -q -k "not slow" -x`
- **Por área**
  - `poetry run pytest tests/data/ -q`
  - `poetry run pytest tests/optimization/ -q`
  - `poetry run pytest tests/backtesting/ -q`

## Via Makefile

- `make test-fast`
- `make test`
- `make test-cov`

## Estrutura

- `tests/data/`: limpeza, calendário, loaders e retornos.
- `tests/estimators/`: μ/Σ, BL, robustez e validações.
- `tests/optimization/`: constraints, solvers, CVaR e pós-processamento.
- `tests/backtesting/`: engine walk-forward, execução, métricas e ledger.
- `tests/unit/`: casos unitários curtos (opcional, quando fizer sentido).
- `tests/integration/`: demos e testes de integração do pipeline.
- `tests/performance/`: benchmarks (quando estáveis).

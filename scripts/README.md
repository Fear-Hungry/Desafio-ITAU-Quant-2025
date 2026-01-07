# Scripts (`scripts/`)

Este diretório contém os scripts executáveis do projeto. Em geral, prefira
rodar via `poetry run ...` a partir da raiz do repositório.

## Estrutura

```
scripts/
├── run_01_data_pipeline.py           # Pipeline: download → preprocess → retornos
├── run_02_estimate_params.py         # Pipeline: estima μ/Σ (artefatos em data/processed)
├── run_03_optimize.py                # Pipeline: otimização a partir de artefatos/config
├── run_master_validation.py          # Orquestrador (modo quick/full/production)
├── validate_configs.py               # Validação de YAMLs
├── consolidate_oos_metrics.py        # Consolida métricas a partir de nav_daily.csv
├── generate_oos_figures.py           # Figuras OOS a partir dos artefatos consolidados
├── baselines/                        # Utilitários de baselines (ex.: turnover por janela)
├── data/                             # Scripts auxiliares de dados (ex.: RF via FRED)
├── examples/                         # Exemplos executáveis (ARARA básico/robusto)
├── production/                       # Rotinas de produção (ERC v1/v2)
├── research/                         # Experimentos e análises (walk-forward, sensitivities)
├── validation/                       # Smokes e validações offline/constraints/estimators
└── utils/                            # Utilitários pequenos (ex.: helpers do Makefile)
```

## Entrypoints recomendados

- **Reproduzir OOS canônico**
  - `make reproduce-oos` (atalho) ou a sequência no `README.md`
- **Smoke offline (sem rede)**
  - `poetry run python scripts/validation/offline_data_smoke.py`
- **Walk-forward (principal)**
  - `poetry run python scripts/research/run_backtest_walkforward.py`
- **Orquestração completa**
  - `poetry run python scripts/run_master_validation.py --mode full`
- **Checagens de documentação**
  - `make docs` (valida links internos) e `make serve-docs` (static em localhost)

## Via CLI (quando disponível)

O projeto expõe uma CLI unificada em `arara-quant`:

```bash
poetry run arara-quant --help
poetry run arara-quant optimize --config configs/optimizer_example.yaml
poetry run arara-quant backtest --config configs/optimizer_example.yaml --no-dry-run --wf-report
```

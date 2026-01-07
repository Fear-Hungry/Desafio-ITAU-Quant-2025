# Quickstart

Guia rápido para instalar, rodar um smoke e reproduzir o OOS canônico.

Para referência completa de comandos e orquestração, veja:

- `README.md`
- `docs/ORCHESTRATION_GUIDE.md`
- `docs/QUICK_START_COMMANDS.md`

## 1) Instalação

Pré-requisitos: Python `>= 3.11` e Poetry.

```bash
poetry --version
python --version
poetry install --sync
```

## 2) Smoke (offline, sem rede)

```bash
poetry run python scripts/validation/offline_data_smoke.py
```

## 3) Reproduzir OOS canônico

Atalho:

```bash
make reproduce-oos
```

Ou em etapas:

```bash
poetry run python scripts/run_01_data_pipeline.py --force-download --start 2010-01-01
poetry run python scripts/research/run_backtest_walkforward.py
poetry run python scripts/consolidate_oos_metrics.py --psr-n-trials 1
poetry run python scripts/generate_oos_figures.py
```

Se você tiver a série de T-Bill (FRED) disponível:

```bash
poetry run python scripts/consolidate_oos_metrics.py \
  --riskfree-csv data/processed/riskfree_tbill_daily.csv \
  --psr-n-trials 1
```

## 4) CLI (interface unificada)

```bash
poetry run arara-quant --help
poetry run arara-quant optimize --config configs/optimizer_example.yaml
poetry run arara-quant backtest --config configs/optimizer_example.yaml --no-dry-run --wf-report
```

## 5) Onde editar configuração

- YAMLs de referência vivem em `configs/` (guia: `configs/README.md`).
- Dados e artefatos locais: `data/` (guia: `data/README.md`).


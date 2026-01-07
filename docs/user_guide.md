# Guia do Usuário

Este guia descreve os fluxos de execução recomendados do projeto PRISM-R,
separando o que é **reprodutível** (scripts/configs) do que é **exploratório**
(notebooks).

## 1) Ambiente

```bash
poetry install --sync
make validate-configs
```

Smoke rápido (offline):

```bash
poetry run python scripts/validation/offline_data_smoke.py
```

## 2) Fluxo A — Reproduzir o OOS canônico

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

## 3) Fluxo B — Pipeline “numerado” (dados → μ/Σ → otimização)

1. **Dados (download + preprocess + retornos)**

   ```bash
   poetry run python scripts/run_01_data_pipeline.py --force-download --start 2010-01-01
   ```

   Saída padrão: `data/processed/returns_arara.parquet`.

2. **Estimativa de parâmetros (μ/Σ)**

   ```bash
   poetry run python scripts/run_02_estimate_params.py --annualize
   ```

   Saídas padrão: `data/processed/mu_estimate.parquet` e `data/processed/cov_estimate.parquet`.

3. **Otimização**

   ```bash
   poetry run python scripts/run_03_optimize.py --risk-aversion 4.0 --max-weight 0.15
   ```

   Saída padrão: `outputs/results/optimized_weights.parquet`.

## 4) Fluxo C — Orquestração/validação (pipeline completo)

```bash
poetry run python scripts/run_master_validation.py --mode quick --skip-download
poetry run python scripts/run_master_validation.py --mode full
```

Guia detalhado: `docs/ORCHESTRATION_GUIDE.md`.

## 5) Testes

```bash
make test-fast
make lint
make format-check
```

Ou diretamente:

```bash
poetry run pytest -q -k "not slow" -x
```

## 6) Configuração

- YAMLs em `configs/` (guia: `configs/README.md`)
- Dados e artefatos locais em `data/` (guia: `data/README.md`)


# Dados (`data/`)

Este diretório guarda dados **locais** e artefatos intermediários do pipeline.
Por padrão, a maior parte do conteúdo é gerada automaticamente e fica fora do
controle de versão (ver `.gitignore`).

## Estrutura

```
data/
├── raw/         # caches de downloads (ex.: preços ajustados)
├── processed/   # artefatos processados (ex.: retornos em Parquet)
└── results/     # exports manuais/snapshots (opcional)
```

## Pipeline padrão

1. **Baixar/atualizar preços e gerar retornos**

   ```bash
   poetry run python scripts/run_01_data_pipeline.py --force-download --start 2010-01-01
   ```

   - Entrada default: `data/raw/prices_arara.csv`
   - Saída default: `data/processed/returns_arara.parquet`

2. **(Opcional) Taxa livre de risco (T-Bill via FRED)**

   ```bash
   poetry run python scripts/data/fetch_tbill_fred.py --start 2010-01-01 --end 2025-12-31
   ```

   - Saída: `data/processed/riskfree_tbill_daily.csv`

## Offline / sem rede

Para validar a pilha sem depender de APIs externas, rode:

```bash
poetry run python scripts/validation/offline_data_smoke.py
```

Esse smoke cria um dataset sintético pequeno em `data/raw`/`data/processed`,
executa pré-processamento e roda um walk-forward curto.

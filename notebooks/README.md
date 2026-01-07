# Notebooks (`notebooks/`)

Este diretório contém notebooks usados para exploração de dados, prototipagem e
análise de resultados. Eles são complementares aos scripts e à CLI — o pipeline
reprodutível do projeto roda via `scripts/` e `configs/`.

## Notebooks disponíveis

- `notebooks/01-data-exploration.ipynb`: exploração e sanity checks de dados.
- `notebooks/02-model-prototyping.ipynb`: protótipos de estimadores/otimização.
- `notebooks/03-results-analysis.ipynb`: análise e leitura dos artefatos gerados.
- `notebooks/04-or-optimization-experiments.ipynb`: centraliza experimentos de Pesquisa Operacional (mean-CVaR + turnover).

Notebooks antigos/rascunhos podem ser movidos para `notebooks/archive/`.

## Como rodar

```bash
poetry install
poetry run python -m ipykernel install --user --name arara-quant-lab
poetry run jupyter lab
```

## Dica de workflow

- Gere dados/artefatos primeiro com `make reproduce-oos` (ou com os comandos do
  README principal) e use os notebooks para inspecionar `data/processed/` e
  `outputs/`.

# Configurações (`configs/`)

Este diretório contém arquivos YAML que parametrizam o comportamento do projeto
(universo, estimadores, otimização, backtest/walk-forward e rotinas de produção).

## Como usar

- **CLI (recomendado):**
  - `poetry run arara-quant optimize --config configs/optimizer_example.yaml`
  - `poetry run arara-quant backtest --config configs/optimizer_example.yaml --no-dry-run --wf-report`
- **Scripts:**
  - `poetry run python scripts/research/run_backtest_walkforward.py`

## Validação

Os YAMLs são validados por schemas (Pydantic) em `src/arara_quant/config/`.

- `make validate-configs`
- `poetry run python scripts/validate_configs.py`

## Arquivos principais (guia rápido)

- **Universo**
  - `configs/universe_arara.yaml`: universo ARARA padrão.
  - `configs/universe_arara_robust.yaml`: universo robusto (ex.: ajustes em cripto).
  - `configs/universe_liquid.yaml`: alternativa mais líquida/restrita.
- **Otimização + backtest (walk-forward)**
  - `configs/optimizer_example.yaml`: configuração de referência (exemplo e defaults).
  - `configs/optimizer_regime_aware.yaml`: variação com lógica de regimes.
  - `configs/optimizer_adaptive_hedge.yaml`: variação com overlay/hedge adaptativo.
  - `configs/backtest_full.yaml`: backtest mais completo (quando aplicável).
- **Exemplos (scripts em `scripts/examples/`)**
  - `configs/portfolio_arara_basic.yaml`: parâmetros “básicos” do exemplo.
  - `configs/portfolio_arara_robust.yaml`: parâmetros “robustos” do exemplo.
- **Produção**
  - `configs/production_erc_v2.yaml`: configuração recomendada (ERC calibrado).

## Convenções

- Caminhos relativos (ex.: `configs/universe_arara.yaml`) são resolvidos a partir
  do diretório do YAML ou do `project_root` definido em `Settings`.
- Quando `data.returns` não é fornecido, o backtest busca por padrão
  `data/processed/returns_arara.parquet` (gerado por `scripts/run_01_data_pipeline.py`).


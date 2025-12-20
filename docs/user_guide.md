# Guia do Usuário

Este guia fornece um passo-a-passo para rodar o pipeline completo da carteira
ARARA usando os *entrypoints* numerados em `scripts/`. Cada etapa pode ser
executada isoladamente durante o desenvolvimento ou encadeada para reproduzir
os resultados do relatório.

## 1. Ambiente

```bash
poetry install
poetry run pytest        # garante que os testes unitários passam
poetry run ruff check    # valida estilo
```

## 2. Pipeline principal

1. **Ingestão de dados**

   ```bash
   poetry run python scripts/run_01_data_pipeline.py
   ```

   - Verifica `data/raw/prices_arara.csv`; faz download via Yahoo Finance se o
     arquivo não existir ou se `--force-download` for passado.
   - Salva retornos logarítmicos em `data/processed/returns_arara.parquet`.

2. **Estimativa de parâmetros (μ/Σ)**

   ```bash
   poetry run python scripts/run_02_estimate_params.py --annualize
   ```

   - Usa janela padrão de 252 pregões e estimadores robustos (Huber + Ledoit-Wolf).
   - Escrita dos artefatos: `data/processed/mu_estimate.parquet` e
     `data/processed/cov_estimate.parquet`.

3. **Otimização da carteira**

   ```bash
   poetry run python scripts/run_03_optimize.py --risk-aversion 4.0 --max-weight 0.15
   ```

   - Resolve o programa média-variância com limites long-only e cap individual.
   - Exporta `outputs/results/optimized_weights.parquet` com os pesos ordenados.

4. **Backtest walk-forward**

   ```bash
   poetry run python scripts/run_04_backtest.py --output backtest_summary.json
   ```

   - Reutiliza `configs/optimizer_example.yaml` e gera métricas em `outputs/reports/`.
   - Use `--dry-run` para validar a configuração sem simular.

## 3. Testes

- **Unitários**: `poetry run pytest tests/unit`
- **Integração**: `poetry run pytest tests/integration`
- **Performance (opcional)**: adicione casos em `tests/performance` quando houver
  benchmarks estáveis.

## 4. Scripts legados

Os scripts anteriores continuam disponíveis em `scripts/legacy/`. Eles cobrem
experimentos específicos como `scripts/legacy/run_portfolio_arara_robust.py` e comparações de
estimadores. Atualize apontadores ou notebooks para o novo caminho,
por exemplo:

```bash
poetry run python scripts/legacy/run_portfolio_arara_robust.py
```

## 5. Próximos passos sugeridos

- Adicionar testes de performance para monitorar regressões de solver.
- Integrar geração automática de documentação a partir das docstrings em
  `docs/api/`.
- Referenciar este guia a partir do README para facilitar o onboarding.

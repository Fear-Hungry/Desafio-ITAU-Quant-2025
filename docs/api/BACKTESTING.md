# Backtesting API

Este documento lista os principais pontos de entrada do módulo
`arara_quant.backtesting` usados pelo pipeline e pela CLI.

## `run_backtest`

```python
from arara_quant.backtesting import run_backtest

result = run_backtest(
    config_path="configs/optimizer_example.yaml",
    dry_run=False,
)
print(result.metrics)
```

**Assinatura (alto nível):**

- `run_backtest(config_path: str | Path | None = None, *, dry_run: bool = True, ...) -> BacktestResult`

O `BacktestResult` encapsula métricas, ledger e (opcionalmente) séries temporais.
Para serialização, use `BacktestResult.to_dict(...)`.

## Tipos principais

- `BacktestConfig` (`arara_quant.backtesting.engine.BacktestConfig`)
- `BacktestResult` (`arara_quant.backtesting.engine.BacktestResult`)

## Walk-forward

O split temporal é gerado por:

- `generate_walk_forward_splits` (`arara_quant.backtesting.walk_forward`)

e o relatório (tabelas/plots) é consolidado em:

- `arara_quant.evaluation.walkforward_report`


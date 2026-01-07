# Optimization API

Este documento lista os principais pontos de entrada do módulo
`arara_quant.optimization` usados pelo pipeline e pela CLI.

## `run_optimizer`

```python
from arara_quant.optimization.solvers import run_optimizer

result = run_optimizer(
    config_path="configs/optimizer_example.yaml",
    dry_run=False,
)
print(result.status(), result.metrics)
```

**Assinatura (alto nível):**

- `run_optimizer(config_path: str | Path | None = None, *, dry_run: bool = True, ...) -> OptimizationResult`

O `OptimizationResult` retorna pesos (`pd.Series`) e um dicionário de métricas.
Para serialização, use `OptimizationResult.to_dict(include_weights=True)`.

## Mean-Variance (QP)

- `MeanVarianceConfig` (`arara_quant.optimization.core.mv_qp.MeanVarianceConfig`)
- `solve_mean_variance(mu: pd.Series, cov: pd.DataFrame, config: MeanVarianceConfig) -> MeanVarianceResult`

## CVaR (LP)

- `CvarConfig` (`arara_quant.optimization.core.cvar_lp.CvarConfig`)
- `solve_cvar_lp(returns: pd.DataFrame, mu: pd.Series, config: CvarConfig) -> CvarResult`

## Budgets e constraints

- Budgets por classe: `arara_quant.risk.budgets`
- Constraints gerais: `arara_quant.risk.constraints.build_constraints`


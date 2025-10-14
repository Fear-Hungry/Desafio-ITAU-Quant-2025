"""Blueprint for composable CVXPy constraint builders.

Objetivo
--------
Centralizar a geração de restrições reutilizáveis adotadas pelos diversos
problemas convexos (MV, CVaR, Sharpe, etc.).

Componentes sugeridos
---------------------
- `build_budget_constraints(weights, config)`:
    * garante ∑ w = 1, limites de alavancagem (∑ |w| ≤ L_max) e caixa mínima.
- `build_bound_constraints(weights, lower, upper)`:
    * bounds por ativo (ex.: 0 ≤ w_i ≤ 0.10) e por clusters (soma ≤ limites).
- `build_turnover_constraints(weights, prev_weights, max_turnover)`:
    * ∑ |w - w_prev| ≤ turnover_max.
- `build_sector_exposure_constraints(weights, sector_map, limits)`:
    * garante limites por setor/região/tema via matrizes de agregação.
- `build_risk_constraints(weights, cov, config)`:
    * Value-at-Risk, CVaR, volatilidade máxima, tracking error.
- `build_cost_terms(weights, trade_vars, cost_model)`:
    * retorna penalidades/auxiliares para custos lineares/quadráticos.
- `compose_constraints(problem_config)`:
    * Função orquestradora que chama helpers baseados em um dict/dataclass
      (ex.: `OptimizationConfig`) e retorna lista de `cvxpy.Constraint`.

Considerações
-------------
- Utilizar verificações assertivas (shape dos vetores, soma de limites).
- Permitir geração separada de penalidades (expressões cvxpy) para ser somada ao
  objetivo quando apropriado.
- Documentar cada tipo de restrição e sua motivação.

Testes recomendados
-------------------
- `tests/optimization/test_constraints_builder.py` com:
    * budget/box constraints simples,
    * limites setoriais,
    * checagem de turnover máximo,
    * comportamento quando config está ausente (deve ignorar restrição).
"""

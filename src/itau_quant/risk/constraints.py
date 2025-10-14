"""Blueprint for reusable portfolio risk constraints.

Objetivo
--------
Disponibilizar construtores modulares de restrições que podem ser plugados em
diversos otimizadores (MV, CVaR, Sharpe, GA).

Componentes sugeridos
---------------------
- `weight_sum_constraint(weights, target=1.0)`
    Garante soma dos pesos igual ao target.
- `box_constraints(weights, lower, upper)`
    Bounds por ativo, incluindo tratamento de ativos sem short (lower=0).
- `group_constraints(weights, groups, limits)`
    Limites por cluster/segmento (usa budgets do módulo `risk.budgets`).
- `factor_exposure_constraints(weights, factor_loadings, exposure_limits)`
    Restringe exposição a fatores sistemáticos.
- `leverage_constraint(weights, max_leverage)`
    Impõe ∑ |w| ≤ L_max.
- `tracking_error_constraint(weights, benchmark_weights, cov, max_te)`
    Controla tracking error via restrição quadrática.
- `turnover_constraint(weights, prev_weights, max_turnover)`
    Opcional: ∑ |w - w_prev| ≤ limite.
- `build_constraints(config, context)`
    Orquestrador: traduz config declarativa em lista de `cvxpy.Constraint`.

Considerações
-------------
- Garantir que cada função retorne lista de restrições e metadados para logging.
- Incluir asserts para shapes/dtypes e mensagens claras em caso de erro.
- Integrar com `optimization.core.constraints_builder` como camada subjacente.
- Permitir uso independente (ex.: validar portfólio ex-post).

Testes recomendados
-------------------
- `tests/risk/test_constraints.py` cobrindo:
    * soma de pesos = 1,
    * limites por setor com dados sintéticos,
    * tracking error calculado corretamente (usando QA manual),
    * comportamento com constraints conflitantes (deve lançar exceção).
"""

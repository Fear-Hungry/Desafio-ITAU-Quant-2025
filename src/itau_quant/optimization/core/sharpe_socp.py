"""Blueprint for Sharpe ratio maximisation via SOCP.

Objetivo
--------
Transformar a maximização do Sharpe em um problema SOCP (Second-Order Cone
Program) compatível com CVXPy, incorporando custos e restrições práticas.

Componentes sugeridos
---------------------
- `build_sharpe_socp(mu, cov, costs=None, turnover=None)`
    Cria a estrutura base (variáveis auxiliares, restrições, objetivo) segundo a
    formulação clássica: maximizar t, sujeito a ||Σ^{1/2} w||₂ ≤ 1, μᵀw ≥ t.
- `add_cost_terms(problem, trade_vars, cost_model)`
    Integra custos lineares/quadráticos na forma de penalidades ou restrições.
- `solve_sharpe_socp(mu, cov, config, solver_opts)`
    Função que monta, escolhe solver (ECOS, SCS, MOSEK) e retorna pesos.
- `normalize_weights(weights)`
    Garante soma 1 e tratamento de valores numéricos extremos.
- `sharpe_socp(mu, cov, config)`
    API pública orquestrando builder, solver e pós-processamento.

Considerações
-------------
- Tratar casos de covariância singular (usar regularização opcional).
- Permitir inclusão de retornos excessos (subtraindo rf antes de entrar no modelo).
- Usar `solver_utils` para logs, tolerâncias, escolha dinâmica de solver.
- Documentar parâmetros suportados (turnover_max, leverage, custos fixos).

Testes recomendados
-------------------
- `tests/optimization/test_sharpe_socp.py` com:
    * caso sem custos comparado a solução analítica (Sharpe clássico),
    * inclusão de custos verificando redução de turnover,
    * comportamento com cov quase singular (regularização),
    * checagem de normalização dos pesos.
"""

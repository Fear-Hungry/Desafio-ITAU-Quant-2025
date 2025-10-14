"""Blueprint for risk parity optimisation routines.

Objetivo
--------
Produzir pesos que igualam contribuições de risco (marginal × peso) entre ativos
ou clusters, com abordagens eficientes e estáveis.

Componentes sugeridos
---------------------
- `risk_contribution(weights, cov)`
    Calcula contribuições marginais e percentuais.
- `solve_log_barrier(cov, target_risk=None, bounds=None)`
    Formulação convexa (log-barrier) conforme Maillard/Roncalli.
- `iterative_risk_parity(cov, init_weights=None, tol=1e-6)`
    Método alternado multiplicativo (Riccati) para casos gerais.
- `cluster_risk_parity(cov, clusters)`
    Variante hierárquica agrupando ativos e aplicando risk parity por grupo.
- `risk_parity(weights_init, cov, config)`
    Wrapper que escolhe método, verifica convergência e aplica fallback (ex. HRP).

Considerações
-------------
- Garantir covariância positiva definida (usar `project_to_psd`).
- Normalizar pesos após cada iteração.
- Definir critérios de parada (norma do gradiente, variação relativa).
- Expor logs com progresso e motivo de fallback.

Testes recomendados
-------------------
- `tests/optimization/test_risk_parity.py` cobrindo:
    * caso bidimensional com solução analítica simples,
    * matriz quase singular (teste de robustez/fallback),
    * comparação com HRP para base sintética,
    * verificação de que contribuições de risco convergem para igualdade.
"""

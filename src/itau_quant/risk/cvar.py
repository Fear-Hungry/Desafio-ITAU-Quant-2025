"""Blueprint for CVaR (Conditional Value-at-Risk) optimisation helpers.

Objetivo
--------
Construir utilitários que formulam CVaR como problema linear/quadrático seguindo
Rockafellar-Uryasev, integrando com CVXPy e outros solvers.

Componentes sugeridos
---------------------
- `build_cvar_lp(returns_matrix, weights_var, alpha)`
    Cria variáveis auxiliares (VaR scalar, u_i) e restrições:
    u_i ≥ 0, u_i ≥ -r_iᵀ w - VaR, CVaR = VaR + (1/(1-α)) * mean(u_i).
- `cvar_objective(u_vars, var_var, alpha)`
    Retorna expressão CVaR para minimizar (ou incluir como restrição ≤ limite).
- `add_cvar_constraint(problem, max_cvar)`
    Impõe CVaR ≤ limite estabelecido.
- `solve_cvar_portfolio(returns_matrix, mu, alpha, config)`
    Exemplo de uso: minimização de CVaR com retorno mínimo.
- `historical_scenarios(prices_or_returns, window)`
    Helper para gerar matriz de cenários (log retornos) usada na formulação.
- `validate_cvar_inputs(returns_matrix, alpha)`
    Checagem de dimensões, NaNs, α ∈ (0,1).

Considerações
-------------
- Certificar-se de que a formulação é compatível com equivalentes do CVXPy.
- Permitir normalização por capital (CVaR em pontos percentuais).
- Integrar com `optimization.core.cvar_lp` (que monta o problema completo).
- Fornecer utilidades para sensitividade (ex.: gradiente dCVaR/dw).

Testes recomendados
-------------------
- `tests/risk/test_cvar.py` cobrindo:
    * comparação com cálculo manual para portfólio de 2 ativos,
    * validação de que a solução minimiza perdas de cauda conforme esperado,
    * checagem de limites (ex.: CVaR ≤ max_cvar) via solver mockado,
    * tratamento de α extremos (0.90, 0.99).
"""

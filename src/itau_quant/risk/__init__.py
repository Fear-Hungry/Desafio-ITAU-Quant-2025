"""Risk management utilities (budgets, constraints, CVaR, measures).

Componentes expostos
--------------------
- `budgets` → definição de limites por segmento/fator.
- `constraints` → construção de restrições reutilizáveis.
- `cvar` → formulação LP/convexa do CVaR.
- `measures` → métricas puras de risco/performance.

Importe via ``from itau_quant.risk import ...`` para manter encapsulamento das
rotinas internas.
"""

"""Blueprint for portfolio risk budgets by segment/factor.

Objetivo
--------
Modelar limites de alocação por classes (setores, fatores, regiões) e expor
funções que traduzam esses limites em restrições/diagnósticos reutilizáveis.

Componentes sugeridos
---------------------
- `RiskBudget` dataclass\n
    Campos: `name`, `tickers`, `min_weight`, `max_weight`, `target`, `tolerance`.\n
- `load_budgets(config)`\n
    Constrói lista de budgets a partir da configuração (YAML/JSON) do projeto.\n
- `validate_budgets(budgets, universe)`\n
    Assegura que conjuntos são disjuntos (ou lida com sobreposições) e que os\n
    limites são consistentes (ex.: soma dos máximos ≥ 1).\n
- `budgets_to_constraints(budgets, weights_var)`\n
    Retorna restrições CVXPy do tipo ∑_{i∈grupo} w_i ≤ max_weight, ≥ min_weight.\n
- `budget_slack(weights, budgets)`\n
    Calcula excedentes/folgas para reporting (ex.: Slack positivo indica espaço).\n
- `aggregate_by_budget(weights, returns, budgets)`\n
    Consolida métricas (retorno, risco) por grupo para dashboards.\n

Considerações
-------------\n
- Permitir hierarquias (ex.: grupo pai "Renda Fixa" e subgrupos "IG", "HY").\n
- Suportar budgets dinâmicos (dependem do regime de mercado) via função callback.\n
- Integrar com `optimization.core.constraints_builder` e `portfolio.rebalancer`.\n
- Garantir documentação clara de como budgets interagem com outras restrições.\n

Testes recomendados
-------------------\n
- `tests/risk/test_budgets.py` cobrindo:\n
    * conversão para restrições CVXPy (soma dentro dos limites),\n
    * validação de budgets inválidos (max < min),\n
    * agregação de métricas por grupo retornando totais corretos.\n
"""

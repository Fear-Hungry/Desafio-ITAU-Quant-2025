"""Blueprint for auxiliary penalty functions (CVXPy compatible).

Objetivo
--------
Fornecer termos de penalização/regularização que podem ser adicionados ao
objetivo dos problemas convexos, modelando preferências adicionais.

Componentes sugeridos
---------------------
- `l1_penalty(weights, gamma)`
    Penaliza soma de valores absolutos (sparsidade suave).
- `l2_penalty(weights, gamma)`
    Ridge / Tikhonov para regularizar norma L2.
- `group_lasso_penalty(weights, groups, gamma)`
    Penaliza soma das normas L2 por grupo (clusters setoriais, fatores).
- `cardinality_soft_penalty(weights, k_target, gamma, method="log")`
    Aproximação suave para cardinalidade (log, SCAD, MCP, etc.).
- `turnover_penalty(weights, prev_weights, gamma)`
    Penaliza movimentação de carteira.
- `penalty_factory(config)`
    Retorna lista de termos a partir de configurações declarativas.

Considerações
-------------
- Todas as funções devem retornar expressões CVXPy prontas para somar ao
  objetivo: ex. `gamma * cp.norm1(weights)`.
- Documentar efeitos esperados e compatibilidade com diferentes solvers.
- Permitir combinações múltiplas (L1 + turnover + grupo) de forma simples.

Testes recomendados
-------------------
- `tests/optimization/test_penalties.py` verificando:
    * expressão retornada é convexa (`expr.is_convex()`),
    * cardinalidade suave entrega gradiente finito,
    * penalização de grupos responde a clusters vazios/nulos.
"""

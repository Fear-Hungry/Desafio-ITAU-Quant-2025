"""Blueprint for cardinality control heuristic (MIQP-free).

Objetivo
--------
Reduzir o número de ativos (||w||₀ ≤ K) sem resolver problemas inteiros,
via heurísticas eficientes que interagem com o núcleo convexo.

Abordagens sugeridas
--------------------
- `greedy_selection(mu, cov, costs, k)`\n
    Seleciona ativos iterativamente maximizando ganho incremental (ex.: contribuição\n
    ao Sharpe) até atingir K.\n
- `beam_search_selection(mu, cov, costs, k, beam_width)`\n
    Mantém múltiplos candidatos em cada nível para escapar de ótimos locais.\n
- `prune_after_optimisation(weights, k, method=\"magnitude\")`\n
    Após resolver o problema contínuo, mantém apenas os maiores pesos e renormaliza.\n
- `reoptimize_with_subset(subset, data, core_solver)`\n
    Rodar novamente o núcleo convexo no subconjunto escolhido para refinar pesos.\n
- `cardinality_pipeline(data, k, strategy, config)`\n
    Orquestra heurísticas pré ou pós-otimização, escolhendo a abordagem conforme\n
    trade-off tempo x qualidade.\n

Considerações
-------------\n
- Garantir balanço entre diversificação e performance (ex.: penalizar ativos muito correlacionados).\n
- Permitir imposição de limites por setor/região (ex.: ao selecionar, respeitar mix).\n
- Lidar com custos de transação (ex.: preferir ativos já na carteira atual).\n
- Retornar tanto subconjunto quanto pesos reotimizados e estatísticas (turnover, Sharpe).\n

Testes recomendados
-------------------\n
- `tests/optimization/heuristics/test_cardinality.py` cobrindo:\n
    * heurística gulosa em caso pequeno comparado ao ótimo MIQP conhecido,\n
    * beam search evitando escolha redundante em ativos correlacionados,\n
    * pós-processamento (prune + reoptimize) mantendo soma = 1.\n
"""

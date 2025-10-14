"""Blueprint for GA fitness evaluation.

Objetivo
--------
Avaliar cada indivíduo do GA executando o núcleo de otimização e convertendo
o resultado em escalar de fitness (retorno ajustado ao risco/custos).

Componentes sugeridos
---------------------
- `build_candidate_solution(individual, data, config)`
    Prepara inputs para o núcleo convexo (subconjunto de ativos, parâmetros).
- `run_core_optimizer(candidate, core_solver)`
    Invoca `optimization.core` (MV, CVaR, Sharpe) conforme tipo definido no
    indivíduo/config, capturando pesos, métricas e status.
- `compute_fitness(weights, metrics, penalties, config)`
    Combina retorno esperado, risco, custos, turnover e penalidades (cardinalidade,
    concentração) em escalar. Possível fórmula: `fitness = sharpe - γ_cost*turnover`.
- `evaluate_population(population, data, core_solver, config, parallel=False)`
    Loop vectorizado/paralelo que retorna lista de resultados por indivíduo
    (fitness, pesos, métricas detalhadas, status).
- `handle_failures(individual, error, fallback_strategy)`
    Trata casos em que solver falha (ex.: atribuir fitness muito baixo, tentar HRP).

Considerações
-------------
- Registrar tempo de execução e número de falhas para monitorar performance.
- Permitir caching de resultados se um indivíduo for avaliado novamente.
- Assegurar que fitness seja maior/igual para melhores soluções (monotonicidade).
- Incluir métricas extras: drawdown, CVaR, turnover, diversidade dos ativos.

Testes recomendados
-------------------
- `tests/optimization/ga/test_evaluation.py` com mocks:
    * núcleo convexo retornando pesos previsíveis,
    * fitness respondendo a diferentes configurações (ex.: penalidade de custo),
    * tratamento de exceções (solver failure → fitness reduzido).
"""

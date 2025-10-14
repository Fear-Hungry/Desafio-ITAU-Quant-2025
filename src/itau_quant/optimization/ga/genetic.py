"""Blueprint for the hybrid genetic algorithm main loop.

Objetivo
--------
Meta-heurística que combina busca evolutiva (subconjunto de ativos, hyperparams)
com núcleos convexos (ex.: CVaR/MV) para encontrar carteiras robustas.

Fluxo sugerido
--------------
1. `initialize_population(config)`
    - Usa utilitários de `population.py` para gerar indivíduos com diversidade
      (subconjuntos distintos, hiperparâmetros variados).
2. `evaluate_population(pop, data)`
    - Avalia fitness via `evaluation.py` (roda solver convexo, computa métricas).
3. `select_parents(pop, fitness)`
    - Estratégias de seleção (torneio/roleta) definidas em `selection.py`.
4. `apply_crossover(parents)`
    - Combinação de genes (ativos/param) usando operadores de `crossover.py`.
5. `apply_mutation(children)`
    - Perturbações controladas via `mutation.py` (flip de ativos, jitter em lambdas).
6. `elitism(pop, fitness, elite_ratio)`
    - Preserva melhores indivíduos para próxima geração.
7. `termination_criteria(history, config)`
    - Verifica convergência: max generations, estagnação, tempo máximo.

Considerações
-------------
- Integrar logs (melhor fitness por geração, diversidade).
- Permitir paralelização da avaliação (multiprocessing ou joblib).
- Interagir com núcleo convexo respeitando limites (timeout, max_failures).
- Guardar histórico completo (pesos, parâmetros, métricas) para report.

Testes recomendados
-------------------
- `tests/optimization/ga/test_genetic.py` cobrindo:
    * evolução básica em cenário sintético (fitness melhora ao longo das gerações),
    * tratamento de parâmetros inválidos (raises claros),
    * integração com mocks de `evaluation` para testar fallback quando solver falha.
"""

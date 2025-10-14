"""Blueprint for genetic algorithm selection mechanisms.

Objetivo
--------
Escolher indivíduos (pais) para reprodução balanceando exploração/exploração.

Métodos sugeridos
-----------------
- `tournament_selection(pop, fitness, tournament_size, rng)`
    Seleção por torneio (melhor entre subconjunto aleatório).
- `roulette_wheel_selection(pop, fitness, rng)`
    Probabilidade proporcional ao fitness (roulette).
- `stochastic_universal_sampling(pop, fitness, rng)`
    Variante mais estável da roleta.
- `diversity_preserving_selection(pop, fitness, diversity_metric, rng)`
    Favorece indivíduos diversos (ex.: distância Hamming entre subsets).
- `selection_pipeline(pop, fitness, config)`
    Composição/fallback de múltiplas estratégias.

Considerações
-------------
- Normalizar fitness quando necessário (ex.: garantir valores positivos).
- Incluir elitismo (manter top N) separadamente no loop principal.
- Metric de diversidade pode usar Jaccard entre subconjuntos de ativos.

Testes recomendados
-------------------
- `tests/optimization/ga/test_selection.py` com:
    * probabilidade crescente para indivíduos com fitness maior,
    * comportamento determinístico dado seed,
    * preservação de diversidade mínima quando configurada.
"""

"""Blueprint for genetic algorithm crossover operators.

Objetivo
--------
Definir estratégias de recombinação entre indivíduos para explorar novas
configurações de ativos e hiperparâmetros.

Operadores sugeridos
--------------------
- `single_point_crossover(parent_a, parent_b, rng)`
    Combina subconjuntos dividindo o vetor de decisão em um ponto.
- `uniform_crossover(parent_a, parent_b, prob=0.5)`
    Escolhe gene a gene (ativo, lambda, tau) com probabilidade uniforme.
- `blend_crossover(params_a, params_b, alpha)`
    Para hiperparâmetros contínuos (lambda, eta) usar BLX-α ou SBX.
- `subset_exchange(parent_a, parent_b, k)`
    Troca subconjunto fixo de ativos entre pais.
- `crossover_factory(config)`
    Seleciona operador(es) conforme configuração ou combina múltiplos com pesos.

Considerações
-------------
- Garantir que resultados respeitem constraints básicas (ex.: cardinalidade ≤ K).
- Normalizar pesos/hiperparâmetros após o crossover (ex.: clip em ranges válidos).
- Manter reprodutibilidade usando `numpy.random.Generator` passado como argumento.

Testes recomendados
-------------------
- `tests/optimization/ga/test_crossover.py` com:
    * validação de que o output possui o mesmo número de genes,
    * manutenção de limites (probabilidade, ranges),
    * comportamento determinístico com seed fixo.
"""

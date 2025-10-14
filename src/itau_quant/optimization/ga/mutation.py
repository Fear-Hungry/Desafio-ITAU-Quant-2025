"""Blueprint for genetic algorithm mutation operators.

Objetivo
--------
Adicionar diversidade controlada à população evitando convergência prematura.

Operadores sugeridos
--------------------
- `flip_asset_selection(individual, prob, rng)`
    Inverte participação de ativos com probabilidade baixa (bit-flip).
- `gaussian_jitter_params(individual, sigma, bounds, rng)`
    Adiciona ruído gaussiano a hiperparâmetros contínuos (lambda, eta).
- `discrete_adjustment(individual, param_name, values, prob)`
    Ajusta parâmetros discretos (K clusters) escolhendo novo valor.
- `swap_assets(individual, available_assets, rng)`
    Substitui ativos menos relevantes por novos candidatos.
- `mutation_pipeline(individual, config, rng)`
    Combina operadores acima controlando taxa de mutação global.

Considerações
-------------
- Garantir limites após mutação (clip em bounds, cardinalidade ≤ K).
- Possibilitar mutações adaptativas (probabilidade decai à medida que fitness melhora).
- Registrar alterações para debug (quais genes foram mutados).

Testes recomendados
-------------------
- `tests/optimization/ga/test_mutation.py` com:
    * validação de bounds pós-mutation,
    * impacto na diversidade (ex.: contagem de ativos únicos),
    * determinismo com seed fixo.
"""

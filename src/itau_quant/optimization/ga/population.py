"""Blueprint for GA individual representation and population initialisation.

Objetivo
--------
Definir como um indivíduo é codificado (ativos selecionados + hiperparâmetros)
e como gerar populações iniciais diversas e factíveis.

Componentes sugeridos
---------------------
- `Individual` (dataclass/NamedTuple)
    Contém campos como: `assets_mask`, `lambda_risk`, `eta_turnover`, `tau_bl`,
    `k_clusters`, `penalty_weights`, `metadata`.
- `encode_individual(assets, params, universe)`
    Converte uma lista de tickers e dict de parâmetros em representação padrão
    (ex.: máscara booleana ordenada pelo universo).
- `decode_individual(individual, universe)`
    Retorna lista de ativos selecionados e dicionário de hiperparâmetros.
- `random_individual(universe, config, rng)`
    Amostra subconjunto respeitando cardinalidade mínima/máxima e gera
    hiperparâmetros dentro de ranges (`config.hyperparams`).
- `diversified_population(universe, config, size, rng)`
    Cria população inicial mesclando estratégias: uniformemente por setor,
    enviesada por liquidez, baseada em loadings de PCA ou carteiras históricas.
- `warm_start_population(universe, historical_weights, config)`
    Gera indivíduos a partir de soluções existentes (ex.: MV, HRP) convertendo
    pesos em máscaras + hiperparâmetros default.
- `ensure_feasible(individual, constraints)`
    Ajusta cardinalidade, bounds e outros limites antes de avaliação (ex.: se
    número de ativos > K, remove os menores pesos).

Considerações
-------------
- Usar `numpy.random.Generator` injetado para repetibilidade.
- Permitir anexar metadados (origem: random, warm_start, mutation) úteis em logs.
- Prever serialização (`to_dict`/`from_dict`) para checkpoints do GA.
- Validar inputs e lançar erros claros (ex.: universo vazio, ranges inválidos).

Testes recomendados
-------------------
- `tests/optimization/ga/test_population.py` cobrindo:
    * encoding/decoding idempotente pelo universo fornecido,
    * cardinalidade respeitada nos indivíduos aleatórios,
    * diversidade (ex.: métricas de Jaccard entre indivíduos),
    * warm start reproduzindo subconjuntos de carteiras conhecidas.
"""

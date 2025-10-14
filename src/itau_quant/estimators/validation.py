"""Model validation utilities blueprint (purged/embargoed CV).

Objetivo
--------
Evitar leakage ao avaliar estimadores/otimizadores com dados temporais.

Componentes sugeridos
---------------------
- `temporal_split(index, n_splits, min_train, min_test)`:
    Gera janelas rolling/expanding respeitando ordem temporal.
- `purge_train_indices(train_idx, test_idx, purge_window)`:
    Remove observações treinadas que precedem imediatamente o bloco de teste.
- `apply_embargo(train_idx, test_idx, embargo_pct)`:
    Retira observações pós-teste para evitar feedback (López de Prado).
- `PurgedKFold` (classe):
    Implementa interface scikit-learn (``split``/``get_n_splits``) aplicando as
    funções acima.
- `evaluate_estimator(estimator_fn, data, scoring, splitter)`:
    Função helper para rodar cross-validation purged/embargoed de forma padrão.

Considerações
-------------
- `index` deverá ser `DatetimeIndex` ou array ordenado crescente.
- Suportar diferentes janelas de purging (dias) e embargo percentual.
- Garantir que conjuntos de treino/teste não se sobreponham.
- Logar/emitir exceções quando conjuntos forem muito pequenos.

Testes recomendados
-------------------
- `tests/estimators/test_validation.py` com:
    * verificação de não sobreposição (treino ∩ teste = ∅),
    * checagem de purging correto dado um ``purge_window`` em dias,
    * comportamento com séries curtas (deve levantar erro claro),
    * integração com um estimador dummy (ex.: média) conferindo pipeline completo.
"""

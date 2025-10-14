"""Blueprint for optimisation post-processing utilities.

Objetivo
--------
Garantir que os pesos resultantes dos solvers atendam às restrições práticas
antes de serem enviados para backtest ou execução.

Componentes sugeridos
---------------------
- `project_to_simplex(weights)`
    Projeta vetor em simplex (∑ w = 1, w_i ≥ 0).
- `clip_to_bounds(weights, lower, upper)`
    Aplica limites inferiores/superiores garantindo consistência numérica.
- `rebalance_equal_weight(selected_assets)`
    Distribui igualmente entre conjunto de ativos selecionados.
- `round_to_lots(weights, lot_size)`
    Ajusta pesos para múltiplos discretos (round lots, fractional shares).
- `enforce_cardinality(weights, k)`
    Zera ativos de menor peso mantendo apenas os maiores K.
- `postprocess_pipeline(weights, config)`
    Orquestra as etapas acima conforme ``OptimizationConfig``.

Considerações
-------------
- Preservar a soma total após cada transformação (renormalizar se preciso).
- Garantir que outputs mantenham tipos (Series/DataFrame) com índice original.
- Lidar com `NaN` substituindo por 0 antes de projetar.
- Permitir logging das etapas (antes/depois) para auditoria.

Testes recomendados
-------------------
- `tests/optimization/test_postprocess.py` cobrindo:
    * projeção no simplex (soma = 1, não-negativos),
    * clipping respeitando bounds customizados,
    * round lots resultando em soma próxima do target,
    * pipeline completo com múltiplas transformações.
"""

"""Blueprint for rounding and minimum-lot adjustments.

Objetivo
--------
Transformar pesos contínuos produzidos pelos otimizadores em ordens viáveis,
respeitando tamanhos mínimos de lote, custos de arredondamento e limitações de
capital disponível.

Componentes sugeridos
---------------------
- `weights_to_shares(weights, capital, prices)`
    Converte pesos para quantidade de shares/contracts antes do arredondamento.
- `round_to_lots(shares, lot_size_map, method="nearest")`
    Aplica arredondamento para múltiplos de lote usando diferentes critérios
    (mais próximo, para baixo, heurística de custo mínimo).
- `shares_to_weights(shares, capital, prices)`
    Recalcula pesos após arredondamento.
- `allocate_residual_cash(weights, capital, prices, priority="largest_weight")`
    Distribui residual de caixa (positivo ou negativo) respeitando bounds.
- `estimate_rounding_costs(original_weights, rounded_weights, prices, cost_model)`
    Quantifica custo extra (slippage, spreads) decorrente do ajuste.
- `rounding_pipeline(weights, prices, capital, config)`
    Orquestra as etapas acima, retornando `RoundedPortfolioResult` com pesos
    finais, transações, custos e diagnósticos.

Considerações
-------------
- Suportar ativos sem lotes mínimos (lot=1) e segmentos fracionários (lot < 1).
- Tratar ativos inviáveis (preço > capital disponível) zerando e documentando.
- Garantir soma dos pesos ≈ 1 (ou sinalizada quando residuais são levados ao
  caixa separado).
- Integrar com `portfolio.rebalancer` e `optimization.core.postprocess`.
- Expor logs detalhados por ativo (peso original, arredondado, delta, custo).

Testes recomendados
-------------------
- `tests/portfolio/test_rounding.py` cobrindo:
    * conversão pesos→shares→pesos mantendo consistência numérica,
    * diferentes métodos de arredondamento (nearest vs. floor),
    * distribuição de residual de caixa e respeito a bounds,
    * cálculo correto do custo adicional em cenários sintéticos.
"""

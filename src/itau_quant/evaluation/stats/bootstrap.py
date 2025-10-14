"""Blueprint for bootstrap-based confidence intervals.

Objetivo
--------
Mensurar incerteza das métricas da estratégia usando reamostragem apropriada
para séries temporais (block/bootstrap estacionário).

Componentes sugeridos
---------------------
- `block_bootstrap(returns, block_size, n_samples, random_state=None)`
    Gera trajetórias reamostradas preservando autocorrelação intra-bloco.
- `stationary_bootstrap(returns, p, n_samples, random_state=None)`
    Variante com tamanhos de bloco aleatórios (Politis & Romano).
- `bootstrap_metric(metric_fn, returns, n_samples, **kwargs)`
    Avalia uma métrica (ex.: Sharpe, drawdown) nas amostras reamostradas.
- `confidence_interval(samples, alpha=0.05, method="percentile")`
    Calcula ICs (percentil, BCa) e retorna dict com limites inferior/superior.
- `compare_vs_benchmark(metric_samples, benchmark_metric)`
    Estatística de p-valor ou probabilidade de outperformance vs. 1/N.

Considerações
-------------
- Suportar DataFrames multi-ativos (bootstrap por coluna ou portfólio agregado).
- Validar que block_size ≤ len(returns) e tratar séries curtas.
- Permitirem seeds reprodutíveis via numpy Generator.
- Integrar com ``evaluation.stats.performance`` para métricas compostas.

Testes recomendados
-------------------
- `tests/evaluation/stats/test_bootstrap.py` cobrindo:
    * reamostragem preservando média aproximada em dados IID,
    * comportamento com block_size=1 (deve equivaler ao bootstrap clássico),
    * cálculo correto de ICs para métricas conhecidas,
    * comparação com benchmark retornando valores coerentes (0.5 em casos simétricos).
"""

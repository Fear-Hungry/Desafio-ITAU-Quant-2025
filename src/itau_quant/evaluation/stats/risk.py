"""Blueprint for realised risk metrics.

Objetivo
--------
Quantificar risco observado da estratégia e decompor contribuições por ativo.

Componentes sugeridos
---------------------
- `max_drawdown(returns)`
    Calcula drawdown máximo e série completa para integração com plots.
- `conditional_value_at_risk(returns, alpha=0.95, method="historical")`
    CVaR (expected shortfall) para retornos ou P&L.
- `tracking_error(strategy_returns, benchmark_returns, periods_per_year)`
    Desvio padrão do excesso de retorno em relação a benchmark.
- `realized_leverage(weights, prices)`
    Estima alavancagem atingida (∑ |peso|) ou notional em relação ao patrimônio.
- `risk_contribution(weights, cov)`
    Calcula contribuição marginal e percentual por ativo/cluster.
- `beta_to_benchmark(strategy_returns, benchmark_returns)`
    Beta histórico (regressão simples).
- `aggregate_risk_metrics(returns, weights=None, benchmark=None, cov=None)`
    Consolida resultados em DataFrame amigável.

Considerações
-------------
- Garantir consistência de frequências (retornos diários x covariância anualizada).
- Tratar séries com NaNs aplicando ``dropna`` sincronizado.
- Oferecer opções de smoothing para CVaR (ex.: janela rolling).
- Compatibilizar saída com `evaluation.report` e `evaluation.plots`.

Testes recomendados
-------------------
- `tests/evaluation/stats/test_risk.py` cobrindo:
    * drawdown de série conhecida (ex.: monotonicamente crescente → drawdown zero),
    * CVaR comparado a implementação manual para distribuição simples,
    * tracking error retornando zero quando estratégia=benchmark,
    * risk contribution somando ao risco total estimado.
"""

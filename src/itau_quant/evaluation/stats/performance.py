"""Blueprint for performance metrics computation.

Objetivo
--------
Calcular carteira de métricas padronizadas que resumem performance absoluta e
relativa da estratégia.

Componentes sugeridos
---------------------
- `annualized_return(returns, periods_per_year)`
    Converte retorno acumulado para base anual, tratando valores faltantes.
- `annualized_volatility(returns, periods_per_year)`
    Volatilidade com ajuste de frequência.
- `sharpe_ratio(returns, rf=0.0, method="HAC")`
    Sharpe com correção de autocorrelação (Newey-West / HAC) opcional.
- `sortino_ratio(returns, rf=0.0, target=0.0)`
    Downside risk via semidesvio.
- `calmar_ratio(returns)`
    Retorno CAGR dividido pelo drawdown máximo.
- `hit_rate(returns)`
    Proporção de períodos positivos.
- `excess_vs_benchmark(strategy_returns, benchmark_returns)`
    Medidas relativas (alpha, tracking error, information ratio simplificado).
- `aggregate_performance(returns, benchmark=None, periods_per_year=252)`
    Wrapper que computa todas as métricas e devolve DataFrame/Series amigável.

Considerações
-------------
- Tratar séries com NaNs (ex.: usar alinhamento com `dropna()` ou fill adequado).
- Definir convenções claras (retornos em decimal, rf na mesma base temporal).
- Permitir múltiplas colunas (portfolios diferentes) com resultado multi-index.
- Documentar unidades de saída (ex.: volatilidade anualizada).

Testes recomendados
-------------------
- `tests/evaluation/stats/test_performance.py` cobrindo:
    * equivalência com resultados conhecidos (Sharpe=1 em série ideal),
    * manipulação de benchmark vs. estratégia,
    * robustez a séries curtas (mensagens claras),
    * diferentes frequências (diária, mensal) ajustando periods_per_year.
"""

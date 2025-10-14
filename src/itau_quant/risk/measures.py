"""Blueprint for pure risk/return measures.

Objetivo
--------
Oferecer funções determinísticas para calcular métricas de risco e desempenho
ajustado, reutilizadas em avaliação, backtesting e triggers.

Componentes sugeridos
---------------------
- `volatility(returns, periods_per_year)`\n
    Volatilidade anualizada (std) com tratamento de NaNs.\n
- `sharpe_ratio(returns, rf=0.0, method=\"HAC\", periods_per_year=252)`\n
    Sharpe tradicional ou com correção de autocorrelação (Heteroskedasticity\n
    and Autocorrelation Consistent - Newey-West).\n
- `sortino_ratio(returns, rf=0.0, target=0.0, periods_per_year=252)`\n
    Usa semidesvio inferior para medir risco downside.\n
- `max_drawdown(nav_series)`\n
    Retorna drawdown máximo, duração e série completa.\n
- `historical_cvar(returns, alpha=0.95)`\n
    CVaR empírico (Expected Shortfall) calculado diretamente dos retornos.\n
- `tracking_error(strategy_returns, benchmark_returns, periods_per_year=252)`\n
    Desvio padrão do excesso de retorno.\n
- `information_ratio(strategy_returns, benchmark_returns, periods_per_year=252)`\n
    Relação entre excesso e tracking error.\n
- `rolling_metric(series, window, func)`\n
    Helper genérico para gerar métricas rolling (ex.: rolling Sharpe/drawdown).\n

Considerações
-------------\n
- Funções devem aceitar Series/DataFrames e preservar rótulos.\n
- Garantir robustez a séries curtas (retornar NaN ou lançar erro claro).\n
- Documentar unidades de saída (ex.: retorno anualizado em decimal).\n
- Não depender de estado global nem mutar inputs (puras).\n
- Compartilhadas com `evaluation.stats` e `portfolio.triggers`.\n

Testes recomendados
-------------------\n
- `tests/risk/test_measures.py` cobrindo:\n
    * comparação com cálculos conhecidos (Sharpe=1 em série 45°),\n
    * handling de retornos todos iguais (vol=0, ratio → inf/NaN controlado),\n
    * validação do CVaR empírico vs. percentis manuais,\n
    * tracking error/information ratio com benchmark idêntico (TE=0).\n
"""

"""Blueprint for standard performance tearsheets.

Objetivo
--------
Produzir conjunto canônico de gráficos/tabelas que resumem o desempenho da
estratégia em formato comparável entre experimentos.

Componentes sugeridos
---------------------
- `plot_cumulative_returns(returns, benchmark=None)`
    Curvas acumuladas da estratégia vs. benchmark (com shading de drawdowns).
- `plot_drawdown(returns)`
    Série temporal do drawdown máximo e highlights das piores quedas.
- `plot_rolling_sharpe(returns, window=252)`
    Sharpe rolling com janela configurável e intervalo de confiança.
- `plot_rolling_volatility(returns, window=252)`
    Volatilidade realizada acumulada.
- `plot_risk_contribution(weights, cov)`
    Barras stack ou heatmap mostrando contribuição marginal e percentual dos
    ativos/ clusters de risco.
- `plot_turnover(orders)`
    Evolução do turnover vs. limites/custos.
- `generate_tearsheet(figures, layout="grid")`
    Monta layout final (ex.: Matplotlib subplots ou Dash/Plotly) pronto para
    exportação.

Requisitos adicionais
---------------------
- Oferecer versões estáticas (Matplotlib) e interativas (Plotly) quando possível.
- Garantir acessibilidade (cores, legendas, formatos de data).
- Funções devem aceitar ``Axis`` opcional para integração com notebooks.
- Permitir exportar em carrosel ou PDF via `evaluation.report`.

Testes recomendados
-------------------
- `tests/evaluation/plots/test_tearsheet.py` com dados dummy verificando:
    * que cada função retorna figura/axis válidos,
    * manuseio de benchmarks e valores ausentes,
    * consistência no número de pontos desenhados vs. série fornecida.
"""

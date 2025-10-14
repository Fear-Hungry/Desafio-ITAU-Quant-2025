"""Blueprint for strategy diagnostic visualisations.

Objetivo
--------
Criar gráficos que auxiliem a auditar estabilidade, sensibilidade e sinais da
estratégia para além da tearsheet tradicional.

Gráficos sugeridos
------------------
- `plot_weight_stability(weights, rolling_window)`
    Heatmap/line chart mostrando variação de pesos ao longo do tempo.
- `plot_signal_distribution(signals)`
    Histogramas/violin plots de sinais (score de fatores, exposures) com destaque
    para caudas e porcentagem de posições long/short.
- `plot_parameter_sensitivity(results, param_grid)`
    Visualizações (contour/scatter) comparando métricas (Sharpe/CVaR) enquanto
    parâmetros críticos variam (lambda risk-aversion, eta turnover, tau BL, K clusters).
- `plot_turnover_vs_cost(weights, costs)`
    Relaciona turnover realizado com custos estimados para entender impactos.
- `plot_drawdown_contributors(drawdown_series, weight_history)`
    Identifica quais ativos contribuíram para drawdowns específicos.

Requisitos
----------
- Usar Matplotlib ou Plotly com tema consistente e legendas claras.
- Garantir compatibilidade com notebooks e exportação para PNG/SVG.
- Incluir funções helper para salvar gráficos em diretórios específicos
  (`reports/figures`, por exemplo).
- Os gráficos devem receber DataFrames com índices de datas para facilitar
  alinhamento com outras métricas.

Testes recomendados
-------------------
- `tests/evaluation/plots/test_diagnostics.py` cobrindo:
    * geração sem erro com dados sintéticos,
    * validação básica das dimensões (ex.: peso vs. índice temporal),
    * comportamento quando dados estão vazios (mensagem amigável ou gráfico vazio).
"""

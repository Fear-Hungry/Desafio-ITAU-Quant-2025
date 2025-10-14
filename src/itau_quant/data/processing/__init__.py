"""Transformações core para padronizar dados de mercado.

Submódulos
----------
`clean`\n
    - Conversion helpers (`ensure_dtindex`, `normalize_index`).\n
    - Validação de painéis (`validate_panel`).\n
    - Estatísticas/filtros de liquidez (`compute_liquidity_stats`, `filter_liquid_assets`).\n
    - Tratamento de outliers (`winsorize_outliers`).\n
\n
`returns`\n
    - Cálculo de retornos log/percentuais (`calculate_returns`).\n
    - Derivação de excess returns vs. risk-free (`compute_excess_returns`).\n
\n
`calendar`\n
    - Agendas de rebalance baseadas no índice observado (BMS/BME/Weekly).\n
    - Navegação next/previous trading day e clamp de datas.\n
\n
`corporate_actions`\n
    - Blueprint para ajustes retroativos (splits/dividendos) ainda a implementar.\n
"""

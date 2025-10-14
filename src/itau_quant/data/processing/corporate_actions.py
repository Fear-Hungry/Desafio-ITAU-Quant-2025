# Placeholder implementation notice for future contributors:
# - Escopo: encapsular ajustes retroativos (splits, dividendos em dinheiro,
#   spin-offs, fusões) garantindo continuidade das séries.
# - API mínima proposta:
#     * `load_corporate_actions(tickers, start=None, end=None, source="tiingo")`
#           - Retorna DataFrame estruturado: columns `event_type`, `ex_date`,
#             `effective_date`, `ratio`, `cash_amount`, `ticker`.
#     * `calculate_adjustment_factors(actions_df, index)`
#           - Constrói fatores cumulativos para preço e volume alinhados ao índice
#             do painel principal (usa `ensure_dtindex`).
#     * `apply_price_adjustments(prices, factors)`
#           - Aplica fatores multiplicativos nos preços históricos; idem para
#             volumes caso disponíveis.
#     * `apply_return_adjustments(returns, dividends)`
#           - Ajusta retornos simples quando somente ``Close`` está disponível.
# - Integração com `DataLoader`:
#     * Rodar logo após `normalize_index` e antes de `filter_liquid_assets` para
#       não impactar métricas de liquidez.
#     * Persistir fatores em `data/processed/corporate_actions/` usando
#       `storage.save_parquet` e hash de requisição para auditoria.
# - Considerações técnicas:
#     * Suportar merges de múltiplos eventos no mesmo dia (ex.: dois splits).
#     * Dividends podem ser em cash ou percentuais; alinhar com timezone tz-naive.
#     * Permitir fallback para arquivos CSV locais quando APIs não disponíveis.
# - Testes recomendados (`tests/data/processing/test_corporate_actions.py` futuro):
#     * Split 2:1 seguido de dividendo extraordinário.
#     * Spin-off que requer fator proporcional (ex.: ticker original perde 20%).
#     * Eventos inexistentes para parte dos tickers (fatores = 1).

"""Ajusta precos para eventos corporativos.

Implementar rotinas para aplicar splits, dividendos e outros eventos quando a
fonte nao entrega dados ajustados, preservando continuidade de series.
"""

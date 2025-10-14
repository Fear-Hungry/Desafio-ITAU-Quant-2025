# Placeholder implementation notice for future contributors:
# - Objetivo: replicar o padrão de `sources.yf`/`sources.csv` para ETFs/ETNs
#   cripto, provendo dados prontos para `processing.clean`.
# - API sugerida:
#     * `download_crypto_prices(tickers, start=None, end=None, provider="tiingo", fields=("Adj Close",))`
#          - Retornar DataFrame wide com índice diário.
#          - Permitir seleção de campos extras (volume, NAV, premium).
#     * `download_crypto_ohlcv(...)`
#          - Para casos onde OHLC completo é requerido em backtests intradiários.
# - Providers esperados: Coinbase Advanced, Kaiko, Tiingo, Binance, AlphaVantage.
#     * Cada provider deve ter helper: `_load_from_tiingo`, `_load_from_kaiko`, etc.
#     * Implementar retries exponenciais, tratamento de rate-limit (`429`) e
#       logs amigáveis (`logger.warning`).
# - Normalização/cross-cutting concerns:
#     * `_sanitize_symbols(tickers)` — padroniza separadores, sufixos regionais.
#     * `_convert_currency(df, from_currency, to_currency, fx_source)` — caso o
#       ativo seja cotado em outra moeda (ex.: BTC em USD vs BRL).
#     * `_ensure_session_calendar(df, calendar)` — fecha gaps forçando forward-fill
#       ou reamostragem conforme calendário alvo (CME/NYSE/NasdaqCrypto).
# - Metadados:
#     * Retornar dicionário adicional ou armazenar em `df.attrs` com informações:
#       exchange, base_currency, quote_currency, timezone e provider.
# - Persistência opcional:
#     * Reutilizar `cache.request_hash` + `storage.save_parquet` criando diretório
#       `data/raw/crypto` para facilitar auditoria.
# - Testes sugeridos (`tests/data/sources/test_crypto.py` futuro):
#     * Mock de provider retornando OHLCV e validação de normalização de colunas.
#     * Erro amigável quando provider desconhecido é solicitado.
#     * Conversão de timezone/moeda preservando ordem cronológica.

"""Coleta dados de ETFs cripto spot.

Implementar integracao com provedores suportados, normalizar ticker, moeda e
retornar OHLCV com ajustes necessarios.
"""

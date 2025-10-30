"""Coleção de conectores para fontes externas.

Itens disponíveis
-----------------
`yf`
    ``download_prices`` baixa preços ajustados via Yahoo Finance.

`fred`
    ``download_dtb3`` obtém a taxa livre de risco diária (DTB3).

`csv`
    ``load_price_panel`` padroniza ingestões locais verificando schema.

`crypto`
    ``download_crypto_prices``/``download_crypto_ohlcv`` com suporte a vários
    provedores (Tiingo, Coinbase, Binance) e cache integrado.

Todos retornam outputs alinhados para consumo em ``processing.clean`` e no
``DataLoader``.
"""

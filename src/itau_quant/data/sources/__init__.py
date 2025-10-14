"""Coleção de conectores para fontes externas.

Itens disponíveis
-----------------\n
`yf`\n
    ``download_prices`` baixa preços ajustados via Yahoo Finance.\n
\n
`fred`\n
    ``download_dtb3`` obtém a taxa livre de risco diária (DTB3).\n
\n
`csv`\n
    ``load_price_panel`` padroniza ingestões locais verificando schema.\n
\n
`crypto`\n
    Blueprint para futuros conectores de ETFs/ETNs cripto.\n
\n
Todos retornam outputs alinhados para consumo em `processing.clean` e no\n+`DataLoader`.\n"""

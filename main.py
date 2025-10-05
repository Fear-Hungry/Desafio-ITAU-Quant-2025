"""Baixa históricos de preços via yfinance e monta um DataFrame combinado.

Este script faz o seguinte, linha a linha:
- Define uma lista de tickers (ETFs/ativos) em `acoes`.
- Para cada ticker baixa o histórico do último mês usando yfinance.
- Extrai Open/High/Low/Close/Volume, renomeia colunas sufixando com o ticker
    (ex.: 'SPY_Close') e guarda cada DataFrame em `dataframes_por_acao`.
- Por fim junta todos os DataFrames por índice de data (join outer) em
    `todos_dados` e mostra as primeiras linhas.

Observação: o arquivo foi mantido funcionalmente igual — apenas os comentários
foram organizados e esclarecidos para facilitar aprendizado.
"""

# Imports: padrão (stdlib) primeiro, depois terceiros
import os  # funções para trabalhar com sistema de arquivos (se necessário)
import time  # usado para pausar entre requisições e evitar throttling

import pandas as pd  # manipulação principal de tabelas (DataFrame)
import yfinance as yf  # cliente para baixar dados do Yahoo Finance
from IPython.display import display  # formata DataFrames em notebooks

# %%
# Lista de símbolos (tickers) que serão consultados no Yahoo Finance.
# Organizados por categoria para facilitar leitura; cada item é uma string
# reconhecida pelo Yahoo Finance (ex.: 'SPY' para o ETF S&P 500).
acoes = [
    # Ações EUA (amplo)
    "SPY", "QQQ", "IWM",

    # Mercados desenvolvidos ex-EUA e emergentes
    "EFA", "EEM",

    # Setores dos EUA (exemplos comuns)
    "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLK", "XLI", "XLB", "XLRE", "XLU",

    # ETFs por fator/estratégia
    "USMV", "MTUM", "QUAL", "VLUE", "SIZE",

    # Imobiliário (REITs)
    "VNQ", "VNQI",

    # Títulos do Tesouro (curto/médio/longo prazo)
    "SHY", "IEI", "IEF", "TLT",

    # TIPS (proteção contra inflação)
    "TIP",

    # Crédito/High yield/emerging market debt
    "LQD", "HYG", "EMB", "EMLC",

    # Commodities / USD
    "GLD", "DBC", "UUP",

    # ETFs de criptoativos (tokens sintéticos/ETFs relacionados)
    "IBIT", "ETHA",
]

# Dicionário para guardar um DataFrame por ação (chave = ticker, valor = df)
# Estruturas para armazenar resultados
dataframes_por_acao = {}  # formato: { 'SPY': DataFrame, 'QQQ': DataFrame, ... }
todos_dados = pd.DataFrame()  # DataFrame que conterá o resultado combinado (inicialmente vazio)

# %%
# Loop principal: para cada ticker, baixar e preparar o DataFrame
for acao in acoes:
    # Cria o wrapper do yfinance para acessar dados deste ticker
    ticker = yf.Ticker(acao)

    # Solicita o histórico do último mês. O DataFrame retornado costuma
    # conter colunas: Open, High, Low, Close, Volume (index = datas).
    dados = ticker.history(period="1mo")

    # Se não houver dados, registra a ocorrência e segue para o próximo
    if dados.empty:
        print(f"⚠️ Sem dados para {acao}, pulando...")
        continue

    # Seleciona as colunas de interesse e copia para evitar referências
    # diretas ao objeto retornado por yfinance.
    df_acao = dados[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Renomeia as colunas para incluir o ticker como sufixo. Isso evita
    # colisões quando juntarmos vários DataFrames (ex.: SPY_Close).
    df_acao.rename(
        columns={
            "Open": f"{acao}_Open",
            "High": f"{acao}_High",
            "Low": f"{acao}_Low",
            "Close": f"{acao}_Close",
            "Volume": f"{acao}_Volume",
        },
        inplace=True,
    )

    # Guarda o DataFrame individual no dicionário para uso posterior
    dataframes_por_acao[acao] = df_acao
    # Mensagem de progresso para o usuário
    print(f"✅ Dados de {acao} salvos individualmente")
    # Pequena pausa entre requisições para ser gentil com a API (throttling)
    time.sleep(1)

# %%
# Depois de baixar, mostramos um resumo do que foi obtido
print(f"\n📈 Total de ações processadas: {len(dataframes_por_acao)}")
print("Ações com dados disponíveis:")

for acao, df in dataframes_por_acao.items():
    # Assume que o índice do df é DatetimeIndex; mostramos número de registros
    # e o intervalo de datas (primeira e última linha).
    print(
        f"  • {acao}: {len(df)} registros, de {df.index[0].strftime('%Y-%m-%d')} a {df.index[-1].strftime('%Y-%m-%d')}"
    )

# %%
# Exibe os primeiros registros de cada DataFrame para inspeção rápida
print("\n🔍 Primeiros registros de cada ação:")
for acao, df in list(dataframes_por_acao.items()):
    print(f"\n{acao}:")
    # Em um Jupyter Notebook, display() renderiza uma tabela bonita.
    display(df.head())

# %%
# Cria um DataFrame combinado alinhado por data (índice). Usamos join outer
# para preservar datas que possam existir em alguns ativos e não em outros.
print("Mostrando todas as ações disponíveis:\n")
for acao in dataframes_por_acao:
    if todos_dados.empty:
        # Inicializa com o primeiro DataFrame
        todos_dados = dataframes_por_acao[acao]
    else:
        # Junta por índice de data mantendo todas as datas (outer)
        todos_dados = todos_dados.join(dataframes_por_acao[acao], how="outer")

# Exibe as primeiras linhas do DataFrame combinado
display(todos_dados.head())

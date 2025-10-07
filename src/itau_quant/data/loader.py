import pandas as pd
from pathlib import Path

# Define caminhos relativos à localização deste arquivo para robustez
# src/itau_quant/data/loader.py
# O diretório do projeto é 3 níveis acima
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def load_asset_prices(file_name: str) -> pd.DataFrame:
    """
    Carrega os preços dos ativos de um arquivo CSV no diretório de dados brutos.

    Args:
        file_name: O nome do arquivo a ser carregado (ex: 'asset_prices.csv').

    Returns:
        Um DataFrame do pandas com os preços dos ativos.
    """
    raw_file_path = RAW_DATA_DIR / file_name
    if not raw_file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado em: {raw_file_path}")

    # Supõe que a primeira coluna é a data e a define como índice
    df = pd.read_csv(raw_file_path, index_col=0, parse_dates=True)
    return df


def calculate_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula os retornos percentuais (decimais) a partir de um DataFrame de preços.

    Args:
        prices_df: DataFrame com preços dos ativos.

    Returns:
        DataFrame com os retornos dos ativos.
    """
    return prices_df.pct_change().dropna()


def preprocess_data(raw_file_name: str, processed_file_name: str) -> pd.DataFrame:
    """
    Orquestra o pipeline completo de pré-processamento de dados:
    1. Carrega os preços brutos.
    2. Calcula os retornos.
    3. Salva os retornos processados em um novo arquivo.
    4. Retorna o DataFrame de retornos.

    Args:
        raw_file_name: Nome do arquivo de dados brutos.
        processed_file_name: Nome do arquivo para salvar os dados processados.

    Returns:
        DataFrame com retornos limpos e prontos para análise.
    """
    print("Iniciando pré-processamento de dados...")

    # Garante que o diretório de destino exista
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_asset_prices(raw_file_name)
    returns = calculate_returns(prices)

    processed_file_path = PROCESSED_DATA_DIR / processed_file_name
    returns.to_parquet(processed_file_path)

    print(f"Dados processados e salvos em: {processed_file_path}")
    return returns

# arara_quant.data — Camada de Dados

Camada de aquisição, limpeza, calendário, transformações e armazenamento de dados
para a carteira ARARA. Estruturada para reprodutibilidade (seed e cache),
substituição de fontes (yfinance ↔ CSV/API) e testes unitários.

## Visão Geral

- Facade: `loader.py` (classe DataLoader, DataBundle)
- Fontes: `sources/` (yfinance, FRED, CSV)
- Processamento: `processing/` (calendar, returns, clean)
- Persistência: `storage.py` (Parquet zstd)
- Utilidades: `paths.py` (diretórios), `universe.py` (tickers), `cache.py` (hash de pedidos)

```
src/arara_quant/data/
├── loader.py            # Orquestra download → limpeza → retornos → excess → BMS → disco
├── paths.py             # PROJECT_ROOT/DATA_DIR/RAW/PROCESSED
├── universe.py          # ARARA_TICKERS, get_arara_universe()
├── storage.py           # save_parquet/load_parquet (zstd)
├── cache.py             # request_hash(tickers, start, end)
├── sources/
│   ├── yf.py            # download_prices(tickers, start, end, with_volume=False) → (prices, volume|None)
│   ├── fred.py          # download_dtb3(start, end) → rf diário (B, ffill)
│   └── csv.py           # leitores alternativos (futuro)
├── processing/
│   ├── clean.py         # ensure_dtindex, normalize_index, validate_panel
│   ├── returns.py       # calculate_returns, compute_excess_returns
│   └── calendar.py      # BMS/BME/weekly, next/prev day, clamp_to_index, rebalance_schedule
└── __init__.py          # API pública reexportada
```

## API Pública Principal

- `DataLoader(tickers=None, start=None, end=None, mode="BMS").load() -> DataBundle`
  - prices (DataFrame), returns (DataFrame), rf_daily (Series), excess_returns (DataFrame), bms (DatetimeIndex), inception_mask (Series)
- `get_arara_universe() -> list[str]`
- `business_month_starts/ends(idx) -> DatetimeIndex`, `rebalance_schedule(index, mode)`
- `calculate_returns(prices, method="log") -> DataFrame`, `compute_excess_returns(returns, rf_daily) -> DataFrame`

## Fluxo do DataLoader

1) `sources.yf.download_prices(..., with_volume=True)` → (prices, volume|None)
2) `processing.clean.normalize_index` e `validate_panel`
3) `processing.returns.calculate_returns` e `compute_excess_returns`
4) `processing.calendar.rebalance_schedule`
5) Armazenamento com `storage.save_parquet` usando `cache.request_hash` no nome

## Convenções

- Índices sempre `DatetimeIndex` sem timezone, ordenados e únicos.
- Retornos em decimais (0.01 = 1%). Retornos log como padrão.
- `rf_daily` alinhado a dias úteis (B) com forward-fill.
- Compressão Parquet: `zstd`.

## Exemplos

### 1) Smoke sem rede (sintético)
```python
import pandas as pd
from arara_quant.data import business_month_starts, calculate_returns, compute_excess_returns

idx = pd.bdate_range("2024-01-01", periods=10)
prices = pd.DataFrame({"SPY": 100.0, "EFA": 50.0}, index=idx).cumsum()
rets = calculate_returns(prices, method="log")
rf = pd.Series([0.0001, 0.0002, 0.0001], index=idx[:3])
excess = compute_excess_returns(rets, rf)
bms = business_month_starts(prices.index)
print(rets.shape, excess.shape, len(bms))
```

### 2) DataLoader (requer rede)
```python
from arara_quant.data import DataLoader

bundle = DataLoader(start="2015-01-01", end="2025-01-01", mode="BMS").load()
print(bundle.returns.tail())
```

### 3) Persistência (Parquet zstd)
```python
from pathlib import Path
from arara_quant.data.storage import save_parquet, load_parquet
from arara_quant.data.paths import PROCESSED_DATA_DIR

path = PROCESSED_DATA_DIR / "example.parquet"
save_parquet(path, bundle.returns)
df = load_parquet(path)
print(df.shape)
```

## Testes

- `tests/data/test_clean.py`, `test_returns_excess.py`, `test_calendar.py` e smoke do loader (skip rede).

# Data API

Este documento lista os principais pontos de entrada do módulo
`arara_quant.data` usados por scripts e testes offline.

## `DataLoader`

`DataLoader` é a fachada de alto nível para baixar/validar dados, calcular
retornos e persistir artefatos com hash determinístico.

```python
from arara_quant.data.loader import DataLoader

bundle = DataLoader(start="2010-01-01", end="2025-10-09").load()
returns = bundle.returns
```

**Retorno:** `DataBundle`, com:

- `prices`: preços ajustados (DataFrame)
- `returns`: log-returns diários (DataFrame)
- `rf_daily`: taxa livre de risco diária (Series)
- `excess_returns`: retornos em excesso a RF (DataFrame)
- `bms`: agenda de rebalance (DatetimeIndex)

## Caminhos e storage

- Paths padronizados: `arara_quant.data.paths` (`RAW_DATA_DIR`, `PROCESSED_DATA_DIR`)
- Cache determinístico: `arara_quant.data.cache.request_hash`
- Persistência: `arara_quant.data.storage.save_parquet` / `load_parquet`


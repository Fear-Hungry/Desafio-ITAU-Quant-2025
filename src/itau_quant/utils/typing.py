"""Blueprint for shared typing aliases and protocols.

Objetivo
--------
Definir aliases e Protocols reutilizáveis para padronizar tipagem em todo o
projeto (evitando importações repetidas de pandas/np typing modules).

Componentes sugeridos
---------------------\n
- `DataFrame`, `Series`, `ArrayLike`\n
    TypeAlias apontando para `pd.DataFrame`, `pd.Series`, `np.ndarray | pd.Series`.\n
- `PricePanel`, `ReturnPanel`\n
    Aliases específicos com docs descrevendo shape esperado.\n
- `SupportsToDataFrame` Protocol\n
    Exige método `to_dataframe()` retornando DataFrame.\n
- `OptimizerProtocol`\n
    Define interface padrão usada pelo rebalancer (método `solve(mu, cov, config)`).\n
- `StrategyResult` TypedDict/dataclass\n
    Estrutura padronizada para resultados de backtest (nav, trades, metrics).\n
- `PathLike` alias\n
    Reexport de `os.PathLike[str] | str`.\n

Considerações
-------------\n
- Documentar versões mínimas de pandas/numpy para compatibilidade typing (``from pandas import DataFrame`` etc.).\n
- Permitir fallback quando pandas/numpy não disponíveis (ex.: usar `typing.TYPE_CHECKING`).\n
- Expor no `__all__` os aliases principais para consumo em outros módulos.\n

Testes recomendados\n
-------------------\n
- `tests/utils/test_typing.py` (mocks) garantindo que Protocols funcionam com classes dummy.\n
"""

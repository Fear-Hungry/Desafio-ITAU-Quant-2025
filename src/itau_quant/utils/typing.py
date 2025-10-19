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

"""
shared_types.py

Módulo central para aliases de tipo e protocolos reutilizáveis em todo o projeto.

Considerações de Versão:
A importação direta de tipos como `from pandas import DataFrame` requer:
- pandas >= 1.4.0
- numpy >= 1.21.0
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, TYPE_CHECKING, Union

try:  # Python < 3.10 fallback
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - executed only on older interpreters
    from typing_extensions import TypeAlias  # type: ignore[assignment]

# --- Exposição explícita dos tipos públicos ---
__all__ = [
    "DataFrame",
    "Series",
    "ArrayLike",
    "PricePanel",
    "ReturnPanel",
    "PathLike",
    "SupportsToDataFrame",
    "OptimizerProtocol",
    "StrategyResult",
]

# --- Fallback para ambientes sem pandas/numpy ---
# O bloco `if TYPE_CHECKING:` é lido por type checkers (Mypy, Pyright),
# mas é ignorado em tempo de execução. Isso evita erros de importação.
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray

    # --- Aliases Genéricos ---
    DataFrame: TypeAlias = pd.DataFrame
    Series: TypeAlias = pd.Series
    ArrayLike: TypeAlias = Union[NDArray[np.float64], pd.Series]

else:
    # Em tempo de execução, se as bibliotecas não estiverem instaladas,
    # os tipos são definidos como 'Any' para evitar erros.
    DataFrame: TypeAlias = Any
    Series: TypeAlias = Any
    ArrayLike: TypeAlias = Any


# --- Aliases de Domínio Específico ---

#: DataFrame onde colunas são ativos e o índice é um DatetimeIndex com preços.
PricePanel: TypeAlias = DataFrame

#: DataFrame onde colunas são ativos e o índice é um DatetimeIndex com retornos.
ReturnPanel: TypeAlias = DataFrame


# --- Aliases de I/O ---

#: Alias para tipos que representam caminhos de arquivo.
PathLike: TypeAlias = Union[str, os.PathLike[str]]


# --- Protocolos (Interfaces Estruturais) ---

class SupportsToDataFrame(Protocol):
    """
    Protocolo para objetos que podem ser convertidos para um pandas DataFrame.
    """
    def to_dataframe(self) -> DataFrame:
        """Retorna a representação do objeto como um DataFrame."""
        ...


class OptimizerProtocol(Protocol):
    """
    Protocolo que define a interface de um otimizador de portfólio.

    Qualquer classe que implemente o método `solve` com a assinatura correspondente
    satisfaz este protocolo (duck typing).
    """
    def solve(
        self,
        mu: Series,
        cov: DataFrame,
        config: Dict[str, Any]
    ) -> Series:
        """
        Executa a otimização e retorna os pesos ótimos do portfólio.

        Args:
            mu (Series): Vetor de retornos esperados (índice = ativos).
            cov (DataFrame): Matriz de covariância (índice/colunas = ativos).
            config (Dict[str, Any]): Dicionário com configurações do otimizador.

        Returns:
            Series: Pesos ótimos do portfólio (índice = ativos).
        """
        ...


# --- Estruturas de Dados Tipadas ---

@dataclass
class StrategyResult:
    """
    Estrutura padronizada para encapsular os resultados de um backtest.
    """
    nav: Series  #: Série temporal do Net Asset Value (NAV) da estratégia.
    trades: DataFrame  #: DataFrame com o registro de todas as operações.
    metrics: Dict[str, float]  #: Dicionário com métricas de performance (ex: Sharpe, Drawdown).

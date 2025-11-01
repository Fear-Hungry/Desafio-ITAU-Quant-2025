"""Utilitários para modelagem de custos de transação.

O módulo provê funções numéricas simples que cobrem três blocos frequentes
na modelagem de carteiras:

``linear``
    Conversão de custos declarados em *basis points* (bps) para números
    decimais e cálculo do valor monetário negociado.

``slippage``
    Estimativa de impacto de mercado pela forma clássica de *square-root
    impact* em função da razão entre o notional negociado e o volume diário
    médio (ADV).

``optimisation helpers``
    Construção de vetores prontos para alimentar objetivos lineares dos
    otimizadores (ex.: :math:`c^T |w - w_{prev}|`).

As funções aceitam ``numpy.ndarray`` ou ``pandas.Series`` e devolvem o mesmo
tipo quando o resultado é vetorial. Scalars são suportados e *broadcast* para
o comprimento adequado. Entradas contendo ``NaN`` produzem ``ValueError`` para
evitar a propagação silenciosa de erros numéricos.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "bps_to_decimal",
    "linear_cost_vector",
    "linear_transaction_cost",
    "slippage_square_root_bps",
    "slippage_transaction_cost",
    "transaction_cost_vector",
]


ArrayLike = Sequence[float] | np.ndarray | pd.Series


def _is_scalar(value: object) -> bool:
    """Return ``True`` when *value* behaves like a scalar for numpy."""

    return np.isscalar(value)


def _as_array(
    value: ArrayLike | float, *, name: str, length: int | None = None
) -> np.ndarray:
    """Convert supported inputs to a 1-D ``np.ndarray``.

    Parameters
    ----------
    value:
        Scalar, sequence, ``np.ndarray`` ou ``pd.Series`` com os dados.
    name:
        Nome usado em mensagens de erro.
    length:
        Comprimento esperado; quando informado aplicamos *broadcast* para
        valores escalares e validamos vetores com tamanho incompatível.
    """

    if _is_scalar(value):
        if length is None:
            return np.asarray([float(value)], dtype=float)
        return np.full(length, float(value), dtype=float)

    if isinstance(value, pd.Series):
        array = value.to_numpy(dtype=float)
    elif isinstance(value, np.ndarray):
        array = value.astype(float, copy=False)
    else:
        array = np.asarray(list(value), dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")

    if length is not None and array.size != length:
        raise ValueError(
            f"{name} must contain {length} elements; received {array.size}"
        )

    if np.isnan(array).any():
        raise ValueError(f"{name} contains NaN values")

    return array


def _maybe_series(
    template: ArrayLike | float, values: np.ndarray, *, name: str
) -> np.ndarray | pd.Series:
    """Match the output type to *template* when it is a pandas ``Series``."""

    if isinstance(template, pd.Series):
        return pd.Series(values, index=template.index, name=name)
    return values


def _broadcast_template(*candidates: ArrayLike | float) -> ArrayLike | float:
    """Return the first pandas ``Series`` candidate for result wrapping."""

    for candidate in candidates:
        if isinstance(candidate, pd.Series):
            return candidate
    return 0.0  # sentinel meaning "no wrapping"


def bps_to_decimal(value: ArrayLike | float) -> np.ndarray | pd.Series | float:
    """Convert basis points to decimal form.

    Examples
    --------
    >>> bps_to_decimal(20)
    0.002
    >>> bps_to_decimal([10, 25])
    array([0.001 , 0.0025])
    """

    if _is_scalar(value):
        return float(value) / 10_000.0

    arr = _as_array(value, name="value")
    out = arr / 10_000.0
    return _maybe_series(value, out, name="decimal_cost")


def linear_cost_vector(
    costs_bps: ArrayLike | float, *, notional: float = 1.0
) -> np.ndarray | pd.Series | float:
    """Return the cost vector :math:`c` for |Delta w| given linear bps.

    Parameters
    ----------
    costs_bps:
        Custo linear em basis points. Scalars são aplicados uniformemente a
        todos os ativos.
    notional:
        Valor de referência da carteira. Por padrão assumimos pesos
        normalizados (notional = 1.0).
    """

    if notional <= 0:
        raise ValueError("notional must be positive")

    if _is_scalar(costs_bps):
        return float(costs_bps) * notional / 10_000.0

    arr = _as_array(costs_bps, name="costs_bps")
    vector = arr * notional / 10_000.0
    return _maybe_series(costs_bps, vector, name="linear_cost")


def linear_transaction_cost(
    weights: ArrayLike,
    prev_weights: ArrayLike,
    costs_bps: ArrayLike | float,
    *,
    notional: float = 1.0,
    aggregate: bool = True,
) -> float | np.ndarray | pd.Series:
    """Compute realised linear costs in currency units.

    The function implements::

        cost_i = |w_i - w_i^{prev}| * notional * (bps_i / 10_000)

    and optionally aggregates the result across assets.
    """

    weights_arr = _as_array(weights, name="weights")
    prev_arr = _as_array(prev_weights, name="prev_weights", length=weights_arr.size)
    trades = np.abs(weights_arr - prev_arr)

    costs_arr = _as_array(costs_bps, name="costs_bps", length=weights_arr.size)
    per_asset_costs = trades * notional * costs_arr / 10_000.0

    if aggregate:
        return float(np.sum(per_asset_costs))

    template = _broadcast_template(costs_bps, weights, prev_weights)
    return _maybe_series(template, per_asset_costs, name="linear_cost")


def slippage_square_root_bps(
    trade_notional: ArrayLike | float,
    adv: ArrayLike | float,
    *,
    coefficient: float = 50.0,
    exponent: float = 0.5,
    min_bps: float = 0.0,
    max_bps: float | None = None,
) -> np.ndarray | pd.Series | float:
    """Square-root impact model expressed em bps.

    ``coefficient`` representa o impacto (em bps) quando o *participation
    rate* é igual a 1. Valores mínimos/máximos podem ser impostos para evitar
    resultados numéricos extremos em universos ilíquidos.
    """

    if coefficient < 0 or exponent <= 0:
        raise ValueError("coefficient must be non-negative and exponent positive")

    if min_bps < 0:
        raise ValueError("min_bps cannot be negative")

    trade_arr = _as_array(trade_notional, name="trade_notional")
    adv_arr = _as_array(adv, name="adv", length=trade_arr.size)

    if np.any((trade_arr > 0) & (adv_arr <= 0)):
        raise ValueError("adv must be positive for assets with non-zero trades")

    ratio = np.divide(
        trade_arr, adv_arr, out=np.zeros_like(trade_arr), where=adv_arr > 0
    )
    impact = coefficient * np.power(ratio, exponent)
    if min_bps:
        impact = np.maximum(impact, min_bps)
    if max_bps is not None:
        if max_bps < min_bps:
            raise ValueError("max_bps must be greater or equal to min_bps")
        impact = np.minimum(impact, max_bps)

    template = _broadcast_template(trade_notional, adv)
    return _maybe_series(template, impact, name="slippage_bps")


def slippage_transaction_cost(
    weights: ArrayLike,
    prev_weights: ArrayLike,
    adv: ArrayLike | float,
    *,
    notional: float = 1.0,
    coefficient: float = 50.0,
    exponent: float = 0.5,
    min_bps: float = 0.0,
    max_bps: float | None = None,
    aggregate: bool = True,
) -> float | np.ndarray | pd.Series:
    """Estimate slippage costs using :func:`slippage_square_root_bps`."""

    weights_arr = _as_array(weights, name="weights")
    prev_arr = _as_array(prev_weights, name="prev_weights", length=weights_arr.size)
    trades_notional = np.abs(weights_arr - prev_arr) * notional

    impact_bps = slippage_square_root_bps(
        trades_notional,
        adv,
        coefficient=coefficient,
        exponent=exponent,
        min_bps=min_bps,
        max_bps=max_bps,
    )

    impact_arr = _as_array(impact_bps, name="slippage_bps", length=weights_arr.size)
    per_asset = trades_notional * impact_arr / 10_000.0

    if aggregate:
        return float(np.sum(per_asset))

    template = _broadcast_template(impact_bps, weights, prev_weights)
    return _maybe_series(template, per_asset, name="slippage_cost")


def transaction_cost_vector(
    weights: ArrayLike,
    prev_weights: ArrayLike,
    *,
    linear_bps: ArrayLike | float | None = None,
    adv: ArrayLike | float | None = None,
    notional: float = 1.0,
    coefficient: float = 50.0,
    exponent: float = 0.5,
    min_slippage_bps: float = 0.0,
    max_slippage_bps: float | None = None,
) -> np.ndarray | pd.Series:
    """Return per-asset costs combining linear and slippage components.

    The result is a *vector* (same dimension of the universo de ativos)
    representando o custo monetário de efetuar o rebalanceamento sugerido. O
    valor agregado pode ser obtido com ``vector.sum()``.
    """

    weights_arr = _as_array(weights, name="weights")
    prev_arr = _as_array(prev_weights, name="prev_weights", length=weights_arr.size)
    template = _broadcast_template(linear_bps, adv, weights, prev_weights)

    per_asset = np.zeros_like(weights_arr)

    if linear_bps is not None:
        per_asset += _as_array(
            linear_transaction_cost(
                weights_arr,
                prev_arr,
                linear_bps,
                notional=notional,
                aggregate=False,
            ),
            name="linear_component",
            length=weights_arr.size,
        )

    if adv is not None:
        per_asset += _as_array(
            slippage_transaction_cost(
                weights_arr,
                prev_arr,
                adv,
                notional=notional,
                coefficient=coefficient,
                exponent=exponent,
                min_bps=min_slippage_bps,
                max_bps=max_slippage_bps,
                aggregate=False,
            ),
            name="slippage_component",
            length=weights_arr.size,
        )

    return _maybe_series(template, per_asset, name="transaction_cost")

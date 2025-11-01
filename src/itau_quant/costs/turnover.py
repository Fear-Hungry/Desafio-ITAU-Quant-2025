r"""Funções auxiliares de turnover para rebalanceamento de carteiras.

O objetivo é disponibilizar blocos reutilizáveis tanto para avaliação (*post
trade*) quanto para modelagem em otimizadores:

``l1_turnover``
    Calcula :math:`\lVert w - w_{prev} \rVert_1`, métrica clássica de giro.

``normalised_turnover``
    Retorna o turnover normalizado (metade da norma L1), frequentemente usado
    em relatórios financeiros.

``turnover_penalty`` / ``turnover_penalty_vector``
    Facilita a inclusão de penalidades suaves :math:`\eta \lVert \Delta w\rVert_1`.

``turnover_violation`` / ``is_within_turnover``
    Utilitários para validar limites duros de giro.

As funções aceitam ``numpy.ndarray`` ou ``pandas.Series`` e devolvem o mesmo
tipo quando o resultado é vetorial, espelhando o comportamento do módulo
``transaction_costs``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "l1_turnover",
    "normalised_turnover",
    "turnover_penalty",
    "turnover_penalty_vector",
    "turnover_violation",
    "is_within_turnover",
    "trades",
]


ArrayLike = Sequence[float] | np.ndarray | pd.Series


def _is_scalar(value: object) -> bool:
    return np.isscalar(value)


def _as_array(
    value: ArrayLike | float, *, name: str, length: int | None = None
) -> np.ndarray:
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


def _template(*candidates: ArrayLike | float) -> ArrayLike | float:
    for candidate in candidates:
        if isinstance(candidate, pd.Series):
            return candidate
    return 0.0


def _wrap(
    values: np.ndarray, template: ArrayLike | float, name: str
) -> np.ndarray | pd.Series:
    if isinstance(template, pd.Series):
        return pd.Series(values, index=template.index, name=name)
    return values


def trades(weights: ArrayLike, prev_weights: ArrayLike) -> np.ndarray:
    """Return signed rebalance trades (``w - w_prev``)."""

    weights_arr = _as_array(weights, name="weights")
    prev_arr = _as_array(prev_weights, name="prev_weights", length=weights_arr.size)
    return weights_arr - prev_arr


def l1_turnover(
    weights: ArrayLike, prev_weights: ArrayLike, *, aggregate: bool = True
) -> float | np.ndarray | pd.Series:
    r"""Compute :math:`\lVert w - w_{prev} \rVert_1`."""

    delta = np.abs(trades(weights, prev_weights))
    if aggregate:
        return float(delta.sum())

    template = _template(weights, prev_weights)
    return _wrap(delta, template, name="l1_turnover")


def normalised_turnover(weights: ArrayLike, prev_weights: ArrayLike) -> float:
    """Return half of the L1 turnover, common in performance reports."""

    return 0.5 * float(l1_turnover(weights, prev_weights, aggregate=True))


def turnover_penalty(
    weights: ArrayLike,
    prev_weights: ArrayLike,
    eta: float,
    *,
    normalised: bool = False,
) -> float:
    """Return soft penalty ``eta * ||Delta w||_1`` (optionally normalised)."""

    if eta < 0:
        raise ValueError("eta must be non-negative")

    base = l1_turnover(weights, prev_weights, aggregate=True)
    if normalised:
        base *= 0.5
    return float(eta) * float(base)


def turnover_penalty_vector(
    weights: ArrayLike,
    prev_weights: ArrayLike,
    eta: float,
    *,
    normalised: bool = False,
    aggregate: bool = False,
) -> float | np.ndarray | pd.Series:
    """Return per-asset contribution of the soft penalty.

    When ``aggregate`` is ``True`` the function collapses to
    :func:`turnover_penalty`.
    """

    if eta < 0:
        raise ValueError("eta must be non-negative")

    delta = np.abs(trades(weights, prev_weights))
    if normalised:
        delta *= 0.5
    contribution = float(eta) * delta

    if aggregate:
        return float(contribution.sum())

    template = _template(weights, prev_weights)
    return _wrap(contribution, template, name="turnover_penalty")


def turnover_violation(
    weights: ArrayLike,
    prev_weights: ArrayLike,
    max_turnover: float,
    *,
    normalised: bool = True,
    tol: float = 1e-9,
) -> float:
    """Return how much the turnover exceeds the allowed limit (>= 0)."""

    if max_turnover < 0:
        raise ValueError("max_turnover must be non-negative")

    realised = (
        normalised_turnover(weights, prev_weights)
        if normalised
        else float(l1_turnover(weights, prev_weights, aggregate=True))
    )
    violation = realised - max_turnover
    if violation <= tol:
        return 0.0
    return float(violation)


def is_within_turnover(
    weights: ArrayLike,
    prev_weights: ArrayLike,
    max_turnover: float,
    *,
    normalised: bool = True,
    tol: float = 1e-9,
) -> bool:
    """Check if the turnover constraint is satisfied within tolerance."""

    return (
        turnover_violation(
            weights,
            prev_weights,
            max_turnover,
            normalised=normalised,
            tol=tol,
        )
        == 0.0
    )

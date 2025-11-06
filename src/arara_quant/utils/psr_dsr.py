"""Probabilistic/Deflated Sharpe helpers with Newey–West ν estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["PSRDSRResult", "psr_dsr"]


def _normal_cdf(x: float) -> float:
    """Standard normal CDF without SciPy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _inv_normal_cdf(p: float) -> float:
    """Acklam's approximation of Φ^{-1}(p) (sufficient for stats work)."""
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("p must be in (0, 1)")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        num = ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]
        den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        x = (num * q) / den
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    # One Newton step refines accuracy.
    e = _normal_cdf(x) - p
    x = x - e / (math.sqrt(2.0 * math.pi) * math.exp(-0.5 * x * x))
    return x


def _auto_newey_west_lag(length: int) -> int:
    """Plugin rule of thumb for NW lag (>=1 when sample is reasonable)."""
    if length <= 2:
        return 1
    lag = int(np.floor(4.0 * (length / 100.0) ** (2.0 / 9.0)))
    return max(1, lag)


def _effective_sample_size_newey_west(returns: pd.Series, max_lag: Optional[int] = None) -> float:
    """Compute Newey–West-style effective ν (guards against autocorr)."""
    values = pd.Series(returns).dropna().values.astype(float)
    n_obs = len(values)
    if n_obs < 3:
        return float(max(n_obs, 1))

    if max_lag is None:
        lag = _auto_newey_west_lag(n_obs)
    else:
        lag = int(max(1, min(max_lag, n_obs - 1)))

    centered = values - values.mean()
    denom = np.dot(centered, centered)
    if denom <= 0:
        return float(n_obs)

    acc = 0.0
    for k in range(1, lag + 1):
        numerator = np.dot(centered[k:], centered[:-k])
        rho_k = numerator / denom if denom != 0 else 0.0
        weight = 1.0 - k / (lag + 1.0)  # Bartlett weights
        acc += weight * rho_k

    variance_inflation = 1.0 + 2.0 * acc
    if variance_inflation <= 0:
        variance_inflation = 1e-12

    nu_eff = n_obs / variance_inflation
    return float(np.clip(nu_eff, 2.0, n_obs))


@dataclass
class PSRDSRResult:
    """Container with PSR/DSR stats for downstream reporting."""

    sharpe_ann: float
    mean: float
    std: float
    periods_per_year: int
    psr: float
    dsr: float
    s0: float
    s_star: float
    nu_eff: float
    skew: float
    excess_kurtosis: float
    nw_lag: int
    N: int

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow copy for JSON serialization/reporting."""
        return self.__dict__.copy()


def psr_dsr(
    returns: pd.Series,
    *,
    S0: float = 0.0,
    N: int = 1,
    periods_per_year: int = 252,
    nw_max_lag: Optional[int] = None,
) -> PSRDSRResult:
    """Compute Probabilistic and Deflated Sharpe ratios for a return series."""
    r = pd.Series(returns).dropna().astype(float)
    n_obs = len(r)
    std = r.std(ddof=1)
    if n_obs < 3 or std == 0:
        nu_eff = float(max(n_obs, 2))
        return PSRDSRResult(
            sharpe_ann=0.0,
            mean=float(r.mean()) if n_obs else 0.0,
            std=float(std),
            periods_per_year=int(periods_per_year),
            psr=0.5,
            dsr=0.5,
            s0=float(S0),
            s_star=float(S0),
            nu_eff=nu_eff,
            skew=0.0,
            excess_kurtosis=0.0,
            nw_lag=1,
            N=max(1, N),
        )

    mu = r.mean()
    sharpe_ann = (mu / std) * math.sqrt(periods_per_year)
    skew = float(r.skew())
    excess_kurt = float(r.kurt())

    nw_lag = _auto_newey_west_lag(n_obs) if nw_max_lag is None else int(max(1, min(nw_max_lag, n_obs - 1)))
    nu_eff = _effective_sample_size_newey_west(r, max_lag=nw_lag)

    S = sharpe_ann
    gamma3 = skew
    gamma4 = excess_kurt + 3.0  # convert from excess to raw kurtosis
    numerator = 1.0 - gamma3 * S + ((gamma4 - 1.0) / 4.0) * (S**2)
    numerator = max(numerator, 1e-12)
    denom = max(nu_eff - 1.0, 1.0)
    se_s = math.sqrt(numerator / denom)

    z_psr = (S - S0) / se_s
    psr = _normal_cdf(z_psr)

    N_eff = max(1, int(N))
    if N_eff == 1:
        s_star = S0
    else:
        z_star = _inv_normal_cdf(1.0 - 1.0 / N_eff)
        s_star = S0 + z_star * se_s

    z_dsr = (S - s_star) / se_s
    dsr = _normal_cdf(z_dsr)

    return PSRDSRResult(
        sharpe_ann=float(S),
        mean=float(mu),
        std=float(std),
        periods_per_year=int(periods_per_year),
        psr=float(psr),
        dsr=float(dsr),
        s0=float(S0),
        s_star=float(s_star),
        nu_eff=float(nu_eff),
        skew=float(skew),
        excess_kurtosis=float(excess_kurt),
        nw_lag=int(nw_lag),
        N=int(N_eff),
    )

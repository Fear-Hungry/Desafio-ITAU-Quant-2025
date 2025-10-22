"""Skill diagnostics for expected return estimators.

Tests whether μ estimates have predictive power for future returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "information_coefficient",
    "predictive_r2",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "skill_report",
    "SkillReport",
]


def information_coefficient(
    mu_estimates: pd.DataFrame,
    realized_returns: pd.DataFrame,
    *,
    method: str = "spearman",
) -> pd.Series:
    """Compute cross-sectional IC between forecasts and realizations.

    Parameters
    ----------
    mu_estimates : pd.DataFrame
        Forecasted returns (rows = time periods, cols = assets)
    realized_returns : pd.DataFrame
        Realized returns (rows = time periods, cols = assets)
    method : str
        Correlation method: 'pearson' or 'spearman'

    Returns
    -------
    pd.Series
        IC for each time period
    """
    if mu_estimates.shape != realized_returns.shape:
        raise ValueError("mu_estimates and realized_returns must have same shape")

    ic_values = []
    for t in mu_estimates.index:
        mu_t = mu_estimates.loc[t].dropna()
        ret_t = realized_returns.loc[t].dropna()

        common = mu_t.index.intersection(ret_t.index)
        if len(common) < 3:
            ic_values.append(np.nan)
            continue

        if method == "spearman":
            ic, _ = stats.spearmanr(mu_t.loc[common], ret_t.loc[common])
        else:
            ic, _ = stats.pearsonr(mu_t.loc[common], ret_t.loc[common])

        ic_values.append(ic)

    return pd.Series(ic_values, index=mu_estimates.index, name="IC")


def predictive_r2(
    mu_estimates: pd.DataFrame,
    realized_returns: pd.DataFrame,
) -> dict[str, float]:
    """Compute predictive R² from panel regression μ̂ → r.

    Parameters
    ----------
    mu_estimates : pd.DataFrame
        Forecasted returns (rows = time, cols = assets)
    realized_returns : pd.DataFrame
        Realized returns (rows = time, cols = assets)

    Returns
    -------
    dict
        Contains 'r2', 'r2_adj', 'beta', 'beta_pval', 'n_obs'
    """
    # Stack to long format
    mu_long = mu_estimates.stack().rename("mu_hat")
    ret_long = realized_returns.stack().rename("return")

    df = pd.concat([mu_long, ret_long], axis=1).dropna()

    if len(df) < 10:
        return {
            "r2": np.nan,
            "r2_adj": np.nan,
            "beta": np.nan,
            "beta_pval": np.nan,
            "n_obs": len(df),
        }

    X = df["mu_hat"].values
    y = df["return"].values

    # OLS: y = alpha + beta * X
    X_ols = np.column_stack([np.ones(len(X)), X])
    try:
        beta_hat = np.linalg.lstsq(X_ols, y, rcond=None)[0]
        y_pred = X_ols @ beta_hat

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        n, k = len(y), 2
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else np.nan

        # t-stat for beta
        residuals = y - y_pred
        mse = ss_res / (n - k) if n > k else np.nan
        var_beta = mse * np.linalg.inv(X_ols.T @ X_ols)
        se_beta = np.sqrt(np.diag(var_beta))[1]
        t_stat = beta_hat[1] / se_beta if se_beta > 0 else 0.0
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))

    except np.linalg.LinAlgError:
        return {
            "r2": np.nan,
            "r2_adj": np.nan,
            "beta": np.nan,
            "beta_pval": np.nan,
            "n_obs": len(df),
        }

    return {
        "r2": float(r2),
        "r2_adj": float(r2_adj),
        "beta": float(beta_hat[1]),
        "beta_pval": float(p_val),
        "n_obs": len(df),
    }


def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    *,
    sharpe_benchmark: float = 0.0,
    skew: float | None = None,
    kurtosis: float | None = None,
) -> float:
    """Compute Probabilistic Sharpe Ratio (PSR).

    PSR = Prob(SR_true > SR_benchmark | observed data)

    Based on Bailey & López de Prado (2012).

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Time series of returns
    sharpe_benchmark : float
        Benchmark Sharpe to test against (default 0)
    skew : float, optional
        Empirical skewness (computed if None)
    kurtosis : float, optional
        Empirical excess kurtosis (computed if None)

    Returns
    -------
    float
        PSR ∈ [0, 1]
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values

    returns = np.asarray(returns).ravel()
    if len(returns) < 3:
        return np.nan

    sr_obs = returns.mean() / (returns.std(ddof=1) + 1e-12)
    n = len(returns)

    if skew is None:
        skew = float(stats.skew(returns, bias=False))
    if kurtosis is None:
        kurtosis = float(stats.kurtosis(returns, bias=False))

    # Variance of SR estimator
    var_sr = (1 + 0.5 * sr_obs**2 - skew * sr_obs + (kurtosis / 4) * sr_obs**2) / n

    if var_sr <= 0:
        return np.nan

    # Z-score
    z = (sr_obs - sharpe_benchmark) / np.sqrt(var_sr)
    psr = float(stats.norm.cdf(z))

    return psr


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    *,
    n_trials: int = 1,
    sharpe_benchmark: float = 0.0,
    var_sharpes: float | None = None,
) -> float:
    """Compute Deflated Sharpe Ratio (DSR).

    Accounts for multiple testing / selection bias.
    Based on Bailey & López de Prado (2014).

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Time series of returns
    n_trials : int
        Number of strategies tested (default 1 = no adjustment)
    sharpe_benchmark : float
        Benchmark Sharpe (default 0)
    var_sharpes : float, optional
        Variance of Sharpes across trials (if known)

    Returns
    -------
    float
        DSR ∈ [0, 1]
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values

    returns = np.asarray(returns).ravel()
    if len(returns) < 3:
        return np.nan

    sr_obs = returns.mean() / (returns.std(ddof=1) + 1e-12)
    n = len(returns)

    skew = float(stats.skew(returns, bias=False))
    kurtosis = float(stats.kurtosis(returns, bias=False))

    var_sr = (1 + 0.5 * sr_obs**2 - skew * sr_obs + (kurtosis / 4) * sr_obs**2) / n

    if var_sr <= 0:
        return np.nan

    # Expected maximum Sharpe under null (Bonferroni correction)
    if var_sharpes is None:
        var_sharpes = var_sr  # Conservative assumption

    euler_gamma = 0.5772156649
    expected_max_sr = (
        sharpe_benchmark + np.sqrt(var_sharpes) *
        ((1 - euler_gamma) * stats.norm.ppf(1 - 1 / n_trials) + euler_gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    )

    # Deflated statistic
    z = (sr_obs - expected_max_sr) / np.sqrt(var_sr)
    dsr = float(stats.norm.cdf(z))

    return dsr


@dataclass
class SkillReport:
    """Results from mu skill diagnostic."""

    ic_mean: float
    ic_std: float
    ic_tstat: float
    ic_pval: float
    ic_hit_rate: float

    r2: float
    r2_adj: float
    beta: float
    beta_pval: float

    sharpe_forecast: float
    psr: float
    dsr: float

    n_periods: int
    n_obs: int

    has_skill: bool
    recommendation: str


def skill_report(
    returns: pd.DataFrame,
    estimator: Callable[[pd.DataFrame], pd.Series],
    *,
    window: int = 252,
    step: int = 21,
    n_trials: int = 1,
    ic_threshold: float = 0.05,
    psr_threshold: float = 0.60,
) -> SkillReport:
    """Run comprehensive skill diagnostic on μ estimator.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns (rows = time, cols = assets)
    estimator : callable
        Function that takes returns DataFrame and returns μ Series
    window : int
        Estimation window (default 252 days)
    step : int
        Step size for rolling windows (default 21 days)
    n_trials : int
        Number of strategies tested (for DSR adjustment)
    ic_threshold : float
        Minimum |IC| to consider skillful
    psr_threshold : float
        Minimum PSR to consider skillful

    Returns
    -------
    SkillReport
    """
    n = len(returns)
    if n < window + step:
        raise ValueError(f"Need at least {window + step} observations, got {n}")

    mu_forecasts = []
    realized_rets = []

    indices = range(window, n - step, step)

    for i in indices:
        train = returns.iloc[i - window : i]
        test = returns.iloc[i : i + step]

        try:
            mu_hat = estimator(train)
            mu_forecasts.append(mu_hat)
            realized_rets.append(test.mean())  # Average return over test period
        except Exception:
            continue

    if len(mu_forecasts) < 2:
        return SkillReport(
            ic_mean=np.nan, ic_std=np.nan, ic_tstat=np.nan, ic_pval=np.nan,
            ic_hit_rate=np.nan, r2=np.nan, r2_adj=np.nan, beta=np.nan,
            beta_pval=np.nan, sharpe_forecast=np.nan, psr=np.nan, dsr=np.nan,
            n_periods=0, n_obs=0, has_skill=False,
            recommendation="Insufficient data for skill test"
        )

    mu_df = pd.DataFrame(mu_forecasts)
    ret_df = pd.DataFrame(realized_rets)

    # IC analysis
    ic_series = information_coefficient(mu_df, ret_df, method="spearman")
    ic_clean = ic_series.dropna()

    ic_mean = float(ic_clean.mean())
    ic_std = float(ic_clean.std())
    ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_clean)) + 1e-12)
    ic_pval = 2 * (1 - stats.t.cdf(abs(ic_tstat), len(ic_clean) - 1))
    ic_hit_rate = float((ic_clean > 0).mean())

    # R² analysis
    r2_results = predictive_r2(mu_df, ret_df)

    # Construct synthetic portfolio returns from forecasts (equal-weight top quintile)
    portfolio_rets = []
    for i in range(len(mu_df)):
        mu_t = mu_df.iloc[i].dropna()
        ret_t = ret_df.iloc[i].dropna()

        common = mu_t.index.intersection(ret_t.index)
        if len(common) < 5:
            continue

        # Top 20% by forecast
        top_quintile = mu_t.loc[common].nlargest(max(1, len(common) // 5)).index
        port_ret = ret_t.loc[top_quintile].mean()
        portfolio_rets.append(port_ret)

    if len(portfolio_rets) > 0:
        port_series = pd.Series(portfolio_rets)
        sharpe_forecast = port_series.mean() / (port_series.std() + 1e-12) * np.sqrt(252 / step)
        psr = probabilistic_sharpe_ratio(port_series, sharpe_benchmark=0.0)
        dsr = deflated_sharpe_ratio(port_series, n_trials=n_trials, sharpe_benchmark=0.0)
    else:
        sharpe_forecast = np.nan
        psr = np.nan
        dsr = np.nan

    # Decision logic
    has_skill = (
        abs(ic_mean) >= ic_threshold and
        ic_pval < 0.05 and
        psr >= psr_threshold
    )

    if has_skill:
        recommendation = f"μ estimator shows skill (IC={ic_mean:.3f}, PSR={psr:.2f}). Safe to use in optimization."
    elif abs(ic_mean) < ic_threshold:
        recommendation = f"IC too low ({ic_mean:.3f}). Use μ=0 or risk-based approach (min-var, risk parity)."
    elif psr < psr_threshold:
        recommendation = f"PSR too low ({psr:.2f}). Likely overfit. Shrink μ aggressively (γ ≥ 0.75) or use μ=0."
    else:
        recommendation = "Inconclusive. Proceed with caution and heavy shrinkage."

    return SkillReport(
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_tstat=ic_tstat,
        ic_pval=ic_pval,
        ic_hit_rate=ic_hit_rate,
        r2=r2_results["r2"],
        r2_adj=r2_results["r2_adj"],
        beta=r2_results["beta"],
        beta_pval=r2_results["beta_pval"],
        sharpe_forecast=sharpe_forecast,
        psr=psr,
        dsr=dsr,
        n_periods=len(mu_forecasts),
        n_obs=r2_results["n_obs"],
        has_skill=has_skill,
        recommendation=recommendation,
    )

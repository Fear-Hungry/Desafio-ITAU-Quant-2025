import math

import numpy as np
import pandas as pd

from arara_quant.utils.psr_dsr import psr_dsr


def test_psr_high_sharpe_has_near_certain_probability():
    np.random.seed(123)
    periods = 5 * 252
    daily_mu = 0.001  # ~25% annualized mean
    daily_sigma = 0.005  # ~8% annualized vol
    returns = pd.Series(np.random.normal(daily_mu, daily_sigma, size=periods))

    result = psr_dsr(returns, S0=0.0, N=10, periods_per_year=252)

    assert result.psr > 0.999
    assert result.dsr <= result.psr
    assert math.isclose(result.periods_per_year, 252)


def test_psr_dsr_decreases_with_more_trials():
    np.random.seed(7)
    returns = pd.Series(np.random.normal(0.0005, 0.01, size=252 * 3))

    res_single = psr_dsr(returns, S0=0.0, N=1)
    res_many = psr_dsr(returns, S0=0.0, N=250)

    assert res_single.psr >= res_many.dsr
    assert res_many.dsr < res_single.psr
    assert res_many.s_star > res_single.s_star


def test_psr_handles_degenerate_series():
    returns = pd.Series([0.0, 0.0, 0.0])
    result = psr_dsr(returns)
    assert result.psr == 0.5
    assert result.dsr == 0.5
    assert result.nu_eff >= 2.0

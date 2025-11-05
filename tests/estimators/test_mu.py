"""Tests for expected-return estimators."""

import numpy as np
import pandas as pd
from arara_quant.estimators import mu as mu_estimators


def _toy_returns() -> pd.DataFrame:
    data = np.array(
        [
            [0.01, 0.015],
            [0.012, 0.016],
            [0.009, 0.014],
            [0.013, 0.017],
        ]
    )
    return pd.DataFrame(data, columns=["AssetA", "AssetB"])


def test_mean_return_simple_matches_numpy():
    returns = _toy_returns()
    result = mu_estimators.mean_return(returns, method="simple")
    expected = returns.mean(axis=0)
    pd.testing.assert_series_equal(result, expected.astype(float))


def test_mean_return_geometric():
    returns = pd.DataFrame(
        [[0.01, 0.02], [0.015, -0.005], [0.0, 0.01]],
        columns=["AssetA", "AssetB"],
    )
    result = mu_estimators.mean_return(returns, method="geometric")
    growth = (1.0 + returns).prod(axis=0) ** (1.0 / len(returns)) - 1.0
    pd.testing.assert_series_equal(result, growth.astype(float))


def test_huber_mean_downweights_outlier():
    returns = _toy_returns().copy()
    returns.loc[0, "AssetA"] = 0.5  # strong outlier
    huber_mean, weights = mu_estimators.huber_mean(returns, c=1.5)
    simple_mean = mu_estimators.mean_return(returns)
    clean_mean = mu_estimators.mean_return(_toy_returns())

    assert abs(huber_mean["AssetA"] - clean_mean["AssetA"]) < abs(
        simple_mean["AssetA"] - clean_mean["AssetA"]
    )
    assert weights.loc[0, "AssetA"] < 1.0


def test_student_t_mean_close_to_sample_for_moderate_nu():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0.01, 0.02, size=(200, 2)), columns=["A", "B"])
    student = mu_estimators.student_t_mean(returns, nu=8.0)
    sample = returns.mean()
    np.testing.assert_allclose(student.to_numpy(), sample.to_numpy(), atol=5e-4)


def test_bayesian_shrinkage_mean_moves_towards_prior():
    returns = _toy_returns()
    prior = pd.Series({"AssetA": 0.02, "AssetB": 0.01})
    shrunk = mu_estimators.bayesian_shrinkage_mean(returns, prior=prior, strength=0.5)
    sample = returns.mean()
    assert np.all(np.sign(shrunk - sample) == np.sign(prior - sample))


def test_shrunk_mean_approaches_zero_prior():
    returns = _toy_returns()
    shrunk = mu_estimators.shrunk_mean(returns, strength=0.5)
    sample = returns.mean()
    expected = sample * 0.5  # zero prior
    pd.testing.assert_series_equal(shrunk, expected.astype(float))


def test_confidence_intervals_contains_sample_mean():
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.normal(0.01, 0.02, size=(100, 2)), columns=["A", "B"])
    ci = mu_estimators.confidence_intervals(
        returns, method="bootstrap", alpha=0.1, n_bootstrap=300, random_state=1
    )
    sample = returns.mean()
    assert ((sample >= ci["lower"]) & (sample <= ci["upper"])).all()


def test_blend_with_black_litterman_passthrough():
    mu_prior = pd.Series({"A": 0.01, "B": 0.015})
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.05]], index=["A", "B"], columns=["A", "B"]
    )
    blended = mu_estimators.blend_with_black_litterman(mu_prior, cov, views=None)
    pd.testing.assert_series_equal(blended, mu_prior.astype(float))


def test_annualize_simple_and_compound():
    mu = pd.Series({"A": 0.01, "B": 0.015})
    simple = mu_estimators.annualize(mu, periods_per_year=12, compound=False)
    compound = mu_estimators.annualize(mu, periods_per_year=12, compound=True)
    pd.testing.assert_series_equal(simple, mu * 12)
    expected_compound = (1.0 + mu) ** 12 - 1.0
    pd.testing.assert_series_equal(compound, expected_compound)

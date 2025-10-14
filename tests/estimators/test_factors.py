"""Unit tests for factor modelling helpers."""

import numpy as np
import pandas as pd
import pytest

from itau_quant.estimators import factors as factors_mod


def _build_prices() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    data = np.linspace(100, 110, num=len(index))
    prices = pd.DataFrame(
        {
            "AssetA": data,
            "AssetB": data * 1.01,
        },
        index=index,
    )
    return prices


def _build_factors() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    rng = np.random.default_rng(42)
    values = rng.normal(scale=0.01, size=(len(index), 2))
    return pd.DataFrame(values, index=index, columns=["MKT", "SMB"])


def test_prepare_factor_data_zscores_and_aligns():
    prices = _build_prices()
    factors = _build_factors()

    asset_z, factor_z = factors_mod.prepare_factor_data(
        prices, factors, window=None
    )

    assert asset_z.index.equals(factor_z.index)
    assert not asset_z.isna().any().any()
    assert not factor_z.isna().any().any()

    np.testing.assert_allclose(asset_z.mean().to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(factor_z.mean().to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(asset_z.std(ddof=0).to_numpy(), 1.0, atol=1e-12)
    np.testing.assert_allclose(factor_z.std(ddof=0).to_numpy(), 1.0, atol=1e-12)


def test_time_series_regression_recovers_known_betas():
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=120, freq="D")
    factors = pd.DataFrame(
        rng.normal(size=(len(index), 2)), index=index, columns=["MKT", "SMB"]
    )
    betas_true = pd.DataFrame(
        [[1.2, 0.5, -0.3], [0.8, -0.1, 0.6]],
        index=["MKT", "SMB"],
        columns=["Asset1", "Asset2", "Asset3"],
    )
    alpha_true = pd.Series([0.01, -0.005, 0.002], index=betas_true.columns)

    returns = factors.to_numpy() @ betas_true.to_numpy()
    returns = returns + alpha_true.to_numpy()
    returns_df = pd.DataFrame(returns, index=index, columns=betas_true.columns)

    betas, alphas, residuals = factors_mod.time_series_regression(
        returns_df, factors
    )

    np.testing.assert_allclose(betas, betas_true, atol=1e-10)
    np.testing.assert_allclose(alphas, alpha_true, atol=1e-10)
    np.testing.assert_allclose(residuals.to_numpy(), 0.0, atol=1e-10)


def test_cross_sectional_regression_recovers_premia():
    betas = pd.DataFrame(
        [[1.0, 0.5, -0.2], [0.2, -0.1, 0.3]],
        index=["MKT", "SMB"],
        columns=["Asset1", "Asset2", "Asset3"],
    )
    premia_true = pd.Series({"MKT": 0.04, "SMB": -0.01})
    alpha_true = 0.002
    future_returns = betas.T @ premia_true + alpha_true

    premia, alpha = factors_mod.cross_sectional_regression(
        betas, future_returns, add_constant=True
    )

    np.testing.assert_allclose(premia, premia_true, atol=1e-10)
    assert alpha is not None
    assert abs(alpha - alpha_true) < 1e-10


def test_shrink_betas_methods():
    betas = pd.DataFrame(
        [[0.5, -0.2], [1.0, -1.5]],
        index=["MKT", "SMB"],
        columns=["Asset1", "Asset2"],
    )

    ridge = factors_mod.shrink_betas(betas, method="ridge", alpha=0.5)
    np.testing.assert_allclose(ridge, betas / 1.5)

    lasso = factors_mod.shrink_betas(betas, method="lasso", alpha=0.3)
    expected_lasso_vals = np.sign(betas.to_numpy()) * np.maximum(
        np.abs(betas.to_numpy()) - 0.3, 0.0
    )
    expected_lasso = pd.DataFrame(
        expected_lasso_vals, index=betas.index, columns=betas.columns
    )
    np.testing.assert_allclose(lasso, expected_lasso)

    grand_mean = factors_mod.shrink_betas(betas, method="grand_mean", alpha=0.2)
    mean_beta = betas.mean(axis=1)
    target = pd.DataFrame(
        np.repeat(mean_beta.values[:, None], betas.shape[1], axis=1),
        index=betas.index,
        columns=betas.columns,
    )
    expected_grand = 0.8 * betas + 0.2 * target
    np.testing.assert_allclose(grand_mean, expected_grand)


def test_factor_covariance_methods_return_psd():
    rng = np.random.default_rng(1)
    factors = pd.DataFrame(rng.normal(size=(100, 3)), columns=["MKT", "SMB", "HML"])

    sample = factors_mod.factor_covariance(factors, method="sample")
    lw = factors_mod.factor_covariance(factors, method="ledoit_wolf")

    for cov in (sample, lw):
        eigvals = np.linalg.eigvalsh(cov.to_numpy())
        assert np.all(eigvals >= -1e-9)


def test_implied_asset_returns_combines_premia_and_alpha():
    betas = pd.DataFrame(
        [[1.0, 0.5], [0.2, -0.1]],
        index=["MKT", "SMB"],
        columns=["Asset1", "Asset2"],
    )
    premia = pd.Series({"MKT": 0.05, "SMB": 0.01})
    alpha = pd.Series({"Asset1": 0.002, "Asset2": -0.001})

    implied = factors_mod.implied_asset_returns(betas, premia, alpha)
    expected = betas.T @ premia + alpha
    np.testing.assert_allclose(implied.sort_index(), expected.sort_index())


def test_principal_component_factors_reconstructs_data():
    rng = np.random.default_rng(123)
    T, N, K = 120, 4, 2
    loadings_true = rng.normal(size=(N, K))
    factor_ts = rng.normal(size=(T, K))
    returns = factor_ts @ loadings_true.T
    returns_df = pd.DataFrame(returns, columns=[f"Asset{i}" for i in range(N)])

    factor_returns, loadings, explained = factors_mod.principal_component_factors(
        returns_df, n_components=K
    )

    reconstructed = factor_returns.to_numpy() @ loadings.to_numpy().T
    demeaned = returns_df - returns_df.mean(axis=0)
    np.testing.assert_allclose(reconstructed, demeaned.to_numpy(), atol=1e-10)
    assert explained.sum() == pytest.approx(1.0, rel=1e-9)

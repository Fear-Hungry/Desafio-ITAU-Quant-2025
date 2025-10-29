import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal

from itau_quant.estimators.bl import (
    black_litterman,
    build_projection_matrix,
    posterior_returns,
    reverse_optimization,
    view_uncertainty,
    _solve_psd,
)


@pytest.fixture
def market_data() -> dict:
    assets = ["A", "B", "C"]
    cov = pd.DataFrame(
        np.diag([0.04, 0.09, 0.16]),
        index=assets,
        columns=assets,
    )
    weights = pd.Series([1.0 / 3] * 3, index=assets)
    pi, delta = reverse_optimization(weights, cov, risk_aversion=3.0)
    return {"assets": assets, "cov": cov, "weights": weights, "pi": pi, "delta": delta}


def test_reverse_optimization_infers_risk_aversion():
    assets = ["A", "B", "C"]
    cov = pd.DataFrame(
        np.diag([0.04, 0.09, 0.16]),
        index=assets,
        columns=assets,
    )
    weights = pd.Series([0.2, 0.5, 0.3], index=assets)
    market_return = 0.14
    risk_free = 0.02

    pi, delta = reverse_optimization(
        weights=weights,
        cov=cov,
        market_return=market_return,
        risk_free=risk_free,
    )

    variance = float(weights.to_numpy() @ cov.to_numpy() @ weights.to_numpy())
    expected_delta = (market_return - risk_free) / variance
    expected_pi = cov @ weights * expected_delta

    assert delta == pytest.approx(expected_delta)
    assert_series_equal(pi, expected_pi, check_names=False)


def test_reverse_optimization_requires_non_zero_risk_premium():
    assets = ["A", "B", "C"]
    cov = pd.DataFrame(
        np.diag([0.04, 0.09, 0.16]),
        index=assets,
        columns=assets,
    )
    weights = pd.Series([1.0 / 3] * 3, index=assets)

    with pytest.raises(
        ValueError, match="market_return must differ from risk_free"
    ):
        reverse_optimization(
            weights=weights,
            cov=cov,
            market_return=0.08,
            risk_free=0.08,
        )


def test_black_litterman_without_views_returns_prior(market_data):
    result = black_litterman(
        cov=market_data["cov"],
        pi=market_data["pi"],
        tau=0.05,
        return_intermediates=True,
    )

    assert_series_equal(result["mu_bl"], market_data["pi"], check_names=False)
    assert_frame_equal(result["cov_bl"], market_data["cov"])
    expected_sigma = market_data["cov"] * 0.05
    assert_frame_equal(
        result["intermediates"]["mean_uncertainty"], expected_sigma.astype(float)
    )
    assert "Omega" in result["intermediates"]
    assert result["intermediates"]["Omega"].shape == (0, 0)


def test_black_litterman_without_views_add_uncertainty(market_data):
    tau = 0.1
    result = black_litterman(
        cov=market_data["cov"],
        pi=market_data["pi"],
        tau=tau,
        add_mean_uncertainty=True,
        return_intermediates=True,
    )

    expected_sigma = market_data["cov"] * tau
    assert_frame_equal(
        result["intermediates"]["mean_uncertainty"], expected_sigma.astype(float)
    )
    assert_frame_equal(
        result["cov_bl"],
        (market_data["cov"] + expected_sigma).astype(float),
    )


def test_black_litterman_absolute_view_diagonal_mode(market_data):
    view = {
        "type": "absolute",
        "asset": "A",
        "expected_return": 0.08,
        "confidence": 0.6,
    }

    result = black_litterman(
        cov=market_data["cov"],
        weights=market_data["weights"],
        risk_aversion=market_data["delta"],
        views=[view],
        tau=0.05,
        return_intermediates=True,
    )

    mu_bl = result["mu_bl"]
    cov_bl = result["cov_bl"]
    omega = result["intermediates"]["Omega"]
    mean_uncertainty = result["intermediates"]["mean_uncertainty"]

    assert mu_bl["A"] == pytest.approx(0.064, rel=1e-6)
    assert mu_bl["B"] == pytest.approx(market_data["pi"]["B"], rel=1e-9)
    assert mu_bl["C"] == pytest.approx(market_data["pi"]["C"], rel=1e-9)

    c = np.clip(view["confidence"], 0.0, 1.0)
    expected_alpha = np.clip((1 - c) / max(c, 1e-6), 1e-4, 1e6)
    expected_omega = 0.05 * 0.04 * expected_alpha
    assert omega.shape == (1, 1)
    assert omega[0, 0] == pytest.approx(expected_omega, rel=1e-9)

    assert_frame_equal(cov_bl, market_data["cov"])
    assert mean_uncertainty.loc["A", "A"] > 0
    assert mean_uncertainty.loc["B", "B"] > 0
    assert mean_uncertainty.loc["C", "C"] > 0


def test_black_litterman_relative_view_with_custom_omega(market_data):
    view = {
        "type": "relative",
        "long": ["B"],
        "short": ["C"],
        "expected_spread": 0.01,
        "confidence": 0.7,
    }
    custom_omega = np.array([[0.002]])

    result = black_litterman(
        cov=market_data["cov"],
        pi=market_data["pi"],
        views=[view],
        tau=0.05,
        omega_mode="custom",
        user_Omega=custom_omega,
        return_intermediates=True,
    )

    mu_bl = result["mu_bl"]
    P = result["intermediates"]["P"]
    Omega = result["intermediates"]["Omega"]
    pi = market_data["pi"]

    assert_allclose(Omega, custom_omega)
    assert_allclose(P, np.array([[0.0, 0.5, -0.5]]))
    assert mu_bl["B"] > pi["B"]
    assert mu_bl["C"] < pi["C"]

    prior_spread = np.asarray(P @ pi.to_numpy()).item()
    posterior_spread = np.asarray(P @ mu_bl.to_numpy()).item()
    prior_distance = abs(view["expected_spread"] - prior_spread)
    posterior_distance = abs(view["expected_spread"] - posterior_spread)
    assert posterior_distance < prior_distance


def test_black_litterman_zero_tau_with_views_returns_prior(market_data):
    view = {
        "type": "absolute",
        "asset": "A",
        "expected_return": 0.08,
        "confidence": 0.6,
    }

    result = black_litterman(
        cov=market_data["cov"],
        pi=market_data["pi"],
        views=[view],
        tau=0.0,
        return_intermediates=True,
    )

    assert_series_equal(result["mu_bl"], market_data["pi"], check_names=False)
    assert_frame_equal(result["cov_bl"], market_data["cov"])
    zeros = pd.DataFrame(
        np.zeros_like(market_data["cov"].to_numpy()),
        index=market_data["cov"].index,
        columns=market_data["cov"].columns,
    )
    assert_frame_equal(result["intermediates"]["mean_uncertainty"], zeros.astype(float))


def test_black_litterman_tau_limits_with_custom_omega(market_data):
    view = {
        "type": "absolute",
        "asset": "A",
        "expected_return": 0.08,
        "confidence": 0.8,
    }
    custom_omega = np.array([[0.005]])

    low_tau = black_litterman(
        cov=market_data["cov"],
        pi=market_data["pi"],
        views=[view],
        tau=1e-6,
        omega_mode="custom",
        user_Omega=custom_omega,
    )

    high_tau = black_litterman(
        cov=market_data["cov"],
        pi=market_data["pi"],
        views=[view],
        tau=10.0,
        omega_mode="custom",
        user_Omega=custom_omega,
    )

    assert abs(low_tau["mu_bl"]["A"] - market_data["pi"]["A"]) < 1e-4
    assert abs(high_tau["mu_bl"]["A"] - view["expected_return"]) < 1e-3
    assert_series_equal(
        low_tau["mu_bl"][["B", "C"]],
        market_data["pi"][["B", "C"]],
        check_names=False,
    )
    assert_series_equal(
        high_tau["mu_bl"][["B", "C"]],
        market_data["pi"][["B", "C"]],
        check_names=False,
    )


def test_posterior_returns_without_views_returns_tau_sigma():
    assets = ["A", "B", "C"]
    pi = pd.Series([0.01, 0.02, 0.015], index=assets)
    cov = pd.DataFrame(
        np.diag([0.04, 0.09, 0.16]),
        index=assets,
        columns=assets,
    )
    tau = 0.05
    P = np.zeros((0, len(assets)))
    Q = np.zeros(0)
    Omega = np.zeros((0, 0))

    mu_bl, cov_bl, sigma = posterior_returns(
        pi,
        cov,
        P,
        Q,
        Omega,
        tau,
        return_sigma_posterior=True,
    )

    expected_sigma = cov * tau
    assert_series_equal(mu_bl, pi.astype(float), check_names=False)
    assert_frame_equal(cov_bl, cov.astype(float))
    assert_frame_equal(sigma, expected_sigma.astype(float))


def test_posterior_returns_raises_when_q_length_mismatch():
    assets = ["A", "B", "C"]
    pi = pd.Series([0.01, 0.015, 0.02], index=assets)
    cov = pd.DataFrame(
        np.diag([0.04, 0.09, 0.16]),
        index=assets,
        columns=assets,
    )
    P = np.array([[1.0, -1.0, 0.0]])
    Q = np.zeros(0)
    Omega = np.eye(1)

    with pytest.raises(ValueError):
        posterior_returns(pi, cov, P, Q, Omega, tau=0.05)


def test_posterior_returns_zero_tau_returns_prior():
    assets = ["A", "B", "C"]
    pi = pd.Series([0.01, 0.02, 0.015], index=assets)
    cov = pd.DataFrame(
        np.diag([0.05, 0.06, 0.07]),
        index=assets,
        columns=assets,
    )
    P = np.array([[1.0, 0.0, -1.0]])
    Q = np.array([0.02])
    Omega = np.eye(1)

    mu_bl, cov_bl, sigma = posterior_returns(
        pi,
        cov,
        P,
        Q,
        Omega,
        tau=0.0,
        add_mean_uncertainty=True,
        return_sigma_posterior=True,
    )

    zeros = pd.DataFrame(
        np.zeros_like(cov.to_numpy()),
        index=cov.index,
        columns=cov.columns,
    )
    assert_series_equal(mu_bl, pi.astype(float), check_names=False)
    assert_frame_equal(cov_bl, cov.astype(float))
    assert_frame_equal(sigma, zeros.astype(float))


def test_black_litterman_add_mean_uncertainty_flag(market_data):
    view = {
        "type": "absolute",
        "asset": "A",
        "expected_return": 0.08,
        "confidence": 0.5,
    }

    baseline = black_litterman(
        cov=market_data["cov"],
        pi=market_data["pi"],
        views=[view],
        tau=0.05,
    )

    enriched = black_litterman(
        cov=market_data["cov"],
        pi=market_data["pi"],
        views=[view],
        tau=0.05,
        add_mean_uncertainty=True,
    )

    assert_frame_equal(baseline["cov_bl"], market_data["cov"])
    assert enriched["cov_bl"].loc["A", "A"] > market_data["cov"].loc["A", "A"]
    assert enriched["cov_bl"].loc["B", "B"] >= market_data["cov"].loc["B", "B"]
    assert enriched["cov_bl"].loc["C", "C"] >= market_data["cov"].loc["C", "C"]


def test_build_projection_matrix_relative_weights_respect_normalization():
    assets = ["A", "B", "C"]
    views = [
        {
            "type": "relative",
            "long": ["A", "B"],
            "short": ["C"],
            "expected_spread": 0.015,
            "confidence": 0.5,
        }
    ]

    P, Q, confidences = build_projection_matrix(views, assets)

    assert_allclose(Q, np.array([0.015]))
    assert_allclose(confidences, np.array([0.5]))
    assert_allclose(P.sum(axis=1), np.zeros(1))
    assert np.isclose(np.sum(np.abs(P[0])), 1.0)


def test_posterior_mean_identity_holds(market_data):
    tau = 0.05
    view = {
        "type": "absolute",
        "asset": "A",
        "expected_return": 0.08,
        "confidence": 0.7,
    }
    views = [view]
    P, Q, confidences = build_projection_matrix(views, market_data["assets"])
    Omega = view_uncertainty(
        views=views,
        tau=tau,
        cov=market_data["cov"],
        P=P,
        confidences=confidences,
    )

    mu_bl, _, _ = posterior_returns(
        pi=market_data["pi"],
        cov=market_data["cov"],
        P=P,
        Q=Q,
        Omega=Omega,
        tau=tau,
    )

    pi_vec = market_data["pi"].to_numpy()
    cov_mat = market_data["cov"].to_numpy()
    tau_sigma = tau * cov_mat

    # Forma 1
    lhs_adjustment = _solve_psd(P @ tau_sigma @ P.T + Omega, (Q - P @ pi_vec)[:, None]).ravel()
    mu_form1 = pi_vec + tau_sigma @ P.T @ lhs_adjustment

    # Forma 2
    inv_tau_sigma = _solve_psd(tau_sigma, np.eye(len(pi_vec)))
    Omega_inv = _solve_psd(Omega, np.eye(Omega.shape[0]))
    rhs = inv_tau_sigma @ pi_vec + P.T @ Omega_inv @ Q
    mu_form2 = _solve_psd(inv_tau_sigma + P.T @ Omega_inv @ P, rhs)

    assert_allclose(mu_form1, mu_form2, rtol=1e-8, atol=1e-8)
    assert_allclose(mu_form1, mu_bl.to_numpy(), rtol=1e-8, atol=1e-8)


def test_posterior_returns_without_views_preserves_uncertainty(market_data):
    tau = 0.12
    n_assets = len(market_data["assets"])
    empty_P = np.zeros((0, n_assets))
    empty_Q = np.zeros(0)
    empty_Omega = np.zeros((0, 0))

    mu_bl, cov_bl, sigma_post = posterior_returns(
        pi=market_data["pi"],
        cov=market_data["cov"],
        P=empty_P,
        Q=empty_Q,
        Omega=empty_Omega,
        tau=tau,
        return_sigma_posterior=True,
    )

    assert_series_equal(mu_bl, market_data["pi"], check_names=False)
    expected_sigma = market_data["cov"] * tau
    assert_frame_equal(cov_bl, market_data["cov"])
    assert_frame_equal(sigma_post, expected_sigma.astype(float))


def test_posterior_returns_fast_path_skips_sigma_inverse(market_data, monkeypatch):
    call_log = []
    original_solve = _solve_psd

    def tracking_solve(A: np.ndarray, B: np.ndarray, jitter: float = 1e-10):
        call_log.append(B.shape)
        if B.ndim == 2 and B.shape[0] == B.shape[1] and np.allclose(
            B, np.eye(B.shape[0])
        ):
            raise AssertionError("Fast path should not solve against identity.")
        return original_solve(A, B, jitter=jitter)

    monkeypatch.setattr(
        "itau_quant.estimators.bl._solve_psd",
        tracking_solve,
        raising=True,
    )

    tau = 0.05
    view = {
        "type": "absolute",
        "asset": "A",
        "expected_return": 0.07,
        "confidence": 0.6,
    }
    views = [view]
    P, Q, confidences = build_projection_matrix(views, market_data["assets"])
    Omega = view_uncertainty(
        views=views,
        tau=tau,
        cov=market_data["cov"],
        P=P,
        confidences=confidences,
    )

    mu_bl, cov_bl, sigma_post = posterior_returns(
        pi=market_data["pi"],
        cov=market_data["cov"],
        P=P,
        Q=Q,
        Omega=Omega,
        tau=tau,
    )

    assert_frame_equal(cov_bl, market_data["cov"])
    assert not mu_bl.equals(market_data["pi"])
    assert sigma_post is None
    assert call_log  # ensure path executed

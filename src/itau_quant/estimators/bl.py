"""Black-Litterman estimator blueprint.

Objetivo
--------
Fornecer funções para gerar retornos esperados combinando equilíbrio de mercado
com opiniões discretas, seguindo o framework de Black-Litterman.

Componentes sugeridos
---------------------
- `reverse_optimization(weights, cov, risk_aversion)`:
    Derive o vetor de retornos implícitos (π) a partir das alocações de mercado.
- `build_projection_matrix(views)`:
    Converte views estruturadas (dict/dataclass) em matrizes ``P`` e ``Q``.
- `view_uncertainty(views, tau, cov)`:
    Constrói ``Omega`` (matriz de incerteza das views) suportando opções:
    * escala diagonal usando volatilidade do ativo;
    * fator comum (idêntico para todas as views);
    * matriz customizada fornecida pelo usuário.
- `posterior_returns(pi, cov, P, Q, Omega, tau)`:
    Combina prior e views retornando μ_BL e matriz de covariância ajustada.
- `black_litterman(...)`:
    Função principal que orquestra as etapas acima, aplicando verificações
    numéricas (PSD, condicionamento) e normalizando pesos.

Checklist de implementação
--------------------------
- Validar dimensões entre ``P``, ``Q`` e número de ativos.
- Usar fator de escala ``tau`` configurável.
- Permitir prior via pesos de mercado ou vetor já calculado.
- Suportar views absolutas e relativas.
- Explicar claramente no docstring das funções como os inputs devem ser
  estruturados (e.g., `views=[{"type": "absolute", "ticker": "SPY", ...}]`).
- Incluir testes em `tests/estimators/test_bl.py` cobrindo:
    * ausência de views (retorno igual ao prior),
    * views absolutas simples,
    * views relativas com Omega customizado,
    * comportamento quando tau → 0 (prior domina) e tau grande (views dominam).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, Sequence[float], pd.Series]
MatrixLike = Union[np.ndarray, pd.DataFrame]

# ------------------
# Core Functions
# ------------------


def reverse_optimization(
    weights: Union[pd.Series, Sequence[float], np.ndarray],
    cov: Union[pd.DataFrame, np.ndarray],
    risk_aversion: Optional[float] = None,
    market_return: Optional[float] = None,
    risk_free: float = 0.0,
) -> Tuple[pd.Series, float]:
    """
    Obtém retornos de equilíbrio (π) via reverse optimization.

    Regras:
    - Se risk_aversion (δ) fornecido: π = δ Σ w.
    - Senão, se market_return fornecido: δ = (market_return - risk_free)/(wᵀ Σ w).
    - Retorna (pi, delta).

    Parameters
    ----------
    weights : Series ou array (n,)
        Pesos de mercado (somam ~1).
    cov : DataFrame ou ndarray (n,n)
        Matriz de covariância.
    risk_aversion : float, opcional
        Parâmetro δ. Se None, será inferido de market_return.
    market_return : float, opcional
        Retorno esperado do portfólio de mercado (annualizado).
    risk_free : float
        Taxa livre de risco (para inferir prêmio de risco).

    Returns
    -------
    pi : Series
        Retornos implícitos de equilíbrio.
    delta : float
        Aversão a risco utilizada.
    """
    if isinstance(cov, pd.DataFrame):
        cov_df = cov.astype(float).copy()
        asset_names = list(cov_df.columns)
    else:
        cov_arr = np.asarray(cov, dtype=float)
        if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
            raise ValueError("cov must be a square matrix.")
        asset_names = None
        cov_df = pd.DataFrame(cov_arr)

    if isinstance(weights, pd.Series):
        weights_series = weights.astype(float)
        if asset_names is None:
            asset_names = list(weights_series.index)
    else:
        weights_arr = np.asarray(weights, dtype=float).flatten()
        if weights_arr.ndim != 1:
            raise ValueError("weights must be one-dimensional.")
        if asset_names is None:
            asset_names = list(range(len(weights_arr)))
        weights_series = pd.Series(weights_arr, index=asset_names, dtype=float)

    if len(weights_series) != cov_df.shape[0]:
        raise ValueError("weights and cov dimension mismatch.")

    cov_df.index = asset_names
    cov_df.columns = asset_names
    weights_series = weights_series.reindex(asset_names)
    if weights_series.isnull().any():
        raise ValueError("weights must include all assets present in cov.")

    total_weight = float(weights_series.sum())
    if np.isclose(total_weight, 0.0):
        raise ValueError("weights must sum to a non-zero value.")
    if not np.isclose(total_weight, 1.0):
        weights_series = weights_series / total_weight

    cov_matrix = cov_df.to_numpy()
    weights_vec = weights_series.to_numpy()

    if risk_aversion is not None and risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive when provided.")

    inferred_delta = risk_aversion
    if inferred_delta is None:
        if market_return is None:
            raise ValueError("market_return required when risk_aversion is None.")
        port_var = float(weights_vec @ cov_matrix @ weights_vec)
        if port_var <= 0 or np.isclose(port_var, 0.0):
            raise ValueError(
                "Portfolio variance must be strictly positive to infer risk aversion."
            )
        risk_premium = float(market_return - risk_free)
        if np.isclose(risk_premium, 0.0):
            raise ValueError(
                "market_return must differ from risk_free to infer risk_aversion."
            )
        inferred_delta = risk_premium / port_var
        if inferred_delta <= 0:
            raise ValueError("Computed risk_aversion must be positive.")

    pi_values = cov_matrix @ weights_vec * float(inferred_delta)
    pi_series = pd.Series(pi_values, index=asset_names, dtype=float)

    return pi_series, float(inferred_delta)


def build_projection_matrix(
    views: Optional[List[Dict[str, Any]]],
    assets: Sequence[str],
    normalize_relative: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constrói P, Q e vetor de confidences a partir de lista de views.

    Parameters
    ----------
    views : list[dict]
        Cada view deve ter "type": "absolute" ou "relative".
        O campo "confidence" deve estar em [0, 1]; valores > 1 são truncados.
        Formatos:
            Absolute:
            {"type":"absolute","asset":"AAPL","expected_return":0.08,"confidence":0.7}
            Relative:
            {"type":"relative","long":["AAPL"],"short":["MSFT"],
                "expected_spread":0.02,"confidence":0.6,
                "long_weights":[... opcional ...],"short_weights":[... opcional ...]}
    assets : sequência de str
        Ordem global dos ativos (colunas de Σ).
    normalize_relative : bool
        Se True, normaliza pesos long e short para soma de absolutos igual a 1.

    Returns
    -------
    P : ndarray (k, n)
    Q : ndarray (k,)
    confidences : ndarray (k,)
    """
    if views is None or len(views) == 0:
        n_assets = len(assets)
        return np.zeros((0, n_assets)), np.zeros(0), np.zeros(0)

    asset_list = list(assets)
    asset_to_idx = {asset: idx for idx, asset in enumerate(asset_list)}

    P_rows: List[np.ndarray] = []
    Q_vals: List[float] = []
    confidences: List[float] = []

    for view in views:
        if "type" not in view:
            raise ValueError("Each view must declare a 'type'.")
        vtype = view["type"]
        confidence = float(view.get("confidence", 1.0))
        if confidence < 0 or confidence > 1:
            raise ValueError("View confidence must be in [0, 1].")

        row = np.zeros(len(asset_list), dtype=float)

        if vtype == "absolute":
            asset = view.get("asset")
            if asset not in asset_to_idx:
                raise ValueError(f"Asset '{asset}' not present in assets universe.")
            if "expected_return" not in view:
                raise ValueError("Absolute views require 'expected_return'.")
            row[asset_to_idx[asset]] = 1.0
            q_val = float(view["expected_return"])

        elif vtype == "relative":
            long_assets = view.get("long")
            short_assets = view.get("short")
            if not long_assets or not short_assets:
                raise ValueError(
                    "Relative views require non-empty 'long' and 'short' assets."
                )
            long_weights = view.get("long_weights")
            short_weights = view.get("short_weights")

            if long_weights is None:
                long_weights = [1.0 / len(long_assets)] * len(long_assets)
            if short_weights is None:
                short_weights = [1.0 / len(short_assets)] * len(short_assets)

            if len(long_weights) != len(long_assets):
                raise ValueError(
                    "Length of long_weights must match length of long assets."
                )
            if len(short_weights) != len(short_assets):
                raise ValueError(
                    "Length of short_weights must match length of short assets."
                )

            for asset, weight in zip(long_assets, long_weights):
                if asset not in asset_to_idx:
                    raise ValueError(f"Asset '{asset}' not present in assets universe.")
                row[asset_to_idx[asset]] += float(weight)
            for asset, weight in zip(short_assets, short_weights):
                if asset not in asset_to_idx:
                    raise ValueError(f"Asset '{asset}' not present in assets universe.")
                row[asset_to_idx[asset]] -= float(weight)

            if normalize_relative:
                scale = np.sum(np.abs(row))
                if scale > 0:
                    row = row / scale

            if "expected_spread" not in view:
                raise ValueError("Relative views require 'expected_spread'.")
            q_val = float(view["expected_spread"])
        else:
            raise ValueError("View type must be 'absolute' or 'relative'.")

        P_rows.append(row)
        Q_vals.append(q_val)
        confidences.append(confidence)

    P = np.vstack(P_rows) if P_rows else np.zeros((0, len(asset_list)))
    Q = np.asarray(Q_vals, dtype=float)
    conf_array = np.asarray(confidences, dtype=float)
    return P, Q, conf_array


def view_uncertainty(
    views: Optional[List[Dict[str, Any]]],
    tau: float,
    cov: Union[pd.DataFrame, np.ndarray],
    P: np.ndarray,
    confidences: np.ndarray,
    mode: str = "diagonal",
    user_Omega: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    min_var: float = 1e-12,
) -> np.ndarray:
    """
    Constrói matriz Ω (k×k) de incerteza das views.

    Estratégias:
    - diagonal (default): Ω_i = α_i * P_i (τΣ) P_iᵀ,
        com α_i = (1 - c_i)/c_i limitado a [1e-4, 1e6]
    - scalar: fator único baseado em média de P_i (τΣ) P_iᵀ
    - custom: usa user_Omega validada

    Parameters
    ----------
    views : lista de views (apenas usada para validação de k)
    tau : float
    cov : Σ
    P : ndarray (k,n)
    confidences : ndarray (k,)
    mode : {"diagonal","scalar","custom"}
    user_Omega : matriz customizada (k,k)
    min_var : float
        Piso numérico para variâncias.

    Returns
    -------
    Omega : ndarray (k,k)
    """
    k = P.shape[0]
    if k == 0:
        return np.zeros((0, 0))

    if confidences.shape[0] != k:
        raise ValueError(
            "Confidences vector size must match number of views (rows in P)."
        )

    if np.any(confidences < 0):
        raise ValueError("Confidences must be non-negative.")

    if isinstance(cov, pd.DataFrame):
        cov_df = cov.astype(float)
    else:
        cov_arr = np.asarray(cov, dtype=float)
        if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
            raise ValueError("cov must be a square matrix.")
        cov_df = pd.DataFrame(cov_arr)

    if P.shape[1] != cov_df.shape[0]:
        raise ValueError("P columns must match number of assets in cov.")

    tau_cov = tau * cov_df.to_numpy()
    base_vars = np.einsum("ij,jk,ik->i", P, tau_cov, P)
    base_vars = np.maximum(base_vars, min_var)

    mode = mode.lower()
    if mode not in {"diagonal", "scalar", "custom"}:
        raise ValueError("mode must be 'diagonal', 'scalar', or 'custom'.")

    if mode == "diagonal":
        c = np.clip(confidences, 0.0, 1.0)
        alpha = (1.0 - c) / np.maximum(c, 1e-6)
        alpha = np.clip(alpha, 1e-4, 1e6)
        diag_vals = np.maximum(base_vars * alpha, min_var)
        Omega = np.diag(diag_vals)
        return Omega

    if mode == "scalar":
        c = np.clip(confidences, 0.0, 1.0)
        alpha = (1.0 - c) / np.maximum(c, 1e-6)
        alpha = np.clip(alpha, 1e-4, 1e6)
        scalar = float(np.mean(base_vars * alpha))
        scalar = max(scalar, min_var)
        return np.eye(k) * scalar

    if user_Omega is None:
        raise ValueError("user_Omega must be provided when mode='custom'.")

    omega = np.asarray(user_Omega, dtype=float)
    if omega.shape != (k, k):
        raise ValueError("user_Omega must have shape (k, k).")
    omega = 0.5 * (omega + omega.T)
    eigvals = np.linalg.eigvalsh(omega)
    if np.any(eigvals < -1e-10):
        raise ValueError("user_Omega must be positive semi-definite.")
    np.fill_diagonal(omega, np.maximum(np.diag(omega), min_var))
    return omega


def posterior_returns(
    pi: Union[pd.Series, np.ndarray],
    cov: Union[pd.DataFrame, np.ndarray],
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float,
    *,
    add_mean_uncertainty: bool = False,
    return_sigma_posterior: bool = False,
    solver_jitter: float = 1e-10,
) -> Tuple[pd.Series, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Calcula o par (μ_BL, Σ) e, opcionalmente, a matriz de covariância da
    incerteza dos retornos esperados (posterior do mean).

    Parameters
    ----------
    pi : prior (n,)
    cov : Σ (n,n)
    P : (k,n)
    Q : (k,)
    Omega : (k,k)
    tau : float
    add_mean_uncertainty : bool
        Se True, adiciona a matriz da incerteza do mean (M) ao retorno de Σ.
    return_sigma_posterior : bool
        Se True, retorna DataFrame com M = [(τΣ)^-1 + PᵀΩ^-1P]^-1.
    solver_jitter : float
        Perturbação na diagonal usada para estabilizar sistemas lineares.

    Returns
    -------
    mu_bl : Series (n,)
    cov_bl : DataFrame (n,n)
    sigma_post : DataFrame (n,n) ou None

    Notes
    -----
    Quando tau → 0 e não há views (P vazio), o prior domina: μ_BL = π e
    a matriz de incerteza do mean retornada é nula.
    Se não houver views (P vazio) e tau > 0, mantemos σ_post = tau * Σ,
    preservando a incerteza original do prior.
    Quando há views (k > 0) e tau <= 0, retornamos o prior e σ_post = 0.
    Quando add_mean_uncertainty e return_sigma_posterior são ambos falsos,
    evitamos calcular σ_post para reduzir custo computacional.
    """
    if isinstance(cov, pd.DataFrame):
        cov_df = cov.astype(float)
        assets = list(cov_df.columns)
    else:
        cov_arr = np.asarray(cov, dtype=float)
        if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
            raise ValueError("cov must be a square matrix.")
        cov_df = pd.DataFrame(cov_arr)
        assets = list(cov_df.columns)

    if isinstance(pi, pd.Series):
        pi_series = pi.astype(float)
    else:
        pi_array = np.asarray(pi, dtype=float).flatten()
        pi_series = pd.Series(pi_array, index=assets, dtype=float)

    tau_value = float(tau)
    cov_array = cov_df.to_numpy()
    n = cov_df.shape[0]

    P_matrix = np.asarray(P, dtype=float)
    if P_matrix.ndim != 2:
        raise ValueError("P must be a 2D array.")
    k, n_cols = P_matrix.shape
    if n_cols != n:
        raise ValueError("P must have the same number of columns as assets.")

    if k == 0:
        mu_series = pi_series.copy()
        sigma_post_arr = (
            cov_array * tau_value if tau_value > 0 else np.zeros_like(cov_array)
        )
        cov_values = cov_array.copy()
        if add_mean_uncertainty:
            cov_values = cov_values + sigma_post_arr
        cov_bl = pd.DataFrame(cov_values, index=cov_df.index, columns=cov_df.columns)
        if return_sigma_posterior or add_mean_uncertainty:
            sigma_post_df = pd.DataFrame(
                sigma_post_arr,
                index=cov_df.index,
                columns=cov_df.columns,
            )
        else:
            sigma_post_df = None
        return mu_series, cov_bl, sigma_post_df

    if tau_value <= 0:
        mu_series = pi_series.copy()
        sigma_post_arr = np.zeros_like(cov_array)
        cov_values = cov_array.copy()
        if add_mean_uncertainty:
            cov_values = cov_values + sigma_post_arr
        cov_bl = pd.DataFrame(cov_values, index=cov_df.index, columns=cov_df.columns)
        sigma_post_df = (
            pd.DataFrame(
                sigma_post_arr,
                index=cov_df.index,
                columns=cov_df.columns,
            )
            if (return_sigma_posterior or add_mean_uncertainty)
            else None
        )
        return mu_series, cov_bl, sigma_post_df

    Q_vec = np.asarray(Q, dtype=float).reshape(-1)
    if Q_vec.shape[0] != k:
        raise ValueError("Q length must match number of views.")

    Omega_array = np.asarray(Omega, dtype=float)
    if Omega_array.shape != (k, k):
        raise ValueError("Omega must be (k, k).")

    tau_sigma = tau_value * cov_array
    pi_vec = pi_series.to_numpy()

    P_pi = P_matrix @ pi_vec
    A = P_matrix @ tau_sigma @ P_matrix.T + Omega_array
    adjustment_rhs = (Q_vec - P_pi)[:, None]
    middle = _solve_psd(A, adjustment_rhs, jitter=solver_jitter).ravel()
    mu_vec = pi_vec + tau_sigma @ P_matrix.T @ middle

    cov_bl_values = cov_array.copy()
    sigma_post_df: Optional[pd.DataFrame] = None

    if return_sigma_posterior or add_mean_uncertainty:
        inv_tau_sigma = _solve_psd(tau_sigma, np.eye(n), jitter=solver_jitter)
        Omega_inv = _solve_psd(Omega_array, np.eye(k), jitter=solver_jitter)
        posterior_precision = inv_tau_sigma + P_matrix.T @ Omega_inv @ P_matrix
        sigma_post = _solve_psd(posterior_precision, np.eye(n), jitter=solver_jitter)
        sigma_post = 0.5 * (sigma_post + sigma_post.T)
        if add_mean_uncertainty:
            cov_bl_values = cov_bl_values + sigma_post
        sigma_post_df = pd.DataFrame(
            sigma_post,
            index=cov_df.index,
            columns=cov_df.columns,
        )

    mu_bl = pd.Series(mu_vec, index=pi_series.index, dtype=float)
    cov_bl = pd.DataFrame(cov_bl_values, index=cov_df.index, columns=cov_df.columns)

    return mu_bl, cov_bl, sigma_post_df


def _project_psd(mat: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Projeta uma matriz para ser PSD.

    Parameters
    ----------
    mat : ndarray (n,n)
    epsilon : float
        Piso numérico para variâncias.

    Returns
    -------
    mat : ndarray (n,n)
    """
    sym_mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym_mat)
    eigvals_clipped = np.clip(eigvals, epsilon, None)
    projected = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return 0.5 * (projected + projected.T)


def _solve_psd(lhs: np.ndarray, rhs: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """
    Resolve sistemas lineares assumindo matriz PSD via Cholesky com jitter.

    Parameters
    ----------
    lhs : ndarray
        Matriz (n,n) simétrica/PSD.
    rhs : ndarray
        Matriz (n, m) ou vetor (n,).
    jitter : float
        Perturbação adicionada à diagonal para estabilidade numérica.

    Returns
    -------
    ndarray
        Solução do sistema `lhs * X = rhs`.
    """
    matrix = np.asarray(lhs, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("lhs must be a square matrix.")

    rhs_arr = np.asarray(rhs, dtype=float)
    sym_matrix = 0.5 * (matrix + matrix.T)
    jittered = sym_matrix + jitter * np.eye(sym_matrix.shape[0])

    try:
        chol = np.linalg.cholesky(jittered)
        intermediate = np.linalg.solve(chol, rhs_arr)
        solution = np.linalg.solve(chol.T, intermediate)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(jittered, rhs_arr, rcond=None)
    return solution


def black_litterman(
    cov: Union[pd.DataFrame, np.ndarray],
    weights: Optional[Union[pd.Series, Sequence[float]]] = None,
    pi: Optional[Union[pd.Series, Sequence[float]]] = None,
    risk_aversion: Optional[float] = None,
    market_return: Optional[float] = None,
    risk_free: float = 0.0,
    views: Optional[List[Dict[str, Any]]] = None,
    tau: float = 0.025,
    omega_mode: str = "diagonal",
    user_Omega: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    assets: Optional[Sequence[str]] = None,
    ensure_psd: bool = True,
    psd_epsilon: float = 1e-9,
    return_intermediates: bool = False,
    add_mean_uncertainty: bool = False,
    solver_jitter: float = 1e-10,
) -> Dict[str, Any]:
    """
    Orquestra o processo Black-Litterman.

    Retorna dicionário com chaves:
        mu_bl, cov_bl, (opcional) intermediates

    Regras:
    - Se pi ausente, requer weights + (risk_aversion ou market_return).
    - Sem views -> mu_bl = pi.
    - Para tau = 0, o prior domina (μ_BL = π).

    Parameters
    ----------
    add_mean_uncertainty : bool
        Se True, devolve Σ + M onde M é a incerteza do mean.
    solver_jitter : float
        Perturbação numérica aplicada nas resoluções de sistemas lineares.
    """
    if tau < 0:
        raise ValueError("tau must be non-negative.")

    if isinstance(cov, pd.DataFrame):
        cov_df = cov.astype(float).copy()
        inferred_assets = list(cov_df.columns)
    else:
        cov_arr = np.asarray(cov, dtype=float)
        if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
            raise ValueError("cov must be a square matrix.")
        if assets is None:
            inferred_assets = [f"a{i}" for i in range(cov_arr.shape[0])]
        else:
            inferred_assets = list(assets)
            if len(inferred_assets) != cov_arr.shape[0]:
                raise ValueError("Provided assets length must match cov dimension.")
        cov_df = pd.DataFrame(cov_arr, index=inferred_assets, columns=inferred_assets)

    if ensure_psd:
        projected = _project_psd(cov_df.to_numpy(), psd_epsilon)
        cov_df.loc[:, :] = projected

    if assets is None:
        assets_list = inferred_assets
    else:
        assets_list = list(assets)
        if len(assets_list) != cov_df.shape[0]:
            raise ValueError("assets length must match covariance dimension.")
        cov_df = cov_df.loc[assets_list, assets_list]

    pi_series: Optional[pd.Series] = None
    delta_used: Optional[float] = risk_aversion
    weights_series: Optional[pd.Series] = None

    if weights is not None:
        if isinstance(weights, pd.Series):
            weights_series = weights.astype(float).reindex(assets_list)
        else:
            weights_arr = np.asarray(weights, dtype=float).flatten()
            if len(weights_arr) != len(assets_list):
                raise ValueError("weights length must match number of assets.")
            weights_series = pd.Series(weights_arr, index=assets_list, dtype=float)
        if weights_series.isnull().any():
            raise ValueError("weights must include every asset in assets list.")
        weights_series = weights_series / weights_series.sum()

    if pi is not None:
        if isinstance(pi, pd.Series):
            pi_series = pi.astype(float).reindex(assets_list)
        else:
            pi_array = np.asarray(pi, dtype=float).flatten()
            if len(pi_array) != len(assets_list):
                raise ValueError("pi length must match number of assets.")
            pi_series = pd.Series(pi_array, index=assets_list, dtype=float)
        if pi_series.isnull().any():
            raise ValueError("pi must provide values for all assets.")
    else:
        if weights_series is None:
            raise ValueError("Provide either pi or weights to infer the prior.")
        pi_series, delta_used = reverse_optimization(
            weights_series,
            cov_df,
            risk_aversion=risk_aversion,
            market_return=market_return,
            risk_free=risk_free,
        )

    P, Q, confidences = build_projection_matrix(views, assets_list)

    sigma_post: Optional[pd.DataFrame] = None
    omega_matrix: np.ndarray = np.zeros((0, 0))

    if P.shape[0] == 0:
        mu_bl = pi_series.copy()
        cov_values = cov_df.to_numpy()
        if add_mean_uncertainty or return_intermediates:
            if tau > 0:
                sigma_arr = cov_values * tau
            else:
                sigma_arr = np.zeros_like(cov_values)
            sigma_post = pd.DataFrame(
                sigma_arr,
                index=cov_df.index,
                columns=cov_df.columns,
            )
            if add_mean_uncertainty:
                cov_values = cov_values + sigma_arr
        cov_bl = pd.DataFrame(cov_values, index=cov_df.index, columns=cov_df.columns)
    else:
        Omega = view_uncertainty(
            views=views,
            tau=tau,
            cov=cov_df,
            P=P,
            confidences=confidences,
            mode=omega_mode,
            user_Omega=user_Omega,
        )

        mu_bl, cov_bl, sigma_post = posterior_returns(
            pi=pi_series,
            cov=cov_df,
            P=P,
            Q=Q,
            Omega=Omega,
            tau=tau,
            add_mean_uncertainty=add_mean_uncertainty,
            return_sigma_posterior=return_intermediates,
            solver_jitter=solver_jitter,
        )
        omega_matrix = Omega

    if ensure_psd:
        cov_values = _project_psd(cov_bl.to_numpy(), psd_epsilon)
        cov_bl = pd.DataFrame(cov_values, index=cov_bl.index, columns=cov_bl.columns)

    result = {"mu_bl": mu_bl, "cov_bl": cov_bl}

    if return_intermediates:
        if pi_series is None:
            raise RuntimeError("pi_series is not defined.")
        if sigma_post is None:
            sigma_post = pd.DataFrame(
                np.zeros_like(cov_df.to_numpy()),
                index=cov_df.index,
                columns=cov_df.columns,
            )
        result["intermediates"] = {
            "pi": pi_series,
            "P": P,
            "Q": Q,
            "Omega": omega_matrix,
            "tau": tau,
            "delta": delta_used,
            "confidences": confidences,
            "mean_uncertainty": sigma_post,
        }
        if weights_series is not None:
            result["intermediates"]["weights"] = weights_series

    return result

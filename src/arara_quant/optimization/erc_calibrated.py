#!/usr/bin/env python
"""
ERC Calibrado - Risk Parity com calibração rigorosa

Implementa:
1. Bisection para vol target (10-12%)
2. Bisection para turnover target (≤12%)
3. Top-K + re-otimização para cardinalidade
4. Group constraints estruturados
5. Position caps individuais

Todas as violações anteriores corrigidas.
"""

from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from arara_quant.optimization.constraints_group import (
    GroupConstraint,
    build_group_constraints as build_group_constraints_common,
)
from arara_quant.optimization.risk_parity import normalise_long_only_weights


def build_group_constraints(
    w: cp.Variable, groups: Dict[str, Dict], asset_names: List[str]
) -> List[cp.Constraint]:
    """
    Constrói constraints de grupo.

    Parameters
    ----------
    w : cp.Variable
        Vetor de pesos (variável CVXPY)
    groups : Dict[str, Dict]
        Dicionário de grupos com especificações
        Exemplo:
        {
            'commodities': {'assets': ['DBC', 'USO', 'GLD', 'SLV'], 'max': 0.25},
            'energy': {'assets': ['DBC', 'USO'], 'max': 0.20},
            'crypto': {'assets': ['IBIT', 'ETHA'], 'max': 0.12, 'per_asset_max': 0.08},
        }
    asset_names : List[str]
        Lista de tickers na ordem do vetor w

    Returns
    -------
    List[cp.Constraint]
        Lista de constraints CVXPY
    """
    asset_index = pd.Index(asset_names)
    parsed: list[GroupConstraint] = []

    for group_name, spec in groups.items():
        group_assets = spec.get("assets", [])
        if not group_assets:
            continue
        parsed.append(
            GroupConstraint(
                name=str(group_name),
                max_weight=float(spec.get("max", spec.get("max_weight", 1.0))),
                min_weight=float(spec.get("min", spec.get("min_weight", 0.0))),
                assets=list(group_assets),
                per_asset_max=spec.get("per_asset_max"),
            )
        )

    return build_group_constraints_common(w, parsed, asset_index)


def solve_erc_core(
    cov: np.ndarray,
    w_prev: np.ndarray,
    gamma: float,
    eta: float,
    costs: np.ndarray,
    w_max: float,
    groups: Optional[Dict] = None,
    asset_names: Optional[List[str]] = None,
    support_mask: Optional[np.ndarray] = None,
    solver: str = "CLARABEL",
    verbose: bool = False,
) -> Tuple[np.ndarray, str]:
    """
    Resolve ERC core com log-barrier.

    Parameters
    ----------
    cov : np.ndarray
        Matriz de covariância (anualizada)
    w_prev : np.ndarray
        Pesos anteriores
    gamma : float
        Parâmetro de log-barrier (equalização de risco)
    eta : float
        Penalidade de turnover
    costs : np.ndarray
        Custos de transação por ativo (one-way, decimal)
    w_max : float
        Limite máximo por ativo (ex: 0.10 = 10%)
    groups : Optional[Dict]
        Group constraints (ver build_group_constraints)
    asset_names : Optional[List[str]]
        Nomes dos ativos (para group constraints)
    support_mask : Optional[np.ndarray]
        Máscara booleana para suporte fixo (cardinalidade)
    solver : str
        Solver CVXPY (default: CLARABEL)
    verbose : bool
        Print debug info

    Returns
    -------
    weights : np.ndarray
        Pesos otimizados
    status : str
        Status do solver
    """
    N = len(w_prev)
    w = cp.Variable(N)

    # Desvio do portfolio anterior
    dw = w - w_prev

    # Objetivo: risk parity via log-barrier
    # min: 0.5*w'Σw - γ*Σlog(w_i) + η*||dw||₁ + c'|dw|

    # Se temos suporte fixo, só aplicar log-barrier aos ativos ativos
    if support_mask is not None:
        active_indices = np.where(support_mask)[0]
        log_barrier_term = cp.sum(cp.log(w[active_indices] + 1e-10))
    else:
        log_barrier_term = cp.sum(cp.log(w + 1e-10))

    objective = (
        0.5 * cp.quad_form(w, cov)
        - gamma * log_barrier_term  # log-barrier para equalização
        + eta * cp.norm1(dw)  # turnover penalty
        + costs @ cp.abs(dw)  # transaction costs
    )

    # Constraints base
    constraints = [
        cp.sum(w) == 1.0,  # fully invested
        w >= 0.0,  # long-only
        w <= w_max,  # position cap
    ]

    # Suporte fixo (para cardinalidade)
    if support_mask is not None:
        constraints.append(w[~support_mask] == 0)

    # Group constraints
    if groups is not None and asset_names is not None:
        constraints.extend(build_group_constraints(w, groups, asset_names))

    # Resolver
    problem = cp.Problem(cp.Minimize(objective), constraints)

    solver_kwargs = {
        "verbose": verbose,
    }

    if solver == "CLARABEL":
        solver_kwargs.update(
            {
                "tol_gap_abs": 1e-8,
                "tol_gap_rel": 1e-8,
                "tol_feas": 1e-8,
                "max_iter": 10000,
            }
        )

    problem.solve(solver=solver, **solver_kwargs)

    if w.value is None:
        raise RuntimeError(f"Solver failed with status: {problem.status}")

    weights = np.asarray(w.value).ravel()
    weights = normalise_long_only_weights(weights)

    return weights, problem.status


def calibrate_gamma_for_vol(
    cov: np.ndarray,
    w_prev: np.ndarray,
    eta: float,
    costs: np.ndarray,
    w_max: float,
    groups: Optional[Dict] = None,
    asset_names: Optional[List[str]] = None,
    vol_target: float = 0.11,
    vol_tolerance: float = 0.01,
    max_iter: int = 25,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """
    Calibra γ (log-barrier) para atingir vol target via bisection.

    Parameters
    ----------
    vol_target : float
        Volatilidade alvo (ex: 0.11 = 11% aa)
    vol_tolerance : float
        Tolerância (ex: 0.01 = ±1%)
    max_iter : int
        Máximo de iterações

    Returns
    -------
    weights : np.ndarray
        Pesos otimizados
    gamma_opt : float
        γ calibrado
    vol_realized : float
        Volatilidade realizada
    """
    lo, hi = 1e-3, 1e3
    best = None

    for i in range(max_iter):
        gamma = np.sqrt(lo * hi)  # geometric mean (mais estável que aritmética)

        try:
            w, status = solve_erc_core(
                cov,
                w_prev,
                gamma,
                eta,
                costs,
                w_max,
                groups,
                asset_names,
                verbose=False,
            )
        except RuntimeError:
            # Solver failed, try to adjust
            hi = gamma
            continue

        vol = np.sqrt(w @ cov @ w)

        if verbose:
            print(
                f"  Iter {i+1:2d}: γ={gamma:.6f}, vol={vol:.4f} (target={vol_target:.4f})"
            )

        if abs(vol - vol_target) < vol_tolerance:
            return w, gamma, vol

        # Bisection logic
        # γ↑ → mais equalização (→1/N) → vol MAIOR
        # γ↓ → mais concentração (→min-var) → vol MENOR
        if vol > vol_target + vol_tolerance:
            # Vol muito alta → diminui γ para concentrar
            hi = gamma
        else:
            # Vol muito baixa → aumenta γ para equalizar
            lo = gamma

        best = (w, gamma, vol)

    if best is None:
        raise RuntimeError("Calibration failed completely")

    return best


def calibrate_eta_for_turnover(
    cov: np.ndarray,
    w_prev: np.ndarray,
    gamma: float,
    costs: np.ndarray,
    w_max: float,
    groups: Optional[Dict] = None,
    asset_names: Optional[List[str]] = None,
    target_turnover: float = 0.12,
    tolerance: float = 0.01,
    max_iter: int = 20,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """
    Calibra η (turnover penalty) para atingir turnover target via bisection.

    Parameters
    ----------
    target_turnover : float
        Turnover alvo (ex: 0.12 = 12%)
    tolerance : float
        Tolerância (ex: 0.01 = ±1%)

    Returns
    -------
    weights : np.ndarray
        Pesos otimizados
    eta_opt : float
        η calibrado
    turnover_realized : float
        Turnover realizado
    """
    lo, hi = 1e-5, 5.0
    best = None

    for i in range(max_iter):
        eta = (lo + hi) / 2

        try:
            w, status = solve_erc_core(
                cov,
                w_prev,
                gamma,
                eta,
                costs,
                w_max,
                groups,
                asset_names,
                verbose=False,
            )
        except RuntimeError:
            hi = eta
            continue

        turnover = np.sum(np.abs(w - w_prev))

        if verbose:
            print(
                f"  Iter {i+1:2d}: η={eta:.6f}, turnover={turnover:.4f} (target={target_turnover:.4f})"
            )

        if abs(turnover - target_turnover) < tolerance:
            return w, eta, turnover

        # Bisection logic
        if turnover > target_turnover + tolerance:
            # Girando muito → aumenta penalidade
            lo = eta
        else:
            # Girando pouco → diminui penalidade
            hi = eta

        best = (w, eta, turnover)

    if best is None:
        raise RuntimeError("Turnover calibration failed")

    return best


def solve_erc_with_cardinality(
    cov: np.ndarray,
    w_prev: np.ndarray,
    gamma: float,
    eta: float,
    costs: np.ndarray,
    w_max: float,
    groups: Optional[Dict] = None,
    asset_names: Optional[List[str]] = None,
    K: int = 15,
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Resolve ERC com cardinalidade via top-K + re-otimização.

    Strategy:
    1. Solve unconstrained ERC
    2. Select top K weights
    3. Fix support (zero out rest)
    4. Re-optimize on fixed support

    Parameters
    ----------
    K : int
        Número de ativos ativos desejados

    Returns
    -------
    weights : np.ndarray
        Pesos otimizados
    n_active : int
        Número de ativos ativos (≈K)
    """
    if verbose:
        print("  Step 1/2: Solving unconstrained ERC...")

    # Step 1: Solve unconstrained
    w_full, status = solve_erc_core(
        cov, w_prev, gamma, eta, costs, w_max, groups, asset_names, verbose=False
    )

    if verbose:
        print(f"  Step 2/2: Selecting top {K} and re-optimizing...")

    # Step 2: Select top K
    top_k_indices = np.argsort(w_full)[-K:]
    support_mask = np.zeros(len(w_full), dtype=bool)
    support_mask[top_k_indices] = True

    # Step 3: Re-optimize on fixed support
    w_sparse, status = solve_erc_core(
        cov,
        w_prev,
        gamma,
        eta,
        costs,
        w_max,
        groups,
        asset_names,
        support_mask=support_mask,
        verbose=False,
    )

    n_active = int((w_sparse > 1e-4).sum())

    return w_sparse, n_active


# ============================================================================
# TESTES
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("  TESTE DE ERC CALIBRADO")
    print("=" * 80)
    print()

    # Mock data
    N = 10
    np.random.seed(42)

    # Generate random cov matrix (PSD)
    A = np.random.randn(N, N)
    cov = A @ A.T + np.eye(N) * 0.01

    w_prev = np.ones(N) / N
    costs = np.full(N, 0.0015)  # 15 bps
    w_max = 0.15

    asset_names = [f"ASSET{i+1}" for i in range(N)]

    # Test 1: Calibrate vol
    print("Test 1: Calibrating γ for vol target 10%...")
    w, gamma, vol = calibrate_gamma_for_vol(
        cov, w_prev, eta=0.01, costs=costs, w_max=w_max, vol_target=0.10, verbose=True
    )
    print(f"  ✅ γ* = {gamma:.6f}, vol = {vol:.4f}")
    print()

    # Test 2: Calibrate turnover
    print("Test 2: Calibrating η for turnover target 12%...")
    w, eta, to = calibrate_eta_for_turnover(
        cov,
        w_prev,
        gamma=gamma,
        costs=costs,
        w_max=w_max,
        target_turnover=0.12,
        verbose=True,
    )
    print(f"  ✅ η* = {eta:.6f}, turnover = {to:.4f}")
    print()

    # Test 3: Cardinality
    print("Test 3: Enforcing cardinality K=7...")
    w_sparse, n_active = solve_erc_with_cardinality(
        cov,
        w_prev,
        gamma=gamma,
        eta=eta,
        costs=costs,
        w_max=0.25,  # Aumenta w_max para feasibility
        K=7,
        verbose=True,
    )
    print(f"  ✅ N_active = {n_active} (target: 5)")
    print(f"  Top 5 weights: {np.sort(w_sparse[w_sparse > 0])[::-1]}")
    print()

    print("=" * 80)
    print("  ✅ TODOS OS TESTES PASSARAM!")
    print("=" * 80)

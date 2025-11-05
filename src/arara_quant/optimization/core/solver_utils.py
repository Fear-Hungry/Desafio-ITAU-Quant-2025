"""Utility helpers for CVXPy-based solvers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import cvxpy as cp

__all__ = [
    "SolverSummary",
    "select_solver",
    "solve_problem",
]


@dataclass(frozen=True)
class SolverSummary:
    status: str
    solver: str
    value: float
    runtime: float
    primal_residual: float | None
    dual_residual: float | None

    def is_optimal(self) -> bool:
        return self.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


def select_solver(preferred: str | None = None) -> str:
    """Return an installed solver name following a priority order."""

    installed = {solver.upper() for solver in cp.installed_solvers()}
    if preferred:
        candidate = preferred.upper()
        if candidate in installed:
            return candidate
    for solver in ("ECOS", "SCS", "OSQP"):
        if solver in installed:
            return solver
    if installed:
        return sorted(installed)[0]
    raise RuntimeError("No CVXPy solver available. Install ECOS, OSQP or SCS.")


def solve_problem(
    problem: cp.Problem,
    *,
    solver: str | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
) -> SolverSummary:
    """Solve ``problem`` and return a structured summary."""

    chosen_solver = select_solver(solver)
    kwargs = dict(solver_kwargs or {})

    start = time.perf_counter()
    problem.solve(solver=chosen_solver, **kwargs)
    runtime = time.perf_counter() - start

    status = problem.status or cp.UNKNOWN
    primal = getattr(problem, "primal_residual", None)
    dual = getattr(problem, "dual_residual", None)

    return SolverSummary(
        status=status,
        solver=chosen_solver,
        value=float(problem.value) if problem.value is not None else float("nan"),
        runtime=float(runtime),
        primal_residual=float(primal) if primal is not None else None,
        dual_residual=float(dual) if dual is not None else None,
    )

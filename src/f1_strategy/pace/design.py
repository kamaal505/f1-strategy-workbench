"""Deterministic penalised sparse design for conditional robust pace models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor
import warnings
from ..contracts import WorkbenchError

ALPHA = 1e-4


def design_spec(frame: pd.DataFrame, multiplier: float = 1.) -> dict:
    if multiplier not in (0.1, 1., 10.):
        raise ValueError("penalty multiplier must be 0.1, 1 or 10")
    groups = {}
    for name in ("driver_number", "session_id", "driver_session", "layout_id", "spec_era_id"):
        groups[name] = sorted(frame[name].dropna().astype(str).unique().tolist())
    return {"groups": groups, "multiplier": multiplier}


def design_matrix(frame: pd.DataFrame, spec: dict):
    """Return CSR matrix, column names and unscaled ridge penalties.

    Unknown groups have all-zero indicators, giving global partial pooling.
    """
    n = len(frame)
    columns, names, penalties = [], [], []
    multiplier = spec["multiplier"]
    def add(name, values, penalty):
        columns.append(sparse.csr_matrix(np.asarray(values, dtype=np.float64).reshape(n, 1)))
        names.append(name)
        penalties.append(penalty * multiplier)
    compound = frame["compound"].astype(str).to_numpy()
    age = frame["tyre_age_laps"].to_numpy(dtype=np.float64)
    a = np.maximum(age - 2., 0.)
    b = np.maximum(age - 15., 0.)
    for c in ("SOFT", "HARD"):
        add(f"compound:{c}", compound == c, .1)
    for c in ("SOFT", "MEDIUM", "HARD"):
        add(f"age:{c}", (compound == c) * a, .0001)
        add(f"tail:{c}", (compound == c) * b, .0001)
    for name in ("warmup_first", "warmup_second"):
        add(name, frame[name].to_numpy(dtype=np.float64), .1)
    temperature = frame["track_temperature_delta_c"].to_numpy(dtype=np.float64)
    add("temperature_delta", np.nan_to_num(temperature, nan=0.), .1)
    add("temperature_missing", ~np.isfinite(temperature), .1)
    lap = frame["lap_number"].to_numpy(dtype=np.float64) - 1.
    for group, penalty in (("driver_number", 1.), ("session_id", 1.), ("driver_session", 1.), ("layout_id", .1), ("spec_era_id", 20.)):
        vals = frame[group].astype(str).to_numpy()
        for value in spec["groups"][group]:
            mask = vals == value
            add(f"{group}:{value}", mask, penalty)
            if group == "spec_era_id":
                add(f"spec_age:{value}", mask * a, 20.)
                add(f"spec_tail:{value}", mask * b, 20.)
            if group == "layout_id":
                for c in ("SOFT", "MEDIUM", "HARD"):
                    add(f"layout_age:{value}:{c}", mask * (compound == c) * a, 20.)
            if group == "session_id":
                add(f"session_drift:{value}", mask * lap, 1000.)
    return sparse.hstack(columns, format="csr", dtype=np.float64), tuple(names), np.asarray(penalties)


def _initial_parameters(matrix, target):
    """Condition the exact Huber solve with bounded sparse robust ridge starts.

    These are optimizer initial values only. The final sklearn Huber solve
    retains its original objective, penalties and convergence requirement.
    """
    n, p = matrix.shape
    augmented = sparse.hstack((matrix, np.ones((n, 1))), format="csr")
    ridge = sparse.hstack(
        (sparse.eye(p, format="csr") * np.sqrt(ALPHA), sparse.csr_matrix((p, 1))),
        format="csr",
    )
    rhs = np.zeros(n + p)
    rhs[:n] = target
    def solve(design):
        solution = lsqr(design, rhs, atol=1e-10, btol=1e-10, iter_lim=2000)
        if solution[1] == 7 or not np.isfinite(solution[0]).all():
            raise WorkbenchError("fit_not_converged", "Pace initialization did not converge")
        return solution[0]
    initial = solve(sparse.vstack((augmented, ridge), format="csr"))
    sigma = 1.
    for _ in range(3):
        residual = target - augmented @ initial
        sigma = max(float(np.median(np.abs(residual - np.median(residual)))) * .95, 1e-6)
        root_weight = np.sqrt(np.minimum(1., 1.35 * sigma / np.maximum(np.abs(residual), 1e-12)))
        rhs[:n] = target * root_weight
        initial = solve(sparse.vstack(
            (augmented.multiply(root_weight[:, None]), ridge * np.sqrt(sigma)), format="csr"))
    return initial[:-1], float(initial[-1]), sigma


def fit_robust(frame: pd.DataFrame, spec: dict, net_fuel_gain: float):
    matrix, names, penalties = design_matrix(frame, spec)
    scale = np.sqrt(ALPHA / penalties)
    target = frame.lap_time_s.to_numpy(dtype=np.float64) + net_fuel_gain * (frame.lap_number.to_numpy(dtype=np.float64) - 1.)
    # Translation leaves the objective unchanged but avoids optimizing a ~90s
    # intercept from zero alongside sub-second effects and a small robust scale.
    target_origin = float(np.median(target))
    scaled = matrix.multiply(scale).tocsr()
    centered = target - target_origin
    coefficients, intercept, sigma = _initial_parameters(scaled, centered)
    estimator = HuberRegressor(epsilon=1.35, alpha=ALPHA, max_iter=1000, tol=1e-6, fit_intercept=True, warm_start=True)
    estimator.coef_, estimator.intercept_, estimator.scale_ = coefficients, intercept, sigma
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(scaled, centered)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught) or estimator.n_iter_ >= estimator.max_iter:
        raise WorkbenchError("fit_not_converged", "Robust pace regression did not converge")
    return float(estimator.intercept_ + target_origin), np.asarray(estimator.coef_ * scale, dtype=np.float64), names

"""Conditional quantile marginals with the base block rank dependence.

The four HGB quantiles do not identify an entire distribution. We explicitly
assume a piecewise-linear inverse CDF with linear outer tails on [0,1], centered
to zero mean so the separately fitted conditional mean remains the estimand.
This is not bootstrapped ML parameter uncertainty or joint strategy calibration.
"""
from __future__ import annotations
import numpy as np
from ..contracts import WorkbenchError

LEVELS = np.array([0., .025, .10, .90, .975, 1.], dtype=np.float64)


def map_residual_draws(draws, pool, quantiles_s=None, active=None):
    """Map preallocated draw-by-lap block errors; unchanged for base forecasts.

    Common raw block draws retain paired random ranks across plans and settings.
    A false mask entry retains the empirical hierarchical residual distribution
    for outlap/warm-up/insufficient-history fallback.
    """
    if quantiles_s is None:
        return draws
    raw = np.asarray(draws, dtype=np.float64)
    q = np.asarray(quantiles_s, dtype=np.float64)
    if raw.ndim != 2 or q.shape != (raw.shape[1], 4) or not np.isfinite(q).all():
        raise WorkbenchError("numerical_error", "Invalid conditional residual distribution")
    mask = np.ones(raw.shape[1], dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if mask.shape != (raw.shape[1],):
        raise WorkbenchError("numerical_error", "Invalid conditional residual mask")
    if not mask.any():
        return draws
    reference = np.sort(np.asarray(pool, dtype=np.float64).ravel())
    if not len(reference) or not np.isfinite(reference).all():
        raise WorkbenchError("insufficient_evidence", "Residual rank support unavailable")
    output = raw.copy()
    for lap in np.flatnonzero(mask):
        values = np.sort(q[lap])
        knots = np.empty(6, dtype=np.float64)
        knots[1:5] = values
        knots[0] = values[0] - (values[1] - values[0]) / 3.
        knots[5] = values[3] + (values[3] - values[2]) / 3.
        center = float(np.sum((knots[:-1] + knots[1:]) * np.diff(LEVELS) * .5))
        left = np.searchsorted(reference, raw[:, lap], side="left")
        right = np.searchsorted(reference, raw[:, lap], side="right")
        ranks = (left + right + 1.) / (2. * (len(reference) + 1.))
        output[:, lap] = np.interp(ranks, LEVELS, knots) - center
    return output

"""Empirical marginal interval adjustment for held-out event predictions."""
from __future__ import annotations

import math
import numpy as np


def conformal_adjustment(lower, upper, observed, events, coverage: float) -> dict:
    """Compute the finite-sample order statistic or explicitly report sparsity.

    Only finite complete observations count; events identify independent held-out
    races, not laps. This is empirical marginal calibration, not a guarantee of
    exchangeability for correlated prefixes.
    """
    if not 0 < coverage < 1:
        raise ValueError("coverage must be strictly between zero and one")
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    events = np.asarray(events)
    if not (lower.shape == upper.shape == observed.shape == events.shape) or lower.ndim != 1:
        raise ValueError("calibration arrays must be equally sized vectors")
    valid = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(observed)
    if np.any(lower[valid] > upper[valid]):
        raise ValueError("lower interval bound exceeds upper bound")
    count = int(valid.sum())
    event_count = len(set(events[valid].tolist()))
    if count < 30 or event_count < 2:
        return {"status": "uncalibrated", "n_scores": count, "n_events": event_count, "adjustment_s": None}
    scores = np.maximum(np.maximum(lower[valid] - observed[valid], observed[valid] - upper[valid]), 0.)
    index = math.ceil((count + 1) * coverage) - 1
    if index >= count:
        return {"status": "uncalibrated", "n_scores": count, "n_events": event_count, "adjustment_s": None}
    value = float(np.partition(scores, index)[index])
    return {"status": "calibrated", "n_scores": count, "n_events": event_count, "adjustment_s": value}

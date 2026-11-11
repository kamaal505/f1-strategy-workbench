"""Deterministic cluster resampling for robust regression parameter paths."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .design import fit_robust


def bootstrap_fits(events: list[pd.DataFrame], spec: dict, net_fuel_gain: float,
                   *, seed: int, count: int = 200) -> np.ndarray:
    """Draw entire events and then driver-stints within each sampled event.

    The output is float64 [replicate, intercept+coefficients]. The supplied
    design is fixed, so an absent category contributes zero to that draw.
    """
    if count != 200:
        raise ValueError("the serving bundle requires exactly 200 bootstrap replicates")
    rng = np.random.default_rng(seed)
    pools = [list(group for _, group in event.groupby(["driver_number", "stint_id"], sort=True))
             for event in events]
    if any(not pool for pool in pools):
        raise ValueError("every sampled event must have an eligible stint")
    coefficients = []
    for _ in range(count):
        pieces = []
        for event_index in rng.integers(len(events), size=len(events)):
            stints = pools[event_index]
            pieces.extend(stints[index] for index in rng.integers(len(stints), size=len(stints)))
        sampled = pd.concat(pieces, ignore_index=True)
        intercept, values, _ = fit_robust(sampled, spec, net_fuel_gain)
        coefficients.append(np.concatenate(([intercept], values)))
    return np.asarray(coefficients, dtype=np.float64)

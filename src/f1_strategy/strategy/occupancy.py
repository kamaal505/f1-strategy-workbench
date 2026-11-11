"""Conditional absolute pit-lane occupancy, distinct from net race-time loss."""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.stats import truncnorm


def pit_occupancy_projection(state, scenario, n_draws, seed):
    results = {}
    mean = scenario.pit.absolute_lane_duration_s
    sd = scenario.pit.absolute_lane_duration_sd_s
    for number, driver in sorted(state.drivers.items()):
        if not driver.in_pit:
            continue
        if mean is None or driver.pit_entry_ns is None:
            results[number] = {"available": False, "reason": "Absolute lane duration or pit-entry time unavailable"}
            continue
        elapsed = (state.cutoff_ns - driver.pit_entry_ns) / 1e9
        if elapsed < 0:
            results[number] = {"available": False, "reason": "Pit-entry time is after the decision cutoff"}
            continue
        if sd is None or sd == 0:
            if mean <= elapsed:
                results[number] = {"available": False, "reason": "Fixed absolute duration does not exceed elapsed occupancy"}
                continue
            remaining = np.full(n_draws, mean - elapsed)
        else:
            keyed = int.from_bytes(hashlib.sha256(f"{seed}:occupancy:{number}".encode()).digest()[:16], "big")
            rng = np.random.default_rng(keyed)
            samples = truncnorm.rvs((elapsed - mean) / sd, np.inf, loc=mean, scale=sd, size=n_draws, random_state=rng)
            remaining = samples - elapsed
        results[number] = {"available": True, "elapsed_s": elapsed,
                           "remaining_occupancy_mean_s": float(remaining.mean()),
                           "remaining_occupancy_interval80_s": np.quantile(remaining, [.1, .9]).tolist(),
                           "remaining_occupancy_draws_s": remaining.tolist(),
                           "assumption": "Absolute pit-line occupancy conditioned on already elapsed time; not net race-time loss or a stationary service estimate."}
    return results

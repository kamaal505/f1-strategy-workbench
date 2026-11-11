"""Event-held-out empirical calibration of lap and cumulative intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .adaptation import adapt_prefix
from .calibration import conformal_adjustment
from .design import design_matrix
from .training import issued_prefix


def select_residual_pool(compound, era, compound_pools, spec_pools):
    """Choose exactly the same spec-era/compound fallback at fit and serving."""
    return spec_pools.get((era, compound), compound_pools.get(compound))


def issued_parameter_paths(prefix, horizon, spec, fit, bootstrap, net_gain, *, crossing=None):
    """Issue a same-stint forecast using the released history and crossing."""
    last = prefix.iloc[-1] if crossing is None else crossing
    future = last.to_frame().T.iloc[[0] * horizon].copy()
    offsets = np.arange(1, horizon + 1)
    future["lap_number"] = int(last.lap_number) + offsets
    future["tyre_age_laps"] = int(last.tyre_age_laps) + offsets
    # At the crossing only observations released by then may set future temperature.
    future["track_temperature_delta_c"] = prefix.iloc[-1].cutoff_temperature_delta_c
    future["warmup_first"] = 0.
    future["warmup_second"] = (offsets == 1).astype(float) * float(last.warmup_first)
    past_matrix, _, _ = design_matrix(prefix, spec)
    future_matrix, _, _ = design_matrix(future, spec)
    past_correction = net_gain * (prefix.lap_number.to_numpy(dtype=float) - 1.)
    future_correction = net_gain * (future.lap_number.to_numpy(dtype=float) - 1.)
    past_mean = fit[0] + past_matrix @ fit[1:] - past_correction
    past_parameters = (past_matrix @ bootstrap[:, 1:].T).T + bootstrap[:, :1] - past_correction
    _, parameter_anchor, slope, adapted = adapt_prefix(
        prefix, past_mean, past_parameters, current_age=int(last.tyre_age_laps), same_stint=True)
    paths = (future_matrix @ bootstrap[:, 1:].T).T + bootstrap[:, :1] - future_correction
    paths += parameter_anchor[:, None]
    if adapted:
        paths += slope * offsets
    return np.asarray(paths, dtype=np.float64)


def calibrate_events(events, spec, fit, bootstrap, blocks_by_compound, net_gain, *, seed, spec_pools=None):
    observations = {(h, kind, coverage): [[], [], [], []] for h in (1, 3, 5, 10)
                    for kind in ("lap", "cumulative") for coverage in (.8, .95)}
    rng = np.random.default_rng(seed)
    for event in events:
        all_rows = event["features"]
        for _, driver in all_rows.groupby("driver_number", sort=True):
            driver = driver.sort_values("lap_number", kind="stable").reset_index(drop=True)
            for start in range(len(driver) - 1):
                last = driver.iloc[start]
                if (pd.isna(last.lap_end_ns) or pd.isna(last.release_ns)
                        or last.release_ns != last.lap_end_ns or pd.isna(last.tyre_age_laps)):
                    continue
                prefix = issued_prefix(event, int(last.lap_end_ns), last.driver_number)
                if len(prefix) < 3:
                    continue
                for horizon in (1, 3, 5, 10):
                    if start + horizon >= len(driver):
                        continue
                    target = driver.iloc[start + 1:start + 1 + horizon]
                    if target.lap_number.isna().any() or target.tyre_age_laps.isna().any():
                        continue
                    lapnums = target.lap_number.to_numpy(dtype=int)
                    ages = target.tyre_age_laps.to_numpy(dtype=int)
                    offsets = np.arange(1, horizon + 1)
                    if (not target.eligible.all()
                            or np.any(lapnums != int(last.lap_number) + offsets)
                            or np.any(ages != int(last.tyre_age_laps) + offsets)
                            or not (target.stint_id == last.stint_id).all()):
                        continue
                    compound = last.compound
                    blocks = select_residual_pool(compound, last.spec_era_id, blocks_by_compound, spec_pools or {})
                    if blocks is None or not len(blocks):
                        continue
                    paths = issued_parameter_paths(prefix, horizon, spec, fit, bootstrap, net_gain, crossing=last)
                    for i in range(0, horizon, 3):
                        chosen = blocks[rng.integers(len(blocks), size=len(paths))]
                        length = min(3, horizon - i)
                        paths[:, i:i + length] += chosen[:, :length]
                    actual = target.lap_time_s.to_numpy(dtype=float)
                    for kind, sample, observed in (("lap", paths[:, -1], actual[-1]),
                                                   ("cumulative", paths.sum(axis=1), actual.sum())):
                        for coverage in (.8, .95):
                            alpha = (1. - coverage) / 2.
                            bounds = np.quantile(sample, (alpha, 1. - alpha))
                            arrays = observations[(horizon, kind, coverage)]
                            arrays[0].append(bounds[0]); arrays[1].append(bounds[1]); arrays[2].append(observed)
                            arrays[3].append(str(last.session_id))
    output = {}
    for (h, kind, coverage), (lower, upper, observed, ids) in observations.items():
        output[f"{h}:{kind}:{int(round(coverage * 100))}"] = conformal_adjustment(lower, upper, observed, ids, coverage)
    return output

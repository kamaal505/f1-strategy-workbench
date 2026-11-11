"""Leave-one-event-out residual construction with rolling prefix adaptation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .design import design_matrix, design_spec, fit_robust
from .adaptation import adapt_prefix
from .training import issued_prefix


def event_held_out_residuals(events, net_fuel_gain: float, multiplier: float) -> pd.DataFrame:
    """Predict each event from all other fit events and only earlier driver laps.

    Each residual is from a fit that excludes the entire scored event. The
    per-driver intercept uses at most five eligible preceding observations.
    """
    result = []
    for index, event in enumerate(events):
        target = event["features"].query("eligible").copy()
        training = pd.concat([other["features"].query("eligible") for j, other in enumerate(events) if j != index], ignore_index=True)
        spec = design_spec(training, multiplier)
        intercept, coefficients, _ = fit_robust(training, spec, net_fuel_gain)
        target = target.sort_values(["driver_number", "lap_number"], kind="stable").copy()
        matrix, _, _ = design_matrix(target, spec)
        predicted = intercept + matrix @ coefficients - net_fuel_gain * (target.lap_number.to_numpy(dtype=float) - 1.)
        raw = target.lap_time_s.to_numpy(dtype=float) - predicted
        adapted = np.empty(len(target), dtype=np.float64)
        all_rows = event["features"]
        for i, row in enumerate(target.itertuples()):
            previous = all_rows.loc[(all_rows.driver_number == row.driver_number)
                                    & (all_rows.lap_number == row.lap_number - 1)]
            if (previous.empty or pd.isna(previous.iloc[-1].lap_end_ns)
                    or pd.isna(previous.iloc[-1].release_ns)
                    or previous.iloc[-1].release_ns != previous.iloc[-1].lap_end_ns):
                adapted[i] = np.nan
                continue
            crossing = previous.iloc[-1]
            prefix = issued_prefix(event, int(crossing.lap_end_ns), row.driver_number)
            if prefix.empty:
                adapted[i] = raw[i]
                continue
            past_matrix, _, _ = design_matrix(prefix, spec)
            past = intercept + past_matrix @ coefficients - net_fuel_gain * (prefix.lap_number.to_numpy(dtype=float) - 1.)
            same_stint = crossing.source_stint == row.source_stint
            anchor, _, slope, active = adapt_prefix(prefix, past, past[None, :],
                current_age=int(crossing.tyre_age_laps) if pd.notna(crossing.tyre_age_laps) else None,
                same_stint=same_stint)
            correction = slope * (row.tyre_age_laps - crossing.tyre_age_laps) if active and same_stint else 0.
            adapted[i] = raw[i] - anchor - correction
        target["residual_s"] = adapted
        result.append(target)
    return pd.concat(result, ignore_index=True)

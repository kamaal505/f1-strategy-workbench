"""Past-only rolling baseline; it does not support compound-switch advice."""
from __future__ import annotations

import numpy as np

from ..contracts import WorkbenchError


def rolling_forecast(eligible_driver_laps, *, race_laps, assumptions):
    """Forecast from the latest five eligible laps and past-only residuals.

    Returns conditional means, centered residuals and a weak-baseline label.
    The model holds tyre effects constant and cannot compare compounds.
    """
    source = eligible_driver_laps.sort_values("lap_number", kind="stable").tail(5)
    if len(source) < 3:
        raise WorkbenchError("insufficient_evidence", "Rolling pace requires three eligible prefix laps")
    laps = np.asarray(race_laps)
    if laps.ndim != 1 or laps.dtype.kind not in "iu" or np.any(laps <= int(source.lap_number.max())):
        raise WorkbenchError("invalid_input", "Rolling forecast requires future integer lap numbers")
    net = assumptions.fuel_gain_s_per_lap - assumptions.track_drift_s_per_lap
    adjusted = source.lap_time_s.to_numpy(dtype=np.float64) + net * (source.lap_number.to_numpy(dtype=np.float64) - 1.)
    center = float(np.mean(adjusted))
    return {"mean_s": center - net * (laps.astype(np.float64) - 1.),
            "residual_s": adjusted - center, "flags": ("rolling_baseline", "no_compound_switch_advice")}

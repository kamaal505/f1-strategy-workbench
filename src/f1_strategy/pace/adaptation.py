"""Numerical prefix adaptation shared by serving and held-out scoring."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import HuberRegressor


def adapt_prefix(prefix, past_mean, past_parameters, *, current_age, same_stint):
    """Return central and replicate anchors plus a penalised current-stint slope."""
    if not len(prefix):
        return 0., np.zeros(past_parameters.shape[0], dtype=np.float64), 0., False
    observed = prefix.lap_time_s.to_numpy(dtype=np.float64)
    residual = observed - past_mean
    anchor = float(np.median(residual[-5:]))
    parameter_anchor = np.median(observed[-5:] - past_parameters[:, -5:], axis=1)
    slope, adapted = 0., False
    if same_stint and current_age is not None and "source_stint" in prefix:
        current_mask = (prefix.source_stint == prefix.source_stint.iloc[-1]).to_numpy()
        current = prefix.loc[current_mask]
        if len(current) >= 8 and current.tyre_age_laps.max() - current.tyre_age_laps.min() >= 6:
            x = current.tyre_age_laps.to_numpy(dtype=np.float64) - current_age
            y = observed[current_mask] - past_mean[current_mask] - anchor
            fitted = HuberRegressor(alpha=50., epsilon=1.35, max_iter=1000, tol=1e-6).fit(x.reshape(-1, 1), y)
            if fitted.n_iter_ < fitted.max_iter:
                slope, adapted = float(fitted.coef_[0]), True
    return anchor, parameter_anchor, slope, adapted

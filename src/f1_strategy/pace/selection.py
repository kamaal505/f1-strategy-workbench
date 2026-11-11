"""Choose pooling strength using only earlier development events."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .design import design_matrix, design_spec, fit_robust


def select_penalty(training_events, tuning_events, *, net_gain):
    """Score dated tuning events against strictly earlier training events.

    Events are (max-used UTC ISO timestamp, eligible frame) pairs. When fewer
    than two chronological folds are possible, preserve the fixed multiplier 1.
    """
    folds = []
    for target_time, target in sorted(tuning_events, key=lambda item: item[0]):
        earlier = [event for event_time, event in training_events if event_time < target_time and not event.empty]
        if len(earlier) >= 4 and not target.empty:
            folds.append((pd.concat(earlier, ignore_index=True), target))
    if len(folds) < 2:
        return 1., {"status": "default_insufficient_folds", "folds": len(folds)}
    scores = {}
    for multiplier in (.1, 1., 10.):
        errors = []
        for source, target in folds:
            spec = design_spec(source, multiplier)
            intercept, coefficients, _ = fit_robust(source, spec, net_gain)
            matrix, _, _ = design_matrix(target, spec)
            pred = intercept + matrix @ coefficients - net_gain * (target.lap_number.to_numpy(dtype=float) - 1.)
            errors.append(float(np.mean(np.abs(target.lap_time_s.to_numpy(dtype=float) - pred))))
        scores[str(multiplier)] = float(np.mean(errors))
    selected = min((.1, 1., 10.), key=lambda multiplier: (scores[str(multiplier)], -multiplier))
    return selected, {"status": "selected", "folds": len(folds), "mae_s": scores}

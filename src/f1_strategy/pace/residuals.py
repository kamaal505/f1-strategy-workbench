"""Within-stint residual paths and their explicitly disclosed pooling level."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_residual_blocks(rows: pd.DataFrame, *, spec_era_id: str, compound: str,
                          min_residuals: int = 30) -> tuple[np.ndarray, str]:
    """Pool first by physical spec, then relative compound, then all dry tyres.

    Rows must be out-of-event predictions adapted only on prior laps; this
    function cannot establish that provenance by itself. Windows never bridge
    an event, driver, stint, or a missing lap. Each selected pool is centered.
    """
    for label, selected in (
        ("spec_era", rows.loc[rows.spec_era_id == spec_era_id]),
        ("compound", rows.loc[rows.compound == compound]),
        ("all_dry", rows),
    ):
        selected = selected.loc[np.isfinite(selected.residual_s)]
        if len(selected) < min_residuals:
            continue
        center = float(selected.residual_s.mean())
        blocks = []
        for _, group in selected.groupby(["session_id", "driver_number", "stint_id"], sort=True, dropna=False):
            group = group.sort_values("lap_number", kind="stable")
            lap = group.lap_number.to_numpy(dtype=int)
            residual = group.residual_s.to_numpy(dtype=np.float64) - center
            for i in range(len(group) - 2):
                if lap[i + 1] == lap[i] + 1 and lap[i + 2] == lap[i] + 2:
                    blocks.append(residual[i:i + 3])
        if blocks:
            return np.asarray(blocks, dtype=np.float64), label
    return np.empty((0, 3), dtype=np.float64), "insufficient_residuals"

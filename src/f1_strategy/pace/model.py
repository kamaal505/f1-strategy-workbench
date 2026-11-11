"""In-memory view of a verified, numeric-only hierarchical model bundle."""
from __future__ import annotations

from datetime import datetime
import numpy as np

from ..contracts import WorkbenchError
from .design import design_matrix


class HierarchicalPaceModel:
    def __init__(self, model_id, metadata, arrays):
        self.model_id = model_id
        self.metadata = metadata
        self.arrays = arrays
        self.data_cutoff_utc = datetime.fromisoformat(metadata["data_cutoff_utc"])
        self.spec = metadata["design"]
        self.feature_names = tuple(metadata["feature_names"])
        self.support = metadata["support"]
        self.lineage = metadata["lineage"]
        self.n_parameter_replicates = 200
        self.assumption_grid = tuple(tuple(value) for value in metadata["assumption_grid"])

    def _variant(self, assumptions):
        key = (assumptions.fuel_gain_s_per_lap, assumptions.track_drift_s_per_lap)
        for index, pair in enumerate(self.assumption_grid):
            if all(abs(a - b) < 1e-12 for a, b in zip(pair, key)):
                return self.metadata["variant_indices"][index]
        raise WorkbenchError("missing_model_variant", "Pace model does not contain the requested fuel/track assumptions")

    def predict_laps(self, features, *, assumptions, replicate_index=None):
        variant = self._variant(assumptions)
        matrix, names, _ = design_matrix(features, self.spec)
        if names != self.feature_names:
            raise WorkbenchError("corrupt_artifact", "Pace design does not match model coefficients")
        if replicate_index is None:
            coefficients = self.arrays[f"fit_{variant}"]
        else:
            if not isinstance(replicate_index, int) or not 0 <= replicate_index < 200:
                raise WorkbenchError("invalid_input", "Invalid parameter replicate index")
            coefficients = self.arrays[f"bootstrap_{variant}"][replicate_index]
        correction = (assumptions.fuel_gain_s_per_lap - assumptions.track_drift_s_per_lap) * (features.lap_number.to_numpy(dtype=float) - 1.)
        return np.asarray(coefficients[0] + matrix @ coefficients[1:] - correction, dtype=np.float64)

    def predict_parameter_laps(self, features, *, assumptions):
        variant = self._variant(assumptions)
        matrix, names, _ = design_matrix(features, self.spec)
        if names != self.feature_names:
            raise WorkbenchError("corrupt_artifact", "Pace design does not match model coefficients")
        coefficients = self.arrays[f"bootstrap_{variant}"]
        correction = (assumptions.fuel_gain_s_per_lap - assumptions.track_drift_s_per_lap) * (features.lap_number.to_numpy(dtype=float) - 1.)
        return np.asarray((matrix @ coefficients[:, 1:].T).T + coefficients[:, :1] - correction, dtype=np.float64)

    def residual_blocks(self, compound, *, assumptions):
        variant = self._variant(assumptions)
        return self.arrays.get(f"residual_{variant}_{compound}", np.empty((0, 3), dtype=np.float64))

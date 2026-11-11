"""Persisted identity and provenance for the weak prefix-only rolling model."""
from datetime import datetime

from ..contracts import WorkbenchError


class RollingPaceModel:
    def __init__(self, model_id, metadata):
        self.model_id = model_id
        self.metadata = metadata
        self.data_cutoff_utc = datetime.fromisoformat(metadata["data_cutoff_utc"])
        self.feature_names = ()
        self.support = {}
        self.lineage = metadata["lineage"]
        self.n_parameter_replicates = 1
        self.assumption_grid = tuple(tuple(x) for x in metadata["assumption_grid"])

    def predict_laps(self, features, *, assumptions, replicate_index=None):
        raise WorkbenchError("insufficient_evidence", "Rolling predictions require a released driver prefix")

    def residual_blocks(self, compound, *, assumptions):
        raise WorkbenchError("insufficient_evidence", "Rolling residuals require a released driver prefix")

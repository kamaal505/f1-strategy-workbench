"""Shared validated controls and immutable runtime interfaces.

Seconds are float64; event times are stream-relative int64 nanoseconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, Annotated
import re

import numpy as np
import pandas as pd
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPLAY_NOTICE = "Event-time reconstructed replay; historical corrections and original arrival times are not fully observable."
ID_PATTERN = r"^[0-9a-f]{64}$"
ArtifactId = Annotated[str, Field(pattern=ID_PATTERN)]
DryCompound = Literal["SOFT", "MEDIUM", "HARD"]

class WorkbenchError(Exception):
    def __init__(self, code: str, detail: str):
        self.code, self.detail = code, detail
        super().__init__(detail)

class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

class IngestRequest(FrozenModel):
    year: int = Field(ge=2022, le=2024)
    round_number: int = Field(ge=1, le=24)
    session: Literal["R"] = "R"
    telemetry: bool = False

class SnapshotRef(FrozenModel):
    snapshot_id: ArtifactId
    session_id: str
    path: Path

class Availability(FrozenModel):
    status: Literal["available", "empty", "unavailable"]
    reason: str | None = None
    rows: int = Field(default=0, ge=0)

class FileEntry(FrozenModel):
    path: str
    sha256: ArtifactId
    rows: int = Field(ge=0)

class SnapshotManifest(FrozenModel):
    schema_version: Literal[1] = 1
    snapshot_id: ArtifactId
    session_id: str
    synthetic: bool
    year: int = Field(ge=2022, le=2024)
    round_number: int = Field(ge=1, le=24)
    session: Literal["R"] = "R"
    stream_origin_utc: datetime
    race_start_ns: int = Field(ge=0)
    scheduled_laps: int = Field(gt=0)
    entries: tuple[dict[str, Any], ...]
    event_metadata: dict[str, Any]
    event_metadata_digest: ArtifactId
    acquired_utc: datetime
    source_ids: tuple[str, ...]
    source_endpoints: tuple[str, ...] = ()
    versions: dict[str, str]
    lock_digest: ArtifactId
    load_options: dict[str, bool]
    availability: dict[str, Availability]
    quality: dict[str, Any] = Field(default_factory=dict)
    files: tuple[FileEntry, ...]
    caveats: tuple[str, ...] = (REPLAY_NOTICE, "Original receipt timestamps unavailable.")

    @field_validator("stream_origin_utc", "acquired_utc")
    @classmethod
    def aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("timezone-aware UTC required")
        return v.astimezone(timezone.utc)

@dataclass(frozen=True)
class Snapshot:
    ref: SnapshotRef
    manifest: SnapshotManifest
    tables: dict[str, pa.Table]

    def table(self, name: str) -> pa.Table:
        try:
            return self.tables[name]
        except KeyError:
            raise WorkbenchError("missing_stream", f"Stream {name} unavailable") from None

class PaceAssumptions(FrozenModel):
    fuel_gain_s_per_lap: float = Field(default=0.06, ge=0, le=1)
    track_drift_s_per_lap: float = Field(default=0.0, ge=-1, le=1)
    track_temperature_c: float | None = None

DEFAULT_GRID = tuple(PaceAssumptions(fuel_gain_s_per_lap=g, track_drift_s_per_lap=d)
                     for g in (0.04, 0.06, 0.08) for d in (-0.02, 0.0, 0.02))

class TrainingSpec(FrozenModel):
    training_snapshot_ids: tuple[ArtifactId, ...]
    calibration_snapshot_ids: tuple[ArtifactId, ...]
    data_cutoff_utc: datetime
    model_kind: Literal["rolling", "hierarchical", "hgb"]
    seed: int = 2026
    hyperparameters: dict[str, float | int] = Field(default_factory=dict)
    assumption_grid: tuple[PaceAssumptions, ...] = DEFAULT_GRID
    tuning_snapshot_ids: tuple[ArtifactId, ...] = ()
    selection_snapshot_ids: tuple[ArtifactId, ...] = ()

    @model_validator(mode="after")
    def valid_training(self):
        if self.data_cutoff_utc.tzinfo is None:
            raise ValueError("timezone-aware data cutoff required")
        groups = (self.training_snapshot_ids, self.calibration_snapshot_ids)
        if not self.training_snapshot_ids or any(len(set(x)) != len(x) for x in groups):
            raise ValueError("nonempty unique training IDs required")
        if set(groups[0]) & set(groups[1]):
            raise ValueError("training and calibration events must be disjoint")
        allowed = {"penalty_multiplier", "n_bootstraps", "epsilon", "max_iter", "max_leaf_nodes", "min_samples_leaf", "learning_rate", "l2_regularization"}
        if set(self.hyperparameters) - allowed:
            raise ValueError("undocumented hyperparameter")
        required = {(p.fuel_gain_s_per_lap, p.track_drift_s_per_lap) for p in DEFAULT_GRID}
        actual = {(p.fuel_gain_s_per_lap, p.track_drift_s_per_lap) for p in self.assumption_grid}
        if not required <= actual:
            raise ValueError("mandatory nine-setting assumption grid required")
        return self

class ModelRef(FrozenModel):
    model_id: ArtifactId
    path: Path
    model_kind: Literal["rolling", "hierarchical", "hgb"]

class TyreSet(FrozenModel):
    set_id: str = Field(min_length=1, max_length=100)
    compound: DryCompound
    age_laps: int = Field(ge=0)
    life_limit_laps: int | None = Field(default=None, gt=0)
    provenance: Literal["assumed", "operator_documented"]

    @model_validator(mode="after")
    def within_life(self):
        if self.life_limit_laps is not None and self.age_laps > self.life_limit_laps:
            raise ValueError("set age exceeds life limit")
        return self

class PitAssumptions(FrozenModel):
    net_transit_loss_s: float = Field(default=19.0, ge=0)
    net_transit_sd_s: float = Field(default=2.0, ge=0)
    service_s: float = Field(default=3.0, ge=1)
    service_sd_s: float = Field(default=0.5, ge=0)
    outlap_warmup_s: float = Field(default=1.0, ge=0)
    outlap_warmup_sd_s: float = Field(default=0.5, ge=0)
    absolute_lane_duration_s: float | None = Field(default=None, gt=0)
    absolute_lane_duration_sd_s: float | None = Field(default=None, ge=0)
    sc_net_transit_factor: float = Field(default=0.55, ge=0, le=1)
    vsc_net_transit_factor: float = Field(default=0.65, ge=0, le=1)
    provenance: str = "assumed"

class NeutralisationSpec(FrozenModel):
    kind: Literal["SC", "VSC"]
    start_after_leader_laps: int = Field(ge=0)
    duration_laps: int = Field(gt=0)
    pit_entry_open: bool = True
    pit_exit_open: bool = True

class RuleOverride(FrozenModel):
    min_dry_specs: int = Field(default=2, ge=1, le=3)
    mandatory_race_compounds: tuple[DryCompound, ...] = ("MEDIUM", "HARD")
    max_set_life_laps: int | None = Field(default=None, gt=0)
    pit_entry_open: bool = True
    pit_exit_open: bool = True
    source_reference: str = Field(min_length=1)
    provenance: Literal["assumed", "operator_documented"]

class ScenarioSpec(FrozenModel):
    schema_version: Literal[1] = 1
    name: str
    inventory_by_driver: dict[str, tuple[TyreSet, ...]]
    inventory_provenance: str
    pace: PaceAssumptions = PaceAssumptions()
    pit: PitAssumptions = PitAssumptions()
    neutralisations: tuple[NeutralisationSpec, ...] = ()
    rival_policy: Literal["age_threshold"] = "age_threshold"
    max_additional_stops: int = Field(default=3, ge=0, le=3)
    overtaking_margin_s: float = Field(default=0.6, ge=0)
    following_headway_s: float = Field(default=0.3, ge=0)
    traffic_gap_s: float = Field(default=1.5, ge=0)
    traffic_penalty_s: float = Field(default=0.4, ge=0)
    dry_condition_assumption: Literal["none", "dry"] = "none"
    rules_override: RuleOverride | None = None

    @model_validator(mode="after")
    def unique_sets(self):
        ids = [s.set_id for sets in self.inventory_by_driver.values() for s in sets]
        if len(ids) != len(set(ids)):
            raise ValueError("set IDs must be globally unique")
        return self

class PlannedStop(FrozenModel):
    pit_lap: int = Field(gt=0)
    new_set_id: str

class StrategyPlan(FrozenModel):
    plan_id: str
    branch: Literal["pit_now", "stay_out"]
    stops: tuple[PlannedStop, ...]

@dataclass(frozen=True)
class DriverState:
    driver_number: str
    completed_laps: int
    last_crossing_ns: int
    progress_laps: float
    compound: str | None
    tyre_age_laps: int | None
    mounted_set_id: str
    used_compounds: frozenset[str]
    in_pit: bool = False
    pit_entry_ns: int | None = None
    gap_s: float | None = None
    stale: bool = False
    typical_lap_s: float | None = None
    flags: tuple[str, ...] = ()

@dataclass(frozen=True)
class RaceState:
    snapshot_id: str
    session_id: str
    driver_number: str
    completed_lap: int
    cutoff_ns: int
    max_feature_release_ns: int
    scheduled_laps: int
    race_start_ns: int
    stream_origin_utc: datetime
    year: int
    round_number: int
    synthetic: bool
    laps: pd.DataFrame
    weather: pd.DataFrame
    track_status: pd.DataFrame
    session_status: pd.DataFrame
    messages: pd.DataFrame
    pit_events: pd.DataFrame
    drivers: dict[str, DriverState]
    flags: tuple[str, ...] = ()
    event_metadata: dict[str, Any] = field(default_factory=dict)
    source_lineage: tuple[dict[str, Any], ...] = ()

    @property
    def cutoff_utc(self) -> datetime:
        from datetime import timedelta
        return self.stream_origin_utc + timedelta(microseconds=self.cutoff_ns / 1000)

    @property
    def race_start_utc(self) -> datetime:
        from datetime import timedelta
        return self.stream_origin_utc + timedelta(microseconds=self.race_start_ns / 1000)

@dataclass(frozen=True)
class StintForecast:
    driver_number: str
    compound: str
    race_laps: np.ndarray
    tyre_ages: np.ndarray
    mean_s: np.ndarray
    parameter_mean_s: np.ndarray
    residual_blocks_s: np.ndarray
    interval80_s: np.ndarray
    interval95_s: np.ndarray
    cumulative_intervals: dict[str, Any]
    support: dict[str, Any]
    assumptions: PaceAssumptions
    flags: tuple[str, ...]
    cutoff_ns: int
    # Optional challenger conditional residual marginals. Base three-lap blocks
    # supply the serial rank dependence, not a second additive noise source.
    residual_quantiles_s: np.ndarray | None = None  # horizon x [2.5,10,90,97.5]%
    ml_residual_mask: np.ndarray | None = None  # horizon; false retains base residuals

class PaceModel(Protocol):
    model_id: str
    data_cutoff_utc: datetime
    feature_names: tuple[str, ...]
    support: dict
    n_parameter_replicates: int
    assumption_grid: tuple[tuple[float, float], ...]

    def predict_laps(self, features: pd.DataFrame, *, assumptions: PaceAssumptions,
                     replicate_index: int | None = None) -> np.ndarray: ...
    def residual_blocks(self, compound: str, *, assumptions: PaceAssumptions) -> np.ndarray: ...

class DecisionResult(FrozenModel):
    schema_version: Literal[1] = 1
    analysis_id: ArtifactId
    snapshot_id: ArtifactId
    session_id: str
    driver_number: str
    completed_lap: int
    cutoff_ns: int
    replay_standard: Literal["event_time_reconstructed"] = "event_time_reconstructed"
    model_id: ArtifactId
    model_data_cutoff_utc: datetime
    scenario: ScenarioSpec
    n_draws: int = Field(ge=100, le=5000)
    seed: int
    status: Literal["ok", "insufficient_evidence", "unsupported", "infeasible"]
    recommendation: Literal["pit_now", "stay_out", "no_clear_advantage", "unavailable"]
    candidate_summaries: tuple[dict[str, Any], ...] = ()
    selected_plan_ids: dict[str, str] = Field(default_factory=dict)
    delta_stay_minus_pit_s: float | None = None
    prob_pit_faster: float | None = Field(default=None, ge=0, le=1)
    intervals: dict[str, Any] = Field(default_factory=dict)
    pit_exit_projection: dict[str, Any] = Field(default_factory=dict)
    position_distribution: dict[str, Any] = Field(default_factory=dict)
    sensitivity: tuple[dict[str, Any], ...] = ()
    quality_flags: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    evidence: dict[str, Any] = Field(default_factory=dict)
    comparison_scope: str
    identity_inputs: dict[str, Any]
    prefix_state: dict[str, Any]
    forecast_series: dict[str, Any] = Field(default_factory=dict)
    simulation_series: dict[str, Any] = Field(default_factory=dict)
    source_lineage: tuple[dict[str, Any], ...]
    payload_checksum: ArtifactId

class ReportRef(FrozenModel):
    analysis_id: ArtifactId
    path: Path
    analysis_path: Path
    manifest_path: Path

class EvaluationSpec(FrozenModel):
    schema_version: Literal[1] = 1
    corpus_asset: str = "evaluation/corpus.json"
    partition: Literal["development", "holdout"]
    decision_lap_start: int = Field(default=10, gt=0)
    decision_lap_step: int = Field(default=5, gt=0)
    end_margin_laps: int = Field(default=5, ge=0)
    horizons: tuple[int, ...] = (1, 3, 5, 10)
    seed: int = 2026
    frozen_settings_manifest: str | None = None

    @model_validator(mode="after")
    def holdout_frozen(self):
        if any(h not in (1, 3, 5, 10) for h in self.horizons):
            raise ValueError("supported horizons are 1,3,5,10")
        if self.partition == "holdout" and not self.frozen_settings_manifest:
            raise ValueError("holdout requires pre-existing frozen settings manifest")
        return self

class PublicUseAuthorization(FrozenModel):
    source_ids: tuple[str, ...]
    session_ids: tuple[str, ...]
    scopes: tuple[Literal["source_access", "snapshot_retention", "public_display", "derived_report_download", "model_server_use", "public_asset_distribution"], ...]
    expires_utc: datetime | None
    permission_reference_id: str = Field(min_length=1)
    confirmed_utc: datetime

    @field_validator("expires_utc", "confirmed_utc")
    @classmethod
    def aware_auth(cls, v):
        if v is not None and v.tzinfo is None:
            raise ValueError("timezone-aware UTC required")
        return v

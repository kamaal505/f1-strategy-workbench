"""Admission and provenance of immutable earlier-event training inputs."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from ..contracts import TrainingSpec, WorkbenchError
from ..data.snapshots import open_snapshot
from .features import eligible_laps


def admitted_events(training: TrainingSpec, *, root: Path):
    """Open every explicitly listed event and reject future or overlapping roles.

    The latest released observation, not the acquisition date or scheduled end,
    bounds its information time. A source with no timed lap cannot be admitted.
    """
    roles = {"fit": training.training_snapshot_ids,
             "calibration": training.calibration_snapshot_ids,
             "tuning": training.tuning_snapshot_ids,
             "selection": training.selection_snapshot_ids}
    fit = set(roles["fit"])
    calibration = set(roles["calibration"])
    if calibration & (fit | set(roles["tuning"]) | set(roles["selection"])):
        raise WorkbenchError("invalid_training", "Calibration events must be disjoint from fit and selection events")
    result = {}
    session_snapshots = {}
    for role, identifiers in roles.items():
        for identifier in identifiers:
            if identifier in result:
                result[identifier]["roles"].append(role)
                continue
            snapshot = open_snapshot(root, identifier)
            manifest = snapshot.manifest
            previous = session_snapshots.setdefault(manifest.session_id, identifier)
            if previous != identifier:
                raise WorkbenchError("invalid_training", "Multiple snapshots of the same session cannot be used across training roles")
            laps = snapshot.table("laps").to_pandas()
            end = laps.release_ns.dropna()
            if end.empty:
                raise WorkbenchError("insufficient_evidence", "Training event has no timed laps")
            # All canonical temporal streams may contribute to fitting or correction.
            max_ns = int(end.max())
            for name in ("pit_events", "weather", "track_status", "session_status", "messages"):
                table = snapshot.tables.get(name)
                if table is not None and table.num_rows:
                    values = table.column("release_ns").to_pandas().dropna()
                    if not values.empty:
                        max_ns = max(max_ns, int(values.max()))
            end_utc = manifest.stream_origin_utc + timedelta(microseconds=max_ns / 1000)
            if end_utc >= training.data_cutoff_utc:
                raise WorkbenchError("future_artifact", "A training, tuning or calibration source is not earlier than the cutoff")
            def frame(name):
                table = snapshot.tables.get(name)
                if table is None:
                    from ..data.schemas import SCHEMAS
                    return pd.DataFrame(columns=SCHEMAS[name].names)
                return table.to_pandas()
            features = eligible_laps(laps, frame("pit_events"), frame("track_status"), frame("weather"),
                                     cutoff_ns=max_ns, event_metadata=manifest.event_metadata,
                                     session_id=manifest.session_id, historical=True)
            lineage = {"snapshot_id": identifier, "session_id": manifest.session_id,
                       "source_ids": list(manifest.source_ids), "synthetic": manifest.synthetic,
                       "max_used_utc": end_utc.isoformat()}
            features["driver_session"] = manifest.session_id + ":" + features.driver_number.astype(str)
            first_weather = frame("weather")
            first_weather = first_weather.dropna(subset=["release_ns"]).sort_values("release_ns")
            valid_temperature = first_weather.track_temp_c.dropna()
            reference_temperature = float(valid_temperature.iloc[0]) if len(valid_temperature) else float("nan")
            features["track_temperature_delta_c"] = features.track_temperature_c - reference_temperature
            observations = first_weather.dropna(subset=["release_ns", "track_temp_c"])
            times = observations.release_ns.to_numpy(dtype="int64")
            temperatures = observations.track_temp_c.to_numpy(dtype=float)
            import numpy as np
            cutoff_times = features.lap_end_ns.to_numpy(dtype=float, na_value=np.nan)
            indices = np.searchsorted(times, np.nan_to_num(cutoff_times, nan=-1), side="right") - 1
            features["cutoff_temperature_delta_c"] = [
                float(temperatures[i] - reference_temperature) if i >= 0 else float("nan") for i in indices
            ]
            features["stint_id"] = features.driver_number.astype(str) + ":" + features.source_stint.astype(str)
            result[identifier] = {"snapshot": snapshot, "features": features, "lineage": lineage, "roles": [role]}
    # Held-out calibration is chronologically after every fitted or selected
    # event, not merely disjoint by identifier.
    earlier = [item for item in result.values() if set(item["roles"]) & {"fit", "tuning", "selection"}]
    for item in result.values():
        if "calibration" not in item["roles"]:
            continue
        manifest = item["snapshot"].manifest
        start = manifest.stream_origin_utc + timedelta(microseconds=manifest.race_start_ns / 1000)
        for source in earlier:
            from datetime import datetime
            if datetime.fromisoformat(source["lineage"]["max_used_utc"]) >= start:
                raise WorkbenchError("future_artifact", "Calibration event does not follow all fitting and selection events")
    return result


def issued_prefix(event, cutoff_ns, driver_number):
    """Reconstruct the eligible driver history as released at a crossing."""
    from ..replay import _deletions
    cache = event.setdefault("_issued_prefixes", {})
    key = (int(cutoff_ns), str(driver_number))
    if key in cache:
        return cache[key]
    snapshot = event["snapshot"]
    def frame(name):
        table = snapshot.tables.get(name)
        if table is None:
            from ..data.schemas import SCHEMAS
            return pd.DataFrame(columns=SCHEMAS[name].names)
        return table.to_pandas()
    messages = frame("messages")
    messages = messages.loc[messages.release_ns.notna() & (messages.release_ns <= cutoff_ns)].sort_values("release_ns", kind="stable")
    laps = frame("laps")
    laps = laps.loc[laps.lap_end_ns.notna() & (laps.lap_end_ns <= cutoff_ns)]
    laps, _ = _deletions(laps, messages)
    manifest = snapshot.manifest
    weather = frame("weather")
    prefix = eligible_laps(laps, frame("pit_events"), frame("track_status"), weather,
                           cutoff_ns=cutoff_ns, event_metadata=manifest.event_metadata,
                           session_id=manifest.session_id)
    prefix = prefix.loc[(prefix.driver_number.astype(str) == str(driver_number)) & prefix.eligible].copy()
    if prefix.empty:
        cache[key] = prefix
        return prefix
    weather = weather.loc[weather.release_ns.notna() & (weather.release_ns <= cutoff_ns)].sort_values("release_ns", kind="stable")
    temperatures = weather.track_temp_c.dropna()
    reference = float(temperatures.iloc[0]) if len(temperatures) else float("nan")
    current = float(temperatures.iloc[-1]) if len(temperatures) else float("nan")
    prefix["driver_session"] = manifest.session_id + ":" + prefix.driver_number.astype(str)
    prefix["track_temperature_delta_c"] = prefix.track_temperature_c - reference
    prefix["cutoff_temperature_delta_c"] = current - reference
    prefix["stint_id"] = prefix.driver_number.astype(str) + ":" + prefix.source_stint.astype(str)
    cache[key] = prefix
    return prefix

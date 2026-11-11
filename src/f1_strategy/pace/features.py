"""Past-only construction of eligible lap observations for pace estimation.

The input may be a complete earlier event or a replay prefix.  Times are stream
nanoseconds, and all as-of joins use observations available at lap start.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DRY = frozenset(("SOFT", "MEDIUM", "HARD"))


def eligible_laps(laps: pd.DataFrame, pit_events: pd.DataFrame,
                  track_status: pd.DataFrame, weather: pd.DataFrame, *,
                  cutoff_ns: int, event_metadata: dict, session_id: str,
                  historical: bool = False) -> pd.DataFrame:
    """Return clean timed laps with start-known covariates and explicit exclusions.

    Slow laps are retained. A missing status or ambiguous track interval is not
    assumed green. Final deletion flags are usable only for prior completed races.
    """
    df = laps.loc[laps.release_ns.notna() & (laps.release_ns <= cutoff_ns)].copy()
    df = df.sort_values(["driver_number", "lap_number", "release_ns"], kind="stable")
    if df.empty:
        return df
    pit = pit_events.loc[pit_events.release_ns.notna() & (pit_events.release_ns <= cutoff_ns)]
    pit_laps = set(zip(pit.driver_number.astype(str), pit.source_lap_number))
    exits = {(str(row.driver_number), int(row.source_lap_number)) for row in pit.itertuples()
             if row.kind == "exit" and pd.notna(row.source_lap_number)}
    status = track_status.loc[track_status.release_ns.notna() & (track_status.release_ns <= cutoff_ns)].sort_values("release_ns")
    w = weather.loc[weather.release_ns.notna() & (weather.release_ns <= cutoff_ns)].sort_values("release_ns")
    status_times = status.release_ns.to_numpy(dtype=np.int64)
    status_values = status.status.to_numpy()
    weather_times = w.release_ns.to_numpy(dtype=np.int64)
    weather_values = w.track_temp_c.to_numpy(dtype=float, na_value=np.nan)
    rainfall = w.rainfall.to_numpy() if "rainfall" in w else np.full(len(w), None)
    reasons, temps, warm1, warm2, gaps = [], [], [], [], []
    starts = df.lap_start_ns.to_numpy(dtype=float, na_value=np.nan)
    # A gap is measured only from crossings already observable at lap start.
    crossing = df.loc[df.lap_end_ns.notna() & df.release_ns.notna(), ["driver_number", "lap_end_ns", "release_ns"]]
    crossing_times = crossing.lap_end_ns.to_numpy(dtype=np.int64)
    crossing_releases = crossing.release_ns.to_numpy(dtype=np.int64)
    crossing_drivers = crossing.driver_number.astype(str).to_numpy()
    for row, start in zip(df.itertuples(), starts):
        why = []
        lap = int(row.lap_number) if pd.notna(row.lap_number) else 0
        driver = str(row.driver_number)
        end = row.lap_end_ns
        if not np.isfinite(start) or pd.isna(end) or end <= start or pd.isna(row.lap_time_s) or row.lap_time_s <= 0:
            why.append("invalid_timing")
        if lap <= 1:
            why.append("race_start")
        if row.compound not in DRY or pd.isna(row.tyre_age_laps) or row.tyre_age_laps < 0:
            why.append("unknown_tyre")
        if pd.notna(row.fastf1_generated) and bool(row.fastf1_generated):
            why.append("generated")
        if historical and pd.notna(getattr(row, "final_deleted", None)) and bool(row.final_deleted):
            why.append("deleted")
        if not historical and bool(getattr(row, "deleted_as_of", False)):
            why.append("deleted")
        if (driver, lap) in pit_laps:
            why.append("pit_lap")
        if np.isfinite(start) and pd.notna(end):
            idx = np.searchsorted(status_times, start, side="right") - 1
            stop = np.searchsorted(status_times, end, side="right")
            if idx < 0 or any(v != "GREEN" for v in status_values[max(idx, 0):stop]):
                why.append("not_green_or_unknown")
            wi = np.searchsorted(weather_times, start, side="right") - 1
            temp = weather_values[wi] if wi >= 0 else np.nan
            weather_stop = np.searchsorted(weather_times, end, side="right")
            conditions = rainfall[max(wi, 0):weather_stop]
            if wi < 0 or any(pd.isna(value) for value in conditions):
                why.append("weather_unknown")
            if any(pd.notna(value) and bool(value) for value in conditions):
                why.append("rain")
            previous = crossing_times[(crossing_drivers != driver) & (crossing_times <= start) & (crossing_releases <= start)]
            gap = (start - previous.max()) / 1e9 if previous.size else np.nan
            if np.isfinite(gap) and gap <= 1.5:
                why.append("traffic")
        else:
            temp, gap = np.nan, np.nan
        temps.append(temp)
        gaps.append(gap)
        warm1.append((driver, lap - 1) in exits)
        warm2.append((driver, lap - 2) in exits)
        reasons.append(tuple(why))
    df["exclusion_reasons"] = reasons
    df["eligible"] = [not x for x in reasons]
    df["track_temperature_c"] = temps
    df["start_gap_s"] = gaps
    df["traffic_unknown"] = df.start_gap_s.isna()
    df["warmup_first"] = np.asarray(warm1, dtype=np.float64)
    df["warmup_second"] = np.asarray(warm2, dtype=np.float64)
    df["layout_id"] = str(event_metadata.get("layout_id") or f"unknown:{session_id}")
    eras = event_metadata.get("spec_era_by_compound", {})
    df["spec_era_id"] = [eras.get(c) or f"unknown:{session_id}:{c}" for c in df.compound]
    return df

"""Convert returned FastF1 frames into time-released canonical Arrow tables.

The source frames are retained independently; normalization never fills missing
values from later observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from .schemas import table_from_rows


def _value(row, key):
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _ns(value):
    if value is None or pd.isna(value):
        return None
    return int(pd.Timedelta(value).value)


def _seconds(value):
    ns = _ns(value)
    return None if ns is None else ns / 1e9


def _int(value):
    return None if value is None else int(value)


def _str(value):
    return None if value is None else str(value)


def _common(session_id, stream, index, ns, basis="estimated_event"):
    return dict(session_id=session_id, source_row_id=f"{stream}:{index}", event_ns=ns,
                release_ns=ns, time_basis=basis if ns is not None else "unknown")


def source_table(frame):
    """Preserve source columns, ordering, nulls and Arrow-representable dtypes."""
    return pa.Table.from_pandas(pd.DataFrame(frame).reset_index(drop=True), preserve_index=False)


def normalize_streams(session_id, origin, frames):
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        origin = origin.tz_localize("UTC")
    else:
        origin = origin.tz_convert("UTC")
    rows = {name: [] for name in ("laps", "pit_events", "weather", "track_status", "session_status", "messages")}
    for i, (_, row) in enumerate(frames.get("laps", pd.DataFrame()).iterrows()):
        get = lambda key: _value(row, key)
        end = _ns(get("Time"))
        compound = _str(get("Compound"))
        compound = compound.upper() if compound else "UNKNOWN"
        if compound not in {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}:
            compound = "UNKNOWN"
        lap = dict(_common(session_id, "laps", i, end), driver_number=_str(get("DriverNumber")),
                   driver_code=_str(get("Driver")), lap_number=_int(get("LapNumber")),
                   lap_start_ns=_ns(get("LapStartTime")), lap_end_ns=end,
                   lap_time_s=_seconds(get("LapTime")), sector1_s=_seconds(get("Sector1Time")),
                   sector2_s=_seconds(get("Sector2Time")), sector3_s=_seconds(get("Sector3Time")),
                   source_stint=_int(get("Stint")), compound=compound, tyre_age_laps=_int(get("TyreLife")),
                   fresh_tyre=get("FreshTyre"), source_track_status=_str(get("TrackStatus")),
                   is_accurate=get("IsAccurate"), fastf1_generated=get("FastF1Generated"),
                   final_deleted=get("Deleted"), final_deleted_reason=_str(get("DeletedReason")),
                   source_position=_int(get("Position")))
        rows["laps"].append(lap)
        for kind, column in (("entry", "PitInTime"), ("exit", "PitOutTime")):
            ns = _ns(get(column))
            if ns is not None:
                rows["pit_events"].append(dict(_common(session_id, f"laps:{i}:{kind}", 0, ns),
                    driver_number=lap["driver_number"], kind=kind, source_lap_number=lap["lap_number"]))
    status_map = {"1": "GREEN", "2": "YELLOW", "4": "SC", "5": "RED", "6": "VSC", "7": "VSC_ENDING"}
    for name in ("weather", "track_status", "session_status", "messages"):
        for i, (_, row) in enumerate(frames.get(name, pd.DataFrame()).iterrows()):
            get = lambda key: _value(row, key)
            utc = None
            if name == "messages":
                utc = get("Time") if get("Utc") is None else get("Utc")
                if utc is not None:
                    utc = pd.Timestamp(utc)
                    if utc.tzinfo is None:
                        utc = utc.tz_localize("UTC")
                    else:
                        utc = utc.tz_convert("UTC")
                ns = None if utc is None else int((utc - origin).value)
            else:
                ns = _ns(get("Time"))
            result = _common(session_id, name, i, ns, "message_timestamp" if name == "messages" else "estimated_event")
            if name == "weather":
                for dest, src in (("air_temp_c", "AirTemp"), ("track_temp_c", "TrackTemp"), ("humidity_pct", "Humidity"),
                                  ("pressure_mbar", "Pressure"), ("rainfall", "Rainfall"), ("wind_direction_deg", "WindDirection"), ("wind_speed_ms", "WindSpeed")):
                    result[dest] = get(src)
            elif name == "track_status":
                code = _str(get("Status"))
                result.update(source_status=code, status=status_map.get(code, "UNKNOWN"), message=_str(get("Message")))
            elif name == "session_status":
                result["status"] = _str(get("Status"))
            else:
                result.update(utc=utc, category=_str(get("Category")), message=_str(get("Message")), status=_str(get("Status")),
                              flag=_str(get("Flag")), scope=_str(get("Scope")), sector=_int(get("Sector")),
                              driver_number=_str(get("RacingNumber")), lap_number=_int(get("Lap")))
            rows[name].append(result)
    return {name: table_from_rows(name, data) for name, data in rows.items()
            if name in frames or name == "pit_events"}


def quality_report(tables):
    """Retain all rows and report suspect observations, not corrected values."""
    laps = tables["laps"].to_pandas()
    findings = {}
    def record(name, mask):
        selected = laps.loc[mask.fillna(False)]
        findings[name] = {"count": int(len(selected)), "drivers": sorted(str(x) for x in selected.driver_number.dropna().unique())}
    record("duplicate_driver_lap", laps.duplicated(["driver_number", "lap_number"], keep=False))
    record("negative_timestamp", (laps.lap_start_ns < 0) | (laps.lap_end_ns < 0))
    record("reversed_timestamp", laps.lap_end_ns < laps.lap_start_ns)
    record("nonpositive_duration", laps.lap_time_s <= 0)
    record("missing_mandatory_time", laps.lap_start_ns.isna() | laps.lap_end_ns.isna() | laps.lap_time_s.isna())
    sector_sum = laps.sector1_s + laps.sector2_s + laps.sector3_s
    record("sector_sum_mismatch", (sector_sum - laps.lap_time_s).abs() > .01)
    ordered = laps.sort_values(["driver_number", "source_stint", "lap_number"], na_position="last")
    diff = ordered.groupby(["driver_number", "source_stint"], dropna=False).tyre_age_laps.diff()
    record("tyre_age_regression", diff.reindex(laps.index) < 0)
    pits = tables["pit_events"].to_pandas().sort_values("event_ns")
    unmatched = []
    for driver, group in pits.groupby("driver_number"):
        opened = False
        for row in group.itertuples():
            if row.kind == "entry":
                if opened:
                    unmatched.append(driver)
                opened = True
            elif opened:
                opened = False
            else:
                unmatched.append(driver)
        if opened:
            unmatched.append(driver)
    findings["unmatched_pit_events"] = {"count": len(unmatched), "drivers": sorted(set(unmatched))}
    return findings

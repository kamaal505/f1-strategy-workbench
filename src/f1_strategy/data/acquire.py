"""Explicitly authorized FastF1 acquisition; never imported by offline readers."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

import pyarrow as pa
from ..contracts import IngestRequest, SnapshotRef, WorkbenchError
from ..identity import digest, lock_digest
from .normalize import normalize_streams, quality_report, source_table
from .snapshots import publish_snapshot


def acquire_session(request: IngestRequest, *, root: Path) -> SnapshotRef:
    if os.environ.get("F1_SOURCE_ACCESS_CONFIRMED") != "1" or os.environ.get("F1_SNAPSHOT_RETENTION_CONFIRMED") != "1":
        raise WorkbenchError("authorization_required", "Operator must confirm source access and snapshot retention permissions")
    from ..deployment import authorize_lineage
    authorize_lineage(
        ({"session_id": f"{request.year}-{request.round_number:02d}-R",
          "source_ids": [f"fastf1:{request.year}-{request.round_number:02d}-R"], "synthetic": False},),
        ("source_access", "snapshot_retention"),
    )
    # Importing FastF1 and contacting the source occur only after operator confirmation.
    import fastf1
    cache = root / "cache" / "fastf1"
    if root.is_symlink() or (root / "cache").is_symlink() or cache.is_symlink():
        raise WorkbenchError("unsafe_path", "Symlink cache root forbidden")
    cache.mkdir(parents=True, exist_ok=True)
    try:
        fastf1.Cache.configure(cache_dir=str(cache))
        session = fastf1.get_session(request.year, request.round_number, "R", backend="fastf1")
        session.load(laps=True, weather=True, messages=True, telemetry=request.telemetry)
    except Exception:
        raise WorkbenchError("acquisition_failed", "FastF1 session acquisition failed; check authorized access and upstream availability") from None
    session_id = f"{request.year}-{request.round_number:02d}-R"
    frames = {}
    availability = {}
    attrs = {"laps": "laps", "weather": "weather_data", "track_status": "track_status",
             "session_status": "session_status", "messages": "race_control_messages"}
    for name, attr in attrs.items():
        try:
            frame = getattr(session, attr)
            if frame is None:
                raise ValueError
            frames[name] = pd.DataFrame(frame).copy()
        except Exception:
            availability[name] = {"status": "unavailable", "reason": "Source stream unavailable", "rows": 0}
    for name in ("laps", "track_status", "session_status"):
        if name not in frames or frames[name].empty:
            raise WorkbenchError("missing_stream", f"Required stream {name} missing or empty")
    origin = session.t0_date
    distance = session.total_laps
    if origin is None or distance is None or int(distance) <= 0:
        raise WorkbenchError("missing_metadata", "Stream origin or scheduled race distance unavailable")
    origin = pd.Timestamp(origin)
    origin = origin.tz_localize("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")
    started = frames["session_status"]
    started = started.loc[started["Status"].astype(str).str.lower().eq("started"), "Time"].dropna()
    if started.empty:
        raise WorkbenchError("missing_metadata", "Race start timestamp unavailable")
    start_ns = int(pd.Timedelta(started.iloc[0]).value)
    if start_ns < 0:
        raise WorkbenchError("missing_metadata", "Invalid race start timestamp")
    tables = normalize_streams(session_id, origin, frames)
    for name, frame in frames.items():
        tables[f"source/{name}"] = source_table(frame)
    quality = quality_report(tables)
    for kind, attr in (("car", "car_data"), ("position", "pos_data")):
        stream_name = f"{kind}_samples"
        if not request.telemetry:
            availability[stream_name] = {"status": "unavailable", "reason": "Not requested", "rows": 0}
            continue
        try:
            sample_tables = []
            streams = getattr(session, attr)
            count = 0
            for driver, frame in sorted(streams.items()):
                if not str(driver).isdigit():
                    continue
                frame = pd.DataFrame(frame).copy()
                tables[f"source/{kind}/{driver}"] = source_table(frame)
                time_values = pd.to_timedelta(frame.get("SessionTime", pd.Series([pd.NaT] * len(frame))), errors="coerce")
                event_times = [None if pd.isna(value) else int(value.value) for value in time_values]
                canonical_frame = frame.copy()
                canonical_frame.insert(0, "driver_number", str(driver))
                canonical_frame.insert(0, "time_basis", ["estimated_event" if value is not None else "unknown" for value in event_times])
                canonical_frame.insert(0, "release_ns", pd.array(event_times, dtype="Int64"))
                canonical_frame.insert(0, "event_ns", pd.array(event_times, dtype="Int64"))
                canonical_frame.insert(0, "source_row_id", [f"{kind}:{driver}:{i}" for i in range(len(frame))])
                canonical_frame.insert(0, "session_id", session_id)
                sample_tables.append(source_table(canonical_frame))
                count += len(frame)
                channels = ("Speed", "Throttle", "Brake") if kind == "car" else ("X", "Y", "Z")
                missing = [c for c in channels if c not in frame]
                times = pd.to_timedelta(frame.get("SessionTime", pd.Series(dtype="timedelta64[ns]")), errors="coerce").dropna().sort_values().diff().dt.total_seconds().dropna()
                quality.setdefault("telemetry", {})[f"{kind}/{driver}"] = {"rows": len(frame), "missing_channels": missing,
                    "null_counts": {c: int(frame[c].isna().sum()) if c in frame else len(frame) for c in channels},
                    "median_cadence_s": float(times.median()) if len(times) else None}
            if sample_tables:
                tables[stream_name] = pa.concat_tables(sample_tables, promote_options="permissive")
            availability[stream_name] = {"status": "available" if count else "empty", "reason": None, "rows": count}
        except Exception:
            availability[stream_name] = {"status": "unavailable", "reason": "Source telemetry unavailable", "rows": 0}
    entries = []
    try:
        for driver in session.drivers:
            info = session.get_driver(driver)
            entries.append({"driver_number": str(driver), "driver_code": str(info.get("Abbreviation", ""))})
    except Exception:
        entries = [{"driver_number": str(d), "driver_code": ""} for d in sorted(tables["laps"].column("driver_number").unique().to_pylist()) if d is not None]
    event_metadata = {"layout_id": f"event:{session_id}", "rules_provenance": "unverified", "nomination_provenance": "unverified",
                      "race_date": origin.date().isoformat(), "source_reference": "FastF1 session metadata; event-specific sporting applicability unverified"}
    metadata = dict(schema_version=1, session_id=session_id, synthetic=False, year=request.year, round_number=request.round_number,
                    session="R", stream_origin_utc=origin.to_pydatetime(), race_start_ns=start_ns, scheduled_laps=int(distance),
                    entries=tuple(entries), event_metadata=event_metadata, event_metadata_digest=digest(event_metadata),
                    acquired_utc=datetime.now(timezone.utc), source_ids=(f"fastf1:{session_id}",),
                    source_endpoints=("FastF1 timing API",), versions={name: version(name) for name in ("fastf1", "pandas", "pyarrow", "numpy")},
                    lock_digest=lock_digest(), load_options={"laps": True, "weather": True, "messages": True, "telemetry": request.telemetry},
                    availability=availability, quality=quality)
    return publish_snapshot(root=root, metadata=metadata, tables=tables)
